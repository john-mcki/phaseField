// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/type_enums.h>
#include <prismspf/core/variable_attribute_loader.h>
#include <prismspf/core/variable_container.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

void
CustomAttributeLoader::load_variable_attributes()
{
  set_variable_name(0, "u");
  set_variable_type(0, FieldInfo::TensorRank::Vector);
  set_variable_equation_type(0, TimeIndependent);
  set_dependencies_gradient_term_rhs(0,"grad(u),C,p1");
  set_dependencies_gradient_term_lhs(0, "grad(change(u)),p1");

  set_variable_name(1, "S");
  set_variable_type(1, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(1, Auxiliary);
  set_dependencies_value_term_rhs(1, "grad(u),C,p1");
  //set_dependencies_gradient_term_rhs(1,"grad(u),C,p"); //Double check gradient dependencies

  set_variable_name(2, "C");
  set_variable_type(2, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(2, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(2, "C,grad(S),p1,grad(p1)");
  set_dependencies_gradient_term_rhs(2, "C,grad(C),grad(S),grad(p1)");

  set_variable_name(3, "p1");
  set_variable_type(3, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(3, Constant);

  set_variable_name(4, "p2");
  set_variable_type(4, FieldInfo::TensorRank::Scalar);
  set_variable_equation_type(4, Constant);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
  //Gradient of Hydrostatic Stress
  ScalarGrad Sx = variable_list.template get_gradient<ScalarGrad>(1);
  //Concentration
  ScalarValue C = variable_list.template get_value<ScalarValue>(2);
  ScalarGrad Cx = variable_list.template get_gradient<ScalarGrad>(2);
  //Order Parameter
  ScalarValue p = variable_list.template get_value<ScalarValue>(3);
  ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(3);
  ScalarValue px_mag(1e-6); //Initial value is equal to offset
  for (unsigned int i = 0; i < dim; i++)
    {
      px_mag += px[i] * px[i];
    }
  px_mag = std::sqrt(px_mag);
  ScalarValue dt = this->get_timestep();
  ScalarValue B_Neu = -kc * (C - C_ref);//for now there is no traction, only flux term is mass
  ScalarValue C_term1 = (diffusivity/p) * (px * Cx);
  ScalarValue C_term2 = -px/p * Sx * diffusivity * (omega * C)/(R * Temp);
  ScalarValue C_term3 = (px_mag/p) * diffusivity * B_Neu;
  ScalarGrad Cx_term1 = -diffusivity * Cx;
  ScalarGrad Cx_term2 = diffusivity * (omega * C)/(R * Temp) * Sx;
  ScalarValue eq_C = C + (dt * (C_term1 + C_term2 + C_term3));
  ScalarGrad eq_Cx = dt * (Cx_term1 + Cx_term2);

  variable_list.set_value_term(2, eq_C);
  variable_list.set_gradient_term(2, eq_Cx);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{
  if (index == 0)
    {
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue C = variable_list.template get_value<ScalarValue>(2);
      ScalarValue p = variable_list.template get_value<ScalarValue>(3);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega/3 * (C - C_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, p * ux, stress);
      variable_list.set_gradient_term(0, -stress); //check sign if results are weird
    }
  // would grabbing the displacement term gradient be sufficient since it was just updated to be stress?
  if (index == 1)
    {
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue hydrostatic_stress(0.0); //zeroing out S at every term may not be efficient
      for (unsigned int i = 0; i < dim; i++)
        {
          hydrostatic_stress += 1.0/3 * ux[i][i];
        }
      variable_list.set_value_term(1, hydrostatic_stress);
    }   
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_lhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{
  if (index == 0)
    {
      VectorGrad change_ux = variable_list.template get_symmetric_gradient<VectorGrad>(0, Change);
      ScalarValue p = variable_list.template get_value<ScalarValue>(3);
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, p * change_ux, stress);
      variable_list.set_gradient_term(0, stress, Change);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_postprocess_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE