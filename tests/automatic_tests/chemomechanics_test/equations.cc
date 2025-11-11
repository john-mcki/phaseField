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
  set_variable_name(0, "C");
  set_variable_type(0, Scalar);
  set_variable_equation_type(0, ImplicitTimeDependent);
  set_dependencies_value_term_rhs(0, "C, old_1(C), grad(S), p, grad(p)");
  set_dependencies_gradient_term_rhs(0, "grad(C), grad(p)");
  set_dependencies_value_term_lhs(0, "change(C),grad(S),p, grad(p)");
  set_dependencies_gradient_term_lhs(0, "grad(change(C)),grad(S),grad(p)");

  set_variable_name(1, "u");
  set_variable_type(1, Vector);
  set_variable_equation_type(1, TimeIndependent);
  set_dependencies_value_term_rhs(1, "");
  set_dependencies_gradient_term_rhs(1,"grad(u),C,p");
  set_dependencies_value_term_lhs(1, "");
  set_dependencies_gradient_term_lhs(1, "grad(change(u)),grad(p)");

  set_variable_name(2, "S");
  set_variable_type(2, Vector);
  set_variable_equation_type(2, Auxiliary);
  set_dependencies_value_term_rhs(2, "grad(u),C,p");
  set_dependencies_gradient_term_rhs(2,"grad(u),C,p"); //Double check gradient dependencies
  //set_dependencies_value_term_lhs(0, "change(C),p");
  //set_dependencies_gradient_term_lhs(0, "grad(change(C)),grad(p)");

  set_variable_name(3, "p");
  set_variable_type(3, Scalar);
  set_variable_equation_type(3, Constant);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{
  if ((index == 0) && (solve_block == 0))
    {
      //Concentration
      ScalarValue C = variable_list.template get_value<ScalarValue>(0);
      ScalarGrad Cx = variable_list.template get_gradient<ScalarGrad>(0);
      ScalarValue C_old = variable_list.template get_value<ScalarValue>(0,OldOne);
      VectorGrad Sx = variable_list.template get_gradient<VectorGrad>(2);
      ScalarValue p = variable_list.template get_value<ScalarValue>(3);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(3);
      ScalarValue px_mag(1e-6); //Initial value is equal to offset
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = get_timestep();
      ScalarGrad Sxh;
      for (unsigned int i = 0; i < dim; i++)
        {
          Sxh[i] = Sx[i][i];
        }
      ScalarValue B_Neu = -0.1 * (C - C_ref) + Sxh.norm_square()/diffusivity;
      
      ScalarValue C_term1 = (diffusivity/p) * (px * Cx);
      ScalarValue C_term2 = -(px/p) * Sxh * (omega * C)/(R * Temp);
      ScalarValue C_term3 = (px_mag/p) * diffusivity * B_Neu;
      ScalarValue eq_C = C_old - C + (dt * (C_term1 + C_term2 + C_term3));
      ScalarGrad eq_Cx = -dt * (diffusivity * Cx - (omega * C)/(R*Temp) * Sxh); //double-check

      variable_list.set_value_term(0, eq_C);
      variable_list.set_gradient_term(0, eq_Cx);
    }
  if ((index == 1) && (solve_block == 1)) //Implementing a displacement update segment, could be the wrong approach
    {
      ScalarValue C = variable_list.template get_value<ScalarValue>(0);
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(1);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega/3 * (C - C_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(compliance, ux, stress);
      variable_list.set_gradient_term(1, -stress); //check sign if results are weird
    }
  if ((index == 2) && (solve_block == 0))
    {
      ScalarValue C = variable_list.template get_value<ScalarValue>(0);
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(1);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega/3 * (C - C_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(compliance, ux, stress);
      variable_list.set_gradient_term(2, stress); //check sign if results are weird
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
      //Concentration to be changed
      ScalarValue change_C = variable_list.template get_value<ScalarValue>(0, Change);
      ScalarGrad change_Cx = variable_list.template get_gradient<ScalarGrad>(0, Change);

      ScalarValue C = variable_list.template get_value<ScalarValue>(0);

      //Getting the stress gradient
      VectorGrad Sx = variable_list.template get_gradient<VectorGrad>(2);

      //Order Parameter, needed but not changed
      ScalarValue p = variable_list.template get_value<ScalarValue>(3);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(3);

      //domain gradient magnitude
      ScalarValue px_mag(1e-6); //Initial value is equal to offset
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = get_timestep();
      ScalarGrad Sxh;
      for (unsigned int i = 0; i < dim; i++)
        {
          Sxh[i] = Sx[i][i];
        }
      ScalarValue LHS_C_term1 = -1.0 * (diffusivity/p) * (px * change_Cx);
      ScalarValue LHS_C_term2 = px/p * omega/(R*Temp) * (Sxh - 22.5 * C * omega * change_Cx); //22.5 found in compliance tensor
      ScalarValue LHS_C_term3 = (px_mag/p) * 0.1 * diffusivity * change_C;
      ScalarGrad LHS_Cx_term1 = -diffusivity * change_Cx;
      ScalarGrad LHS_Cx_term2 = omega/(R * Temp) * (Sxh - (22.5 * C * omega * change_Cx));
      ScalarValue eq_change_C = change_C + dt * (LHS_C_term1 + LHS_C_term2 + LHS_C_term3);
      ScalarGrad eq_change_Cx = dt * (LHS_Cx_term1 + LHS_Cx_term2);

      variable_list.set_value_term(0, eq_change_C, Change);
      variable_list.set_gradient_term(0, eq_change_Cx, Change);
    }
  if (index == 1)
    {
      VectorGrad change_ux = variable_list.template get_symmetric_gradient<VectorGrad>(1, Change);
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(compliance, change_ux, stress);
      variable_list.set_gradient_term(1, stress, Change);
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