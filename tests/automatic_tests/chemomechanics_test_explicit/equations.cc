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
  set_variable_equation_type(0, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(0, "C,p,grad(u),grad(p),grad(S)");
  set_dependencies_gradient_term_rhs(0, "C,grad(C),grad(u),grad(p),grad(S)");
  //set_solve_block(0,1);

  set_variable_name(1, "u");
  set_variable_type(1, Vector);
  set_variable_equation_type(1, TimeIndependent);
  //set_dependencies_value_term_rhs(1, "");
  set_dependencies_gradient_term_rhs(1,"grad(u),C");
  //set_dependencies_value_term_lhs(1, "");
  set_dependencies_gradient_term_lhs(1, "grad(change(u))");

  set_variable_name(2, "S");
  set_variable_type(2, Scalar);
  set_variable_equation_type(2, Auxiliary);
  set_dependencies_value_term_rhs(2, "grad(u),C");
  //set_dependencies_gradient_term_rhs(2,""); //Double check gradient dependencies
  //set_solve_block(2,0);

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
{
    {  
      //Concentration, scalar value may not be needed
      ScalarValue C = variable_list.template get_value<ScalarValue>(0);
      ScalarGrad Cx = variable_list.template get_gradient<ScalarGrad>(0);
      //Gradient of Hydrostatic Stress
      ScalarGrad Sx = variable_list.template get_gradient<ScalarGrad>(2);
      //Order Parameter
      ScalarValue p = variable_list.template get_value<ScalarValue>(3);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(3);
      //Unit normal vector for Li concentration
      ScalarValue px_mag(1e-6); //Initial value is equal to offset
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = this->get_timestep();
      ScalarValue B_Neu = -0.1 * (C - C_ref); //+ Sx.norm_square()/diffusivity; // Neumann Condition
      ScalarValue C_term1 = (diffusivity/p) * (px * Cx);
      ScalarValue C_term2 = -px/p * Sx * (omega * C)/(R * Temp);
      ScalarValue C_term3 = (px_mag/p) * diffusivity * B_Neu;
      ScalarGrad Cx_term1 = diffusivity * Cx;
      ScalarGrad Cx_term2 = (omega * C)/(R * Temp) * Sx;
      ScalarValue eq_C = C + (dt * (C_term1 + C_term2 + C_term3));
      ScalarGrad eq_Cx = -dt * (Cx_term1 + Cx_term2);  
      variable_list.set_value_term(0, eq_C);
      variable_list.set_gradient_term(0, eq_Cx);
    }
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
  if (index == 2)
    {
      //Concentration, scalar value may not be needed
      ScalarValue C = variable_list.template get_value<ScalarValue>(0);
      //Displacement gradient
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(1);
      //Grabbing hydrostatic stress
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega/3 * (C - C_ref);
        }
      ScalarValue hydrostatic_stress(0.0);
      //Turning the gradient into a divergence
      for (unsigned int i = 0; i < dim; i++)
        {
          hydrostatic_stress += 22.5/3 * ux[i][i]; //22.5 is value in compliance tensor
      }
      variable_list.set_value_term(2, hydrostatic_stress);
    }
  if (index == 1)
    {
      //Concentration
      ScalarValue C = variable_list.template get_value<ScalarValue>(0);

      //Displacement gradient
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(1);

      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega/3 * (C - C_ref);
        }
          
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(compliance, ux, stress);
      variable_list.set_gradient_term(1, -stress);
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
  if (index == 1)
    {
      VectorGrad change_ux =
        variable_list.template get_symmetric_gradient<VectorGrad>(1, Change);
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