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
  set_variable_equation_type(0, TimeIndependent);
  set_dependencies_value_term_rhs(0, "C, C_old, p");
  set_dependencies_gradient_term_rhs(0, "grad(C), grad(p)");
  set_dependencies_value_term_lhs(0, "change(C),p");
  set_dependencies_gradient_term_lhs(0, "grad(change(C)),grad(p)");
  set_solve_block(0, 0);

  set_variable_name(1, "C_old");
  set_variable_type(1, Scalar);
  set_variable_equation_type(1, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(1, "C");
  set_solve_block(1, 1);

  set_variable_name(2, "p");
  set_variable_type(2, Scalar);
  set_variable_equation_type(2, Constant);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
  if (solve_block == 1)
    {
      variable_list.set_value_term(1, variable_list.template get_value<ScalarValue>(0));
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
  if ((solve_block == 0) && (index == 0))
    {
      ScalarValue C = variable_list.template get_value<ScalarValue>(0);
      ScalarGrad Cx = variable_list.template get_gradient<ScalarGrad>(0);
      ScalarValue C_old = variable_list.template get_value<ScalarValue>(1);
      ScalarValue p = variable_list.template get_value<ScalarValue>(2);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(2);
      ScalarValue px_mag(1e-6); 
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = get_timestep();
      ScalarValue B_Neu = -0.1 * (C - concentration_ref);
      ScalarValue C_term1 = (diffusivity/p) * (px * Cx);
      ScalarValue C_term2 = (px_mag/p) * diffusivity * B_Neu;
      ScalarValue eq_C = C_old - C + (dt * (C_term1 + C_term2));
      ScalarGrad eq_Cx = -diffusivity * dt * Cx;
      variable_list.set_value_term(0, eq_C);
      variable_list.set_gradient_term(0, eq_Cx);
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
  ScalarValue change_C = variable_list.template get_value<ScalarValue>(0, Change);
  ScalarGrad change_Cx = variable_list.template get_gradient<ScalarGrad>(0, Change);
  ScalarValue p = variable_list.template get_value<ScalarValue>(2);
  ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(2);
  ScalarValue px_mag(1e-6);
  for (unsigned int i = 0; i < dim; i++)
    {
      px_mag += px[i] * px[i];
    }
  px_mag = std::sqrt(px_mag);
  ScalarValue dt = get_timestep();
  ScalarValue LHS_C_term1 = -1.0 * (diffusivity/p) * (px * change_Cx);
  ScalarValue LHS_C_term2 = (px_mag/p) * 0.1 * diffusivity * change_C;
  ScalarValue eq_change_C = change_C + dt * (LHS_C_term1 + LHS_C_term2);
  ScalarGrad eq_change_Cx = change_Cx * diffusivity * dt;
  variable_list.set_value_term(0, eq_change_C, Change);
  variable_list.set_gradient_term(0, eq_change_Cx, Change);
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