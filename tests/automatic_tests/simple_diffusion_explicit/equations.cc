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
  set_dependencies_value_term_rhs(0, "C,p1,grad(p1)");
  set_dependencies_gradient_term_rhs(0, "grad(C),grad(p1)");

  set_variable_name(1, "p1");
  set_variable_type(1, Scalar);
  set_variable_equation_type(1, Constant);
  //set_dependencies_value_term_rhs(1, "C,p")
  //set_dependencies_gradient_term_rhs(1,"grad(n)")

  set_variable_name(2, "p2");
  set_variable_type(2, Scalar);
  set_variable_equation_type(2, Constant);
  //set_dependencies_value_term_rhs(1, "C,p")
  //set_dependencies_gradient_term_rhs(1,"grad(n)")

  set_variable_name(3, "p3");
  set_variable_type(3, Scalar);
  set_variable_equation_type(3, Constant);
  //set_dependencies_value_term_rhs(1, "C,p")
  //set_dependencies_gradient_term_rhs(1,"grad(n)")
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
  //Concentration, scalar value may not be needed
  //ScalarValue offset(1e-10); //Offset value for the domain parameter
  ScalarValue C = variable_list.template get_value<ScalarValue>(0);
  ScalarGrad Cx = variable_list.template get_gradient<ScalarGrad>(0);

  //Order Parameter
  ScalarValue p1 = variable_list.template get_value<ScalarValue>(1);
  ScalarGrad p1x = variable_list.template get_gradient<ScalarGrad>(1);

  //Unit normal vector for Li concentration
  ScalarValue p1x_mag(1e-6); //Initial value is equal to offset
  for (unsigned int i = 0; i < dim; i++)
    {
      p1x_mag += p1x[i] * p1x[i];
    }
  p1x_mag = std::sqrt(p1x_mag);
  //ScalarGrad n = p1x/p1x_mag;
  ScalarValue dt = this->get_timestep();
  ScalarValue B_Neu = -0.1 * (C - concentration_ref);
  ScalarValue C_term1 = (diffusivity/p1) * (p1x * Cx);
  ScalarValue C_term2 = (p1x_mag/p1) * diffusivity * B_Neu;
  ScalarValue eq_C = (C + (dt * (C_term1 + C_term2)));
  ScalarGrad eq_Cx = (-diffusivity * dt * Cx);  
  //Evolving
  variable_list.set_value_term(0, eq_C);
  variable_list.set_gradient_term(0, eq_Cx);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_nonexplicit_lhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block,
  [[maybe_unused]] Types::Index                           index) const
{}

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