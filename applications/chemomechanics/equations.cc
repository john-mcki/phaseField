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
  set_variable_type(0, Vector);
  set_variable_equation_type(0, TimeIndependent);
  set_dependencies_gradient_term_rhs(0,"grad(u),C_old,p");
  set_dependencies_gradient_term_lhs(0,"grad(change(u)),p");
  set_solve_block(0, 0);

  set_variable_name(1, "S");
  set_variable_type(1, Scalar);
  set_variable_equation_type(1, Auxiliary);
  set_dependencies_value_term_rhs(1, "grad(u),C_old,p");
  set_solve_block(1, 1);

  set_variable_name(2, "C_new");
  set_variable_type(2, Scalar);
  set_variable_equation_type(2, TimeIndependent);
  set_dependencies_value_term_rhs(2, "C_new, grad(C_new), C_old, grad(S), p, grad(p)");
  set_dependencies_gradient_term_rhs(2, "C_new, grad(C_new), grad(S)");
  set_dependencies_value_term_lhs(2, "change(C_new),C_old,grad(S),p, grad(p)");
  set_dependencies_gradient_term_lhs(2, "grad(change(C_new)),C_old,grad(S),grad(p)");
  set_solve_block(2, 2);

  set_variable_name(3, "C_old");
  set_variable_type(3, Scalar);
  set_variable_equation_type(3, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(3, "C_new");
  set_solve_block(3, 3);

  set_variable_name(4, "p");
  set_variable_type(4, Scalar);
  set_variable_equation_type(4, Constant);

  set_variable_name(5, "stress_diag"); 
  set_variable_type(5, Vector);
  set_variable_equation_type(5, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(5,"grad(u)");
  set_is_postprocessed_field(5, true);

  set_variable_name(6, "stress_offdiag"); 
  set_variable_type(6, Vector);
  set_variable_equation_type(6, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(6,"grad(u)");
  set_is_postprocessed_field(6, true);

  set_variable_name(7, "stress_eigs"); 
  set_variable_type(7, Vector);
  set_variable_equation_type(7, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(7,"grad(u)");
  set_is_postprocessed_field(7, true);
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
  if (solve_block == 3)
    {
      variable_list.set_value_term(3, variable_list.template get_value<ScalarValue>(2));
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
  if ((index == 0) && (solve_block == 0)) //Step 1: solve displacement using old C
    {
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue C = variable_list.template get_value<ScalarValue>(3);
      ScalarValue p = variable_list.template get_value<ScalarValue>(4);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega/3 * (C - C_ref); //units of length???
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, p * ux, stress);
      variable_list.set_gradient_term(0, -stress); //check sign if results are weird
    }
  if ((index == 1) && (solve_block == 1)) //Step 2: solve hydrostatic stress
    {
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue C = variable_list.template get_value<ScalarValue>(3);
      ScalarValue p = variable_list.template get_value<ScalarValue>(4);
      //Grabbing hydrostatic stress
      ScalarValue hydrostatic_stress(0.0);
      for (unsigned int i = 0; i < dim; i++)
        {
          hydrostatic_stress += youngs_modulus/(1 - 2 * poisson) * (ux[i][i] - omega/3 * (C - C_ref));
        }
      variable_list.set_value_term(1, p * hydrostatic_stress);
    }
  if ((index == 2) && (solve_block == 2)) //step 3: solve concentration, TODO: review B_neu for mechanics term
    {
      ScalarGrad Sx = variable_list.template get_gradient<ScalarGrad>(1);
      ScalarValue C = variable_list.template get_value<ScalarValue>(2);
      ScalarGrad Cx = variable_list.template get_gradient<ScalarGrad>(2);
      ScalarValue C_old = variable_list.template get_value<ScalarValue>(3);
      ScalarValue p = variable_list.template get_value<ScalarValue>(4);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(4);
      ScalarValue px_mag(1e-6);
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = this->get_timestep();
      ScalarValue B_Neu = -0.01 * (1.0/diffusivity) * (C - C_ref);// + Sx.norm_square()/diffusivity; //TODO double check this line
      ScalarValue C_term1 = (diffusivity/p) * (px * Cx);
      //ScalarValue C_term2 = -px/p * Sx * (omega * C * diffusivity)/(R * Temp);
      ScalarValue C_term2 = -px/p * Sx * (omega * diffusivity)/(R * Temp); //C get's removed by way of different free-energy formulation
      ScalarValue C_term3 = (px_mag/p) * diffusivity * B_Neu;
      ScalarGrad Cx_term1 = -diffusivity * Cx;
      ScalarGrad Cx_term2 = (omega * C * diffusivity)/(R * Temp) * Sx;
      ScalarValue eq_C = C_old - C + (dt * (C_term1 + C_term2 + C_term3));
      ScalarGrad eq_Cx = dt * (Cx_term1 + Cx_term2);

      variable_list.set_value_term(2, eq_C);
      variable_list.set_gradient_term(2, eq_Cx);
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
  if ((index == 0) && (solve_block == 0))
    {
      VectorGrad change_ux = variable_list.template get_symmetric_gradient<VectorGrad>(0, Change);
      ScalarValue p = variable_list.template get_value<ScalarValue>(4);
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(stiffness, p * change_ux, stress);
      variable_list.set_gradient_term(0, stress, Change);
    }
  if ((index == 2) && (solve_block == 2)) //NOTE: same solve_block as rhs C solve
    {
      //Concentration to be changed
      ScalarValue change_C = variable_list.template get_value<ScalarValue>(2, Change);
      ScalarGrad change_Cx = variable_list.template get_gradient<ScalarGrad>(2, Change);
      ScalarValue C = variable_list.template get_value<ScalarValue>(3); //using c_old
      //Gradient of Hydrostatic Stress
      ScalarGrad Sx = variable_list.template get_gradient<ScalarGrad>(1);
      //Order Parameter, needed but not changed
      ScalarValue p = variable_list.template get_value<ScalarValue>(4);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(4);

      //domain gradient magnitude
      ScalarValue px_mag(1e-6); //Initial value is equal to offset
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = get_timestep();
      ScalarValue LHS_C_term1 = -(diffusivity/p) * (px * change_Cx);
      ScalarValue LHS_C_term2 = px/p * (omega * diffusivity)/(R*Temp) * (Sx * change_C - (youngs_modulus/(1 - 2 * poisson) * C * omega * change_Cx));//C term got removed in RHS needs to be removed here too, check your math
      ScalarValue LHS_C_term3 = (px_mag/p) * 0.1 * diffusivity * change_C;
      ScalarGrad LHS_Cx_term1 = diffusivity * change_Cx;
      ScalarGrad LHS_Cx_term2 = -(omega * diffusivity)/(R * Temp) * (Sx * change_C - (youngs_modulus/(1 - 2 * poisson) * C * omega * change_Cx));
      ScalarValue eq_change_C = change_C + dt * (LHS_C_term1 + LHS_C_term2 + LHS_C_term3);
      ScalarGrad eq_change_Cx = dt * (LHS_Cx_term1 + LHS_Cx_term2);

      variable_list.set_value_term(2, eq_change_C, Change);
      variable_list.set_gradient_term(2, eq_change_Cx, Change);
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::compute_postprocess_explicit_rhs(
  [[maybe_unused]] VariableContainer<dim, degree, number> &variable_list,
  [[maybe_unused]] const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point_loc,
  [[maybe_unused]] const dealii::VectorizedArray<number> &element_volume,
  [[maybe_unused]] Types::Index                           solve_block) const
{
    VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
    VectorGrad stress;
    compute_stress<dim, ScalarValue>(stiffness, ux, stress);
    ScalarGrad stress_diag;
    ScalarGrad stress_offdiag;
    for (unsigned int i = 0; i < dim; ++i)
      {
        stress_diag[i] = stress[i][i];
      }
    stress_offdiag[0] = stress[0][1];
    stress_offdiag[1] = stress[0][2];
    stress_offdiag[2] = stress[1][2];
    variable_list.set_value_term(5, stress_diag);
    variable_list.set_value_term(6, stress_offdiag);
    ScalarGrad stress_eigs;
    for (unsigned int k = 0; k < stress[0][0].size(); ++k)
      {
        dealii::SymmetricTensor<2, dim, number> stress_sym;
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              {
                stress_sym[i][j] = stress[i][j][k];
              }
          }
        std::array<number, dim> stress_eigvals = dealii::eigenvalues(stress_sym);
        for (unsigned int i = 0; i < dim; ++i)
          {
            stress_eigs[i][k] = stress_eigvals[i];
          }
      }
    variable_list.set_value_term(7, stress_eigs);
}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE