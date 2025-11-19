// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/type_enums.h>
#include <prismspf/core/variable_attribute_loader.h>
#include <prismspf/core/variable_container.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE
template <typename T, unsigned int dim> 
dealii::Tensor<2, voigt_tensor_size<dim>, T>
form_Cij(const dealii::Tensor<1, dim, T> &Cv1,
        const dealii::Tensor<1, dim, T> &Cv2,
        const dealii::Tensor<1, dim, T> &Cv3,
        const dealii::Tensor<1, dim, T> &Cv4,
        const dealii::Tensor<1, dim, T> &Cv5,
        const dealii::Tensor<1, dim, T> &Cv6,
        const dealii::Tensor<1, dim, T> &Cv7);
void
CustomAttributeLoader::load_variable_attributes()
{
  std::string c_vec_string = ", Cv1, Cv2, Cv3, Cv4, Cv5, Cv6, Cv7";
  set_variable_name(0, "u");
  set_variable_type(0, Vector);
  set_variable_equation_type(0, TimeIndependent);
  //set_dependencies_gradient_term_rhs(2,"grad(u),C_old,p");
  //set_dependencies_gradient_term_lhs(2, "grad(change(u)),p");
  set_dependencies_gradient_term_rhs(0, std::string("grad(u),C_old,p")+c_vec_string);
  set_dependencies_gradient_term_lhs(0, std::string("grad(change(u)),p")+c_vec_string);
  set_solve_block(0, 0);

  for (unsigned int i = 1; i <= 7; ++i)
    {
      set_variable_name(i, "Cv" + std::to_string(i));
      set_variable_type(i, Vector);
      set_variable_equation_type(i, Constant);
      set_dependencies_gradient_term_rhs(i, "");
      set_dependencies_gradient_term_lhs(i, "");
    }

  set_variable_name(8, "stress_diag"); 
  set_variable_type(8, Vector);
  set_variable_equation_type(8, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(8, std::string("grad(u)")+c_vec_string);
  set_is_postprocessed_field(8, true);

  set_variable_name(9, "stress_offdiag"); 
  set_variable_type(9, Vector);
  set_variable_equation_type(9, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(9, std::string("grad(u)")+c_vec_string);
  set_is_postprocessed_field(9, true);

  set_variable_name(10, "stress_eigs"); 
  set_variable_type(10, Vector);
  set_variable_equation_type(10, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(10, std::string("grad(u)")+c_vec_string);
  set_is_postprocessed_field(10, true);

  set_variable_name(11, "S");
  set_variable_type(11, Scalar);
  set_variable_equation_type(11, Auxiliary);
  set_dependencies_value_term_rhs(11, "grad(u),C_old,p");
  set_solve_block(11, 1);

  set_variable_name(12, "C_new");
  set_variable_type(12, Scalar);
  set_variable_equation_type(12, TimeIndependent);
  set_dependencies_value_term_rhs(12, "C_new, grad(C_new), C_old, grad(S), p, grad(p)");
  set_dependencies_gradient_term_rhs(12, "C_new, grad(C_new), grad(S)");
  set_dependencies_value_term_lhs(12, "change(C_new),C_old,grad(S),p, grad(p)");
  set_dependencies_gradient_term_lhs(12, "grad(change(C_new)),C_old,grad(S),grad(p)");
  set_solve_block(12, 2);

  set_variable_name(13, "C_old");
  set_variable_type(13, Scalar);
  set_variable_equation_type(13, ExplicitTimeDependent);
  set_dependencies_value_term_rhs(13, "C_new");
  set_solve_block(13, 3);

  set_variable_name(14, "p");
  set_variable_type(14, Scalar);
  set_variable_equation_type(14, Constant);
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
      variable_list.set_value_term(13, variable_list.template get_value<ScalarValue>(12));
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
      // Compute the RHS of the momentum balance equation
      dealii::Tensor<2, voigt_tensor_size<dim>, dealii::VectorizedArray<number>> 
      C = form_Cij<dealii::VectorizedArray<number>,dim>(
            variable_list.template get_value<ScalarGrad>(1),
            variable_list.template get_value<ScalarGrad>(2),
            variable_list.template get_value<ScalarGrad>(3),
            variable_list.template get_value<ScalarGrad>(4),
            variable_list.template get_value<ScalarGrad>(5),
            variable_list.template get_value<ScalarGrad>(6),
            variable_list.template get_value<ScalarGrad>(7));
      ScalarValue Con = variable_list.template get_value<ScalarValue>(13);
      ScalarValue p = variable_list.template get_value<ScalarValue>(14);
      for (unsigned int i = 0; i < dim; i++)
        {
          ux[i][i] -= omega/3 * (Con - C_ref);
        }
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(compliance, p * ux, stress);
      variable_list.set_gradient_term(0, -stress); //check sign if results are weird
    }
  if ((index == 11) && (solve_block == 1)) //Step 2: solve hydrostatic stress
    {
      VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
      ScalarValue C = variable_list.template get_value<ScalarValue>(13);
      ScalarValue p = variable_list.template get_value<ScalarValue>(14);
      //Grabbing hydrostatic stress
      ScalarValue hydrostatic_stress(0.0);
      for (unsigned int i = 0; i < dim; i++)
        {
          hydrostatic_stress += 22.5/(1 - 2 * 0.3) * (ux[i][i] - omega/3 * (C - C_ref));
        }
      variable_list.set_value_term(11, p * hydrostatic_stress);
    }
  if ((index == 12) && (solve_block == 2)) //step 3: solve concentration, TODO: review B_neu for mechanics term
    {
      ScalarValue C = variable_list.template get_value<ScalarValue>(12);
      ScalarGrad Cx = variable_list.template get_gradient<ScalarGrad>(12);
      ScalarValue C_old = variable_list.template get_value<ScalarValue>(13);
      ScalarGrad Sx = variable_list.template get_gradient<ScalarGrad>(11);
      ScalarValue p = variable_list.template get_value<ScalarValue>(14);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(14);
      ScalarValue px_mag(1e-6);
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = this->get_timestep();
      ScalarValue B_Neu = -0.1 * (C - C_ref);// + Sx.norm_square()/diffusivity; //TODO double check this line
      ScalarValue C_term1 = (diffusivity/p) * (px * Cx);
      ScalarValue C_term2 = -px/p * Sx * (omega * C * diffusivity)/(R * Temp);
      ScalarValue C_term3 = (px_mag/p) * diffusivity * B_Neu;
      ScalarGrad Cx_term1 = -diffusivity * Cx;
      ScalarGrad Cx_term2 = (omega * C * diffusivity)/(R * Temp) * Sx;
      ScalarValue eq_C = C_old - C + (dt * (C_term1 + C_term2 + C_term3));
      ScalarGrad eq_Cx = dt * (Cx_term1 + Cx_term2); //double-check

      variable_list.set_value_term(12, eq_C);
      variable_list.set_gradient_term(12, eq_Cx);
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
      dealii::Tensor<2, voigt_tensor_size<dim>, dealii::VectorizedArray<number>> 
      C = form_Cij<dealii::VectorizedArray<number>,dim>(
            variable_list.template get_value<ScalarGrad>(1),
            variable_list.template get_value<ScalarGrad>(2),
            variable_list.template get_value<ScalarGrad>(3),
            variable_list.template get_value<ScalarGrad>(4),
            variable_list.template get_value<ScalarGrad>(5),
            variable_list.template get_value<ScalarGrad>(6),
            variable_list.template get_value<ScalarGrad>(7));
      VectorGrad change_ux = variable_list.template get_symmetric_gradient<VectorGrad>(0, Change);
      ScalarValue p = variable_list.template get_value<ScalarValue>(14);
      VectorGrad stress;
      compute_stress<dim, ScalarValue>(compliance, p * change_ux, stress);
      variable_list.set_gradient_term(0, stress, Change);
    }
  if ((index == 12) && (solve_block == 2)) //NOTE: same solve_block as rhs C solve
    {
      //Concentration to be changed
      ScalarValue change_C = variable_list.template get_value<ScalarValue>(12, Change);
      ScalarGrad change_Cx = variable_list.template get_gradient<ScalarGrad>(12, Change);
      ScalarValue C = variable_list.template get_value<ScalarValue>(13); //using c_old
      //Gradient of Hydrostatic Stress
      ScalarGrad Sx = variable_list.template get_gradient<ScalarGrad>(11);
      //Order Parameter, needed but not changed
      ScalarValue p = variable_list.template get_value<ScalarValue>(14);
      ScalarGrad px = variable_list.template get_gradient<ScalarGrad>(14);

      //domain gradient magnitude
      ScalarValue px_mag(1e-6); //Initial value is equal to offset
      for (unsigned int i = 0; i < dim; i++)
        {
          px_mag += px[i] * px[i];
        }
      px_mag = std::sqrt(px_mag);
      ScalarValue dt = get_timestep();
      ScalarValue LHS_C_term1 = -(diffusivity/p) * (px * change_Cx);
      ScalarValue LHS_C_term2 = px/p * (omega * diffusivity)/(R*Temp) * (Sx * change_C - (22.5/(1 - 2 * 0.3) * C * omega * change_Cx)); //22.5 found in compliance tensor
      ScalarValue LHS_C_term3 = (px_mag/p) * 0.1 * diffusivity * change_C;
      ScalarGrad LHS_Cx_term1 = diffusivity * change_Cx;
      ScalarGrad LHS_Cx_term2 = -(omega * diffusivity)/(R * Temp) * (Sx * change_C - (22.5/(1 - 2 * 0.3) * C * omega * change_Cx));
      ScalarValue eq_change_C = change_C + dt * (LHS_C_term1 + LHS_C_term2 + LHS_C_term3);
      ScalarGrad eq_change_Cx = dt * (LHS_Cx_term1 + LHS_Cx_term2);

      variable_list.set_value_term(12, eq_change_C, Change);
      variable_list.set_gradient_term(12, eq_change_Cx, Change);
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
    dealii::Tensor<2, voigt_tensor_size<dim>, dealii::VectorizedArray<number>> 
    C = form_Cij<dealii::VectorizedArray<number>,dim>(
          variable_list.template get_value<ScalarGrad>(1),
          variable_list.template get_value<ScalarGrad>(2),
          variable_list.template get_value<ScalarGrad>(3),
          variable_list.template get_value<ScalarGrad>(4),
          variable_list.template get_value<ScalarGrad>(5),
          variable_list.template get_value<ScalarGrad>(6),
          variable_list.template get_value<ScalarGrad>(7));
    VectorGrad ux = variable_list.template get_symmetric_gradient<VectorGrad>(0);
    VectorGrad stress;
    compute_stress<dim, ScalarValue>(C, ux, stress);
    ScalarGrad stress_diag;
    ScalarGrad stress_offdiag;
    for (unsigned int i = 0; i < dim; ++i)
      {
        stress_diag[i] = stress[i][i];
      }
    stress_offdiag[0] = stress[0][1];
    stress_offdiag[1] = stress[0][2];
    stress_offdiag[2] = stress[1][2];
    variable_list.set_value_term(8, stress_diag);
    variable_list.set_value_term(9, stress_offdiag);
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
    variable_list.set_value_term(10, stress_eigs);
}

template <typename T, unsigned int dim> 
dealii::Tensor<2, voigt_tensor_size<dim>, T>
form_Cij(const dealii::Tensor<1, dim, T> &Cv1,
        const dealii::Tensor<1, dim, T> &Cv2,
        const dealii::Tensor<1, dim, T> &Cv3,
        const dealii::Tensor<1, dim, T> &Cv4,
        const dealii::Tensor<1, dim, T> &Cv5,
        const dealii::Tensor<1, dim, T> &Cv6,
        const dealii::Tensor<1, dim, T> &Cv7)
{
  if constexpr (dim != 3)
    {
      throw("This function is designed for 3-D elastic constants only.");
    }
  if constexpr (dim == 3)
    {
        dealii::Tensor<2, voigt_tensor_size<dim>, T> C;
        C[0][0] = Cv1[0];
        C[1][1] = Cv1[1];
        C[2][2] = Cv1[2];
        C[3][3] = Cv2[0];
        C[4][4] = Cv2[1];
        C[5][5] = Cv2[2];
        C[0][1] = Cv3[0];
        C[1][0] = Cv3[0];
        C[0][2] = Cv3[1];
        C[2][0] = Cv3[1];
        C[0][3] = Cv3[2];
        C[3][0] = Cv3[2];
        C[0][4] = Cv4[0];
        C[4][0] = Cv4[0];
        C[0][5] = Cv4[1];
        C[5][0] = Cv4[1];
        C[1][2] = Cv4[2];
        C[2][1] = Cv4[2];
        C[1][3] = Cv5[0];
        C[3][1] = Cv5[0];
        C[1][4] = Cv5[1];
        C[4][1] = Cv5[1];
        C[1][5] = Cv5[2];
        C[5][1] = Cv5[2];
        C[2][3] = Cv6[0];
        C[3][2] = Cv6[0];
        C[2][4] = Cv6[1];
        C[4][2] = Cv6[1];
        C[2][5] = Cv6[2];
        C[5][2] = Cv6[2];
        C[3][4] = Cv7[0];
        C[4][3] = Cv7[0];
        C[3][5] = Cv7[1];
        C[5][3] = Cv7[1];
        C[4][5] = Cv7[2];
        C[5][4] = Cv7[2];
        return C;
    }
}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE