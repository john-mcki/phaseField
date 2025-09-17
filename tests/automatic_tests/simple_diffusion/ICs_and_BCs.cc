// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/initial_conditions.h>
#include <prismspf/core/nonuniform_dirichlet.h>

#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/utilities/utilities.h>

#include <prismspf/config.h>

#include <cmath>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_initial_condition(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{
  //JM the space is a cube, so axis is not important
  const double dim_size = 
    this->get_user_inputs().get_spatial_discretization().get_size()[0];
  //std::vector<double> center = {dim_size/2,dim_size/2,dim_size/2};
  std::vector<double> center = {dim_size/2,dim_size/2};
  const double radius = 15.0;
  const double concentration = 0.1;
  double dist = 0.0;
  for (unsigned int i = 0; i < dim; i++)
    {
      dist += (point[i] - center[i]) * (point[i] - center[i]);
    }
  dist = std::sqrt(dist);
  double domain_parameter_1 = 0.5 - 0.5*std::tanh((dist - radius));
  double domain_parameter_2 = 1.0 - domain_parameter_1; //domain parameter for some material
  double domain_parameter_3 = 1.0;
  if (point[1] < 10.0)
    {
      domain_parameter_3 = 0.0; //domain material for bottom plate
    }
  double offset = 1e-10;
  if (index == 0)
    {
      scalar_value = concentration * domain_parameter_1 + offset; //setting concentration of li
    }
  if (index == 1)
    {
      //Setting the domain parameter
      scalar_value = domain_parameter_1 + offset;
    }
  if (index == 2)
    {
      //Setting the domain parameter
      scalar_value = domain_parameter_3 + offset;
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_nonuniform_dirichlet(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &boundary_id,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE