#include <core/matrixFreePDE.h>
#include <grains/FloodFiller.h>
#include <grains/OrderParameterRemapper.h>
#include <grains/SimplifiedGrainRepresentation.h>

// vmult operation for LHS
template <int dim, int degree>
void
MatrixFreePDE<dim, degree>::reassignGrains()
{
  // log time
  computing_timer.enter_subsection("matrixFreePDE: reassignGrains");

  pcout << "Reassigning grains...\n";

  // Get the index of the first scalar field (used to get the FE object and
  // DOFHandler)
  unsigned int scalar_field_index = 0;
  for (const auto &[index, variable] : var_attributes)
    {
      if (variable.var_type == SCALAR)
        {
          scalar_field_index = index;
          break;
        }
    }

  // Create the simplified grain representations
  QGaussLobatto<dim>       quadrature2(degree + 1);
  FloodFiller<dim, degree> flood_filler(*FESet.at(scalar_field_index), quadrature2);

  std::vector<GrainSet<dim>> grain_sets;

  unsigned int op_list_index = 0;
  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      if (op_list_index < userInputs.variables_for_remapping.size())
        {
          if (fieldIndex == userInputs.variables_for_remapping.at(op_list_index))
            {
              op_list_index++;

              std::vector<GrainSet<dim>> single_OP_grain_sets;
              flood_filler.calcGrainSets(*FESet.at(scalar_field_index),
                                         *dofHandlersSet_nonconst.at(scalar_field_index),
                                         solutionSet.at(fieldIndex),
                                         userInputs.order_parameter_threshold,
                                         1.0 + userInputs.order_parameter_threshold,
                                         0,
                                         fieldIndex,
                                         single_OP_grain_sets);

              grain_sets.insert(grain_sets.end(),
                                single_OP_grain_sets.begin(),
                                single_OP_grain_sets.end());
            }
        }
    }

  // Set the grain indices to unique values
  for (unsigned int g = 0; g < grain_sets.size(); g++)
    {
      grain_sets.at(g).setGrainIndex(g);
    }

  std::vector<SimplifiedGrainRepresentation<dim>> old_grain_representations =
    simplified_grain_representations;
  simplified_grain_representations.clear();
  for (unsigned int g = 0; g < grain_sets.size(); g++)
    {
      SimplifiedGrainRepresentation<dim> simplified_grain_representation(
        grain_sets.at(g));

      pcout << "Grain: " << simplified_grain_representation.getGrainId() << " "
            << simplified_grain_representation.getOrderParameterId()
            << " Center: " << simplified_grain_representation.getCenter()(0) << " "
            << simplified_grain_representation.getCenter()(1) << std::endl;

      simplified_grain_representations.push_back(simplified_grain_representation);
    }

  SimplifiedGrainManipulator<dim> simplified_grain_manipulator;

  if (currentIncrement > 0 || userInputs.load_grain_structure)
    {
      simplified_grain_manipulator.transferGrainIds(old_grain_representations,
                                                    simplified_grain_representations);
    }

  simplified_grain_manipulator.reassignGrains(simplified_grain_representations,
                                              userInputs.buffer_between_grains,
                                              userInputs.variables_for_remapping);

  for (unsigned int g = 0; g < this->simplified_grain_representations.size(); g++)
    {
      pcout << "Grain: " << simplified_grain_representations[g].getGrainId() << " "
            << simplified_grain_representations[g].getOrderParameterId()
            << " Center: " << simplified_grain_representations[g].getCenter()(0) << " "
            << simplified_grain_representations[g].getCenter()(1) << std::endl;
    }

  OrderParameterRemapper<dim> order_parameter_remapper;
  order_parameter_remapper.remap(simplified_grain_representations,
                                 solutionSet,
                                 *dofHandlersSet_nonconst.at(scalar_field_index),
                                 FESet.at(scalar_field_index)->dofs_per_cell);

  pcout << "Reassigning grains completed.\n\n";

  // end log
  computing_timer.leave_subsection("matrixFreePDE: reassignGrains");
}

template class MatrixFreePDE<2, 1>;
template class MatrixFreePDE<3, 1>;

template class MatrixFreePDE<2, 2>;
template class MatrixFreePDE<3, 2>;

template class MatrixFreePDE<3, 3>;
template class MatrixFreePDE<2, 3>;

template class MatrixFreePDE<3, 4>;
template class MatrixFreePDE<2, 4>;

template class MatrixFreePDE<3, 5>;
template class MatrixFreePDE<2, 5>;

template class MatrixFreePDE<3, 6>;
template class MatrixFreePDE<2, 6>;