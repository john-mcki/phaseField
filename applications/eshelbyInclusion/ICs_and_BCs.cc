// ===========================================================================
// FUNCTION FOR INITIAL CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                                            [[maybe_unused]] const unsigned int index,
                                            [[maybe_unused]] double            &scalar_IC,
                                            [[maybe_unused]] Vector<double>    &vector_IC)
{
  // ---------------------------------------------------------------------
  // ENTER THE INITIAL CONDITIONS HERE
  // ---------------------------------------------------------------------
  // Enter the function describing conditions for the fields at point "p".
  // Use "if" statements to set the initial condition for each variable
  // according to its variable index

  for (unsigned int d = 0; d < dim; d++)
    {
      vector_IC(d) = 0.0;
    }

  // --------------------------------------------------------------------------
}

// ===========================================================================
// FUNCTION FOR NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS
// ===========================================================================

// First we need a Kronecker Delta
int
kDelta(int i, int j)
{
  if (i == j)
    {
      return 1;
    }
  else
    {
      return 0;
    }
}

template <int dim, int degree>
void
customPDE<dim, degree>::setNonUniformDirichletBCs(
  [[maybe_unused]] const Point<dim>  &p,
  [[maybe_unused]] const unsigned int index,
  [[maybe_unused]] const unsigned int direction,
  [[maybe_unused]] const double       time,
  [[maybe_unused]] double            &scalar_BC,
  [[maybe_unused]] Vector<double>    &vector_BC)
{
  // --------------------------------------------------------------------------
  // ENTER THE NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS HERE
  // --------------------------------------------------------------------------
  // Enter the function describing conditions for the fields at point "p".
  // Use "if" statements to set the boundary condition for each variable
  // according to its variable index. This function can be left blank if there
  // are no non-uniform Dirichlet boundary conditions. For BCs that change in
  // time, you can access the current time through the variable "time". The
  // boundary index can be accessed via the variable "direction", which starts
  // at zero and uses the same order as the BC specification in parameters.in
  // (i.e. left = 0, right = 1, bottom = 2, top = 3, front = 4, back = 5).

  // -------------------------------------------------------------------------

  for (unsigned int i = 0; i < dim; i++)
    {
      double A = (incRadius * incRadius * incRadius) /
                 (6 * (1 - poisson)); // All constants for the displacement equation
      double dist =
        sqrt((p[0] - centerX) * (p[0] - centerX) + (p[1] - centerY) * (p[1] - centerY) +
             (p[2] - centerZ) * (p[2] - centerZ)); // distance from center of inclusion
      // double sfts = 0.01;
      double G = 0.0;
      for (unsigned int j = 0; j < dim; j++)
        {
          for (unsigned int k = 0; k < dim; k++)
            {
              double g = (1 - 2 * poisson) * (kDelta(i, j) * ((p[k] - centerZ) / dist) +
                                              kDelta(i, k) * ((p[j] - centerY) / dist) -
                                              kDelta(j, k) * ((p[i] - centerX) / dist)) +
                         3 * ((p[i] - centerX) / dist) * ((p[j] - centerY) / dist) *
                           ((p[k] - centerZ) / dist);
              double sfts = kDelta(j, k) * 0.01;
              G += sfts * g;
            }
        }
      vector_BC(i) = -1.0 * A * (1 / (dist * dist)) * G;
    }
}
