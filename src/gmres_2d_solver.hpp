#include <blitz/array.h>
#include "TArray.hpp"
#include "grad.hpp"
#include "Parformer.hpp"

/* Public interface of the 2D Poisson solver, simplified to assume uniform
   boundary conditions at all 4 (2 in a symmetric case) walls.

   The return value follows the GMRES_Solver::Solve convention documented
   at src/gmres.hpp: > 0 converged, and the value is the total inner
   iteration count; < 0 NOT converged, with the same count as the
   magnitude; the all-zero right-hand side short-circuits to 1.  A
   non-positive return means soln is not a solution of the system, so a
   caller must end the run with a meaningful error rather than use it. */
int poisson_2d(TArrayn::DTArray &resid, TArrayn::DTArray &soln, TArrayn::Grad *gradient, Transformer::S_EXP type_x, double helmholtz, double zbc_a, double zbc_b, double xbc_a = 0, double xbc_b = 0);
