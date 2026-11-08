#include <mpi.h>
#include <blitz/array.h>
#include "../gmres_1d_solver.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Tests the production 1D Chebyshev Poisson/Helmholtz solver (poisson_1d, used by ESolver for
   each horizontal wavenumber on a Chebyshev z grid) against a manufactured solution.

   Domain z in [-1, 1] (length 2); grid point j is z_j = -cos(pi j/(N-1)), so j = 0 is the
   bottom. Problem: u'' - h u = f, with a_bot u - b_bot u' at the bottom and a_top u + b_top u'
   at the top (outward normals). Exact solution u = cos(pi z) + z^2/2.

   For pure Neumann with h = 0 the solver solves the bordered system u'' + c = f, mean_j(u) = 0,
   so the error is measured against u - mean_j(u).

   Usage: test_poisson1d.x [N] (default 32). poisson_1d keeps static state, so one N per run. */

using blitz::Array;

static double u_exact(double z) {
    return cos(M_PI * z) + 0.5 * z * z;
}
static double u_exact_first_deriv(double z) {
    return -M_PI * sin(M_PI * z) + z;
}
static double u_exact_second_deriv(double z) {
    return -M_PI * M_PI * cos(M_PI * z) + 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int N = 0;
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0) N = 32;
    Array<double, 1> z(N), resid(N), soln(N), exact(N);
    for (int j = 0; j < N; j++) z(j) = -cos(M_PI * j / (N - 1));

    const double s2       = 1 / sqrt(2.0);
    const double bcs[][2] = {
        {1,  0 },
        {s2, s2},
        {0,  1 }
    }; // (a, b): Dirichlet, Robin, Neumann
    const char  *names[] = {"Dirichlet", "Robin", "Neumann"};
    const double helms[] = {0, 1, 100, 1e4, 1e6};

    for (int b = 0; b < 3; b++) {
        for (int h = 0; h < 5; h++) {
            double a = bcs[b][0], bb = bcs[b][1], helm = helms[h];
            for (int j = 0; j < N; j++) {
                resid(j) = u_exact_second_deriv(z(j)) - helm * u_exact(z(j));
                exact(j) = u_exact(z(j));
            }
            resid(0)     = a * u_exact(-1) - bb * u_exact_first_deriv(-1);
            resid(N - 1) = a * u_exact(1) + bb * u_exact_first_deriv(1);
            soln         = 0;
            int itc      = poisson_1d(resid, soln, 2.0, helm, a, a, bb, bb);
            if (a == 0 && helm == 0) exact = exact - mean(exact);
            double err = max(abs(soln - exact));
            printf("N=%3d %-9s h=%5g: iterations %4d, max error %.3e, mean(soln) %.3e\n", N, names[b], helm, itc, err, mean(soln));
        }
    }
    MPI_Finalize();
    return 0;
}
