#include "mpi.h"
#include <gtest/gtest.h>
#include <blitz/array.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gmres.hpp"
#include "gmres_1d_solver.hpp"
#include "gmres_1d_solver_impl.hpp"
#include "test_helpers.hpp"

/* Tests of the 1D Chebyshev Poisson/Helmholtz solver that ESolver uses for each horizontal
   wavenumber on an unmapped Chebyshev z grid: u'' - h u = f on a domain of length L, with
   a_bot u - b_bot u' at the bottom and a_top u + b_top u' at the top (outward normals). Grid
   point j is z_j = -(L/2) cos(pi j/(N-1)), so j = 0 is the bottom.

   poisson_1d keeps static state that fixes N and L for the whole process, so most tests drive
   the production kernel (cheb_d2) and GMRES_Solver directly, at tight tolerance; one test calls
   the poisson_1d wrapper itself at a single N. SPINS_TEST_REPORT=1 prints iteration counts. */

using spins_test::cheb_values;
using spins_test::eps;
using spins_test::report;

namespace {
    struct BC {
        double      a, b;
        const char *name;
    };
    constexpr BC dirichlet{1.0, 0.0, "Dirichlet"}, robin{M_SQRT1_2, M_SQRT1_2, "Robin"}, neumann{0.0, 1.0, "Neumann"};

    // Manufactured solution with a full-amplitude top Chebyshev mode, which the grid represents
    // exactly: u = T_(N-1)(s) + s^2/2 + s/3, s = 2z/L.
    void exact(int N, double L, double z, double &u, double &du, double &d2u) {
        const double s = 2 * z / L;
        double       T, dT, d2T;
        cheb_values(N - 1, s, T, dT, d2T);
        u   = T + 0.5 * s * s + s / 3;
        du  = (dT + s + 1.0 / 3) * (2 / L);
        d2u = (d2T + 1) * (4 / (L * L));
    }

    double zgrid(int N, double L, int j) {
        return -0.5 * L * std::cos(M_PI * j / (N - 1));
    }

    struct Solve {
        int                 iterations;
        std::vector<double> u, expect;
        double              extra, scale, fscale; // scale = max|expect|, fscale = max|rhs|
    };

    // Builds the right-hand side for `exact`, solves with the production kernel, and returns the
    // solution next to the expected one (mean-removed for the singular pure-Neumann problem).
    Solve solve(int N, double L, BC bc, double h, double tol) {
        cheb_d2 kernel(N, L);
        kernel.set_bc(h, bc.a, bc.a, bc.b, bc.b);
        GMRES_Solver<cheb_d2> gmres(&kernel);
        cheb_d2::RType        r = kernel.alloc_resid();
        cheb_d2::BType        x = kernel.alloc_basis();
        Solve                 out{0, std::vector<double>(N), std::vector<double>(N), 0, 0, 0};
        for (int j = 0; j < N; j++) {
            double u, du, d2u;
            exact(N, L, zgrid(N, L, j), u, du, d2u);
            out.expect[j]        = u;
            (*r->zline)(0, 0, j) = d2u - h * u;
            if (j == 0) (*r->zline)(0, 0, j) = bc.a * u - bc.b * du;
            if (j == N - 1) (*r->zline)(0, 0, j) = bc.a * u + bc.b * du;
        }
        for (int j = 0; j < N; j++) out.fscale = std::fmax(out.fscale, std::fabs((*r->zline)(0, 0, j)));
        r->extra       = 0;
        out.iterations = gmres.Solve(x, r, tol, 20, 10);
        for (int j = 0; j < N; j++) out.u[j] = (*x->zline)(0, 0, j);
        out.extra = x->extra;
        if (bc.a == 0 && h == 0) { // solution fixed only up to a constant: compare mean-removed
            double mean = 0;
            for (double v : out.expect) mean += v / N;
            for (double &v : out.expect) v -= mean;
        }
        for (double v : out.expect) out.scale = std::fmax(out.scale, std::fabs(v));
        kernel.free_resid(r);
        kernel.free_basis(x);
        return out;
    }

    double max_err(const Solve &s) {
        double e = 0;
        for (size_t j = 0; j < s.u.size(); j++) e = std::fmax(e, std::fabs(s.u[j] - s.expect[j]));
        return e / s.scale;
    }
} // namespace

// U7 + U9: solutions the grid represents exactly come back to the solver tolerance, for every
// boundary condition, Helmholtz shifts from 0 to 1e8, two domain lengths and even and odd N.
// For pure Neumann with h = 0, the bordered system must also pin the mean and leave extra ~ 0.
TEST(Cheb1DSolver, RepresentableSolutionsAreExact) {
    // At GMRES tolerance 1e-13 every case converges in <= 20 iterations. The worst error (N = 64,
    // L = 0.5) varies between runs because FFTW_MEASURE can pick different plans: 2.8e-10 typically,
    // up to 1.0e-9 in 80 samples, inside N^4 eps = 3.7e-9; a wrong operator gives errors of order 1e-2.
    const double gmres_tol = 1e-13, tol = 5e-9; // relative max error
    double       worst = 0;
    for (int N : {16, 17, 32, 33, 64}) {
        for (double L : {2.0, 0.5}) {
            for (BC bc : {dirichlet, robin, neumann}) {
                for (double h : {0.0, 1.0, 1e2, 1e4, 1e6, 1e8}) {
                    const Solve s = solve(N, L, bc, h, gmres_tol);
                    worst         = std::fmax(worst, max_err(s));
                    EXPECT_GT(s.iterations, 0) << bc.name << ", N = " << N << ", L = " << L << ", h = " << h << ": GMRES did not converge";
                    EXPECT_LT(max_err(s), tol) << bc.name << ", N = " << N << ", L = " << L << ", h = " << h;
                    if (bc.a == 0 && h == 0) {
                        double mean = 0;
                        for (double v : s.u) mean += v / N;
                        EXPECT_LT(std::fabs(mean), 10 * N * eps * s.scale) << "pure Neumann mean, N = " << N << ", L = " << L;
                        EXPECT_LT(std::fabs(s.extra), tol * s.fscale) << "pure Neumann compatibility constant, N = " << N << ", L = " << L;
                    }
                }
            }
        }
    }
    if (report()) std::printf("[ report ] 1D solver, representable solutions: worst relative error %.3g\n", worst);
}

// U8: the banded finite-difference preconditioner must be built on the solver's own grid. Its
// three-point second difference is exact for quadratics, and its one-sided boundary derivative
// is exact for linears, so applying the preconditioner to the operator's image of z^2
// (Dirichlet) or z (Robin) must return that function, to round-off of the banded solve
// (which grows like N^2). Built on a grid twice the domain length, it returns 4z^2 - 3.
TEST(Cheb1DPreconditioner, ExactOnLowDegreePolynomials) {
    for (int N : {16, 33, 64}) {
        const double tol = 100 * double(N) * N * eps;
        for (double L : {2.0, 0.5}) {
            for (double h : {0.0, 1.0, 1e2}) {
                for (BC bc : {dirichlet, robin}) {
                    const bool quad = (bc.b == 0); // z^2 for Dirichlet, z for Robin
                    cheb_d2    kernel(N, L);
                    kernel.set_bc(h, bc.a, bc.a, bc.b, bc.b);
                    cheb_d2::RType r     = kernel.alloc_resid();
                    cheb_d2::BType x     = kernel.alloc_basis();
                    double         scale = 0;
                    for (int j = 0; j < N; j++) {
                        const double z = zgrid(N, L, j), u = quad ? z * z : z, du = quad ? 2 * z : 1, d2u = quad ? 2 : 0;
                        (*r->zline)(0, 0, j) = d2u - h * u;
                        if (j == 0) (*r->zline)(0, 0, j) = bc.a * u - bc.b * du;
                        if (j == N - 1) (*r->zline)(0, 0, j) = bc.a * u + bc.b * du;
                        scale = std::fmax(scale, std::fabs(u));
                    }
                    kernel.precondition(r, x);
                    double err = 0;
                    for (int j = 0; j < N; j++) {
                        const double z = zgrid(N, L, j);
                        err            = std::fmax(err, std::fabs((*x->zline)(0, 0, j) - (quad ? z * z : z)));
                    }
                    EXPECT_LT(err / scale, tol) << bc.name << (quad ? ", u = z^2" : ", u = z") << ", N = " << N << ", L = " << L << ", h = " << h;
                    kernel.free_resid(r);
                    kernel.free_basis(x);
                }
            }
        }
    }
}

// U10: with a preconditioner that is consistent with the operator, adding a Helmholtz shift h
// cannot slow GMRES down: the shift enters both exactly on the diagonal, and P^-1 A tends to the
// identity as h grows. A mis-scaled preconditioner breaks this at intermediate h, while leaving
// iteration counts at h = 0 unchanged (they are invariant to a constant rescaling of P).
TEST(Cheb1DSolver, HelmholtzShiftDoesNotSlowConvergence) {
    const double gmres_tol = 1e-10;
    const int    slack     = 2; // iterations
    for (int N : {32, 64, 128}) {
        for (BC bc : {dirichlet, robin}) {
            const int base = solve(N, 2.0, bc, 0.0, gmres_tol).iterations;
            if (report()) std::printf("[ report ] %-9s N = %3d iterations: h=0 %d", bc.name, N, base);
            for (double h : {1.0, 1e2, 1e4, 1e6}) {
                const int its = solve(N, 2.0, bc, h, gmres_tol).iterations;
                if (report()) std::printf(", h=%g %d", h, its);
                EXPECT_GT(its, 0) << bc.name << ", N = " << N << ", h = " << h << ": GMRES did not converge";
                EXPECT_LE(its, base + slack) << bc.name << ", N = " << N << ": h = " << h << " needs " << its << " iterations against " << base << " at h = 0";
            }
            if (report()) std::printf("\n");
        }
    }
}

// The poisson_1d wrapper itself (production tolerance 1e-7), at the one N it may use per process.
TEST(Poisson1DWrapper, SolvesRepresentableSolutions) {
    const int    N = 32;
    const double L = 2.0, tol = 1e-5; // relative max error at GMRES tolerance 1e-7
    for (BC bc : {dirichlet, robin, neumann}) {
        for (double h : {0.0, 1.0, 1e2, 1e4}) {
            blitz::Array<double, 1> resid(N), soln(N);
            std::vector<double>     expect(N);
            double                  scale = 0, mean = 0;
            for (int j = 0; j < N; j++) {
                double u, du, d2u;
                exact(N, L, zgrid(N, L, j), u, du, d2u);
                expect[j] = u;
                resid(j)  = d2u - h * u;
                if (j == 0) resid(j) = bc.a * u - bc.b * du;
                if (j == N - 1) resid(j) = bc.a * u + bc.b * du;
                mean += u / N;
            }
            const int its = poisson_1d(resid, soln, L, h, bc.a, bc.a, bc.b, bc.b);
            double    err = 0;
            for (int j = 0; j < N; j++) {
                const double e = (bc.a == 0 && h == 0) ? expect[j] - mean : expect[j];
                err            = std::fmax(err, std::fabs(soln(j) - e));
                scale          = std::fmax(scale, std::fabs(e));
            }
            EXPECT_GT(its, 0) << bc.name << ", h = " << h;
            EXPECT_LT(err / scale, tol) << bc.name << ", h = " << h;
        }
    }
}
