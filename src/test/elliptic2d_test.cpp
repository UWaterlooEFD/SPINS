/* Tests of the 2D elliptic solver on mapped grids (test U11 of the verification plan): GMRES with
   the finite-difference multigrid preconditioner, as ESolver uses it for the pressure and viscous
   solves of mapped and Chebyshev x Chebyshev cases.

   The grid is periodic in x (Lx = 2) over a cosine hill h(x) = A (1 + cos(pi x)) / 2 with a flat
   lid at z = 1 and Chebyshev z, and the Jacobian is built the way FluidEvolve builds it for a
   mapped case. Two measurements:
   - the convergence factor of Richardson iteration u <- u + P(f - A u), with A the spectral
     operator and P one preconditioner cycle, which measures how well the multigrid cycle
     approximates the inverse of the operator GMRES actually solves;
   - poisson_2d on a manufactured solution, Dirichlet and Neumann (pressure-like), giving the
     iteration count, the final residual and the solution error.
   Run with SPINS_TEST_REPORT=1 to print the tables. */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "gmres_2d_solver.hpp"
#include "gmres_2d_solver_impl.hpp"
#include "test_helpers.hpp"

using namespace TArrayn;
using namespace Transformer;
using spins_test::report;

namespace {
    blitz::firstIndex ii;
    blitz::thirdIndex kk;

    const double Lx = 2.0, H = 1.0;

    // A mapped 2D grid (Nx x 1 x Nz): x_i = (i + 1/2) Lx / Nx, z = h(x) + (H - h(x)) (1 - b) / 2 with
    // b_k = cos(k pi / (Nz - 1)), so k = 0 is the bottom. The Jacobian is computed as FluidEvolve
    // computes it: spectral derivatives of the grids, with x's linear part removed, then inverted.
    struct MappedGrid {
        int                      Nx, Nz;
        double                   amp;
        std::unique_ptr<Grad>    grad;
        std::unique_ptr<DTArray> x, z;
        double                   h(double xv) const { return amp * (1 + std::cos(M_PI * xv)) / 2; }
        double                   hx(double xv) const { return -amp * M_PI * std::sin(M_PI * xv) / 2; }

        MappedGrid(int nx, int nz, double a) : Nx(nx), Nz(nz), amp(a) {
            auto alloc = [&]() { return std::unique_ptr<DTArray>(alloc_array(Nx, 1, Nz)); };
            x = alloc(), z = alloc();
            auto xa = alloc(), xb = alloc(), za = alloc(), zb = alloc(), xg = alloc(), jac = alloc(), det = alloc();
            for (int i = x->lbound(firstDim); i <= x->ubound(firstDim); i++)
                for (int k = 0; k < Nz; k++) {
                    const double xv = (i + 0.5) * Lx / Nx, b = std::cos(k * M_PI / (Nz - 1));
                    (*x)(i, 0, k) = xv;
                    (*z)(i, 0, k) = h(xv) + (H - h(xv)) * (1 - b) / 2;
                }
            grad = std::make_unique<Grad>(Nx, 1, Nz, FOURIER, FOURIER, CHEBY);
            grad->set_jac(firstDim, firstDim, 1);
            grad->set_jac(secondDim, secondDim, 1);
            grad->set_jac(thirdDim, thirdDim, 1);
            *xg = *x - Lx * (0.5 + ii) / Nx; // FluidEvolve removes the linear part for trig x
            grad->setup_array(xg.get(), FOURIER, FOURIER, CHEBY);
            grad->get_dx(xa.get());
            *xa = *xa + Lx / (2 * M_PI);
            grad->get_dz(xb.get());
            grad->setup_array(z.get(), FOURIER, FOURIER, CHEBY);
            grad->get_dx(za.get());
            grad->get_dz(zb.get());
            *det = (*xa) * (*zb) - (*xb) * (*za);
            *jac = *zb / *det;
            grad->set_jac(firstDim, firstDim, 0, jac.get());
            *jac = -*za / *det;
            grad->set_jac(firstDim, thirdDim, 0, jac.get());
            *jac = -*xb / *det;
            grad->set_jac(thirdDim, firstDim, 0, jac.get());
            *jac = *xa / *det;
            grad->set_jac(thirdDim, thirdDim, 0, jac.get());
            grad->set_jac(secondDim, secondDim, 1);
        }
    };

    // Manufactured solution u = cos(pi x) e^z + z^2 and its derivatives
    double u_ex(double x, double z) {
        return std::cos(M_PI * x) * std::exp(z) + z * z;
    }
    double ux_ex(double x, double z) {
        return -M_PI * std::sin(M_PI * x) * std::exp(z);
    }
    double uz_ex(double x, double z) {
        return std::cos(M_PI * x) * std::exp(z) + 2 * z;
    }
    double lap_ex(double x, double z) {
        return (1 - M_PI * M_PI) * std::cos(M_PI * x) * std::exp(z) + 2;
    }

    // Grids are kept for the whole run: poisson_2d and Cheb_2dmg rebuild their preconditioner only
    // when the Grad pointer changes, so a new Grad must never reuse a freed one's address.
    std::vector<std::unique_ptr<MappedGrid>> grids;
    MappedGrid                              &new_grid(int nx, int nz, double a) {
                                     grids.push_back(std::make_unique<MappedGrid>(nx, nz, a));
                                     return *grids.back();
    }

    // Convergence factor of preconditioned Richardson iteration on a Dirichlet problem
    double richardson_factor(MappedGrid &g, int iters, double &first) {
        Cheb_2dmg op(g.Nx, g.Nz);
        op.set_grad(g.grad.get());
        op.set_bc(0, 1, 0, FOURIER, 0, 0);
        ubox *u = op.alloc_basis(), *uex = op.alloc_basis(), *du = op.alloc_basis();
        fbox *f = op.alloc_resid(), *r = op.alloc_resid(), *Au = op.alloc_resid();
        for (int i = g.x->lbound(firstDim); i <= g.x->ubound(firstDim); i++)
            for (int k = 0; k < g.Nz; k++) (*uex->gridbox)(i, 0, k) = u_ex((*g.x)(i, 0, k), (*g.z)(i, 0, k));
        uex->sigma = 0, u->sigma = 0;
        op.matrix_multiply(uex, f);
        f->mean = 0;
        op.basis_scale(u, 0);
        double r0 = 0, rk = 0;
        for (int it = 0; it <= iters; it++) {
            op.matrix_multiply(u, Au);
            op.resid_copy(r, f);
            op.resid_fma(r, Au, -1);
            rk = std::sqrt(op.resid_dot(r, r));
            if (it == 0) r0 = rk;
            if (it == 1) first = rk / r0;
            if (it == iters) break;
            op.precondition(r, du);
            op.basis_fma(u, du, 1);
        }
        op.free_basis(u), op.free_basis(uex), op.free_basis(du);
        op.free_resid(f), op.free_resid(r), op.free_resid(Au);
        return std::pow(rk / r0, 1.0 / iters);
    }

    struct SolveResult {
        int    its;
        double err;
    };

    // poisson_2d on the manufactured solution; neumann = false gives Dirichlet rows
    SolveResult solve_manufactured(MappedGrid &g, bool neumann) {
        auto f  = std::unique_ptr<DTArray>(alloc_array(g.Nx, 1, g.Nz));
        auto u  = std::unique_ptr<DTArray>(alloc_array(g.Nx, 1, g.Nz));
        auto ue = std::unique_ptr<DTArray>(alloc_array(g.Nx, 1, g.Nz));
        *u      = 0.0;
        for (int i = g.x->lbound(firstDim); i <= g.x->ubound(firstDim); i++)
            for (int k = 0; k < g.Nz; k++) {
                const double xv = (*g.x)(i, 0, k), zv = (*g.z)(i, 0, k);
                (*ue)(i, 0, k) = u_ex(xv, zv);
                (*f)(i, 0, k)  = lap_ex(xv, zv);
                if (k == 0) // bottom: outward normal (h', -1) / |.|
                    (*f)(i, 0, k) = neumann ? (g.hx(xv) * ux_ex(xv, zv) - uz_ex(xv, zv)) / std::sqrt(1 + g.hx(xv) * g.hx(xv)) : u_ex(xv, zv);
                if (k == g.Nz - 1) (*f)(i, 0, k) = neumann ? uz_ex(xv, zv) : u_ex(xv, zv);
            }
        const int its = neumann ? poisson_2d(*f, *u, g.grad.get(), FOURIER, 0, 0, 1) : poisson_2d(*f, *u, g.grad.get(), FOURIER, 0, 1, 0);
        if (neumann) { // solution defined up to a constant: compare with means removed
            const double n = double(g.Nx) * g.Nz;
            *u             = *u - pssum(sum(*u)) / n;
            *ue            = *ue - pssum(sum(*ue)) / n;
        }
        return {its, psmax(max(abs(*u - *ue))) / psmax(max(abs(*ue)))};
    }
} // namespace

// Configurations, with values measured on 2026-10-24 (with the line_solve fix; results do not
// depend on it when each configuration runs alone). One test each so that a failing or slow
// configuration can run alone. Stored values make preconditioner changes visible: the factor-2
// cross-derivative change, for example, moves the hill first-step ratios to about 0.015.
struct Config {
    double amp;     // hill amplitude (depth 1, period 2)
    int    n;       // grid n x (n + 1)
    double first;   // Richardson residual ratio after one step
    double mean;    // mean Richardson factor over 10 steps
    int    its_dir; // poisson_2d iterations, Dirichlet
    int    its_neu; // poisson_2d iterations, Neumann
};
std::string config_name(const testing::TestParamInfo<Config> &info) {
    return "hill" + std::to_string(int(10 * info.param.amp)) + "_N" + std::to_string(info.param.n);
}
const Config configs[] = {
    {0.0, 32,  0.014, 0.665, 7,  5 },
    {0.0, 64,  0.013, 0.713, 7,  5 },
    {0.0, 128, 0.011, 0.733, 7,  5 },
    {0.2, 32,  0.046, 0.671, 22, 20},
    {0.2, 64,  0.045, 0.720, 22, 16},
    {0.4, 32,  0.091, 0.683, 26, 28},
    {0.4, 64,  0.091, 0.749, 26, 25},
    {0.4, 128, 0.091, 0.779, 26, 19},
};
const int    its_tol = 2;    // iteration counts may move by one or two with round-off (FFTW plans)
const double err_tol = 2e-6; // GMRES stops at relative residual 1e-5; measured errors <= 5.1e-7

class Elliptic2D : public testing::TestWithParam<Config> { };

// The multigrid cycle as a preconditioner for the spectral operator. Richardson iteration
// u <- u + P(f - A u) must converge at a rate that barely degrades with N.
TEST_P(Elliptic2D, PreconditionerRichardsonFactor) {
    const Config c     = GetParam();
    MappedGrid  &g     = new_grid(c.n, c.n + 1, c.amp);
    double       first = 0;
    const double rho   = richardson_factor(g, 10, first);
    if (report() && master()) printf("Richardson: hill %.1f, %d x %d: first-step ratio %.3f, mean factor over 10 steps %.3f\n", c.amp, c.n, c.n + 1, first, rho);
    EXPECT_NEAR(first, c.first, 0.01);
    EXPECT_NEAR(rho, c.mean, 0.05);
}

// Iteration counts and errors of the full solve, Dirichlet and Neumann (pressure-like, indefinite)
TEST_P(Elliptic2D, DirichletSolve) {
    const Config      c = GetParam();
    const SolveResult s = solve_manufactured(new_grid(c.n, c.n + 1, c.amp), false);
    if (report() && master()) printf("poisson_2d Dirichlet: hill %.1f, %d x %d: %d iterations, max relative error %.3e\n", c.amp, c.n, c.n + 1, s.its, s.err);
    EXPECT_NEAR(s.its, c.its_dir, its_tol);
    EXPECT_LT(s.err, err_tol);
}

TEST_P(Elliptic2D, NeumannSolve) {
    const Config      c = GetParam();
    const SolveResult s = solve_manufactured(new_grid(c.n, c.n + 1, c.amp), true);
    if (report() && master()) printf("poisson_2d Neumann: hill %.1f, %d x %d: %d iterations, max relative error %.3e\n", c.amp, c.n, c.n + 1, s.its, s.err);
    EXPECT_NEAR(s.its, c.its_neu, its_tol);
    EXPECT_LT(s.err, err_tol);
}

INSTANTIATE_TEST_SUITE_P(Hills, Elliptic2D, testing::ValuesIn(configs), config_name);

// Free the grids before main() calls MPI_Finalize (their transforms hold MPI resources).
class GridCleanup : public testing::Environment {
  public:
    void TearDown() override { grids.clear(); }
};
testing::Environment *const grid_cleanup = testing::AddGlobalTestEnvironment(new GridCleanup);
