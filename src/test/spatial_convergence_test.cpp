/* Spatial convergence on functions no grid represents exactly (study N7 of the verification
   plan): errors must fall exponentially with N until they reach a round-off plateau, about N eps
   for Fourier derivatives and N^2 eps (first derivative) or N^4 eps (second) for Chebyshev ones.
   The 2D Chebyshev x Chebyshev Poisson solve (Haidvogel & Zang 1979, u = sin 4 pi x sin 4 pi z)
   instead levels off where its GMRES stops (relative residual 1e-5 in poisson_2d).
   SPINS_TEST_REPORT=1 prints the error tables. */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "Parformer.hpp"
#include "T_util.hpp"
#include "gmres_2d_solver.hpp"
#include "grad.hpp"
#include "test_helpers.hpp"

using namespace TArrayn;
using namespace Transformer;
using spins_test::eps;
using spins_test::make_local_line;
using spins_test::report;

namespace {
    struct Fn {
        std::function<double(double)> f, d1, d2;
    };

    // Max errors of d/dz and d2/dz2 relative to max |f'| and max |f''| on a line
    void line_errors(const Fn &fn, int N, S_EXP type, double &e1, double &e2) {
        auto                f = make_local_line(N), d1 = make_local_line(N), d2 = make_local_line(N);
        Trans1D             tform(1, 1, N, TArrayn::thirdDim, type, MPI_COMM_SELF);
        std::vector<double> z(N);
        for (int k = 0; k < N; k++) z[k] = (type == CHEBY) ? std::cos(M_PI * k / (N - 1)) : 2 * M_PI * k / N;
        for (int k = 0; k < N; k++) (*f)(0, 0, k) = fn.f(z[k]);
        if (type == CHEBY) {
            deriv_cheb(*f, tform, *d1);
            deriv_cheb(*d1, tform, *d2);
        } else {
            deriv_fft(*f, tform, *d1);
            deriv_fft(*d1, tform, *d2);
        }
        double s1 = 0, s2 = 0;
        e1 = e2 = 0;
        for (int k = 0; k < N; k++) {
            s1 = std::fmax(s1, std::fabs(fn.d1(z[k]))), s2 = std::fmax(s2, std::fabs(fn.d2(z[k])));
            e1 = std::fmax(e1, std::fabs((*d1)(0, 0, k) - fn.d1(z[k])));
            e2 = std::fmax(e2, std::fabs((*d2)(0, 0, k) - fn.d2(z[k])));
        }
        e1 /= s1, e2 /= s2;
    }

    // Errors over N must fall by at least `drop` from the first N to the plateau, and the plateau
    // (the last N) must sit under the round-off bound
    void check_convergence(const char *what, const std::vector<int> &Ns, const std::vector<double> &err, double drop, double plateau) {
        if (report() && master()) {
            printf("%s:", what);
            for (size_t i = 0; i < Ns.size(); i++) printf(" N %d %.2e;", Ns[i], err[i]);
            printf(" plateau bound %.1e\n", plateau);
        }
        EXPECT_LT(err.back(), plateau) << what;
        EXPECT_LT(err.back(), err.front() / drop) << what;
    }
} // namespace

// exp(sin x) is entire and periodic: Fourier coefficients fall faster than exponentially
TEST(SpatialConvergence, FourierExpSin) {
    const Fn            fn{[](double x) { return std::exp(std::sin(x)); }, [](double x) { return std::cos(x) * std::exp(std::sin(x)); }, [](double x) { return (std::cos(x) * std::cos(x) - std::sin(x)) * std::exp(std::sin(x)); }};
    std::vector<int>    Ns = {8, 12, 16, 24, 32, 48, 64};
    std::vector<double> e1, e2;
    for (int N : Ns) {
        double a, b;
        line_errors(fn, N, FOURIER, a, b);
        e1.push_back(a), e2.push_back(b);
    }
    check_convergence("Fourier d/dx exp(sin x)", Ns, e1, 1e8, 20 * 64 * eps);
    check_convergence("Fourier d2/dx2 exp(sin x)", Ns, e2, 1e8, 20 * 64 * 64 * eps);
}

// tanh(5z) (poles at +-i pi/10) and exp(-20 z^2) (entire, but narrow) on Gauss-Lobatto points
TEST(SpatialConvergence, ChebyshevTanhAndGaussian) {
    const Fn               tanh5{[](double z) { return std::tanh(5 * z); }, [](double z) { return 5 / std::pow(std::cosh(5 * z), 2); }, [](double z) { return -50 * std::tanh(5 * z) / std::pow(std::cosh(5 * z), 2); }};
    const Fn               gauss{[](double z) { return std::exp(-20 * z * z); }, [](double z) { return -40 * z * std::exp(-20 * z * z); }, [](double z) { return (1600 * z * z - 40) * std::exp(-20 * z * z); }};
    const std::vector<int> Ns = {16, 24, 32, 48, 64, 96, 128};
    for (auto [name, fn] : {std::make_pair("tanh(5z)", tanh5), std::make_pair("exp(-20 z^2)", gauss)}) {
        std::vector<double> e1, e2;
        for (int N : Ns) {
            double a, b;
            line_errors(fn, N, CHEBY, a, b);
            e1.push_back(a), e2.push_back(b);
        }
        const double N = Ns.back();
        check_convergence((std::string("Chebyshev d/dz ") + name).c_str(), Ns, e1, 1e6, 10 * N * N * eps);
        check_convergence((std::string("Chebyshev d2/dz2 ") + name).c_str(), Ns, e2, 1e4, 10 * N * N * N * N * eps);
    }
}

// Haidvogel & Zang: lap u = -32 pi^2 u with u = sin 4 pi x sin 4 pi z = 0 on the walls of [-1, 1]^2,
// through poisson_2d (GMRES with the multigrid preconditioner) on a Chebyshev x Chebyshev grid.
TEST(SpatialConvergence, ChebyshevPoisson2DHaidvogelZang) {
    std::vector<int>                   Ns = {16, 24, 32, 48, 64};
    std::vector<double>                err;
    std::vector<std::unique_ptr<Grad>> keep; // poisson_2d caches its preconditioner per Grad address
    for (int N : Ns) {
        keep.push_back(std::make_unique<Grad>(N, 1, N, CHEBY, FOURIER, CHEBY));
        Grad &g = *keep.back();
        g.set_jac(firstDim, firstDim, 1), g.set_jac(secondDim, secondDim, 1), g.set_jac(thirdDim, thirdDim, 1);
        std::unique_ptr<DTArray> f(alloc_array(N, 1, N)), u(alloc_array(N, 1, N)), ue(alloc_array(N, 1, N));
        for (int i = f->lbound(firstDim); i <= f->ubound(firstDim); i++)
            for (int k = 0; k < N; k++) {
                const double x = std::cos(M_PI * i / (N - 1)), z = std::cos(M_PI * k / (N - 1));
                (*ue)(i, 0, k) = std::sin(4 * M_PI * x) * std::sin(4 * M_PI * z);
                (*f)(i, 0, k)  = (i == 0 || i == N - 1 || k == 0 || k == N - 1) ? 0 : -32 * M_PI * M_PI * (*ue)(i, 0, k);
            }
        poisson_2d(*f, *u, &g, CHEBY, 0, 1, 0, 1, 0);
        err.push_back(psmax(max(abs(*u - *ue))));
    }
    // measured: 2.7e-2 (N 16), 3.1e-6 (24), then 1.3e-7 to 2.4e-7 (N 32-64): the level GMRES's 1e-5 residual
    // allows, where the spatial error alone would be near 1e-12 (see the derivative tests above)
    check_convergence("poisson_2d Haidvogel-Zang (Chebyshev x Chebyshev)", Ns, err, 1e4, 2e-6);
}
