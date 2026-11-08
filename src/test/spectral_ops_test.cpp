#include "mpi.h"
#include <gtest/gtest.h>
#include <blitz/array.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "Parformer.hpp"
#include "T_util.hpp"
#include "test_helpers.hpp"

/* Exactness tests for the 1D spectral operators: every mode a grid represents must be
   differentiated (and filtered) exactly, up to round-off that grows like N^(2k) eps for the
   k-th Chebyshev derivative and like N eps for trigonometric bases. Errors are reported as
   multiples of those scales, so a tolerance means the same thing at every N.
   SPINS_TEST_REPORT=1 prints the worst measured ratios. */

using spins_test::cheb_values;
using spins_test::eps;
using spins_test::make_local_line;
using spins_test::max_abs_diff;
using spins_test::report;
using TArrayn::DTArray;
using Transformer::S_EXP;
using Transformer::Trans1D;
using Transformer::TransWrapper;

// U1: d/dz and d2/dz2 of every Chebyshev polynomial T_0..T_(N-1) on the Gauss-Lobatto points
// z_k = cos(pi k/(N-1)), even and odd N. Round-off scales: N^2 eps for D1, N^4 eps for D2.
TEST(ChebyshevDerivative, EveryModeIsExact) {
    // In units of N^2 eps max|T_n'| and N^4 eps max|T_n''|. Measured worst: 0.87 and 0.24.
    constexpr double tol1 = 10.0, tol2 = 10.0;
    double           worst1 = 0, worst2 = 0;
    for (int N : {8, 9, 16, 17, 32, 33, 64, 65}) {
        auto    f = make_local_line(N), d1 = make_local_line(N), d2 = make_local_line(N);
        Trans1D tform(1, 1, N, TArrayn::thirdDim, Transformer::CHEBY, MPI_COMM_SELF);
        for (int n = 0; n < N; n++) {
            std::vector<double> e1(N), e2(N);
            double              s1 = 1, s2 = 1; // max|T_n'|, max|T_n''| (at least 1)
            for (int k = 0; k < N; k++) {
                double T;
                cheb_values(n, std::cos(M_PI * k / (N - 1)), T, e1[k], e2[k]);
                (*f)(0, 0, k) = T;
                s1            = std::fmax(s1, std::fabs(e1[k]));
                s2            = std::fmax(s2, std::fabs(e2[k]));
            }
            TArrayn::deriv_cheb(*f, tform, *d1);
            TArrayn::deriv_cheb(*d1, tform, *d2);
            const double r1 = max_abs_diff(*d1, e1) / (double(N) * N * eps * s1);
            const double r2 = max_abs_diff(*d2, e2) / (std::pow(double(N), 4) * eps * s2);
            worst1          = std::fmax(worst1, r1);
            worst2          = std::fmax(worst2, r2);
            EXPECT_LT(r1, tol1) << "D1 of T_" << n << ", N = " << N;
            EXPECT_LT(r2, tol2) << "D2 of T_" << n << ", N = " << N;
        }
    }
    if (report() && master()) std::printf("[ report ] Chebyshev modes: worst D1 %.3g N^2 eps, worst D2 %.3g N^4 eps\n", worst1, worst2);
}

// U2: round-off growth on a smooth, fully resolved function, N = 32..1024. A badly formed
// operator shows up as growth faster than N^2 (D1) or N^4 (D2).
TEST(ChebyshevDerivative, RoundOffGrowthIsBounded) {
    // In units of N^2 eps max|f'| and N^4 eps max|f''|. Measured worst: 0.16 and 0.016.
    const double tol1 = 1.0, tol2 = 1.0;
    double       worst1 = 0, worst2 = 0;
    for (int N : {32, 64, 128, 256, 512, 1024}) {
        auto                f = make_local_line(N), d1 = make_local_line(N), d2 = make_local_line(N);
        Trans1D             tform(1, 1, N, TArrayn::thirdDim, Transformer::CHEBY, MPI_COMM_SELF);
        std::vector<double> e1(N), e2(N);
        double              s1 = 0, s2 = 0;
        for (int k = 0; k < N; k++) {
            const double z = std::cos(M_PI * k / (N - 1)), ez = std::exp(z), s = std::sin(3 * z), c = std::cos(3 * z);
            (*f)(0, 0, k) = ez * s;
            e1[k]         = ez * (s + 3 * c);
            e2[k]         = ez * (6 * c - 8 * s);
            s1            = std::fmax(s1, std::fabs(e1[k]));
            s2            = std::fmax(s2, std::fabs(e2[k]));
        }
        TArrayn::deriv_cheb(*f, tform, *d1);
        TArrayn::deriv_cheb(*d1, tform, *d2);
        const double r1 = max_abs_diff(*d1, e1) / (double(N) * N * eps * s1);
        const double r2 = max_abs_diff(*d2, e2) / (std::pow(double(N), 4) * eps * s2);
        worst1          = std::fmax(worst1, r1);
        worst2          = std::fmax(worst2, r2);
        EXPECT_LT(r1, tol1) << "D1, N = " << N;
        EXPECT_LT(r2, tol2) << "D2, N = " << N;
    }
    if (report() && master()) std::printf("[ report ] Chebyshev round-off sweep: worst D1 %.3g N^2 eps, worst D2 %.3g N^4 eps\n", worst1, worst2);
}

// U3: Fourier derivative on [0, 2 pi), x_k = 2 pi k/N. Every wavenumber below Nyquist is exact.
// For even N the Nyquist mode (-1)^k has no real derivative; SPINS's convention is to return 0.
TEST(FourierDerivative, EveryModeIsExactAndNyquistIsDropped) {
    const double tol   = 10.0; // in units of N eps max(1, m)
    double       worst = 0;
    for (int N : {8, 9, 16, 17, 64, 65}) {
        auto    f = make_local_line(N), d = make_local_line(N);
        Trans1D tform(1, 1, N, TArrayn::thirdDim, Transformer::FOURIER, MPI_COMM_SELF);
        for (int m = 0; m <= N / 2; m++) {
            const bool          nyquist = (2 * m == N);
            std::vector<double> e(N);
            for (int k = 0; k < N; k++) {
                const double x = 2 * M_PI * k / N;
                (*f)(0, 0, k)  = nyquist ? (k % 2 ? -1.0 : 1.0) : std::cos(m * x + 0.3);
                e[k]           = nyquist ? 0.0 : -m * std::sin(m * x + 0.3);
            }
            TArrayn::deriv_fft(*f, tform, *d);
            const double r = max_abs_diff(*d, e) / (N * eps * std::fmax(1.0, m));
            worst          = std::fmax(worst, r);
            EXPECT_LT(r, tol) << "m = " << m << ", N = " << N << (nyquist ? " (Nyquist)" : "");
        }
    }
    if (report() && master()) std::printf("[ report ] Fourier modes: worst %.3g N eps max(1,m)\n", worst);
}

// U4: cosine and sine derivatives on [0, pi] at the half-sample points x_k = (k + 1/2) pi/N used
// by REDFT10/RODFT10. d/dx cos(mx) = -m sin(mx) for m = 0..N-1; d/dx sin(mx) = m cos(mx) for
// m = 1..N, where the top sine's derivative, N cos(N x_k), vanishes on the grid.
TEST(CosineSineDerivative, EveryModeIsExact) {
    const double tol   = 10.0; // in units of N eps max(1, m)
    double       worst = 0;
    for (int N : {8, 9, 16, 17, 64, 65}) {
        auto    f = make_local_line(N), d = make_local_line(N);
        Trans1D cos_t(1, 1, N, TArrayn::thirdDim, Transformer::COSINE, MPI_COMM_SELF);
        Trans1D sin_t(1, 1, N, TArrayn::thirdDim, Transformer::SINE, MPI_COMM_SELF);
        for (int m = 0; m <= N; m++) {
            std::vector<double> e(N);
            if (m < N) { // cosine mode m
                for (int k = 0; k < N; k++) {
                    const double x = (k + 0.5) * M_PI / N;
                    (*f)(0, 0, k)  = std::cos(m * x);
                    e[k]           = -m * std::sin(m * x);
                }
                TArrayn::deriv_dct(*f, cos_t, *d);
                const double r = max_abs_diff(*d, e) / (N * eps * std::fmax(1.0, m));
                worst          = std::fmax(worst, r);
                EXPECT_LT(r, tol) << "cos(" << m << " x), N = " << N;
            }
            if (m > 0) { // sine mode m
                for (int k = 0; k < N; k++) {
                    const double x = (k + 0.5) * M_PI / N;
                    (*f)(0, 0, k)  = std::sin(m * x);
                    e[k]           = m * std::cos(m * x);
                }
                TArrayn::deriv_dst(*f, sin_t, *d);
                const double r = max_abs_diff(*d, e) / (N * eps * std::fmax(1.0, m));
                worst          = std::fmax(worst, r);
                EXPECT_LT(r, tol) << "sin(" << m << " x), N = " << N;
            }
        }
    }
    if (report() && master()) std::printf("[ report ] cosine/sine modes: worst %.3g N eps max(1,m)\n", worst);
}

// U5: the exponential filter multiplies mode k by 1 for |k| <= c k_max and by
// exp(-s ((|k| - c k_max)/(k_max - c k_max))^p) above, so the highest mode gets exp(-s).
// k_max is the highest wavenumber the grid holds: N-1 for cosine and Chebyshev, N for sine and
// floor(N/2) for Fourier. It is computed here independently of TransWrapper::max_wavenum.
TEST(SpectralFilter, TransferFunctionMatchesDefinition) {
    const double cutoff = 0.6, order = 2.0, strength = 20.0;
    const double tol = 100.0; // in units of N eps
    struct Basis {
        S_EXP       type;
        const char *name;
    };
    for (Basis b : {
             Basis{Transformer::COSINE,  "cosine"   },
             Basis{Transformer::SINE,    "sine"     },
             Basis{Transformer::CHEBY,   "Chebyshev"},
             Basis{Transformer::FOURIER, "Fourier"  }
    }) {
        for (int N : {16, 17, 32, 33}) {
            int kmin = 0, kmax = N - 1;
            if (b.type == Transformer::SINE) kmin = 1, kmax = N;
            if (b.type == Transformer::FOURIER) kmax = N / 2;
            const double kc = cutoff * kmax;
            auto         f  = make_local_line(N);
            TransWrapper tw(1, 1, N, Transformer::NONE, Transformer::NONE, b.type, MPI_COMM_SELF);
            for (int m = kmin; m <= kmax; m++) {
                std::vector<double> in(N);
                for (int k = 0; k < N; k++) {
                    switch (b.type) {
                        case Transformer::COSINE:
                            in[k] = std::cos(m * (k + 0.5) * M_PI / N);
                            break;
                        case Transformer::SINE:
                            in[k] = std::sin(m * (k + 0.5) * M_PI / N);
                            break;
                        case Transformer::CHEBY:
                            in[k] = std::cos(m * M_PI * k / (N - 1));
                            break;
                        default:
                            in[k] = std::cos(m * 2 * M_PI * k / N);
                            break;
                    }
                    (*f)(0, 0, k) = in[k];
                }
                const double sigma = (m <= kc) ? 1.0 : std::exp(-strength * std::pow((m - kc) / (kmax - kc), order));
                TArrayn::filter3(*f, tw, Transformer::NONE, Transformer::NONE, b.type, cutoff, order, strength);
                std::vector<double> expect(N);
                for (int k = 0; k < N; k++) expect[k] = sigma * in[k];
                const double r = max_abs_diff(*f, expect) / (N * eps);
                EXPECT_LT(r, tol) << b.name << " mode " << m << " of k_max " << kmax << ", N = " << N << ", expected factor " << sigma;
            }
        }
    }
}
