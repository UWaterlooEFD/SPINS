/* Tests of the analysis routines in Science/: quadrature weights, background potential energy and
   the equations of state (test U16 of the verification plan).

   compute_quadweights must integrate polynomials of degree <= N - 1 exactly on a Chebyshev grid
   (Clenshaw-Curtis). compute_Background_PE must give BPE = PE for a fluid that is already sorted,
   and PE - BPE = g rho_0 drho H^2 Lx Ly / 8 for a lock-exchange state; on the uniform (trig) grid
   both hold to round-off, on any number of ranks. lineos takes its expansion coefficients from
   nleos by forward differences, whose error is measured against a Richardson-extrapolated central
   difference. Run with SPINS_TEST_REPORT=1 to print the measured errors. */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>

#include "Science.hpp"
#include "test_helpers.hpp"

using namespace TArrayn;
using NSIntegrator::FREE_SLIP;
using NSIntegrator::NO_SLIP;
using NSIntegrator::PERIODIC;
using spins_test::eps;
using spins_test::report;

namespace {
    blitz::firstIndex  ii;
    blitz::secondIndex jj;
    blitz::thirdIndex  kk;

    std::unique_ptr<DTArray> make_split(int Nx, int Ny, int Nz) {
        return std::make_unique<DTArray>(alloc_lbound(Nx, Ny, Nz), alloc_extent(Nx, Ny, Nz), alloc_storage(Nx, Ny, Nz));
    }

    // Sum_j w_j x_j^p on the Chebyshev points x_j = (L/2) cos(j pi / (N - 1)) of a weight vector
    // indexed by global point number. The x weights are split across ranks (summed over them),
    // the y and z weights are held whole on every rank.
    double cheb_moment(const blitz::Array<double, 1> &w, int N, double L, int p, bool split) {
        double s = 0;
        for (int j = w.lbound(firstDim); j <= w.ubound(firstDim); j++) s += w(j) * std::pow(L / 2 * std::cos(j * M_PI / (N - 1)), p);
        return split ? pssum(s) : s;
    }
} // namespace

// Clenshaw-Curtis weights integrate x^p exactly for even p <= N - 1 (odd p vanish by symmetry), in
// each of the three directions, which compute_quadweights codes separately.
TEST(Quadrature, ClenshawCurtisIntegratesPolynomials) {
    const double L = 3.0;
    for (int N : {8, 9, 16, 17, 32, 33}) {
        compute_quadweights(N, N, N, L, L, L, NO_SLIP, NO_SLIP, NO_SLIP);
        const blitz::Array<double, 1> *w[3] = {get_quad_x(), get_quad_y(), get_quad_z()};
        for (int d = 0; d < 3; d++) {
            double       worst        = 0;
            const double should_split = (d == 0);
            for (int p = 0; p <= N - 1; p += 2) {
                const double exact = 2 * std::pow(L / 2, p + 1) / (p + 1);
                worst              = std::fmax(worst, std::fabs(cheb_moment(*w[d], N, L, p, should_split) - exact) / exact);
            }
            if (report() && master()) printf("Clenshaw-Curtis N = %d, direction %d: max relative moment error %.3e\n", N, d, worst);
            EXPECT_LT(worst, 50 * eps) << "N = " << N << ", direction " << d;
        }
    }
}

// The trapezoid weights of the trig directions are L / N and sum to L.
TEST(Quadrature, TrigWeightsAreUniform) {
    const int    N = 12;
    const double L = 2.5;
    compute_quadweights(N, N, N, L, L, L, FREE_SLIP, PERIODIC, FREE_SLIP);
    const blitz::Array<double, 1> *w[3] = {get_quad_x(), get_quad_y(), get_quad_z()};
    for (int d = 0; d < 3; d++) {
        EXPECT_LT(psmax(max(abs(*w[d] - L / N))), 4 * eps * L / N);
        EXPECT_NEAR(d == 0 ? pssum(sum(*w[d])) : sum(*w[d]), L, 8 * eps * L);
    }
}

namespace {
    // Grid shape of the BPE tests (uniform trig grid unless noted). compute_Background_PE keeps static
    // state sized by the first call, and sortarray warns on shape changes, so all share one shape.
    const int    bNx = 64, bNy = 1, bNz = 32;
    const double bLx = 2.0, bLy = 1.0, bLz = 0.5, g = 9.81, rho_0 = 1000.0;

    struct BpeSetup {
        std::unique_ptr<DTArray> rho, quad3;
        BpeSetup() : rho(make_split(bNx, bNy, bNz)), quad3(make_split(bNx, bNy, bNz)) {
            compute_quadweights(bNx, bNy, bNz, bLx, bLy, bLz, FREE_SLIP, FREE_SLIP, FREE_SLIP);
            *quad3 = (*get_quad_x())(ii) * (*get_quad_y())(jj) * (*get_quad_z())(kk);
        }
        // Height above the bottom of grid level k (half-sample grid)
        static double zk(int k) { return (k + 0.5) * bLz / bNz; }
        // PE = sum g rho_0 (1 + rho') z dV with the same weights, summed over ranks
        double pe() const {
            double s = 0;
            for (int i = rho->lbound(firstDim); i <= rho->ubound(firstDim); i++)
                for (int j = rho->lbound(secondDim); j <= rho->ubound(secondDim); j++)
                    for (int k = rho->lbound(thirdDim); k <= rho->ubound(thirdDim); k++) s += g * rho_0 * (1 + (*rho)(i, j, k)) * zk(k) * (*quad3)(i, j, k);
            return pssum(s);
        }
    };
} // namespace

// A density that already decreases with height is its own sorted state: BPE = PE.
TEST(BackgroundPE, SortedProfileEqualsPE) {
    BpeSetup s;
    *s.rho     = -0.01 * tanh((BpeSetup::zk(0) + kk * bLz / bNz - bLz / 2) / 0.1) + 0 * ii + 0 * jj;
    double bpe = 0;
    compute_Background_PE(bpe, *s.rho, *s.quad3, bNx, bNy, bNz, bLx, bLy, bLz, g, rho_0, 0);
    const double pe = s.pe();
    if (report() && master()) printf("Sorted profile: BPE %.15e, PE %.15e, relative difference %.3e\n", bpe, pe, (bpe - pe) / pe);
    EXPECT_LT(std::fabs(bpe - pe), 20 * eps * pe);
}

// Lock exchange: heavy fluid (rho' = +d/2) in the left half, light (-d/2) in the right half.
// Sorted, the heavy half fills the bottom half-depth, so PE - BPE = g rho_0 d H^2 Lx Ly / 8.
TEST(BackgroundPE, LockExchangeAPE) {
    BpeSetup     s;
    const double d = 0.02;
    *s.rho         = where(ii < bNx / 2, d / 2, -d / 2) + 0 * jj + 0 * kk;
    double bpe     = 0;
    compute_Background_PE(bpe, *s.rho, *s.quad3, bNx, bNy, bNz, bLx, bLy, bLz, g, rho_0, 1);
    const double ape = s.pe() - bpe, exact = g * rho_0 * d * bLz * bLz * bLx * bLy / 8;
    if (report() && master()) printf("Lock exchange: APE %.15e, exact %.15e, relative error %.3e\n", ape, exact, (ape - exact) / exact);
    // APE is a small difference of two large terms (PE ~ g rho_0 H^2 L / 2), so scale by PE
    EXPECT_LT(std::fabs(ape - exact), 20 * eps * s.pe());
}

namespace {
    // BPE/PE - 1 for rho' = -0.01 (z - H/2) on a mapped grid (periodic x, Chebyshev z) over the
    // bottom amp (1 + cos(2 pi x / Lx)) / 2, with map_iwave's volume weights. Same shape as above.
    double mapped_bpe_bias(double amp) {
        compute_quadweights(bNx, bNy, bNz, bLx, bLy, bLz, PERIODIC, FREE_SLIP, NO_SLIP);
        auto                    rho = make_split(bNx, bNy, bNz), quad3 = make_split(bNx, bNy, bNz);
        blitz::Array<double, 1> hill(split_range(bNx));
        double                  pe = 0;
        for (int i = rho->lbound(firstDim); i <= rho->ubound(firstDim); i++) {
            hill(i) = amp * (1 + std::cos(2 * M_PI * (i + 0.5) / bNx)) / 2;
            for (int k = 0; k < bNz; k++) {
                const double z    = hill(i) + (bLz - hill(i)) * (0.5 - 0.5 * std::cos(M_PI * k / (bNz - 1)));
                (*rho)(i, 0, k)   = -0.01 * (z - bLz / 2);
                (*quad3)(i, 0, k) = (*get_quad_x())(i) * (*get_quad_y())(0) * (*get_quad_z())(k) * (bLz - hill(i)) / bLz;
                pe += g * rho_0 * (1 + (*rho)(i, 0, k)) * z * (*quad3)(i, 0, k);
            }
        }
        pe         = pssum(pe);
        double bpe = 0;
        compute_Background_PE(bpe, *rho, *quad3, bNx, bNy, bNz, bLx, bLy, bLz, g, rho_0, 0, false, true, hill);
        return bpe / pe - 1;
    }
} // namespace

// Over a hill a density that depends on z only is again its own sorted state, so BPE may differ
// from PE only by the stacking of Chebyshev cell volumes into horizontal slabs, the error of a
// flat bottom, on any number of ranks. Stacking each cell with the area at its top instead biased
// BPE high at first order, by an amount that changed with the number of ranks.
TEST(BackgroundPE, MappedSortedProfileEqualsPE) {
    const double flat = mapped_bpe_bias(0.0), hill = mapped_bpe_bias(0.5 * bLz);
    if (report() && master()) printf("Mapped BPE/PE - 1: flat bottom %.4e, hill of half the depth %.4e\n", flat, hill);
    EXPECT_LT(std::fabs(hill), 3 * std::fabs(flat));
}

namespace {
    // Richardson-extrapolated central difference of f at x: error O(h^4) + O(eps f / h)
    template <class F>
    double ref_derivative(F f, double x) {
        const double h  = 1e-2;
        const double d1 = (f(x + h) - f(x - h)) / (2 * h), d2 = (f(x + h / 2) - f(x - h / 2)) / h;
        return (4 * d2 - d1) / 3;
    }
} // namespace

// lineos's alpha and beta are forward differences with step 1e-8 on a density of ~1e3 kg/m^3, so
// round-off alone allows errors of about eps * 1e3 / 1e-8 ~ 2e-5 kg/m^3 per unit. Measure them.
TEST(EOS, LinearCoefficientsMatchNonlinearDerivatives) {
    double worst_a = 0, worst_b = 0, worst_ra = 0, worst_rb = 0;
    for (double T0 : {0.0, 4.0, 10.0, 20.0, 30.0})
        for (double S0 : {0.0, 10.0, 35.0}) {
            const double a_ref = ref_derivative([&](double T) { return nleos_inline(T, S0); }, T0);
            const double b_ref = S0 > 0.02 ? ref_derivative([&](double S) { return nleos_inline(T0, S); }, S0) : (nleos_inline(T0, 1e-6) - nleos_inline(T0, 0)) / 1e-6;
            const double ea = std::fabs(compute_alpha(T0, S0) - a_ref), eb = std::fabs(compute_beta(T0, S0) - b_ref);
            worst_a = std::fmax(worst_a, ea), worst_b = std::fmax(worst_b, eb);
            if (std::fabs(a_ref) > 1e-2) worst_ra = std::fmax(worst_ra, ea / std::fabs(a_ref));
            worst_rb = std::fmax(worst_rb, eb / std::fabs(b_ref));
            EXPECT_EQ(compute_rho0(T0, S0), nleos_inline(T0, S0));
        }
    if (report() && master())
        printf("lineos coefficients: max |alpha error| %.3e kg/m^3/K (relative %.3e where |alpha| > 0.01), "
               "max |beta error| %.3e kg/m^3/(g/kg) (relative %.3e)\n",
               worst_a, worst_ra, worst_b, worst_rb);
    // Bound set from the round-off estimate above (2e-5) with a factor-of-5 margin
    EXPECT_LT(worst_a, 1e-4);
    EXPECT_LT(worst_b, 1e-4);
}

// The array nleos repeats nleos_inline's coefficients, so the two must agree for S >= 0. For S < 0
// nleos_inline clamps S to 0 everywhere while nleos clamps only inside the square root; record that.
TEST(EOS, ArrayFormMatchesInline) {
    const int Nx = 6, Ny = 1, Nz = 9;
    auto      T = make_split(Nx, Ny, Nz), S = make_split(Nx, Ny, Nz), rho = make_split(Nx, Ny, Nz);
    *T = -2.0 + 6.0 * ii + 0 * jj + 0 * kk;
    *S = -0.5 + 5.0 * kk + 0 * ii + 0 * jj; // -0.5 ... 39.5
    nleos(*rho, *T, *S);
    double worst = 0, neg = 0;
    for (int i = rho->lbound(firstDim); i <= rho->ubound(firstDim); i++)
        for (int k = 0; k < Nz; k++) {
            const double d = std::fabs((*rho)(i, 0, k) - nleos_inline((*T)(i, 0, k), (*S)(i, 0, k)));
            if ((*S)(i, 0, k) >= 0) worst = std::fmax(worst, d / (*rho)(i, 0, k));
            else neg = std::fmax(neg, d);
        }
    worst = psmax(worst), neg = psmax(neg);
    if (report() && master()) printf("nleos array vs inline: max relative difference %.3e for S >= 0; %.3e kg/m^3 at S = -0.5\n", worst, neg);
    EXPECT_LT(worst, 4 * eps);
}

// quadeos is a parabola through nleos's freshwater maximum: its vertex value must be nleos at Tmd,
// and Tmd must sit at the maximum of nleos(T, 0) to within the fit's stated grid (0.01 K).
TEST(EOS, QuadeosVertexMatchesNonlinear) {
    const int    Nx = 4, Ny = 1, Nz = 1;
    auto         T = make_split(Nx, Ny, Nz), rho = make_split(Nx, Ny, Nz);
    const double Tmd = 3.973973973973974;
    *T               = Tmd;
    quadeos(*rho, *T);
    const double vertex = (*rho)(rho->lbound());
    // golden-section search for the maximum of nleos(T, 0)
    double       a = 3.0, b = 5.0;
    const double r = (std::sqrt(5.0) - 1) / 2;
    for (int it = 0; it < 80; it++) {
        const double c = b - r * (b - a), d = a + r * (b - a);
        if (nleos_inline(c, 0) > nleos_inline(d, 0)) b = d;
        else a = c;
    }
    const double Tmax = (a + b) / 2;
    if (report() && master()) printf("quadeos: vertex %.12f at Tmd %.6f; nleos maximum %.12f at T %.6f\n", vertex, Tmd, nleos_inline(Tmax, 0), Tmax);
    EXPECT_LT(std::fabs(vertex - nleos_inline(Tmd, 0)), 1e-9);
    EXPECT_LT(std::fabs(Tmax - Tmd), 0.01);
}
