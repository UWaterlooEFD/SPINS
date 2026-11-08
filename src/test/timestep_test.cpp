#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "Timestep.hpp"

/* U6: the variable-step weights from Timestep::get_coeff. With levels t[-2..1], the left-hand
   weights differentiate the interpolating polynomial at t[1], and the right-hand weights
   extrapolate the explicit terms from t[-2..0] to t[1]. For the full third-order scheme the
   derivative must be exact for cubics and the extrapolation exact for quadratics, for any step
   sequence. The start-up orders (repeated levels) must be exact one degree lower. */

using Timestep::get_coeff;
using Timestep::Stepped;

namespace {
    // Sum_i w[i] p(t[i]) for p(t) = (t - t1)^j, over levels lo..hi.
    double apply(const Stepped<double> &w, const Stepped<double> &t, int lo, int hi, int j) {
        double s = 0;
        for (int i = lo; i <= hi; i++) s += w[i] * std::pow(t[i] - t[1], j);
        return s;
    }

    // Checks the weights for one set of levels. `order` is the scheme order the levels imply.
    void check_weights(const Stepped<double> &t, const int order, const char *what) {
        Stepped<double> L(4), R(4);
        get_coeff(t, L, R);
        // Sums are scaled to O(1): the left weights are O(1/dt) and (t - t1)^j is O(dt^j).
        const double dt = t[1] - t[0], tol = 1e-12;
        for (int j = 0; j <= order; j++) {
            // d/dt (t - t1)^j at t1 is 1 for j = 1 and 0 otherwise.
            const double expect = (j == 1) ? 1.0 : 0.0;
            EXPECT_NEAR(apply(L, t, -2, 1, j) / std::pow(dt, j - 1), expect, tol) << what << ": derivative weights, degree " << j;
        }
        // Extrapolation to t1 of (t - t1)^j is 1 for j = 0 and 0 otherwise.
        for (int j = 0; j < order; j++) {
            const double expect = (j == 0) ? 1.0 : 0.0;
            EXPECT_NEAR(apply(R, t, -2, 0, j) / std::pow(dt, j), expect, tol) << what << ": extrapolation weights, degree " << j;
        }
        EXPECT_EQ(R[1], 1.0) << what << ": implicit-term weight";
    }
} // namespace

TEST(TimestepWeights, ThirdOrderIsExactForRandomStepRatios) {
    std::mt19937                     gen(12345);
    std::uniform_real_distribution<> ratio(0.5, 1.5);
    for (int trial = 0; trial < 200; trial++) {
        Stepped<double> t(4);
        double          h = 1e-2 * ratio(gen);
        t[-2]             = 1.0;
        t[-1]             = t[-2] + h;
        h *= ratio(gen);
        t[0] = t[-1] + h;
        h *= ratio(gen);
        t[1] = t[0] + h;
        check_weights(t, 3, "random ratios");
    }
}

TEST(TimestepWeights, StartupOrdersAreExactOneDegreeLower) {
    Stepped<double> t(4);
    // First step: t[-2] == t[-1] == t[0] gives the first-order (BDF1) formula.
    t[-2] = t[-1] = t[0] = 0.0;
    t[1]                 = 1.0 / 64;
    check_weights(t, 1, "BDF1 start-up");
    // Second step: t[-2] == t[-1] gives the second-order formula.
    t[-2] = t[-1] = 0.0;
    t[0]          = 1.0 / 64;
    t[1]          = 2.0 / 64;
    check_weights(t, 2, "BDF2 start-up");
}

namespace {
    // N3: zero stability under a constant step ratio r. For y' = 0 the scheme is
    // y1 = -(L[-2] y[-2] + L[-1] y[-1] + L[0] y[0]) / L[1]; a perturbed history leaves the constant
    // solution and the differences y_n - y_(n-1) evolve with the parasitic roots. Returns their
    // growth per step over steps 30-60.
    double parasitic_growth(const double r) {
        Stepped<double> t(4), L(4), R(4);
        double          y[3] = {1.0, 1.0, 1.0 + 1e-3}, h = r * r, d30 = 0, d60 = 0;
        t[-2] = 0.0, t[-1] = 1.0, t[0] = 1.0 + r;
        for (int n = 1; n <= 60; n++) {
            t[1] = t[0] + h;
            get_coeff(t, L, R);
            const double y1 = -(L[-2] * y[0] + L[-1] * y[1] + L[0] * y[2]) / L[1];
            if (n == 30) d30 = std::fabs(y1 - y[2]);
            if (n == 60) d60 = std::fabs(y1 - y[2]);
            y[0] = y[1], y[1] = y[2], y[2] = y1;
            t[-2] = t[-1], t[-1] = t[0], t[0] = t[1];
            h *= r;
        }
        return std::pow(d60 / d30, 1.0 / 30);
    }
} // namespace

// Sustained step growth: the variable-step third-order scheme damps history perturbations at
// SPINS's growth cap (1.5 per step) and amplifies them beyond r ~ 1.65. The start-up ramp
// doubles its steps (r = 2), where perturbations grow by 1.43 per step; that is why its first-step
// error is amplified (about 3x with the dt/64 start, 15x with dt/512).
TEST(TimestepWeights, ZeroStabilityUnderSustainedStepGrowth) {
    const double g15 = parasitic_growth(1.5), g16 = parasitic_growth(1.6), g17 = parasitic_growth(1.7), g20 = parasitic_growth(2.0);
    if (std::getenv("SPINS_TEST_REPORT")) printf("parasitic growth per step: r = 1.5 %.4f, 1.6 %.4f, 1.7 %.4f, 2.0 %.4f\n", g15, g16, g17, g20);
    EXPECT_LT(g15, 0.9); // measured 0.8811
    EXPECT_LT(g16, 1.0); // 0.9848
    EXPECT_GT(g17, 1.0); // 1.0922
    EXPECT_NEAR(g20, 1.4257, 0.005);
}
