#ifndef SPINS_TEST_HELPERS_HPP
#define SPINS_TEST_HELPERS_HPP 1
/* Small shared helpers for the GoogleTest suite. */
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include "Par_util.hpp"
#include "TArray.hpp"

namespace spins_test {
    const double eps = std::numeric_limits<double>::epsilon();

    // Set SPINS_TEST_REPORT=1 to have tests print measured error ratios and iteration counts.
    inline bool report() {
        return std::getenv("SPINS_TEST_REPORT") != nullptr;
    }

    // A 1 x 1 x N array, laid out like the legacy tests and the 1D GMRES solver lay them out.
    inline std::unique_ptr<TArrayn::DTArray> make_line(int N) {
        return std::make_unique<TArrayn::DTArray>(alloc_lbound(1, 1, N), alloc_extent(1, 1, N), alloc_storage(1, 1, N));
    }

    // The same line held whole by this rank (MPI_COMM_SELF), for operator tests that use their
    // own transforms: with Nx = 1 a split over several ranks would leave some with no points.
    // Pair it with transforms built on MPI_COMM_SELF; each rank then runs the test on its own.
    inline std::unique_ptr<TArrayn::DTArray> make_local_line(int N) {
        return std::make_unique<TArrayn::DTArray>(alloc_lbound(1, 1, N, MPI_COMM_SELF), alloc_extent(1, 1, N, MPI_COMM_SELF), alloc_storage(1, 1, N, MPI_COMM_SELF));
    }

    // T_n(z), T_n'(z) and T_n''(z) from the three-term recurrences, an independent reference
    // for the transform-based Chebyshev operators.
    inline void cheb_values(int n, double z, double &T, double &dT, double &d2T) {
        double t0 = 1, t1 = z, d0 = 0, d1 = 1, s0 = 0, s1 = 0;
        if (n == 0) {
            T = 1, dT = 0, d2T = 0;
            return;
        }
        for (int k = 1; k < n; k++) {
            const double t2 = 2 * z * t1 - t0, d2 = 2 * t1 + 2 * z * d1 - d0, s2 = 4 * d1 + 2 * z * s1 - s0;
            t0 = t1, t1 = t2, d0 = d1, d1 = d2, s0 = s1, s1 = s2;
        }
        T = t1, dT = d1, d2T = s1;
    }

    // max_k |a(0, 0, k) - b[k]|
    inline double max_abs_diff(const TArrayn::DTArray &a, const std::vector<double> &b) {
        double err = 0;
        for (int k = 0; k < int(b.size()); k++) err = std::fmax(err, std::fabs(a(0, 0, k) - b[k]));
        return err;
    }
} // namespace spins_test

#endif
