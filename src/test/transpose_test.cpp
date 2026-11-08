#include "mpi.h"
#include <gtest/gtest.h>
#include <blitz/array.h>
#include <complex>

#include "Splits.hpp"

/* U13: the MPI slab transposes (Transposer, as used by Parformer and ESolver to move a split
   from x to z and back) must deliver every element to the right place, and the round trip must
   be bitwise exact. Each element's value encodes its global (i, j, k), so a misplaced element
   is detected wherever it lands. Covers double and complex data, sizes not divisible by the rank
   count and Ny = 1 (2D runs). Run under MPI for the parallel cases, e.g.

       mpirun -np 4 ./spins_test --gtest_filter='Transpose*'

   Counts are summed over ranks, so every rank reaches the same verdict. */

namespace {
    using blitz::Array;
    using blitz::GeneralArrayStorage;
    using blitz::TinyVector;

    double code(int i, int j, int k) {
        return i + 1000.0 * j + 1e6 * k + 0.25;
    }
    template <class T>
    T value(int i, int j, int k);
    template <>
    double value<double>(int i, int j, int k) {
        return code(i, j, k);
    }
    template <>
    std::complex<double> value<std::complex<double>>(int i, int j, int k) {
        return {code(i, j, k), -code(i, j, k) - 0.5};
    }

    // Number of elements of a that differ from value(i, j, k) (or from b, if given).
    template <class T>
    long mismatches(const Array<T, 3> &a, const Array<T, 3> *b = nullptr) {
        long bad = 0;
        for (int i = a.lbound(0); i <= a.ubound(0); i++)
            for (int j = a.lbound(1); j <= a.ubound(1); j++)
                for (int k = a.lbound(2); k <= a.ubound(2); k++)
                    if (a(i, j, k) != (b ? (*b)(i, j, k) : value<T>(i, j, k))) bad++;
        return bad;
    }

    long global_sum(long x) {
        long s = 0;
        MPI_Allreduce(&x, &s, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        return s;
    }

    template <class T>
    void check_round_trip(int sx, int sy, int sz) {
        Transposer<T>          tr(sx, sy, sz, blitz::firstDim, blitz::thirdDim);
        TinyVector<int, 3>     lb, ex;
        GeneralArrayStorage<3> order;
        tr.source_alloc(lb, ex, order);
        Array<T, 3> src(lb, ex, order), back(lb, ex, order);
        tr.dest_alloc(lb, ex, order);
        Array<T, 3> dst(lb, ex, order);
        for (int i = src.lbound(0); i <= src.ubound(0); i++)
            for (int j = src.lbound(1); j <= src.ubound(1); j++)
                for (int k = src.lbound(2); k <= src.ubound(2); k++) src(i, j, k) = value<T>(i, j, k);

        tr.transpose(src, dst);
        const long held = global_sum(dst.numElements()), wrong = global_sum(mismatches(dst));
        tr.back_transpose(dst, back);
        const long lost = global_sum(mismatches(back, &src));

        int np;
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        const char *type = sizeof(T) == sizeof(double) ? "double" : "complex";
        EXPECT_EQ(held, long(sx) * sy * sz) << type << " " << sx << "x" << sy << "x" << sz << ", np = " << np << ": elements held after transpose";
        EXPECT_EQ(wrong, 0) << type << " " << sx << "x" << sy << "x" << sz << ", np = " << np << ": misplaced elements after transpose";
        EXPECT_EQ(lost, 0) << type << " " << sx << "x" << sy << "x" << sz << ", np = " << np << ": round trip not bitwise";
    }

    // Sizes with every split dimension at least the rank count (4), including uneven splits and
    // Ny = 1.
    const int sizes[][3] = {
        {16, 16, 16},
        {17, 13, 11},
        {16, 1,  16},
        {9,  1,  7 },
        {5,  3,  4 },
        {4,  1,  4 }
    };
} // namespace

TEST(Transpose, RealRoundTripIsExact) {
    for (auto &s : sizes) check_round_trip<double>(s[0], s[1], s[2]);
}

TEST(Transpose, ComplexRoundTripIsExact) {
    for (auto &s : sizes) check_round_trip<std::complex<double>>(s[0], s[1], s[2]);
}
