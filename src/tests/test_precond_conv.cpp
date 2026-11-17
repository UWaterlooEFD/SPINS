/* test_precond_conv.cpp -- convergence regression for the 1D Chebyshev
   line solve's preconditioner (the build_precond node grid).
 *
 * WHAT IT IS.  A battery of (N, helm, BC class, RHS pool) points runs the
 * LIBRARY's own poisson_1d() (the shipped path: cheb_d2 + GMRES_Solver +
 * build_precond's tridiagonal band) on a smooth and a rough right-hand
 * side, and asserts two things per point:
 *
 *   (1) SOLUTION.  The returned line matches a direct reference solve of
 *       the SAME discrete system to a stated relative tolerance.  The
 *       reference is not a transcription: the (N+1)x(N+1) pair-state
 *       operator of the config is recovered by unit-vector probing of a
 *       second, identically-configured cheb_d2's own matrix_multiply()
 *       (mode 1 of the recipe documented below: the library's kernel is
 *       used via ../gmres_1d_solver_impl.hpp,
 *       NOT a test-local copy: the local-copy symbol collision is what
 *       test_gmres.cpp's #define guard exists to dodge), then solved by a
 *       partial-pivoting LU written in this file.  A Krylov method that
 *       converges must land on the direct answer; this pins what the
 *       solve delivers, not how many steps it takes.
 *
 *   (2) ITERATION COUNT.  The returned itc is at or below a committed
 *       per-point bound.  The bounds are placed BETWEEN the shipped-node
 *       and the operator-node counts of the preconditioner: the shipped
 *       build_precond used to place its band on nodes of half-width
 *       |length| while the operator it preconditions
 *       (matrix_multiply: deriv_cheb scaled by 1/(length/2), twice) lives
 *       on |length|/2, and on that half-width the smooth-pool pressure
 *       line costs ~4x the iterations it needs on the operator's own
 *       nodes.  Re-introducing the mis-scaled nodes therefore FAILS THIS
 *       TEST LOUDLY, at every bound but the two that guard the DC and
 *       Dirichlet-Poisson points (whose counts barely move -- the fixed
 *       band must merely stay converged and correct there).
 *
 * Exit status is 0 only if every point passes both assertions; any
 * non-converged (itc < 0) or over-bound or off-reference point makes the
 * exit non-zero.  One line per point is printed to stdout:
 *
 *   PRECOND_CONV <label> N=<N> helm=<h> cls=<NN|DD> pool=<sm|rg> \
 *                  itc=<k> bound=<b> rel=<e> tol=<t> <PASS|FAIL>
 *
 * USAGE:  test_precond_conv_x [N]      (N = 256 or 1024, default 256)
 * Runs in seconds under mpirun -n 1; no case files, no grid beyond the
 * committed points.  Like poisson_1d() itself the kernel is built once
 * per process, so ONE N per invocation.
 */

#include <mpi.h>

#include "../gmres.hpp"
#include "../gmres_1d_solver.hpp"
#include "../gmres_1d_solver_impl.hpp"

#include <blitz/array.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>

/* ------------------------------------------------------------------ */
/* the committed battery                                              */
/* ------------------------------------------------------------------ */

/* One line-solve point.  `pool` 0 = smooth, 1 = rough (white noise).
   BC classes: NN = pressure (a=0,0 b=1,1), DD = velocity (a=1,1 b=0,0) --
   the two classes the model actually solves (ESolver.cpp vertical lines),
   at helms taken from a survey of the model's own driven lines; the
   values, with the configs that produced them, are the literals below. */
struct point {
    const char *label;
    double      helm;
    int         a_top, a_bot, b_top, b_bot;
    int         pool; /* 0 smooth, 1 rough */
    int         bound;
};

/* The helmholtz numbers this test drives, committed as literals so the
   test is self-contained (the same values the survey above used). */
#define HELM_NN_1 9.869604401089358e+00
#define HELM_NN_2 1.3052551820440676e+03
#define HELM_NN_3 2.1340561143814957e+04
#define HELM_DD_1 9.1556308014086240e+06
#define HELM_DD_2 9.8287200000000000e+06

/* N = 256 (the grid of the reported case, Nz = 256).  Bounds committed from
   measurements on both sides of the node line:
   with the band built on nodes of half-width
   |length| the smooth pressure lines cost 15-32 iterations, with the
   operator's |length|/2 nodes 3-13; each bound sits between.  P0 (DC) and
   V0 (Dirichlet Poisson) are guards, not straddles -- their count barely
   moves with the nodes, so their bound is the measured shipped value and
   only requires the fixed band to stay converged and correct.  The DD
   helm~1e7 lines are ulp-sensitive in k (src/TArray.cpp:23 plans with
   FFTW_MEASURE -- plans are timed, so the plan can differ between builds):
   over 9 repeats of the
   FIXED binary V1 measured 16-25, V3 25-26 and V4 18-26, while all six
   repeats of the SHIPPED binary sat at 31-32; the DD bounds sit above
   the observed fixed-band band and below the shipped one. */
static const point battery_256[] = {
    {"P0", 0.0,       0, 0, 1, 1, 0, 8 }, /* NN DC: guard */
    {"P1", HELM_NN_1, 0, 0, 1, 1, 0, 8 },
    {"P2", HELM_NN_2, 0, 0, 1, 1, 0, 10},
    {"P3", HELM_NN_3, 0, 0, 1, 1, 0, 10},
    {"P4", HELM_NN_1, 0, 0, 1, 1, 1, 18},
    {"P5", HELM_NN_2, 0, 0, 1, 1, 1, 18},
    {"V0", 0.0,       1, 1, 0, 0, 0, 4 }, /* DD Poisson: guard */
    {"V1", HELM_DD_1, 1, 1, 0, 0, 0, 28},
    {"V2", HELM_DD_2, 1, 1, 0, 0, 0, 28},
    {"V3", HELM_DD_1, 1, 1, 0, 0, 1, 29},
    {"V4", HELM_DD_2, 1, 1, 0, 0, 1, 28},
};
static const int n_256 = (int)(sizeof(battery_256) / sizeof(point));

/* N = 1024: the production line size.  V3 is the class that hit the
   solver's iteration cap (itc = -60, NOT converged) on the mis-scaled
   band with a rough right-hand side -- the fixed band reaches 18, so
   the bound sits between the measurement and the cap.  At this grid even
   the DD helm~1e7 lines were bit-stable over 9 repeats (14-19 fixed,
   58/cap shipped); the bounds keep the same between-post-and-shipped
   shape as N=256.  Q0/V0 are the guards (see battery_256). */
static const point battery_1024[] = {
    {"Q0", 0.0,       0, 0, 1, 1, 0, 14}, /* guard */
    {"Q1", HELM_NN_1, 0, 0, 1, 1, 0, 8 },
    {"Q2", HELM_NN_2, 0, 0, 1, 1, 0, 10},
    {"Q3", HELM_NN_3, 0, 0, 1, 1, 0, 10},
    {"Q4", HELM_NN_1, 0, 0, 1, 1, 1, 20},
    {"V0", 0.0,       1, 1, 0, 0, 0, 9 }, /* guard */
    {"V1", HELM_DD_1, 1, 1, 0, 0, 0, 30},
    {"V2", HELM_DD_2, 1, 1, 0, 0, 0, 30},
    {"V3", HELM_DD_1, 1, 1, 0, 0, 1, 24},
    {"V4", HELM_DD_2, 1, 1, 0, 0, 1, 24},
};
static const int n_1024 = (int)(sizeof(battery_1024) / sizeof(point));

/* Relative-tolerance pin on the solution (assertion 1).  The library
   solves to a 1e-7 relative Krylov tolerance (poisson_1d's
   Solve(ans, resid, 1e-7, 20, 3)); 1e-5 carries the operator's
   conditioning between them and is still 5+ orders below the drift a
   wrong preconditioner path produces. */
#define SOL_RTOL 1e-5

/* ------------------------------------------------------------------ */
/* the right-hand sides (deterministic -- the bounds are only real     */
/* if every machine builds the same vectors)                           */
/* ------------------------------------------------------------------ */

/* smooth: a C-infinity function of the Chebyshev coordinate
   sigma_i = cos(pi i/(N-1)) (the operator's nodes are (length/2)*sigma_i
   up to sign; nothing here needs the nodes themselves). */
static void fill_smooth(double *r, int N) {
    for (int i = 0; i < N; i++) {
        double sigma = cos(M_PI * i / (N - 1));
        r[i]         = sin(M_PI * sigma) + 0.3 * cos(3.0 * M_PI * sigma);
    }
}

/* rough: white noise from a fixed integer LCG (reproducible bit-for-bit
   on any two's-complement machine -- the same reason the bounds below
   are meaningful). */
static void fill_rough(double *r, int N) {
    unsigned long s = 20261008UL;
    for (int i = 0; i < N; i++) {
        s    = (1103515245UL * s + 12345UL) & 0x7fffffffUL;
        r[i] = (double)s / 1073741824.0 - 1.0;
    }
}

/* ------------------------------------------------------------------ */
/* the direct reference: probe the pair state, solve it densely        */
/* ------------------------------------------------------------------ */

/* Plain partial-pivoting LU on a dense M x M row-major matrix.  M is at
   most 1025, this runs once per point, and no LAPACK is pulled in that
   the test binary would not already have.  Returns 0 on success. */
static int lu_solve(double *A, double *b, int M) {
    for (int k = 0; k < M; k++) {
        int    piv = k;
        double mx  = fabs(A[k * M + k]);
        for (int i = k + 1; i < M; i++)
            if (fabs(A[i * M + k]) > mx) {
                mx  = fabs(A[i * M + k]);
                piv = i;
            }
        if (mx == 0.0) return 1;
        if (piv != k) {
            for (int j = k; j < M; j++) {
                double t       = A[k * M + j];
                A[k * M + j]   = A[piv * M + j];
                A[piv * M + j] = t;
            }
            double t = b[k];
            b[k]     = b[piv];
            b[piv]   = t;
        }
        for (int i = k + 1; i < M; i++) {
            double f     = A[i * M + k] / A[k * M + k];
            A[i * M + k] = f;
            if (f != 0.0) {
                for (int j = k + 1; j < M; j++) A[i * M + j] -= f * A[k * M + j];
                b[i] -= f * b[k];
            }
        }
    }
    for (int i = M - 1; i >= 0; i--) {
        for (int j = i + 1; j < M; j++) b[i] -= A[i * M + j] * b[j];
        b[i] /= A[i * M + i];
    }
    return 0;
}

/* The pair-state operator of config `p` on grid (N, length), probed
   column-by-column from the library's own matrix_multiply() on a
   SECOND, identically-configured cheb_d2 (poisson_1d's kernel is a
   private static).  The output pair state is zeroed before every apply,
   exactly the state gmres.hpp's fresh vectors are in.  The extra channel
   is live only for the pure-Neumann helm=0 class (matrix_multiply /
   precondition, gmres_1d_solver.cpp:119,252); for every other config the
   pair state's row N and column N are identically zero and the reference
   is the NxN z-block alone. */
static double *probe_operator(int N, double length, const point &p, int &M) {
    cheb_d2 *k = new cheb_d2(N, length);
    k->set_bc(p.helm, p.a_top, p.a_bot, p.b_top, p.b_bot);
    cheb_d2::RType r = k->alloc_resid();
    cheb_d2::BType x = k->alloc_basis();

    int dc = (p.a_top == 0 && p.a_bot == 0 && p.helm == 0);
    M      = dc ? N + 1 : N;

    double *A = new double[M * M];
    memset(A, 0, (size_t)M * M * sizeof(double));

    for (int j = 0; j < M; j++) {
        /* input: e_j on the pair state (zline column j, or the extra channel) */
        *(x->zline) = 0.0;
        x->extra    = 0.0;
        if (j < N) (*x->zline)(0, 0, j) = 1.0;
        else x->extra = 1.0;
        /* zero the output pair state (see the op1d-oracle probe header) */
        *(r->zline) = 0.0;
        r->extra    = 0.0;

        k->matrix_multiply(x, r);

        if (dc) {
            for (int i = 0; i < N; i++) A[i * M + j] = (*r->zline)(0, 0, i);
            A[N * M + j] = r->extra;
        } else {
            for (int i = 0; i < N; i++) A[i * M + j] = (*r->zline)(0, 0, i);
        }
    }
    k->free_resid(r);
    k->free_basis(x);
    delete k;
    return A;
}

/* ------------------------------------------------------------------ */
/* one point: solve (library) - reference (probe+LU) - report          */
/* ------------------------------------------------------------------ */

static int run_point(int N, double length, const point &p) {
    blitz::Array<double, 1> resid(N), soln(N);
    if (p.pool == 0) fill_smooth(resid.data(), N);
    else fill_rough(resid.data(), N);

    int itc = poisson_1d(resid, soln, length, p.helm, p.a_top, p.a_bot, p.b_top, p.b_bot);

    /* Direct reference of the SAME discrete system.  The GMRES right-hand
       side carries extra = 0 (poisson_1d), so the dense RHS is
       [resid; 0]. */
    int     M = 0;
    double *A = probe_operator(N, length, p, M);
    double *b = new double[M];
    for (int i = 0; i < N; i++) b[i] = resid(i);
    if (M > N) b[N] = 0.0;
    int bad = lu_solve(A, b, M);

    double rel = -1.0;
    if (!bad) {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < N; i++) {
            double d = fabs(soln(i) - b[i]);
            if (d > num) num = d;
            if (fabs(b[i]) > den) den = fabs(b[i]);
        }
        rel = (den > 0.0) ? num / den : num;
    }

    int pass = (itc >= 0) && (itc <= p.bound) && !bad && (rel >= 0.0) && (rel <= SOL_RTOL);
    printf("PRECOND_CONV %s N=%d helm=%.6g cls=%s pool=%s itc=%d bound=%d rel=%.3e tol=%.1e %s\n", p.label, N, p.helm, (p.a_top == 0) ? "NN" : "DD", (p.pool == 0) ? "sm" : "rg", itc, p.bound, rel, SOL_RTOL, pass ? "PASS" : "FAIL");
    fflush(stdout);

    delete[] A;
    delete[] b;
    return pass ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int N = 256;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) {
        fprintf(stderr, "usage: %s [N]   (N = 256 or 1024)\n", argv[0]);
        MPI_Finalize();
        return 2;
    }

    const point *bat;
    int          npts;
    if (N == 256) {
        bat  = battery_256;
        npts = n_256;
    } else if (N == 1024) {
        bat  = battery_1024;
        npts = n_1024;
    } else {
        fprintf(stderr, "test_precond_conv: no committed battery for N=%d (256 or 1024)\n", N);
        MPI_Finalize();
        return 2;
    }

    /* The case's length argument as ESolver passes it (the solver
       negates it internally; the operator's nodes are +-|length|/2). */
    double length = 2.0;

    int fails = 0;
    for (int i = 0; i < npts; i++) fails += run_point(N, length, bat[i]);

    printf("PRECOND_CONV SUMMARY N=%d points=%d failures=%d\n", N, npts, fails);
    MPI_Finalize();
    return (fails == 0) ? 0 : 1;
}
