/* test_plan_cache.cpp -- what a cached FFTW plan is keyed on.  The driver is
   a repro: it exercises the stale-address collision, so it can be read
   before the fix that removes it.

   Plan_spec (src/Plan_holder.hpp) named a cached plan by (source address,
   destination address, kind, dimension), while the plan itself was built from
   n = extent(dimension) and the strides of whichever array happened to ask
   first (src/TArray.cpp).  Two geometries at one pair of addresses is not
   exotic, it is what an allocator does all day, and the second geometry then
   runs the first geometry's plan.  The route's stale-geometry guard
   (dct0_route_is_stale, 2026-10-10) catches that collision for ROUTED plans
   only; the rest of the hole is the plain case: a cached
   plain-r2r plan for a different n at a reused address is hit and returns wrong
   numbers.  This driver hits it, in every plan class that can be hit in-bounds,
   and counts the cost of the other half of the record.

   The address-reuse pattern is set up rather than hoped for.  The cache lives
   inside the source TArray, so the exercise needs a live TArray object whose
   geometry changes underneath its own cache: two geometries with the SAME total
   element count are laid over ONE pair of fftw_alloc'd buffers, and Blitz's
   reference() rebinds storage and shape in place (plans, a member of the
   derived TArray, survives).  The addresses are therefore identical by
   construction -- no allocator behaviour to pray over.  Every pair of geometries
   here keeps the stale plan inside its own buffers, so the signature is a WRONG
   NUMBER rather than a segfault, which is what makes it worth pinning: the
   reference for each is the same transform done by a FRESH array pair, i.e. a
   fresh cache, i.e. a plan built for the length actually in front of it.

     part 1  minimum-stride, out-of-place, ordinary fftw_plan_many_r2r:
             (8,1,24) then (24,1,8), both z contiguous, 192 doubles.  Neither
             length routes (LPF(2*23)=23, LPF(2*7)=7), so the route's guard is
             off-screen.  The stale plan's last touched index is 7*24+23 = 191,
             inside the buffer.  This is the branch the live z leg uses.
     part 2  five independent array pairs at ONE routed geometry (z = 192, the
             live dip length, batch 8): how many plans, and how much of the
             record's "+100 MiB of per-plan scratch" does one shape cost?
             (counted in the shipped model: 67 plans for one
             (M=382, batch=256) shape in the live case.)
     part 3  the same five pairs at one NON-routed geometry (z = 160): the
             per-array plan cache is per ARRAY, so the multiplication is not a
             property of the route at all.
     part 4  the complex classes on one shared pair of buffers: real-to-complex
             and complex-to-real, geometry A then geometry B.  The c2r leg is
             the one a geometry-keyed cache silently zeroes (fftw_execute_dft_c2r
             does not overwrite the imaginary parts FFTW thinks are redundant),
             so its signature is worth seeing next to the others.
     part 5  the intermediate-stride class (the multi-plan one, which caches
             several plans with per-plane offsets): (4,6,8) then (8,3,8),
             transform along y.

   Convention per src/tests/test_cheby.cpp: MPI_Init, master() prints,
   machine-readable PLAN_CACHE lines for whatever harness runs this driver
   to assert on.  The exit status is 0 unless a
   part whose claim is "the addresses collided" did not actually collide, in
   which case it says so and returns 2 -- a driver must never report success on
   a run that did not exercise the hazard. */
#include <mpi.h>
#include <blitz/array.h>
#include <complex>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../Par_util.hpp"
#include "../TArray.hpp"

/* TArray.cpp's plan-cache-miss counter (defined there, declared nowhere). */
extern int plan_counts;

using namespace blitz;
using namespace TArrayn;
using std::complex;
using std::vector;

/* blitz::thirdDim and TArrayn::thirdDim are the same enumerator, but both
   namespaces are imported above (as test_cheby.cpp does), so the transform
   dimension is named once here instead of qualified at every call. */
static const Dimension ZDIM = TArrayn::thirdDim;
static const Dimension YDIM = TArrayn::secondDim;

/* z contiguous, like the live z leg (SPINS' alloc_storage comment: "single
   processor runs use default C-order", and the z leg does hit the
   minimum-stride branch). */
static GeneralArrayStorage<3> z_first(void) {
    GeneralArrayStorage<3> s;
    s.ordering()[0] = TArrayn::thirdDim;
    s.ordering()[1] = TArrayn::secondDim;
    s.ordering()[2] = TArrayn::firstDim;
    return s;
}

/* f(i,j,k): the same values in the hazard array and in its reference, index for
   index, so any difference can only have come from the plan. */
static void fill(Array<double, 3> &a) {
    for (int i = a.lbound(0); i <= a.ubound(0); i++)
        for (int j = a.lbound(1); j <= a.ubound(1); j++)
            for (int k = a.lbound(2); k <= a.ubound(2); k++) a(i, j, k) = 1.0 + 0.25 * i - 0.05 * j + 0.1 * cos(0.7 * k) + 0.02 * k;
}

static void fill(Array<complex<double>, 3> &a) {
    for (int i = a.lbound(0); i <= a.ubound(0); i++)
        for (int j = a.lbound(1); j <= a.ubound(1); j++)
            for (int k = a.lbound(2); k <= a.ubound(2); k++) a(i, j, k) = complex<double>(0.3 * i - 0.1 * j, 0.2 * cos(0.4 * k) - 0.05 * k);
}

/* max |x - y| over the pair and max |y| (the scale to quote it against), real. */
static void compare(Array<double, 3> &x, Array<double, 3> &y, double *absmax, double *scale) {
    *absmax = 0.0;
    *scale  = 0.0;
    for (int i = x.lbound(0); i <= x.ubound(0); i++)
        for (int j = x.lbound(1); j <= x.ubound(1); j++)
            for (int k = x.lbound(2); k <= x.ubound(2); k++) {
                double d = fabs(x(i, j, k) - y(i, j, k));
                if (d > *absmax) *absmax = d;
                if (fabs(y(i, j, k)) > *scale) *scale = fabs(y(i, j, k));
            }
}

static void compare(Array<complex<double>, 3> &x, Array<complex<double>, 3> &y, double *absmax, double *scale) {
    *absmax = 0.0;
    *scale  = 0.0;
    for (int i = x.lbound(0); i <= x.ubound(0); i++)
        for (int j = x.lbound(1); j <= x.ubound(1); j++)
            for (int k = x.lbound(2); k <= x.ubound(2); k++) {
                double d = fabs(x(i, j, k) - y(i, j, k));
                if (d > *absmax) *absmax = d;
                if (fabs(y(i, j, k)) > *scale) *scale = fabs(y(i, j, k));
            }
}

/* The one report line every part ends with: the collision flag that makes the
   claim meaningful, the plans the second call cost (0 == a stale plan was
   reused), and how far the answer strayed from the fresh-cache reference. */
static void report(const char *name, int collide, int rebuilt, double absmax, double scale) {
    if (master()) {
        printf("PLAN_CACHE %s_address_collision = %d\n", name, collide);
        printf("PLAN_CACHE %s_plans_rebuilt = %d\n", name, rebuilt);
        printf("PLAN_CACHE %s_absmax = %.6e\n", name, absmax);
        printf("PLAN_CACHE %s_scale = %.6e\n", name, scale);
        printf("PLAN_CACHE %s_rel = %.6e\n", name, absmax / (scale == 0.0 ? 1.0 : scale));
    }
}

/* How many plans does one geometry cost when K independent array pairs ask for
   it, and do the K answers agree?  The live case's answer for one routed
   (M=382, batch=256) shape was 67 plans and 67 even-extension scratch pairs
   counted the same way below; plan_counts is
   incremented on every plan-cache miss, so its delta over the K calls is the
   count directly, with no interposer. */
static void plan_cost_of_K(const char *name, int K, TinyVector<int, 3> shape, GeneralArrayStorage<3> store, int *collide_bad) {
    size_t   nbytes = sizeof(double) * shape[0] * shape[1] * shape[2];
    double **sb     = new double *[K];
    double **db     = new double *[K];
    for (int p = 0; p < K; p++) {
        sb[p] = (double *)fftw_alloc_real(nbytes);
        db[p] = (double *)fftw_alloc_real(nbytes);
    }
    int    before         = plan_counts;
    int    collide        = 0;
    int    align_mismatch = 0;
    double worst          = 0.0;
    for (int p = 0; p < K; p++) {
        DTArray src(Array<double, 3>(sb[p], shape, store));
        DTArray dst(Array<double, 3>(db[p], shape, store));
        if (p and ((sb[p] == sb[0]) or (db[p] == db[0]))) { collide = 1; }
        /* A plan may only be shared by arrays of one alignment class (see
           align_class, Plan_holder.hpp), so the count below means what it is
           supposed to mean only if these pairs are all one class. */
        if (p and (align_class(sb[p]) != align_class(sb[0]) or align_class(db[p]) != align_class(db[0]))) { align_mismatch++; }
        fill(src);
        src.transform<double>(dst, ZDIM, TArrayn::DCT0);
        if (p) {
            DTArray first(Array<double, 3>(db[0], shape, store));
            double  am = 0.0, sc = 0.0;
            compare(dst, first, &am, &sc);
            if (am / (sc + 1e-300) > worst) worst = am / (sc + 1e-300);
        }
    }
    int plans = plan_counts - before;
    if (master()) {
        printf("PLAN_CACHE %s_pairs = %d\n", name, K);
        printf("PLAN_CACHE %s_n = %d\n", name, shape[2]);
        printf("PLAN_CACHE %s_howmany = %d\n", name, (int)(shape[0] * shape[1] * shape[2]) / shape[2]);
        printf("PLAN_CACHE %s_plans = %d\n", name, plans);
        printf("PLAN_CACHE %s_distinct_addresses = %d\n", name, collide ? 0 : 1);
        printf("PLAN_CACHE %s_align_class_mismatch = %d\n", name, align_mismatch);
        printf("PLAN_CACHE %s_align_class_src0 = %d\n", name, align_class(sb[0]));
        printf("PLAN_CACHE %s_worst_pair_rel = %.6e\n", name, worst);
    }
    if (collide) (*collide_bad)++;
    for (int p = 0; p < K; p++) {
        fftw_free(sb[p]);
        fftw_free(db[p]);
    }
    delete[] sb;
    delete[] db;
}

/* One address-reuse hazard: transform geometry A, rebind the SAME live TArray
   pair onto geometry B over the SAME buffers, transform again, and compare the
   second answer against a fresh pair that had no choice but to plan for B. */
static void hazard_real(const char *name, TinyVector<int, 3> A, TinyVector<int, 3> B, GeneralArrayStorage<3> store, Dimension dim, int *collide_bad) {
    size_t total = (size_t)A[0] * A[1] * A[2];
    if ((size_t)B[0] * B[1] * B[2] != total) {
        printf("hazard_real: geometries must hold the same number of elements\n");
        exit(3);
    }
    double *sbuf = (double *)fftw_alloc_real(total);
    double *dbuf = (double *)fftw_alloc_real(total);
    double *rib  = (double *)fftw_alloc_real(total);
    double *rob  = (double *)fftw_alloc_real(total);

    DTArray src(Array<double, 3>(sbuf, A, store)), dst(Array<double, 3>(dbuf, A, store));
    DTArray ref_s(Array<double, 3>(rib, B, store)), ref_d(Array<double, 3>(rob, B, store));

    fill(src);
    int before = plan_counts;
    src.transform<double>(dst, dim, TArrayn::DCT0); /* a plan for A's length */
    before = plan_counts;

    src.reference(Array<double, 3>(sbuf, B, store)); /* same address, new geometry */
    dst.reference(Array<double, 3>(dbuf, B, store));
    fill(src);
    src.transform<double>(dst, dim, TArrayn::DCT0);
    int rebuilt = plan_counts - before;

    fill(ref_s);
    ref_s.transform<double>(ref_d, dim, TArrayn::DCT0);

    double absmax = 0.0, scale = 0.0;
    compare(dst, ref_d, &absmax, &scale);
    int collide = ((const char *)src.data() == (const char *)sbuf) && ((const char *)dst.data() == (const char *)dbuf);
    if (not collide) { (*collide_bad)++; }
    if (master()) {
        printf("PLAN_CACHE %s_extent_first_call = %d\n", name, A[dim]);
        printf("PLAN_CACHE %s_extent_second_call = %d\n", name, B[dim]);
    }
    report(name, collide, rebuilt, absmax, scale);

    fftw_free(sbuf);
    fftw_free(dbuf);
    fftw_free(rib);
    fftw_free(rob);
}

/* The complex pair of classes, same trick.  The real buffer holds 192 doubles
   under both geometries; the complex buffer holds the larger of the two
   half-spectra. */
static void hazard_complex(const char *name, TinyVector<int, 3> Ar, TinyVector<int, 3> Br, GeneralArrayStorage<3> store, int r2c, int *collide_bad) {
    size_t             total = (size_t)Ar[0] * Ar[1] * Ar[2];
    TinyVector<int, 3> Ac(Ar[0], Ar[1], Ar[2] / 2 + 1), Bc(Br[0], Br[1], Br[2] / 2 + 1);
    size_t             nc = (size_t)Ac[0] * Ac[1] * Ac[2], nb = (size_t)Bc[0] * Bc[1] * Bc[2];
    size_t             cplx = (nc > nb ? nc : nb) * sizeof(fftw_complex);

    double       *sbuf = (double *)fftw_alloc_real(total);
    double       *dbuf = (double *)fftw_alloc_real(total);
    double       *rib  = (double *)fftw_alloc_real(total);
    double       *rob  = (double *)fftw_alloc_real(total);
    fftw_complex *scb  = (fftw_complex *)fftw_alloc_complex(cplx);
    fftw_complex *dcb  = (fftw_complex *)fftw_alloc_complex(cplx);
    fftw_complex *ric  = (fftw_complex *)fftw_alloc_complex(cplx);
    fftw_complex *roc  = (fftw_complex *)fftw_alloc_complex(cplx);

    /* r2c: real source, complex destination.  c2r: the other way round, and the
       complex array is the one whose cache would go stale. */
    int    collide = 0;
    int    rebuilt = 0;
    double absmax = 0.0, scale = 0.0;
    if (r2c) {
        DTArray src(Array<double, 3>(sbuf, Ar, store));
        CTArray dst(Array<complex<double>, 3>((complex<double> *)dcb, Ac, store));
        DTArray ref_s(Array<double, 3>(rib, Br, store));
        CTArray ref_d(Array<complex<double>, 3>((complex<double> *)roc, Bc, store));
        fill(src);
        src.transform<complex<double>>(dst, ZDIM, TArrayn::FFTR);
        src.reference(Array<double, 3>(sbuf, Br, store));
        dst.reference(Array<complex<double>, 3>((complex<double> *)dcb, Bc, store));
        fill(src);
        int before = plan_counts;
        src.transform<complex<double>>(dst, ZDIM, TArrayn::FFTR);
        rebuilt = plan_counts - before;
        fill(ref_s);
        ref_s.transform<complex<double>>(ref_d, ZDIM, TArrayn::FFTR);
        compare(dst, ref_d, &absmax, &scale);
        collide = ((const char *)src.data() == (const char *)sbuf) && ((const char *)dst.data() == (const char *)dcb);
    } else {
        CTArray src(Array<complex<double>, 3>((complex<double> *)scb, Ac, store));
        DTArray dst(Array<double, 3>(dbuf, Ar, store));
        CTArray ref_s(Array<complex<double>, 3>((complex<double> *)ric, Bc, store));
        DTArray ref_d(Array<double, 3>(rob, Br, store));
        fill(src);
        src.transform<double>(dst, ZDIM, TArrayn::IFFTR);
        src.reference(Array<complex<double>, 3>((complex<double> *)scb, Bc, store));
        dst.reference(Array<double, 3>(dbuf, Br, store));
        fill(src);
        int before = plan_counts;
        src.transform<double>(dst, ZDIM, TArrayn::IFFTR);
        rebuilt = plan_counts - before;
        fill(ref_s);
        ref_s.transform<double>(ref_d, ZDIM, TArrayn::IFFTR);
        compare(dst, ref_d, &absmax, &scale);
        collide = ((const char *)src.data() == (const char *)scb) && ((const char *)dst.data() == (const char *)dbuf);
    }
    if (not collide) { (*collide_bad)++; }
    report(name, collide, rebuilt, absmax, scale);

    fftw_free(sbuf);
    fftw_free(dbuf);
    fftw_free(rib);
    fftw_free(rob);
    fftw_free(scb);
    fftw_free(dcb);
    fftw_free(ric);
    fftw_free(roc);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    GeneralArrayStorage<3> store = z_first();
    int                    bad   = 0;

    /* part 1: the minimum-stride class, ordinary r2r -- the branch the live z
       leg uses, at lengths too short to route. */
    hazard_real("p1", TinyVector<int, 3>(8, 1, 24), TinyVector<int, 3>(24, 1, 8), store, ZDIM, &bad);

    /* parts 2 and 3: what one geometry costs when several arrays ask for it. */
    plan_cost_of_K("p2", 5, TinyVector<int, 3>(4, 2, 192), store, &bad); /* the live dip length */
    plan_cost_of_K("p3", 5, TinyVector<int, 3>(4, 2, 160), store, &bad); /* not a dip length */

    /* part 4: the complex classes on one shared pair of buffers. */
    hazard_complex("p4a", TinyVector<int, 3>(8, 1, 24), TinyVector<int, 3>(24, 1, 8), store, 1, &bad);
    hazard_complex("p4b", TinyVector<int, 3>(8, 1, 24), TinyVector<int, 3>(24, 1, 8), store, 0, &bad);

    /* part 5: the intermediate-stride class -- y is neither the fastest nor the
       slowest stride, so this is the branch that caches SEVERAL plans with
       per-plane offsets. */
    hazard_real("p5", TinyVector<int, 3>(4, 6, 8), TinyVector<int, 3>(8, 3, 8), store, YDIM, &bad);

    if (master()) { printf("PLAN_CACHE parts_with_failed_premise = %d\n", bad); }

    MPI_Finalize();
    return bad == 0 ? 0 : 2;
}
