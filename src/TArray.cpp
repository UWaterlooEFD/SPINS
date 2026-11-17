#include "TArray.hpp"
#include "Plan_holder.hpp"
#include <assert.h>
#include <fftw3.h> // FFTW
/* Implementation of TArray transform functions for the relevant cases.
   A truly general implementation is not possible because fftw:
      1) Only provides a limited set of implementations itself, since it is
      a C library, and
      2) Uses quite distinct function calls for different types of transforms
      (real-to-complex, complex-to-complex, cosine/sine), even ignoring type
      differences.  
   Most of the useful transforms only make sense for certain datatypes.  For
   example, a cosine transform requires both the source and destination arrays
   be real-valued.  Thus, we take advantage of template specialization in 
   C++ to do some additional compile-time error checking.
*/
int plan_counts = 0;
namespace TArrayn {

    using blitz::TinyVector;

    /* ---------------- DCT0 through FFTW's own real-to-complex route ------
       DCT0 is FFTW_REDFT00, and FFTW implements that kind as its own
       even-extension real DFT: REDFT00(n) reduces to a real-input DFT of
       length M = 2(n-1).  Where M carries a large prime factor that
       sub-problem is handed to FFTW's real-input generator, whose candidate
       list holds no Bluestein kernel, and the transform falls off a cliff --
       55 ns/point at the live Nz = 192 against 4.7 ns/point at 193.  Handing
       the identical problem to the COMPLEX generator (the even extension in
       scratch, one fftw_plan_many_dft_r2c, then reading the real bins back)
       reaches dft-bluestein and recovers 3.93-3.96x at Nz = 192 under
       FFTW_MEASURE and 2.36-2.38x under FFTW_ESTIMATE, reproducing REDFT00 to
       <= 6e-15 relative.

       Because that is a last-ulp reformulation and not the same arithmetic,
       the route is confined by two conditions evaluated at PLAN time only (a
       plan-cache miss; the cache hides all of this from execute()): the length
       of the line, and the shape of the plan site being asked to serve it.
       Anything that fails either guard takes today's fftw_plan_many_r2r call,
       with the same arguments, which is what leaves the cases this does not
       serve bit-for-bit unchanged. */

    /* Largest prime factor of m, by 2-then-odd trial division: O(sqrt m), and
       free because it runs once per plan-cache miss. */
    static int largest_prime_factor(int m) {
        if (m <= 3) return m; /* 0, 1, 2 and 3 are their own answer */
        int best = 1;
        while ((m % 2) == 0) {
            best = 2;
            m /= 2;
        }
        for (int d = 3; d * d <= m; d += 2) {
            while ((m % d) == 0) {
                best = d;
                m /= d;
            }
        }
        if (m > 1) best = m;
        return best;
    }

    /* The length half of the rule.  The cut is >= 127, not the predictor's
       "bad stratum" of LPF > 40, because the win tracks the COST of the
       kernel REDFT00 lands on and that saturates only well above 40: every
       routed length wins by a margin (the weakest measured cell is 1.69x)
       while the excluded 40 < LPF < 127 lengths are small wins (1.04-2.07x)
       and n = 160 (LPF 53) is a measured LOSS, 0.81-0.84x under FFTW_ESTIMATE
       at both batch sizes tested.  Moving the cut is a measurement, not a
       policy knob. */
    static int dct0_dip_length(int n) {
        return largest_prime_factor(2 * (n - 1)) >= 127;
    }

    /* The plan-site half.  Three sites reach the code below: the largest
       stride (one plan, transforming IN PLACE -- at np=1 the no-slip x leg),
       the minimum stride (one plan, out of place -- the z leg, which is what
       this route serves), and the intermediate stride (one plan per plane).

       The in-place site is NOT routed.  Its route needs a gather and a copy
       back whose measured price is a win at batch footprints of 128 KiB and
       510 KiB (2.42-2.53x at Nx = 128) and a LOSS at 12.0 MiB (0.86-0.96x at
       (4095, 384)), and no shipped case has a no-slip x leg at a dip length:
       the only (NO_SLIP,NO_SLIP) configuration in the tree,
       src/cases/dipole/spins.conf, is at Nx = Nz = 256, which LPF 17 leaves
       unrouted in any case.  The x leg therefore stays on REDFT00 and the
       constant below records that as a decision rather than an omission -- a
       future leg turns it on here, with the 0.5-12 MiB band measured instead
       of extrapolated). */
    static const int DCT0_ROUTE_IN_PLACE = 0;

    /* The out-of-place class the z route was measured at: the line carries the
       minimum stride, and successive lines lie end to end. */
    static int dct0_r2c_out_of_place_class(int n, int num_transforms, int stride, int distance) {
        if (n < 2 || num_transforms < 1) return 0;
        return (stride == 1 || stride == -1) && distance == n;
    }

    /* The in-place class as it was measured: stride == howmany, distance == 1,
       and a batch footprint no larger than the largest one that won (512 KiB)
       -- beyond that the copy-back is cache-bound and costs more than the
       kernel saves.  Reached only if DCT0_ROUTE_IN_PLACE is turned on. */
    static int dct0_r2c_in_place_class(int n, int num_transforms, int stride, int distance) {
        if (n < 2 || num_transforms < 1) return 0;
        if (stride != num_transforms || distance != 1) return 0;
        return (long long)n * num_transforms * (long long)sizeof(double) <= 524288LL;
    }

    /* The dispatch: the length AND the site's own measured class. */
    static int dct0_r2c_route(int n, int num_transforms, int stride, int distance, int in_place) {
        if (!dct0_dip_length(n)) return 0;
        if (in_place) return DCT0_ROUTE_IN_PLACE && dct0_r2c_in_place_class(n, num_transforms, stride, distance);
        return dct0_r2c_out_of_place_class(n, num_transforms, stride, distance);
    }

    //const int FFTW_PLAN_TYPE = FFTW_EXHAUSTIVE;
    /* Plan selection: FFTW_MEASURE times candidate algorithms, so the plan
       (and hence the rounding of the result) can vary from run to run.  A
       build with -DSPINS_FFTW_ESTIMATE selects FFTW_ESTIMATE instead, which
       picks an algorithm by heuristic alone: the same binary on the same
       machine then always builds the same plans, and no timing loop runs at
       startup.  The default (no flag) is the upstream MEASURE behaviour. */
#ifdef SPINS_FFTW_ESTIMATE
    const int FFTW_PLAN_TYPE = FFTW_ESTIMATE;
#else
    const int FFTW_PLAN_TYPE = FFTW_MEASURE;
#endif

    /* Real-to-real transforms; currently only DCT is implemented.  A DST may
   eventually be useful. */
    template <>
    template <>
    void DTArray::transform<double>(Array<double, 3> &dest, Dimension dimension, Trans trans_type) {
        /* First and foremostly, do error checking on everything possible. */

        /* Assert that the bounds and storage of the source and destination arrays
      are the same */
        assert(all(this->lbound() == dest.lbound()));
        assert(all(this->ubound() == dest.ubound()));
        assert(all(this->stride() == dest.stride()));
        //   assert(!any(blitz_isnan(*this)));

        /* It is perhaps unnecessary, but assert that the storage is continguous */
        assert(this->isStorageContiguous() && dest.isStorageContiguous());

        assert(trans_type == DCT0 || trans_type == DCT1 || trans_type == IDCT1 || trans_type == DST1 || trans_type == IDST1);

        /* Whether the plan for this call is already built is decided inside the
      branches below, once the geometry the call hands FFTW is known: Plan_spec
      keys on that geometry (extent, batch, strides, distances, in-place-ness and
      alignment), so a call whose lines have a different length is a miss rather
      than a stale hit -- the hole the address key left open for every transform
      in the model. */

        fftw_r2r_kind plan_type;
        switch (trans_type) {
            case DCT0:
                plan_type = FFTW_REDFT00;
                break;
            case DCT1:
                plan_type = FFTW_REDFT10;
                break;
            case IDCT1:
                plan_type = FFTW_REDFT01;
                break;
            case DST1:
                plan_type = FFTW_RODFT10;
                break;
            case IDST1:
                plan_type = FFTW_RODFT01;
                break;
            default:
                abort();
        }
        /* A novel geometry: the plan is built below, and kept for the next call
      that asks for the same one. */
        if (this->stride(dimension) == max(this->stride())) {
            /* If this stride is the largest, it means we can do the transform with
         a single FFTW plan.  Each element is separated by stride(dimension),
         and each array is a "distance" of min(stride()) apart. */
            int stride         = this->stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int distance       = this->stride(this->ordering(0));
            int n              = this->extent(dimension);

            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, stride, distance, stride, distance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {                                   // We've seen this plan before
                cached->execute(this->data(), dest.data()); // Execute the plan
                return;                                     // and we're done.
            }
            plan_counts++;

            /* When creating plans with anything other than FFTW_ESTIMATE, the
         plan creation is destructive of the contents of the array.  Since
         we don't want to do a transform of gibberish, we need to copy our
         array's contents someplace safe. */

            Baseclass temp = this->copy();

            fftw_plan major_plan = fftw_plan_many_r2r(1, &n, num_transforms, this->data(), 0, stride, distance, dest.data(), 0, stride, distance, &plan_type, FFTW_PLAN_TYPE);
            assert(major_plan); // Assert that the plan indeed was created

            (*this) = temp; // Copy data back over;

            // Store the plan in the cache and execute
            Plan_cache::remember(this_transform, Plan_holder(major_plan, Plan_holder::R2R)).execute(this->data(), dest.data());
            return;
        } else if (abs(this->stride(dimension)) == abs(this->stride(this->ordering(0)))) {
            /* Minimum stride case -- we have a number of 1D arrays to transform, and
         the beginning of the next is located immediately after the end of
         the previous */
            int stride         = this->stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int distance       = this->extent(dimension);
            int n              = distance;

            /* The route is decided before the cache is consulted, because it
               changes which key the call has: a routed plan is bound to its own
               scratch, not to these arrays, so its key leaves the addresses, the
               alignment and the in-place-ness out (route_plan_key) and every array
               of the shape shares it -- which is the point of the route's entry
               being shared at all (measured:
               counted 67 of them, each with its own scratch, for one shape). */
            const int    routed         = (trans_type == DCT0 && this->data() != dest.data() && dct0_r2c_route(n, num_transforms, stride, distance, 0));
            Plan_spec    this_transform = routed ? route_plan_key(trans_type, dimension, n, num_transforms, stride, distance) : plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, stride, distance, stride, distance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            /* The dipping lengths, out of place only (policy block at the top
               of this file): extend each line to M = 2(n-1) samples in
               scratch, hand that to the complex generator, and read the n
               real bins back -- X_k = Re F_k, which is REDFT00 to <= 6e-15.
               One plan, and it is planned on the scratch rather than on the
               model's arrays, which is also why this path needs no
               temp = this->copy() guard: FFTW's destructive MEASURE planning
               never sees them.

               The ORDER below is load-bearing.  FFTW_MEASURE planning destroys
               the contents of the arrays handed to the planner, so the plan is
               built on the scratch BEFORE the extension is written into it;
               filling first loses the answer without a crash.  The gather and
               the read-out live in Plan_holder, so that a cached plan replays
               the whole pipeline. */
            if (routed) {
                const int m = 2 * (n - 1); /* the extension length */
                assert(m / 2 + 1 == n);    /* ...which r2c answers with exactly n bins */

                double       *ext  = reinterpret_cast<double *>(fftw_alloc_real((size_t)m * num_transforms));
                fftw_complex *bins = reinterpret_cast<fftw_complex *>(fftw_alloc_complex((size_t)n * num_transforms));
                assert(ext && bins);

                /* The embeds are spelled out, rather than left to the NULL
                   convention, because the output line is n bins rather than
                   m. */
                int       inembed    = m;
                int       onembed    = n;
                fftw_plan major_plan = fftw_plan_many_dft_r2c(1, &m, num_transforms, ext, &inembed, 1, m, bins, &onembed, 1, n, FFTW_PLAN_TYPE);
                assert(major_plan);

                Plan_cache::remember(this_transform, Plan_holder(major_plan, Plan_holder::R2C, ext, bins, n, num_transforms, stride, distance)).execute(this->data(), dest.data());
                return;
            }

            Baseclass temp = this->copy();

            fftw_plan major_plan = fftw_plan_many_r2r(1, &n, num_transforms, this->data(), 0, stride, distance, dest.data(), 0, stride, distance, &plan_type, FFTW_PLAN_TYPE);
            assert(major_plan);

            *this = temp;
            Plan_cache::remember(this_transform, Plan_holder(major_plan, Plan_holder::R2R)).execute(this->data(), dest.data());
            return;
        } else {
            /* General case: intermediate stride.  Unfortunately, FFTW3 doesn't
         provide a simple, one-plan solution for this, so we have to build
         a number of plans.  Each will transform a single (memory-contiguous)
         2D slice of the 3D array. */
            assert(dimension == this->ordering(1)); // Sanity check

            int stride   = this->stride(dimension);
            int largedim = this->ordering(2); // Major dimension
            int smalldim = this->ordering(0); // Minor dimension

            int num_transforms = this->extent(smalldim);
            int distance       = this->stride(smalldim);
            int n              = this->extent(dimension);

            /* One plan per 2D plane, each built at its own offset into the two
         arrays.  The offsets are recorded so execute() can re-apply them to the
         arrays of a later call, which is what lets this class of plan be shared
         by geometry too; they are byte offsets from data(), and the assert below
         pins the first plane to data() so that anchor means the same thing in
         every array of the geometry. */
            /* The number of planes, and the alignment class of the second plane's
         base (there is no second one for a single plane, when the first stands
         in), are part of what identifies these plans; see Plan_spec. */
            const int    planes         = this->extent(largedim);
            const void  *splane2        = planes > 1 ? (const void *)(this->data() + this->stride(largedim)) : 0;
            const void  *dplane2        = planes > 1 ? (const void *)(dest.data() + dest.stride(largedim)) : 0;
            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, stride, distance, stride, distance, planes, splane2, dplane2);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass         temp = this->copy();
            vector<fftw_plan> planvec;          // Vector to hold all of our generated plans
            vector<size_t>    soffset, doffset; // ...and where each plane starts

            TinyVector<int, 3> base; // TinyVector to hold the index of our transform
            base(smalldim)  = this->lbound(smalldim);
            base(dimension) = this->lbound(dimension);
            base(largedim)  = this->lbound(largedim); // the corner the offsets below are measured from
            assert(&((*this)(base)) == this->data() && &(dest(base)) == dest.data());

            /* Now, loop over all of our 2D planes and build the FFTW plans.
         As a schematic example, we have a plan for each of (0,0,0), (0,0,1),
         etc -- corresponding to the planes (x,y,0), (x,y,1), etc. */
            for (int kk = this->lbound(largedim); kk <= this->ubound(largedim); kk++) {
                base(largedim)       = kk;
                fftw_plan major_plan = fftw_plan_many_r2r(1, &n, num_transforms, &((*this)(base)), 0, stride, distance, &(dest(base)), 0, stride, distance, &plan_type, FFTW_PLAN_TYPE);
                assert(major_plan);

                planvec.push_back(major_plan);
                soffset.push_back((size_t)((char *)&((*this)(base)) - (char *)this->data()));
                doffset.push_back((size_t)((char *)&(dest(base)) - (char *)dest.data()));
                /* These offsets are re-applied to whichever array of this geometry
             asks for the plan next (see Plan_holder), so they have to be a
             property of the geometry and not of the array: assert the planes are
             uniformly packed.  The first is pinned to data() above, so a uniform
             step fixes every plane's offset from n and num_transforms alone. */
                if (soffset.size() > 1) {
                    assert(soffset.back() - soffset[soffset.size() - 2] == soffset[1] - soffset[0]);
                    assert(doffset.back() - doffset[doffset.size() - 2] == doffset[1] - doffset[0]);
                }
            }

            *this = temp;

            Plan_cache::remember(this_transform, Plan_holder(planvec, Plan_holder::R2R, soffset, doffset)).execute(this->data(), dest.data());
            //      fprintf(stderr,"Plan (array) made and executing: %ld\n",(&plans[this_transform]));
            return;
        }
    }
    typedef complex<double> cdouble;

    /* Real to complex transform, for the FFT of a real-valued source */
    template <>
    template <>
    void DTArray::transform<cdouble>(Array<cdouble, 3> &dest, Dimension dimension, Trans trans_type) {
        /* Error checking is slightly more difficult with R2C transforms, because
      the sizes do not match up perfectly.  A (one-dimension) array of N
      real values transforms to floor(N/2+1) complex values -- the
      ``other half'' are fixed because of symmetry. */

        /* lbounds, however, should be the same. */
        assert(all(this->lbound() == dest.lbound()));
        assert(all(this->ordering() == dest.ordering()));
        assert(this->isStorageContiguous());
        assert(dest.isStorageContiguous());
        assert(trans_type == FFTR);
        //   assert(!any(blitz_isnan(*this)));

        for (Dimension d = firstDim; d <= int(thirdDim); d = Dimension(int(d) + 1)) {
            if (d != dimension) {
                assert(this->ubound(d) == dest.ubound(d));
            } else {
                /* The transformed dimension is halved as part of the real-to-
            complex transform.  A size 1 array (only DC) has a size 1
            output, a size 2 array (DC and Nyquist) has a size 2 output,
            a size 3 array (DC and two non-Nyquist) also has a siz 2 output
            (but the second frequency is not necessarily real), etc. */
                assert(this->lbound(d) == 0);
                assert(dest.lbound(d) == 0);
                assert(dest.ubound(d) == (this->ubound(d) + 1) / 2);
            }
        }
        /* End of error checking */

        /* The plan is looked up per branch, on the geometry that branch hands
           FFTW (see Plan_spec). */
        if (this->stride(dimension) == max(this->stride())) {
            /* Major dimension */
            int sstride        = this->stride(dimension);
            int dstride        = dest.stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int sdistance      = this->stride(this->ordering(0));
            int ddistance      = dest.stride(dest.ordering(0));
            int n              = this->extent(dimension);

            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, sstride, sdistance, dstride, ddistance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass temp = this->copy();

            fftw_plan major_plan = fftw_plan_many_dft_r2c(1, &n, num_transforms, this->data(), 0, sstride, sdistance, reinterpret_cast<fftw_complex *>(dest.data()), 0, dstride, ddistance, FFTW_PLAN_TYPE);
            assert(major_plan); // Assert that the plan was created

            (*this) = temp;

            Plan_cache::remember(this_transform, Plan_holder(major_plan, Plan_holder::R2C)).execute(this->data(), dest.data());
            return;
        } else if (this->stride(dimension) == this->stride(this->ordering(0))) {
            /* Minimum stride case */
            int sstride        = this->stride(dimension);
            int dstride        = dest.stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int sdistance      = this->extent(dimension);
            int ddistance      = dest.extent(dimension);
            int n              = sdistance;

            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, sstride, sdistance, dstride, ddistance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass temp = this->copy();

            fftw_plan minor_plan = fftw_plan_many_dft_r2c(1, &n, num_transforms, this->data(), 0, sstride, sdistance, reinterpret_cast<fftw_complex *>(dest.data()), 0, dstride, ddistance, FFTW_PLAN_TYPE);
            assert(minor_plan);

            (*this) = temp;

            Plan_cache::remember(this_transform, Plan_holder(minor_plan, Plan_holder::R2C)).execute(this->data(), dest.data());
            return;
        } else { // Intermediate case
            assert(dimension == this->ordering(1));
            int sstride        = this->stride(dimension);
            int dstride        = dest.stride(dimension);
            int num_transforms = this->extent(this->ordering(0));
            int sdistance      = this->stride(this->ordering(0));
            int ddistance      = dest.stride(dest.ordering(0));
            int n              = this->extent(dimension);

            int smalldim = this->ordering(0);
            int largedim = this->ordering(2);

            /* The number of planes, and the alignment class of the second plane's
         base, are part of what identifies these plans; see Plan_spec. */
            const int    planes         = this->extent(largedim);
            const void  *splane2        = planes > 1 ? (const void *)(this->data() + this->stride(largedim)) : 0;
            const void  *dplane2        = planes > 1 ? (const void *)(dest.data() + dest.stride(largedim)) : 0;
            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, sstride, sdistance, dstride, ddistance, planes, splane2, dplane2);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass         temp = this->copy();
            vector<fftw_plan> planvec;
            vector<size_t>    soffset, doffset;

            TinyVector<int, 3> base;
            base(smalldim)  = this->lbound(smalldim);
            base(dimension) = this->lbound(dimension);
            base(largedim)  = this->lbound(largedim); // the corner the offsets below are measured from
            /* The per-plane offsets are relative to data(); see the r2r case. */
            assert(&((*this)(base)) == this->data() && &(dest(base)) == dest.data());

            for (int kk = this->lbound(largedim); kk <= this->ubound(largedim); kk++) {
                base(largedim)       = kk;
                fftw_plan major_plan = fftw_plan_many_dft_r2c(1, &n, num_transforms, &((*this)(base)), 0, sstride, sdistance, reinterpret_cast<fftw_complex *>(&dest(base)), 0, dstride, ddistance, FFTW_PLAN_TYPE);
                assert(major_plan);
                planvec.push_back(major_plan);
                soffset.push_back((size_t)((char *)&((*this)(base)) - (char *)this->data()));
                doffset.push_back((size_t)((char *)&(dest(base)) - (char *)dest.data()));
                /* These offsets are re-applied to whichever array of this geometry
             asks for the plan next (see Plan_holder), so they have to be a
             property of the geometry and not of the array: assert the planes are
             uniformly packed.  The first is pinned to data() above, so a uniform
             step fixes every plane's offset from n and num_transforms alone. */
                if (soffset.size() > 1) {
                    assert(soffset.back() - soffset[soffset.size() - 2] == soffset[1] - soffset[0]);
                    assert(doffset.back() - doffset[doffset.size() - 2] == doffset[1] - doffset[0]);
                }
            }

            (*this) = temp;

            Plan_cache::remember(this_transform, Plan_holder(planvec, Plan_holder::R2C, soffset, doffset)).execute(this->data(), dest.data());
            return;
        }
    }

    /* Complex to real transforms, adapted from real-to-complex above */
    template <>
    template <>
    void CTArray::transform<double>(Array<double, 3> &dest, Dimension dimension, Trans trans_type) {
        /* Error checking is slightly more difficult with C2R transforms, because
      the sizes do not match up perfectly.  A (one-dimension) array of N
      real values transforms to floor(N/2+1) complex values -- the
      ``other half'' are fixed because of symmetry. */

        /* lbounds, however, should be the same. */
        assert(all(this->lbound() == dest.lbound()));
        assert(all(this->ordering() == dest.ordering()));
        assert(this->isStorageContiguous());
        assert(dest.isStorageContiguous());
        assert(trans_type == IFFTR);
        //   assert(!any(blitz_isnan(real(*this))));
        //   assert(!any(blitz_isnan(imag(*this))));

        for (Dimension d = firstDim; d <= int(thirdDim); d = Dimension(int(d) + 1)) {
            if (d != dimension) {
                assert(this->ubound(d) == dest.ubound(d));
            } else {
                /* The transformed dimension is halved as part of the real-to-
            complex transform.  A size 1 array (only DC) has a size 1
            output, a size 2 array (DC and Nyquist) has a size 2 output,
            a size 3 array (DC and two non-Nyquist) also has a siz 2 output
            (but the second frequency is not necessarily real), etc. */
                assert(this->lbound(d) == 0);
                assert(dest.lbound(d) == 0);
                assert(this->ubound(d) == (dest.ubound(d) + 1) / 2);
            }
        }
        /* End of error checking */

        /* The plan is looked up per branch, on the geometry that branch hands
           FFTW (see Plan_spec). */
        if (this->stride(dimension) == max(this->stride())) {
            /* Major dimension */
            int sstride        = this->stride(dimension);
            int dstride        = dest.stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int sdistance      = this->stride(this->ordering(0));
            int ddistance      = dest.stride(dest.ordering(0));
            int n              = dest.extent(dimension);

            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, sstride, sdistance, dstride, ddistance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass temp = this->copy();

            fftw_plan major_plan = fftw_plan_many_dft_c2r(1, &n, num_transforms, reinterpret_cast<fftw_complex *>(this->data()), 0, sstride, sdistance, (dest.data()), 0, dstride, ddistance, FFTW_PLAN_TYPE | FFTW_PRESERVE_INPUT);
            assert(major_plan); // Assert that the plan was created

            (*this) = temp;

            Plan_cache::remember(this_transform, Plan_holder(major_plan, Plan_holder::C2R)).execute(this->data(), dest.data());
            return;
        } else if (this->stride(dimension) == this->stride(this->ordering(0))) {
            /* Minimum stride case */
            int sstride        = this->stride(dimension);
            int dstride        = dest.stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int sdistance      = this->extent(dimension);
            int ddistance      = dest.extent(dimension);
            int n              = ddistance;

            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, sstride, sdistance, dstride, ddistance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass temp = this->copy();

            fftw_plan minor_plan = fftw_plan_many_dft_c2r(1, &n, num_transforms, reinterpret_cast<fftw_complex *>(this->data()), 0, sstride, sdistance, (dest.data()), 0, dstride, ddistance, FFTW_PLAN_TYPE);
            assert(minor_plan);

            (*this) = temp;

            Plan_cache::remember(this_transform, Plan_holder(minor_plan, Plan_holder::C2R)).execute(this->data(), dest.data());
            return;
        } else { // Intermediate case
            assert(dimension == this->ordering(1));
            int sstride        = this->stride(dimension);
            int dstride        = dest.stride(dimension);
            int num_transforms = this->extent(this->ordering(0));
            int sdistance      = this->stride(this->ordering(0));
            int ddistance      = dest.stride(dest.ordering(0));
            int n              = dest.extent(dimension);

            int smalldim = this->ordering(0);
            int largedim = this->ordering(2);

            /* The number of planes, and the alignment class of the second plane's
         base, are part of what identifies these plans; see Plan_spec. */
            const int    planes         = this->extent(largedim);
            const void  *splane2        = planes > 1 ? (const void *)(this->data() + this->stride(largedim)) : 0;
            const void  *dplane2        = planes > 1 ? (const void *)(dest.data() + dest.stride(largedim)) : 0;
            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, sstride, sdistance, dstride, ddistance, planes, splane2, dplane2);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass         temp = this->copy();
            vector<fftw_plan> planvec;
            vector<size_t>    soffset, doffset;

            TinyVector<int, 3> base;
            base(smalldim)  = this->lbound(smalldim);
            base(dimension) = this->lbound(dimension);
            base(largedim)  = this->lbound(largedim); // the corner the offsets below are measured from
            /* The per-plane offsets are relative to data(); see the r2r case. */
            assert(&((*this)(base)) == this->data() && &(dest(base)) == dest.data());

            for (int kk = this->lbound(largedim); kk <= this->ubound(largedim); kk++) {
                base(largedim)       = kk;
                fftw_plan major_plan = fftw_plan_many_dft_c2r(1, &n, num_transforms, reinterpret_cast<fftw_complex *>(&(*this)(base)), 0, sstride, sdistance, (&dest(base)), 0, dstride, ddistance, FFTW_PLAN_TYPE);
                assert(major_plan);
                planvec.push_back(major_plan);
                soffset.push_back((size_t)((char *)&((*this)(base)) - (char *)this->data()));
                doffset.push_back((size_t)((char *)&(dest(base)) - (char *)dest.data()));
                /* These offsets are re-applied to whichever array of this geometry
             asks for the plan next (see Plan_holder), so they have to be a
             property of the geometry and not of the array: assert the planes are
             uniformly packed.  The first is pinned to data() above, so a uniform
             step fixes every plane's offset from n and num_transforms alone. */
                if (soffset.size() > 1) {
                    assert(soffset.back() - soffset[soffset.size() - 2] == soffset[1] - soffset[0]);
                    assert(doffset.back() - doffset[doffset.size() - 2] == doffset[1] - doffset[0]);
                }
            }

            (*this) = temp;

            Plan_cache::remember(this_transform, Plan_holder(planvec, Plan_holder::C2R, soffset, doffset)).execute(this->data(), dest.data());
            return;
        }
    }

    /* Complex->Complex FFT, based off of the DCT code.  Preconditions on array
   size remain the same, since unlike real-to-complex transforms a dimension
   is not halved in size. */
    template <>
    template <>
    void CTArray::transform<cdouble>(Array<cdouble, 3> &dest, Dimension dimension, Trans trans_type) {
        /* First and foremostly, do error checking on everything possible. */

        /* Assert that the bounds and storage of the source and destination arrays
      are the same */
        assert(all(this->lbound() == dest.lbound()));
        assert(all(this->ubound() == dest.ubound()));
        assert(all(this->stride() == dest.stride()));
        //   assert(!any(blitz_isnan(real(*this))));
        //   assert(!any(blitz_isnan(imag(*this))));

        /* It is perhaps unnecessary, but assert that the storage is continguous */
        assert(this->isStorageContiguous() && dest.isStorageContiguous());

        assert(trans_type == FFT || trans_type == IFFT);

        /* The plan is looked up per branch, on the geometry that branch hands
           FFTW (see Plan_spec). */
        int trans_sign;
        /* Check for forward or backward FFT */
        if (trans_type == FFT) trans_sign = FFTW_FORWARD;
        if (trans_type == IFFT) trans_sign = FFTW_BACKWARD;
        /* This is a novel transform, one we haven't seen before.  Build the
      FFTW plans */
        if (this->stride(dimension) == max(this->stride())) {
            /* If this stride is the largest, it means we can do the transform with
         a single FFTW plan.  Each element is separated by stride(dimension),
         and each array is a "distance" of min(stride()) apart. */
            int stride         = this->stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int distance       = this->stride(this->ordering(0));
            int n              = this->extent(dimension);

            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, stride, distance, stride, distance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            /* When creating plans with anything other than FFTW_ESTIMATE, the
         plan creation is destructive of the contents of the array.  Since
         we don't want to do a transform of gibberish, we need to copy our
         array's contents someplace safe. */

            Baseclass temp = this->copy();

            fftw_plan major_plan = fftw_plan_many_dft(1, &n, num_transforms, reinterpret_cast<fftw_complex *>(this->data()), 0, stride, distance, reinterpret_cast<fftw_complex *>(dest.data()), 0, stride, distance, trans_sign, FFTW_PLAN_TYPE);
            assert(major_plan); // Assert that the plan indeed was created

            (*this) = temp; // Copy data back over;

            // Store the plan in the cache and execute
            Plan_cache::remember(this_transform, Plan_holder(major_plan, Plan_holder::C2C)).execute(this->data(), dest.data());
            return;
        } else if (abs(this->stride(dimension)) == abs(this->stride(this->ordering(0)))) {
            /* Minimum stride case -- we have a number of 1D arrays to transform, and
         the beginning of the next is located immediately after the end of
         the previous */
            int stride         = this->stride(dimension);
            int num_transforms = this->numElements() / this->extent(dimension);
            int distance       = this->extent(dimension);
            int n              = distance;

            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, stride, distance, stride, distance);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass temp = this->copy();

            fftw_plan major_plan = fftw_plan_many_dft(1, &n, num_transforms, reinterpret_cast<fftw_complex *>(this->data()), 0, stride, distance, reinterpret_cast<fftw_complex *>(dest.data()), 0, stride, distance, trans_sign, FFTW_PLAN_TYPE);
            assert(major_plan);

            *this = temp;
            Plan_cache::remember(this_transform, Plan_holder(major_plan, Plan_holder::C2C)).execute(this->data(), dest.data());
            return;
        } else {
            /* General case: intermediate stride.  Unfortunately, FFTW3 doesn't
         provide a simple, one-plan solution for this, so we have to build
         a number of plans.  Each will transform a single (memory-contiguous)
         2D slice of the 3D array. */
            assert(dimension == this->ordering(1)); // Sanity check

            int stride   = this->stride(dimension);
            int largedim = this->ordering(2); // Major dimension
            int smalldim = this->ordering(0); // Minor dimension

            int num_transforms = this->extent(smalldim);
            int distance       = this->stride(smalldim);
            int n              = this->extent(dimension);

            /* The number of planes, and the alignment class of the second plane's
         base (there is no second one for a single plane, when the first stands
         in), are part of what identifies these plans; see Plan_spec. */
            const int    planes         = this->extent(largedim);
            const void  *splane2        = planes > 1 ? (const void *)(this->data() + this->stride(largedim)) : 0;
            const void  *dplane2        = planes > 1 ? (const void *)(dest.data() + dest.stride(largedim)) : 0;
            Plan_spec    this_transform = plan_key(trans_type, dimension, this->data(), dest.data(), n, num_transforms, stride, distance, stride, distance, planes, splane2, dplane2);
            Plan_holder *cached         = Plan_cache::lookup(this_transform);
            if (cached) {
                cached->execute(this->data(), dest.data());
                return;
            }
            plan_counts++;

            Baseclass         temp = this->copy();
            vector<fftw_plan> planvec;          // Vector to hold all of our generated plans
            vector<size_t>    soffset, doffset; // ...and where each plane starts

            TinyVector<int, 3> base; // TinyVector to hold the index of our transform
            base(smalldim)  = this->lbound(smalldim);
            base(dimension) = this->lbound(dimension);
            base(largedim)  = this->lbound(largedim); // the corner the offsets below are measured from
            /* The per-plane offsets are relative to data(); see the r2r case. */
            assert(&((*this)(base)) == this->data() && &(dest(base)) == dest.data());

            /* Now, loop over all of our 2D planes and build the FFTW plans.
         As a schematic example, we have a plan for each of (0,0,0), (0,0,1),
         etc -- corresponding to the planes (x,y,0), (x,y,1), etc. */
            for (int kk = this->lbound(largedim); kk <= this->ubound(largedim); kk++) {
                base(largedim)       = kk;
                fftw_plan major_plan = fftw_plan_many_dft(1, &n, num_transforms, reinterpret_cast<fftw_complex *>(&((*this)(base))), 0, stride, distance, reinterpret_cast<fftw_complex *>(&(dest(base))), 0, stride, distance, trans_sign, FFTW_PLAN_TYPE);
                assert(major_plan);

                planvec.push_back(major_plan);
                soffset.push_back((size_t)((char *)&((*this)(base)) - (char *)this->data()));
                doffset.push_back((size_t)((char *)&(dest(base)) - (char *)dest.data()));
                /* These offsets are re-applied to whichever array of this geometry
             asks for the plan next (see Plan_holder), so they have to be a
             property of the geometry and not of the array: assert the planes are
             uniformly packed.  The first is pinned to data() above, so a uniform
             step fixes every plane's offset from n and num_transforms alone. */
                if (soffset.size() > 1) {
                    assert(soffset.back() - soffset[soffset.size() - 2] == soffset[1] - soffset[0]);
                    assert(doffset.back() - doffset[doffset.size() - 2] == doffset[1] - doffset[0]);
                }
            }

            *this = temp;

            Plan_cache::remember(this_transform, Plan_holder(planvec, Plan_holder::C2C, soffset, doffset)).execute(this->data(), dest.data());
            return;
        }
    }

    Trans invert_trans(Trans in) {
        /* Helper function to return proper inverse transform for a given type */
        switch (in) {
            case DCT0:
                return DCT0;
            case DCT1:
                return IDCT1;
            case IDCT1:
                return DCT1;
            case DST1:
                return IDST1;
            case IDST1:
                return DST1;
            case FFTR:
                return IFFTR;
            case IFFTR:
                return FFTR;
            case FFT:
                return IFFT;
            case IFFT:
                return FFT;
            default:
                abort();
        }
    }

} // namespace TArrayn
