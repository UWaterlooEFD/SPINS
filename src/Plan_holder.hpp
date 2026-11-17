/* Plan_holder.hpp -- utility classes for building the cache of FFTW plans */
#ifndef PLAN_HOLDER_HPP // prevent double inclusion
#define PLAN_HOLDER_HPP 1

#ifdef __DEPRECATED
    #define _OLD_DEPRECATED
    #undef __DEPRECATED
#endif
#include <ext/hash_map>
#ifdef _OLD_DEPRECATED
    #define __DEPRECATED
#endif
#include <utility>
#include <vector>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

namespace TArrayn {

    using __gnu_cxx::hash_map;
    using __gnu_cxx::hash;
    using std::make_pair;
    using std::pair;
    using std::vector;
    using std::cout;
    using std::endl;

    /* What FFTW will assume about a pointer, as a small integer.

       FFTW plans on the alignment of the arrays it is handed, and its generic
       execute entry points (fftw_execute_r2r &c, the ones that let a plan be
       reused for arrays other than the ones it was planned on) do NOT check that
       the arrays they are given still satisfy it -- the manual says a plan "may
       embody an assumption about alignment ... potentially producing wrong
       answers or crashing", and means it.  Measured on this vendor
       (fftw-3.3.8-sse2-avx):
       executing a plan on a differently aligned array returns garbage
       (relative error 4.7e3) or, over a sweep of seven shapes, writes nothing
       at all, silently.  Executing it on a different array of the SAME alignment
       class is bit-identical, for all four transform families and in both plan
       regimes.  So alignment belongs in the key, and alignment class -- not
       address -- is the thing a plan can observe.

       FFTW's own fftw_alignment_of() is the class: it reports the pointer's
       offset, in doubles, from the nearest address aligned to the largest vector
       this build was configured for (this vendor: 64 bytes, so the class is 0 for
       an aligned pointer and 8 for one displaced by a single double).  It is the
       right granularity precisely because it is the number the planner itself
       consults -- every narrower codelet's test is implied by it.

       A coarser class of our own is not an option, and neither is a finer one.
       Measured here, fftw_alloc_real hands out memory aligned to 32 bytes and not
       to 64 (20000 allocations: every one 32-byte aligned, and addr %% 64 lands on
       either 0 or 32), so a hand-rolled class on (% 64) split one geometry's
       arrays into two, and bought a second plan for nothing; fftw_alignment_of
       calls all of them class 0.  The driver's PLAN_CACHE p3_align_class_mismatch
       line is that measurement, and it is what keeps "one plan per geometry" a
       claim about geometry rather than about allocator luck. */
    inline int align_class(const void *p) {
        return fftw_alignment_of((double *)(void *)p);
    }

    struct Plan_spec {
        /* FFTW requires transforms to be 'planned' in advance and executed later.
      This is how the library achieves most of its performance gains -- otherwise
      expensive optimizations can be run once and then taken advantage of many
      times.  Unfortunately, it means that the transforms don't play nicely with
      an object-oriented perspective, where I'd like a transform to be an
      orthogonal operation on the array type.

      The quick-and-ugly solution is to create a plan each time we want to do a
      transform and delete the plan afterwards.  This is bad for obvious reasons.
      Instead, we want to make the plan once and keep it in a cache; the plan
      itself requires negligible memory inside FFTW, so there aren't worries
      about total cache size.

      Step zero is to build a datatype that can uniquely identify a plan, given
      the constraints of SPINS operation.  What identifies a plan is the geometry
      it was planned for -- the same set of values that is handed to
      fftw_plan_many_* in src/TArray.cpp -- plus what FFTW will assume about the
      arrays (in-place-ness and alignment, both above).  Addresses used to be
      here instead of all this, which was a defect wearing the costume of a key:
      two transforms of different length at a reused pair of addresses -- what an
      allocator does all day -- hit the same entry, and the second one ran the
      first one's plan and got wrong numbers without a murmur; the driver in
      src/tests/test_plan_cache.cpp reproduces the
      signature in every plan class).  Keyed on geometry, an address is not part
      of a plan's identity at all, which is also what lets one plan, and one
      scratch buffer, serve a geometry however many arrays ask for it.

      planes and the plane alignment classes are here for the intermediate-stride
      case, where one transform needs one plan per 2D plane (see Plan_holder):
      those plans are built at offsets into the arrays, so the number of planes is
      part of the plan's identity, and -- because what a plan can observe about an
      offset is its alignment -- so is the alignment class of the second plane's
      base.  With the packed layout these arrays are asserted to have, those two
      fix the alignment class of every plane. */

      public:
        const int type, dimension;                        // what, and along which axis
        const int n, howmany;                             // line length, and how many lines
        const int sstride, sdistance, dstride, ddistance; // how the lines are laid out
        const int planes;                                 // plans per call (1 unless intermediate stride)
        const int in_place;                               // planned on one array for both roles?
        const int salign, dalign;                         // alignment class at plan time
        const int splane_align, dplane_align;             // ...of the next plane, when there is one

        Plan_spec(int type_, int dimension_, int n_, int howmany_, int sstride_, int sdistance_, int dstride_, int ddistance_, int planes_, int in_place_, int salign_, int dalign_, int splane_align_, int dplane_align_) :
            type(type_), dimension(dimension_), n(n_), howmany(howmany_), sstride(sstride_), sdistance(sdistance_), dstride(dstride_), ddistance(ddistance_), planes(planes_), in_place(in_place_), salign(salign_), dalign(dalign_), splane_align(splane_align_), dplane_align(dplane_align_) { }
        bool operator==(const Plan_spec &rhs) const {
            // Equality testing
            return (type == rhs.type && dimension == rhs.dimension && n == rhs.n && howmany == rhs.howmany && sstride == rhs.sstride && sdistance == rhs.sdistance && dstride == rhs.dstride && ddistance == rhs.ddistance && planes == rhs.planes && in_place == rhs.in_place && salign == rhs.salign && dalign == rhs.dalign && splane_align == rhs.splane_align && dplane_align == rhs.dplane_align);
        }
    };

    struct plan_hash {
        /* An unfortunate ugliness of the STL extension hash_map is that the
      hash function must be overidden as a structure with operator(),
      rather than as a simple templated function. */
        size_t operator()(const Plan_spec &p) const {
            size_t h = hash<int>()(p.type);
            h        = h * 31 + hash<int>()(p.dimension);
            h        = h * 31 + hash<int>()(p.n);
            h        = h * 31 + hash<int>()(p.howmany);
            h        = h * 31 + hash<int>()(p.sstride);
            h        = h * 31 + hash<int>()(p.sdistance);
            h        = h * 31 + hash<int>()(p.dstride);
            h        = h * 31 + hash<int>()(p.ddistance);
            h        = h * 31 + hash<int>()(p.planes);
            h        = h * 31 + hash<int>()(p.in_place);
            h        = h * 31 + hash<int>()(p.salign);
            h        = h * 31 + hash<int>()(p.dalign);
            h        = h * 31 + hash<int>()(p.splane_align);
            h        = h * 31 + hash<int>()(p.dplane_align);
            return h;
        }
    };

    /* FFTW documents howmany >= 1 in every fftw_plan_many_* call.  A transform
       count of zero means this process owns no lines of the axis that is about
       to be transformed at all -- the rank count exceeds the extent.  What it
       does about that is NOT a fatal error on this vendor: measured on
       fftw-3.3.8,
       fftw_plan_many_r2r and fftw_plan_many_dft_r2c both return a perfectly
       good plan at howmany = 0, under ESTIMATE and MEASURE alike.  So the
       documented precondition is enforced here rather than trusted to a vendor
       that does not check it: an empty batch means the decomposition has run
       out of grid, which is a setup error the model can name in its own words
       and fftw cannot.  Every fftw_plan_many_* call in TArray.cpp goes through
       a Plan_spec, so guarding the two key builders below covers all of them
       (and any that get added).  One caveat for whoever promotes this: no
       communicator is in scope here (this header does no MPI), so the rank
       exits rather than the communicator being aborted, and a rank-local exit
       can leave a neighbour blocked in its next collective -- the very failure
       this guard family exists to end.  Accepted because nothing measured in a
       running case reaches this branch (every zero-count path dies upstream of
       the plan site) and the driver reaches it with no MPI neighbours; if a
       case ever does trip it, thread a comm in and abort that instead. */
    inline void check_num_transforms(int dimension, int n, int howmany) {
        if (howmany < 1) {
            fprintf(stderr, "FATAL: a %d-point transform along dimension %d has no lines to transform: this process owns no part of the array it was asked to transform (np exceeds an axis of the grid) -- ending the run.\n", n, dimension);
            fflush(stderr);
            exit(1);
        }
    }

    /* The key for one transform call: the geometry that fftw_plan_many_* is
       about to be handed, plus what FFTW will assume about the arrays it is
       planned on.  The defaults are the single-plan case; the intermediate-stride
       branches pass the plane count and the byte step between planes.

       The addresses are used here only to ask about alignment and in-place-ness,
       which is what a plan can actually observe about them; they are not stored,
       and no part of a cached plan's identity depends on which array a geometry
       arrived in. */
    inline Plan_spec plan_key(int type, int dimension, const void *source, const void *destination, int n, int howmany, int sstride, int sdistance, int dstride, int ddistance, int planes = 1, const void *source_plane2 = 0, const void *dest_plane2 = 0) {
        check_num_transforms(dimension, n, howmany);
        /* For the multi-plan class the plans are built one per plane, at offsets
           into the arrays, so the plane-to-plane step is part of the plan's
           identity -- and what FFTW can observe about that step is its effect on
           alignment, which is why the step enters as the alignment class of the
           SECOND plane's base.  With the packed (asserted contiguous) layout these
           arrays must have, that and the first plane's class fix the class of
           every plane.  Single-plan calls pass no second plane, and the first
           plane is the array itself. */
        const void *s2 = source_plane2 ? source_plane2 : source;
        const void *d2 = dest_plane2 ? dest_plane2 : destination;
        return Plan_spec(type, dimension, n, howmany, sstride, sdistance, dstride, ddistance, planes, source == destination, align_class(source), align_class(destination), align_class(s2), align_class(d2));
    }

    /* The key for a routed DCT0 call.  The model's arrays are deliberately NOT
       part of this plan's identity: the plan is bound to the holder's own
       even-extension scratch (whose alignment is the holder's business, and the
       same for every entry of a geometry), and the arrays are only read and
       written element by element by the gather and the scatter.  Address,
       alignment and in-place-ness therefore say nothing about the plan here --
       which is exactly why one routed plan, and one scratch, serve every array
       of the shape. */
    inline Plan_spec route_plan_key(int type, int dimension, int n, int howmany, int stride, int distance) {
        check_num_transforms(dimension, n, howmany);
        return Plan_spec(type, dimension, n, howmany, stride, distance, stride, distance, 1, 0, 0, 0, 0, 0);
    }

    class Plan_holder {
        /* FFTW plans are, to quote the documentation, ``an opaque pointer type''.
   We need to keep them around for repeated transforms (the inner loops of
   SPINS), so we want to wrap the plans in a simple class.  The largest
   advantage is that user code will not have to worry at all about properly
   destroying the plans, since it's simple to wrap them up in a reference-
   counted container (this class).  Also, we can entirely wrap the
   fftw_execute_plan function in an object-oriented manner.

   One niggling detail is that transforms of 3D arrays aren't always simple.
   If the dimension to be transformed has the smallest stride, it's easy --
   the transform becomes that of many 1D arrays, interleaved one right after
   the prior.  If the stride is the largest, then the case is also simple --
   there are many arrays, but with a single large stride between elements and a
   single small distance (probably 1) between each array.

   The intermediate case is more difficult, however -- a single stride/distance
   pair is not sufficient.  Thus, we must store and execute many plans, each
   transforming (properly) a 2D slice of the 3D array.  We can wrap that in this
   class by using a vector, allowing user code to only care about ->execute().

   What changed with the geometry key:
   an entry no longer remembers the arrays its plan was planned on, because it no
   longer needs to.  execute() takes the arrays of the call, and passes them to
   FFTW's generic entry points, so a plan keyed on a geometry can serve that
   geometry in whatever array it shows up in next -- which is what makes one plan
   (and, for the routed class, one scratch buffer) serve a whole model's worth of
   arrays.  For the intermediate case the plans were built at offsets into the
   arrays, so those offsets are kept and re-applied to the call's arrays; they are
   relative to data(), which is the (lbound, lbound, lbound) corner, and the first
   plane is asserted to sit exactly there. */
      public:
        /* Which of FFTW's four families a cached plan belongs to, which is what
           selects the execute entry point.  This is not the Trans enum of
           TArray.hpp: this header is included by it, so it cannot name it, and a
           plan does not need to know whether it is a DCT0 or a DCT1 (FFTW baked
           that into the plan) -- only which of the four entry points to call. */
        enum Family {
            R2R, // real to real (the Chebyshev transforms)
            R2C, // real to complex
            C2R, // complex to real
            C2C  // complex to complex
        };

      private:
        vector<fftw_plan> plans;
        vector<size_t>    source_offset, dest_offset; // bytes, one per plan (0 unless multi-plan)
        Family            family;
        int              *refcount;
        /* The DCT0 r2c route (see the policy block in TArray.cpp) plans against
           scratch buffers, so those buffers have to outlive every execute() and
           belong to the cached plan.  Because the plan is now keyed on its
           geometry, one entry -- and therefore exactly one pair of these buffers
           -- serves a geometry however many arrays ask for it, which is the whole
           point of the change (the shipped model counted
           67 entries, and 67 scratch pairs, for one shape in the live case).
           They are inert -- route_in == 0 -- for every plan built the ordinary
           way, which is all any other transform in the model gets. */
        double       *route_in;  /* the even-extension scratch, m per line */
        fftw_complex *route_out; /* the r2c bins, n per line */
        int           route_n, route_howmany, route_stride, route_distance;
        void          _init_route() {
                     route_in  = 0;
                     route_out = 0;
                     route_n = route_howmany = route_stride = route_distance = 0;
        }
        void _free_route() {
            if (route_in) {
                fftw_free(route_in);
                route_in = 0;
            }
            if (route_out) {
                fftw_free(route_out);
                route_out = 0;
            }
        }
        void _check_and_delete() {
            if (refcount && *refcount == 0) { // Last reference to the plans
                delete refcount;
                refcount = 0;
                for (vector<fftw_plan>::iterator i = plans.begin(); i < plans.end(); i++) { fftw_destroy_plan(*i); }
                _free_route(); // ...and with them the scratch they were planned on
            }
        }

      public:
        Plan_holder() // Default constructor -- empty plans
            :
            plans(0), source_offset(0), dest_offset(0), family(R2R), refcount(0) {
            _init_route();
        }
        Plan_holder(fftw_plan singleplan, Family which) // Single plan constructor
            :
            plans(1), source_offset(1, 0), dest_offset(1, 0), family(which) {
            plans[0] = singleplan;
            refcount = new int(1);
            _init_route();
        }
        /* The routed constructor: one plan, the scratch it was planned on, and the
     geometry of the calls it serves (which is what drives the gather and the
     read-out in execute() below).  It takes ownership of in and out, which must
     come from fftw_alloc_real / fftw_alloc_complex.  The model's own arrays are
     deliberately NOT remembered: they are arguments of execute(). */
        Plan_holder(fftw_plan singleplan, Family which, double *in, fftw_complex *out, int n, int howmany, int stride, int distance) : plans(1), source_offset(1, 0), dest_offset(1, 0), family(which) {
            plans[0] = singleplan;
            refcount = new int(1);
            _init_route();
            route_in       = in;
            route_out      = out;
            route_n        = n;
            route_howmany  = howmany;
            route_stride   = stride;
            route_distance = distance;
        }
        /* The intermediate-stride constructor: one plan per 2D plane, each at its
     own offset into the source and destination. */
        Plan_holder(vector<fftw_plan> manyplans, Family which, vector<size_t> soffset, vector<size_t> doffset) : plans(manyplans), source_offset(soffset), dest_offset(doffset), family(which) {
            assert(soffset.size() == manyplans.size() && doffset.size() == manyplans.size());
            refcount = new int(1);
            _init_route();
        }
        Plan_holder(const Plan_holder &copyfrom) { // Copy constructor
            refcount = copyfrom.refcount;
            if (refcount) (*refcount)++;
            plans          = copyfrom.plans;
            source_offset  = copyfrom.source_offset;
            dest_offset    = copyfrom.dest_offset;
            family         = copyfrom.family;
            route_in       = copyfrom.route_in;  // shared, never owned twice: the
            route_out      = copyfrom.route_out; // refcount says who frees them
            route_n        = copyfrom.route_n;
            route_howmany  = copyfrom.route_howmany;
            route_stride   = copyfrom.route_stride;
            route_distance = copyfrom.route_distance;
        }
        ~Plan_holder() { // destructor
            if (refcount) (*refcount)--;
            _check_and_delete();
        }
        Plan_holder &operator=(const Plan_holder &rhs) { // assignment
            if (refcount) {
                (*refcount)--;
                _check_and_delete();
            }
            refcount = rhs.refcount;
            if (refcount) (*refcount)++;
            plans          = rhs.plans;
            source_offset  = rhs.source_offset;
            dest_offset    = rhs.dest_offset;
            family         = rhs.family;
            route_in       = rhs.route_in;
            route_out      = rhs.route_out;
            route_n        = rhs.route_n;
            route_howmany  = rhs.route_howmany;
            route_stride   = rhs.route_stride;
            route_distance = rhs.route_distance;
            return (*this);
        }
        /* Does this entry carry the r2c route, and so own scratch? */
        bool routed() const { return route_in != 0; }
        /* Run the plan(s) on the arrays of THIS call.  Offsets are applied first
           (they are 0 for every case but the intermediate-stride one), and the
           family picks FFTW's generic entry point -- which is what allows a plan
           built on one array to run on another of the same geometry, the claim
           measured on the fftw-3.3.8-sse2-avx build named above. */
        void execute(void *source, void *destination) {
            assert(source && destination);
            if (!route_in) { // The plain case: run the plan(s) where they point
                for (size_t p = 0; p < plans.size(); p++) {
                    char *s = (char *)source + source_offset[p];
                    char *d = (char *)destination + dest_offset[p];
                    switch (family) {
                        case R2C:
                            fftw_execute_dft_r2c(plans[p], (double *)s, (fftw_complex *)d);
                            break;
                        case C2R:
                            fftw_execute_dft_c2r(plans[p], (fftw_complex *)s, (double *)d);
                            break;
                        case C2C:
                            fftw_execute_dft(plans[p], (fftw_complex *)s, (fftw_complex *)d);
                            break;
                        default:
                            fftw_execute_r2r(plans[p], (double *)s, (double *)d);
                            break;
                    }
                }
                return;
            }
            /* The routed pipeline: fill the even extension of every line, run the
               plan, then take the real bins.  The extension is written here, on
               every call, because MEASURE planning destroyed whatever was in the
               scratch while it timed its candidates -- filling before planning
               would transform stale data without a murmur.

               The scratch is shared, so two transforms must not be in the middle
               of this pipeline at once.  SPINS is single-threaded per rank (there
               is no use of threads or OpenMP anywhere in src/), which is what the
               plan cache itself and FFTW's own planner already assume; there is no
               portable way to assert it here, so it is stated as a precondition of
               the cache (see Plan_cache). */
            assert(source != destination); // the route is out-of-place by policy
            const int m = 2 * (route_n - 1);
            for (int j = 0; j < route_howmany; j++) {
                const double *src = (const double *)source + j * route_distance;
                double       *w   = route_in + j * m;
                for (int k = 0; k < route_n; k++) w[k] = src[k * route_stride];
                for (int k = route_n; k < m; k++) w[k] = src[(2 * route_n - 2 - k) * route_stride];
            }
            fftw_execute_dft_r2c(plans[0], route_in, route_out);
            for (int j = 0; j < route_howmany; j++) {
                double             *dst = (double *)destination + j * route_distance;
                const fftw_complex *f   = route_out + j * route_n;
                for (int k = 0; k < route_n; k++) dst[k * route_stride] = f[k][0];
            }
        }
    };

    /* Every plan SPINS has ever built, keyed on its geometry (see Plan_spec).

       This used to be a member of TArray, one map per array, which is what turned
       a single geometry into many plans: the live case built 67 plans and kept 67
       even-extension scratch buffers for one (M=382, batch=256) shape, because
       the model has 67 arrays of it (counted in the shipped model).
       Keyed on geometry, the shape of the container doesn't matter for
       correctness -- so it may as well be the thing that makes each geometry cost
       one plan.  A probe-then-plan scheme left the multiplication in place; this
       stops creating the duplicates, which is the only way to stop paying for
       them.

       Not thread safe, exactly like FFTW's own planner and the map it replaces;
       SPINS runs one rank per thread and never forks a transform (see src/ for
       any use of threads: there are none).

       The table is deliberately leaked.  Destroying FFTW plans during static
       destruction means calling into libfftw3 after whatever it does at exit, in a
       library that is loaded as a shared object -- an order-dependent crash at the
       worst possible moment for diagnosis, spent to free a few dozen plans.  It
       also would not be safe in principle: an entry cannot be released when the
       array that first asked for it goes away, because another array of the same
       geometry is about to ask for the same plan. */
    class Plan_cache {
      public:
        /* The cached plan for this geometry, or 0 if there is none yet. */
        static Plan_holder *lookup(const Plan_spec &key) {
            hash_map<Plan_spec, Plan_holder, plan_hash>::iterator seen = table().find(key);
            return (seen == table().end()) ? 0 : &seen->second;
        }
        /* Take ownership of a freshly built plan and hand back the cache's copy,
           which is the one to execute (and to execute from now on).  If the key
           were already present the plan would be destroyed with the temporary,
           which the assert says can only mean the key failed to pin a plan. */
        static Plan_holder &remember(const Plan_spec &key, const Plan_holder &entry) {
            pair<hash_map<Plan_spec, Plan_holder, plan_hash>::iterator, bool> placed = table().insert(make_pair(key, entry));
            assert(placed.second);
            return placed.first->second;
        }
        /* For diagnostics and the tests: how many geometries have been planned.
           plan_counts (TArray.cpp) counts the same events from the other side --
           plans actually built -- and they agree because the key pins the plan. */
        static size_t size() { return table().size(); }

      private:
        static hash_map<Plan_spec, Plan_holder, plan_hash> &table() {
            static hash_map<Plan_spec, Plan_holder, plan_hash> *the_table = new hash_map<Plan_spec, Plan_holder, plan_hash>();
            return *the_table;
        }
    };

} // namespace TArrayn
#endif
