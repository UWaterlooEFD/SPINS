#ifndef MULTIGRID_HPP
#define MULTIGRID_HPP 1

#include <mpi.h>
#include <blitz/array.h>
#include <vector>
#include "umfpack.h"

/* Header file for our required 2D multigrid solver, operating on a (split) grid
   with the numerical Jacobian entries (from the spectral model) providing varying
   coefficients */

extern double coarse_solve_time, redblack_time, apply_op_time;
extern int    coarse_solve_count, redblack_count, apply_op_count;

enum SYM_TYPE {
    SYM_NONE = 0, // No symmetry
    SYM_PERIODIC, // Periodic
    SYM_EVEN,
    SYM_ODD
};

enum CYCLE_TYPE { // Multigrid cycle type
    CYCLE_V,      // down-up, no recursion
    CYCLE_W,      // down-up-down-up with full recursion
    CYCLE_F,      // down-up-down-up with recursion only on the first half
    CYCLE_NONE    // Debugging -- relax only
};

/* ----------------------------------------------------------------------
   The batched line solve: several independent line solves, interleaved

   One Thomas sweep is a serial dependency chain of length nz: the measured
   line solve is latency-bound inside that chain rather than bound by its
   memory traffic, so nb sweeps of independent lines interleaved lane-by-lane
   keep nb chains in flight where the single-line call keeps one.  The batch
   is a block of successive same-colour lines, never a whole colour: the
   measured curve is flat to B=8 and rises again past it, which is the block
   outgrowing the nearest cache rather than the interleaving stopping working.

   The band of each staged line is still built per line by line_band_build
   (the metric varies with x, so there is nothing to hoist); what batching
   changes is only where its storage lives -- a lane of this struct, rather
   than one band that is rebuilt under the single-line call.
   ---------------------------------------------------------------------- */

/* The widest lane block this interface can hold.  The width a given level
   actually uses is line_batch_width(nz), below. */
#define THOMAS_BATCH_LANES 8

/* A block of lines staged for one batched solve.  Lane b owns a block of
   3*nz doubles inside `block` (the layout line_band_set documents, which is
   what keeps that function's stale-values guard inside one lane) and nz
   doubles each of `f` and `u`.  `gen[b]` is the setup generation lane b's
   band holds, so the guard zeroes a lane once per setup rather than once per
   line: the band views themselves are per-build (see line_batch_lane), which
   keeps this struct free of any type the class cannot see yet. */
struct Line_batch {
    double *block;                   /* 3*nz*nb_max: one band-sized block per lane */
    double *f, *u;                   /* nz*nb_max each: residual in, solution out */
    long    gen[THOMAS_BATCH_LANES]; /* the setup generation each lane's band holds */
    int     ix[THOMAS_BATCH_LANES];  /* the x-line each lane last held */
    int     nb_max;                  /* lanes this solver stages into (0: never) */
    int     nz;                      /* z points per line */
};

/* Point a batch of nb_max lanes of nz-point lines at caller-owned storage:
   `block` holds 3*nz*nb_max doubles (3*nz per lane, in lane order) and `fu`
   holds 2*nz*nb_max (the residuals of all lanes, then their solutions).
   A freshly-set lane belongs to no setup, exactly as line_band_set leaves a
   freshly-set band. */
void line_batch_set(struct Line_batch &batch, double *block, double *fu, int nb_max, int nz);

/* The lane width a line of nz points is solved in: the measured batch-size
   curve (measured as a batch-size sweep) is flat from B=1 to B=8 for
   short lines and turns earlier for long ones, where the block itself stops
   fitting in the nearest cache.  A rule over the line length, because the
   line length is a property of the multigrid hierarchy a level is built on,
   not of the case file. */
int line_batch_width(int nz);

class MG_Solver {
  public:
    /* Constructor.  Gets x and z values of the numerical (perfectly orthogonal)
         grid, for construction of the 1D finite difference operators.  These arrays
         must also be for the full domain, since boundary points are special.

         If symmetry_problem is true, then xvals is extended by two to allow for
         a ghost-point type handling of the left and right boundaries */
    MG_Solver(blitz::Array<double, 1> xvals, blitz::Array<double, 1> zvals, SYM_TYPE symmetry_problem, MPI_Comm comm = MPI_COMM_WORLD);

    /* Set the specific form of symmetry.  This is not in the constructor because
         it might change between invocations on the same grid.  As an example, on
         a rectanguar grid with free-slip conditions, u (orthogonal to the boundary)
         is odd symmetry while w is even.  Pressure likewise has even symmetry */
    void set_x_symmetry(SYM_TYPE type);

    /* Note that after this point all arrays specified are split by processor */

    /* Problem setup, which takes the several coefficients.  With the exception
         of DC, these will probably not change between invocations.  For
         the multilevel algorithm, this can call downwards recursively with
         coarsened versions of these operators */
    void problem_setup(blitz::Array<double, 2> &Uxx, blitz::Array<double, 2> &Uzz, blitz::Array<double, 2> &Uxz, blitz::Array<double, 2> &Ux, blitz::Array<double, 2> &Uz);

    /* Setting the DC term.  This will change based on timestep, diffusivity,
         or spanwise wavenumber (in the case of a reduced 3D problem).  The
         general case has this varying, but I much prefer not to deal with that
         right now */
    void helmholtz_setup(double c);

    /* BC setup; so many terms are necessary because physical grid lines might not
         intersect the boundary at a right angle.  Indeed, they won't at the bottom
         even in the case of a hill.  To avoid a duplicate function spec, this
         fuction gets called twice with dim=0 for x and dim=1 for z*/
    void bc_setup(int dim, blitz::Array<double, 1> u_min, blitz::Array<double, 1> uz_min, blitz::Array<double, 1> ux_min, blitz::Array<double, 1> u_max, blitz::Array<double, 1> uz_max, blitz::Array<double, 1> ux_max);

    /* Perform a multigrid cycle, or at the coarse level do a direct solve
         via UMFPACK.  Takes as input the desired cycle type (F-cycle or V-cycle),
         and input residual f, and returns as output the (approximate) u.

         For the coarse level, this does the coarse solve regardless of cycle type.

         The bonus_in and bonus_out terms are required for indefinite problems, where
         u=constant is a solution to A*u=0.  In that case, we regularize the problem
         by requiring mean(u)==bonus_in, and in return the f applied to get that
         value is really f-bonus_out (on interior terms).  This seems a bit picky,
         but it's required to make gmres behave */
    /* This is the public facing implementation, which load-balances f and u
         and calls cycle_p */
    void cycle(CYCLE_TYPE cycle, blitz::Array<double, 2> &f, blitz::Array<double, 2> &u, double bonus_in, double &bonus_out, int pre = 2, int mid = 2, int post = 2);

    //   protected:
    /* Local array size */
    int local_size_x;
    int local_x_lbound, local_x_ubound;
    int coarse_x_lbound, coarse_x_ubound;

    int size_x, size_z;

    /* The red sweep's batched line solve: the
      interior lines of one colour are independent Jacobi solves of the same
      length, so they are staged into a block of lanes and solved by one
      interleaved recurrence.  Owned by this level -- deliberately not function
      static storage -- so that two MG_Solver levels (or two threads) can never
      share, let alone corrupt, one another's lanes.  z is never coarsened, so
      one allocation serves every sweep of this level.  Empty (nb_max==0) in a
      THOMAS_BATCH=false build, which is the single-line arm. */
    struct Line_batch   line_batch;
    std::vector<double> line_batch_block, line_batch_fu;

    /* Solve the lines staged in line_batch and write them back into u; this is
      also where the block that carries the local lbound sends it left. */
    void line_batch_flush(blitz::Array<double, 2> &u, int nb, int left_neighbour, MPI_Request &send_request_left);

    // Parallelization parameters
    int myrank, nproc;

    // Parameters for the coarse-grid solving
    bool coarse_symbolic_ok, // Whether the coarse grid symbolic factorizing is okay
        coarse_numeric_ok,   // Same with the numerical factor
        // And parameters to control when a symbolic factor is not okay
        any_dxz,    // Whether there's any nonzero dxz terms in the domain
        bc_tangent, // Whether there's any tangent derivatives in the BCs
        bc_normal;  // Same, normal derivatives

    // numeric and symbolic factors from UMFPACK
    void *numeric_factor, *symbolic_factor;

    // Arrays for the sparse representation of the coarse operator, see UMFPACK
    // documentation for the full information.  The short version is that
    // the entry A_double[A_cols[i]+j] is in column i, and row A_rows[A_cols[i]+j]
    int  sparse_size; // number of entries in sparse array
    int *A_cols,      // Array of column indices
        *A_rows;      // Array of row index entries
    double *A_double; // Array of entries in the sparse matrix

    // Builds the sparse operator, storing proper entries in the above arrays
    // and allocating if necessary
    void build_sparse_operator();

    // Performs consistency checks on whether a symbolic coarse grid factor
    // is OK based on current conditions and the coarse-grid boolean parameters
    // (see above)
    void check_bc_consistency();

    /* One dimensional FD operators */
    /* Interpretation:
         Op(i,j) is the operator for point i, referring to:
            j=0 -- left (minus) neighbour
            j=1 -- self
            j=2 -- right (plus) neighbour
      */
    blitz::Array<double, 2> Dx, Dxx, Dz, Dzz;

    /* Local coefficients for the 2D operator */
    blitz::Array<double, 2> uxx, uzz, uxz, ux, uz;

    /* Local arrays for load-balanced f and u arrays, since those
         may differ from input arrays */
    blitz::Array<double, 2> f_balance, u_balance;

    /* c, for the c*U term in the equation */
    double helm_parameter;

    /* Bumped every time this level's operator setup is (re)built -- see
      problem_setup, helmholtz_setup and bc_setup, which each increment it
      alongside the coarse_numeric_ok flag they already invalidate.  It is
      the generation stamp that line_band_build() checks: band values belong
      to the setup that produced them, and nothing may be relied across a
      generation change. */
    long setup_generation;

    /* Local bc coefficients */
    blitz::Array<double, 1> u_left, u_right, ux_left, ux_right, uz_left, uz_right, u_top, u_bot, ux_top, ux_bot, uz_top, uz_bot;

    /* True if the problem is indefinite, such that A*(constant) = 0.  
         See comments for cycle() for the implications */
    bool indefinite_problem;

    /* Symmetry type */
    SYM_TYPE symmetry_type;

    /* Perform one red-black relaxation pass */
    void do_redblack(blitz::Array<double, 2> &f, blitz::Array<double, 2> &u);

    /* Apply the operator "forwards" */
    void apply_operator(blitz::Array<double, 2> &u, blitz::Array<double, 2> &f);

    /* Coarse u, f arrays for multilevel problem */
    blitz::Array<double, 2> coarse_u, coarse_f;

    /* Local, nonbalanced coarsened arrays.  Since the array gets roughly halved,
         we can either rebalance and then coarsen or coarsen and then rebalance.
         This takes the latter approach */
    blitz::Array<double, 2> local_coarse_2d;
    blitz::Array<double, 1> local_coarse_1d;

    /* If we coarsen an even number of points (odd number of
         intervals), then the "include every other point" strategy
         doesn't work (given that we keep the boundaries).  We have
         to thus pick an interval to keep between coarsening levels;
         keeping the largest interval on the fine grid seems to be
         an excellent choice */
    int  kept_interval;
    bool coarsest_level;

    /* (semi)-coarsen a variable on a 2D grid.  The extra 
         boolean (even) tells the coarsening operator to 
         treat the problem as having even symmetry (if any) 
         rather than whatever has been specified.  This is 
         useful for the coefficients and boundary conditions, 
         since they'll never have odd symmetry */
    /* This operator does not rebalance -- it will simply write
         into local_coarse_2d (1d for line version) */
    void coarsen_grid(blitz::Array<double, 2> &q, bool even = false);
    /* And do the same to a line.  */
    void coarsen_line(blitz::Array<double, 1> &q, bool even = false);

    /* Interpolate (from local_coarse_2d) to full-sized array */
    void interpolate_grid(blitz::Array<double, 2> &q_fine);
    // Interpolate line commented out -- I don't see a need for it
    //      void interpolate_line(blitz::Array<double,1> & q_coarse,
    //            blitz::Array<double,1> & q_fine, bool even=false);
    /* Local cycle, which implements the public function but assumes all relevant
         arrays are properly load-balanced */
    void _cycle(CYCLE_TYPE cycle, blitz::Array<double, 2> &f, blitz::Array<double, 2> &u, double bonus_in, double &bonus_out, int pre, int mid, int post);

    /* Rebalances an array; an array that is orignally spit in orig is
         reassigned to balance, such that the global (compressed) array
         is exactly the same */
    // Arrays to hold origin, destination bounds for array rebalancing (coarse)
    enum BalanceGroup {
        FFine   = 0,
        UFine   = 1,
        FCoarse = 2,
        UCoarse = 3
    };
    int *to_lbounds[4], *to_ubounds[4], *tb_lbounds[4], *tb_ubounds[4];
    bool rebalance_setup[4];
    void rebalance_array(blitz::Array<double, 2> &orig, blitz::Array<double, 2> &balance, MPI_Comm c, BalanceGroup btype);
    void rebalance_line(blitz::Array<double, 1> &o, blitz::Array<double, 1> &b, MPI_Comm c, BalanceGroup btype);

    MPI_Comm   my_comm;
    MPI_Comm   coarse_comm;   // Communicator for coarse grid
    MG_Solver *coarse_solver; // Coarse solver
};

/* Build a 1D finite-difference operator for arbitrary x-grid, see mkfd.m
   as reference in MATLAB */
void get_fd_operator(blitz::Array<double, 1> &x, blitz::Array<double, 2> &Dx, blitz::Array<double, 2> &Dxx);

/* Get the local splitting of N points in the multigrid context.  This might (will)
   differ from the splitting in spectral context, because we want a minimum of 2 lines
   per processor, and preferably each processor will have an even number of lines
   to minimize communication.  e.g.
   4 4 4 4 6 is preferable to
   4 4 4 5 5, since the latter will mix up red/black identities
   (RBRB-RBRB-RBRB-RBRB-RBRBRB vs.
    RBRB-RBRB-RBRB-RBRBR-BRBRB -- the latter will involve a double-stall on 
                                  processor 4 for boundary data */
void get_local_split(int extent, int rank, int nproc, int &l_lbound, int &l_ubound);

/* ----------------------------------------------------------------------
   The line-solve band: the tridiagonal storage one line solve needs

   What a line solve needs is a tridiagonal band: three diagonals of nz
   doubles.  It used to be assembled out of blitz expression templates into
   scratch local to line_solve; it is now assembled by line_band_build(),
   whose arguments are only double, int and long, so a caller that owns no
   blitz arrays at all can build exactly the same band.  The 1D solver's line
   solves (poisson_1d and the GMRES kernel behind it, gmres_1d_solver.cpp)
   are the designed second consumer: what such a caller passes is

     per line   -- A,B,C, the three coefficient rows of
                  A*u_zz + B*u_z + C*u (= the same operator its cheb_d2
                  kernel applies, read off its own FD operator arrays), and
                  the four boundary scalars a0,b0 (bottom) a1,b1 (top) that
                  poisson_1d already takes;
     per level  -- a Z_operator: the three columns of the first- and
                  second-derivative stencils.  These are per-level *arrays*,
                  not scalars: on a mapped grid the metric varies with x, so
                  a caller must re-read them for every line and may not
                  hoist them out of the line loop;
     per setup  -- the setup generation, described below.

   Storage: one caller-owned block of 3*nz contiguous doubles, handed in as
   a Line_band view (line_band_set).  The build allocates nothing, so a
   batched caller can lay out as many bands as it likes -- one block of
   3*nz*nb, one band per line, whatever suits its access pattern -- and no
   part of this interface changes.  The 2D caller (do_redblack) owns one
   band per sweep and rebuilds it for every line of that sweep.

   Setup-generation guard: band values belong to the operator setup that
   built them.  line_band_build() re-adopts (zeroes, then re-stamps)
   storage whose generation differs from the level's current one before it
   writes anything, so a band can never carry a value from a superseded
   setup into a solve.  Note what this is *not*: this branch caches nothing,
   the build recomputes every entry on every call, and the guard costs one
   integer compare per line solve (the band build is only ~8% of a
   multigrid cycle, which is why this is an invalidation contract rather than a
   cache).  It is here so that any reuse a later caller adds is
   invalidation-checked by construction, with no hashing, no cross-line key
   and nothing at all carried across timesteps.
   ---------------------------------------------------------------------- */

/* The per-level z-direction stencil, as plain column pointers.  dz[k] and
   dzz[k] point at the k-th entry (0 = minus neighbour, 1 = self,
   2 = plus neighbour) of the operator local to z=0; successive z-points are
   `stride` doubles apart, so nothing here assumes any particular storage
   order. */
struct Z_operator {
    const double *dz[3], *dzz[3];
    int           stride;
};

/* A tridiagonal band: three diagonals of nz doubles, viewed out of a
   caller-owned block of 3*nz of them. */
struct Line_band {
    double *an, *bn, *cn; /* sub-, main-, super-diagonal, nz entries each */
    int     nz;           /* entries per diagonal (the line's length) */
    long    generation;   /* the setup generation these values belong to */
};

/* Point a band at a caller-owned block of 3*nz doubles.  The block is
   neither read nor owned here; a freshly-set band belongs to no setup. */
void line_band_set(struct Line_band &band, double *block, int nz);

/* Point a z-operator at the stencil columns of two (nz,3) arrays.  This is
   the one blitz-aware piece of the boundary; a caller without blitz fills
   the same struct from its own storage. */
void z_operator_set(blitz::Array<double, 2> &Dz, blitz::Array<double, 2> &Dzz, struct Z_operator &zop);

/* Build the band of one line solve (see the notes above).  Writes every
   entry of an[0..nz-1], bn[0..nz-1], cn[0..nz-1]. */
void line_band_build(struct Line_band &band, const struct Z_operator &zop, long setup_generation, const double *A, const double *B, const double *C, double a0, double a1, double b0, double b1);

/* Perform a line solve, which is the 1d problem:
   A(z)*u_zz + B(z)*u_z + C(z) = f(z), with
   a0*u+b0*u_z = f0 @ z=0 (bottom)
   a1*u+b1*u_z = f1 @ z=1 (top)

   Unlike the 1D gmres controller, we're not going to go to
   great lengths to preserve a factored operator.  Firstly,
   there's no chance of an indefinite 1D problem here, and
   secondly most of the time we're going to be doing a -lot-
   of line solves, and the only way to cache things would
   be to store all of those operators.  Bad for memory.

   The band is built into `band`, caller-owned storage of this level's
   length, under the level's current setup generation `setup_generation`. */
void line_solve(blitz::Array<double, 1> &u, blitz::Array<double, 1> &f, blitz::Array<double, 1> &A, blitz::Array<double, 1> &B, blitz::Array<double, 1> &C, double a0, double a1, double b0, double b1, blitz::Array<double, 2> &Dz, blitz::Array<double, 2> &Dzz, struct Line_band &band, long setup_generation);

/* ----------------------------------------------------------------------
   The batched line solve -- the band-side of it (see the notes above)
   ---------------------------------------------------------------------- */

/* Adopt lane `lane`'s band for one build: a per-line view of the lane's own
   3*nz block, stamped with the generation that lane was last built under. */
void line_batch_lane(struct Line_batch &batch, int lane, struct Line_band &band);

/* Solve nb staged lines: the same recurrence line_solve runs on one line, with
   the lane loop innermost.  Lane b's band must have been built (per line, by
   the caller) into its block, and its residual copied into batch.f + b*nz;
   the solution comes back in batch.u + b*nz.  nb must not exceed nb_max. */
void line_solve_batch(int nb, struct Line_batch &batch);

#endif
