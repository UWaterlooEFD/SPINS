/* test_np_guard.cpp -- does an ill-posed rank count fail the run, or hang it?
   There was no guard on a rank count exceeding the extent of an axis, and
   this driver is the two shapes that defect takes.

   The two defects share one contract: an ill-posed setup must end the run with
   a meaningful error, never a silent hang: that is the contract for a
   failed setup in this model, and it is what both arms assert.
   Two arms, one driver; the arm is argv[1], so a caller can ask for
   either one.

   part 1  arm "split"  --  MG_Solver's level-0 x-split (get_local_split) hands
       out lines in PAIRS, so once np > size_x/2 the last ranks own no line at
       all.  A rank that owns nothing never runs its red sweep, so the halo
       Isend its neighbour waits on is never issued and the first sweep
       deadlocks.  The grid here is SIZE_X = 8 lines, so the arm is admissible
       up to np = 4 and ill-posed from np = 5.  Admissible -> prints
           NP_GUARD split = passed
       and exits 0; ill-posed -> the guard's FATAL line and a nonzero exit.
       Measured before the guard existed, the ill-posed arm takes the
       ADMISSIBLE branch -- MG_Solver tolerates an ownerless rank and the driver
       returns 0 -- so what this arm pins is the REFUSAL (message plus nonzero
       exit), not a hang: the deadlock is in the red sweep, which
       this driver deliberately never runs.

   part 2  arm "plan" [sx sy sz dim]  --  every fftw_plan_many_* call takes
       howmany = num_transforms, which TArray computes as
       numElements()/extent(dimension) (and as an extent, on the plane-stride
       branch).  Both are 0 when a process owns no lines of the axis, which
       FFTW's manual forbids (howmany >= 1) even though this vendor's library
       measures as accepting it -- see the comment on check_num_transforms in
       Plan_holder.hpp.  The arm asks for a transform of an array that has no lines at all -- the
       shape a rank owns when an axis has run out, with the MPI left out.  The
       geometry is printed first because which of TArray::transform's three
       stride branches such an array takes is a blitz stride question, not an
       obvious one, and the arm's outcome depends on it.

   Convention per src/tests/test_cheby.cpp: MPI_Init, machine-readable
   NP_GUARD lines for the caller to assert on. */
#include <mpi.h>
#include <blitz/array.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../TArray.hpp"
#include "../multigrid.hpp"

using namespace blitz;
using namespace TArrayn;

#define SIZE_X 8
#define SIZE_Z 16

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int myrank, numproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    const char *arm = (argc > 1) ? argv[1] : "split";

    if (!strcmp(arm, "split")) {
        /* The full-domain grids MG_Solver wants (x includes the boundaries for
           SYM_NONE, exactly as gmres_2d_solver::build_precond hands it). */
        firstIndex       ii;
        Array<double, 1> xvals(SIZE_X), zvals(SIZE_Z);
        xvals = -cos(ii * M_PI / (SIZE_X - 1));
        zvals = -cos(ii * M_PI / (SIZE_Z - 1));
        /* The guard is evaluated before this line is entered: if the split is
           ill-posed nothing below runs, and if it is admissible the object is
           built, sized and recursed as in the model. */
        MG_Solver miggy(xvals, zvals, SYM_NONE, MPI_COMM_WORLD);
        if (myrank == 0) fprintf(stdout, "NP_GUARD split = passed\n");
    } else if (!strcmp(arm, "plan")) {
        int            sx = (argc > 2) ? atoi(argv[2]) : 4;
        int            sy = (argc > 3) ? atoi(argv[3]) : 0;
        int            sz = (argc > 4) ? atoi(argv[4]) : 16;
        int            td = (argc > 5) ? atoi(argv[5]) : 0;
        TArray<double> source(sx, sy, sz), dest(sx, sy, sz);
        /* The count the plan builders are about to be handed, printed BEFORE
           the transform so it is on the record even when the guard fires. */
        int extent = (int)source.extent(td == 0 ? blitz::firstDim : (td == 1 ? blitz::secondDim : blitz::thirdDim));
        if (myrank == 0)
            fprintf(stdout, "NP_GUARD plan_howmany = %d\nNP_GUARD geom = (%d,%d,%d) dim %d stride (%d,%d,%d) ordering (%d,%d,%d) numElements %d\n", extent > 0 ? (int)source.numElements() / extent : -1, sx, sy, sz, td, (int)source.stride(blitz::firstDim),
                    (int)source.stride(blitz::secondDim), (int)source.stride(blitz::thirdDim), (int)source.ordering(0), (int)source.ordering(1), (int)source.ordering(2), (int)source.numElements());
        if (td == 0) source.transform(dest, Dimension(blitz::firstDim), DCT1);
        else if (td == 1) source.transform(dest, Dimension(blitz::secondDim), DCT1);
        else source.transform(dest, Dimension(blitz::thirdDim), DCT1);
        /* Printed only when the transform came back.  For a zero-count geometry
           that means the plan was built and executed, which is the defect; the
           caller asserts this line is ABSENT there, and present (with a
           nonzero howmany) for an admissible geometry. */
        if (myrank == 0) fprintf(stdout, "NP_GUARD plan = returned\n");
    } else {
        if (myrank == 0) fprintf(stderr, "unknown arm %s (want split or plan)\n", arm);
        MPI_Finalize();
        return 2;
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
