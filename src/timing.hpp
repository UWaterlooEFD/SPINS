/* timing.hpp -- header file for self-timing code */

#ifndef TIMING_HPP
#define TIMING_HPP 1 // Include this header file at most once

/* The timing infrastructure works with a stack-based model.

   At portions of the code to be instrumented, call timing_push("name")
   with some (short but) memortable name for the code segment; after the
   end of the section call timing_pop().  This underlying code will build
   the corresponding call-time-tree and measure (via MPI_Wtime) the time
   between the timing_push and the corresponding timing_pop.

   This model does require that timing be called in a predictable way, such
   that timing_pop is called before any possible exit from the code path.*/

#ifndef TIMING_ENABLE // If timing code is not enabled, define the respective functions as no-ops

    #define timing_push(x)
    #define timing_pop()

#else
    #include <mpi.h> // for the MPI_Comm default argument below

void timing_push(const char *name);
void timing_pop();

/* Rank identity, captured by the run while MPI is alive.  The report fires
   after MPI_Finalize (every case main is `do_run(...); MPI_Finalize();
   return 0;` with the evolver a main-scope object), where no rank query may
   be made -- so the per-rank file names come from here, never from a query
   at the emit site. */
void timing_set_rank(MPI_Comm comm = MPI_COMM_WORLD);

/* When the environment names a prefix (SPINS_TIMING_RANK_FILES), write this
   rank's call tree to <prefix>.rank<NNN>.txt in the workspace, in the same
   format timing_stack_report prints.  Deliberately NOT guarded by master():
   every rank writes its own file.  No-op, and no I/O of any kind, when the
   variable is unset -- which is what keeps the cost of this file the
   library's normal zero. */
void timing_rank_files();
#endif

void timing_stack_report();

#endif
