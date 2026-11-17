# Makefile to build SPINS, generalized for multiple systemd

# The "system.mk" file in the current directory will contain
# system-specific make variables, most notably the
# C/C++ compiler/linker, 
include system.mk
include VERSION


# Compile with debugging flags
DEBUG?=false
# Compile with optimizations
OPTIM?=true
# Compile with extra optimizations that may require significantly
# longer for compilation
SLOW_OPTIM?=false

# Compile with instrumentation support for timing
TIMINGS?=false

# Plan FFTWs with FFTW_ESTIMATE instead of FFTW_MEASURE: deterministic
# (heuristic) plan selection, so a run's results are reproducible
# bit-for-bit and no plan-timing loop runs at startup.  Off by default.
FFTW_ESTIMATE?=false

# Batch the 2D multigrid's red line solves: the
# strictly interior lines of a colour are staged into a block of lanes and
# solved by one interleaved Thomas recurrence, which keeps one dependency
# chain per lane in flight where the single-line solve keeps one in total.
# On by default; `THOMAS_BATCH=false` builds the single-line arm, which is the
# comparison arm the correctness battery measures against (and is byte-identical
# to the tree without this change).  GOTCHA: make does not track flags, so
# flipping this needs the objects deleted first (see system.mk).
THOMAS_BATCH?=true

# If MPICXX isn't separately defined, then set it to be the same
# as CXX
ifeq ($(strip $(MPICXX)),)
   MPICXX:=$(CXX)
endif

# Flag to pass version info to code
VERSION:= -DMAJOR_VERSION=${MAJOR_VERSION} -DMINOR_VERSION=${MINOR_VERSION} -DPATCH_VERSION=${PATCH_VERSION}

# Assemble the CFLAGS
CFLAGS:=$(SYSTEM_CFLAGS) $(MPI_CFLAGS) $(SYSTEM_CXXFLAGS)
LDFLAGS:=$(SYSTEM_LDFLAGS)
ifeq ($(DEBUG),true)
   CFLAGS:=$(CFLAGS) $(DEBUG_CFLAGS)
   LDFLAGS:=$(LDFLAGS) $(DEBUG_LDFLAGS)
endif
ifeq ($(OPTIM),true)
   CFLAGS:=$(CFLAGS) $(OPTIM_CFLAGS)
   LDFLAGS:=$(LDFLAGS) $(OPTIM_LDFLAGS)
   ifeq ($(SLOW_OPTIM),true)
      CFLAGS:=$(CFLAGS) $(EXTRA_OPTIM_CFLAGS)
      LDFLAGS:=$(LDFLAGS) $(EXTRA_OPTIM_LDFLAGS)
   endif
endif

ifeq ($(TIMINGS),true)
   CFLAGS:=$(CFLAGS) -DTIMING_ENABLE
endif

ifeq ($(FFTW_ESTIMATE),true)
   CFLAGS:=$(CFLAGS) -DSPINS_FFTW_ESTIMATE
endif

ifeq ($(THOMAS_BATCH),false)
   CFLAGS:=$(CFLAGS) -DTHOMAS_BATCH_SERIAL
endif


INCLUDE_DIRS := -I../include $(MPI_INCDIR) $(LAPACK_INCDIR) $(BLITZ_INCDIR) $(FFTW_INCDIR) $(UMF_INCDIR)
CFLAGS := $(CFLAGS) $(INCLUDE_DIRS)

LIB_DIRS := -L../lib $(MPI_LIBDIR) $(LAPACK_LIBDIR) $(BLITZ_LIBDIR) $(FFTW_LIBDIR) $(UMF_LIBDIR)
LDLIBS := -lfftw3 $(MPI_LIB) -lumfpack -lamd -lboost_program_options $(LAPACK_LIB) -lblitz -lm -lstdc++ 
LDFLAGS := $(LDFLAGS) $(LIB_DIRS) 

.PHONY: all 
all: tests/test_deriv_x tests/test_write_x tests/test_esolve_x tests/test_heat_x tests/test_ode_x tests/test_ns_x

.PHONY: clean
clean:
	rm -f *.o tests/*.o cases/*.o cases/*.src.? tests/*.src.? Science/*.o BaseCase/*.o cases/*/*.src.? cases/*/*.o

##
## Short-names for source files
##

objfiles: $(shell ls *.cpp | sed -e 's/cpp/o/g')

# SPINS files
SPINS_BASE := TArray.o T_util.o Parformer.o ESolver.o Timestep.o NSIntegrator.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o Options.o BaseCase.o Sorter.o timing.o

BASECASE_CPPS := $(wildcard BaseCase/*.cpp)
BASECASE_OBJS := $(addprefix BaseCase/,$(notdir $(BASECASE_CPPS:.cpp=.o)))

SCIENCE_CPPS := $(wildcard Science/*.cpp)
SCIENCE_OBJS := $(addprefix Science/,$(notdir $(SCIENCE_CPPS:.cpp=.o)))

##
## Recipes
##

# Whether an object sees the headers must be a property of the build, not of
# which rule matched: make prefers the most specific matching rule (shortest
# stem) over the general `%.o: %.cpp *.hpp` at the bottom of this file, so any
# rule that carries its OWN recipe has to name the headers too. An object built
# by a rule that omits them survives a header change, and a promotion that
# moves a class layout then links objects of two eras (measured at the change
# that removed TArray<T>'s plan-cache member: a stale tests/test_cheby.o
# re-linked into the new library computed its test and aborted at teardown in
# glibc).
#
# NSIntegrator.o carries NO recipe, so make takes this object's recipe — and
# with it the general rule's `*.hpp` prerequisites — from that rule; it is not
# header-blind and needs no prerequisite of its own.
NSIntegrator.o: NSIntegrator.cpp NSIntegrator_impl.cc

# The legacy `tests/test%.x` rule cannot build ANY test in this directory:
# it is a pattern stem with a DOT suffix (tests/test<stem>.x from
# tests/test<stem>.cpp), but every real source is tests/test_<name>.cpp and
# every wanted binary is tests/test_<name>_x (underscore) — no stem
# satisfies both (review-verified 2026-10-05: a proper single-% pattern
# target matches fine on GNU make 4.3, so this is a naming-scheme
# mismatch, not a make bug). Explicit static pattern pairs carry the same
# recipe over the real test sources (identical behaviour).
TESTS := $(patsubst tests/test_%.cpp,tests/test_%,$(wildcard tests/test_*.cpp))

$(addsuffix _x,$(TESTS)): %_x: %.o %.src.o ${SCIENCE_OBJS} ${BASECASE_OBJS} ${SPINS_BASE}
	$(LD) ${VERSION} $(LDFLAGS) -o $@ $^ $(LDLIBS)

tests/test%.o: tests/test%.cpp *.hpp
	$(MPICXX) ${VERSION} $(CFLAGS) -o $@ -c $<

# Restored legacy case rules (unchanged from the original Makefile, except the
# header prerequisites: see the note above the NSIntegrator.o rule).
cases/%.o: cases/%.cpp NSIntegrator_impl.cc NSIntegrator.hpp *.hpp
	$(MPICXX) ${VERSION} $(CFLAGS) -o $@ -c  $<

cases/%.x: cases/%.o cases/%.src.o ${SCIENCE_OBJS} ${BASECASE_OBJS} ${SPINS_BASE}
	$(LD) ${VERSION} $(LDFLAGS) -o $@ $^ $(LDLIBS)

cases/%_x: cases/%.o cases/%.src.o ${SCIENCE_OBJS} ${BASECASE_OBJS} ${SPINS_BASE}
	$(LD) ${VERSION} $(LDFLAGS) -o $@ $^ $(LDLIBS)

tests/test%.src.c: tests/test%.cpp CaseFileSource.c
	echo "const char casefilesource[] = {`xxd -i < $<`, 0x00};" > $@
	echo "const char casefilename[] = \"$<\";" >> $@
	cat CaseFileSource.c >> $@

cases/%.src.c: cases/%.cpp CaseFileSource.c
	echo "const char casefilesource[] = {`xxd -i < $<`, 0x00};" > $@
	echo "const char casefilename[] = \"$<\";" >> $@
	cat CaseFileSource.c >> $@

# C compile rule (the generated tests/%.src.c / cases/%.src.c case-file
# embedders) — absent from the legacy Makefile; without it the generic
# tests/test%.x rule has no way to build its tests/test%.src.o. Those
# embedders include no project header today, but the rule carries `*.hpp`
# anyway: whether an object sees a header stays a property of the build, and
# the recompile it costs is one small C file, once per header movement.
%.o: %.c *.hpp
	$(CC) ${VERSION} $(CFLAGS) -o $@ -c $<

%.o: %.cpp	*.hpp
	$(MPICXX) ${VERSION} $(CFLAGS) -o $@ -c $< 

print-%  : ; @echo $* = $($*)

.SECONDARY:
