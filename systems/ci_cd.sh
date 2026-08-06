#!/bin/bash

### This is a system-specific configuration file designed for
### the ci_cd docker container

# System-specific settings for bow.math.private

CC=gcc
CXX=g++
LD=mpicxx

# System-specific compiler flags
SYSTEM_CFLAGS=

SYSTEM_LDFLAGS=

# Compiler flags for debugging
DEBUG_CFLAGS="-g -DBZ_DEBUG"
DEBUG_LDFLAGS=

# Compiler flags for optimization
OPTIM_CFLAGS="-O3"
OPTIM_LDFLAGS=$OPTIM_CFLAGS

# Compiler flags for extra optimization, such as -ip -ipo on icc
EXTRA_OPTIM_CFLAGS=
EXTRA_OPTIM_LDFLAGS=$EXTRA_OPTIM_CFLAGS

# Library names/locations/flags for MPI-compilation.  This will
# probably not be necessary on systems with a working mpicc alias
# When using SGI/HPE MPT, after loading the module, mpicxx is in search rules
# and it sets the rest of the references it needs.
MPICXX=mpicxx
#MPI_CFLAGS=
#MPI_LIBDIR="-L${MPI_LIB}"
MPI_LIB="-lmpi_cxx -lmpi"
#MPI_INCDIR="-I${MPI_INCLUDE}"

# Library names/locations for LAPACK
LAPACK_LIB="-lopenblas"
LAPACK_LIBDIR=
LAPACK_INCDIR=

# Library locations for blitz; leave blank to use system-installed
# or compiled-with-this-package version
BLITZ_LIBDIR=
BLITZ_INCDIR=

# Library locations for fftw
FFTW_LIBDIR=
FFTW_INCDIR=

# Library locations for UMFPACK
UMF_INCDIR="-I/usr/include/suitesparse"
UMF_LIBDIR=

# Location/library for BLAS
BLAS_LIB=
BLAS_LIBDIR=
BLAS_INCDIR=

