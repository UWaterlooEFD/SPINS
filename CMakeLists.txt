cmake_minimum_required(VERSION 3.15)
project(SPINS)

set (CMAKE_CONFIGURATION_TYPES Release Debug)
set (CMAKE_BUILD_TYPE Release)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 -DBZ_DEBUG")

set (MPI_ROOT "/usr/lib/x86_64-linux-gnu/openmpi/")
set (BLITZ_ROOT "/usr/local")
set (SUITESPARSE_INCLUDE_ROOT "/usr/include/suitesparse/")

set (base_include_paths "${MPI_ROOT}/include" "${BLITZ_ROOT}" "${SUITESPARSE_INCLUDE_ROOT}")
set (base_library_paths "${MPI_ROOT}/lib ${BLITZ_ROOT}/lib")

# Hacky method of extracting the contents of the src/VERSION file into variables.
execute_process(COMMAND bash -c ". ${CMAKE_CURRENT_SOURCE_DIR}/src/VERSION && echo -n $MAJOR_VERSION" OUTPUT_VARIABLE MAJOR_VERSION)
execute_process(COMMAND bash -c ". ${CMAKE_CURRENT_SOURCE_DIR}/src/VERSION && echo -n $MINOR_VERSION" OUTPUT_VARIABLE MINOR_VERSION)
execute_process(COMMAND bash -c ". ${CMAKE_CURRENT_SOURCE_DIR}/src/VERSION && echo -n $PATCH_VERSION" OUTPUT_VARIABLE PATCH_VERSION)

message("-- SPINS CMAKE BUILD --")
message("-- MAJOR VERSION: ${MAJOR_VERSION}")
message("-- MINOR VERSION: ${MINOR_VERSION}")
message("-- PATCH VERSION: ${PATCH_VERSION}")

# Resolve dependency locations.
find_library(BLITZ NAMES "blitz" PATHS "${BLITZ_ROOT}/lib" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/lib64")
find_library(BOOST_PROGRAM_OPTIONS NAMES "boost_program_options" PATHS "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/lib64")
find_library(FFTW3 NAMES "fftw3" PATHS "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/lib64")
find_library(LAPACK NAMES "lapack" PATHS "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/lib64")
find_library(MPI NAMES "mpi" PATHS "${MPI_ROOT}/lib" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/lib64")
find_library(MPI_CXX NAMES "mpi_cxx" PATHS "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/lib64")
find_library(UMFPACK NAMES "umfpack" PATHS "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/lib64")

message("-- Found ${BLITZ}")
message("-- Found ${BOOST_PROGRAM_OPTIONS}")
message("-- Found ${FFTW3}")
message("-- Found ${LAPACK}")
message("-- Found ${MPI}")
message("-- Found ${MPI_CXX}")
message("-- Found ${MPI}")
message("-- Found ${UMFPACK}")

set (spins_base_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/TArray.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/T_util.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Parformer.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/ESolver.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Timestep.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/NSIntegrator.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Splits.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Par_util.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Split_reader.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/gmres.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/gmres_1d_solver.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/gmres_2d_solver.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/grad.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/multigrid.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Options.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/src/Sorter.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/timing.cpp"
)

set (science_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/get_quads.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_quadweights.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/top_stress_x.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/top_stress_y.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/bottom_stress_x.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/bottom_stress_y.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/read_2d_restart.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/find_expansion.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/swap_trig.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_vort_z.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_vort_x.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_vort_y.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/dissipation.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/bottom_slope.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_baroclinic_vort_x.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_baroclinic_vort_y.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_baroclinic_vort.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/enstrophy_density.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/enstrophy_stretch_production.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/vortex_stretch_x.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/vortex_stretch_y.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/vortex_stretch_z.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_lambda2.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/R_invt.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/Q_invt.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/QSPCount.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_background_PE.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_background_PE.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/compute_BPE_from_internal.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/lineos.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/nleos.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/quadeos.cpp"
)

set (basecase_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/automatic_grid.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/check_and_dump.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/check_timestep.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/parse_boundary_conditions.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/stresses_bottom.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/stresses_top.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/successful_dump.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/write_chain.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/write_diagnostics.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/src/BaseCase/write_plot_times.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Science/read_2d_slice.cpp"
)

add_library(spins_base STATIC ${spins_base_sources})
target_include_directories (spins_base PUBLIC ${base_include_paths})
target_compile_definitions (spins_base PUBLIC MAJOR_VERSION=${MAJOR_VERSION} MINOR_VERSION=${MINOR_VERSION} PATCH_VERSION=${PATCH_VERSION})

add_library(spins_science STATIC ${science_sources})
target_include_directories (spins_science PUBLIC ${base_include_paths})

add_library(spins_basecase STATIC ${basecase_sources})
target_include_directories (spins_basecase PUBLIC ${base_include_paths})
target_compile_definitions (spins_basecase PUBLIC MAJOR_VERSION=${MAJOR_VERSION} MINOR_VERSION=${MINOR_VERSION} PATCH_VERSION=${PATCH_VERSION})
target_link_libraries(spins_basecase PRIVATE spins_base spins_science)


add_executable(nonhydro_sw.x "${CMAKE_CURRENT_SOURCE_DIR}/src/nonhydro_sw.cpp")
target_include_directories (nonhydro_sw.x PUBLIC ${base_include_paths})
target_compile_definitions (nonhydro_sw.x PUBLIC MAJOR_VERSION=${MAJOR_VERSION} MINOR_VERSION=${MINOR_VERSION} PATCH_VERSION=${PATCH_VERSION})
target_link_libraries(nonhydro_sw.x PRIVATE spins_base)
target_link_libraries(nonhydro_sw.x PUBLIC ${MPI_CXX} ${MPI} ${BOOST_PROGRAM_OPTIONS}
${UMFPACK} ${LAPACK} ${BLITZ} ${FFTW3})


# Include test cases (some seem to be broken, incompatible with the current SPINS API.)
include ("src/tests/tests.cmake")

# Include real cases.
include ("src/cases/cases.cmake")