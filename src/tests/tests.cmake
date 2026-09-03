set (tests_TARGETS
#  SplitTest  # -- Broken.
  test_cheby
#  test_chebysolve  # -- Broken.
#  test_csolve2d  # -- Broken.
  test_deriv  
#  test_esolve  # -- Broken.
  test_gmres
#  test_heat  # -- Broken.
  test_mg
#  test_min_ns  # -- Broken.
  test_ode   
#  test_par  # -- Broken.
#  test_pesolve  # -- Broken.
#  test_pheat  # -- Broken.
#  test_write  # -- Broken.
)

foreach( target ${tests_TARGETS})
    set (testfilename "tests/${target}.cpp")

    add_executable (${target}.x "src/tests/${target}.cpp")

    # Pass versioning info to C pre-processor
    target_compile_definitions (${target}.x PUBLIC MAJOR_VERSION=${MAJOR_VERSION} MINOR_VERSION=${MINOR_VERSION} PATCH_VERSION=${PATCH_VERSION})
    set_target_properties (${target}.x PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/src/tests")
    target_include_directories (${target}.x PUBLIC ${base_include_paths})
    target_link_directories (${target}.x PUBLIC ${base_library_paths})
    target_link_libraries(${target}.x PRIVATE spins_base spins_science spins_basecase)
    target_link_libraries (${target}.x PUBLIC ${MPI_CXX} ${MPI} ${BOOST_PROGRAM_OPTIONS}
                                              ${UMFPACK} ${LAPACK} ${BLITZ} ${FFTW3}
    )
endforeach()