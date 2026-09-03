set (cases_TARGETS
  brydon_test
  derivatives
  dipole
  gravity_current
  #jmd_test  # -- Broken.
  kh_billow
  lineos_test
  map_iwave
  mode1_mode2
  mode2_ISW
  nleos_test
  qsp
  quadeos_test
  #salt_temp  # -- Broken.
  wave_reader
)

foreach( target ${cases_TARGETS})
    set (casefilename "cases/${target}/${target}.cpp")

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generate_${target}.src.sh "#!/bin/bash
        set -e
        echo \"
        #define CASE_NAME \\\"${target}\\\"
        #include <stdio.h>
        #include <string.h>

        /* Write the source code of the case to spinscase.cpp */
        void WriteCaseFileSource(void)
        {
            const char casefilesource[] = {`xxd -i < ${CMAKE_CURRENT_SOURCE_DIR}/src/cases/${target}/${target}.cpp`, 0x00};
            const char casefilename[] = \\\"{CMAKE_CURRENT_SOURCE_DIR}/src/cases/${target}/${target}.cpp\\\";
            char* filename;
            if ( strcmp(casefilename, \\\"src/cases/derivatives/derivatives.cpp\\\") == 0 ) {
                filename = \\\"derivatives.cpp\\\";
            } else {
                filename = \\\"spinscase.cpp\\\";
            }
            FILE* fid;
            fid = fopen(filename,\\\"w\\\");
            fprintf(fid,\\\"/* %s */\\\\n%s\\\", filename, casefilesource);
            fclose(fid);
        }
        \" > ${CMAKE_CURRENT_BINARY_DIR}/src/cases/${target}/${target}.src.c
        "
    )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cases/${target}/spins.conf
        ${CMAKE_CURRENT_BINARY_DIR}/src/cases/${target}/spins.conf
        COPYONLY
    )

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/src/cases/${target}/${target}.src.c
        COMMAND /bin/bash "${CMAKE_CURRENT_BINARY_DIR}/generate_${target}.src.sh"
        DEPENDS
            ${PROJECT_SOURCE_DIR}/src/cases/${target}/${target}.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Ran pre-build step for ${target}: Embed SPINS casefile source for write upon execution."
        VERBATIM
    )

    add_executable (${target}.x "${CMAKE_CURRENT_SOURCE_DIR}/src/cases/${target}/${target}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/src/cases/${target}/${target}.src.c")

    # Pass versioning info to C pre-processor
    target_compile_definitions (${target}.x PUBLIC MAJOR_VERSION=${MAJOR_VERSION} MINOR_VERSION=${MINOR_VERSION} PATCH_VERSION=${PATCH_VERSION})
    set_target_properties (${target}.x PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/src/cases/${target}")
    target_include_directories (${target}.x PUBLIC ${base_include_paths})
    target_link_directories (${target}.x PUBLIC ${base_library_paths})
    target_link_libraries(${target}.x PRIVATE spins_science spins_basecase spins_base)
    target_link_libraries (${target}.x PUBLIC ${MPI_CXX} ${MPI} ${BOOST_PROGRAM_OPTIONS}
                                              ${UMFPACK} ${LAPACK} ${BLITZ} ${FFTW3}
    )
endforeach()
