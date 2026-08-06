cmake_minimum_required(VERSION 3.14)
project(my_project)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  spins_test
  derivatives_test.cpp
  ../TArray.cpp
  ../grad.cpp
  ../Parformer.cpp
  ../T_util.cpp
  ../ESolver.cpp
  ../Splits.cpp
  ../gmres_2d_solver.cpp
  ../gmres_1d_solver.cpp
  ../Par_util.cpp
  ../multigrid.cpp
)
target_link_libraries(
  spins_test
  GTest::gtest_main
  blitz
  fftw3
  umfpack
  openblas
  mpi_cxx
  mpi
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++17")

#message("--source folder: ${PROJECT_SOURCE_DIR}")
target_include_directories(spins_test
  PRIVATE
  /usr/local/include
  /usr/lib/x86_64-linux-gnu/openmpi/include
  /usr/include/suitesparse
  /usr/local/include/suitesparse
  ${PROJECT_SOURCE_DIR}/..
)


include(GoogleTest)
gtest_discover_tests(spins_test)
