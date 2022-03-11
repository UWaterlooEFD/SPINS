#include "../Science.hpp"
#include "boost/lexical_cast.hpp"
#include <algorithm>
#include <blitz/array.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <string>
#include <vector>

using namespace Transformer;
using namespace blitz;

class QSPVector {
private:
  double *data;
  int length;
  int rows;
  int cols;

public:
  explicit QSPVector(int length)
      : data((double *)calloc(length, sizeof(double))), length(length), rows(0),
        cols(0) {
    if (!data) {
      std::cout << "Error! Data could not be initialized.\n";
    }
  }
  QSPVector(int Nx, int Ny)
      : data((double *)calloc(Nx * Ny, sizeof(double))), length(Nx * Ny),
        rows(Nx), cols(Ny) {
    if (!data) {
      std::cout << "Error! Data could not be initialized.\n";
    }
  }
  double *raw() const { return data; }
  int Length() const { return length; }
  int Rows() const { return rows; }
  int Cols() const { return cols; }
  double operator[](int i) const {
    assert(i >= 0 && i < length);
    return data[i];
  }
  double operator()(int row, int col) const {
    assert(row >= 0 && row < rows);
    assert(col >= 0 && col < cols);
    return data[(row * cols) + col];
  }
  double &operator()(int row, int col) {
    assert(row >= 0 && row < rows);
    assert(col >= 0 && col < cols);
    return data[(row * cols) + col];
  }
  ~QSPVector() { free(data); }
};

int index(int row, int col, int rows, int cols) {
  assert(row >= 0 && row < rows);
  assert(col >= 0 && col < cols);
  return (row * cols) + col;
}

void QSPCount(const TArrayn::DTArray &t, const TArrayn::DTArray &u,
              const TArrayn::DTArray &v, const TArrayn::DTArray &w,
              const char T1_name, const char S1_name, const int NS,
              const int NT, double T1_max, double S1_max, double T1_min,
              double S1_min, const int Nx, const int Ny, const int Nz,
              string filename, const int plotnum, bool mapped,
              TArrayn::DTArray *xgrid, TArrayn::DTArray *ygrid,
              TArrayn::DTArray *zgrid) {

  int local_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
  const TArrayn::DTArray *T1_ptr = NULL, *S1_ptr = NULL;

  switch (T1_name) {
  case 'u':
    T1_ptr = &u;
    break;
  case 'v':
    T1_ptr = &v;
    break;
  case 'w':
    T1_ptr = &w;
    break;
  case 'k':
    break;
  case 't':
    T1_ptr = &t;
    break;
  case 'T':
    T1_ptr = &t;
    break;
  default:
    return;
  }

  switch (S1_name) {
  case 'u':
    S1_ptr = &u;
    break;
  case 'v':
    S1_ptr = &v;
    break;
  case 'w':
    S1_ptr = &w;
    break;
  case 'k':
    break;
  case 't':
    S1_ptr = &t;
    break;
  case 'T':
    S1_ptr = &t;
    break;
  default:
    return;
  }

  int i_low = u.lbound(blitz::firstDim);
  int j_low = u.lbound(blitz::secondDim);
  int k_low = u.lbound(blitz::thirdDim);
  int i_high = u.ubound(blitz::firstDim);
  int j_high = u.ubound(blitz::secondDim);
  int k_high = u.ubound(blitz::thirdDim);

  // This block calculates the global min/max values, in case the user
  // didn't want to specify them.
  double double_max = std::numeric_limits<double>::max();
  double double_min = std::numeric_limits<double>::min();
  if (T1_max == double_max || S1_max == double_max || S1_max == double_min ||
      S1_min == double_min) {

    // If One of the variables is K.E or rho, we need to hand-roll the max/min
    // since, in general, the index of max(u) is the same as the index of max(v)
    if (T1_name == 'k' || T1_name == 't' || S1_name == 'k' || S1_name == 't') {
      double ke_max = -double_max, ke_min = double_max;
      double rho_max = -double_max, rho_min = double_max;

      // Main hand-rolled loop
      for (int i = i_low; i <= i_high; i++) {
        for (int j = j_low; j <= j_high; j++) {
          for (int k = k_low; k <= k_high; k++) {
            double ke_current = 0, tmp = 0;
            if (Nx > 1) {
              tmp = u(i, j, k);
              ke_current += tmp * tmp;
            }
            if (Ny > 1) {
              tmp = v(i, j, k);
              ke_current += tmp * tmp;
            }
            if (Nz > 1) {
              tmp = w(i, j, k);
              ke_current += tmp * tmp;
            }
            ke_current = 0.5 * ke_current;
            double rho_current = eqn_of_state_t(t(i, j, k));
            ke_max = ke_current > ke_max ? ke_current : ke_max;
            ke_min = ke_current < ke_min ? ke_current : ke_min;
            rho_max = rho_current > rho_max ? rho_current : rho_max;
            rho_min = rho_current < rho_min ? rho_current : rho_min;
          }
        }
      }

      double glob_ke_max, glob_ke_min, glob_rho_max, glob_rho_min;
      MPI_Allreduce(&ke_max, &glob_ke_max, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&ke_min, &glob_ke_min, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&rho_max, &glob_rho_max, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&rho_min, &glob_rho_min, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);

      switch (T1_name) {
      case 'k':
        T1_max = glob_ke_max;
        T1_min = glob_ke_min;
        break;
      case 't':
        T1_max = glob_rho_max;
        T1_min = glob_rho_min;
        break;
      default:
        T1_max = psmax(max(*T1_ptr));
        T1_min = psmin(min(*T1_ptr));
        break;
      }

      switch (S1_name) {
      case 'k':
        S1_max = glob_ke_max;
        S1_min = glob_ke_min;
        break;
      case 't':
        S1_max = glob_rho_max;
        S1_min = glob_rho_min;
        break;
      default:
        S1_max = psmax(max(*S1_ptr));
        S1_min = psmin(min(*S1_ptr));
        break;
      }
    } else { // !(cond1 || cond2) == !cond1 && !cond2
      S1_max = psmax(max(*S1_ptr));
      S1_min = psmin(min(*S1_ptr));
      T1_max = psmax(max(*T1_ptr));
      T1_min = psmin(min(*T1_ptr));
    }
  }

  double hS = (S1_max - S1_min) / (double)NS;
  double hT = (T1_max - T1_min) / (double)NT;
  double hS_inv = 1 / hS;
  double hT_inv = 1 / hT;

  QSPVector local_hist(NS, NT);
  QSPVector global_z_max(Nx, Ny);
  QSPVector global_z_min(Nx, Ny);
  // Find the range of Lz values per 2D-slice
  if (mapped) {
    QSPVector local_z_max(Nx, Ny);
    QSPVector local_z_min(Nx, Ny);
    //  We are slicing as if we are doing zgrid[i, j, :]
    for (int ii = i_low; ii <= i_high; ii++) {
      for (int jj = j_low; jj <= j_high; jj++) {
        // min is set to the highest possible value so it always gets changed
        double tmp_z_min = std::numeric_limits<double>::max();
        // max is set to the lowest possible value so it always gets changed
        double tmp_z_max = -tmp_z_min;
        double tmp;
        for (int kk = k_low; kk <= k_high; kk++) {
          tmp = (*zgrid)(ii, jj, kk);
          if (tmp > tmp_z_max) {
            tmp_z_max = tmp;
          } else if (tmp < tmp_z_min) {
            tmp_z_min = tmp;
          }
        }
        local_z_max(ii, jj) = tmp_z_max;
        local_z_min(ii, jj) = tmp_z_min;
      }
    }
    MPI_Allreduce(local_z_min.raw(), global_z_min.raw(), Nx * Ny, MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_z_max.raw(), global_z_max.raw(), Nx * Ny, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
  }

  // Main loop for QSP
  double Tval, Sval, tmp;
  for (int i = i_low; i <= i_high; i++) {
    for (int j = j_low; j <= j_high; j++) {
      for (int k = k_low; k <= k_high; k++) {

        if (T1_name == 'k') {
          Tval = 0;
          if (Nx > 1) {
            tmp = u(i, j, k);
            Tval += tmp * tmp;
          }
          if (Ny > 1) {
            tmp = v(i, j, k);
            Tval += tmp * tmp;
          }
          if (Nz > 1) {
            tmp = w(i, j, k);
            Tval += tmp * tmp;
          }
          Tval = 0.5 * Tval;
        } else if (T1_name == 't') {
          tmp = (*T1_ptr)(i, j, k);
          Tval = eqn_of_state_t(tmp);
        } else {
          Tval = (*T1_ptr)(i, j, k);
        }
        int idxT = floor((Tval - T1_min) * hT_inv);
        if (idxT < 0) {
          idxT = 0;
        } else if (idxT >= NT) {
          idxT = 0;
        }

        if (S1_name == 'k') {
          Sval = 0;
          if (Nx > 1) {
            tmp = u(i, j, k);
            Sval += tmp * tmp;
          }
          if (Ny > 1) {
            tmp = v(i, j, k);
            Sval += tmp * tmp;
          }
          if (Nz > 1) {
            tmp = w(i, j, k);
            Sval += tmp * tmp;
          }
          Sval = 0.5 * Sval;
        } else if (S1_name == 't') {
          tmp = (*S1_ptr)(i, j, k);
          Sval = eqn_of_state_t(tmp);
        } else {
          Sval = (*S1_ptr)(i, j, k);
        }
        int idxS = floor((Sval - S1_min) * hS_inv);
        if (idxS < 0) {
          idxS = 0;
        } else if (idxS >= NS) {
          idxS = 0;
        }

        double volume_weight;
        if (mapped) {
          // Calculate the Lz range
          double Lzmax_now = global_z_max(ii, jj);
          double Lzmin_now = global_z_min(ii, jj);

          // Calculate the arc length
          double arc, z_high, z_low, cos_high, cos_low;
          if (k > 0 && k < Nz - 1) {
            z_high = (double)(k + 1);
            z_low = (double)(k - 1);
          } else if (k == 0) {
            z_high = (double)(k + 1);
            z_low = (double)(k);
          } else if (k == Nz - 1) {
            z_high = (double)(k);
            z_low = (double)(k - 1);
          } else { // Failure
            std::cout << "k was out of bounds somehow. Failing...\n";
            return;
          }
          cos_high = std::cos(M_PI * z_high / (double)Nz);
          cos_low = std::cos(M_PI * z_low / (double)Nz);
          arc = 0.5 * (cos_low - cos_high);

          // Calculate the volume weight
          volume_weight = arc * (Lzmax_now - Lzmin_now);
        } else {
          volume_weight = 1.0;
        }

        // local_hist[index(idxS, idxT, NS, NT)] += volume_weight;
        local_hist(idxS, idxT) += volume_weight;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD); // Wait for everyone to finish
  if (local_rank == 0) {
    QSPVector glob_hist(NS, NT);
    MPI_Reduce(local_hist.raw(), glob_hist.raw(), // send and receive buffers
               NS * NT, MPI_DOUBLE,               // count and datatype
               MPI_SUM, 0,      // Reduction operator and root process #
               MPI_COMM_WORLD); // Communicator
    filename = filename + "." + boost::lexical_cast<string>(plotnum) + ".csv";
    std::fstream outfile;
    outfile.open(filename.c_str(), std::ios_base::out);
    if (outfile.is_open()) {
      outfile << T1_max << ',' << T1_min << ',' << S1_max << ',' << S1_min;
      for (int i = 4; i < NT; i++) {
        outfile << ',' << 0;
      }
      outfile << std::endl;
      for (int ii = 0; ii < NS; ii++) {
        outfile << glob_hist[index(ii, 0, NS, NT)];
        for (int jj = 1; jj < NT; jj++) {
          outfile << ',' << glob_hist[index(ii, jj, NS, NT)];
        }
        outfile << std::endl;
      }
    }
  } else {
    MPI_Reduce(local_hist.raw(), NULL, // send and receive buffers
               NS * NT, MPI_DOUBLE,    // count and datatype
               MPI_SUM, 0,             // Reduction operator and root process #
               MPI_COMM_WORLD);        // Communicator
  }
}
