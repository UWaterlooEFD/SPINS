#include "../Science.hpp"
#include "math.h"
#include "../Par_util.hpp"
#include "stdio.h"
#include "../Split_reader.hpp"
#include "../T_util.hpp"
#include "../Parformer.hpp"
#include "../Sorter.hpp"
#include <numeric>

using blitz::Array;
using blitz::cos;
using namespace TArrayn;
using namespace NSIntegrator;
using namespace Transformer;
using namespace std;

bool compare_pairs(pair<double, double> a, pair<double, double> b) {
    return a.first < b.first;
}

// Compute Background Potential Energy (BPE)
void compute_Background_PE(double &BPE_tot, TArrayn::DTArray &rho, TArrayn::DTArray &quad3, int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double g, double rho_0, int iter, bool dimensional_rho, bool mapped, Array<double, 1> hill) {
    // Tensor variables for indexing
    blitz::firstIndex  ii;
    blitz::secondIndex jj;
    blitz::thirdIndex  kk;
    // Set mpi parameters
    int myrank, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // Arrays for sorting
    static Array<double, 3>             sort_rho, sort_quad;
    static vector<double>               sort_hill(Nx), sort_dx(Nx);
    static vector<double>               Lx_partsum(Nx), hillvol_partsum(Nx);
    double                             *local_vols   = new double[numprocs];
    double                             *local_vols_r = new double[numprocs];
    static double                       hill_max;
    static double                       hill_vol;
    static vector<pair<double, double>> height_width(Nx);

    // Stuff to do once at the beginning
    if (iter == 0) {
        // adjust if mapped
        if (mapped) {
            // information about the hill
            hill_vol = pssum(sum(hill * Ly * (*get_quad_x())));
            hill_max = psmax(max(hill));

            /* copy and sort the hill */
            double *hill_tmp  = new double[Nx];
            double *hill_tmp2 = new double[Nx];
            double *dx_tmp    = new double[Nx];
            double *dx_tmp2   = new double[Nx];
            // create temporary empty arrays for holding the grid and hill
            for (int II = 0; II < Nx; II++) {
                if ((II >= hill.lbound(firstDim)) and (II <= hill.ubound(firstDim))) {
                    // populate these arrays with their processor's information
                    hill_tmp2[II] = hill(II);
                    dx_tmp2[II]   = (*get_quad_x())(II);
                } else {
                    // else set to zero
                    hill_tmp2[II] = 0.;
                    dx_tmp2[II]   = 0.;
                }
            }
            // share across processors
            MPI_Allreduce(hill_tmp2, hill_tmp, Nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(dx_tmp2, dx_tmp, Nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // put into a double pair
            for (int II = 0; II < Nx; II++) {
                height_width[II].first  = hill_tmp[II];
                height_width[II].second = dx_tmp[II];
            }
            // sort the height_width pair according to the height
            sort(height_width.begin(), height_width.end(), compare_pairs);
            // put sorted pairs into separate vectors
            for (int II = 0; II < Nx; II++) {
                sort_hill[II] = height_width[II].first;
                sort_dx[II]   = height_width[II].second;
            }
            // Compute cumulative sums of hill volume (reusing Lx_partsum)
            for (int II = 0; II < Nx; II++) { Lx_partsum[II] = sort_hill[II] * sort_dx[II]; }
            std::partial_sum(Lx_partsum.begin(), Lx_partsum.end(), hillvol_partsum.begin());
            // Compute cumulative sums of dxs
            std::partial_sum(sort_dx.begin(), sort_dx.end(), Lx_partsum.begin());
            // clean-up
            delete[] dx_tmp;
            delete[] dx_tmp2;
            delete[] hill_tmp;
            delete[] hill_tmp2;
        }
    }

    // Compute sorted rho
    sortarray(rho, quad3, sort_rho, sort_quad);

    // Volume of memory-local portion of sorted array
    double local_vol = sum(sort_quad);

    // Share local volume information with other processors
    // ie. processor 0 has lightest fluid
    for (int II = 0; II < numprocs; II++) { local_vols[II] = (II <= myrank) ? local_vol : 0; }
    MPI_Allreduce(local_vols, local_vols_r, numprocs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // The total volume of fluid with greater or equal density
    // to the lightest element on the local domain
    double base_vol = local_vols_r[myrank];

    // Now compute BPE from the sorted rho field, stacking this processor's cells downwards from
    // the height that base_vol fills to
    double BPE = 0;
    if (!mapped) {
        double tmpH = base_vol / Lx / Ly, dH = 0, Area_star = Lx * Ly;
        for (int II = sort_rho.lbound(firstDim); II <= sort_rho.ubound(firstDim); II++) {
            for (int KK = sort_rho.lbound(thirdDim); KK <= sort_rho.ubound(thirdDim); KK++) {
                for (int JJ = sort_rho.lbound(secondDim); JJ <= sort_rho.ubound(secondDim); JJ++) {
                    // spread volume over domain and compute BPE for that cell
                    dH = sort_quad(II, JJ, KK) / Area_star;
                    if (dimensional_rho) {
                        // rho is dimensional density
                        BPE += g * sort_rho(II, JJ, KK) * (tmpH - 0.5 * dH) * dH * Area_star;
                    } else {
                        // rho is density anomaly
                        BPE += g * rho_0 * (1 + sort_rho(II, JJ, KK)) * (tmpH - 0.5 * dH) * dH * Area_star;
                    }
                    tmpH -= dH;
                }
            }
        }
    } else {
        // Over a hill the horizontal area changes with height, so each cell spans the heights to
        // which the fluid below it and the fluid up to its top fill (fill_height, exact for the
        // sorted hill profile). One area per cell (the area at its top) biases BPE high at first
        // order, and by an amount that depends on where each processor's portion starts.
        const double hill_area   = hill_vol / Ly; // hill cross-section in the x-z plane
        int          kfill       = Nx - 1;
        auto         fill_height = [&](double vol) {
            const double a = vol / Ly; // fluid cross-section below the fill height
            if (a + hill_area >= hill_max * Lx) return (a + hill_area) / Lx;
            // smallest kfill with a <= the cross-section below sort_hill[kfill]; vol only decreases
            // from call to call, so the search moves down from where the last call left it
            while (kfill > 0 && a <= sort_hill[kfill - 1] * Lx_partsum[kfill - 1] - hillvol_partsum[kfill - 1]) { kfill--; }
            if (kfill == 0) return sort_hill[0];
            return sort_hill[kfill] - (sort_hill[kfill] * Lx_partsum[kfill] - hillvol_partsum[kfill] - a) / Lx_partsum[kfill - 1];
        };
        double vol_top = base_vol, h_top = fill_height(vol_top);
        for (int II = sort_rho.lbound(firstDim); II <= sort_rho.ubound(firstDim); II++) {
            for (int KK = sort_rho.lbound(thirdDim); KK <= sort_rho.ubound(thirdDim); KK++) {
                for (int JJ = sort_rho.lbound(secondDim); JJ <= sort_rho.ubound(secondDim); JJ++) {
                    const double vol_bot = vol_top - sort_quad(II, JJ, KK), h_bot = fill_height(vol_bot);
                    const double h_mid = 0.5 * (h_top + h_bot);
                    if (dimensional_rho) {
                        BPE += g * sort_rho(II, JJ, KK) * h_mid * sort_quad(II, JJ, KK);
                    } else {
                        BPE += g * rho_0 * (1 + sort_rho(II, JJ, KK)) * h_mid * sort_quad(II, JJ, KK);
                    }
                    vol_top = vol_bot, h_top = h_bot;
                }
            }
        }
    }

    // Share BPE with other processors
    MPI_Allreduce(&BPE, &BPE_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // delete variables
    delete[] local_vols;
    delete[] local_vols_r;
}
