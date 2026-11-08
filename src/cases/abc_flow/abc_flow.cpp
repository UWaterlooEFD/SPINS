/* Verification case: decaying Arnold-Beltrami-Childress (ABC) flow in a triply periodic box.

   u = (A sin kz + C cos ky, B sin kx + A cos kz, C sin ky + B cos kx) exp(-nu k^2 t)

   is an exact solution of the incompressible Navier-Stokes equations: it is divergence-free and a
   Beltrami flow (curl u = k u), so (u . grad) u = grad(|u|^2/2) is a pure gradient, which the
   pressure projection removes exactly. On a Fourier grid with a few points per wavelength every
   field and product is represented exactly and lies below the filter cutoff, so the only error
   left is the time stepper's on the linear decay, which is tiny for small nu dt.

   The case exercises 3D Fourier transforms in all directions, every velocity component, the
   projection of a pure gradient and the MPI slab transposes. Distinct A, B and C make component
   and index permutations visible. It writes, every step, the max error of each component
   relative to the exact amplitude, max |div u| and the kinetic-energy ratio to the exact value.
   With max_error > 0 it aborts (exit code 3) at the first step whose error exceeds it. */

#include "../../BaseCase.hpp"
#include "../../Options.hpp"
#include "../../Science.hpp"

blitz::firstIndex  ii;
blitz::secondIndex jj;
blitz::thirdIndex  kk;

// Grid
int     Nx, Ny, Nz;
double  Lx, Ly, Lz;
DIMTYPE intype_x, intype_y, intype_z;
string  grid_type[3];
// Physics and flow
double visco, A, B, C;
int    kwave;
// Time and output
double final_time, plot_interval, dt_max, max_error;
bool   restarting;
double initial_time;
int    restart_sequence;
int    iter = 0;

class userControl : public BaseCase {
  public:
    Array<double, 1> xx, yy, zz;
    Grad            *gradient_op;
    DTArray         *temp, *ue, *ve, *we;
    int              plot_number;
    double           next_plot;

    double  length_x() const { return Lx; }
    double  length_y() const { return Ly; }
    double  length_z() const { return Lz; }
    int     size_x() const { return Nx; }
    int     size_y() const { return Ny; }
    int     size_z() const { return Nz; }
    DIMTYPE type_x() const { return intype_x; }
    DIMTYPE type_y() const { return intype_y; }
    DIMTYPE type_z() const { return intype_z; }
    void    set_grad(Grad *in_grad) { gradient_op = in_grad; }
    double  get_visco() const { return visco; }
    double  init_time() const { return initial_time; }
    int     get_restart_sequence() const { return restart_sequence; }
    double  get_dt_max() const { return dt_max; }
    double  get_next_plot() { return next_plot; }
    int     numtracers() const { return 0; }

    double decay(double t) const { return exp(-visco * kwave * kwave * t); }

    void exact(double t, DTArray &u, DTArray &v, DTArray &w) {
        const double e = decay(t), k = kwave;
        u = (A * sin(k * zz(kk)) + C * cos(k * yy(jj)) + 0 * xx(ii)) * e;
        v = (B * sin(k * xx(ii)) + A * cos(k * zz(kk)) + 0 * yy(jj)) * e;
        w = (C * sin(k * yy(jj)) + B * cos(k * xx(ii)) + 0 * zz(kk)) * e;
    }

    void init_vels(DTArray &u, DTArray &v, DTArray &w) {
        if (restarting) {
            init_vels_restart(u, v, w);
        } else {
            exact(initial_time, u, v, w);
            write_array(u, "u", plot_number);
            write_array(v, "v", plot_number);
            write_array(w, "w", plot_number);
        }
    }

    void analysis(double time, DTArray &u, DTArray &v, DTArray &w, vector<DTArray *> &tracers, DTArray &pressure) {
        if (!temp) {
            temp = alloc_array(Nx, Ny, Nz);
            ue   = alloc_array(Nx, Ny, Nz);
            ve   = alloc_array(Nx, Ny, Nz);
            we   = alloc_array(Nx, Ny, Nz);
        }
        const double amp = (fabs(A) + fabs(B) + fabs(C)) * decay(time);
        // Errors against the exact solution, one component at a time
        exact(time, *ue, *ve, *we);
        const double err[3] = {psmax(max(abs(u - *ue))) / amp, psmax(max(abs(v - *ve))) / amp, psmax(max(abs(w - *we))) / amp};
        // Divergence, through the same gradient operator the integrator uses
        Transformer::S_EXP expan[3];
        find_expansion(grid_type, expan, "u");
        gradient_op->setup_array(&u, expan[0], expan[1], expan[2]);
        gradient_op->get_dx(temp, false);
        find_expansion(grid_type, expan, "v");
        gradient_op->setup_array(&v, expan[0], expan[1], expan[2]);
        gradient_op->get_dy(temp, true);
        find_expansion(grid_type, expan, "w");
        gradient_op->setup_array(&w, expan[0], expan[1], expan[2]);
        gradient_op->get_dz(temp, true);
        const double max_div = psmax(max(abs(*temp))) / (amp * kwave);
        // Kinetic energy against the exact (1/2)(A^2 + B^2 + C^2) V exp(-2 nu k^2 t)
        const double dV    = Lx * Ly * Lz / (double(Nx) * Ny * Nz);
        const double ke    = pssum(sum(0.5 * (u * u + v * v + w * w))) * dV;
        const double ke_ex = 0.5 * (A * A + B * B + C * C) * Lx * Ly * Lz * decay(time) * decay(time);

        if (master()) {
            string header, line;
            add_diagnostic("Iter", iter, header, line);
            add_diagnostic("Time", time, header, line);
            add_diagnostic("Err_u", err[0], header, line);
            add_diagnostic("Err_v", err[1], header, line);
            add_diagnostic("Err_w", err[2], header, line);
            add_diagnostic("Max_div", max_div, header, line);
            add_diagnostic("KE_ratio_m1", ke / ke_ex - 1, header, line);
            write_diagnostics(header, line, iter, restarting);
            fprintf(stdout, "[%d] %.6f: errors %.3e %.3e %.3e, div %.3e\n", iter, time, err[0], err[1], err[2], max_div);
        }
        if (max_error > 0 && fmax(err[0], fmax(err[1], err[2])) > max_error) {
            if (master()) fprintf(stderr, "abc_flow: FAIL at t = %g, error above max_error = %g\n", time, max_error);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        if ((time - next_plot) > -1e-6) {
            plot_number++;
            write_array(u, "u", plot_number);
            write_array(v, "v", plot_number);
            write_array(w, "w", plot_number);
            next_plot += plot_interval;
        }
        iter++;
    }

    userControl() : xx(split_range(Nx)), yy(Ny), zz(Nz), gradient_op(0), temp(0), ue(0), ve(0), we(0), plot_number(restart_sequence), next_plot(initial_time + plot_interval) {
        compute_quadweights(size_x(), size_y(), size_z(), length_x(), length_y(), length_z(), type_x(), type_y(), type_z());
        automatic_grid(0, 0, 0, &xx, &yy, &zz);
    }
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    options_init();
    option_category("Grid options");
    add_option("Nx", &Nx, 16, "Number of points in X");
    add_option("Ny", &Ny, 16, "Number of points in Y");
    add_option("Nz", &Nz, 16, "Number of points in Z");
    option_category("Flow");
    add_option("visco", &visco, 0.05, "Viscosity");
    add_option("A", &A, 1.0, "ABC coefficient A");
    add_option("B", &B, 0.8164965809277260, "ABC coefficient B");
    add_option("C", &C, 0.5773502691896258, "ABC coefficient C");
    add_option("kwave", &kwave, 1, "Wavenumber (box length 2 pi/kwave)");
    option_category("Temporal options");
    add_option("final_time", &final_time, 1.0, "Final time");
    add_option("plot_interval", &plot_interval, 0.5, "Time between field writes");
    add_option("dt_max", &dt_max, 0.01, "Maximum time step");
    add_option("max_error", &max_error, 0.0, "Abort if a component's relative error exceeds this (0: off)");
    option_category("Restart options");
    add_option("restart", &restarting, false, "Restart from prior output time");
    add_option("restart_time", &initial_time, 0.0, "Time to restart from");
    add_option("restart_sequence", &restart_sequence, 0, "Sequence number to restart from");
    option_category("Filter options");
    add_option("f_cutoff", &f_cutoff, 0.6, "Filter cut-off frequency");
    add_option("f_order", &f_order, 2.0, "Filter order");
    add_option("f_strength", &f_strength, 20.0, "Filter strength");
    options_parse(argc, argv);

    Lx = Ly = Lz = 2 * M_PI / kwave;
    grid_type[0] = grid_type[1] = grid_type[2] = "FOURIER";
    parse_boundary_conditions(grid_type[0], grid_type[1], grid_type[2], intype_x, intype_y, intype_z);

    userControl              mycode;
    FluidEvolve<userControl> do_stuff(&mycode);
    do_stuff.initialize();
    if (master()) fprintf(stdout, "ABC flow: %d x %d x %d, nu = %g, A B C = %g %g %g, k = %d\n", Nx, Ny, Nz, visco, A, B, C, kwave);
    do_stuff.do_run(final_time);
    MPI_Finalize();
    return 0;
}
