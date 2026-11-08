/* Verification case with selectable exact solutions (tests R1, R2, R4-R12, R14 and U12 of the
   verification plan). Set `problem` in spins.conf; each problem fixes its own grid, physics and
   reference.

   Linear problems on modes the grid represents exactly have a "discrete-exact" reference: the
   mode amplitudes are advanced with SPINS's own get_coeff weights on the same time levels the
   integrator used (start-up included), so any difference beyond round-off means the integrator
   does not apply its scheme as written. All problems also report the error against the
   continuous exact solution, which measures the scheme's accuracy.

     tg_periodic      2D Taylor-Green vortex, periodic x and z                 discrete-exact
     tg_freeslip      2D Taylor-Green vortex, free-slip walls in x and z       discrete-exact
     tg_rotating      horizontal Taylor-Green flow in 3D with Coriolis forcing  discrete-exact
     mms_steady       steady manufactured flow with non-gradient advection     exact steady state
     mms_linear       the same flow with amplitude 1 + b t                     continuous
     poiseuille       body-force-driven channel flow, Chebyshev walls in z     exact steady state
     womersley        oscillating channel flow, Chebyshev walls in z           continuous
     ekman            two-wall Ekman channel, Chebyshev walls in z             continuous steady
     inertial         uniform inertial oscillation                             discrete-exact
     igw              plane internal gravity wave, exact at any amplitude      discrete-exact
     tracer_fourier   diffusing tracer mode                                    discrete-exact
     tracer_advect    tracer mode advected by a uniform flow                   discrete-exact
     tracer_cheb_mode tracer mode between zero-Dirichlet Chebyshev walls in x  continuous
     tracer_cheb_bc   the same mode, with its zero wall values supplied through
                      tracer_bc_forcing (the boundary-value code path)         continuous
     mapped_metric    no flow; derivatives on a grid mapped over a cosine hill  metric identities
     mapped_metric_fs the same with free-slip x (cosine hill, even about walls)
     mapped_metric_gauss  the same with periodic x and a Gaussian hill
     rest_freeslip    stratified fluid at rest, free-slip z (cosine density)   u = 0
     rest_noslip      the same with Chebyshev z (linear density)               u = 0
     rest_mapped      the same over a cosine hill (mapped grid)                u = 0
     rest_mapped_f    the same with Coriolis forcing                           u = 0
     noslip_quad      steady flow psi = (1 - z^2)^2 sin x between no-slip walls; its wall
                      shear has non-zero divergence, so the splitting leaves an error
     noslip_cubic     psi = (1 - z^2)^3 sin x, zero wall-shear divergence      exact steady state
     noslip_box_zquad psi = (1 - x^2)^3 (1 - z^2)^2 in a no-slip box (splitting error at the
                      z walls); noslip_box_xquad is its transpose (error at the x walls)
     budget_energy    unforced periodic flow, filter off: KE, dissipation and enstrophy
     budget_bpe       resting fluid, diffusing cosine density mode: BPE, PE    discrete-exact
     mms_osc          the mms flow with amplitude 1 + 0.5 sin(2 pi t)            continuous
     gs_noslip        Guermond-Shen field between no-slip walls, with a wall-normal pressure
                      gradient: velocity, divergence and pressure errors in L-inf and L2

   Options nu (viscosity) and dt_pattern, dt_scale, dt_ratio (prescribed time-step sequences, see
   check_timestep) serve the time-order studies.

   Coriolis and buoyancy are supplied in forcing(), as SPINS cases do: u_f = f v, v_f = -f u and
   w_f = -g rho, with the background stratification as tracer forcing (N^2/g) w (igw) or in the
   tracer itself (rest_*).

   Diagnostics per step: Err_vel and Err_T against the primary reference (discrete-exact where
   available), Cont_vel and Cont_T against the continuous solution, all relative to the
   reference's own scale (absolute where the reference is zero). Some problems add columns:
   metric and derivative errors (mapped_metric*), max |div u| (noslip_*), the integrals KE, Diss
   (SPINS's dissipation()) and Enst (enstrophy_density()) for budget_energy, and BPE
   (compute_Background_PE), PE and the exact BPE rate for budget_bpe. With max_error > 0 the run
   aborts (exit 3) once Err_vel or Err_T exceeds it. */

#include "../../BaseCase.hpp"
#include "../../Options.hpp"
#include "../../Timestep.hpp"
#include <complex>
#include <cstring>
#include <utility>

blitz::firstIndex            ii;
blitz::secondIndex           jj;
blitz::thirdIndex            kk;
typedef std::complex<double> cplx;

string  problem;
int     Nx, Ny, Nz, ntracers = 0;
double  Lx, Ly = 1, Lz, MinX = 0, MinY = 0, MinZ = 0;
DIMTYPE intype_x, intype_y, intype_z;
double  visco = 0, kappa = 0, rot_f = 0, N2 = 0, grav = 1, U0 = 1, wave_k = 1, wave_m = 1, omega = 0, growth_b = 0;
double  final_time, plot_interval = 1e6, dt_max, max_error;
double  hill = 0, a0 = 0.01; // mapped-bottom amplitude (0: unmapped); budget_bpe density amplitude
bool    gauss_hill = false;
int     psi_px = 0, psi_pz = 0; // noslip_*: streamfunction factors (0: sin s, p: (1 - s^2)^p)
string  grid_type[3];           // boundary types as Science/ functions expect them
string  mode_file = "mode.txt"; // eig_*: eigenmode written by scripts/in/eigen_modes.py on this grid
double  eps_amp = 1e-6, Ra = 1000;
string  dt_pattern = "default"; // time-step pattern for order studies (see check_timestep)
double  dt_scale = 1, dt_ratio = 1.5;
int     iter = 0;

bool starts_with(const string &s, const char *prefix) {
    return s.compare(0, strlen(prefix), prefix) == 0;
}
bool manufactured() { // problems forced towards a manufactured flow u = g(t) U(x, z)
    return problem == "mms_steady" || problem == "mms_linear" || problem == "mms_osc" || starts_with(problem, "noslip_");
}
// The time factor g(t) of the manufactured flows, and g'(t)
double amp(double t) {
    return problem == "mms_osc" ? 1 + growth_b * sin(omega * t) : 1 + growth_b * t;
}
double amp_t(double t) {
    return problem == "mms_osc" ? growth_b * omega * cos(omega * t) : growth_b;
}

// d^n/ds^n of one factor of a separable streamfunction: sin(s) (p = 0) or (1 - s^2)^p
double psi_factor(int p, int n, double s) {
    if (p == 0) return sin(s + n * M_PI / 2);
    double r = 0, c = 1; // c = (-1)^j C(p, j), the coefficient of s^(2j)
    for (int j = 0; j <= p; j++) {
        double d = 1; // d^n/ds^n s^(2j) = d s^(2j - n)
        for (int q = 0; q < n; q++) d *= 2 * j - q;
        if (2 * j >= n) r += c * d * pow(s, 2 * j - n);
        c *= -double(p - j) / (j + 1);
    }
    return r;
}

/* A mode amplitude vector X (n <= 2 complex components) obeying X' = lam X + M X, with lam
   treated implicitly and M explicitly, advanced with Timestep::get_coeff on given time levels. */
struct Recurrence {
    int    n       = 0;
    double lam     = 0;
    cplx   M[2][2] = {
          {0, 0},
          {0, 0}
    };
    double t[4]; // levels -2, -1, 0, 1
    cplx   X[4][2];
    void   start(double t0, cplx x0, cplx x1) {
          for (int l = 0; l < 4; l++) t[l] = t0, X[l][0] = x0, X[l][1] = x1;
    }
    void step(double tnew) {
        for (int l = 0; l < 3; l++) t[l] = t[l + 1], X[l][0] = X[l + 1][0], X[l][1] = X[l + 1][1];
        t[3] = tnew;
        Timestep::Stepped<double> T(4), L(4), R(4);
        for (int l = 0; l < 4; l++) T[l - 2] = t[l];
        Timestep::get_coeff(T, L, R);
        cplx acc[2] = {0, 0};
        for (int l = 0; l < 3; l++)
            for (int a = 0; a < n; a++) {
                acc[a] -= L[l - 2] * X[l][a];
                for (int b = 0; b < n; b++) acc[a] += R[l - 2] * M[a][b] * X[l][b];
            }
        for (int a = 0; a < n; a++) X[3][a] = acc[a] / (L[1] - lam);
    }
    cplx x(int a) const { return X[3][a]; }
};

class userControl : public BaseCase {
  public:
    Array<double, 1> xx, yy, zz;
    DTArray         *ru, *rv, *rw, *rT, *cu, *cv, *cw, *cT; // primary and continuous references
    DTArray         *mU, *mW, *mNU, *mNW, *mLU, *mLW;       // manufactured-solution fields
    DTArray         *zgrid, *quad3, *t1, *t2;               // physical z, quadrature weights, temporaries
    DTArray         *gPX, *gPZ, *gP;                        // gs_noslip: spatial parts of grad p and p
    Grad            *gradient_op;
    Recurrence       rec;
    bool             discrete;
    double           metric_zz = 0, metric_xz = 0, deriv_x = 0, deriv_z = 0; // mapped_metric*

    void set_grad(Grad *in_grad) { gradient_op = in_grad; }
    bool is_mapped() const { return hill != 0; }
    void do_mapping(DTArray &xg, DTArray &yg, DTArray &zg) {
        xg = xx(ii) + 0 * jj + 0 * kk;
        yg = yy(jj) + 0 * ii + 0 * kk;
        zg = *zgrid;
    }
    // Bottom height: cosine hill, periodic over Lx or even about free-slip walls, or a Gaussian
    double bottom(double x) const {
        if (gauss_hill) return hill * exp(-pow((x - MinX - Lx / 2) / (0.1 * Lx), 2));
        return hill * (1 + cos((intype_x == PERIODIC ? 2 : 1) * M_PI * (x - MinX) / Lx)) / 2;
    }
    // Initial and exact density of the rest_* problems (kappa = 0, so it never changes)
    void rest_density(DTArray &T) {
        if (problem == "rest_freeslip") T = 0.5 * cos(M_PI * ((*zgrid) - MinZ) / Lz);
        else T = -0.5 * ((*zgrid) - MinZ - Lz / 2);
    }
    // gs_noslip: max over x of |div u|, |u - u_exact| and the pressure error at each z level, written
    // to profile.txt at the final time (the near-wall layer against wall distance)
    void write_profile(const DTArray &div, const DTArray &vel_err2, const DTArray &p_err) {
        vector<double> prof(3 * Nz, 0.0);
        for (int i = div.lbound(0); i <= div.ubound(0); i++)
            for (int j = div.lbound(1); j <= div.ubound(1); j++)
                for (int k = div.lbound(2); k <= div.ubound(2); k++) {
                    prof[3 * k]     = fmax(prof[3 * k], fabs(div(i, j, k)));
                    prof[3 * k + 1] = fmax(prof[3 * k + 1], sqrt(vel_err2(i, j, k)));
                    prof[3 * k + 2] = fmax(prof[3 * k + 2], fabs(p_err(i, j, k)));
                }
        MPI_Allreduce(MPI_IN_PLACE, prof.data(), 3 * Nz, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (master()) {
            FILE *f = fopen("profile.txt", "w");
            fprintf(f, "z, Div, Vel_err, P_err\n");
            for (int k = 0; k < Nz; k++) fprintf(f, "%.17e, %.17e, %.17e, %.17e\n", zz(k), prof[3 * k], prof[3 * k + 1], prof[3 * k + 2]);
            fclose(f);
        }
    }
    // mapped_metric*: D_z(z) = 1 and D_x(z) = 0, and the derivatives of a smooth function of the
    // physical coordinates, through the Jacobian FluidEvolve built for the mapped grid
    void compute_metric() {
        S_EXP ex[3];
        find_expansion(grid_type, ex, "rho"); // a scalar: even about free-slip walls
        gradient_op->setup_array(zgrid, ex[0], ex[1], ex[2]);
        gradient_op->get_dz(t1);
        metric_zz = psmax(max(abs(*t1 - 1)));
        gradient_op->get_dx(t1);
        metric_xz      = psmax(max(abs(*t1)));
        const double k = (intype_x == PERIODIC ? 2 : 1) * M_PI / Lx;
        DTArray     &f = *ru, &fx = *rv, &fz = *rw;
        if (intype_x == PERIODIC) { // f = sin(kx) z^2 + cos(kx) z
            f  = sin(k * xx(ii)) * (*zgrid) * (*zgrid) + cos(k * xx(ii)) * (*zgrid);
            fx = k * (cos(k * xx(ii)) * (*zgrid) * (*zgrid) - sin(k * xx(ii)) * (*zgrid));
            fz = 2 * sin(k * xx(ii)) * (*zgrid) + cos(k * xx(ii)) + 0 * jj;
        } else { // f = cos(kx) z^2 + z, even about the walls
            f  = cos(k * xx(ii)) * (*zgrid) * (*zgrid) + (*zgrid);
            fx = -k * sin(k * xx(ii)) * (*zgrid) * (*zgrid);
            fz = 2 * cos(k * xx(ii)) * (*zgrid) + 1;
        }
        gradient_op->setup_array(&f, ex[0], ex[1], ex[2]);
        gradient_op->get_dx(t1);
        deriv_x = psmax(max(abs(*t1 - fx))) / psmax(max(abs(fx)));
        gradient_op->get_dz(t1);
        deriv_z = psmax(max(abs(*t1 - fz))) / psmax(max(abs(fz)));
    }

    double  length_x() const override { return Lx; }
    double  length_y() const override { return Ly; }
    double  length_z() const override { return Lz; }
    int     size_x() const override { return Nx; }
    int     size_y() const override { return Ny; }
    int     size_z() const override { return Nz; }
    DIMTYPE type_x() const override { return intype_x; }
    DIMTYPE type_y() const override { return intype_y; }
    DIMTYPE type_z() const override { return intype_z; }
    double  get_visco() const override { return visco; }
    double  get_diffusivity(int) const override { return kappa; }
    double  get_rot_f() const override { return rot_f; }
    double  init_time() const override { return 0; }
    double  get_dt_max() const override { return dt_max; }
    double  get_next_plot() override { return plot_interval; }

    /* Time-step patterns for order studies. FluidEvolve asks once per outer step (the first is the
       start-up block of substeps summing to it) and takes the answer as given. Steps are
       dt_scale * dt_max * q_n:
         default      BaseCase's choice (CFL, 1.5 growth cap, dt_max)
         constant     q_n = 1
         alternating  q_n = 1.4, 1/1.4, 1.4, ...
         random       seeded random walk, ratios in [0.7, 1.4], q kept in [0.5, 2]
         sawtooth     q grows by 1.5 per step and halves once it exceeds 2
         cut          q = 1, then 0.1 from half the final time on (an abrupt reduction)
         grow         q_n = dt_ratio^(n - 20) up to n = 20, then 1 (sustained step growth)
       The last step is cut to land on final_time the way FluidEvolve does it. */
    int          outer  = 0;
    double       q_last = 1;
    unsigned int lcg    = 12345u;
    double       check_timestep(double step, double now) {
              if (dt_pattern == "default") return BaseCase::check_timestep(step, now);
        double q = 1;
              if (dt_pattern == "alternating") {
                  q = (outer % 2 == 0) ? 1.4 : 1 / 1.4;
        } else if (dt_pattern == "random") {
                  if (outer > 0) {
                      lcg            = 1664525u * lcg + 1013904223u;
                      const double r = 0.7 + 0.7 * (lcg / 4294967296.0);
                      q              = (q_last * r >= 0.5 && q_last * r <= 2) ? q_last * r : q_last / r;
            }
        } else if (dt_pattern == "sawtooth") {
                  if (outer > 0) q = (q_last <= 2) ? 1.5 * q_last : q_last / 2;
        } else if (dt_pattern == "cut") {
                  q = (now < 0.5 * final_time * (1 - 1e-12)) ? 1 : 0.1;
        } else if (dt_pattern == "grow") {
                  q = pow(dt_ratio, fmin(outer, 20) - 20);
        } else if (dt_pattern != "constant") {
                  if (master()) fprintf(stderr, "verify_exact: unknown dt_pattern '%s'\n", dt_pattern.c_str());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
              q_last         = q, outer++;
              const double h = dt_scale * dt_max * q;
              return (h >= (1 - 1e-8) * (final_time - now)) ? (1 + 5e-9) * (final_time - now) : h;
    }
    int  numtracers() const { return ntracers; }
    int  numActive() const { return (problem == "igw" || starts_with(problem, "rest_") || starts_with(problem, "eig_")) ? ntracers : 0; }
    void tracer_bc_z(int, double &dir, double &neu) const { // fixed-temperature walls for Rayleigh-Benard
        const bool rb = starts_with(problem, "eig_rb");
        dir           = rb ? 1 : 0;
        neu           = rb ? 0 : 1;
    }
    int  numPassive() const { return ntracers - numActive(); }
    bool tracer_bc_forcing() const { return problem == "tracer_cheb_bc"; }
    void tracer_bc_x(int, double &dir, double &neu) const {
        const bool cheb = problem.compare(0, 11, "tracer_cheb") == 0;
        dir             = cheb ? 1 : 0;
        neu             = cheb ? 0 : 1;
    }

    DTArray *alloc() { return alloc_array(Nx, Ny, Nz); }

    // d^i/dx^i d^j/dz^j of the manufactured streamfunction: psi = sin x sin z + 0.5 cos 2x sin z
    // (mms_*, budget_energy) or the separable psi_factor(psi_px, x) psi_factor(psi_pz, z) (noslip_*)
    void psi_deriv(int i, int j, DTArray &out) {
        if (starts_with(problem, "noslip_")) {
            for (int l = out.lbound(0); l <= out.ubound(0); l++)
                for (int m = out.lbound(1); m <= out.ubound(1); m++)
                    for (int n = out.lbound(2); n <= out.ubound(2); n++) out(l, m, n) = psi_factor(psi_px, i, xx(l)) * psi_factor(psi_pz, j, zz(n));
            return;
        }
        const double a[2] = {1.0, 0.5}, kx[2] = {1, 2}, px[2] = {0, M_PI / 2}, kz[2] = {1, 1}, pz[2] = {0, 0};
        out = 0;
        for (int m = 0; m < 2; m++) out = out + a[m] * pow(kx[m], i) * pow(kz[m], j) * sin(kx[m] * xx(ii) + px[m] + i * M_PI / 2) * sin(kz[m] * zz(kk) + pz[m] + j * M_PI / 2) + 0 * yy(jj);
    }

    // Shape of each discrete-exact problem's fields for amplitudes X (real parts taken).
    void fields_from(cplx X0, cplx X1, DTArray &u, DTArray &v, DTArray &w, DTArray *T) {
        const double k = wave_k, m = wave_m;
        u = 0, v = 0, w = 0;
        if (T) *T = 0;
        if (problem == "tg_periodic" || problem == "tg_freeslip") {
            u = real(X0) * U0 * sin(k * xx(ii)) * cos(k * zz(kk)) + 0 * yy(jj);
            w = -real(X0) * U0 * cos(k * xx(ii)) * sin(k * zz(kk)) + 0 * yy(jj);
        } else if (problem == "tg_rotating") {
            u = real(X0) * U0 * sin(k * xx(ii)) * cos(k * yy(jj)) + 0 * zz(kk);
            v = -real(X0) * U0 * cos(k * xx(ii)) * sin(k * yy(jj)) + 0 * zz(kk);
        } else if (problem == "inertial") {
            u = real(X0) + 0 * xx(ii) + 0 * yy(jj) + 0 * zz(kk);
            v = imag(X0) + 0 * xx(ii) + 0 * yy(jj) + 0 * zz(kk);
        } else if (problem == "igw") { // psi = A cos(theta), b = B sin(theta), rho = -b/g
            u  = -m * real(X0) * sin(k * xx(ii) + m * zz(kk)) + 0 * yy(jj);
            w  = k * real(X0) * sin(k * xx(ii) + m * zz(kk)) + 0 * yy(jj);
            *T = -real(X1) / grav * sin(k * xx(ii) + m * zz(kk)) + 0 * yy(jj);
        } else if (problem == "tracer_fourier") {
            *T = real(X0) * cos(k * xx(ii)) + 0 * yy(jj) + 0 * zz(kk);
        } else if (problem == "tracer_advect") { // T = Re(c exp(ikx)), uniform u = U0
            u  = U0 + 0 * xx(ii) + 0 * yy(jj) + 0 * zz(kk);
            *T = real(X0) * cos(k * xx(ii)) - imag(X0) * sin(k * xx(ii)) + 0 * yy(jj) + 0 * zz(kk);
        } else if (problem == "budget_bpe") {
            *T = real(X0) * a0 * cos(M_PI * (zz(kk) - MinZ) / Lz) + 0 * xx(ii) + 0 * yy(jj);
        }
    }

    // Continuous exact solution at time t.
    void continuous(double t, DTArray &u, DTArray &v, DTArray &w, DTArray *T) {
        const double k = wave_k, m = wave_m;
        if (problem == "tg_periodic" || problem == "tg_freeslip" || problem == "tg_rotating") {
            fields_from(exp(-2 * visco * k * k * t), 0, u, v, w, T);
        } else if (problem == "inertial") {
            fields_from(U0 * std::exp(cplx(0, -rot_f * t)), 0, u, v, w, T);
        } else if (problem == "igw") {
            const double K2 = k * k + m * m, om = sqrt(N2) * k / sqrt(K2), e = exp(-visco * K2 * t);
            const double A0 = U0, B0 = 0; // X(t) = e (cos(om t) I + sin(om t)/om M) X0
            const double A = e * (cos(om * t) * A0 + sin(om * t) / om * (k / K2) * B0);
            const double B = e * (cos(om * t) * B0 + sin(om * t) / om * (-N2 * k) * A0);
            fields_from(A, B, u, v, w, T);
        } else if (problem == "tracer_fourier") {
            fields_from(exp(-kappa * k * k * t), 0, u, v, w, T);
        } else if (problem == "tracer_advect") {
            fields_from(std::exp(cplx(-kappa * k * k, -k * U0) * t), 0, u, v, w, T);
        } else if (manufactured()) {
            const double g = amp(t);
            u = g * (*mU), v = 0, w = g * (*mW);
        } else if (problem == "gs_noslip") {
            u = sin(t) * (*mU), v = 0, w = sin(t) * (*mW);
        } else if (starts_with(problem, "rest_")) {
            u = 0, v = 0, w = 0;
            rest_density(*T);
        } else if (problem == "budget_bpe") {
            fields_from(exp(-kappa * pow(M_PI / Lz, 2) * t), 0, u, v, w, T);
        } else if (starts_with(problem, "eig_")) { // the base state; the mode is added at t = 0
            u = eU(kk) + 0 * ii + 0 * jj, v = 0, w = 0;
            if (T) *T = eT(kk) + 0 * ii + 0 * jj;
        } else if (problem == "budget_energy" || starts_with(problem, "mapped_metric")) {
            u = 0, v = 0, w = 0; // no reference (budget_energy is checked through its integrals)
        } else if (problem == "poiseuille") {
            u = U0 * (1 - zz(kk) * zz(kk)) + 0 * xx(ii) + 0 * yy(jj), v = 0, w = 0;
        } else if (problem == "womersley") { // u = Re{(F0/(i om))[1 - cosh(qz)/cosh(q)] e^{i om t}}, F0 = U0
            const cplx q = std::sqrt(cplx(0, omega / visco)), c = U0 / cplx(0, omega) * std::exp(cplx(0, omega * t)), ch = std::cosh(q);
            v = 0, w = 0;
            for (int i = u.lbound(0); i <= u.ubound(0); i++)
                for (int j = u.lbound(1); j <= u.ubound(1); j++)
                    for (int l = u.lbound(2); l <= u.ubound(2); l++) u(i, j, l) = real(c * (1.0 - std::cosh(q * zz(l)) / ch));
        } else if (problem == "ekman") { // u + iv = U0 [1 - cosh(qz)/cosh(q)], q = (1+i)/delta
            const double delta = sqrt(2 * visco / rot_f);
            const cplx   q = cplx(1, 1) / delta, ch = std::cosh(q);
            w = 0;
            for (int i = u.lbound(0); i <= u.ubound(0); i++)
                for (int j = u.lbound(1); j <= u.ubound(1); j++)
                    for (int l = u.lbound(2); l <= u.ubound(2); l++) {
                        const cplx W = U0 * (1.0 - std::cosh(q * zz(l)) / ch);
                        u(i, j, l)   = real(W);
                        v(i, j, l)   = imag(W);
                    }
        } else if (problem == "tracer_cheb_mode" || problem == "tracer_cheb_bc") {
            u = 0, v = 0, w = 0;
            *T = sin(M_PI * (xx(ii) - MinX) / Lx) * exp(-kappa * M_PI * M_PI / (Lx * Lx) * t) + 0 * yy(jj) + 0 * zz(kk);
        }
    }

    void init_vels(DTArray &u, DTArray &v, DTArray &w) {
        DTArray *T = ntracers ? alloc() : nullptr;
        continuous(0, u, v, w, T);
        delete T;
        if (problem == "budget_energy") u = *mU, w = *mW; // decays freely from the manufactured flow
        if (starts_with(problem, "eig_")) add_mode(u, mu), add_mode(w, mw);
    }
    void init_tracer(int, DTArray &T) {
        DTArray *u = alloc(), *v = alloc(), *w = alloc();
        continuous(0, *u, *v, *w, &T);
        delete u, delete v, delete w;
        if (starts_with(problem, "eig_")) add_mode(T, mT);
    }

    // eig_*: base profiles (U, U'', background tracer and its second derivative, N^2) on zz, and
    // the mode's complex amplitudes; f += eps_amp Re{a(z) exp(i k x)}
    Array<double, 1> eU, eUpp, eT, eTpp, eN2;
    vector<cplx>     mu, mw, mT;
    double           eig_k = 1, mode_norm = 0, eig_sr = 0, eig_si = 0;
    void             add_mode(DTArray &f, const vector<cplx> &a) {
                    for (int i = f.lbound(0); i <= f.ubound(0); i++)
            for (int j = f.lbound(1); j <= f.ubound(1); j++)
                for (int k = f.lbound(2); k <= f.ubound(2); k++) f(i, j, k) += eps_amp * real(a[k] * std::exp(cplx(0, eig_k * xx(i))));
    }
    void read_mode() {
        FILE *fp = fopen(mode_file.c_str(), "r");
        if (!fp) {
            if (master()) fprintf(stderr, "verify_exact: cannot open mode file '%s'\n", mode_file.c_str());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        if (fscanf(fp, "# %lf %lf %lf", &eig_k, &eig_sr, &eig_si) != 3) MPI_Abort(MPI_COMM_WORLD, 2);
        mu.resize(Nz), mw.resize(Nz), mT.resize(Nz);
        for (int k = 0; k < Nz; k++) {
            double z, a[6];
            if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", &z, a, a + 1, a + 2, a + 3, a + 4, a + 5) != 7 || fabs(z - zz(k)) > 1e-12 * (1 + fabs(z))) {
                if (master()) fprintf(stderr, "verify_exact: mode file '%s' does not match the grid at level %d\n", mode_file.c_str(), k);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            mu[k] = cplx(a[0], a[1]), mw[k] = cplx(a[2], a[3]), mT[k] = cplx(a[4], a[5]);
            mode_norm += norm(mw[k]) * (*get_quad_z())(k);
        }
        fclose(fp);
        if (fabs(eig_k * Lx / (2 * M_PI) - round(eig_k * Lx / (2 * M_PI))) > 1e-9) {
            if (master()) fprintf(stderr, "verify_exact: mode wavenumber %g does not fit Lx = %g\n", eig_k, Lx);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    // Projection of w on the mode: a = sum_z conj(w_hat) W(z) q_z / sum_z |w_hat|^2 q_z, with W(z) the
    // exp(i k x) coefficient of w; a = eps_amp exp(sigma t) for a pure mode
    cplx mode_amplitude(const DTArray &w) {
        cplx s = 0;
        for (int i = w.lbound(0); i <= w.ubound(0); i++)
            for (int j = w.lbound(1); j <= w.ubound(1); j++)
                for (int k = w.lbound(2); k <= w.ubound(2); k++) s += conj(mw[k]) * w(i, j, k) * std::exp(cplx(0, -eig_k * xx(i))) * (*get_quad_z())(k);
        return cplx(pssum(real(s)), pssum(imag(s))) * (2.0 / (Nx * Ny)) / mode_norm;
    }

    void forcing(double t, DTArray &u, DTArray &u_f, DTArray &v, DTArray &v_f, DTArray &w, DTArray &w_f, vector<DTArray *> &tracers, vector<DTArray *> &tracers_f) {
        u_f = 0, v_f = 0, w_f = 0;
        for (auto *tf : tracers_f) *tf = 0;
        if (rot_f != 0) {
            u_f = rot_f * v;
            v_f = -rot_f * u;
        }
        if (manufactured()) {
            const double g = amp(t);
            u_f            = amp_t(t) * (*mU) + g * g * (*mNU) - visco * g * (*mLU);
            w_f            = amp_t(t) * (*mW) + g * g * (*mNW) - visco * g * (*mLW);
        } else if (problem == "gs_noslip") { // u = s(t) U, p = s(t) P with s = sin t
            const double s = sin(t), st = cos(t);
            u_f = st * (*mU) + s * s * (*mNU) - visco * s * (*mLU) + s * (*gPX);
            w_f = st * (*mW) + s * s * (*mNW) - visco * s * (*mLW) + s * (*gPZ);
        } else if (starts_with(problem, "rest_")) {
            w_f = -grav * (*tracers[0]); // the pressure must balance this exactly
        } else if (problem == "poiseuille") {
            u_f = 2 * visco * U0;
        } else if (problem == "womersley") {
            u_f = U0 * cos(omega * t);
        } else if (problem == "ekman") {
            v_f = v_f + rot_f * U0;
        } else if (problem == "igw") {
            w_f           = -grav * (*tracers[0]);
            *tracers_f[0] = (N2 / grav) * w;
        } else if (starts_with(problem, "eig_")) { // hold the base state against diffusion
            u_f = -visco * eUpp(kk) + 0 * ii + 0 * jj;
            if (ntracers) {
                w_f           = -grav * (*tracers[0]);
                *tracers_f[0] = -kappa * eTpp(kk) + (eN2(kk) / grav) * w; // background in T (kh) or in eN2
            }
        }
        // tracer_cheb_bc: with tracer_bc_forcing(), the forcing on boundary points is the boundary
        // value; zero here (set above), so it must reproduce tracer_cheb_mode.
    }

    void analysis(double time, DTArray &u, DTArray &v, DTArray &w, vector<DTArray *> &tracers, DTArray &pressure) {
        DTArray *T = ntracers ? tracers[0] : nullptr;
        if (iter == 0) {
            if (problem == "tg_periodic" || problem == "tg_freeslip" || problem == "tg_rotating") {
                rec.n = 1, rec.lam = -2 * visco * wave_k * wave_k;
                rec.start(time, 1, 0);
            } else if (problem == "inertial") {
                rec.n = 1, rec.M[0][0] = cplx(0, -rot_f);
                rec.start(time, U0, 0);
            } else if (problem == "igw") {
                const double K2 = wave_k * wave_k + wave_m * wave_m;
                rec.n = 2, rec.lam = -visco * K2, rec.M[0][1] = wave_k / K2, rec.M[1][0] = -N2 * wave_k;
                rec.start(time, U0, 0);
            } else if (problem == "tracer_fourier") {
                rec.n = 1, rec.lam = -kappa * wave_k * wave_k;
                rec.start(time, 1, 0);
            } else if (problem == "tracer_advect") {
                rec.n = 1, rec.lam = -kappa * wave_k * wave_k, rec.M[0][0] = cplx(0, -wave_k * U0);
                rec.start(time, 1, 0);
            } else if (problem == "budget_bpe") {
                rec.n = 1, rec.lam = -kappa * pow(M_PI / Lz, 2);
                rec.start(time, 1, 0);
            }
            discrete = rec.n > 0;
            if (starts_with(problem, "mapped_metric")) compute_metric();
        } else if (discrete) {
            rec.step(time);
        }
        continuous(time, *cu, *cv, *cw, T ? cT : nullptr);
        if (discrete) fields_from(rec.x(0), rec.x(1), *ru, *rv, *rw, T ? rT : nullptr);
        DTArray &pu = discrete ? *ru : *cu, &pv = discrete ? *rv : *cv, &pw = discrete ? *rw : *cw;

        double vscale = fmax(psmax(max(abs(*cu))), fmax(psmax(max(abs(*cv))), psmax(max(abs(*cw)))));
        if (vscale == 0) vscale = 1;
        const double err_vel  = fmax(psmax(max(abs(u - pu))), fmax(psmax(max(abs(v - pv))), psmax(max(abs(w - pw))))) / vscale;
        const double cont_vel = fmax(psmax(max(abs(u - *cu))), fmax(psmax(max(abs(v - *cv))), psmax(max(abs(w - *cw))))) / vscale;
        double       err_T = 0, cont_T = 0;
        if (T) {
            double tscale = psmax(max(abs(*cT)));
            if (tscale == 0) tscale = 1;
            err_T  = psmax(max(abs(*T - (discrete ? *rT : *cT)))) / tscale;
            cont_T = psmax(max(abs(*T - *cT))) / tscale;
        }
        vector<pair<string, double>> extra; // problem-specific columns
        if (starts_with(problem, "mapped_metric")) {
            extra = {
                {"Metric_zz", metric_zz},
                {"Metric_xz", metric_xz},
                {"Deriv_x",   deriv_x  },
                {"Deriv_z",   deriv_z  }
            };
        } else if (starts_with(problem, "noslip_")) { // max |du/dx + dw/dz|
            S_EXP ex[3];
            find_expansion(grid_type, ex, "u");
            gradient_op->setup_array(&u, ex[0], ex[1], ex[2]);
            gradient_op->get_dx(t1);
            find_expansion(grid_type, ex, "w");
            gradient_op->setup_array(&w, ex[0], ex[1], ex[2]);
            gradient_op->get_dz(t1, true);
            extra = {
                {"Div", psmax(max(abs(*t1))) / vscale}
            };
        } else if (problem == "budget_energy") {
            dissipation(*t1, u, v, w, gradient_op, grid_type, Nx, Ny, Nz, visco);
            enstrophy_density(*t2, u, v, w, gradient_op, grid_type, Nx, Ny, Nz);
            extra = {
                {"KE",   0.5 * pssum(sum((u * u + v * v + w * w) * (*quad3)))},
                {"Diss", pssum(sum((*t1) * (*quad3)))                        },
                {"Enst", pssum(sum((*t2) * (*quad3)))                        }
            };
        } else if (starts_with(problem, "eig_")) { // mode amplitude and perturbation energy
            const cplx a = mode_amplitude(w);
            *t1          = (u - eU(kk)) * (u - eU(kk)) + v * v + w * w;
            extra        = {
                       {"Amp_re",  real(a)                           },
                       {"Amp_im",  imag(a)                           },
                       {"Pert_KE", 0.5 * pssum(sum((*t1) * (*quad3)))}
            };
        } else if (problem == "gs_noslip") { // absolute errors; pressure with means removed
            const double s = sin(time);
            S_EXP        ex[3];
            find_expansion(grid_type, ex, "u");
            gradient_op->setup_array(&u, ex[0], ex[1], ex[2]);
            gradient_op->get_dx(t1);
            find_expansion(grid_type, ex, "w");
            gradient_op->setup_array(&w, ex[0], ex[1], ex[2]);
            gradient_op->get_dz(t1, true); // div u
            *t2               = (u - s * (*mU)) * (u - s * (*mU)) + (w - s * (*mW)) * (w - s * (*mW));
            const double vol  = pssum(sum(*quad3));
            const double pnum = pssum(sum(pressure * (*quad3))) / vol, pex = s * pssum(sum((*gP) * (*quad3))) / vol;
            *cu   = (pressure - pnum) - (s * (*gP) - pex); // cu is rebuilt at every call
            extra = {
                {"Vel_Linf", sqrt(psmax(max(*t2)))                           },
                {"Vel_L2",   sqrt(pssum(sum((*t2) * (*quad3))) / vol)        },
                {"Div_Linf", psmax(max(abs(*t1)))                            },
                {"Div_L2",   sqrt(pssum(sum((*t1) * (*t1) * (*quad3))) / vol)},
                {"P_Linf",   psmax(max(abs(*cu)))                            },
                {"P_L2",     sqrt(pssum(sum((*cu) * (*cu) * (*quad3))) / vol)}
            };
            if (time >= final_time * (1 - 1e-9)) write_profile(*t1, *t2, *cu);
        } else if (problem == "budget_bpe") { // rho_0 = 1; d BPE/dt = g kappa A (rho(0) - rho(H))
            double bpe = 0;
            compute_Background_PE(bpe, *T, *quad3, Nx, Ny, Nz, Lx, Ly, Lz, grav, 1.0, iter);
            const double pe   = pssum(sum(grav * (1 + *T) * ((*zgrid) - MinZ) * (*quad3)));
            const double rate = grav * kappa * Lx * Ly * 2 * a0 * exp(-kappa * pow(M_PI / Lz, 2) * time);
            extra             = {
                            {"BPE",      bpe },
                            {"PE",       pe  },
                            {"BPE_rate", rate}
            };
        }
        if (master()) {
            string header, line;
            add_diagnostic("Iter", iter, header, line);
            add_diagnostic("Time", time, header, line);
            add_diagnostic("Err_vel", err_vel, header, line);
            add_diagnostic("Err_T", err_T, header, line);
            add_diagnostic("Cont_vel", cont_vel, header, line);
            add_diagnostic("Cont_T", cont_T, header, line);
            for (auto &c : extra) add_diagnostic(c.first, c.second, header, line);
            write_diagnostics(header, line, iter, false);
        }
        if (max_error > 0 && fmax(err_vel, err_T) > max_error) {
            if (master()) fprintf(stderr, "verify_exact (%s): FAIL at t = %g: Err_vel %g, Err_T %g > max_error %g\n", problem.c_str(), time, err_vel, err_T, max_error);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        iter++;
    }

    userControl() : xx(split_range(Nx)), yy(Ny), zz(Nz), gradient_op(0), discrete(false) {
        compute_quadweights(size_x(), size_y(), size_z(), length_x(), length_y(), length_z(), type_x(), type_y(), type_z());
        automatic_grid(MinX, MinY, MinZ, &xx, &yy, &zz);
        ru = alloc(), rv = alloc(), rw = alloc(), rT = alloc(), cu = alloc(), cv = alloc(), cw = alloc(), cT = alloc();
        t1 = alloc(), t2 = alloc(), quad3 = alloc(), zgrid = alloc();
        *quad3 = (*get_quad_x())(ii) * (*get_quad_y())(jj) * (*get_quad_z())(kk);
        // physical z: the automatic grid, or (mapped) the Chebyshev levels stretched over the bottom
        for (int i = zgrid->lbound(0); i <= zgrid->ubound(0); i++)
            for (int j = zgrid->lbound(1); j <= zgrid->ubound(1); j++)
                for (int k = zgrid->lbound(2); k <= zgrid->ubound(2); k++) {
                    const double h    = is_mapped() ? bottom(xx(i)) : 0;
                    (*zgrid)(i, j, k) = is_mapped() ? MinZ + h + (Lz - h) * (0.5 - 0.5 * cos(M_PI * k / (Nz - 1))) : zz(k);
                }
        mU = mW = mNU = mNW = mLU = mLW = 0;
        if (manufactured() || problem == "budget_energy") {
            DTArray *d[10];
            for (auto &p : d) p = alloc();
            // u = psi_z, w = -psi_x; derivatives needed for advection and the Laplacian
            psi_deriv(0, 1, *d[0]); // u
            psi_deriv(1, 0, *d[1]); // -w
            psi_deriv(1, 1, *d[2]); // u_x = -w_z
            psi_deriv(0, 2, *d[3]); // u_z
            psi_deriv(2, 0, *d[4]); // -w_x
            psi_deriv(2, 1, *d[5]);
            psi_deriv(0, 3, *d[6]);
            psi_deriv(3, 0, *d[7]);
            psi_deriv(1, 2, *d[8]);
            mU = alloc(), mW = alloc(), mNU = alloc(), mNW = alloc(), mLU = alloc(), mLW = alloc();
            *mU  = *d[0];
            *mW  = -(*d[1]);
            *mNU = (*mU) * (*d[2]) + (*mW) * (*d[3]);       // u u_x + w u_z
            *mNW = (*mU) * (-(*d[4])) + (*mW) * (-(*d[2])); // u w_x + w w_z
            *mLU = *d[5] + *d[6];                           // lap u = psi_xxz + psi_zzz
            *mLW = -(*d[7] + *d[8]);                        // lap w = -(psi_xxx + psi_xzz)
            for (auto &p : d) delete p;
        }
        if (starts_with(problem, "eig_")) {
            eU.resize(Nz), eUpp.resize(Nz), eT.resize(Nz), eTpp.resize(Nz), eN2.resize(Nz);
            eU = 0, eUpp = 0, eT = 0, eTpp = 0, eN2 = 0;
            for (int k = 0; k < Nz; k++) {
                const double z = zz(k);
                if (problem == "eig_ts") { // plane Poiseuille
                    eU(k) = 1 - z * z, eUpp(k) = -2;
                } else if (problem == "eig_kh") { // thesis section 3.2: tanh shear layer and density
                    const double s = (z - 0.5) / 0.1, c2 = 1 / pow(cosh(s), 2);
                    eU(k) = 0.1808314132 * tanh(s), eUpp(k) = -2 * 0.1808314132 / 0.01 * tanh(s) * c2; // Ri_min = 0.15
                    eT(k) = -0.005 * tanh(s), eTpp(k) = 2 * 0.005 / 0.01 * tanh(s) * c2;
                } else if (starts_with(problem, "eig_rb")) { // unstable, |N^2| = Ra nu kappa / H^4
                    eN2(k) = -Ra * visco * kappa;
                } else if (problem == "eig_imode") { // pycnocline
                    eN2(k) = 1 + 4 / pow(cosh((z - 0.75) / 0.1), 2);
                }
            }
            read_mode();
        }
        gPX = gPZ = gP = 0;
        if (problem == "gs_noslip") {
            // Guermond & Shen (2003): u = pi sin(2 pi z) sin^2(pi x) s(t), w = -pi sin(2 pi x) sin^2(pi z) s(t),
            // p = cos(pi x) sin(pi z) s(t). Stored: the spatial parts of u, w, (u.grad)u / s^2, lap u, p, grad p.
            mU = alloc(), mW = alloc(), mNU = alloc(), mNW = alloc(), mLU = alloc(), mLW = alloc(), gPX = alloc(), gPZ = alloc(), gP = alloc();
            const double P = M_PI;
            for (int i = mU->lbound(0); i <= mU->ubound(0); i++)
                for (int j = mU->lbound(1); j <= mU->ubound(1); j++)
                    for (int k = mU->lbound(2); k <= mU->ubound(2); k++) {
                        const double x = xx(i), z = zz(k);
                        const double a = pow(sin(P * x), 2), a1 = P * sin(2 * P * x), a2 = 2 * P * P * cos(2 * P * x);
                        const double b = sin(2 * P * z), b1 = 2 * P * cos(2 * P * z), b2 = -4 * P * P * sin(2 * P * z);
                        const double c = sin(2 * P * x), c1 = 2 * P * cos(2 * P * x), c2 = -4 * P * P * sin(2 * P * x);
                        const double d = pow(sin(P * z), 2), d1 = P * sin(2 * P * z), d2 = 2 * P * P * cos(2 * P * z);
                        (*mU)(i, j, k)  = P * a * b;
                        (*mW)(i, j, k)  = -P * c * d;
                        (*mNU)(i, j, k) = P * P * (a * a1 * b * b - c * d * a * b1);  // u u_x + w u_z
                        (*mNW)(i, j, k) = P * P * (-a * b * c1 * d + c * c * d * d1); // u w_x + w w_z
                        (*mLU)(i, j, k) = P * (a2 * b + a * b2);
                        (*mLW)(i, j, k) = -P * (c2 * d + c * d2);
                        (*gP)(i, j, k)  = cos(P * x) * sin(P * z);
                        (*gPX)(i, j, k) = -P * sin(P * x) * sin(P * z);
                        (*gPZ)(i, j, k) = P * cos(P * x) * cos(P * z);
                    }
        }
    }
};

// Grid, physics and default time settings of each problem.
bool configure(const string &p) {
    const double twopi = 2 * M_PI;
    string       tx = "FOURIER", ty = "FOURIER", tz = "FOURIER";
    Nx = 32, Ny = 1, Nz = 32, Lx = Lz = twopi, Ly = 1, MinX = MinY = MinZ = 0;
    if (p == "tg_periodic") {
        visco = 0.05;
    } else if (p == "tg_freeslip") {
        tx = tz = "FREE_SLIP", Lx = Lz = M_PI, visco = 0.05;
    } else if (p == "tg_rotating") {
        Nx = Ny = 16, Nz = 4, Ly = twopi, visco = 0.05, rot_f = 1.3;
    } else if (p == "mms_steady" || p == "mms_linear") {
        visco = 0.05, growth_b = (p == "mms_linear") ? 0.5 : 0.0;
    } else if (p == "mms_osc") { // amplitude 1 + 0.5 sin(2 pi t): forcing time levels in every term
        visco = 0.05, growth_b = 0.5, omega = twopi;
    } else if (p == "eig_ts") { // Tollmien-Schlichting wave, Re = 1e4, alpha = 1 (Orszag 1971; Fischer 1997)
        Nx = 16, Nz = 97, tz = "NO_SLIP", MinZ = -1, Lz = 2, visco = 1e-4, eps_amp = 1e-5;
    } else if (p == "eig_stokes") { // least-damped channel mode, no flow
        Nx = 8, Nz = 33, tz = "NO_SLIP", MinZ = -1, Lz = 2, visco = 1;
    } else if (p == "eig_kh") { // Kelvin-Helmholtz, thesis section 3.2, one wavelength 2 pi / 2.38434
        Lx = twopi / 2.38434, Lz = 1, Nx = 64, Nz = 128, tz = "FREE_SLIP", visco = kappa = 1e-6, grav = 9.81, ntracers = 1;
    } else if (p == "eig_rb_ff" || p == "eig_rb_rr") { // Rayleigh-Benard, unit depth, Pr = 1, fixed-temperature walls
        const bool ff = (p == "eig_rb_ff");
        Lx = twopi / (ff ? M_PI / sqrt(2.0) : 3.117), Lz = 1, Nx = 16, Nz = ff ? 32 : 33, tz = ff ? "FREE_SLIP" : "NO_SLIP";
        visco = kappa = 0.01, ntracers = 1;
    } else if (p == "eig_imode") { // inviscid internal mode 1 over a pycnocline, k = 2 pi
        Lx = 1, Lz = 1, Nx = 16, Nz = 65, tz = "NO_SLIP", visco = kappa = 0, ntracers = 1;
    } else if (p == "gs_noslip") { // no-slip walls at z = 0 and 1, wall-normal pressure gradient; p has period 2 in x
        Nx = 64, Nz = 48, Lx = 2, Lz = 1, tz = "NO_SLIP", visco = 1;
    } else if (p == "poiseuille") {
        Nx = 8, Nz = 33, tz = "NO_SLIP", MinZ = -1, Lz = 2, visco = 0.1;
    } else if (p == "womersley") {
        Nx = 4, Nz = 65, tz = "NO_SLIP", MinZ = -1, Lz = 2, visco = 0.01, omega = twopi;
    } else if (p == "ekman") {
        Nx = 4, Nz = 65, tz = "NO_SLIP", MinZ = -1, Lz = 2, visco = 0.01, rot_f = 1;
    } else if (p == "inertial") {
        Nx = Nz = 4, rot_f = twopi;
    } else if (p == "igw") {
        visco = kappa = 0.01, N2 = 4, ntracers = 1, U0 = 0.5;
    } else if (p == "tracer_fourier" || p == "tracer_advect") {
        Nx = Nz = 16, kappa = 0.05, wave_k = 2, ntracers = 1;
    } else if (p == "tracer_cheb_mode" || p == "tracer_cheb_bc") {
        // Chebyshev x needs Chebyshev z too: a Fourier z trips an assertion in TransWrapper, and a
        // free-slip z stops ESolver ("Multigrid/GMRES solver only supported with Chebyshev vertical").
        // T does not depend on z, consistent with the default Neumann tracer BCs there.
        Nx = 33, Nz = 9, tx = tz = "NO_SLIP", Lx = Lz = 1, kappa = 0.05, ntracers = 1;
    } else if (starts_with(p, "mapped_metric")) { // mapping needs Chebyshev z
        Lx = 2, Lz = 1, Nz = 33, tz = "NO_SLIP", hill = 0.3, gauss_hill = (p == "mapped_metric_gauss");
        if (p == "mapped_metric_fs") tx = "FREE_SLIP";
        else if (p != "mapped_metric" && !gauss_hill) return false;
    } else if (p == "rest_freeslip") { // kappa = 0: with diffusion, sloping walls drive real flow
        Lx = 2, Lz = 1, tz = "FREE_SLIP", visco = 0.01, ntracers = 1;
    } else if (p == "rest_noslip" || p == "rest_mapped" || p == "rest_mapped_f") {
        Lx = 2, Lz = 1, Nz = 33, tz = "NO_SLIP", visco = 0.01, ntracers = 1;
        if (p != "rest_noslip") hill = 0.3;
        if (p == "rest_mapped_f") rot_f = 1;
    } else if (p == "noslip_quad" || p == "noslip_cubic") {
        Nx = 16, Nz = 33, tz = "NO_SLIP", MinZ = -1, Lz = 2, visco = 0.2, psi_px = 0, psi_pz = (p == "noslip_quad") ? 2 : 3;
    } else if (p == "noslip_box_zquad" || p == "noslip_box_xquad") {
        Nx = Nz = 33, tx = tz = "NO_SLIP", MinX = MinZ = -1, Lx = Lz = 2, visco = 0.2;
        psi_px = (p == "noslip_box_zquad") ? 3 : 2, psi_pz = (p == "noslip_box_zquad") ? 2 : 3;
    } else if (p == "budget_energy") {
        Nx = Nz = 64, visco = 0.01;
        f_strength = 0; // the budget must close without the filter's dissipation
    } else if (p == "budget_bpe") {
        Nx = 4, Nz = 64, tz = "FREE_SLIP", Lx = Lz = 1, kappa = 0.01, ntracers = 1;
    } else {
        return false;
    }
    parse_boundary_conditions(tx, ty, tz, intype_x, intype_y, intype_z);
    grid_type[0] = tx, grid_type[1] = ty, grid_type[2] = tz;
    return true;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int    Nx_in, Ny_in, Nz_in;
    double nu_in;
    options_init();
    add_option("problem", &problem, "Which exact solution (see verify_exact.cpp)");
    add_option("Nx", &Nx_in, 0, "Override the problem's Nx (0: problem default)");
    add_option("Ny", &Ny_in, 0, "Override the problem's Ny (0: problem default)");
    add_option("Nz", &Nz_in, 0, "Override the problem's Nz (0: problem default)");
    add_option("final_time", &final_time, 1.0, "Final time");
    add_option("dt_max", &dt_max, 0.01, "Maximum time step");
    add_option("max_error", &max_error, 0.0, "Abort if Err_vel or Err_T exceeds this (0: off)");
    add_option("f_cutoff", &f_cutoff, 0.6, "Filter cut-off frequency");
    add_option("f_order", &f_order, 2.0, "Filter order");
    add_option("f_strength", &f_strength, 20.0, "Filter strength");
    add_option("nu", &nu_in, -1.0, "Override the problem's viscosity (negative: problem default)");
    add_option("dt_pattern", &dt_pattern, "default", "Time-step pattern (see userControl::check_timestep)");
    add_option("dt_scale", &dt_scale, 1.0, "Scale of the patterned steps, in units of dt_max");
    add_option("dt_ratio", &dt_ratio, 1.5, "Step ratio of the grow pattern");
    add_option("mode_file", &mode_file, "mode.txt", "eig_*: the eigenmode file (scripts/in/eigen_modes.py)");
    double eps_in, Ra_in;
    add_option("eps_amp", &eps_in, -1.0, "eig_*: mode amplitude (negative: problem default)");
    add_option("Ra", &Ra_in, 1000.0, "eig_rb_*: Rayleigh number");
    options_parse(argc, argv);
    if (!configure(problem)) {
        if (master()) fprintf(stderr, "verify_exact: unknown problem '%s'\n", problem.c_str());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (Nx_in) Nx = Nx_in;
    if (Ny_in) Ny = Ny_in;
    if (Nz_in) Nz = Nz_in;
    if (nu_in >= 0) visco = nu_in;
    if (nu_in >= 0 && starts_with(problem, "eig_rb")) kappa = nu_in; // Pr = 1
    if (eps_in > 0) eps_amp = eps_in;
    Ra = Ra_in;

    userControl              mycode;
    FluidEvolve<userControl> do_stuff(&mycode);
    do_stuff.initialize();
    if (master()) fprintf(stdout, "verify_exact: %s on %d x %d x %d, dt_max %g, final time %g\n", problem.c_str(), Nx, Ny, Nz, dt_max, final_time);
    do_stuff.do_run(final_time);
    MPI_Finalize();
    return 0;
}
