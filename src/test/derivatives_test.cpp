#include "mpi.h"
#include <gtest/gtest.h>
#include <blitz/blitz.h>
#include <blitz/array.h>
#include <memory>
#include <string>

#include "grad.hpp"
#include "Parformer.hpp"

#include "NSIntegrator.hpp"

#include "Science.hpp"

using std::unique_ptr;
using std::cout;
using namespace blitz;
using namespace TArrayn;
using namespace Transformer;

// The fixture for testing Derivatives.
class DerivativesTest : public testing::Test {
 protected:
    const int Nx = 64;
    const int Nz = 64;
    const double Lx = 1.0e3;
    const double Lz = 160.0;
    const double reltol_eps = 1.0e-12;
    double resid = std::numeric_limits<double>::max();

    firstIndex ii;
    secondIndex jj;
    thirdIndex kk;

  unique_ptr<DTArray> x_, z_;
    unique_ptr<DTArray> psi_, u_, w_, vort_;
    unique_ptr<DTArray> unum_, wnum_, vortnum_, psixnum_, psiznum_, test_;
    unique_ptr<Grad> grad_;

  DerivativesTest() {
    x_ = make_unique<DTArray>(Nx, 1, Nz);
    z_ = make_unique<DTArray>(Nx, 1, Nz);

    grad_ = make_unique<Grad>(Nx, 1, Nz, S_EXP::FOURIER, S_EXP::FOURIER, S_EXP::CHEBY); // or whatever expansions, for now.

    psi_ = make_unique<DTArray>(Nx, 1, Nz);
    vort_ = make_unique<DTArray>(Nx, 1, Nz);
    psixnum_ = make_unique<DTArray>(Nx, 1, Nz);
    psiznum_ = make_unique<DTArray>(Nx, 1, Nz);
    unum_ = make_unique<DTArray>(Nx, 1, Nz);
    wnum_ = make_unique<DTArray>(Nx, 1, Nz);
    vortnum_ = make_unique<DTArray>(Nx, 1, Nz);
    test_ = make_unique<DTArray>(Nx, 1, Nz);
    u_ = make_unique<DTArray>(Nx, 1, Nz);
    w_ = make_unique<DTArray>(Nx, 1, Nz); 
  }

  ~DerivativesTest() override {
     // Nada for destructor, 'unique_ptr' self-manages its memory.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).

     // Define analytical answers.

    DTArray &xx = *x_, &zz = *z_;
    DTArray &psi = *psi_, &vort = *vort_,
        &u = *u_, &w = *w_;

    const double dx = Lx / Nx;

    xx = (ii + 0.5) * dx  + 0.0*kk;

    zz = 0.5*(cos(M_PI*kk/(Nz-1))+1.0)*Lz;


    psi = sin(2.0*M_PI*xx/Lx) * sin(2.0*M_PI*zz/Lz);
    vort = -(-(4.0*M_PI*M_PI/Lx/Lx + 4.0*M_PI*M_PI/Lz/Lz))*psi;

    u = -(2.0*M_PI/Lz) * sin(2.0*M_PI*xx/Lx) * cos(2.0*M_PI*zz/Lz);
    w = (2.0*M_PI/Lx) * cos(2.0*M_PI*xx/Lx) * sin(2.0*M_PI*zz/Lz);
  }
};


// Assert first derivate os correct.
TEST_F(DerivativesTest, AssertFirstDerivativesAreCorrect) {
    grad_->setup_array(psi_.get(), S_EXP::FOURIER, S_EXP::FOURIER, S_EXP::CHEBY);
    grad_->set_jac(Dimension::thirdDim, Dimension::thirdDim, 2.0/Lz);
    grad_->set_jac(Dimension::firstDim, Dimension::firstDim, (2.0*M_PI)/Lx);

    grad_->get_dx(psixnum_.get());
    grad_->get_dz(psiznum_.get());

    const DTArray &xx = *x_, &zz = *z_;
    DTArray &psi = *psi_, &vort = *vort_,
        &unum = *unum_, &wnum = *wnum_, &test = *test_;

    // Use streamfunction definition.
    wnum = *psixnum_;
    unum = -(*psiznum_);

    test = unum - *u_;
    resid = sqrt(blitz::sum(test*test))/(Nz*Nx);

    EXPECT_NEAR(sqrt(blitz::sum(unum*unum))/(Nz*Nx), sqrt(blitz::sum(*u_*(*u_)))/(Nz*Nx), reltol_eps)
        << "x-derivative (Fourier) residual: " << resid << '\n';

    test = wnum - *w_;
    resid = sqrt(blitz::sum(test*test))/(Nz*Nx);

    EXPECT_NEAR(sqrt(blitz::sum(wnum*wnum))/(Nz*Nx), sqrt(blitz::sum(*w_*(*w_)))/(Nz*Nx), reltol_eps)
      << "z-derivative (Chebyshev) residual: " << resid << '\n';
}

TEST_F(DerivativesTest, AssertSecondDerivativesAreCorrect) {
    grad_->setup_array(psi_.get(), S_EXP::FOURIER, S_EXP::FOURIER, S_EXP::CHEBY);
    grad_->set_jac(Dimension::thirdDim, Dimension::thirdDim, 2.0/Lz);
    grad_->set_jac(Dimension::firstDim, Dimension::firstDim, (2.0*M_PI)/Lx);

    grad_->get_dx(psixnum_.get());
    grad_->get_dz(psiznum_.get());

    const DTArray &xx = *x_, &zz = *z_;
    DTArray &psi = *psi_, &vort = *vort_, &vortnum = *vortnum_,
        &unum = *unum_, &wnum = *wnum_, &test = *test_;

    // Use streamfunction definition.
    wnum = *psixnum_;
    unum = -(*psiznum_);

    grad_->setup_array(&wnum, S_EXP::FOURIER, S_EXP::FOURIER, S_EXP::CHEBY);
    grad_->get_dx(vortnum_.get());
    (*vortnum_) *= -1.0;

    grad_->setup_array(&unum, S_EXP::FOURIER, S_EXP::FOURIER, S_EXP::CHEBY);
    grad_->get_dz(vortnum_.get(), true);

    test = vortnum - vort;
    resid = sqrt(blitz::sum(test*test))/(Nz*Nx);

    EXPECT_NEAR(sqrt(blitz::sum(vortnum*vortnum))/(Nz*Nx), sqrt(blitz::sum(*vort_*(*vort_)))/(Nz*Nx), reltol_eps)
        << "Vorticity (Chebyshev+Fourier) residual: " << resid << '\n';
}



// Get an MPI-capable testing binary out.
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);

    int ret = RUN_ALL_TESTS();

    MPI_Finalize();
    return ret;
}