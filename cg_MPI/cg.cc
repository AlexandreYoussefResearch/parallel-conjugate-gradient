#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>

/* -------------------------------------------------------------------------- */
#include <mpi.h>
/* -------------------------------------------------------------------------- */


/* For debugging */
const double NEARZERO = 1.0e-14;
const bool DEBUG = true;


/*
Sparse version of the cg solver : MPI implementation
*/
void CGSolverSparse::solve(std::vector<double> & x, int prank, int psize) {

  int size;
  if (prank < psize-1){
    size = m_n/psize;
  }
  else{
     size = m_n - (psize-1)*(m_n/psize);
  }
 

  std::vector<double> r(size);
  std::vector<double> p(size);
  std::vector<double> Ap(size);
  std::vector<double> tmp(size);
  std::vector<double> p_gen(m_n);
 

  // r = b - A * x; suppose x = 0
  r = m_b;

  // p = r;
  p = r;

  // rsold = r' * r;
  std::fill(Ap.begin(), Ap.end(), 0.);

  double rsold_loc = cblas_ddot(size, r.data(), 1, r.data(), 1);
  double rsold_gen;
  MPI_Allreduce(&rsold_loc, &rsold_gen, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


  // Vectors for the AllGatherv() operation
  std::vector<int> recv_count(psize);
  std::vector<int> displs(psize);
  displs[0] = 0;
  recv_count[0] = m_n/psize;
  for (int i=1; i<psize-1;i++){
    displs[i] = i*(m_n/psize);
    recv_count[i] = m_n/psize;
  }
  displs[psize-1] =  (psize-1)*(m_n/psize) ;
  recv_count[psize-1] = m_n - (psize-1)*(m_n/psize) ;


  // Main loop
  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k) {

    MPI_Allgatherv(p.data(), size, MPI_DOUBLE,
                   p_gen.data(), recv_count.data(), displs.data(),
                  MPI_DOUBLE,MPI_COMM_WORLD);

    // Ap = A * p;
    m_A.mat_vec(p_gen, Ap, prank, displs);

    // alpha = rsold / (p' * Ap);
    double prod_loc = cblas_ddot(size, p.data(), 1, Ap.data(), 1);
    double prod_gen;
    MPI_Allreduce(&prod_loc, &prod_gen, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double alpha = rsold_gen / std::max(prod_gen, rsold_gen * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(size, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(size, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    auto rsnew_loc = cblas_ddot(size, r.data(), 1, r.data(), 1);
    double rsnew_gen;
    MPI_Allreduce(&rsnew_loc, &rsnew_gen, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew_gen) < m_tolerance)
      break; // Convergence test

    double beta = rsnew_gen / rsold_gen;



    // p = r + (rsnew / rsold) * p;
    tmp = r;
    cblas_daxpy(size, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;

    // rsold = rsnew;
    rsold_gen = rsnew_gen;

    if (DEBUG && (prank==0)) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold_gen) << "\r" << std::flush;
    }
  }


  if (DEBUG) {

    std::vector<double> x_gen(m_n);

    MPI_Allgatherv(x.data(), size, MPI_DOUBLE,
                   x_gen.data(), recv_count.data(), displs.data(),
                   MPI_DOUBLE,MPI_COMM_WORLD);

    m_A.mat_vec(x_gen, r,prank, displs);

    cblas_daxpy(size, -1., m_b.data(), 1, r.data(), 1);


    double r_prod_loc = cblas_ddot(size, r.data(), 1, r.data(), 1);
    double r_prod_gen;

    MPI_Reduce(&r_prod_loc, &r_prod_gen,1,
                  MPI_DOUBLE,MPI_SUM,0, MPI_COMM_WORLD);

    double b_prod_loc = cblas_ddot(size, m_b.data(), 1, m_b.data(), 1);
    double b_prod_gen;
    MPI_Reduce(&b_prod_loc, &b_prod_gen,1,
                  MPI_DOUBLE,MPI_SUM, 0,MPI_COMM_WORLD);


    if (prank == 0){
      double res = std::sqrt(r_prod_gen)/std::sqrt(b_prod_gen);
      double nx = std::sqrt(cblas_ddot(m_n, x_gen.data(), 1, x_gen.data(), 1));
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold_gen) << ", ||x|| = " << nx
                << ", ||Ax - b||/||b|| = " << res << std::endl;

    }
  }
}

void CGSolverSparse::read_matrix(const std::string & filename, int prank, int psize) {
  m_A.read(filename, prank, psize);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void Solver::init_source_term(double h, int size, int idx_start) {
  m_b.resize(size);

  int iterator = 0;
  for (int i = idx_start; i < idx_start+size; i++) {
    m_b[iterator] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
    iterator++;
  }
}
