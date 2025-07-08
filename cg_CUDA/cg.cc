#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "cg.hh"

/*
Sparse version of the cg solver
*/

void CGSolverSparse::solve(double *x, dim3 block_size) {

  // Allocate device memory
  double *r;
  double *p;
  double *Ap;
  double *tmp;

	cudaMallocManaged(&r, m_n * sizeof(double));
	cudaMallocManaged(&p, m_n * sizeof(double));
	cudaMallocManaged(&Ap, m_n * sizeof(double));
	cudaMallocManaged(&tmp, m_n * sizeof(double));


	double* beta;
  double* alpha;
	double* rsnew;
	double* rsold;
	double* temp_scal;
	cudaMallocManaged(&beta, sizeof(double));
	cudaMallocManaged(&alpha, sizeof(double));
	cudaMallocManaged(&rsnew, sizeof(double));
	cudaMallocManaged(&rsold, sizeof(double));
	cudaMallocManaged(&temp_scal, sizeof(double));

	// run the main function
	solveCG_cuda(x, r, p, Ap, tmp, beta, alpha, rsnew, rsold,  temp_scal, block_size);
  cudaDeviceSynchronize();


  // freeing the memory
  cudaFree(r);
  cudaFree(p);
  cudaFree(Ap);
  cudaFree(tmp);
  cudaFree(beta);
  cudaFree(alpha);
  cudaFree(rsnew);
  cudaFree(rsold);
  cudaFree(temp_scal);
  cudaFree(m_b);
  cudaFree(m_A.irn);
  cudaFree(m_A.jcn);
  cudaFree(m_A.a);
}


void CGSolverSparse::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void CGSolverSparse::init_source_term(double h) {
  cudaMallocManaged(&m_b, m_n*sizeof(double));

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
