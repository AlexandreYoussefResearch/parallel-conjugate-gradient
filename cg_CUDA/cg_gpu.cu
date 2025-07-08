/* -------------------------------------------------------------------------- */
#include "cg.hh"
#include "matrix_coo.hh"
#include <iostream>
#include <exception>
#include <cublas_v2.h>
#include <cuda_runtime.h>
/* -------------------------------------------------------------------------- */


/* For debugging */
const bool DEBUG = true;



// KERNELS

// Matrix vector product
__global__ void matVec(int* irn, int* jcn, double *a, double* b, double* out, int m_n, int nz, bool is_sym) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nz){
        int i = irn[idx];
        int j = jcn[idx];
        double value = a[idx];
		atomicAdd(&(out[i]), value*b[j]);
		if (i!=j && is_sym){
			atomicAdd(&(out[j]), value*b[i]);
		}
	}
}


// Initialize vector to zero
__global__ void setToZero(double* out, int m_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m_n) {
        out[idx] = 0.0;
    }
}


// Sum of two vectors
__global__ void vecSum(double* a, double* b, double* out, int m_n) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < m_n){
		out[index_x] = b[index_x] + a[index_x];
	}
}	


// Difference of two vectors
__global__ void vecDiff(double* a, double* b, double* out, int m_n) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < m_n){
		out[index_x] = a[index_x] - b[index_x];
	}
}

// Scalar vector product
__global__ void vecProd(double* scalar, double* a, double* out, int m_n) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < m_n){
		out[index_x] = a[index_x] * (*scalar);
	}
}

// Copy elements
__global__ void copy(double* in, double* out, int m_n) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	out[index_x] = in[index_x];
}

// Scalar division
__global__ void scalarDivide(double* num, double* den, double* out) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x == 0) {
		*out = *num / *den;
	}
}



// MAIN FUNCTION
void CGSolverSparse::solveCG_cuda(double *x, double *r, double *p, double *Ap, double *tmp, double *beta, double *alpha, double *rsnew, double *rsold,  double *temp_scal, dim3 block_size){

    // Extract the parameters
	int m_n = m_A.m();
	int nz = m_A.nz();
	bool is_sym = m_A.is_sym();
	int  *irn = m_A.data_irn();
	int *jcn = m_A.data_jcn();
	double *a  = m_A.data_a();

	
	// Define the grid sizes
    dim3 grid_size(ceil((double)nz/(double) block_size.x));
	dim3 grid_size_vec(ceil((double)m_n/(double) block_size.x));
	

	// r = b - A * x; -> x=0
	copy<<<grid_size_vec, block_size>>>(m_b,r, m_n);

  	// p = r;
	copy<<<grid_size_vec, block_size>>>(r,p, m_n);

	// rsold = r' * r;
	cublasHandle_t h;
    cublasCreate(&h);
    cublasDdot(h, m_n, r, 1, r, 1, rsold);


	int k = 0;
	 for (; k < m_n; ++k) { 

		// Ap = A * p;
		setToZero<<<grid_size_vec, block_size>>>(Ap, m_n);
		matVec<<<grid_size, block_size>>>(irn, jcn, a, p, Ap, m_n,nz, is_sym);

		// alpha = rsold / (p' * Ap);
		cublasDdot(h, m_n, p, 1, Ap, 1, temp_scal);
		scalarDivide<<<1, 1>>>(rsold, temp_scal, alpha);
		

		// x = x + alpha * p;
		vecProd<<<grid_size_vec, block_size>>>(alpha, p, tmp, m_n);
		vecSum<<<grid_size_vec, block_size>>>(x, tmp, x, m_n);
		cublasDdot(h, m_n, x, 1, x, 1, temp_scal);
		
		// r = r - alpha * Ap;
		vecProd<<<grid_size_vec, block_size>>>(alpha, Ap, tmp, m_n);
		vecDiff<<<grid_size_vec, block_size>>>(r, tmp, r, m_n);

		// rsnew = r' * r;
		cublasDdot(h, m_n, r, 1, r, 1, rsnew);

		cudaDeviceSynchronize();
   		// if sqrt(rsnew) < 1e-10
    	//   break;
        if (std::sqrt(*rsnew) < m_tolerance)
      			break; 

		// beta = rsnew / rsold
		scalarDivide<<<1, 1>>>(rsnew, rsold, beta);

		// p = r + (rsnew / rsold) * p;
		vecProd<<<grid_size_vec, block_size>>>(beta, p, tmp, m_n);
		vecSum<<<grid_size_vec, block_size>>>(r, tmp,p, m_n);
	
		// rsold = rsnew;
		copy<<<1, 1>>>(rsnew,rsold, 1);
		
		if (DEBUG){
			std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(*rsold) << "\r" << std::flush;
		}
	}

	printf("Total number of iterations = %d \n",k);
	
	if (DEBUG) {
		setToZero<<<grid_size_vec, block_size>>>(r, m_n);
		matVec<<<grid_size, block_size>>>(irn, jcn, a, x,r, m_n,nz, is_sym);

		vecDiff<<<grid_size, block_size>>>(m_b, r, r, m_n);
	
		cublasDdot(h, m_n, r, 1, r, 1, rsnew);
		cublasDdot(h, m_n, m_b, 1, m_b, 1, temp_scal);

		auto res = std::sqrt(*rsnew) / std::sqrt(*temp_scal);

		cublasDdot(h, m_n, x, 1, x, 1, temp_scal);

		auto nx = std::sqrt(*temp_scal);
		std::cout << "\t[STEP " << k << "] residual = " << std::scientific
				<< std::sqrt(*rsold) << ", ||x|| = " << nx
				<< ", ||Ax - b||/||b|| = " << res << std::endl;
  	}
}