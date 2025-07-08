#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;

  __host__ __device__ int m() const { return m_m; }
  __host__ __device__ int n() const { return m_n; }

  __host__ __device__ int nz() const { return nz_f;}
  __host__ __device__ int is_sym() const { return m_is_sym; }

  void read(const std::string & filename);


  __host__ __device__ int* data_irn() { return irn; }
  __host__ __device__ int* data_jcn() { return jcn; }
  __host__ __device__ double* data_a() { return a; }

  int *irn;
  int *jcn;
  double *a;
  int nz_f;

private:
  int m_m{0};
  int m_n{0};
  bool m_is_sym{false};
};

#endif // __MATRIX_COO_H_
