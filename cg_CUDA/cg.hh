#include "matrix_coo.hh"
#include <cblas.h>
#include <string>


#ifndef __CG_HH__
#define __CG_HH__



class CGSolverSparse{
public:
  CGSolverSparse() = default;
  void init_source_term(double h);
  void read_matrix(const std::string & filename);
  void solve(double * x, dim3 block_size);
  inline int m() const { return m_m; }
  inline int n() const { return m_n; }
  void tolerance(double tolerance) { m_tolerance = tolerance; }

  void solveCG_cuda(double *x, double *r, double *p, double *Ap, double *tmp, double *beta, double *alpha, double *rsnew, double *rsold,  double *temp_scal,dim3  block_size);


protected:
  int m_m{0};
  int m_n{0};
  double * m_b;
  double m_tolerance{1e-10};

private:
  MatrixCOO m_A;
};

#endif /* __CG_HH__ */