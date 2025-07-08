#include <algorithm>
#include <string>
#include <vector>

/* -------------------------------------------------------------------------- */
#include <mpi.h>
/* -------------------------------------------------------------------------- */


#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  inline int nz() const { return irn.size(); }
  inline int is_sym() const { return m_is_sym; }

  void read(const std::string & filename, int prank, int psize);

  void mat_vec(const std::vector<double> & x, std::vector<double> & y, int prank, const std::vector<int> & displs) {
    std::fill_n(y.begin(), y.size(), 0.);


    for (size_t z = 0; z < irn.size(); ++z) {
      auto i = irn[z];
      auto j = jcn[z];
      auto a_ = a[z];

      auto i2 = i- displs[prank];

      y[i2] += a_ * x[j];

      
    }
  }

  std::vector<int> irn;
  std::vector<int> jcn;
  std::vector<double> a;

private:
  int m_m{0};
  int m_n{0};
  bool m_is_sym{false};
};

#endif // __MATRIX_COO_H_
