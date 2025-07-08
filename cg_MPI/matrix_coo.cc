#include "matrix_coo.hh"
extern "C" {
#include "mmio.h"
}

void MatrixCOO::read(const std::string & fn, int prank, int psize) {
  int nz;

  // For the matrix truncation (scaling analysis) mn_trucated = mn/div
  int div = 1;
  // -----

  int ret_code;
  MM_typecode matcode;
  FILE * f;

  if ((f = fopen(fn.c_str(), "r")) == NULL) {
    printf("Could not open matrix");
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Matrix is sparse
  if (not(mm_is_matrix(matcode) and mm_is_coordinate(matcode))) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &m_m, &m_n, &nz)) != 0) {
    exit(1);
  }

  // For the scaling analysis
  m_m = m_m/div;
  m_n = m_n/div;


  /* reserve memory for matrices Filling the entire matrix*/
  irn.resize(2*nz);
  jcn.resize(2*nz);
  a.resize(2*nz);

  /*  NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  int nz_loc = 0;

  int stride = m_n/psize;
  m_is_sym = mm_is_symmetric(matcode);

  for (int i = 0; i < nz; i++) {
    int I, J;
    double a_;

    fscanf(f, "%d %d %lg\n", &I, &J, &a_);
    I--; /* adjust from 1-based to 0-based */
    J--;

    // Condition for the matrix truncation (scaling analysis)
    if (I < m_n && J < m_n){
      if ((I >= prank*stride) && I< ((prank+1)*stride) && prank < psize-1){
            irn[nz_loc] = I;
            jcn[nz_loc] = J;
            a[nz_loc] = a_;
            nz_loc++;
      }

      else if ((I >= prank*stride) && (prank == psize-1)){
            irn[nz_loc] = I;
            jcn[nz_loc] = J;
            a[nz_loc] = a_;
            nz_loc++;
      }

      if (m_is_sym && (I!=J) && (J >= prank*stride) && J< ((prank+1)*stride) && prank < psize-1){
            irn[nz_loc] = J;
            jcn[nz_loc] = I;
            a[nz_loc] = a_;
            nz_loc++;
      }

      else if (m_is_sym && (I!=J) && (J >= prank*stride) && (prank == psize-1)){
            irn[nz_loc] = J;
            jcn[nz_loc] = I;
            a[nz_loc] = a_;
            nz_loc++;
      }
    }
  }
  
  /* Resize matrix memory */
  irn.resize(nz_loc);
  jcn.resize(nz_loc);
  a.resize(nz_loc);

  if (f != stdin) {
    fclose(f);
  }
}
