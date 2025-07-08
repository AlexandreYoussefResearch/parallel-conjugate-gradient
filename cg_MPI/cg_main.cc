#include "cg.hh"
#include <chrono>
#include <iostream>

/* -------------------------------------------------------------------------- */
#include <mpi.h>
/* -------------------------------------------------------------------------- */


using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code

Author : Alexandre Youssef
*/

int main(int argc, char ** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]"
              << std::endl;
    return 1;
  }

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int prank, psize;

  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);


  if (psize==1){
     printf("Serial code (p=1) \n");
  }
  else{
    if (prank ==0){
      printf("Running with p = %d processes \n", psize);
    }
  }

  // Read the local matrices
  auto t2 = clk::now();
  CGSolverSparse sparse_solver;
  sparse_solver.read_matrix(argv[1], prank, psize);
  second  elapsed2 = clk::now() - t2;

  if (prank == 0){
    std::cout << "Time for matrix reading  = " << elapsed2.count() << " [s]\n";
  }


  int n = sparse_solver.n();
  int m = sparse_solver.m();
  double h = 1. / n;
  int size;
  int idx_start;

  if (prank < psize-1){
    size = n/psize;
    idx_start = prank*size;
  }

  else{
    size = n - (psize-1)*(n/psize);
    idx_start = (psize-1)*(n/psize);
  }

  sparse_solver.init_source_term(h, size, idx_start);

  std::vector<double> x_s(size);
  std::fill(x_s.begin(), x_s.end(), 0.);

  MPI_Barrier(MPI_COMM_WORLD);
  if (prank == 0){
    std::cout << "Call CG sparse on matrix size (" << m << " x " << n << ")"
            << std::endl;
  }

  auto t1 = clk::now();
  sparse_solver.solve(x_s, prank, psize);
  second  elapsed = clk::now() - t1;

  if (prank == 0){
    std::cout << "Time for CG (sparse solver)  = " << elapsed.count() << " [s]\n";
  }

  MPI_Finalize();

  return 0;
}
