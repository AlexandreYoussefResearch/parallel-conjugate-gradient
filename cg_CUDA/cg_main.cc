#include "cg.hh"
#include "cg.cc"
#include <chrono>
#include <iostream>
#include <tuple>

/* -------------------------------------------------------------------------- */
#include <cuda_runtime.h>
/* -------------------------------------------------------------------------- */

typedef std::chrono::high_resolution_clock clk;
typedef std::chrono::duration<double> second;

/*
Implementation of a parallel CUDA CG solver using sparse matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code. 

Author - Alexandre Youssef
*/

static void usage(const std::string & prog_name) {
  std::cerr << prog_name << " <grid_size> <block_size [default: 32]>" << std::endl;
  exit(0);
}

int main(int argc, char ** argv) {

  if (argc < 2) usage(argv[0]);

  // 1D blocks
  dim3 block_size{32, 1};
  if (argc >= 3) {
    try {
      block_size.x = std::stoi(argv[2]);
    } catch(std::invalid_argument &) {
      usage(argv[0]);
    }
  }


 // By default, we use device 0
  
  int dev_id = 0;

  cudaDeviceProp device_prop;
  cudaGetDevice(&dev_id);
  cudaGetDeviceProperties(&device_prop, dev_id);
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                 "threads can use ::cudaSetDevice()"
              << std::endl;
    return -1;
  }

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "cudaGetDeviceProperties returned error code " << error
              << ", line(" << __LINE__ << ")" << std::endl;
    return error;
  } else if (false){
    std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
              << "\" with compute capability " << device_prop.major << "."
              << device_prop.minor << std::endl;
  }


  CGSolverSparse sparse_solver;
  sparse_solver.read_matrix(argv[1]);

  int n = sparse_solver.n();
  double h = 1. / n;
  sparse_solver.init_source_term(h);

  double *x_s;
  cudaMallocManaged(&x_s, n*sizeof(double));
  std::fill(x_s, x_s+n, 0.);

  auto start = clk::now();
  sparse_solver.solve(x_s,block_size);
  auto end = clk::now();

  second time = end - start;

 std::cout << "(" << block_size.x << "x" <<  block_size.y << ") " << n << " Time for CG (CUDA) = "
            << time.count() << " [s]\n";

  cudaFree(x_s);

  return 0;
  
}
