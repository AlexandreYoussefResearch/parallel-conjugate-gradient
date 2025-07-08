# Parallelized Conjugate Gradient Solver

This project implements and benchmarks two parallel versions of the **Conjugate Gradient (CG)** method for solving large, sparse, symmetric positive-definite linear systems of the form *Ax = b*.

Developed as part of a High Performance Computing course at **EPFL (École Polytechnique Fédérale de Lausanne)**, the project aims to assess performance tradeoffs between **multi-core CPU parallelism** (MPI) and **GPU acceleration** (CUDA).

## Project Overview

The CG method is implemented from scratch in C++, based on a standard iterative algorithm with early stopping on residual norm convergence. The following two versions were developed:

1. **MPI Implementation (CPU)**  
   - Distributed-memory parallelism
   - Benchmarking of strong and weak scaling
   - Evaluation against Amdahl’s and Gustafson’s laws

2. **CUDA Implementation (GPU)**  
   - Handwritten CUDA kernels (except for vector dot-product using `cublasDdot`)
   - Tuning of grid/block sizes
   - Performance profiling using NVIDIA tools

## Matrix Handling

- The solver supports both dense and sparse matrices.
- Sparse matrices are handled using **COO format** (triplet representation).
- Inputs are provided in Matrix Market format, e.g. from [sparse.tamu.edu](https://sparse.tamu.edu).

## Performance & Analysis

- Detailed profiling of each component of the CG loop (SpMV, dot products, vector updates)
- Study of the computational cost breakdown and memory bottlenecks
- Comparison of scalability across MPI ranks and GPU occupancy levels
- Accuracy and convergence tracking relative to a serial baseline

## Context

This project was submitted as the final deliverable for the "Parallel and High Performance Computing" course taught by **Dr. Pablo Antolín** at EPFL. The implementation and results were presented in a live oral defense session.

