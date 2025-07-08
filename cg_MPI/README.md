PHPC - CONJUGATE GRADIENT PROJECT - Alexandre Youssef

HOWTO COMPILE AND RUN : MPI Sparse implementation 
(parallelization of the entire CG algorithm)
=====================

Requirements : 

- a recent compiler (like gcc or intel)
- a cblas library (like openblas or intel MKL)

compile on SCITAS clusters :

```
$ module load gcc openblas
$ module load gcc mvapich2
$ make
```

to run, do :

srun ./cgsolver lap2D_5pt_n100.mtx proc

where lap2D_5pt_n100.mtx is the example matrix provided in the statement and proc the number of processes 


The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format.

The input matrix format is [Matrix Market format (.mtx extension)](https://sparse.tamu.edu/). 