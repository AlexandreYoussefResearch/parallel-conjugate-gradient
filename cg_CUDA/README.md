PHPC - CONJUGATE GRADIENT PROJECT - Alexandre Youssef

HOWTO COMPILE AND RUN : CUDA Sparse implementation
=====================

Requirements : 

- a recent compiler (like gcc or intel)
- a cblas library (like openblas or intel MKL)

compile on SCITAS clusters :

```
$ module load gcc openblas
$ module load gcc cuda
$ make
```

to run, do :

srun ./cgsolver lap2D_5pt_n100.mtx block_size

where lap2D_5pt_n100.mtx is the example matrix provided in the statement and block_size the chosen block_size (<1024)




The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format.

The input matrix format is [Matrix Market format (.mtx extension)](https://sparse.tamu.edu/). 