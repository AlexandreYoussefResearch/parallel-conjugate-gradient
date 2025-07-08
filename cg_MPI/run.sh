#!/bin/bash
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454-final
#SBATCH -n 5
#SBATCH -N 1
#SBATCH -t 0-2:00

module purge
module load gcc mvapich2
module load gcc openblas

srun -n 5 cgsolver lap2D_5pt_n100.mtx
