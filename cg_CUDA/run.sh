#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:5:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454-final

module purge
module load gcc cuda
module load gcc openblas

srun ./cgsolver lap2D_5pt_n100.mtx 512