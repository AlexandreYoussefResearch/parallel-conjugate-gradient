#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454-final

module purge
module load gcc cuda
module load gcc openblas

rm output.txt 

echo "dim N time" >> output.txt

for block_size in 2 4 8 16 32 64 128 256 512 1024; do
    srun ./cgsolver lap2D_5pt_n100.mtx $block_size >> output.txt
done
