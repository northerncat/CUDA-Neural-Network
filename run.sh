#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

# Add command line arguments as necessary
mpirun -n 4 ./main

#For profiling run, uncomment the line below
#MV2_USE_CUDA=1 mpirun -np 4 nvprof --output-profile profile.%p.nvprof ./main
