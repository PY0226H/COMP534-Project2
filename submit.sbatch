#!/bin/bash
#SBATCH --export=ALL
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --time=00:05:00 
#SBATCH --partition=interactive
#SBATCH --reservation=comp422

export KMP_AFFINITY=verbose # just to print the bindings
for ((i = 1; i <= 32; i++)); do 
  OMP_PROC_BIND=spread time ./lu-omp 8000 $i
done
