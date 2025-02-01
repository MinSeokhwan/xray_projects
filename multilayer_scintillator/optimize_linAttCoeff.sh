#!/bin/bash

#SBATCH -o /home/gridsan/smin/python_scripts/multilayer_scintillator/slurm/optimize_linAttCoeff.log-%j
#SBATCH -n 1

module load anaconda/2023a
export OMP_NUM_THREADS=1

mpirun -np 1 python /home/gridsan/smin/python_scripts/multilayer_scintillator/optimize_linAttCoeff.py