#!/bin/bash

#SBATCH -o /home/gridsan/smin/python_scripts/multilayer_scintillator/slurm/run_xray_spectrum.log-%j
#SBATCH -n 1

source /etc/profile
module load anaconda/2023a
export OMP_NUM_THREADS=1

mpirun -np 1 python /home/gridsan/smin/python_scripts/multilayer_scintillator/xray_spectrum_generator.py