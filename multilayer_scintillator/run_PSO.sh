#!/bin/bash

#SBATCH -o /home/gridsan/smin/python_scripts/multilayer_scintillator/slurm/run_PSO.log-%j
#SBATCH -n 250

source /etc/profile
module load anaconda/2023a
export OMP_NUM_THREADS=1

. /home/gridsan/smin/geant4/geant4-v11.1.2-install/share/Geant4/geant4make/geant4make.sh
mpirun -np 250 python /home/gridsan/smin/python_scripts/multilayer_scintillator/PSO_geant4.py