#!/bin/bash

#SBATCH -o /home/minseokhwan/xray_projects/multilayer_scintillator/slurm/run_phantom.log-%j
#SBATCH --partition=batch
#SBATCH --job-name=run_phantom
#SBATCH --ntasks=1

export OMP_NUM_THREADS=1
export PATH=/opt/gcc-12/bin:$PATH
export LD_LIBRARY_PATH=/opt/gcc-12/lib64:$LD_LIBRARY_PATH

. /home/minseokhwan/xray_projects/geant4/geant4-v11.2.2-install/share/Geant4/geant4make/geant4make.sh
/appl/intel/oneapi/mpi/2021.8.0/bin/mpirun -n 1 python /home/minseokhwan/xray_projects/multilayer_scintillator/run_phantom_geant4.py