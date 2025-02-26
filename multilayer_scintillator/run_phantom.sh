#!/bin/bash

#SBATCH -o /home/minseokhwan/xray_projects/multilayer_scintillator/slurm/run_phantom.log-%j
#SBATCH --partition=batch
#SBATCH --job-name=run_phantom
#SBATCH --ntasks=48

export OMP_NUM_THREADS=1

##. /home/minseokhwan/xray_projects/geant4/geant4-v11.2.2-install/share/Geant4/geant4make/geant4make.sh

export PATH=/home/minseokhwan/geant4_workdir/bin/Linux-g++:/home/minseokhwan/xray_projects/geant4/geant4-v11.2.2-install/bin:/home/minseokhwan/.local/bin:/opt/anaconda3/bin:/opt/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export LD_LIBRARY_PATH=/opt/gcc-12/lib64:/usr/lib/x86_64-linux-gnu:/opt/anaconda3/lib:/home/minseokhwan/xray_projects/geant4/geant4-v11.2.2-install/lib

##echo $PATH
##echo $LD_LIBRARY_PATH
/appl/intel/oneapi/mpi/2021.8.0/bin/mpirun -n 48 python /home/minseokhwan/xray_projects/multilayer_scintillator/run_phantom_geant4.py