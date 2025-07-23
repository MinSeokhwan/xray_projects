#!/bin/bash
#
#-------------------------------------------------------------
#running a distributed memory MPI job over multiple nodes
#-------------------------------------------------------------
#
#SBATCH -o /nfs/scistore08/roquegrp/smin/xray_projects/xray_scattering_discrimination/slurm/run_simulation.log-%j
#SBATCH --job-name=run_data
#
#Total number of CPU cores to be used for the MPI job
#SBATCH --ntasks=100
#
#Define the number of hours the job should run. 
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=240:00:00
#
#Define the amount of RAM used per CPU in GigaBytes
#In distributed memory applications the total amount of RAM 
#used will be:   number of CPUs * memory per CPU
#SBATCH --mem-per-cpu=5G
#
#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning, 
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Ensure that only nodes of same type are selected
#SBATCH --partition=defaultp
##SBATCH --constraint="eta|epsilon|delta|beta|leonid|serbyn|gamma|zeta"
#
#Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
#
#load an MPI module with SLURM support, or a software module with MPI support
source /nfs/scistore08/roquegrp/smin/geant4/bin/activate
module load openmpi/4.1.6
module load gcc/12.3
module load python/3.12
#
#for pure MPI jobs the number of threads has to be one
export OMP_NUM_THREADS=1

export PATH=/nfs/scistore08/roquegrp/smin/xray_projects/geant4/geant4-v11.2.2-install/bin:$PATH
export LD_LIBRARY_PATH=/nfs/scistore08/roquegrp/smin/xray_projects/geant4/geant4-v11.2.2-install/lib:$LD_LIBRARY_PATH
#
#run the respective binary through SLURM's srun
mpirun -n 60 python /nfs/scistore08/roquegrp/smin/xray_projects/xray_scattering_discrimination/run_simulation.py