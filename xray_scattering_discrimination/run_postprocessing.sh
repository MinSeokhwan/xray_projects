#!/bin/bash
#
#-------------------------------------------------------------
#example script for running a single-CPU serial job via SLURM
#-------------------------------------------------------------
#
#SBATCH --job-name=run_postproc
#SBATCH --output=/nfs/scistore08/roquegrp/smin/xray_projects/xray_scattering_discrimination/slurm/run_postprocessing.log-%j
#
#Total number of CPU cores to be used for the MPI job
#SBATCH --ntasks=1
#
#Define the number of hours the job should run. 
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=24:00:00
#
#Define the amount of RAM used by your job in GigaBytes
#SBATCH --mem-per-cpu=5G
#
#Send emails when a job starts, it is finished or it exits
#SBATCH --mail-user=petermsh513@gmail.com
#SBATCH --mail-type=ALL
#
#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning, 
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#

#SBATCH --partition=defaultp

#Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
#
#for single-CPU jobs make sure that they use a single thread
export OMP_NUM_THREADS=1
#
#load the respective software module you intend to use
source /nfs/scistore08/roquegrp/smin/geant4/bin/activate
module load python/3.12
module load openmpi/4.1.6
#
#
#run the respective binary through SLURM's srun
mpirun -n 1 python /nfs/scistore08/roquegrp/smin/xray_projects/xray_scattering_discrimination/run_postprocessing.py