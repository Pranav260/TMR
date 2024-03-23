#!/bin/bash
#SBATCH --account=project_462000189
#SBATCH --partition=eap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=05:00:00
#SBATCH --output=feat_ext.log
module use /appl/local/csc/soft/ai/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/pranav/everything_at_once
#pip install wrapt
export OMP_NUM_THREADS=16
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun python train_vivit_msr_vtt.py