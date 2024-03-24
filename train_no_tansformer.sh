#!/bin/bash
#SBATCH --account=project_462000189
#SBATCH --partition=eap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=10:00:00
#SBATCH --output=clip_1k_depth_1_bs_128_no_transformer.log
module use /appl/local/csc/soft/ai/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/pranav/everything_at_once
#pip install wrapt
export OMP_NUM_THREADS=16
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun python train.py --config configs/finetuning/clip/clip_no_transformer.yaml