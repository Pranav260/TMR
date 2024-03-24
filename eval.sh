#!/bin/bash
#SBATCH --account=project_462000189
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=04:00:00
#SBATCH --output=zshot_clip_0.1.log
module use /appl/local/csc/soft/ai/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/pranav/everything_at_once
#pip install wrapt
export OMP_NUM_THREADS=16
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun python test.py --n_gpu 1 --config configs/evaluation/stronger_backbone/clip/clip_msrvtt_at_once.yaml   --resume pretrained_models/clip/latest_model.pth