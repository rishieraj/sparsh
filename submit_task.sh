#!/bin/bash
#SBATCH --requeue
#SBATCH --array=0-23
#SBATCH --job-name=tactile_ssl
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account gum
#SBATCH --qos gum
#SBATCH --time=03-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-type=FAIL
#SBATCH --signal=SIGUSR1@90

TASK=$1
SENSOR=$2
PATHS=$3
BASE_MAX_EPOCHS=51
SSL_METHODS=("e2e" "mae" "dino" "dinov2" "ijepa" "vjepa")
TRAIN_DATA_BUDGET=("1.0" "0.33" "0.1" "0.01")
MAX_EPOCHS=("51" "153" "510" "1530")

for ssl_method in "${SSL_METHODS[@]}";
do
  for((i=0; i<${#TRAIN_DATA_BUDGET[@]}; i++)); 
  do
    FLAT_SSL_METHODS+=("$ssl_method")
    FLAT_TRAIN_DATA_BUDGET+=("${TRAIN_DATA_BUDGET[$i]}")
    FLAT_MAX_EPOCHS+=("${MAX_EPOCHS[$i]}")
  done
done
echo ${FLAT_SSL_METHODS[@]}
echo ${FLAT_TRAIN_DATA_BUDGET[@]}
echo ${FLAT_MAX_EPOCHS[@]}

source /data/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate tactile_ssl --no-stack
wandb enabled

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun python train_task.py +experiment=downstream_task/${TASK}/${SENSOR}_${FLAT_SSL_METHODS[$SLURM_ARRAY_TASK_ID]} paths=$PATHS hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=$4 train_data_budget=${FLAT_TRAIN_DATA_BUDGET[$SLURM_ARRAY_TASK_ID]} trainer.max_epochs=${FLAT_MAX_EPOCHS[$SLURM_ARRAY_TASK_ID]}
srun python test_task.py +experiment=downstream_task/${TASK}/${SENSOR}_${FLAT_SSL_METHODS[$SLURM_ARRAY_TASK_ID]} paths=$PATHS hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=$4 experiment_name=dummy train_data_budget=${FLAT_TRAIN_DATA_BUDGET[$SLURM_ARRAY_TASK_ID]}