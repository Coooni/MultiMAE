#!/bin/bash
#SBATCH --job-name=multimae_resume
#SBATCH --partition=nova-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=120:00:00
#SBATCH --output=train_multimae_log_%j.out
#SBATCH --error=train_multimae_err_%j.err

# environment setup
source /work/mech-ai-scratch/bgekim/miniconda3-arm/etc/profile.d/conda.sh
conda activate multimae_env_arm

export PYTHONUNBUFFERED=1

#  move to the working directory
cd /work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE

# add argument for resume
torchrun --nproc_per_node=1 refac_pretrain.py \
    --config /work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/refac_config_pretrain.yaml \
    --output_dir /work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/output/pretrain \
    --resume /work/mech-ai-scratch/bgekim/project/imputation/MultiMAE_NEW/MultiMAE/output/pretrain/pretrain_latest.pth