#!/bin/bash
#SBATCH --job-name=multimae_finetune_yield
#SBATCH --partition=nova-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=300:00:00
#SBATCH --output=finetune_yield_log_%j.out
#SBATCH --error=finetune_yield_err_%j.err

# environment setup
source /work/mech-ai-scratch/bgekim/miniconda3-arm/etc/profile.d/conda.sh
conda activate multimae_env_arm
export PYTHONUNBUFFERED=1

# move to working directory
cd /work/mech-ai-scratch/bgekim/project/MultiMAE_NEW/MultiMAE

torchrun --nproc_per_node=1 refac_finetune.py \
    --config /work/mech-ai-scratch/bgekim/project/MultiMAE_NEW/MultiMAE/refac_config_yield.yaml 