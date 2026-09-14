#!/bin/bash
#SBATCH --job-name=multimae_finetune_yield
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --output=finetune_yield_log_%j.out
#SBATCH --error=finetune_yield_err_%j.err

# environment setup — x86용 conda 환경
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate multimae_env
export PYTHONUNBUFFERED=1

cd /work/mech-ai-scratch/bgekim/project/MultiMAE_NEW/MultiMAE
torchrun --nproc_per_node=1 refac_finetune.py \
    --config /work/mech-ai-scratch/bgekim/project/MultiMAE_NEW/MultiMAE/refac_config_yield.yaml