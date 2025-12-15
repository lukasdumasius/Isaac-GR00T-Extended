#!/bin/bash

#SBATCH --job-name=libero-groot-memory
#SBATCH --output=libero-groot-memory.out
#SBATCH --error=libero-groot-memory.err
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --account=bfxb-delta-gpu

# Set PYTHONPATH to use current repo
export PYTHONPATH=/projects/bfxb/haorany7/MemFusionVLA/Isaac-GR00T-Extended:$PYTHONPATH

# Properly initialize conda in non-interactive bash, then activate gr00t env
eval "$(conda shell.bash hook)"
conda activate gr00t

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gr00t_finetune.py \
  --dataset-path /work/hdd/bfxb/data/libero-hfvla \
  --num-gpus 1 \
  --output-dir /work/hdd/bfxb/checkpoints/libero-groot-memory \
  --max-steps 200000 \
  --data-config libero_data_config:LiberoDataConfig \
  --video-backend torchvision_av \
  --report-to wandb \
  --save-steps 5000 \
  --enable-action-memory \
  --max-memory-size 32 \
  --memory-readout-nhead 4
