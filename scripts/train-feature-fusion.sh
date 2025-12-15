#! /bin/bash
#
# Example training script for GR00T model with intermediate feature fusion
#

#SBATCH --job-name=libero-groot-feature-fusion
#SBATCH --output=/work/hdd/bfxb/checkpoints/libero-groot-feature-fusion/libero-groot-feature-fusion.out
#SBATCH --error=/work/hdd/bfxb/checkpoints/libero-groot-feature-fusion/libero-groot-feature-fusion.err
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpuA100x4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --account=bfxb-delta-gpu

# Load required modules
module load pytorch-conda/2.8

# Ensure user site-packages and lerobot directory are in PYTHONPATH
export PYTHONPATH="/u/ldumasius/.local/lib/python3.11/site-packages:/work/hdd/bfxb/lerobot:$PYTHONPATH"

# Set WandB API key
export WANDB_API_KEY=123

# Change to Isaac-GR00T-Extended-2 directory so gr00t module can be imported
cd /work/hdd/bfxb/Isaac-GR00T-Extended-2

# Use only one GPU
CUDA_VISIBLE_DEVICES=0 python /work/hdd/bfxb/Isaac-GR00T-Extended-2/scripts/gr00t_finetune.py \
  --dataset-path /work/hdd/bfxb/data/libero-hfvla \
  --num-gpus 1 \
  --output-dir /work/hdd/bfxb/checkpoints/libero-groot-feature-fusion \
  --max-steps 100000 \
  --data-config libero_data_config:LiberoDataConfig \
  --video-backend torchvision_av \
  --batch_size 1 \
  --report-to wandb \
  --save-steps 5000 \
  --fusion-config /work/hdd/bfxb/Isaac-GR00T-Extended-2/configs/fusion_intermediate.json


