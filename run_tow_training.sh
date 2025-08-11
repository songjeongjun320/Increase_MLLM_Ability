#!/bin/bash

# ToW Training Execution Script
# Usage: bash run_tow_training.sh

echo "=========================================="
echo "ToW (Thoughts of Words) Training Pipeline"
echo "=========================================="

# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Check if required data exists
if [ ! -f "ToW_koconovel_complete.json" ]; then
    echo "Error: ToW_koconovel_complete.json not found!"
    echo "Please run the ToW dataset generation script first."
    exit 1
fi

# Check if models directory exists
if [ ! -d "/scratch/jsong132/Increase_MLLM_Ability/Base_Models" ]; then
    echo "Warning: Base models directory not found."
    echo "Please update model paths in ToW_Training.py"
fi

# Create output directory
mkdir -p ToW_Models

# Set environment variables for optimal performance
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8

# Optional: Set W&B project
# export WANDB_PROJECT="tow-korean-finetuning"

echo "Starting ToW training pipeline..."
echo "Training 4 models with ToW-augmented Korean data"
echo "Output will be saved to: ToW_Models/"

# Run the training
python ToW_Training.py

echo "Training pipeline completed!"
echo "Check ToW_Models/ for trained models"