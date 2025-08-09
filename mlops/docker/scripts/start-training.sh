#!/bin/bash
# ToW Training Pipeline Startup Script

set -e

echo "Starting ToW Training Pipeline..."
echo "Environment: $MLOPS_ENV"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)')
else:
    print('WARNING: No GPU available for training')
"

# Wait for MLflow
if [ ! -z "$MLFLOW_TRACKING_URI" ] && [[ "$MLFLOW_TRACKING_URI" == http* ]]; then
    echo "Waiting for MLflow server..."
    python3 -c "
import time
import requests
import os

mlflow_uri = os.environ['MLFLOW_TRACKING_URI']
for i in range(30):
    try:
        response = requests.get(f'{mlflow_uri}/health', timeout=5)
        if response.status_code == 200:
            print('MLflow server is ready')
            break
    except Exception as e:
        print(f'Attempt {i+1}/30: MLflow not ready: {e}')
        time.sleep(5)
else:
    print('Warning: MLflow server not available, using local tracking')
"
fi

# Initialize W&B if API key is provided
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Initializing Weights & Biases..."
    wandb login $WANDB_API_KEY
fi

# Create necessary directories
mkdir -p /workspace/logs /workspace/models /workspace/data /workspace/checkpoints /workspace/outputs

# Set default configuration
export PYTHONPATH="/workspace:$PYTHONPATH"

# Check if this is a distributed training setup
if [ ! -z "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    echo "Starting distributed training with $WORLD_SIZE processes..."
    exec /workspace/distributed-training.sh
fi

# Check for training configuration file
TRAINING_CONFIG=${TRAINING_CONFIG_FILE:-/workspace/mlops/configs/training_config.yaml}

if [ ! -f "$TRAINING_CONFIG" ]; then
    echo "Creating default training configuration..."
    cat > "$TRAINING_CONFIG" << EOF
# ToW Training Configuration
model_name: tow-llama-7b
base_model_path: /workspace/models/base
max_seq_length: 2048

# Training parameters
epochs: 3
batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-5
weight_decay: 0.01
warmup_steps: 100

# LoRA parameters
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# ToW-specific parameters
thought_weight: 0.3
cultural_weight: 0.2
translation_weight: 0.5

# Optimization
use_gradient_checkpointing: true
use_mixed_precision: true
optimizer: adamw
lr_scheduler: cosine

# Data paths (update these)
train_dataset_path: /workspace/data/train.jsonl
val_dataset_path: /workspace/data/val.jsonl
test_dataset_path: /workspace/data/test.jsonl

# Validation
eval_steps: 500
save_steps: 1000
logging_steps: 100
early_stopping_patience: 3
EOF
    echo "Default training configuration created at $TRAINING_CONFIG"
    echo "Please update the data paths and model settings before training."
fi

# Check if we should run training immediately or wait for user input
if [ "$AUTO_START_TRAINING" = "true" ]; then
    echo "Starting training automatically..."
    python3 -c "
import asyncio
from mlops.config import load_config
from mlops.pipeline import TrainingPipeline, TrainingConfig
import yaml

# Load configuration
config = load_config()
training_pipeline = TrainingPipeline(config)

# Load training config
with open('$TRAINING_CONFIG', 'r') as f:
    training_config_dict = yaml.safe_load(f)

training_config = TrainingConfig(**training_config_dict)

# Run training
async def main():
    result = await training_pipeline.run_training_pipeline(training_config)
    print(f'Training completed with status: {result.status}')
    return result

asyncio.run(main())
"
else
    echo "Training environment ready!"
    echo "Configuration file: $TRAINING_CONFIG"
    echo ""
    echo "To start training, run:"
    echo "python3 -c \"
import asyncio
from mlops.config import load_config
from mlops.pipeline import TrainingPipeline, TrainingConfig
import yaml

config = load_config()
training_pipeline = TrainingPipeline(config)

with open('$TRAINING_CONFIG', 'r') as f:
    training_config_dict = yaml.safe_load(f)

training_config = TrainingConfig(**training_config_dict)

async def main():
    result = await training_pipeline.run_training_pipeline(training_config)
    print(f'Training completed with status: {result.status}')

asyncio.run(main())
\""
    echo ""
    echo "Or use the interactive training script:"
    echo "python3 scripts/interactive_training.py"
    
    # Keep container running
    exec tail -f /dev/null
fi