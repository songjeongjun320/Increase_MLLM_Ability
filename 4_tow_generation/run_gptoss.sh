#!/bin/bash

echo "=== GPT-OSS 120B Gold Label Generation ==="
echo "Starting memory-optimized loading..."

# System info
echo "GPU Info:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

echo -e "\nCPU Memory Info:"
free -h

# Clean up any existing processes
pkill -f gptoss_hrm8k_generate_gold_word.py 2>/dev/null || true

# Set memory limits and optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
ulimit -v unlimited  # Allow unlimited virtual memory

# Run with optimized settings
python gptoss_hrm8k_generate_gold_word.py 2>&1 | tee generation_log.txt

echo "=== Generation Complete ==="