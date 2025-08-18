#!/bin/bash

echo "=== GPT-OSS 120B vLLM Generation on A100 80GB x2 ==="

# GPU 정보 확인
echo "GPU Information:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used,name --format=csv,noheader,nounits

# A100 80GB x2 특별 설정
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Flash Attention 사용
export TORCH_CUDA_ARCH_LIST="8.0"        # A100 아키텍처
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

# vLLM 최적화 설정
export VLLM_USE_MODELSCOPE=False
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=False
export NCCL_DEBUG=INFO                    # 멀티 GPU 디버깅
export NCCL_IB_DISABLE=1                  # InfiniBand 비활성화 (필요시)

# 메모리 최적화
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo -e "\nStarting vLLM model loading..."
echo "Tensor Parallel Size: 2 (A100 80GB x2)"
echo "Memory utilization: 85% per GPU"

# Python 경로 및 CUDA 확인
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# vLLM 설치 확인
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null || echo "vLLM not installed. Installing..."

# vLLM 설치 (필요시)
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm
fi

# 실행
echo -e "\nRunning vLLM GPT-OSS 120B generation..."
python gptoss_hrm8k_generate_gold_word_vllm.py 2>&1 | tee vllm_generation_log.txt

echo "=== vLLM Generation Complete ==="