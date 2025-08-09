#!/usr/bin/env python3
"""
Download DeepSeek-R1-Distill-Qwen-7B Model
=========================================

Downloads DeepSeek-R1-Distill-Qwen-7B model for Option 2 TOW training.
This is a 7B parameter instruction model with strong multilingual capabilities.
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

def download_deepseek_r1_distill_qwen_7b():
    """Download DeepSeek-R1-Distill-Qwen-7B model"""
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    local_dir = Path(__file__).parent / "deepseek-r1-distill-qwen-7b"
    
    print(f"📥 Downloading {model_name}...")
    print(f"📂 Saving to: {local_dir}")
    
    # Check available disk space (require at least 15GB)
    if torch.cuda.is_available():
        print(f"🔧 CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA not available - model will run on CPU")
    
    try:
        # Download model files
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ Model downloaded successfully to {local_dir}")
        
        # Verify download
        required_files = ["config.json", "model.safetensors.index.json", "tokenizer.json"]
        missing_files = []
        
        for file in required_files:
            if not (local_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Warning: Missing files: {missing_files}")
        else:
            print("✅ All required files present")
            
        return str(local_dir)
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

def test_model_loading():
    """Test loading the downloaded model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = Path(__file__).parent / "deepseek-r1-distill-qwen-7b"
        
        if not model_path.exists():
            print("❌ Model not found. Please download first.")
            return False
            
        print("🧪 Testing model loading...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded: {len(tokenizer.vocab)} tokens")
        
        # Test Korean tokenization
        korean_test = "안녕하세요. 한국어 처리가 가능한지 테스트합니다."
        tokens = tokenizer.encode(korean_test)
        print(f"✅ Korean tokenization test: {len(tokens)} tokens")
        
        # Load model (with low memory usage)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        print(f"✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_model_loading()
    else:
        model_path = download_deepseek_r1_distill_qwen_7b()
        if model_path:
            print(f"\n🚀 Ready to use! Model path: {model_path}")
            print("💡 Test loading with: python download_deepseek_r1_distill_qwen_7b.py --test")