#!/usr/bin/env python3
"""
Download GPT-OSS-20B for ToW Generation
======================================

Downloads OpenAI's GPT-OSS-20B model specifically for Korean ToW generation.
This model is used ONLY for generating English ToW tokens, not for training.

- GPT-OSS-20B: 21B parameters, fits in 16GB memory
- Apache 2.0 license - commercial use allowed
- Korean language support included
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

def download_gpt_oss_20b():
    """Download GPT-OSS-20B model for ToW generation"""
    
    model_name = "openai/gpt-oss-20b"
    local_dir = Path(__file__).parent / "gpt_oss" / "gpt-oss-20b"
    
    print(f"[DOWNLOAD] Downloading {model_name} for ToW generation...")
    print(f"[PATH] Saving to: {local_dir}")
    print("[PURPOSE] Generate English ToW tokens for Korean stories")
    
    # Create directory
    local_dir.parent.mkdir(exist_ok=True)
    
    # Check system requirements
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"[CUDA] CUDA available: {torch.cuda.get_device_name()} ({gpu_memory}GB)")
        
        if gpu_memory < 16:
            print("[WARNING] GPT-OSS-20B requires at least 16GB GPU memory")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("[CANCELLED] Download cancelled")
                return None
    else:
        print("[WARNING] CUDA not available - model will run on CPU (very slow)")
        response = input("Continue with CPU-only? (y/n): ")
        if response.lower() != 'y':
            print("[CANCELLED] Download cancelled")
            return None
    
    try:
        # Download model files
        print("[DOWNLOADING] GPT-OSS-20B model files...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            # Only download essential files
            ignore_patterns=["*.safetensors.index.json", "*.bin.index.json"]
        )
        
        print(f"[SUCCESS] GPT-OSS-20B downloaded successfully to {local_dir}")
        
        # Verify download
        essential_files = ["config.json", "tokenizer.json"]
        missing_files = []
        
        for file in essential_files:
            if not (local_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"[WARNING] Missing files: {missing_files}")
        else:
            print("[SUCCESS] All essential files present")
            
        # Check model size
        total_size = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        print(f"[INFO] Model size: {size_gb:.1f}GB")
            
        return str(local_dir)
        
    except Exception as e:
        print(f"[ERROR] Error downloading model: {e}")
        return None

def test_gpt_oss_loading():
    """Test loading GPT-OSS-20B for Korean ToW generation"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = Path(__file__).parent / "gpt_oss" / "gpt-oss-20b"
        
        if not model_path.exists():
            print("[ERROR] GPT-OSS-20B not found. Please download first.")
            return False
            
        print("[TESTING] GPT-OSS-20B loading for ToW generation...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"[SUCCESS] Tokenizer loaded: {len(tokenizer.vocab)} tokens")
        
        # Test Korean + English tokenization (for ToW generation)
        korean_test = "브루스 리는 쿵푸 영화 감독을 만났다. <ToW>The context suggests professional meeting</ToW>"
        tokens = tokenizer.encode(korean_test)
        print(f"[SUCCESS] Korean + English ToW tokenization: {len(tokens)} tokens")
        
        # Test English-only tokenization (ToW content)
        english_test = "The context suggests a professional meeting between martial artist and filmmaker"
        english_tokens = tokenizer.encode(english_test)
        print(f"[SUCCESS] English-only tokenization: {len(english_tokens)} tokens")
        
        # Load model with memory optimization
        print("[LOADING] GPT-OSS-20B model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True  # GPT-OSS may require this
        )
        print(f"[SUCCESS] GPT-OSS-20B loaded successfully for ToW generation")
        
        # Test inference capability
        test_input = "다음 한국어 문장에 대해 영어로 추론하는 ToW를 생성하세요: 브루스 리는 쿵푸 영화 감독을 만났다."
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        print("[TESTING] ToW generation capability...")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[SUCCESS] ToW generation test successful")
        print(f"[SAMPLE] Generated: {generated_text[-100:]}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing GPT-OSS-20B: {e}")
        print("[INFO] This is normal if the model requires specific setup")
        return False

def show_usage_info():
    """Show information about GPT-OSS-20B usage for ToW generation"""
    print("\n" + "="*60)
    print("[INFO] GPT-OSS-20B Usage for Option 2 TOW Project")
    print("="*60)
    print("[PURPOSE] Generate English ToW tokens for Korean stories")
    print("[WORKFLOW] Korean story -> GPT-OSS-20B -> Korean story + English <ToW>")
    print("[TRAINING] Separate 3-8B models will be trained with this ToW data")
    print("\n[REQUIREMENTS] System Requirements:")
    print("   • 16GB+ GPU memory (recommended)")
    print("   • CUDA support for optimal performance") 
    print("   • ~50GB disk space for model files")
    print("\n[NEXT STEPS]")
    print("   1. Run Korean ToW generation: cd ../4_tow_generation/")
    print("   2. python korean_tow_generator.py")
    print("   3. Train base models: cd ../5_training/")
    print("   4. Compare results: cd ../3_evaluation/")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_gpt_oss_loading()
    elif len(sys.argv) > 1 and sys.argv[1] == "--info":
        show_usage_info()
    else:
        model_path = download_gpt_oss_20b()
        if model_path:
            print(f"\n[SUCCESS] GPT-OSS-20B ready for ToW generation!")
            print(f"[PATH] Model path: {model_path}")
            print("[TESTING] Test loading: python download_gpt_oss_20b.py --test")
            print("[INFO] Usage info: python download_gpt_oss_20b.py --info")
            show_usage_info()