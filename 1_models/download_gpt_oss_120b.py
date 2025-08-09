#!/usr/bin/env python3
"""
Download GPT-OSS-120B for ToW Generation
=======================================

Downloads OpenAI's GPT-OSS-120B model specifically for Korean ToW generation.
This is the larger model alternative to 20B - requires H100 GPU or similar.

- GPT-OSS-120B: 120B parameters, requires ~240GB memory
- Apache 2.0 license - commercial use allowed  
- Korean language support included
- Best ToW generation quality but high resource requirements
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
import psutil

def check_system_requirements():
    """Check if system can handle GPT-OSS-120B"""
    
    print("[SYSTEM CHECK] Checking requirements for GPT-OSS-120B...")
    
    # Check available RAM
    available_ram = psutil.virtual_memory().total // (1024**3)
    print(f"[RAM] Available system RAM: {available_ram}GB")
    
    if available_ram < 64:
        print("[WARNING] GPT-OSS-120B requires at least 64GB system RAM")
        print("[WARNING] Recommended: 128GB+ RAM for stable operation")
    
    # Check GPU memory
    gpu_sufficient = False
    if torch.cuda.is_available():
        total_gpu_memory = 0
        gpu_count = torch.cuda.device_count()
        
        for i in range(gpu_count):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            total_gpu_memory += gpu_memory
            print(f"[GPU {i}] {torch.cuda.get_device_name(i)}: {gpu_memory}GB")
        
        print(f"[GPU TOTAL] Combined GPU memory: {total_gpu_memory}GB")
        
        if total_gpu_memory >= 80:  # H100 or equivalent
            gpu_sufficient = True
            print("[GPU] ✓ Sufficient GPU memory for GPT-OSS-120B")
        else:
            print("[WARNING] GPT-OSS-120B requires 80GB+ GPU memory (H100/A100)")
            print("[WARNING] Current setup may not run the model effectively")
    else:
        print("[WARNING] No CUDA GPU detected - 120B model will be extremely slow on CPU")
    
    # Check disk space
    disk_usage = psutil.disk_usage('/')
    available_disk = disk_usage.free // (1024**3)
    print(f"[DISK] Available disk space: {available_disk}GB")
    
    if available_disk < 300:
        print("[WARNING] GPT-OSS-120B requires ~250GB disk space")
        print("[WARNING] Recommended: 300GB+ free disk space")
    
    return {
        "ram_sufficient": available_ram >= 64,
        "gpu_sufficient": gpu_sufficient,
        "disk_sufficient": available_disk >= 300,
        "total_gpu_memory": total_gpu_memory if torch.cuda.is_available() else 0
    }

def download_gpt_oss_120b():
    """Download GPT-OSS-120B model for ToW generation"""
    
    model_name = "openai/gpt-oss-120b"
    local_dir = Path(__file__).parent / "gpt_oss" / "gpt-oss-120b"
    
    print(f"[DOWNLOAD] Downloading {model_name} for ToW generation...")
    print(f"[PATH] Saving to: {local_dir}")
    print("[PURPOSE] Generate highest quality English ToW tokens for Korean stories")
    print("[SIZE] ~240GB model - this will take significant time and bandwidth")
    
    # Create directory
    local_dir.parent.mkdir(exist_ok=True)
    
    # Check system requirements
    requirements = check_system_requirements()
    
    # Warn about insufficient resources
    warnings = []
    if not requirements["ram_sufficient"]:
        warnings.append("Insufficient RAM")
    if not requirements["gpu_sufficient"]:
        warnings.append("Insufficient GPU memory")
    if not requirements["disk_sufficient"]:
        warnings.append("Insufficient disk space")
    
    if warnings:
        print(f"[WARNING] System issues detected: {', '.join(warnings)}")
        print("[WARNING] GPT-OSS-120B may not run properly on this system")
        print("[ALTERNATIVE] Consider using GPT-OSS-20B instead (download_gpt_oss_20b.py)")
        
        response = input("Continue download anyway? (y/n): ")
        if response.lower() != 'y':
            print("[CANCELLED] Download cancelled")
            print("[ALTERNATIVE] Use GPT-OSS-20B for 16GB GPU systems")
            return None
    else:
        print("[REQUIREMENTS] ✓ System meets GPT-OSS-120B requirements")
    
    # Confirm large download
    print(f"[CONFIRMATION] This will download ~240GB of model files")
    print(f"[CONFIRMATION] Estimated download time: 2-6 hours depending on internet speed")
    response = input("Proceed with download? (y/n): ")
    if response.lower() != 'y':
        print("[CANCELLED] Download cancelled")
        return None
    
    try:
        # Download model files
        print("[DOWNLOADING] GPT-OSS-120B model files (this will take a long time)...")
        print("[INFO] You can interrupt (Ctrl+C) and resume later if needed")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,  # Allow resuming interrupted downloads
            # Download in chunks to handle large files
            ignore_patterns=["*.safetensors.index.json", "*.bin.index.json"]
        )
        
        print(f"[SUCCESS] GPT-OSS-120B downloaded successfully to {local_dir}")
        
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
        
        # Final system check
        if size_gb < 200:
            print("[WARNING] Download may be incomplete (expected ~240GB)")
        else:
            print("[SUCCESS] Download appears complete")
            
        return str(local_dir)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Download interrupted by user")
        print("[INFO] You can resume by running this script again")
        print("[INFO] Partial files will be resumed automatically")
        return None
        
    except Exception as e:
        print(f"[ERROR] Error downloading model: {e}")
        print("[TROUBLESHOOTING] Check internet connection and disk space")
        print("[ALTERNATIVE] Try GPT-OSS-20B instead for lower requirements")
        return None

def test_gpt_oss_120b_loading():
    """Test loading GPT-OSS-120B for Korean ToW generation"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = Path(__file__).parent / "gpt_oss" / "gpt-oss-120b"
        
        if not model_path.exists():
            print("[ERROR] GPT-OSS-120B not found. Please download first.")
            return False
            
        print("[TESTING] GPT-OSS-120B loading for ToW generation...")
        print("[INFO] This may take several minutes due to model size")
        
        # Load tokenizer first (faster)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"[SUCCESS] Tokenizer loaded: {len(tokenizer.vocab)} tokens")
        
        # Test tokenization capabilities  
        korean_test = "브루스 리는 쿵푸 영화 감독을 만났다. <ToW>The context suggests professional meeting</ToW>"
        tokens = tokenizer.encode(korean_test)
        print(f"[SUCCESS] Korean + English ToW tokenization: {len(tokens)} tokens")
        
        # Load model with maximum memory optimization
        print("[LOADING] GPT-OSS-120B model (this will take 5-10 minutes)...")
        print("[INFO] Using all available GPU memory and CPU offloading")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",          # Automatically distribute across GPUs
            low_cpu_mem_usage=True,     # Minimize CPU memory usage
            offload_folder="./offload", # Offload to disk if needed
            trust_remote_code=True      # GPT-OSS may require this
        )
        print(f"[SUCCESS] GPT-OSS-120B loaded successfully for ToW generation")
        
        # Quick inference test
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
        
        # Memory usage info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) // (1024**3)
                reserved = torch.cuda.memory_reserved(i) // (1024**3)
                print(f"[MEMORY] GPU {i} - Allocated: {allocated}GB, Reserved: {reserved}GB")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing GPT-OSS-120B: {e}")
        print("[INFO] This is normal if system doesn't meet requirements")
        print("[SUGGESTION] Try reducing model precision or using smaller model")
        return False

def show_usage_info():
    """Show information about GPT-OSS-120B usage for ToW generation"""
    print("\n" + "="*70)
    print("[INFO] GPT-OSS-120B Usage for Option 2 TOW Project")
    print("="*70)
    print("[PURPOSE] Generate highest quality English ToW tokens for Korean stories")
    print("[WORKFLOW] Korean story → GPT-OSS-120B → Korean story + English <ToW>")
    print("[TRAINING] Separate 3-8B models will be trained with this ToW data")
    print("\n[REQUIREMENTS] System Requirements:")
    print("   • 80GB+ GPU memory (H100, A100 80GB, or multi-GPU setup)")
    print("   • 128GB+ system RAM (recommended)")
    print("   • 300GB+ disk space for model files")
    print("   • High-speed internet for 240GB download")
    print("\n[PERFORMANCE] Expected Performance:")
    print("   • Best ToW generation quality")
    print("   • Better Korean language understanding")
    print("   • More nuanced reasoning in English")
    print("   • Slower inference than 20B model")
    print("\n[ALTERNATIVES] If your system can't handle 120B:")
    print("   • Use GPT-OSS-20B (16GB GPU sufficient)")
    print("   • python download_gpt_oss_20b.py")
    print("\n[NEXT STEPS]")
    print("   1. Run Korean ToW generation: cd ../4_tow_generation/")
    print("   2. python korean_tow_generator.py --model 120b")
    print("   3. Train base models: cd ../5_training/")
    print("   4. Compare results: cd ../3_evaluation/")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_gpt_oss_120b_loading()
    elif len(sys.argv) > 1 and sys.argv[1] == "--info":
        show_usage_info()
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        requirements = check_system_requirements()
        print(f"\n[SUMMARY] System compatibility for GPT-OSS-120B:")
        print(f"   RAM: {'✓' if requirements['ram_sufficient'] else '✗'}")
        print(f"   GPU: {'✓' if requirements['gpu_sufficient'] else '✗'}")
        print(f"   Disk: {'✓' if requirements['disk_sufficient'] else '✗'}")
        
        if all(requirements.values()):
            print("[RESULT] ✓ System ready for GPT-OSS-120B")
        else:
            print("[RESULT] ✗ System may struggle with GPT-OSS-120B")
            print("[RECOMMENDATION] Consider GPT-OSS-20B instead")
    else:
        model_path = download_gpt_oss_120b()
        if model_path:
            print(f"\n[SUCCESS] GPT-OSS-120B ready for ToW generation!")
            print(f"[PATH] Model path: {model_path}")
            print("[TESTING] Test loading: python download_gpt_oss_120b.py --test")
            print("[INFO] Usage info: python download_gpt_oss_120b.py --info")
            print("[CHECK] System check: python download_gpt_oss_120b.py --check")
            show_usage_info()
        else:
            print("\n[FAILED] GPT-OSS-120B download failed")
            print("[ALTERNATIVE] Try GPT-OSS-20B instead:")
            print("   python download_gpt_oss_20b.py")