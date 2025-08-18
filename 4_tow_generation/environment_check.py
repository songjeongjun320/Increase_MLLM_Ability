#!/usr/bin/env python3
"""
환경 진단 스크립트
GPT-OSS 120B 모델 로딩을 위한 환경 점검
"""

import sys
import torch
import os
from pathlib import Path

def check_environment():
    """환경 상태를 체크하고 리포트 생성"""
    
    print("="*60)
    print("GPT-OSS 120B 환경 진단 리포트")
    print("="*60)
    
    # Python 버전
    print(f"Python 버전: {sys.version}")
    print()
    
    # PyTorch 정보
    print("PyTorch 정보:")
    print(f"  버전: {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    GPU {i} 메모리: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    print()
    
    # 라이브러리 버전들
    libraries = [
        'transformers',
        'accelerate', 
        'bitsandbytes',
        'datasets',
        'tokenizers',
        'safetensors',
        'huggingface_hub'
    ]
    
    print("라이브러리 버전:")
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  {lib}: {version}")
        except ImportError:
            print(f"  {lib}: 설치되지 않음")
    print()
    
    # BitsAndBytesConfig 테스트
    print("BitsAndBytesConfig 테스트:")
    try:
        from transformers import BitsAndBytesConfig
        
        # 기본 설정 테스트
        try:
            config = BitsAndBytesConfig(load_in_4bit=True)
            print("  기본 4-bit 설정: 성공")
        except Exception as e:
            print(f"  기본 4-bit 설정 실패: {e}")
        
        # 고급 설정 테스트
        try:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("  고급 4-bit 설정: 성공")
        except Exception as e:
            print(f"  고급 4-bit 설정 실패: {e}")
            
    except ImportError as e:
        print(f"  BitsAndBytesConfig import 실패: {e}")
    print()
    
    # 모델 경로 확인
    model_path = "../1_models/gpt_oss/gpt-oss-120b"
    print(f"모델 경로 확인: {model_path}")
    if Path(model_path).exists():
        print("  모델 디렉토리 존재: 예")
        
        # 모델 파일들 확인
        model_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        for file in model_files:
            file_path = Path(model_path) / file
            if file_path.exists():
                print(f"    {file}: 존재")
            else:
                print(f"    {file}: 없음")
                
        # .safetensors 파일들 확인
        safetensor_files = list(Path(model_path).glob("*.safetensors"))
        print(f"    safetensors 파일 개수: {len(safetensor_files)}")
        
    else:
        print("  모델 디렉토리 존재: 아니오")
    print()
    
    # 메모리 확인
    if torch.cuda.is_available():
        print("GPU 메모리 상태:")
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            free = total - cached
            print(f"  GPU {i}: 총 {total:.1f}GB, 사용 {allocated:.1f}GB, 캐시 {cached:.1f}GB, 여유 {free:.1f}GB")
    print()

def test_basic_transformers():
    """기본 transformers 기능 테스트"""
    print("기본 transformers 기능 테스트:")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("  AutoTokenizer, AutoModelForCausalLM import: 성공")
    except Exception as e:
        print(f"  기본 import 실패: {e}")
        return
    
    # 간단한 모델로 테스트
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("  GPT2 tokenizer 로드: 성공")
    except Exception as e:
        print(f"  GPT2 tokenizer 로드 실패: {e}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        print("  GPT2 model 로드: 성공")
        del model  # 메모리 정리
    except Exception as e:
        print(f"  GPT2 model 로드 실패: {e}")
    print()

def check_gptoss_specific():
    """GPT-OSS 특화 확인"""
    print("GPT-OSS 특화 확인:")
    
    model_path = "../1_models/gpt_oss/gpt-oss-120b"
    
    # config.json 확인
    try:
        import json
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"  모델 타입: {config.get('model_type', 'Unknown')}")
            print(f"  아키텍처: {config.get('architectures', 'Unknown')}")
            print(f"  auto_map: {config.get('auto_map', 'None')}")
            
            if 'auto_map' in config:
                print("  ⚠️  커스텀 모델링 코드 사용됨 (trust_remote_code=True 필요)")
            
        else:
            print("  config.json 파일 없음")
    except Exception as e:
        print(f"  config.json 읽기 실패: {e}")
    
    print()

if __name__ == "__main__":
    check_environment()
    test_basic_transformers()
    check_gptoss_specific()
    
    print("="*60)
    print("진단 완료")
    print("="*60)