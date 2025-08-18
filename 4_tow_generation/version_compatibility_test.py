#!/usr/bin/env python3
"""
라이브러리 버전 호환성 테스트
GPT-OSS 120B와 호환되는 라이브러리 조합 찾기
"""

import subprocess
import sys
import importlib
import pkg_resources

# 테스트할 버전 조합들
VERSION_COMBINATIONS = [
    {
        "name": "최신 안정 버전",
        "versions": {
            "transformers": "4.35.2",
            "torch": "2.1.0",
            "accelerate": "0.24.1",
            "bitsandbytes": "0.41.3"
        }
    },
    {
        "name": "중간 버전",
        "versions": {
            "transformers": "4.33.0",
            "torch": "2.0.1",
            "accelerate": "0.22.0",
            "bitsandbytes": "0.40.0"
        }
    },
    {
        "name": "이전 안정 버전",
        "versions": {
            "transformers": "4.30.0",
            "torch": "1.13.1",
            "accelerate": "0.20.0",
            "bitsandbytes": "0.39.0"
        }
    },
    {
        "name": "GPT-OSS 권장 (추정)",
        "versions": {
            "transformers": "4.28.0",
            "torch": "1.13.0",
            "accelerate": "0.18.0",
            "bitsandbytes": "0.37.0"
        }
    }
]

def get_current_version(package_name):
    """현재 설치된 패키지 버전 확인"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except:
        return "설치되지 않음"

def check_current_environment():
    """현재 환경의 버전 정보 출력"""
    print("="*60)
    print("현재 환경 정보")
    print("="*60)
    
    packages = ["transformers", "torch", "accelerate", "bitsandbytes", "datasets", "tokenizers"]
    
    for package in packages:
        version = get_current_version(package)
        print(f"{package}: {version}")
    
    print()

def test_import_compatibility():
    """현재 환경에서 import 호환성 테스트"""
    print("="*60)
    print("Import 호환성 테스트")
    print("="*60)
    
    tests = [
        ("transformers 기본", "from transformers import AutoTokenizer, AutoModelForCausalLM"),
        ("BitsAndBytesConfig", "from transformers import BitsAndBytesConfig"),
        ("BitsAndBytesConfig 생성", """
from transformers import BitsAndBytesConfig
import torch
config = BitsAndBytesConfig(load_in_4bit=True)
        """),
        ("고급 BitsAndBytesConfig", """
from transformers import BitsAndBytesConfig
import torch
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
        """),
        ("torch 기본", "import torch; print(torch.cuda.is_available())"),
        ("accelerate", "import accelerate"),
        ("bitsandbytes", "import bitsandbytes"),
    ]
    
    for test_name, code in tests:
        try:
            exec(code)
            print(f"✅ {test_name}: 성공")
        except Exception as e:
            print(f"❌ {test_name}: 실패 - {e}")
    
    print()

def test_model_loading_compatibility():
    """간단한 모델로 로딩 호환성 테스트"""
    print("="*60)
    print("모델 로딩 호환성 테스트")
    print("="*60)
    
    # GPT2로 기본 기능 테스트
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("GPT2 tokenizer 테스트...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✅ GPT2 tokenizer 로딩 성공")
        
        print("GPT2 model 테스트...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        print("✅ GPT2 model 로딩 성공")
        
        # 양자화 테스트
        try:
            from transformers import BitsAndBytesConfig
            print("4-bit 양자화 테스트...")
            
            config = BitsAndBytesConfig(load_in_4bit=True)
            model_quantized = AutoModelForCausalLM.from_pretrained(
                "gpt2", 
                quantization_config=config,
                device_map="auto"
            )
            print("✅ 4-bit 양자화 테스트 성공")
            
        except Exception as e:
            print(f"❌ 양자화 테스트 실패: {e}")
        
    except Exception as e:
        print(f"❌ 기본 모델 로딩 실패: {e}")
    
    print()

def generate_requirements_files():
    """다양한 버전 조합의 requirements.txt 파일 생성"""
    print("="*60)
    print("Requirements 파일 생성")
    print("="*60)
    
    for i, combo in enumerate(VERSION_COMBINATIONS):
        filename = f"requirements_combo_{i+1}_{combo['name'].replace(' ', '_')}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"# {combo['name']}\n")
            for package, version in combo['versions'].items():
                f.write(f"{package}=={version}\n")
            
            # 추가 의존성
            f.write("\n# 추가 의존성\n")
            f.write("datasets\n")
            f.write("tokenizers\n")
            f.write("safetensors\n")
            f.write("huggingface_hub\n")
        
        print(f"✅ {filename} 생성 완료")
    
    print()

def create_test_script():
    """각 환경에서 실행할 테스트 스크립트 생성"""
    
    script_content = '''#!/usr/bin/env python3
"""
환경별 GPT-OSS 테스트 스크립트
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def test_gptoss_loading():
    model_path = "../1_models/gpt_oss/gpt-oss-120b"
    
    print("GPT-OSS 120B 로딩 테스트 시작...")
    
    # 방법 1: 기본 로딩
    try:
        print("\\n[테스트 1] 기본 로딩")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("✅ 기본 로딩 성공!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 기본 로딩 실패: {e}")
    
    # 방법 2: trust_remote_code=False
    try:
        print("\\n[테스트 2] trust_remote_code=False")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False)
        print("✅ trust_remote_code=False 성공!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ trust_remote_code=False 실패: {e}")
    
    # 방법 3: 양자화 없이
    try:
        print("\\n[테스트 3] 양자화 없이 float16")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ float16 로딩 성공!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ float16 로딩 실패: {e}")
    
    # 방법 4: CPU only
    try:
        print("\\n[테스트 4] CPU only")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu"
        )
        print("✅ CPU 로딩 성공!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ CPU 로딩 실패: {e}")
    
    print("\\n❌ 모든 로딩 방법 실패")
    return None, None

if __name__ == "__main__":
    print("="*50)
    print("GPT-OSS 120B 환경별 테스트")
    print("="*50)
    
    model, tokenizer = test_gptoss_loading()
    
    if model and tokenizer:
        print("\\n🎉 모델 로딩 성공! 간단한 생성 테스트...")
        try:
            inputs = tokenizer("안녕하세요", return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"생성 결과: {result}")
        except Exception as e:
            print(f"생성 테스트 실패: {e}")
    else:
        print("\\n❌ 모델 로딩 실패")
'''
    
    with open("test_gptoss_environment.py", 'w') as f:
        f.write(script_content)
    
    print("✅ test_gptoss_environment.py 생성 완료")

def main():
    """메인 함수"""
    check_current_environment()
    test_import_compatibility()
    test_model_loading_compatibility()
    generate_requirements_files()
    create_test_script()
    
    print("="*60)
    print("권장 테스트 순서:")
    print("="*60)
    print("1. 현재 환경에서: python environment_check.py")
    print("2. 현재 환경에서: python alternative_loaders.py")
    print("3. 새 환경 만들기: conda create -n gptoss_test python=3.8")
    print("4. 환경 활성화: conda activate gptoss_test")
    print("5. 버전 조합 설치: pip install -r requirements_combo_1_최신_안정_버전.txt")
    print("6. 테스트 실행: python test_gptoss_environment.py")
    print("7. 실패시 다른 requirements 파일로 재시도")
    print()
    print("각 조합을 테스트하여 어떤 버전에서 성공하는지 확인하세요!")

if __name__ == "__main__":
    main()