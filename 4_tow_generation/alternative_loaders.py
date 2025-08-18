#!/usr/bin/env python3
"""
대안적인 GPT-OSS 120B 모델 로딩 방법들
다양한 접근법으로 모델 로딩 시도
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

def method1_direct_loading(model_path):
    """방법 1: 직접 로딩 (가장 기본)"""
    print("\n[방법 1] 직접 로딩")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("  ✅ 성공")
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return None, None

def method2_no_trust_code(model_path):
    """방법 2: trust_remote_code=False"""
    print("\n[방법 2] trust_remote_code=False")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False)
        print("  ✅ 성공")
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return None, None

def method3_force_download(model_path):
    """방법 3: 강제 다운로드"""
    print("\n[방법 3] 강제 다운로드")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            force_download=True,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            force_download=True,
            trust_remote_code=True
        )
        print("  ✅ 성공")
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return None, None

def method4_specific_revision(model_path):
    """방법 4: 특정 revision 사용"""
    print("\n[방법 4] 특정 revision")
    revisions = ["main", "master", None]
    
    for revision in revisions:
        try:
            print(f"    revision: {revision}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=True
            )
            print("  ✅ 성공")
            return model, tokenizer
        except Exception as e:
            print(f"    ❌ revision {revision} 실패: {e}")
            continue
    
    return None, None

def method5_manual_config(model_path):
    """방법 5: 수동 config 수정"""
    print("\n[방법 5] 수동 config 수정")
    
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print("  ❌ config.json 파일 없음")
        return None, None
    
    try:
        # config 백업 및 수정
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        backup_path = config_path.with_suffix('.json.backup')
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # auto_map 제거 시도
        if 'auto_map' in config:
            print("    auto_map 제거 시도")
            config_modified = config.copy()
            del config_modified['auto_map']
            
            temp_config_path = config_path.with_suffix('.json.temp')
            with open(temp_config_path, 'w') as f:
                json.dump(config_modified, f, indent=2)
            
            # 원본 파일 대체
            os.rename(temp_config_path, config_path)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                print("  ✅ 성공 (auto_map 제거)")
                return model, tokenizer
            except Exception as e:
                print(f"  ❌ auto_map 제거 후에도 실패: {e}")
            finally:
                # 원본 복구
                os.rename(backup_path, config_path)
        
        return None, None
        
    except Exception as e:
        print(f"  ❌ config 수정 실패: {e}")
        return None, None

def method6_torch_load(model_path):
    """방법 6: torch.load 직접 사용"""
    print("\n[방법 6] torch.load 직접 사용")
    
    try:
        # safetensors 파일들 찾기
        safetensor_files = list(Path(model_path).glob("*.safetensors"))
        
        if not safetensor_files:
            print("  ❌ safetensors 파일 없음")
            return None, None
        
        print(f"    {len(safetensor_files)}개의 safetensors 파일 발견")
        
        # tokenizer만 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        print("  ✅ tokenizer 로드 성공")
        
        # 이 방법은 복잡하므로 일단 tokenizer만 반환
        return None, tokenizer
        
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return None, None

def method7_transformers_legacy(model_path):
    """방법 7: 레거시 transformers 방식"""
    print("\n[방법 7] 레거시 방식")
    
    try:
        # 버전별 호환성 시도
        from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizer
        
        try:
            tokenizer = GPTNeoXTokenizer.from_pretrained(model_path)
            model = GPTNeoXForCausalLM.from_pretrained(model_path)
            print("  ✅ GPTNeoX로 성공")
            return model, tokenizer
        except Exception as e:
            print(f"  ❌ GPTNeoX 실패: {e}")
        
        # LLaMA 시도
        try:
            from transformers import LlamaForCausalLM, LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
            model = LlamaForCausalLM.from_pretrained(model_path)
            print("  ✅ Llama로 성공")
            return model, tokenizer
        except Exception as e:
            print(f"  ❌ Llama 실패: {e}")
        
        return None, None
        
    except ImportError as e:
        print(f"  ❌ 레거시 모델 import 실패: {e}")
        return None, None

def method8_local_files_only(model_path):
    """방법 8: 로컬 파일만 사용"""
    print("\n[방법 8] 로컬 파일만 사용")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        print("  ✅ 성공")
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return None, None

def method9_minimal_load(model_path):
    """방법 9: 최소 설정으로 로드"""
    print("\n[방법 9] 최소 설정")
    
    try:
        # 가장 기본적인 설정만 사용
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # 기본 타입 사용
            device_map=None,  # device_map 사용 안함
        )
        print("  ✅ 성공")
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return None, None

def test_all_methods(model_path):
    """모든 방법을 순차적으로 테스트"""
    
    methods = [
        method1_direct_loading,
        method2_no_trust_code,
        method3_force_download,
        method4_specific_revision,
        method5_manual_config,
        method6_torch_load,
        method7_transformers_legacy,
        method8_local_files_only,
        method9_minimal_load,
    ]
    
    print("="*60)
    print("GPT-OSS 120B 대안 로딩 방법 테스트")
    print("="*60)
    
    for i, method in enumerate(methods, 1):
        try:
            model, tokenizer = method(model_path)
            if model is not None and tokenizer is not None:
                print(f"\n🎉 성공! 방법 {i}로 모델 로딩 완료")
                
                # 간단한 생성 테스트
                try:
                    inputs = tokenizer("안녕하세요", return_tensors="pt")
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"생성 테스트: {result}")
                except Exception as gen_e:
                    print(f"생성 테스트 실패: {gen_e}")
                
                return model, tokenizer
            
        except Exception as e:
            print(f"\n💥 방법 {i} 예외 발생: {e}")
            continue
    
    print("\n❌ 모든 방법 실패")
    return None, None

if __name__ == "__main__":
    model_path = "../1_models/gpt_oss/gpt-oss-120b"
    model, tokenizer = test_all_methods(model_path)