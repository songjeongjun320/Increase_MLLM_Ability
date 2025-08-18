#!/usr/bin/env python3
"""
안전한 모델 로딩 유틸리티
환경에 관계없이 안정적으로 모델을 로드하는 헬퍼 함수들
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def safe_import_quantization():
    """BitsAndBytesConfig를 안전하게 import"""
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig
    except ImportError:
        print("[WARNING] BitsAndBytesConfig not available")
        return None


def get_model_loading_strategies(model_path, num_gpus, devices):
    """다양한 모델 로딩 전략을 반환"""
    
    BitsAndBytesConfig = safe_import_quantization()
    
    strategies = []
    
    # 양자화 가능한 경우에만 추가
    if BitsAndBytesConfig is not None:
        try:
            # 고급 4-bit 양자화
            strategies.append({
                "name": "4-bit quantization with advanced settings",
                "config": {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    ),
                    "device_map": "auto" if num_gpus > 1 else devices[0],
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.bfloat16,
                }
            })
        except Exception:
            pass
        
        try:
            # 기본 4-bit 양자화
            strategies.append({
                "name": "Basic 4-bit quantization",
                "config": {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                    "device_map": "auto" if num_gpus > 1 else devices[0],
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16,
                }
            })
        except Exception:
            pass
    
    # 양자화 없는 전략들
    strategies.extend([
        {
            "name": "Float16 without quantization",
            "config": {
                "device_map": "auto" if num_gpus > 1 else devices[0],
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16,
            }
        },
        {
            "name": "Basic loading with float16",
            "config": {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
            }
        },
        {
            "name": "Basic loading with auto dtype",
            "config": {
                "trust_remote_code": True,
            }
        },
        {
            "name": "Minimal loading (CPU fallback)",
            "config": {
                "trust_remote_code": True,
                "device_map": "cpu",
            }
        }
    ])
    
    return strategies


def load_model_safe(model_path, num_gpus=None, devices=None):
    """안전하게 모델과 토크나이저를 로드"""
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if devices is None:
        devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else ["cpu"]
    
    print(f"[INFO] Loading model: {model_path}")
    print(f"[INFO] Available devices: {devices}")
    
    # 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[SUCCESS] Tokenizer loaded successfully")
    except Exception as tokenizer_error:
        print(f"[ERROR] Tokenizer loading failed: {tokenizer_error}")
        return None, None
    
    # 모델 로딩 전략들
    strategies = get_model_loading_strategies(model_path, num_gpus, devices)
    
    for strategy in strategies:
        try:
            print(f"[INFO] Trying: {strategy['name']}")
            
            # 설정 준비
            config = strategy['config'].copy()
            
            # quantization_config가 None이면 제거
            if config.get('quantization_config') is None:
                config.pop('quantization_config', None)
            
            # 모델 로드 시도
            model = AutoModelForCausalLM.from_pretrained(model_path, **config)
            
            # 성공적으로 로드된 경우
            print(f"[SUCCESS] Model loaded with strategy: {strategy['name']}")
            
            # 추가 설정
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            
            # 모델 파라미터 확인
            try:
                total_params = sum(p.numel() for p in model.parameters())
                print(f"[INFO] Total model parameters: {total_params:,}")
            except Exception as e:
                print(f"[WARNING] Parameter count failed: {e}")
            
            model.eval()
            return model, tokenizer
            
        except Exception as e:
            print(f"[WARNING] Strategy '{strategy['name']}' failed: {e}")
            continue
    
    print(f"[ERROR] All model loading strategies failed")
    return None, None


def generate_safe(model, tokenizer, prompt, max_new_tokens=50, temperature=0.1):
    """안전하게 텍스트 생성"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # 모델 디바이스 확인
        try:
            model_device = next(iter(model.parameters())).device
        except:
            model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 입력을 모델 디바이스로 이동
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # 생성 설정
        generation_configs = [
            {
                "name": "Standard generation",
                "config": {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'do_sample': True if temperature > 0 else False,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': False,
                    'return_dict_in_generate': True,
                }
            },
            {
                "name": "Simple generation",
                "config": {
                    'max_new_tokens': min(max_new_tokens, 20),
                    'do_sample': False,
                    'pad_token_id': tokenizer.eos_token_id,
                    'use_cache': False,
                }
            },
            {
                "name": "Minimal generation",
                "config": {
                    'max_new_tokens': 10,
                    'pad_token_id': tokenizer.eos_token_id,
                }
            }
        ]
        
        for gen_config in generation_configs:
            try:
                print(f"[INFO] Trying generation: {gen_config['name']}")
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_config['config'])
                    sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
                
                # 새로 생성된 토큰 디코딩
                if len(sequences.shape) > 1 and sequences.shape[0] > 0:
                    new_tokens = sequences[0][inputs['input_ids'].shape[1]:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    print(f"[SUCCESS] Text generated with: {gen_config['name']}")
                    return generated_text.strip()
                
            except Exception as e:
                print(f"[WARNING] Generation '{gen_config['name']}' failed: {e}")
                continue
        
        print("[ERROR] All generation strategies failed")
        return None
        
    except Exception as e:
        print(f"[ERROR] Text generation failed: {e}")
        return None


if __name__ == "__main__":
    # 테스트용
    model_path = "../1_models/gpt_oss/gpt-oss-120b"
    model, tokenizer = load_model_safe(model_path)
    
    if model is not None and tokenizer is not None:
        print("[SUCCESS] Model loading test passed!")
        
        # 간단한 생성 테스트
        test_prompt = "안녕하세요"
        result = generate_safe(model, tokenizer, test_prompt, max_new_tokens=10)
        if result:
            print(f"[SUCCESS] Generation test passed: {result}")
        else:
            print("[WARNING] Generation test failed")
    else:
        print("[ERROR] Model loading test failed")