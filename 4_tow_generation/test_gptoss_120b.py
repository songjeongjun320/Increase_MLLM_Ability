#!/usr/bin/env python3
"""
test_gptoss_120b.py

GPT-OSS 120B 모델을 터미널에서 채팅 형태로 테스트할 수 있는 스크립트입니다.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# 모델 경로
MODEL_PATH = "../1_models/gpt_oss/gpt-oss-120b"

def load_model():
    """GPT-OSS 120B 모델과 토크나이저를 로드합니다."""
    print(f"[INFO] Loading GPT-OSS 120B model: {MODEL_PATH}")
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[SUCCESS] Tokenizer loaded successfully")
    except Exception as e:
        print(f"[ERROR] Tokenizer loading failed: {e}")
        return None, None
    
    # 모델 로딩 전략들
    strategies = [
        {
            "name": "GPU with BFloat16",
            "config": {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }
        },
        {
            "name": "GPU with Float16",
            "config": {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
        },
        {
            "name": "CPU Fallback",
            "config": {
                "device_map": {"": "cpu"},
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
            }
        }
    ]
    
    for strategy in strategies:
        try:
            print(f"[INFO] Trying: {strategy['name']}")
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **strategy['config'])
            model.eval()
            print(f"[SUCCESS] Model loaded with: {strategy['name']}")
            return model, tokenizer
        except Exception as e:
            print(f"[WARNING] {strategy['name']} failed: {e}")
            continue
    
    print("[ERROR] All loading strategies failed")
    return None, None

def generate_response(model, tokenizer, user_input, max_new_tokens=100):
    """사용자 입력에 대한 응답을 생성합니다."""
    try:
        # 프롬프트 구성
        prompt = f"사용자: {user_input}\n어시스턴트:"
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # 모델 디바이스로 이동
        try:
            model_device = next(iter(model.parameters())).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except:
            pass
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 새로 생성된 부분만 디코딩
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        return f"[ERROR] 생성 실패: {e}"

def main():
    """메인 채팅 루프"""
    print("=" * 60)
    print("GPT-OSS 120B 테스트 채팅")
    print("=" * 60)
    print("명령어:")
    print("  - 'quit' 또는 'exit': 종료")
    print("  - 'clear': 화면 지우기")
    print("  - 'help': 도움말")
    print("=" * 60)
    
    # 모델 로드
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("[ERROR] 모델 로딩 실패. 프로그램을 종료합니다.")
        return
    
    print("\n[SUCCESS] 모델이 준비되었습니다. 채팅을 시작하세요!\n")
    
    while True:
        try:
            # 사용자 입력
            user_input = input("사용자: ").strip()
            
            # 종료 명령
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("\n채팅을 종료합니다.")
                break
            
            # 화면 지우기
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # 도움말
            if user_input.lower() == 'help':
                print("\n명령어:")
                print("  - 'quit' 또는 'exit': 종료")
                print("  - 'clear': 화면 지우기")
                print("  - 'help': 도움말")
                print("  - 그 외: 모델과 대화\n")
                continue
            
            # 빈 입력 체크
            if not user_input:
                print("입력을 해주세요.\n")
                continue
            
            # 응답 생성
            print("어시스턴트: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            print()  # 빈 줄 추가
            
        except KeyboardInterrupt:
            print("\n\n[INFO] Ctrl+C 감지. 프로그램을 종료합니다.")
            break
        except EOFError:
            print("\n[INFO] EOF 감지. 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] 예상치 못한 오류: {e}")
            continue

if __name__ == "__main__":
    main()