# -*- coding: utf-8 -*-
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# bitsandbytes 가져오기 시도
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    print("BitsAndBytesConfig를 성공적으로 import했습니다.")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("BitsAndBytesConfig import 실패 - 기본 양자화를 사용합니다.")

def clear_memory():
    """메모리 정리 함수"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ask_deepseek(question, model, tokenizer, device, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """DeepSeek 모델에게 질문하고 답변 받기"""
    # 메모리 정리
    clear_memory()
    
    # 간단한 Q&A 형식 프롬프트
    prompt = f"Question: {question}\nAnswer:"

    print(f"\n--- 입력 프롬프트 ---\n{prompt}\n--------------------")

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # eos_token_id가 여러 개일 수 있으므로, 리스트로 전달
    eos_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'additional_special_tokens_ids'):
        eos_token_ids.extend(tokenizer.additional_special_tokens_ids)
    
    # pad_token_id가 설정되어 있지 않다면 eos_token_id로 설정
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("답변 생성 중...")
    # 생성 파라미터 설정
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids,
        "use_cache": True,
    }

    # 메모리 효율적인 생성을 위해 torch.no_grad() 사용
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

    # 입력 부분을 제외하고 생성된 텍스트만 디코딩
    answer_ids = generated_ids[0][input_ids.shape[-1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # 메모리 정리
    del input_ids, attention_mask, generated_ids
    clear_memory()

    return answer_text.strip()

def load_model():
    """모델과 토크나이저 로딩"""
    global BITSANDBYTES_AVAILABLE
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    
    print(f"'{model_name}' 모델 로딩 중...")
    print("A100 1개에 최적화된 설정으로 로딩합니다.")

    try:
        # 토크나이저 로딩
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # GPU 사용 가능 여부 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {device}")
        
        # GPU 메모리 정보 출력
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 메모리: {gpu_memory:.1f} GB")

        # 모델 로딩 설정
        model_kwargs = {}
        
        if BITSANDBYTES_AVAILABLE:
            print("BitsAndBytesConfig를 사용한 4-bit 양자화 설정")
            try:
                # A100 1개에 최적화된 4-bit 양자화 설정
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "offload_folder": "./offload",
                }
                print("4-bit 양자화 설정 완료")
            except Exception as e:
                print(f"BitsAndBytesConfig 설정 실패: {e}")
                print("기본 설정으로 fallback합니다.")
                BITSANDBYTES_AVAILABLE = False
        
        if not BITSANDBYTES_AVAILABLE:
            print("기본 float16 양자화 설정 사용")
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "offload_folder": "./offload",
            }

        print("모델 로딩 시작... (수 분 소요될 수 있습니다)")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # 추론 모드로 설정
        model.eval()
        
        # 메모리 정리
        clear_memory()

        print(f"'{model_name}' 모델 로딩 완료.")
        
        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.1f} GB allocated, {cached:.1f} GB cached")

        return model, tokenizer, device

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nCUDA out of memory 오류 발생!")
            print("다음 해결 방법들을 시도해보세요:")
            print("1. 더 작은 배치 크기 사용")
            print("2. max_new_tokens 값 줄이기")
            print("3. 더 적극적인 메모리 관리:")
            print("   - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 환경변수 설정")
            print("4. bitsandbytes 재설치")
        else:
            print(f"오류 발생: {e}")
        return None, None, None
    except Exception as e:
        if "bitsandbytes" in str(e).lower() or "quantization" in str(e).lower():
            print(f"\nBitsandbytes 관련 오류: {e}")
            print("\n해결 방법:")
            print("1. bitsandbytes 재설치:")
            print("   pip uninstall bitsandbytes -y")
            print("   pip install bitsandbytes")
        else:
            print(f"예상치 못한 오류 발생: {e}")
        return None, None, None

def interactive_chat():
    """대화형 채팅 시스템"""
    print("=" * 60)
    print("🤖 DeepSeek 대화형 채팅 시스템")
    print("=" * 60)
    
    # 모델 로딩
    model, tokenizer, device = load_model()
    
    if model is None:
        print("모델 로딩에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    print("\n✅ 모델 로딩 완료! 이제 질문해보세요.")
    print("💡 명령어:")
    print("  - 'quit', 'exit', '종료' : 프로그램 종료")
    print("  - 'clear', '클리어' : 화면 정리")
    print("  - 'memory', '메모리' : GPU 메모리 상태 확인")
    print("  - 'settings', '설정' : 생성 설정 변경")
    print("-" * 60)
    
    # 기본 생성 설정
    settings = {
        'max_new_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.9
    }
    
    conversation_count = 0
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input(f"\n[{conversation_count + 1}] 당신: ").strip()
            
            # 종료 명령어 체크
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 채팅을 종료합니다. 감사합니다!")
                break
            
            # 화면 정리 명령어
            elif user_input.lower() in ['clear', '클리어']:
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            # 메모리 상태 확인
            elif user_input.lower() in ['memory', '메모리']:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    cached = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"📊 GPU 메모리 상태:")
                    print(f"  - 할당됨: {allocated:.1f} GB")
                    print(f"  - 캐시됨: {cached:.1f} GB") 
                    print(f"  - 전체: {total:.1f} GB")
                    print(f"  - 사용률: {(allocated/total)*100:.1f}%")
                else:
                    print("CUDA가 사용 불가능합니다.")
                continue
            
            # 설정 변경
            elif user_input.lower() in ['settings', '설정']:
                print(f"\n⚙️ 현재 설정:")
                print(f"  - max_new_tokens: {settings['max_new_tokens']}")
                print(f"  - temperature: {settings['temperature']}")
                print(f"  - top_p: {settings['top_p']}")
                
                try:
                    new_max_tokens = input(f"새로운 max_new_tokens (현재: {settings['max_new_tokens']}): ").strip()
                    if new_max_tokens:
                        settings['max_new_tokens'] = int(new_max_tokens)
                    
                    new_temp = input(f"새로운 temperature (현재: {settings['temperature']}): ").strip()
                    if new_temp:
                        settings['temperature'] = float(new_temp)
                    
                    new_top_p = input(f"새로운 top_p (현재: {settings['top_p']}): ").strip()
                    if new_top_p:
                        settings['top_p'] = float(new_top_p)
                    
                    print("✅ 설정이 업데이트되었습니다!")
                except ValueError:
                    print("❌ 잘못된 값입니다. 설정이 변경되지 않았습니다.")
                continue
            
            # 빈 입력 체크
            elif not user_input:
                print("질문을 입력해주세요.")
                continue
            
            # DeepSeek에게 질문
            print(f"\n🤖 DeepSeek: ", end="", flush=True)
            
            answer = ask_deepseek(
                user_input, 
                model, 
                tokenizer, 
                device,
                max_new_tokens=settings['max_new_tokens'],
                temperature=settings['temperature'],
                top_p=settings['top_p']
            )
            
            print(answer)
            conversation_count += 1
            
            # 주기적으로 메모리 정리 (10번 대화마다)
            if conversation_count % 10 == 0:
                print("\n🔄 메모리 정리 중...")
                clear_memory()
                print("✅ 메모리 정리 완료")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Ctrl+C가 감지되었습니다.")
            user_choice = input("정말 종료하시겠습니까? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '예']:
                break
            else:
                print("계속 진행합니다...")
                continue
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("다시 시도해주세요.")
            continue

def main():
    """메인 함수"""
    interactive_chat()

if __name__ == "__main__":
    main()