# -*- coding: utf-8 -*-
import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정 (성능 최적화)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 오프라인 모드 강제

# bitsandbytes 가져오기 시도
try:
    from transformers import BitsAndBytesConfig
    
    BITSANDBYTES_AVAILABLE = True
    print("✅ BitsAndBytesConfig를 성공적으로 import했습니다.")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("⚠️ BitsAndBytesConfig import 실패 - FP16 양자화를 사용합니다.")

def setup_torch_optimizations():
    """PyTorch 최적화 설정"""
    # 메모리 관리 최적화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # CUDA 캐시 최적화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 메모리 프래그멘테이션 방지
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

def clear_memory():
    """효율적인 메모리 정리 함수"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class OptimizedDeepSeekChat:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_config = None
        
        # 캐시된 토큰 시퀀스들
        self.cached_tokens = {}
        
        # 커스텀 프롬프트 템플릿 정의
        self.prompt_template = """Task Instruction: Given certain text, you need to predict the next word of it. Moreover, before your output, you could first give short thoughts about how you infer the next word based on the provided context.

Here are five examples for the task:

Example 0: 우리는 가끔 온라인 쿠폰과 기타 특별 혜택을 제공합니다. <hCoT> Customers can explore additional ways to find deals beyond online coupons, like subscribing. </hCoT> 또는 제품 연구에 참여하고 싶다면 '홈 제품 배치'를 체크하고 몇 가지 질문에 답해주세요. 무엇을 기다리고 있나요?

Example 1: 방정식 2x + 5 = 17을 풀어보세요. 먼저 양변에서 5를 <hCoT> The context presents an equation 2x + 5 = 17 and mentions subtracting 5 from both sides, so the next word should be '빼면' to describe the subtraction operation. </hCoT> 빼면 2x = 12가 됩니다. 그 다음 양변을 2로 <hCoT> The context shows 2x = 12 and mentions dividing both sides by 2, so the next word should be '나누면' to complete the division step. </hCoT> 나누면 x = 6이 답입니다.

Example 2: Unity에서 2D 객체를 드래그할 때 다른 객체와의 최소 거리는 1.5f입니다. 두 객체가 <hCoT> The context describes distance constraints for 2D objects in Unity, so the next word should be '연결되면' to describe what happens when objects connect. </hCoT> 연결되면 드래그가 더 제한됩니다.

Example 3: 대수학에서 대체는 문자를 숫자로 바꾸는 것입니다. 숫자와 <hCoT> The context explains algebraic substitution involving numbers, so the next word should be '문자' as algebra deals with both numbers and variables. </hCoT> 문자 사이에는 곱셈 기호가 숨겨져 있습니다.

Example 4: 랜달즈빌 이사 회사 Movers MAX 디렉토리는 <hCoT> The context introduces a moving company directory called Movers MAX, so the next word should be '이사' to specify what kind of resources this directory provides. </hCoT> 이사 자원을 위한 원스톱 소스입니다.

Now please give me your prediction for the thought and next word based on the following context:

{user_input_context}

Thought:
Next Word:"""
        
        # 성능 최적화 설정
        setup_torch_optimizations()
        
    def load_model(self) -> bool:
        """최적화된 모델 로딩"""
        print(f"🚀 로컬 경로 '{self.model_path}'에서 모델 로딩 중...")
        print("🔧 A100 최적화 설정 적용... (안정적인 eager attention)")

        try:
            # 모델 설정 파일 확인
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                print(f"❌ config.json을 찾을 수 없습니다: {config_path}")
                return False
            
            # 토크나이저 로딩
            print("📝 토크나이저 로딩 중...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True, 
                    local_files_only=True,
                    use_fast=False,  # fast tokenizer 비활성화로 호환성 개선
                    padding_side="left"
                )
                print("✅ AutoTokenizer로 로딩 성공")
            except Exception as e:
                print(f"❌ AutoTokenizer 로딩 실패: {e}")
                return False
            
            # pad_token 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 디바이스 설정
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🎯 사용 장치: {self.device}")
            
            # GPU 메모리 정보
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"💾 GPU 메모리: {gpu_memory:.1f} GB")

            # 모델 로딩
            try:
                model_kwargs = self._get_optimized_model_config()
                print("🔥 AutoModelForCausalLM으로 모델 로딩 시작...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    **model_kwargs
                )
                print("✅ AutoModelForCausalLM으로 로딩 성공")
            except Exception as e:
                print(f"⚠️ 양자화 모델 로딩 실패, 기본 설정으로 재시도: {e}")
                # 양자화 없이 기본 설정으로 재시도
                try:
                    basic_kwargs = {
                        "trust_remote_code": True,
                        "local_files_only": True,
                        "low_cpu_mem_usage": True,
                        "device_map": "auto",
                        "torch_dtype": torch.float16,
                        "attn_implementation": "eager",
                    }
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **basic_kwargs
                    )
                    print("✅ 기본 설정으로 로딩 성공")
                except Exception as e2:
                    print(f"❌ 모델 로딩 완전 실패: {e2}")
                    return False
            
            # 추론 최적화
            self.model.eval()
            
            # Torch compile 사용 (PyTorch 2.0+) - 선택적 적용
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                print("⚡ Torch compile 적용 시도 중...")
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("✅ Torch compile 적용 완료")
                except Exception as e:
                    print(f"⚠️ Torch compile 실패 (무시하고 계속): {e}")
            
            # 생성 설정 최적화
            self._setup_generation_config()
            
            # 메모리 정리
            clear_memory()
            
            # 메모리 사용량 확인
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(f"📊 GPU 메모리 사용량: {allocated:.1f} GB allocated, {cached:.1f} GB cached")
            
            print("✅ 모델 로딩 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_optimized_model_config(self) -> Dict[str, Any]:
        """최적화된 모델 설정 반환 (호환성 개선)"""
        base_config = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "attn_implementation": "eager",  # FlashAttention 대신 안정적인 eager 사용
        }
        
        # 양자화 설정
        if BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
            print("🔧 4-bit 양자화 설정 (BitsAndBytes)")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                base_config.update({
                    "quantization_config": bnb_config,
                    "torch_dtype": torch.bfloat16,
                })
            except Exception as e:
                print(f"⚠️ 양자화 설정 실패, FP16 사용: {e}")
                base_config["torch_dtype"] = torch.float16
        else:
            print("🔧 FP16 설정")
            base_config["torch_dtype"] = torch.float16
        
        return base_config
    
    def _setup_generation_config(self):
        """최적화된 생성 설정"""
        self.generation_config = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1,  # 빠른 추론을 위해 beam search 비활성화
        }
    
    def _build_optimized_prompt(self, user_input_context: str) -> str:
        """커스텀 프롬프트 템플릿을 사용한 프롬프트 생성"""
        # DeepSeek 형식 확인 후 적용
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # 채팅 템플릿이 있는 경우
            messages = [{"role": "user", "content": self.prompt_template.format(user_input_context=user_input_context)}]
            try:
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                # 채팅 템플릿 적용 실패시 기본 형식 사용
                full_prompt = f"""<｜start▁header▁id｜>user<｜end▁header▁id｜>

{self.prompt_template.format(user_input_context=user_input_context)}<｜eot▁id｜><｜start▁header▁id｜>assistant<｜end▁header▁id｜>

"""
        else:
            # 기본 DeepSeek 형식
            full_prompt = f"""<｜start▁header▁id｜>user<｜end▁header▁id｜>

{self.prompt_template.format(user_input_context=user_input_context)}<｜eot▁id｜><｜start▁header▁id｜>assistant<｜end▁header▁id｜>

"""
        return full_prompt
    
    def ask_deepseek(self, user_input_context: str, max_new_tokens: int = 1024, **kwargs) -> str:
        """커스텀 프롬프트를 사용한 DeepSeek 추론"""
        if self.model is None or self.tokenizer is None:
            return "❌ 모델이 로딩되지 않았습니다."
        
        # 생성 설정 업데이트
        generation_config = self.generation_config.copy()
        generation_config.update(kwargs)
        generation_config["max_new_tokens"] = max_new_tokens
        
        # 커스텀 프롬프트 생성
        prompt = self._build_optimized_prompt(user_input_context)
        
        # 토크나이징 (배치 처리 준비)
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                return_attention_mask=True,
                truncation=True,
                max_length=4096,  # 최대 길이 제한
                padding=False
            )
        except Exception as e:
            print(f"❌ 토크나이징 실패: {e}")
            return "토크나이징 중 오류가 발생했습니다."
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        print(f"🤔 컨텍스트 처리 중: {user_input_context[:50]}{'...' if len(user_input_context) > 50 else ''}")
        
        try:
            # 최적화된 추론
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):  # Mixed precision
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_config
                    )
            
            # 응답 디코딩
            answer_ids = generated_ids[0][input_ids.shape[-1]:]
            full_response = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
            
            # 최종 답변 추출
            final_answer = self._extract_final_answer(full_response)
            
            # 메모리 정리
            del input_ids, attention_mask, generated_ids
            clear_memory()
            
            return final_answer
            
        except Exception as e:
            print(f"❌ 추론 중 오류 발생: {e}")
            clear_memory()
            return f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
    
    def _extract_final_answer(self, response: str) -> str:
        """DeepSeek-R1 응답에서 최종 답변 추출 (최적화)"""
        # thinking 태그 제거
        if "<｜thinking｜>" in response and "<｜/thinking｜>" in response:
            parts = response.split("<｜/thinking｜>")
            final_answer = parts[1].strip() if len(parts) > 1 else response.strip()
        else:
            final_answer = response.strip()
        
        # 불필요한 토큰들 일괄 제거
        unwanted_tokens = ["<｜thinking｜>", "<｜/thinking｜>", "<｜eot▁id｜>", "<｜end▁of▁text｜>"]
        for token in unwanted_tokens:
            final_answer = final_answer.replace(token, "")
        
        return final_answer.strip()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """메모리 상태 반환"""
        if not torch.cuda.is_available():
            return {"error": "CUDA 사용 불가"}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "total_gb": total,
            "usage_percent": (allocated / total) * 100
        }

def check_model_files(model_path: str):
    """모델 파일 구조 확인"""
    print(f"🔍 모델 파일 구조 확인: {model_path}")
    
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    optional_files = [
        "tokenizer.model",
        "vocab.txt",
        "merges.txt"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} 존재")
        else:
            print(f"❌ {file} 누락")
    
    for file in optional_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} 존재 (선택사항)")
    
    # 모델 가중치 파일 확인
    weight_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
    print(f"📦 모델 가중치 파일: {len(weight_files)}개")
    for file in weight_files[:5]:  # 처음 5개만 표시
        print(f"  - {file}")
    if len(weight_files) > 5:
        print(f"  ... 및 {len(weight_files) - 5}개 더")

def interactive_chat():
    """커스텀 프롬프트를 사용한 대화형 채팅 시스템"""
    print("=" * 70)
    print("🚀 DeepSeek 다음 단어 예측 시스템 (커스텀 프롬프트)")
    print("=" * 70)
    
    # 모델 경로
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/DeepSeek_R1_Distill_Llama_70B"
    
    # 모델 파일 구조 확인
    check_model_files(model_path)
    
    # 최적화된 채팅 인스턴스 생성
    chat_system = OptimizedDeepSeekChat(model_path)
    
    # 모델 로딩
    if not chat_system.load_model():
        print("❌ 모델 로딩에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    print("\n✅ 모델 로딩 완료! (다음 단어 예측 모드)")
    print("💡 사용법: 문맥을 입력하면 다음에 올 단어를 예측합니다.")
    print("💡 명령어:")
    print("  - 'quit', 'exit', '종료' : 프로그램 종료")
    print("  - 'clear', '클리어' : 화면 정리")
    print("  - 'memory', '메모리' : GPU 메모리 상태 확인")
    print("  - 'example', '예시' : 사용 예시 보기")
    print("-" * 70)
    
    # 성능 설정
    settings = {
        'max_new_tokens': 512,  # 다음 단어 예측이므로 짧게
        'temperature': 0.3,     # 더 확정적인 예측을 위해 낮게
        'top_p': 0.9,
    }
    
    conversation_count = 0
    
    # 첫 번째 추론 워밍업
    print("🔥 모델 워밍업 중...")
    try:
        warmup_response = chat_system.ask_deepseek("안녕하세요. 오늘", max_new_tokens=50)
        print(f"🔥 워밍업 완료")
    except Exception as e:
        print(f"⚠️ 워밍업 중 오류: {e}")
    
    while True:
        try:
            user_input = input(f"\n[{conversation_count + 1}] 문맥 입력: ").strip()
            
            # 명령어 처리
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 다음 단어 예측 시스템을 종료합니다. 감사합니다!")
                break
            elif user_input.lower() in ['clear', '클리어']:
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif user_input.lower() in ['memory', '메모리']:
                stats = chat_system.get_memory_stats()
                if "error" not in stats:
                    print(f"📊 GPU 메모리 상태:")
                    print(f"  - 할당됨: {stats['allocated_gb']:.1f} GB")
                    print(f"  - 캐시됨: {stats['cached_gb']:.1f} GB")
                    print(f"  - 전체: {stats['total_gb']:.1f} GB")
                    print(f"  - 사용률: {stats['usage_percent']:.1f}%")
                else:
                    print("CUDA가 사용 불가능합니다.")
                continue
            elif user_input.lower() in ['example', '예시']:
                print("📋 사용 예시:")
                print("  입력: '오늘 날씨가 정말'")
                print("  출력: Thought: [모델의 추론 과정] Next Word: 좋네요")
                print("")
                print("  입력: '파이썬에서 리스트를'")
                print("  출력: Thought: [모델의 추론 과정] Next Word: 생성하려면")
                continue
            elif not user_input:
                print("문맥을 입력해주세요.")
                continue
            
            # DeepSeek에게 다음 단어 예측 요청
            import time
            start_time = time.time()
            
            answer = chat_system.ask_deepseek(
                user_input,  # user_input_context로 사용
                max_new_tokens=settings['max_new_tokens'],
                temperature=settings['temperature'],
                top_p=settings['top_p']
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"\n🤖 DeepSeek 예측 결과 ({response_time:.1f}초):")
            print(f"{answer}")
            conversation_count += 1
            
            # 주기적 메모리 정리
            if conversation_count % 5 == 0:
                clear_memory()
            
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
    # 시스템 체크
    print("🔍 시스템 환경 체크...")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 장치: {torch.cuda.get_device_name()}")
    
    # Transformers 버전 확인
    try:
        import transformers
        print(f"Transformers 버전: {transformers.__version__}")
    except:
        pass
    
    interactive_chat()

if __name__ == "__main__":
    main()