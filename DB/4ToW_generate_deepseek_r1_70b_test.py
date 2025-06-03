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

# 멀티 GPU 설정 - A100 80GB x2 최적화
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPU 0, 1 사용

# A100 80GB 최적화를 위한 추가 환경 변수
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True,roundup_power2_divisions:8"
os.environ["NCCL_DEBUG"] = "WARN"  # 멀티 GPU 통신 최적화

# bitsandbytes 가져오기 시도
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    print("✅ BitsAndBytesConfig successfully imported")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("⚠️ BitsAndBytesConfig import failed - using FP16 quantization")

# FlashAttention 사용 가능 여부 확인
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ FlashAttention available for maximum performance")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️ FlashAttention not available - using optimized attention")

def setup_torch_optimizations():
    """PyTorch 최적화 설정 - A100 80GB 특화"""
    # 메모리 관리 최적화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.enable_flash_sdp(True)  # Scaled Dot Product Attention 최적화
    
    # CUDA 캐시 최적화 - A100 80GB 특화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # A100 80GB를 위한 더 큰 메모리 풀 설정
        torch.cuda.set_per_process_memory_fraction(0.95)  # 76GB 사용 가능

def clear_memory():
    """효율적인 메모리 정리 함수 (A100 80GB x2)"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()  # IPC 메모리 정리

def get_multi_gpu_device_map(num_gpus: int = 2):
    """A100 80GB x2 최적 디바이스 맵 생성"""
    if num_gpus == 2:
        # A100 80GB x2 최적 분배 - 70GB 활용
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": 1,
            "lm_head": 1,
        }
        
        # 레이어를 두 GPU에 균등 분배 (70B 모델 기준)
        num_layers = 80  # DeepSeek-R1 Distill Llama 70B 레이어 수
        layers_per_gpu = num_layers // 2
        
        for i in range(num_layers):
            if i < layers_per_gpu:
                device_map[f"model.layers.{i}"] = 0
            else:
                device_map[f"model.layers.{i}"] = 1
        
        return device_map
    else:
        return "auto"

class OptimizedDeepSeekChat:
    def __init__(self, model_path: str, num_gpus: int = 2):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_config = None
        
        # 고성능 캐시 시스템
        self.cached_tokens = {}
        self.kv_cache = None
        
        # 커스텀 프롬프트 템플릿 정의
        self.prompt_template = """Task Instruction: Given certain text, you need to predict the next word of it. Moreover, before your output, you could first give short thoughts about how you infer the next word based on the provided context.

Here are five examples for the task:

Example 0: 우리는 가끔 온라인 쿠폰과 기타 특별 혜택을 제공합니다. <hCoT> Customers can explore additional ways to find deals beyond online coupons, like subscribing. </hCoT> 또는 제품 연구에 참여하고 싶다면 '홈 제품 배치'를 체크하고 몇 가지 질문에 답해주세요. 무엇을 기다리고 있나요?

Example 1: 방정식 2x + 5 = 17을 풀어보세요. 먼저 양변에서 5를 <hCoT> The context presents an equation 2x + 5 = 17 and mentions subtracting 5 from both sides, so the next word should be '빼면' to describe the subtraction operation. </hCoT> 빼면 2x = 12가 됩니다. 그 다음 양변을 2로 <hCoT> The context shows 2x = 12 and mentions dividing both sides by 2, so the next word should be '나누면' to complete the division step. </hCoT> 나누면 x = 6이 답입니다.

Example 2: Unity에서 2D 객체를 드래그할 때 다른 객체와의 최소 거리는 1.5f입니다. 두 객체가 <hCoT> The context describes distance constraints for 2D objects in Unity, so the next word should be '연결되면' to describe what happens when objects connect. </hCoT> 연결되면 드래그가 더 제한됩니다.

Example 3: 대수학에서 대체는 문자를 숫자로 바꾸는 것입니다. 숫자와 <hCoT> The context explains algebraic substitution involving numbers, so the next word should be '문자' as algebra deals with both numbers and variables. </hCoT> 문자 사이에는 곱셈 기호가 숨겨져 있습니다.

Example 4: 랜달즈빌 이사 회사 Movers MAX 디렉토리는 <hCoT> The context introduces a moving company directory called Movers MAX, so the next word should be '이사' to specify what kind of resources this directory provides. </hCoT> 이사 자원을 위한 원스톱 소스입니다.

Now please give me a pair of your prediction for the thought and next word based on the following context:

{user_input_context}

Thought:
Next Word:"""
        
        # A100 80GB 성능 최적화 설정
        setup_torch_optimizations()
        
    def load_model(self) -> bool:
        """A100 80GB x2 최적화된 모델 로딩"""
        print(f"🚀 Loading model from local path '{self.model_path}'...")
        print(f"🔧 Applying A100 80GB x{self.num_gpus} optimization settings...")

        # GPU 정보 출력
        if torch.cuda.is_available():
            print(f"🎯 Available GPUs: {torch.cuda.device_count()}")
            for i in range(min(self.num_gpus, torch.cuda.device_count())):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA is not available")
            return False

        try:
            # 모델 설정 파일 확인
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                print(f"❌ config.json not found: {config_path}")
                return False
            
            # 토크나이저 로딩
            print("📝 Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True, 
                    local_files_only=True,
                    use_fast=True,  # Fast tokenizer for A100 optimization
                    padding_side="left"
                )
                print("✅ AutoTokenizer loaded successfully")
            except Exception as e:
                print(f"❌ AutoTokenizer loading failed: {e}")
                return False
            
            # pad_token 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # A100 80GB x2 디바이스 맵 생성
            device_map = get_multi_gpu_device_map(self.num_gpus)
            print(f"🗺️ A100 80GB x2 device map created")
            
            # 모델 로딩
            try:
                model_kwargs = self._get_optimized_model_config(device_map)
                print("🔥 Starting A100 80GB x2 optimized model loading...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    **model_kwargs
                )
                print("✅ A100 80GB x2 model loading successful")
            except Exception as e:
                print(f"⚠️ Quantized model loading failed, retrying with basic settings: {e}")
                # 양자화 없이 기본 설정으로 재시도
                try:
                    basic_kwargs = {
                        "trust_remote_code": True,
                        "local_files_only": True,
                        "low_cpu_mem_usage": True,
                        "device_map": device_map,
                        "torch_dtype": torch.bfloat16,  # A100에서 bfloat16이 더 빠름
                        "attn_implementation": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "eager",
                        "max_memory": {0: "70GB", 1: "70GB"},  # A100 80GB 최대 활용
                    }
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **basic_kwargs
                    )
                    print("✅ Basic settings A100 80GB x2 loading successful")
                except Exception as e2:
                    print(f"❌ Model loading completely failed: {e2}")
                    return False
            
            # 추론 최적화
            self.model.eval()
            
            # A100 80GB 특화 컴파일 최적화
            try:
                # PyTorch 2.0+ 컴파일 최적화
                if hasattr(torch, 'compile'):
                    print("🚀 Applying PyTorch compile optimization for A100...")
                    self.model = torch.compile(
                        self.model, 
                        mode="max-autotune",  # 최대 성능 모드
                        fullgraph=False,      # 안정성을 위해
                        dynamic=True         # 동적 형태 지원
                    )
                    print("✅ PyTorch compile optimization applied")
            except Exception as e:
                print(f"⚠️ Compile optimization failed: {e}")
            
            # 메인 디바이스 설정 (첫 번째 GPU)
            self.device = torch.device("cuda:0")
            
            # A100 80GB 특화 생성 설정
            self._setup_generation_config()
            
            # 메모리 정리
            clear_memory()
            
            # A100 80GB x2 메모리 사용량 확인
            print("📊 A100 80GB x2 memory usage:")
            total_allocated = 0
            total_cached = 0
            for i in range(min(self.num_gpus, torch.cuda.device_count())):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total_allocated += allocated
                total_cached += cached
                print(f"  GPU {i}: {allocated:.1f} GB allocated, {cached:.1f} GB cached")
            print(f"  Total: {total_allocated:.1f} GB allocated, {total_cached:.1f} GB cached")
            
            print("✅ A100 80GB x2 model loading complete!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_optimized_model_config(self, device_map) -> Dict[str, Any]:
        """A100 80GB x2 최적화된 모델 설정 반환"""
        base_config = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "attn_implementation": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "eager",
            "max_memory": {0: "70GB", 1: "70GB"},  # A100 80GB 각각 70GB까지 사용
        }
        
        # A100 80GB를 위한 고성능 양자화 설정
        if BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
            print("🔧 A100 80GB optimized 4-bit quantization (maximum performance)")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # 4bit 양자화로 더 많은 메모리 확보
                    bnb_4bit_compute_dtype=torch.bfloat16,  # A100에서 bfloat16이 최적
                    bnb_4bit_use_double_quant=True,  # 더블 양자화로 성능 향상
                    bnb_4bit_quant_type="nf4",  # NormalFloat 4bit
                    llm_int8_enable_fp32_cpu_offload=False,  # A100에서는 CPU 오프로드 불필요
                )
                base_config.update({
                    "quantization_config": bnb_config,
                    "torch_dtype": torch.bfloat16,
                })
            except Exception as e:
                print(f"⚠️ Quantization setup failed, using bfloat16: {e}")
                base_config["torch_dtype"] = torch.bfloat16
        else:
            print("🔧 A100 80GB bfloat16 configuration")
            base_config["torch_dtype"] = torch.bfloat16
        
        return base_config
    
    def _setup_generation_config(self):
        """A100 80GB 특화 생성 설정"""
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
            "max_length": 8192,  # A100 80GB에서 더 긴 컨텍스트 지원
            "early_stopping": True,  # 불필요한 생성 중단
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
        """A100 80GB x2 고성능 DeepSeek 추론"""
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded"
        
        # 생성 설정 업데이트
        generation_config = self.generation_config.copy()
        generation_config.update(kwargs)
        generation_config["max_new_tokens"] = max_new_tokens
        
        # 커스텀 프롬프트 생성
        prompt = self._build_optimized_prompt(user_input_context)
        
        # A100 최적화 토크나이징
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                return_attention_mask=True,
                truncation=True,
                max_length=6144,  # A100 80GB에서 더 긴 컨텍스트
                padding=False
            )
        except Exception as e:
            print(f"❌ Tokenization failed: {e}")
            return "Tokenization error occurred"
        
        # 메인 GPU로 입력 전송
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)  # 비동기 전송
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
        
        print(f"🤔 Processing context on A100 80GB x2: {user_input_context[:50]}{'...' if len(user_input_context) > 50 else ''}")
        
        try:
            # A100 80GB 최적화된 추론
            with torch.no_grad():
                # bfloat16 autocast for A100
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
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
            print(f"❌ Inference error: {e}")
            clear_memory()
            return f"Sorry, an error occurred during processing: {str(e)}"
    
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
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """A100 80GB x2 메모리 상태 반환"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_stats = []
        total_allocated = 0
        total_cached = 0
        total_memory = 0
        
        for i in range(min(self.num_gpus, torch.cuda.device_count())):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            gpu_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            gpu_stats.append({
                "gpu_id": i,
                "allocated_gb": allocated,
                "cached_gb": cached,
                "total_gb": gpu_total,
                "usage_percent": (allocated / gpu_total) * 100
            })
            
            total_allocated += allocated
            total_cached += cached
            total_memory += gpu_total
        
        return {
            "gpu_stats": gpu_stats,
            "total_allocated_gb": total_allocated,
            "total_cached_gb": total_cached,
            "total_memory_gb": total_memory,
            "total_usage_percent": (total_allocated / total_memory) * 100
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
    """A100 80GB x2를 사용한 고성능 대화형 채팅 시스템"""
    print("=" * 70)
    print("🚀 DeepSeek Next Word Prediction System (A100 80GB x2 Optimized)")
    print("=" * 70)
    
    # 모델 경로
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/DeepSeek_R1_Distill_Llama_70B"
    
    # 모델 파일 구조 확인
    check_model_files(model_path)
    
    # A100 80GB x2 채팅 인스턴스 생성
    chat_system = OptimizedDeepSeekChat(model_path, num_gpus=2)
    
    # 모델 로딩
    if not chat_system.load_model():
        print("❌ Model loading failed. Terminating program.")
        return
    
    print("\n✅ A100 80GB x2 model loading complete! (Next word prediction mode)")
    print("💡 Usage: Input context to predict the next word")
    print("💡 Commands:")
    print("  - 'quit', 'exit', '종료' : Exit program")
    print("  - 'clear', '클리어' : Clear screen")
    print("  - 'memory', '메모리' : Check A100 80GB x2 memory status")
    print("  - 'example', '예시' : View usage examples")
    print("-" * 70)
    
    # A100 80GB 최적화 성능 설정
    settings = {
        'max_new_tokens': 3072,  # A100 80GB에서 더 긴 생성
        'temperature': 0.8,      # 균형잡힌 창의성
        'top_p': 0.9,
        'repetition_penalty': 1.05,
    }
    
    conversation_count = 0
    
    # A100 80GB 워밍업
    print("🔥 A100 80GB x2 model warmup...")
    try:
        warmup_response = chat_system.ask_deepseek("안녕하세요. 오늘", max_new_tokens=50)
        print(f"🔥 Warmup complete")
    except Exception as e:
        print(f"⚠️ Warmup error: {e}")
    
    while True:
        try:
            user_input = input(f"\n[{conversation_count + 1}] 문맥 입력: ").strip()
            
            # 명령어 처리
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 Terminating A100 80GB x2 next word prediction system. Thank you!")
                break
            elif user_input.lower() in ['clear', '클리어']:
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif user_input.lower() in ['memory', '메모리']:
                stats = chat_system.get_memory_stats()
                if "error" not in stats:
                    print(f"📊 A100 80GB x2 memory status:")
                    for gpu_stat in stats['gpu_stats']:
                        print(f"  GPU {gpu_stat['gpu_id']}:")
                        print(f"    - Allocated: {gpu_stat['allocated_gb']:.1f} GB")
                        print(f"    - Cached: {gpu_stat['cached_gb']:.1f} GB")
                        print(f"    - Total: {gpu_stat['total_gb']:.1f} GB")
                        print(f"    - Usage: {gpu_stat['usage_percent']:.1f}%")
                    print(f"  Total usage: {stats['total_usage_percent']:.1f}%")
                else:
                    print("CUDA not available")
                continue
            elif user_input.lower() in ['example', '예시']:
                print("📋 Usage examples:")
                print("  Input: '오늘 날씨가 정말'")
                print("  Output: Thought: [Model's reasoning process] Next Word: 좋네요")
                print("")
                print("  Input: '파이썬에서 리스트를'")
                print("  Output: Thought: [Model's reasoning process] Next Word: 생성하려면")
                continue
            elif not user_input:
                print("Please input context")
                continue
            
            # DeepSeek A100 80GB x2 고성능 추론
            import time
            start_time = time.time()
            
            answer = chat_system.ask_deepseek(
                user_input,  # user_input_context로 사용
                max_new_tokens=settings['max_new_tokens'],
                temperature=settings['temperature'],
                top_p=settings['top_p'],
                repetition_penalty=settings['repetition_penalty']
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"\n🤖 DeepSeek A100 80GB x2 prediction result ({response_time:.1f}s):")
            print(f"{answer}")
            conversation_count += 1
            
            # A100 80GB 주기적 메모리 정리
            if conversation_count % 5 == 0:  # A100 80GB에서는 덜 자주 정리
                clear_memory()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Ctrl+C detected")
            user_choice = input("Really want to exit? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '예']:
                break
            else:
                print("Continuing...")
                continue
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            print("Please try again")
            continue

def main():
    """메인 함수"""
    # 시스템 체크
    print("🔍 System environment check...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name()}")
    
    # Transformers 버전 확인
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except:
        pass
    
    interactive_chat()

if __name__ == "__main__":
    main()