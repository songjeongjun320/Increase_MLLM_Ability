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

def ask_deepseek(question, model, tokenizer, device, max_new_tokens=200, temperature=0.7, top_p=0.9):
    # 메모리 정리
    clear_memory()
    
    # 모델에 따라 적절한 프롬프트 형식을 구성할 수 있습니다.
    # 예시: "Human: {question}\nAssistant:" 또는 간단히 질문만 전달
    # DeepSeek의 경우, 특별한 지침이 없다면 질문을 바로 전달해도 괜찮을 수 있습니다.
    # 여기서는 간단한 Q&A 형식을 사용해봅니다.
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
        "do_sample": True, # True로 설정해야 temperature, top_p 등이 적용됩니다.
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids, # 리스트로 전달
        "use_cache": True,  # KV 캐시 사용으로 메모리 효율성 향상
    }

    # 메모리 효율적인 생성을 위해 torch.no_grad() 사용
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

    # 입력 부분을 제외하고 생성된 텍스트만 디코딩
    # generated_ids[0] -> 배치 중 첫 번째 결과
    # input_ids.shape[-1] -> 입력 토큰의 길이
    answer_ids = generated_ids[0][input_ids.shape[-1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # 메모리 정리
    del input_ids, attention_mask, generated_ids
    clear_memory()

    return answer_text.strip()

def main():
    global BITSANDBYTES_AVAILABLE  # Add this line to fix the scope issue
    
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
                    bnb_4bit_use_double_quant=True,  # 더 나은 양자화 품질
                    bnb_4bit_quant_type="nf4",       # NF4 양자화 사용
                    bnb_4bit_compute_dtype=torch.bfloat16,  # 계산 시 bfloat16 사용
                )
                
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto",  # 자동으로 GPU 메모리에 맞게 분산
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,  # CPU 메모리 사용량 최소화
                    "offload_folder": "./offload",  # 필요시 디스크로 오프로드
                }
                print("4-bit 양자화 설정 완료")
            except Exception as e:
                print(f"BitsAndBytesConfig 설정 실패: {e}")
                print("기본 설정으로 fallback합니다.")
                BITSANDBYTES_AVAILABLE = False
        
        if not BITSANDBYTES_AVAILABLE:
            print("기본 float16 양자화 설정 사용")
            model_kwargs = {
                "torch_dtype": torch.float16,  # float16으로 메모리 절약
                "device_map": "auto",  # 자동으로 GPU 메모리에 맞게 분산
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

        # 질문 예시
        question = """
            Task Instruction: Given certain text, you need to predict the next
            word of it. Moreover, before your output, you could first give short
            thoughts about how you infer the next word based on the provided
            context.
            Here are five examples for the task:
            Example 0: {"우리는 가끔 온라인 쿠폰과 기타 특별 혜택을 제공합니다. <hCoT> Customers can explore additional ways to find deals beyond online coupons, like subscribing. </hCoT> 또는 제품 연구에 참여하고 싶다면 '홈 제품 배치'를 체크하고 몇 가지 질문에 답해주세요. 무엇을 기다리고 있나요?"}
            Example 1: {"'극성, 비양성 용매'의 예는 무엇인가요? 몇 가지 예로는 메틸렌 <hCoT> The context seeks examples of polar, aprotic solvents; methylene chloride may follow. </hCoT>클로라이드,디에틸 <hCoT> The context lists examples of polar, aprotic solvents like methylene chloride and diethyl ether. </hCoT>에터, 클로로폼 등이 있습니다. 극성 비양성 용매는 유의미한 전하 분리가 있는 <hCoT> A polar aprotic solvent lacks hydrogen bonds, focusing on its chemical properties as a molecule. </hCoT>분자(즉, 극성 용매)로, 용매로 사용될 수 있지만 <hCoT> Polar aprotic solvents have charge separation and do not donate hydrogen ions. </hCoT>산-염기평형을 일으키지 않는 용매입니다. 따라서 물과 플루오르화수소는 <hCoT> The context clarifies \"polar, aprotic solvents\" and excludes water and hydrogen fluoride. </hCoT>확실히 극성 분자이지만, 양성자(${H}^{+}$)를 쉽게 교환하므로 비양성 용매에 해당하지않습니다."}
            Example 2: {"마우스 드래그에 따라 다른 객체 주위에서 객체가 부드럽게 궤도 회전하도록 제한하기\n\n저는 Unity에서 작업하고 있습니다.\n\n저는 각각 2D에서 자유롭게 드래그할 수 있는 특별한 객체들을 가지고 있습니다.\n\n이 객체들은 다른 객체들과 너무 가까워질 수 없습니다 (최소 허용되는 Vector3.distance는 1.5f입니다).\n\n또한 특별한 커넥터들을 가지고 있는데, 이 커넥터는 이 특별한 객체들 중 2개를 자유롭게 연결할 수 있습니다. 두 객체가 <hCoT> The context describes dragging 2D objects with constraints and potential actions upon connecting them. </hCoT>연결되면, 드래그하는 것이 더 제한됩니다.\n\n이 2개의 객체는 서로 너무 멀어질 수 없습니다 (최대 허용되는 Vector3.distance는 5f입니다).\n\n이 2가지 제한은 작동하지만, 문제는 이 제한이 발생하면 객체들이 거의 고정된 상태로 남아 있으며, 제한을 넘어서려고 할 때 객체가 고정된 자리에 그대로 멈추게 된다는 점입니다.
            Example 3: {"대체 대수학 복습 | KS3 수학 자료\n\n## 알아야 할 것들\n\n기억할 사항:\n\n• 대체는 단순히 문자를 숫자로 교체하는 것을 의미합니다.\n• 숫자와 <hCoT> Context explains substitution in algebra, noting no \u201c\u00d7\u201d before a letter. </hCoT> 문자 사이에 $\\times$ 기호가 숨겨져 있으므로, 이를 기억해야 합니다!\n• 분수는 또 다른 방식으로 나눗셈 문제를 나타내는 방법입니다.\n\n그렇다면 대체란 무엇일까요? 대체는 한 가지를 다른 것으로 '바꾸는' 것을 의미합니다. <hCoT> The context explains substitution in algebra, likely leading to an example related to maths. </hCoT> 수학에서는 주로 문자를 숫자로 바꾸는 것을 의미합니다.\n\n**대체를 사용하여 $x + 7$의 값을 구하시오, 여기서 x = 12입니다.**\n\n여기서 $x = 12$라고 주어졌으므로, 우리가 해야 할 일은 식에서 $x$를 12로 바꾸는 것뿐입니다!\n\n$$x+7=12+7=19$$\n\n쉽죠! 다른 연산도 똑같이 처리하면 됩니다!\n\n**대체를 사용하여 $x - 4$의 <hCoT> The context explains using substitution in algebra to simplify expressions, introducing a new problem. </hCoT> 값을 구하시오, 여기서 x = 15 <hCoT> The text explains substitution in algebra, replacing variables with numbers, using examples like \\( x - 4 \\). </hCoT> 입니다.**\n\n $$x-4=15-4=11$$\n\n곱셈 문제는 조금 다릅니다. 왜냐하면 숫자와 문자 사이에 숨겨진 $\\times$ 기호가 있다는 것을 기억해야 하기 때문입니다.\n\n**대체를 사용하여 $5x$의 값을 구하시오, 여기서 x = 13입니다.**\n\n$$5x=5\\times x=5\\times13=65$$\n\n나눗셈 문제는 두 가지 형태로 나올 수 있습니다:\n\n**대체를 사용하여 $x\\div3$의 값을 구하시오, 여기서 x = 9입니다.**\n\n$$x\\div3=9\\div3=3$$\n\n또는 분수 형태로 나와서 변환해야 할 수 있습니다:\n\n**대체를 사용하여 $\\frac{20}{x}$의 값을 구하시오, 여기서 x = 5입니다.**\n\n$$\\frac{20}{x}=20\\div x=20\\div5=4$$\n\n## KS3 수학 <hCoT> The context explains substitution in math, guiding KS3 students with examples. </hCoT> 복습 카드\n\n(77개의 리뷰) ₤8.99\n\n## 예시 문제들\n\n$$12x=12\\times x=12\\times7=84$$\n\n$$\\dfrac{x}{9}=x\\div9=54\\div9=6$$\n\n## KS3 수학 복습 카드\n\n(77개의 리뷰) ₤8.99\n• 모든 주요 KS2 수학 SATs 주제 포함\n• 각 주제에 대한 연습 문제와 답안 제공"}
            Example 4: {"랜달즈빌, 뉴욕의 이사 및 이사 회사: Movers MAX .:\nMovers MAX 이사 디렉토리는 <hCoT> The context suggests a comprehensive resource directory for all moving services and information. </hCoT> 이사 자원을 위한 원스톱 소스입니다.
            Now please give me your prediction for the thought and next word
            based on the following context:
            {미주리에서 초보자를 위한 바비큐 }

            Thought
            Next Word:
        """

        answer = ask_deepseek(question, model, tokenizer, device, max_new_tokens=250)
        print(f"답변: {answer}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nCUDA out of memory 오류 발생!")
            print("다음 해결 방법들을 시도해보세요:")
            print("1. 더 작은 배치 크기 사용")
            print("2. max_new_tokens 값 줄이기 (현재 250 -> 100)")
            print("3. 더 적극적인 메모리 관리:")
            print("   - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 환경변수 설정")
            print("   - export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
            print("4. bitsandbytes 재설치:")
            print("   pip uninstall bitsandbytes -y")
            print("   pip install bitsandbytes")
            print("   pip install --upgrade accelerate transformers")
        else:
            print(f"오류 발생: {e}")
    except Exception as e:
        if "bitsandbytes" in str(e).lower() or "quantization" in str(e).lower():
            print(f"\nBitsandbytes 관련 오류: {e}")
            print("\n해결 방법:")
            print("1. bitsandbytes 재설치:")
            print("   pip uninstall bitsandbytes -y")
            print("   pip install bitsandbytes")
            print("2. 또는 CUDA 버전에 맞는 설치:")
            print("   pip install bitsandbytes --upgrade --force-reinstall")
            print("3. 환경 확인:")
            print("   python -c \"import bitsandbytes; print(bitsandbytes.__version__)\"")
        else:
            print(f"예상치 못한 오류 발생: {e}")

if __name__ == "__main__":
    main()