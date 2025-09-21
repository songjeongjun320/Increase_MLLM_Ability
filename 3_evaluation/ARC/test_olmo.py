import os
import torch
import warnings
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 경고 및 로깅 설정
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 설정
MODEL_PATH = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/olmo-2-0425-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

def load_olmo_model_fixed():
    """수정된 OLMo 모델 로드 함수"""
    logger.info(f"Loading OLMo model from: {MODEL_PATH}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
    
    # PAD 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True
    )
    
    # 임베딩 크기 동기화 - 토크나이저 크기에 맞춤
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    logger.info(f"✅ Model loaded with embedding size: {model.get_input_embeddings().weight.shape[0]}")
    
    return model, tokenizer

def create_token_filter(tokenizer, max_vocab_size=50000):
    """
    안전한 토큰만 사용하도록 필터 생성
    - 낮은 ID의 잘 훈련된 토큰만 허용
    - 문제가 되는 고ID 토큰 차단
    """
    # 차단할 특정 단어들
    blocked_words = [
        "setattr", "PrivateKey", "TestCase", "ForcedSuppressWarnings",
        "komm", "aight", "ılı", "dernier", "cplusplus", "yscale",
        "GLOSS", "VERTISE", "obao", "iyor", "Mey"
    ]
    
    # 차단할 토큰 ID 수집
    blocked_ids = set()
    
    # 특정 단어들의 ID 수집
    for word in blocked_words:
        try:
            ids = tokenizer.encode(word, add_special_tokens=False)
            blocked_ids.update(ids)
        except:
            pass
    
    # 고ID 토큰 차단 (50000 이상)
    for i in range(max_vocab_size, len(tokenizer)):
        blocked_ids.add(i)
    
    return list(blocked_ids)

def safe_generate(model, tokenizer, prompt, max_new_tokens=50):
    """
    안전한 생성 함수
    - Top-k 필터링으로 고품질 토큰만 선택
    - 문제 토큰 차단
    - 보수적인 샘플링 파라미터
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # 차단할 토큰 ID 리스트 생성
    bad_words_ids = create_token_filter(tokenizer)
    
    # 보수적인 생성 파라미터
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.3,  # 매우 낮은 온도
        "top_k": 50,  # 상위 50개 토큰만 고려
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bad_words_ids": [[id] for id in bad_words_ids[:1000]],  # 처음 1000개만 (메모리 제한)
        "min_length": inputs['input_ids'].shape[1] + 5,  # 최소 5개 토큰은 생성
    }
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_config)
    
    # 입력 부분 제외하고 생성된 부분만 디코딩
    generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 후처리: 이상한 문자 제거
    import re
    generated_text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', generated_text)
    generated_text = ' '.join(generated_text.split())  # 중복 공백 제거
    
    return generated_text

def interactive_chat_fixed(model, tokenizer):
    """개선된 대화형 채팅"""
    print("\n" + "="*60)
    print("🤖 수정된 OLMo 채팅 모드")
    print("💡 안전한 토큰만 사용하도록 필터링됨")
    print("📝 종료: 'quit', 'exit', 'q'")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n👤 사용자: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 종료합니다!")
                break
            
            if not user_input:
                continue
            
            print("🤔 생성 중...")
            
            # 안전한 생성 함수 사용
            response = safe_generate(model, tokenizer, user_input)
            
            print(f"🤖 OLMo: {response}")
            
            # 품질 체크
            if len(response) < 10 or any(bad in response.lower() for bad in ['setattr', 'gloss', 'vertise']):
                print("⚠️ 출력 품질이 낮습니다. 모델 재조정이 필요할 수 있습니다.")
        
        except KeyboardInterrupt:
            print("\n\n👋 종료합니다!")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")
            continue

def test_model_quality(model, tokenizer):
    """모델 품질 테스트"""
    test_prompts = [
        "The capital of France is",
        "1 + 1 equals",
        "Hello, my name is",
        "The sun rises in the",
        "Water freezes at"
    ]
    
    print("\n🧪 모델 품질 테스트:")
    print("="*60)
    
    for prompt in test_prompts:
        response = safe_generate(model, tokenizer, prompt, max_new_tokens=10)
        print(f"입력: '{prompt}'")
        print(f"출력: '{response}'")
        print("-"*40)

def main():
    """메인 함수"""
    try:
        # 수정된 모델 로드
        model, tokenizer = load_olmo_model_fixed()
        
        # 품질 테스트
        test_model_quality(model, tokenizer)
        
        # 대화형 채팅
        interactive_chat_fixed(model, tokenizer)
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()