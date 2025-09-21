import os
import torch
import warnings
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# 경고 및 로깅 설정
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 설정
MODEL_PATH = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/olmo-2-0425-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

def load_olmo_model():
    """OLMo 모델과 토크나이저 로드"""
    logger.info(f"Loading OLMo model from: {MODEL_PATH}")
    
    # 토크나이저 로드 - 기본 설정으로 시도
    logger.info("🔧 STEP 1: 기본 토크나이저 로드")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
    
    # 토크나이저 기본 정보 확인
    logger.info(f"📊 원본 토크나이저 정보:")
    logger.info(f"  - Class: {tokenizer.__class__.__name__}")
    logger.info(f"  - Vocab size: {len(tokenizer)}")
    logger.info(f"  - PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"  - EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"  - BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    logger.info(f"  - UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    logger.info(f"  - Padding side: {tokenizer.padding_side}")
    
    # 기본 토크나이저 테스트
    logger.info("🧪 STEP 2: 기본 토크나이저 테스트")
    test_text = "Hello world"
    test_tokens = tokenizer.encode(test_text)
    test_decoded = tokenizer.decode(test_tokens)
    logger.info(f"  Test encode/decode: '{test_text}' -> {test_tokens} -> '{test_decoded}'")
    
    # 문제 토큰들 확인
    logger.info("🚨 STEP 3: 문제 토큰들 확인")
    problem_words = ["setattr", "PrivateKey", "TestCase", "ForcedSuppressWarnings"]
    for word in problem_words:
        try:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            word_decoded = tokenizer.decode(word_tokens)
            logger.info(f"  '{word}' -> {word_tokens} -> '{word_decoded}'")
        except Exception as e:
            logger.error(f"  '{word}' -> ERROR: {e}")
    
    # 최소한의 토크나이저 설정만 적용
    logger.info("⚙️ STEP 4: 최소한의 토크나이저 설정")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"  PAD token set to EOS: {tokenizer.eos_token}")
    
    # padding_side는 기본값 유지 (right)
    logger.info(f"  Padding side: {tokenizer.padding_side} (기본값 유지)")
    
    # 모델 로드 - 양자화 없이
    logger.info("🤖 STEP 5: 모델 로드 (양자화 없음)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,  # bfloat16 대신 float16 사용
        device_map=DEVICE,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    logger.info("✅ Model loaded successfully")
    
    # 모델-토크나이저 호환성 확인
    logger.info("🔍 STEP 6: 모델-토크나이저 호환성 확인")
    model_embed_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"  Model embedding size: {model_embed_size}")
    logger.info(f"  Tokenizer vocab size: {tokenizer_vocab_size}")
    
    if model_embed_size != tokenizer_vocab_size:
        logger.error(f"❌ 크기 불일치! 모델: {model_embed_size}, 토크나이저: {tokenizer_vocab_size}")
        logger.info("🔧 토큰 임베딩 크기 조정 시도...")
        model.resize_token_embeddings(len(tokenizer))
        logger.info("✅ 토큰 임베딩 크기 조정 완료")
    else:
        logger.info("✅ 모델과 토크나이저 크기 일치")
    
    logger.info(f"  Model dtype: {model.dtype}")
    logger.info(f"  Model device: {next(model.parameters()).device}")
    
    return model, tokenizer

def test_simple_generation(model, tokenizer, prompt, test_name):
    """간단한 생성 테스트"""
    logger.info(f"\n=== {test_name} ===")
    logger.info(f"Input prompt: '{prompt}'")
    
    # 토크나이저 테스트
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    logger.info(f"Input tokens: {inputs['input_ids'][0].tolist()}")
    
    # Bad words 필터 (under-trained tokens 차단)
    bad_words = ["setattr", "ForcedSuppressWarnings", "RI", "kommsetattr", "despre", "empire", "FLICT", "PrivateKey", "TestCase"]
    bad_words_ids = []
    for word in bad_words:
        try:
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) > 0:
                bad_words_ids.append(word_ids)
        except:
            continue
    
    # 생성 파라미터
    generation_kwargs = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    
    if bad_words_ids:
        generation_kwargs["bad_words_ids"] = bad_words_ids
        logger.info(f"Bad words filter applied: {len(bad_words_ids)} words")
    
    logger.info(f"Generation parameters: {generation_kwargs}")
    
    # 생성
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    # 결과 분석
    input_length = inputs['input_ids'].shape[1]
    output_only_tokens = outputs[:, input_length:]
    generated_text = tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()
    
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Generated tokens: {output_only_tokens.shape}")
    logger.info(f"Generated token IDs: {output_only_tokens[0].tolist()}")
    logger.info(f"Generated text: '{generated_text}'")
    
    # 개별 토큰 분석
    logger.info("Individual token analysis:")
    for i, token_id in enumerate(output_only_tokens[0].tolist()):
        try:
            token_text = tokenizer.decode([token_id])
            logger.info(f"  Token {i}: ID={token_id}, Text='{token_text}'")
        except Exception as e:
            logger.error(f"  Token {i}: ID={token_id}, Decode error: {e}")
    
    return generated_text

def interactive_chat(model, tokenizer):
    """대화형 채팅 함수"""
    print("\n" + "="*60)
    print("🤖 OLMo 진단 모드 - 단계별 테스트")
    print("💡 질문을 입력하세요 (종료: 'quit', 'exit', 'q')")
    print("🔧 각 단계별로 다른 설정을 테스트합니다")
    print("="*60)
    
    test_configs = [
        {
            "name": "1️⃣ 최소 설정 (Greedy)",
            "params": {
                "max_new_tokens": 20,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        },
        {
            "name": "2️⃣ 기본 샘플링",
            "params": {
                "max_new_tokens": 30,
                "do_sample": True,
                "temperature": 1.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        },
        {
            "name": "3️⃣ 안전한 샘플링",
            "params": {
                "max_new_tokens": 50,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        }
    ]
    
    current_config = 0
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input(f"\n👤 사용자 [{test_configs[current_config]['name']}]: ").strip()
            
            # 종료 조건
            if user_input.lower() in ['quit', 'exit', 'q', '종료']:
                print("👋 대화를 종료합니다!")
                break
            
            # 설정 변경
            if user_input.lower() in ['next', 'n', '다음']:
                current_config = (current_config + 1) % len(test_configs)
                print(f"🔄 설정 변경: {test_configs[current_config]['name']}")
                continue
            
            if user_input.lower() in ['prev', 'p', '이전']:
                current_config = (current_config - 1) % len(test_configs)
                print(f"🔄 설정 변경: {test_configs[current_config]['name']}")
                continue
            
            if not user_input:
                print("❓ 질문을 입력하세요 (next/prev로 설정 변경)")
                continue
            
            print(f"\n🧪 테스트 중: {test_configs[current_config]['name']}")
            
            # 토크나이저 처리
            inputs = tokenizer(user_input, return_tensors="pt").to(DEVICE)
            print(f"📥 입력 토큰 수: {inputs['input_ids'].shape[1]}")
            print(f"📥 입력 토큰 IDs: {inputs['input_ids'][0].tolist()}")
            
            # 현재 설정으로 생성
            generation_kwargs = test_configs[current_config]["params"].copy()
            print(f"⚙️ 생성 설정: {generation_kwargs}")
            
            # 생성
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)
            
            # 결과 추출
            input_length = inputs['input_ids'].shape[1]
            output_only_tokens = outputs[:, input_length:]
            
            print(f"📤 출력 토큰 수: {output_only_tokens.shape[1]}")
            print(f"📤 출력 토큰 IDs: {output_only_tokens[0].tolist()}")
            
            # 각 토큰을 개별적으로 디코딩해서 확인
            print("🔍 개별 토큰 분석:")
            for j, token_id in enumerate(output_only_tokens[0].tolist()):
                try:
                    token_text = tokenizer.decode([token_id])
                    print(f"   Token {j}: ID={token_id} → '{token_text}'")
                except Exception as e:
                    print(f"   Token {j}: ID={token_id} → 디코딩 오류: {e}")
            
            # 전체 텍스트 디코딩
            generated_text = tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()
            print(f"\n🤖 OLMo 전체 답변: '{generated_text}'")
            
            # 다음 설정으로 자동 변경
            if len(generated_text) > 100 or "setattr" in generated_text.lower():
                print("⚠️ 문제가 있는 출력 감지됨")
            else:
                print("✅ 정상적인 출력으로 보임")
        
        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C로 대화를 종료합니다!")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    """메인 함수"""
    try:
        # 모델 로드
        model, tokenizer = load_olmo_model()
        
        # 대화형 채팅 시작
        interactive_chat(model, tokenizer)
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ 치명적 오류: {e}")

if __name__ == "__main__":
    main()
