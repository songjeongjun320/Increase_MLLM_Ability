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
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, padding_side='left')
    
    # OLMo 토크나이저 설정
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"PAD token set to UNK: {tokenizer.unk_token}")
        else:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"PAD token set to EOS: {tokenizer.eos_token}")
    
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        logger.info(f"BOS token set to EOS: {tokenizer.eos_token}")
    
    # 토크나이저 정보 출력
    logger.info(f"Tokenizer class: {tokenizer.__class__.__name__}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(f"PAD: {tokenizer.pad_token_id}, EOS: {tokenizer.eos_token_id}, BOS: {tokenizer.bos_token_id}")
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    
    model.eval()
    logger.info("Model loaded successfully")
    
    # 모델 정보 출력
    model_embed_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model embedding size: {model_embed_size}")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
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

def main():
    """메인 테스트 함수"""
    try:
        # 모델 로드
        model, tokenizer = load_olmo_model()
        
        # 간단한 테스트들
        test_cases = [
            ("Hello", "Simple greeting"),
            ("What is 2+2?", "Simple math question"),
            ("The capital of France is", "Simple completion"),
            ("Once upon a time", "Story beginning"),
            ("Question: What color is the sky?\nAnswer:", "Simple Q&A format")
        ]
        
        results = {}
        for prompt, test_name in test_cases:
            try:
                result = test_simple_generation(model, tokenizer, prompt, test_name)
                results[test_name] = result
            except Exception as e:
                logger.error(f"Error in {test_name}: {e}")
                results[test_name] = f"ERROR: {str(e)}"
        
        # 결과 요약
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)
        for test_name, result in results.items():
            logger.info(f"{test_name}: {result}")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()
