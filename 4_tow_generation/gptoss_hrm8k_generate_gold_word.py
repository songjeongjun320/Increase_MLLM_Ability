#!/usr/bin/env python3
"""
gptoss_generate_gold_word.py

GPT-OSS 120B 모델을 사용하여
문장에서 가장 예측하기 어려운 단어를 JSON 형식으로 생성하고,
그 결과를 파싱하여 최종 데이터셋을 구축합니다.

HRM8K_TEXT 데이터셋의 모든 JSON 파일을 처리합니다.
"""
import json
import os
import re
import glob
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 설정 (Configuration) ---
MODEL_PATH = "../1_models/gpt_oss/gpt-oss-120b"
DATASET_DIR = "../2_datasets/HRM8K_TEXT"
OUTPUT_DIR = "./gold_labels"

# Multi-GPU 설정
NUM_GPUS = torch.cuda.device_count()
DEVICES = [f"cuda:{i}" for i in range(NUM_GPUS)] if NUM_GPUS > 0 else ["cpu"]

# 배치 처리 설정
BATCH_SIZE = 1  # GPT-OSS 120B는 큰 모델이므로 배치 크기를 작게 설정
SAVE_INTERVAL = 50  # 50개 처리할 때마다 저장

# =================================================================
# 수정 1: 모델이 JSON 형식으로 결과를 출력하도록 프롬프트를 변경합니다.
# =================================================================
def create_prompt(sentence: str) -> str:
    """
    모델이 예측하기 가장 어려운 단어를 JSON 형식으로 출력하도록 유도하는
    상세한 Few-shot 프롬프트를 생성합니다.
    """
    return f"""You are a language prediction expert. Your task is to find the single most unpredictable or surprising word in a given Korean sentence. This word is often a proper noun, a specific number, or a key piece of information that cannot be easily guessed.

Analyze the sentence and output your answer in a JSON format with a single key "unpredictable_word". Don't choose proper noun such as name, date, time and number.

---
Example 1:
Sentence: "닫힌구간 [0,2π]에서 정이된 함수 f(x) = acosbx + 3 이 x = π/3에서 최댓값 13을을 갖도록 하는 두 자연수 a,b 의 순서쌍 (a,b) 에 대하여 a+b의 최솟값은?"
JSON Output:
{{
"unpredictable_word": "최솟값은"
}}

---

Example 2:
Sentence: "시각 t = 0 일 때, 출발하여 수직선 위를 움직이는 점 P의 시각 t(t>=0)에서의 위치 x가 x=t^3-(3t^2)/2-6t 이다. 출발한 후 점 P의 운동 방향이 바뀌는 시각에서의 점 P의 가속도는?"
JSON Output:
{{
"unpredictable_word": "가속도는"
}}

---

Example 3:
Sentence: "최고차항의 계수가 1인 삼차함수 f(x)가 f(1)=f(2)=0, f'(0)=-7을 만족시킨다. 원점 O와 점 P(3,f(3))에 대하여 선분 OP가 곡선 y=f(x)와 만나는 점 중 P가 아닌 점을 Q라 하자. 곡선 y=f(x)와 y축 및 선분 OQ로 둘러싸인 부분의 넓이를 A, 곡선 y=f(x)와 선분 PQ로 둘러싸인 부분의 넙이를 B라 할 때, B-A의 값은?"
JSON Output:
{{
"unpredictable_word": "넓이를"
}}

---

Example 4:
Sentence: "방정식 log2(x-3)=log4(3x-5)를 만족시키는 실수 x의 값을 구하시오."
JSON Output:
{{
"unpredictable_word": "x"
}}

---

Example 5:
Sentence: "두 사건 A,B에 대하여 P(A|B)=P(A)=1/2, P(A∩B)=1/5 일 때, P(A∪B)의 값은?"
JSON Output:
{{
"unpredictable_word": "값은"
}}
---

Now, analyze this sentence:
Sentence: "{sentence}"
JSON Output:"""

def load_model():
    """GPT-OSS 120B 모델과 토크나이저를 로드합니다 (강화된 안전 로딩)."""
    print(f"[INFO] Loading GPT-OSS 120B model: {MODEL_PATH}")
    print(f"[INFO] Available devices: {DEVICES}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[SUCCESS] Tokenizer loaded successfully")
    except Exception as tokenizer_error:
        print(f"[ERROR] Tokenizer loading failed: {tokenizer_error}")
        return None, None
    
    # 여러 단계의 fallback 전략
    model_loading_strategies = [
        {
            "name": "Float16 without quantization",
            "config": {
                "device_map": "auto" if NUM_GPUS > 1 else DEVICES[0],
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
    ]
    
    for strategy in model_loading_strategies:
        try:
            print(f"[INFO] Trying: {strategy['name']}")
            
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **strategy['config'])
            
            print(f"[SUCCESS] Model loaded with strategy: {strategy['name']}")
            
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            
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

def generate_with_model(model, tokenizer, prompt, max_new_tokens=50):
    """극도로 안전한 텍스트 생성"""
    try:
        # 다양한 토크나이징 방법
        input_methods = [
            {"truncation": True, "max_length": 1024, "return_tensors": "pt"},
            {"truncation": True, "max_length": 512, "return_tensors": "pt"},
            {"return_tensors": "pt"},
        ]
        
        inputs = None
        for method in input_methods:
            try:
                inputs = tokenizer(prompt, **method)
                break
            except Exception:
                continue
        
        if inputs is None:
            print("[ERROR] All tokenization methods failed")
            return None
        
        # 모델 디바이스 확인
        try:
            model_device = next(iter(model.parameters())).device
        except:
            model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 입력을 모델 디바이스로 이동
        try:
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception as e:
            print(f"[ERROR] Failed to move inputs to device: {e}")
            return None
        
        # 생성 전략들 (안전성 순서)
        generation_strategies = [
            {
                "name": "Minimal safe",
                "config": {
                    'max_new_tokens': min(max_new_tokens, 5),
                    'do_sample': False,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }
            },
            {
                "name": "Conservative",
                "config": {
                    'max_new_tokens': min(max_new_tokens, 15),
                    'do_sample': False,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': False,
                }
            },
            {
                "name": "Standard with sampling",
                "config": {
                    'max_new_tokens': min(max_new_tokens, 30),
                    'do_sample': True,
                    'temperature': 0.1,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': False,
                }
            },
            {
                "name": "Full generation",
                "config": {
                    'max_new_tokens': max_new_tokens,
                    'temperature': 0.1,
                    'do_sample': True,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': False,
                    'return_dict_in_generate': True,
                }
            }
        ]
        
        for strategy in generation_strategies:
            try:
                print(f"[INFO] Trying: {strategy['name']}")
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    try:
                        outputs = model.generate(**inputs, **strategy['config'])
                        sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
                        
                        # 새로 생성된 토큰 디코딩
                        if len(sequences.shape) > 1 and sequences.shape[0] > 0:
                            input_length = inputs['input_ids'].shape[1]
                            new_tokens = sequences[0][input_length:]
                            
                            if len(new_tokens) > 0:
                                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                                print(f"[SUCCESS] Generated with: {strategy['name']}")
                                return generated_text.strip()
                            else:
                                print(f"[WARNING] No new tokens with: {strategy['name']}")
                                continue
                        else:
                            print(f"[WARNING] Invalid output shape with: {strategy['name']}")
                            continue
                            
                    except torch.cuda.OutOfMemoryError as oom:
                        print(f"[ERROR] CUDA OOM with {strategy['name']}: {oom}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                        
                    except Exception as gen_e:
                        print(f"[ERROR] Generation failed with {strategy['name']}: {gen_e}")
                        continue
                
            except Exception as strategy_e:
                print(f"[ERROR] Strategy {strategy['name']} failed: {strategy_e}")
                continue
        
        print("[ERROR] All generation strategies failed")
        return None
        
    except Exception as e:
        print(f"[ERROR] Text generation completely failed: {e}")
        return None

def load_hrm8k_datasets():
    """Load all JSON files from HRM8K_TEXT directory"""
    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    all_data = []
    
    for json_file in json_files:
        print(f"[INFO] Loading file: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # app100개 사용하도록 수정 (전체 데이터 사용)
            # data = data[:10]  # 테스트용 제한 제거
            
            # Standardize data format (convert question field to sentence)
            dataset_name = os.path.basename(json_file).replace('.json', '')
            for i, item in enumerate(data):
                if 'question' in item:
                    new_item = {
                        'id': f"{dataset_name}_{i}",
                        'sentence': item['question'],
                        'original_data': item
                    }
                    all_data.append(new_item)
                    
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file}: {e}")
            continue
    
    print(f"[INFO] Total {len(all_data)} sentences loaded")
    # app100개 처리를 위해 제한을 최대 100개로 설정 (전체 데이터가 100개 미만이면 모두 사용)
    return all_data[:100] if len(all_data) > 100 else all_data

def process_datasets():
    """Process HRM8K_TEXT dataset to generate gold labels"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("[ERROR] Model loading failed. Exiting program.")
        return
    
    # Load data
    all_data = load_hrm8k_datasets()
    if not all_data:
        print("[ERROR] No data loaded.")
        return
    
    results = []
    error_count = 0
    processed_count = 0
    
    print(f"[INFO] Starting gold label generation for {len(all_data)} sentences")
    
    for item in tqdm(all_data, desc="Generating gold labels"):
        processed_count += 1
        
        try:
            prompt = create_prompt(item['sentence'])
            raw_output = generate_with_model(model, tokenizer, prompt)
            
            if raw_output is None:
                error_count += 1
                continue
            
            # JSON parsing
            predicted_word = None
            print(f"[DEBUG] Raw output for {item['id']}: {raw_output[:200]}...")
            
            json_match = re.search(r'{\s*"unpredictable_word":\s*".*?"\s*}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"[DEBUG] Found JSON: {json_str}")
                try:
                    parsed_json = json.loads(json_str)
                    predicted_word = parsed_json.get("unpredictable_word")
                    print(f"[DEBUG] Parsed word: {predicted_word}")
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] JSON decode error: {e}")
            else:
                print(f"[DEBUG] No JSON match found in output")
            
            if not predicted_word:
                print(f"[DEBUG] No predicted word found, skipping item {item['id']}")
                error_count += 1
                continue
            
            # Find predicted word in original sentence
            original_sentence = item['sentence']
            if predicted_word in original_sentence:
                index = original_sentence.find(predicted_word)
                context = original_sentence[:index].strip()
                gold_label = predicted_word
                
                if not context:
                    error_count += 1
                    continue
                
                new_item = {
                    'id': item['id'],
                    'original_sentence': original_sentence,
                    'context': context,
                    'gold_label': gold_label,
                    'raw_output': raw_output,
                    'original_data': item['original_data']
                }
                results.append(new_item)
            else:
                error_count += 1
                continue
                
        except Exception as e:
            print(f"[ERROR] Processing error (ID: {item['id']}): {e}")
            error_count += 1
            continue
        
        # Periodic saving
        if len(results) % SAVE_INTERVAL == 0 and len(results) > 0:
            output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss120b.json")
            print(f"\n[INFO] Intermediate save: saving {len(results)} results")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Final save
    output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss120b.json")
    print(f"\n[SUCCESS] Processing complete!")
    print(f"  - Successfully processed sentences: {len(results)}")
    print(f"  - Errors or skipped sentences: {error_count}")
    print(f"  - Result file: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_datasets()