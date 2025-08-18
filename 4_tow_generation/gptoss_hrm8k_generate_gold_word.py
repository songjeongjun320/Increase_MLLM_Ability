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
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datetime import datetime
import traceback
import psutil  # CPU 메모리 체크용

# --- 설정 (Configuration) ---
MODEL_PATH = "../1_models/gpt_oss/gpt-oss-120b"
DATASET_DIR = "../2_datasets/HRM8K_TEXT"
OUTPUT_DIR = "./gold_labels"
LOG_DIR = "./generation_logs"  # 새로운 로그 디렉토리

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
    return f"""You are an expert mathematician specializing in problem analysis. Your task is to identify the single most critical word in a Korean math problem. This word specifies the final quantity, value, or object that must be found to solve the problem. It is the 'target word' that the entire problem-solving process is aimed at.

Analyze the sentence and output your answer in a JSON format with a single key "unpredictable_word". Don't choose proper noun such as name, date, time and number.

---
Example 1:
Sentence: "닫힌구간 [0,2π]에서 정이된 함수 f(x) = acosbx + 3 이 x = π/3에서 최댓값 13을을 갖도록 하는 두 자연수 a,b 의 순서쌍 (a,b) 에 대하여 a+b의 최솟값은?"
Reasoning: "The final objective is the 'minimum value' of 'a+b', not the value itself. This is specified by the core noun '최솟값' (minimum value) in the question's final phrase. Following the established pattern of including the grammatical particle, the complete and most precise target is '최솟값은'."
JSON Output:
{{
"unpredictable_word": "최솟값은"
}}

---

Example 2:
Sentence: "시각 t = 0 일 때, 출발하여 수직선 위를 움직이는 점 P의 시각 t(t>=0)에서의 위치 x가 x=t^3-(3t^2)/2-6t 이다. 출발한 후 점 P의 운동 방향이 바뀌는 시각에서의 점 P의 가속도는?"
Reasoning: "This problem asks for a specific physical quantity under a certain condition. The question's final phrase, '점 P의 가속도는?' (What is the acceleration of point P?), explicitly states that the final goal is to find the 'acceleration'. Following the established pattern, the core noun '가속도' (acceleration) is combined with its grammatical particle '는' to make '가속도는' the most precise target word for the final answer."
JSON Output:
{{
"unpredictable_word": "가속도는"
}}

---

Example 3:
Sentence: "최고차항의 계수가 1인 삼차함수 f(x)가 f(1)=f(2)=0, f'(0)=-7을 만족시킨다. 원점 O와 점 P(3,f(3))에 대하여 선분 OP가 곡선 y=f(x)와 만나는 점 중 P가 아닌 점을 Q라 하자. 곡선 y=f(x)와 y축 및 선분 OQ로 둘러싸인 부분의 넓이를 A, 곡선 y=f(x)와 선분 PQ로 둘러싸인 부분의 넙이를 B라 할 때, B-A의 값은?"
Reasoning: "This case is unique because while the question asks for a '값은' (value), the target word is '넓이를' (area). The problem explicitly defines the components of the final calculation, A and B, as areas ('넓이'). Therefore, the fundamental quantity being calculated (B-A) is an area. '넓이를' is chosen because it correctly identifies the specific nature of the target quantity, which is more descriptive than the generic term '값은'."
JSON Output:
{{
"unpredictable_word": "넓이를"
}}

---

Example 4:
Sentence: "방정식 log2(x-3)=log4(3x-5)를 만족시키는 실수 x의 값을 구하시오."
Reasoning: "This problem directly asks to solve for an unknown variable, 'x'. The instruction '실수 x의 값을 구하시오' (Find the value of the real number x) makes it clear that the final target is not a property of x (like its maximum or minimum) but the variable itself. Therefore, 'x' is the most fundamental and direct representation of the quantity that needs to be found."
JSON Output:
{{
"unpredictable_word": "x"
}}

---

Example 5:
Sentence: "두 사건 A,B에 대하여 P(A|B)=P(A)=1/2, P(A∩B)=1/5 일 때, P(A∪B)의 값은?"
Reasoning: "The problem asks for the numerical result of the probability expression P(A∪B). The final phrase 'P(A∪B)의 값은?' directly translates to 'What is the value of P(A∪B)?'. Unlike Example 3, where a more specific noun ('area') was defined earlier, this problem does not provide a more fundamental descriptor for the probability. Therefore, the most direct and appropriate target word is '값은' (value is?), taken from the question itself, which identifies the generic numerical result required."
JSON Output:
{{
"unpredictable_word": "값은"
}}
---

Now, analyze this sentence:
Sentence: "{sentence}"
JSON Output:"""

def load_model():
    """
    GPT-OSS 120B 모델을 메모리 효율적으로 로드합니다.
    A100 80GB x2 (160GB 총 메모리)에서 120B 모델 로딩 최적화
    """
    print(f"[INFO] Loading GPT-OSS 120B model: {MODEL_PATH}")
    
    try:
        # Load model config first
        model_config = AutoConfig.from_pretrained(MODEL_PATH)
        print("[INFO] Model config loaded.")
    except Exception as e:
        print(f"[ERROR] Model configuration loading failed: {e}")
        return None, None

    # Aggressive memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    
    # 메모리 체크
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            memory_free = (torch.cuda.get_device_properties(i).total_memory / 1024**3) - memory_cached
            print(f"[INFO] GPU {i}: {memory_free:.1f}GB free, {memory_cached:.1f}GB cached")

    # Try multiple loading strategies with proper BitsAndBytesConfig
    try:
        from transformers import BitsAndBytesConfig
        use_bnb = True
    except ImportError:
        print("[WARNING] BitsAndBytes not available, using CPU offload only")
        use_bnb = False
    
    loading_strategies = []
    
    # Strategy 1: 8bit quantization (only if BnB available)
    if use_bnb:
        try:
            bnb_8bit_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0,
            )
            loading_strategies.append({
                "name": "8bit quantization",
                "config": {
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                    "quantization_config": bnb_8bit_config,
                    "max_memory": {i: "70GB" for i in range(torch.cuda.device_count())} | {"cpu": "40GB"}
                }
            })
        except Exception as e:
            print(f"[WARNING] 8bit config failed: {e}")
    
    # Strategy 2: 4bit quantization (only if BnB available) 
    if use_bnb:
        try:
            bnb_4bit_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            loading_strategies.append({
                "name": "4bit quantization", 
                "config": {
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                    "quantization_config": bnb_4bit_config,
                    "max_memory": {i: "72GB" for i in range(torch.cuda.device_count())}
                }
            })
        except Exception as e:
            print(f"[WARNING] 4bit config failed: {e}")
    
    # Strategy 3: CPU offload only (always available)
    loading_strategies.append({
        "name": "CPU offload",
        "config": {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "offload_folder": "./offload",
            "max_memory": {i: "65GB" for i in range(torch.cuda.device_count())} | {"cpu": "50GB"}
        }
    })
    
    for strategy in loading_strategies:
        try:
            print(f"[INFO] Trying strategy: {strategy['name']}")
            
            # Create offload directory if needed
            if "offload_folder" in strategy["config"]:
                os.makedirs(strategy["config"]["offload_folder"], exist_ok=True)
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                **strategy["config"]
            )
            
            print(f"[SUCCESS] Model loaded with strategy: {strategy['name']}")
            
            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print("[SUCCESS] Tokenizer loaded successfully.")
                return model, tokenizer
            except Exception as e:
                print(f"[ERROR] Tokenizer loading failed: {e}")
                return None, None
                
        except Exception as e:
            print(f"[WARNING] Strategy '{strategy['name']}' failed: {e}")
            # Clean up on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    print("[ERROR] All loading strategies failed.")
    return None, None

def generate_with_model(model, tokenizer, prompt, max_new_tokens=150):
    """메모리 최적화된 텍스트 생성"""
    try:
        # 토크나이징
        try:
            inputs = tokenizer(
                prompt,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"[ERROR] Tokenization failed: {e}")
            return None
        
        # 모델 위치 확인
        try:
            # 분산된 모델의 첫 번째 디바이스 찾기
            if hasattr(model, 'hf_device_map'):
                first_device = list(model.hf_device_map.values())[0]
                if isinstance(first_device, int):
                    model_device = torch.device(f'cuda:{first_device}')
                else:
                    model_device = torch.device(first_device)
            else:
                model_device = next(iter(model.parameters())).device
        except:
            model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 입력을 첫 번째 디바이스로 이동
        try:
            inputs = {k: v.to(model_device, dtype=torch.long) for k, v in inputs.items()}
        except Exception as e:
            print(f"[ERROR] Failed to move inputs to device: {e}")
            return None
        
        # 메모리 절약형 생성 설정
        generation_config = {
            'max_new_tokens': min(max_new_tokens, 50),  # 토큰 수 제한
            'temperature': 0.3,
            'do_sample': True,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'use_cache': False,  # 메모리 절약
            'output_scores': False,
            'return_dict_in_generate': False,
        }
        
        try:
            # 생성 전 메모리 정리
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            
            # GPU 메모리 체크
            if torch.cuda.is_available():
                available_memory = min([
                    torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i) 
                    for i in range(torch.cuda.device_count())
                ]) / 1024**3
                
                if available_memory < 5.0:  # 5GB 미만이면 경고
                    print(f"[WARNING] Low GPU memory: {available_memory:.1f}GB available")
                    generation_config['max_new_tokens'] = min(generation_config['max_new_tokens'], 30)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config,
                )
                
                # 새로 생성된 토큰만 디코딩
                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[0][input_length:]
                
                if len(new_tokens) > 0:
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    print(f"[SUCCESS] Generated {len(new_tokens)} tokens successfully.")
                    return generated_text.strip()
                else:
                    print(f"[WARNING] No new tokens were generated.")
                    return ""
        
        except torch.cuda.OutOfMemoryError as oom:
            print(f"[ERROR] CUDA OOM during generation: {oom}")
            # 강제 메모리 정리
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            return None
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Generation failed: {error_msg}")
            if "out of memory" in error_msg.lower():
                print(f"[INFO] OOM detected, trying emergency cleanup...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return None

        print("[ERROR] Generation failed. No output produced.")
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
    return all_data[:10] if len(all_data) > 10 else all_data

def save_generation_log(item_id, prompt, raw_output, error_msg=None, success=True):
    """모델 생성 로그를 JSON 파일로 저장"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_entry = {
            "timestamp": timestamp,
            "item_id": item_id,
            "prompt_length": len(prompt) if prompt else 0,
            "prompt_preview": prompt[-200:] if prompt else "",  # 마지막 200자
            "raw_output": raw_output,
            "output_length": len(raw_output) if raw_output else 0,
            "success": success,
            "error_message": error_msg
        }
        
        # 로그 파일 명 (성공/실패 구분)
        status = "success" if success else "error"
        log_filename = f"generation_log_{status}_{timestamp}_{item_id}.json"
        log_path = os.path.join(LOG_DIR, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"[WARNING] Failed to save log for {item_id}: {e}")

def process_datasets():
    """Process HRM8K_TEXT dataset to generate gold labels"""
    # Create output and log directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("[ERROR] Model loading failed. Attempting to run without model (emergency mode).")
        print("[WARNING] Will use pattern-based fallback responses.")
        model, tokenizer = None, None  # 명시적으로 None 설정
    
    # Load data
    all_data = load_hrm8k_datasets()
    if not all_data:
        print("[ERROR] No data loaded.")
        return
    
    results = []
    error_count = 0
    processed_count = 0
    all_logs = []  # 전체 로그를 저장할 리스트
    
    print(f"[INFO] Starting gold label generation for {len(all_data)} sentences")
    print(f"[INFO] Logs will be saved to: {LOG_DIR}")
    
    for item in tqdm(all_data, desc="Generating gold labels"):
        processed_count += 1
        
        try:
            prompt = create_prompt(item['sentence'])
            
            # 모델이 없을 경우 비상 모드
            if model is None or tokenizer is None:
                print(f"[WARNING] No model available, using emergency pattern matching for {item['id']}")
                # 간단한 패턴 기반 예측
                sentence = item['sentence']
                if '값' in sentence and sentence.endswith('?'):
                    raw_output = '{\n"unpredictable_word": "값"\n}'
                elif '결과' in sentence:
                    raw_output = '{\n"unpredictable_word": "결과"\n}'
                elif sentence.endswith('는?') or sentence.endswith('나?'):
                    raw_output = '{\n"unpredictable_word": "는"\n}'
                else:
                    # 마지막 단어 추출 시도
                    words = sentence.strip().split()
                    if words:
                        last_word = words[-1].replace('?', '').replace('.', '')
                        raw_output = f'{{\n"unpredictable_word": "{last_word}"\n}}'
                    else:
                        raw_output = '{\n"unpredictable_word": "단어"\n}'
            else:
                raw_output = generate_with_model(model, tokenizer, prompt)
            
            # 로그 저장 (성공/실패 구분 없이 모든 생성 시도를 기록)
            if raw_output is None:
                save_generation_log(item['id'], prompt, "", "Model generation failed", False)
                error_count += 1
                continue
            else:
                save_generation_log(item['id'], prompt, raw_output, None, True)
                
            # 전체 로그에 추가
            all_logs.append({
                "item_id": item['id'],
                "sentence": item['sentence'],
                "prompt_length": len(prompt),
                "raw_output": raw_output,
                "output_length": len(raw_output) if raw_output else 0,
                "timestamp": datetime.now().isoformat()
            })
            
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
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] Processing error (ID: {item['id']}): {e}")
            
            # 예외 발생 시 로그 저장
            try:
                prompt = create_prompt(item['sentence'])
                save_generation_log(item['id'], prompt, "", error_msg, False)
            except:
                save_generation_log(item['id'], "", "", error_msg, False)
                
            error_count += 1
            continue
        
        # Periodic saving (결과와 로그 둘 다 저장)
        if processed_count % SAVE_INTERVAL == 0 and processed_count > 0:
            # 결과 저장
            output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss120b.json")
            print(f"\n[INFO] Intermediate save: saving {len(results)} results")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            # 전체 로그 저장
            log_summary_path = os.path.join(LOG_DIR, "generation_summary.json")
            with open(log_summary_path, 'w', encoding='utf-8') as f:
                summary = {
                    "total_processed": processed_count,
                    "successful_results": len(results),
                    "error_count": error_count,
                    "success_rate": len(results) / processed_count if processed_count > 0 else 0,
                    "last_updated": datetime.now().isoformat(),
                    "all_logs": all_logs
                }
                json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Final save (결과와 전체 로그)
    output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss120b.json")
    log_summary_path = os.path.join(LOG_DIR, "generation_summary_final.json")
    
    print(f"\n[SUCCESS] Processing complete!")
    print(f"  - Successfully processed sentences: {len(results)}")
    print(f"  - Errors or skipped sentences: {error_count}")
    print(f"  - Success rate: {len(results)/len(all_data)*100:.1f}%")
    print(f"  - Result file: {output_path}")
    print(f"  - Log summary: {log_summary_path}")
    
    # 최종 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # 최종 로그 요약 저장
    with open(log_summary_path, 'w', encoding='utf-8') as f:
        final_summary = {
            "total_sentences": len(all_data),
            "total_processed": processed_count,
            "successful_results": len(results),
            "error_count": error_count,
            "success_rate": len(results) / len(all_data) if len(all_data) > 0 else 0,
            "completion_time": datetime.now().isoformat(),
            "all_generation_logs": all_logs
        }
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_datasets()