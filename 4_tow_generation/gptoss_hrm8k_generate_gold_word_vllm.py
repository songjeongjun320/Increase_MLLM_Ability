#!/usr/bin/env python3
"""
gptoss_generate_gold_word_vllm.py

vLLM을 사용하여 GPT-OSS 120B 모델로 
문장에서 가장 예측하기 어려운 단어를 생성합니다.

vLLM은 대용량 모델을 효율적으로 로딩하고 추론할 수 있습니다.
"""
import json
import os
import re
import glob
from tqdm import tqdm
from datetime import datetime
import traceback

# vLLM 관련 import
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed import init_distributed_environment
    VLLM_AVAILABLE = True
except ImportError:
    print("[ERROR] vLLM not available. Install with: pip install vllm")
    VLLM_AVAILABLE = False

# --- 설정 (Configuration) ---
MODEL_PATH = "../1_models/gpt_oss/gpt-oss-120b"
DATASET_DIR = "../2_datasets/HRM8K_TEXT"
OUTPUT_DIR = "./gold_labels"
LOG_DIR = "./generation_logs"

# vLLM 설정
TENSOR_PARALLEL_SIZE = 2  # A100 80GB x2 사용
MAX_MODEL_LEN = 4096      # 최대 시퀀스 길이
BATCH_SIZE = 4            # vLLM은 배치 처리 최적화
SAVE_INTERVAL = 50

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

def load_model_vllm():
    """
    vLLM을 사용하여 GPT-OSS 120B 모델을 로드합니다.
    A100 80GB x2에서 tensor parallel로 효율적 로딩
    """
    if not VLLM_AVAILABLE:
        print("[ERROR] vLLM is not available. Please install it first.")
        return None
        
    print(f"[INFO] Loading GPT-OSS 120B model with vLLM: {MODEL_PATH}")
    print(f"[INFO] Using tensor_parallel_size: {TENSOR_PARALLEL_SIZE}")
    
    try:
        # vLLM 모델 로딩
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,  # 2개 GPU 병렬 사용
            max_model_len=MAX_MODEL_LEN,                # 최대 길이 제한
            trust_remote_code=True,                     # 커스텀 모델 코드 신뢰
            dtype="bfloat16",                          # bfloat16 사용
            gpu_memory_utilization=0.85,               # GPU 메모리 85% 사용
            swap_space=4,                              # 4GB swap space
            enforce_eager=True,                        # eager mode (더 안정적)
        )
        
        print("[SUCCESS] vLLM model loaded successfully.")
        return llm
        
    except Exception as e:
        print(f"[ERROR] vLLM model loading failed: {e}")
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return None

def generate_with_vllm(llm, prompts, max_tokens=50):
    """
    vLLM을 사용하여 배치로 텍스트 생성
    """
    if llm is None:
        return [None] * len(prompts)
    
    try:
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stop=None,
        )
        
        print(f"[INFO] Generating for {len(prompts)} prompts...")
        
        # 배치 생성
        outputs = llm.generate(prompts, sampling_params)
        
        # 결과 추출
        results = []
        for output in outputs:
            if len(output.outputs) > 0:
                generated_text = output.outputs[0].text.strip()
                results.append(generated_text)
            else:
                results.append(None)
        
        print(f"[SUCCESS] Generated {len(results)} outputs successfully.")
        return results
        
    except Exception as e:
        print(f"[ERROR] vLLM generation failed: {e}")
        return [None] * len(prompts)

def load_hrm8k_datasets():
    """Load all JSON files from HRM8K_TEXT directory"""
    json_files = glob.glob(os.path.join(DATASET_DIR, "*.json"))
    all_data = []
    
    for json_file in json_files:
        print(f"[INFO] Loading file: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Standardize data format
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
    return all_data[:100] if len(all_data) > 100 else all_data  # 100개 제한

def save_generation_log(item_id, prompt, raw_output, error_msg=None, success=True):
    """모델 생성 로그를 JSON 파일로 저장"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_entry = {
            "timestamp": timestamp,
            "item_id": item_id,
            "prompt_length": len(prompt) if prompt else 0,
            "prompt_preview": prompt[-200:] if prompt else "",
            "raw_output": raw_output,
            "output_length": len(raw_output) if raw_output else 0,
            "success": success,
            "error_message": error_msg
        }
        
        status = "success" if success else "error"
        log_filename = f"vllm_generation_log_{status}_{timestamp}_{item_id}.json"
        log_path = os.path.join(LOG_DIR, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"[WARNING] Failed to save log for {item_id}: {e}")

def process_datasets():
    """Process HRM8K_TEXT dataset to generate gold labels using vLLM"""
    # Create output and log directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load vLLM model
    llm = load_model_vllm()
    if llm is None:
        print("[ERROR] vLLM model loading failed. Exiting.")
        return
    
    # Load data
    all_data = load_hrm8k_datasets()
    if not all_data:
        print("[ERROR] No data loaded.")
        return
    
    results = []
    error_count = 0
    processed_count = 0
    all_logs = []
    
    print(f"[INFO] Starting gold label generation for {len(all_data)} sentences")
    print(f"[INFO] Using batch size: {BATCH_SIZE}")
    print(f"[INFO] Logs will be saved to: {LOG_DIR}")
    
    # 배치 처리
    for i in range(0, len(all_data), BATCH_SIZE):
        batch = all_data[i:i+BATCH_SIZE]
        prompts = [create_prompt(item['sentence']) for item in batch]
        
        print(f"[INFO] Processing batch {i//BATCH_SIZE + 1}/{(len(all_data)-1)//BATCH_SIZE + 1}")
        
        # vLLM으로 배치 생성
        raw_outputs = generate_with_vllm(llm, prompts)
        
        # 배치 결과 처리
        for j, (item, raw_output) in enumerate(zip(batch, raw_outputs)):
            processed_count += 1
            
            try:
                if raw_output is None:
                    save_generation_log(item['id'], prompts[j], "", "vLLM generation failed", False)
                    error_count += 1
                    continue
                
                save_generation_log(item['id'], prompts[j], raw_output, None, True)
                
                # 전체 로그에 추가
                all_logs.append({
                    "item_id": item['id'],
                    "sentence": item['sentence'],
                    "prompt_length": len(prompts[j]),
                    "raw_output": raw_output,
                    "output_length": len(raw_output),
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
                save_generation_log(item['id'], prompts[j], "", error_msg, False)
                error_count += 1
                continue
        
        # Periodic saving
        if processed_count % SAVE_INTERVAL == 0 and processed_count > 0:
            output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss120b_vllm.json")
            print(f"\n[INFO] Intermediate save: saving {len(results)} results")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            # 전체 로그 저장
            log_summary_path = os.path.join(LOG_DIR, "vllm_generation_summary.json")
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
    
    # Final save
    output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss120b_vllm.json")
    log_summary_path = os.path.join(LOG_DIR, "vllm_generation_summary_final.json")
    
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
    if not VLLM_AVAILABLE:
        print("[ERROR] vLLM not available. Install with:")
        print("pip install vllm")
        exit(1)
        
    process_datasets()