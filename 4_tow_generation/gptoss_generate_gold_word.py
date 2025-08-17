#!/usr/bin/env python3
"""
gptoss_generate_gold_word.py

GPT-OSS 20B 모델을 사용하여
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
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 설정 (Configuration) ---
MODEL_PATH = "/scratch/jsong132/Increase_MLLM_Ability/1_models/gpt_oss/gpt-oss-20b"
DATASET_DIR = "../2_datasets/HRM8K_TEXT"
OUTPUT_DIR = "./gold_labels"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 배치 처리 설정
BATCH_SIZE = 1  # GPT-OSS 20B는 큰 모델이므로 배치 크기를 작게 설정
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
Sentence: "심청효행대상은 가천문화재단 설립자인 이길여 가천길재단 회장이 지난 1999년에 고전소설 '심청전'의 배경인 인천광역시 옹진군 백령면에 심청동상을 제작, 기증한 것을 계기로 제정되었다."
JSON Output:
{{
"unpredictable_word": "배경인"
}}

---

Example 2:
Sentence: "숙소 예약한 시간까지 택시 타고 숙소에 도착하고 싶은데 삼청각에서 출발할 겁니다."
JSON Output:
{{
"unpredictable_word": "도착하고"
}}

---

Example 3:
Sentence: "추천해주시는 곳으로 가볼게요. 목요일에 갈건데 저희가 10명이거든요. 4일 예약이 가능할까요?"
JSON Output:
{{
"unpredictable_word": "가능할까요"
}}

---

Example 4:
Sentence: "C 여학교에서 교원 겸 기숙사 사감 노릇을 하는 B 여사라면 딱장대요 독신주의자요 찰진 야소군으로 유명하다.\\n사십에 가까운 노처녀인 그는 주근깨투성이 얼굴이 처녀다운 맛이란 약에 쓰려도 찾을 수 없을 뿐인가, 시들고 거칠고 마르고 누렇게 뜬 품이 곰팡 슬은 굴비를 생각나게 한다.\\n여러 겹주름이 잡힌 훨렁 벗겨진 이마라든지, 숱이 적어서 법대로 쪽지거나 틀어 올리지를 못하고 엉성하게 그냥 빗어넘긴 머리꼬리가 뒤통수에 염소 똥만 하게 붙은 것이라든지, 벌써 늙어가는 자취를 감출 길이 없었다.\\n뾰족한 입을 앙다물고 돋보기 너머로 쌀쌀한 눈이 노릴 때엔 기숙생들이 오싹하고 몸서리를 치리만큼 그는 엄격하고 매서웠다.\\n이 B 여사가 질겁을 하다시피 싫어하고 미워하는 것은 소위 '러브레터'였다.\\n여학교 기숙사라면 으레 그런 편지가 많이 오는 것이지만 학교로도 유명하고 또 아름다운 여학생이 많은 탓인지 모르되 하루에도 몇 장씩 죽느니 사느니 하는 사랑 타령이 날아들어 왔었다.\\n기숙생에게 오는 사신을 일일이 검토하는 터이니까 그따위 편지도 물론 B 여사의 손에 떨어진다.\\n달짝지근한 사연을 보는 족족 그는 더할 수 없이 흥분되어서 얼굴이 붉으락푸르락, 편지 든 손이 발발 떨리도록 성을 낸다.\\n아무 까닭 없이 그런 편지를 받은 학생이야말로 큰 재변이었다.\\n하학하기가 무섭게 그 학생은 사감실로 불리어 간다.\\n분해서 못 견디겠다는 사람 모양으로 쌔근쌔근하며 방안을 왔다 갔다 하던 그는, 들어오는 학생을 잡아먹을 듯이 노리면서 한 걸음 두 걸음 코가 맞닿을 만큼 바싹 다가들어서 딱 마주 선다."
JSON Output:
{{
"unpredictable_word": "얼굴이"
}}

---

Example 5:
Sentence: "웬 영문인지 알지 못하면서도 선생의 기색을 살피고 겁부터 집어먹은 학생은 한동안 어쩔 줄 모르다가 간신히 모기만 한 소리로,\\n\"저를 부르셨어요?\"\\n하고 묻는다.\\n\"그래 불렀다. 왜!\"\\n팍 무는 듯이 한마디 하고 나서 매우 못마땅한 것처럼 교의를 우당퉁탕 당겨서 철썩 주저앉았다가 그저 서 있는 걸 보면,\\n\"장승이냐? 왜 앉지를 못해!\"\\n하고 또 소리를 빽 지르는 법이었다.\\n스승과 제자는 조그마한 책상 하나를 새에 두고 마주 앉는다.\\n앉은 뒤에도,\\n\"네 죄상을 네가 알지!\"\\n하는 것처럼 아무 말 없이 눈살로 쏘기만 하다가 한참 만에야 그 편지를 끄집어내어 학생의 코앞에 동댕이치며,\\n\"이건 누구한테 오는 거냐?\"\\n하고, 문초를 시작한다.\\n앞장에 제 이름이 쓰였는지라,\\n\"저한테 온 것이에요.\"\\n하고, 대답하지 않을 수 없다.\\n그러면 발신인이 누구인 것을 재차 묻는다.\\n그런 편지의 항용으로 발신인의 성명이 똑똑지 않기 때문에 주저주저하다가 자세히 알 수 없다고 내 대일 양이면,\\n\"너한테 오는 것을 네가 모른단 말이냐?\"\\n고, 불호령을 내린 뒤에 또 사연을 읽어 보라 하여 무심한 학생이 나직나직하나마 꿀 같은 구절을 입술에 올리면, B 여사의 역정은 더욱 심해져서 어느 놈의 소위인 것을 기어이 알려 한다.\\n기실 보지도 듣지도 못한 남성의 한 노릇이요, 자기에게는 아무 죄도 없는 것을 변명하여도 곧이듣지를 않는다.\\n바른대로 아뢰어야 망정이지 그렇지 않으면 퇴학을 시킨다는 둥, 제 이름도 모르는 여자에게 편지할 리가 만무하다는 둥, 필연 행실이 부정한 일이 있으리라는 둥…"
JSON Output:
{{
"unpredictable_word": "집어먹은"
}}
---

Now, analyze this sentence:
Sentence: "{sentence}"
JSON Output:"""

def load_model():
    """GPT-OSS 20B model and tokenizer loading"""
    print(f"[INFO] Loading GPT-OSS 20B model: {MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        model.eval()
        print(f"[INFO] Model loaded successfully on {DEVICE}")
        return model, tokenizer
        
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None, None

def generate_with_model(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text using the model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Ensure all tensors are in the same dtype as the model
        inputs = {k: v.to(device=model.device, dtype=model.dtype) if v.dtype.is_floating_point else v.to(model.device) 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only newly generated tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"[ERROR] Text generation failed: {e}")
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
            
            data = data[:10]
            
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
    # Debug: limit to first 3 items for testing
    return all_data[:3]

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
            output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss20b.json")
            print(f"\n[INFO] Intermediate save: saving {len(results)} results")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Final save
    output_path = os.path.join(OUTPUT_DIR, "hrm8k_gold_labels_gptoss20b.json")
    print(f"\n[SUCCESS] Processing complete!")
    print(f"  - Successfully processed sentences: {len(results)}")
    print(f"  - Errors or skipped sentences: {error_count}")
    print(f"  - Result file: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_datasets()