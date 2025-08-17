#!/usr/bin/env python3
"""
gptoss_generate_tow.py

GPT-OSS 20B 모델을 사용하여 'context'와 'gold_label'을 바탕으로
ToW(Thought-of-Word) 설명을 생성합니다.

gold_word.py에서 생성된 데이터를 입력으로 받습니다.
"""
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 설정 (Configuration) ---
MODEL_PATH = "../1_models/gpt-oss-20b"
INPUT_JSON_PATH = "./gold_labels/hrm8k_gold_labels_gptoss20b.json"
OUTPUT_JSON_PATH = "./tow_data/hrm8k_tow_gptoss20b.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 배치 처리 설정
BATCH_SIZE = 1  # GPT-OSS 20B는 큰 모델이므로 배치 크기를 작게 설정
SAVE_INTERVAL = 50  # 50개 처리할 때마다 저장

# ToW 프롬프트 템플릿
FEW_SHOT_PROMPT_TEMPLATE = """
**[Role and Instructions]**
You are an expert literary critic and a language reasoning AI. Your mission is to analyze and explain precisely why a specific 'Next Word' is the necessary and logical continuation of the given 'Context'. Your entire explanation must be enclosed within <ToW> and </ToW> tags, adhering to the following rules:

1.  **Logical Connection**: Analyze the flow, mood, metaphors, and causal relationships within the context to explain how the word is logically connected.
2.  **Necessity**: Emphasize why this particular word is the most fitting and essential choice compared to any other alternatives.
3.  **Clarity and Brevity**: Provide a concise and clear explanation, focusing on the core reasons.
4.  **Output Language**: Output in English.

---

**[Example 1]**
**Input:**
- **Context:** C 여학교에서 교원 겸 기숙사 사감 노릇을 하는 B 여사라면 딱장대요 독신주의자요 찰진 야소군으로 유명하다. 사십에 가까운 노처녀인 그는 주근깨투성이 얼굴이 처녀다운 맛이란 약에 쓰려도 찾을 수 없을 뿐인가, 시들고 거칠고 마르고 누렇게 뜬 품이
- **Next Word:** 곰팡

**Output:**
<ToW>The context describes the appearance of "Lady B" using a series of negative adjectives such as 'withered,' 'coarse,' 'gaunt,' and 'sallow,' which evoke a sense of lifelessness and age. This process builds a cumulative image of decay and deterioration. The word that most effectively encapsulates and serves as a metaphor for this collective state of negativity is 'mold.' The word 'mold' completes this image of decay and plays a crucial role in powerfully imprinting an unpleasant impression of the character upon the reader.</ToW>

---

**[Example 2]**
**Input:**
- **Context:** 웬 영문인지 알지 못하면서도 선생의 기색을 살피고 겁부터
- **Next Word:** 집어먹은

**Output:**
<ToW>The word '집어먹은' (jibeomeogeun) intensifies the Korean idiom for being frightened, which literally translates to "to eat fear." It changes the nuance from passively feeling fear to being actively and completely consumed by terror. This powerful choice of verb effectively maximizes the scene's tension.</ToW>

---

**[Example 3]**
**Input:**
- **Context:** 소리 나는 방은 어렵지 않게 찾을 수 있었다.\\n찾고는 나무로 깎아 세운 듯이 주춤 걸음을 멈출 만큼 그들은 놀래었다.\\n그런 소리의 출처야말로 자기네 방에서 몇 걸음 안 되는 사감실일 줄이야!\\n그 방에 여전히 사내의 비대발괄하는 푸념이 되풀이되고 있다….\\n\\\"나의 천사, 나의 하늘, 나의 여왕, 나의 목숨, 나의 사랑, 나의 애를 말려 죽이실 테요. 나의 가슴을 뜯어 죽이실 테요. 내 생명을 맡으신 당신의 입술로….\\\"\\n셋째 처녀는 대담스럽게 그 방문을 빠끔히 열었다.\\n그 틈으로 여섯 눈이 방안을 향해 쏘았다.\\n이 어쩐
- **Next Word:** 기괴한

**Output:**
<ToW>The context creates a stark contrast between the strict housemother's persona and the sentimental love declarations heard from her room. This dissonance makes the scene not merely surprising, but illogical and absurd. The word '기괴한' (bizarre/grotesque) is the most precise choice because it perfectly captures this unsettling and nonsensical quality, fitting the surreal climax of the moment.</ToW>


---

**[Actual Work]**

**Input:**
- **Context:** {context}
- **Next Word:** {gold_label}

**Output:**
"""

def load_model():
    """GPT-OSS 20B 모델과 토크나이저를 로드합니다."""
    print(f"[INFO] GPT-OSS 20B 모델을 로드합니다: {MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        model.eval()
        print(f"[INFO] 모델이 {DEVICE}에 성공적으로 로드되었습니다.")
        return model, tokenizer
        
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return None, None

def generate_with_model(model, tokenizer, prompt, max_new_tokens=512):
    """모델을 사용하여 텍스트를 생성합니다."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,  # 약간의 창의성을 허용하되 일관성을 유지
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 새로 생성된 토큰만 디코딩
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"[ERROR] 텍스트 생성 실패: {e}")
        return None

def generate_tow_dataset():
    """GPT-OSS 20B 모델을 사용하여 ToW 데이터셋을 생성합니다."""
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    # 모델 로드
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("[ERROR] 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 입력 데이터 로드
    print(f"[INFO] '{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {INPUT_JSON_PATH}")
        print("[INFO] 먼저 gptoss_generate_gold_word.py를 실행하여 gold label 데이터를 생성하세요.")
        return
    
    # 이미 처리된 결과를 불러와서 이어하기
    results = []
    processed_ids = set()
    if os.path.exists(OUTPUT_JSON_PATH):
        print(f"[INFO] 기존 출력 파일 '{OUTPUT_JSON_PATH}'을(를) 발견했습니다. 이어서 작업을 시작합니다.")
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            results = json.load(f)
        processed_ids = {item['id'] for item in results}
        print(f"[INFO] {len(processed_ids)}개의 항목이 이미 처리되었습니다.")

    # 처리해야 할 데이터만 필터링
    tasks_to_run = [item for item in data if item['id'] not in processed_ids]
    if not tasks_to_run:
        print("[SUCCESS] 모든 항목이 이미 처리되었습니다. 프로그램을 종료합니다.")
        return

    error_count = 0
    last_save_count = len(results)

    print(f"[INFO] 총 {len(tasks_to_run)}개의 신규 항목에 대해 ToW 생성을 시작합니다.")

    for item in tqdm(tasks_to_run, desc="Generating ToW"):
        try:
            # ToW 프롬프트 생성
            prompt = FEW_SHOT_PROMPT_TEMPLATE.format(
                context=item['context'], 
                gold_label=item['gold_label']
            )
            
            # 모델 생성
            tow_content = generate_with_model(model, tokenizer, prompt)
            
            if tow_content is None:
                error_count += 1
                continue
            
            # 기존 item에 'tow' 키 추가
            enhanced_item = item.copy()
            enhanced_item['tow'] = tow_content
            results.append(enhanced_item)

        except Exception as e:
            print(f"[ERROR] 처리 중 오류 발생 (ID: {item['id']}): {e}")
            error_count += 1
            continue
        
        # 주기적 저장
        if len(results) - last_save_count >= SAVE_INTERVAL:
            print(f"\n[INFO] 중간 저장: {len(results)}개의 누적 결과를 파일에 저장합니다.")
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_save_count = len(results)

    # 모든 처리가 끝난 후 최종 저장
    print(f"\n[SUCCESS] ToW 데이터셋 생성이 완료되었습니다.")
    print(f"  - 성공적으로 처리된 신규 항목: {len(tasks_to_run) - error_count}")
    print(f"  - 오류 또는 건너뛴 항목: {error_count}")
    print(f"  - 총 저장된 항목 수: {len(results)}")
    print(f"  - 결과 파일: {OUTPUT_JSON_PATH}")
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_tow_dataset()