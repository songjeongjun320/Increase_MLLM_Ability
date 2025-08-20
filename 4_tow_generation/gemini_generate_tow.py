#!/usr/bin/env python3
"""
generate_tow_batch.py

GCP Vertex AI (Gemini)를 사용하여 'context'와 'gold_label'을 바탕으로
ToW(Thought-of-Word) 설명을 생성합니다.

(수정) 비동기 배치 처리와 주기적 저장을 통해 속도와 안정성을 개선합니다.
"""
import json
import os
import asyncio
import time
from tqdm import tqdm
import random

# =================================================================
# Vertex AI SDK 임포트
# =================================================================
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- 설정 (Configuration) ---
# =================================================================
# GCP 프로젝트 및 Gemini 모델 설정
# =================================================================
PROJECT_ID = "apt-summer-468801-i9"
LOCATION = "us-central1"

GEMINI_MODEL_ID = "gemini-2.0-flash-lite"

INPUT_JSON_PATH = "./gold_labels/klue_next_word_prediction_gemini_2.0-flash-lite.json"
OUTPUT_JSON_PATH = "./klue_tow_gemini_2.0-flash-lite.json"

# =================================================================
# 배치 크기 및 저장 주기 설정
# =================================================================
BATCH_SIZE = 5  # 한 번에 처리할 요청의 수 (병렬 처리 개수)
SAVE_INTERVAL = 1000 # 몇 개의 결과를 처리할 때마다 저장할지 결정

# =================================================================
# 프롬프트는 질문에 제공된 내용을 그대로 사용합니다.
# =================================================================
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
- **Context:** 소리 나는 방은 어렵지 않게 찾을 수 있었다.\n찾고는 나무로 깎아 세운 듯이 주춤 걸음을 멈출 만큼 그들은 놀래었다.\n그런 소리의 출처야말로 자기네 방에서 몇 걸음 안 되는 사감실일 줄이야!\n그 방에 여전히 사내의 비대발괄하는 푸념이 되풀이되고 있다….\n\"나의 천사, 나의 하늘, 나의 여왕, 나의 목숨, 나의 사랑, 나의 애를 말려 죽이실 테요. 나의 가슴을 뜯어 죽이실 테요. 내 생명을 맡으신 당신의 입술로….\"\n셋째 처녀는 대담스럽게 그 방문을 빠끔히 열었다.\n그 틈으로 여섯 눈이 방안을 향해 쏘았다.\n이 어쩐
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

# --- 메인 실행 로직 (비동기 배치 처리 및 주기적 저장) ---
async def generate_tow_dataset_async():
    """Gemini API를 비동기 배치로 호출하고 주기적으로 저장하여 ToW 데이터셋을 생성합니다."""

    # Vertex AI 초기화
    print(f"[INFO] Vertex AI를 초기화합니다. (Project: {PROJECT_ID}, Location: {LOCATION})")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Gemini 모델 로드
    print(f"[INFO] Gemini 모델 '{GEMINI_MODEL_ID}'을(를) 로드합니다.")
    model = GenerativeModel(GEMINI_MODEL_ID)
    
    # 생성 설정: ToW 설명이 길 수 있으므로 max_output_tokens를 충분히 확보
    generation_config = GenerationConfig(
        temperature=0.2, # 약간의 창의성을 허용하되 일관성을 유지
        max_output_tokens=512
    )

    # 입력 데이터 로드
    print(f"[INFO] '{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {INPUT_JSON_PATH}")
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

    # 데이터를 BATCH_SIZE 크기의 묶음(배치)으로 나눔
    batches = [tasks_to_run[i:i + BATCH_SIZE] for i in range(0, len(tasks_to_run), BATCH_SIZE)]
    
    error_count = 0
    last_save_count = len(results)

    print(f"[INFO] 총 {len(tasks_to_run)}개의 신규 항목을 {len(batches)}개의 배치로 나누어 처리를 시작합니다. (배치 크기: {BATCH_SIZE})")

    for batch in tqdm(batches, desc="Generating ToW with Gemini (Batch)"):
        tasks = []
        for item in batch:
            # (수정) ToW 프롬프트 형식에 맞게 context와 gold_label을 사용합니다.
            prompt = FEW_SHOT_PROMPT_TEMPLATE.format(context=item['context'], gold_label=item['gold_label'])
            # model.generate_content_async를 사용하여 비동기 태스크 생성
            task = model.generate_content_async(prompt, generation_config=generation_config)
            tasks.append(task)
        
        # asyncio.gather를 사용해 현재 배치의 모든 API 요청을 병렬로 실행
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"\n[ERROR] 배치 처리 중 심각한 오류 발생: {e}")
            error_count += len(batch)
            continue # 다음 배치로 넘어감

        # 응답 처리
        for item, response in zip(batch, responses):
            if isinstance(response, Exception):
                print(f"\n[Warning] API 요청 실패 (ID: {item['id']}). Error: {response}")
                error_count += 1
                continue

            try:
                # (수정) 결과 처리 로직: 응답 텍스트를 그대로 'tow' 키에 저장합니다.
                tow_content = response.text.strip()
                item['tow'] = tow_content # 기존 item 딕셔너리에 'tow' 키 추가
                results.append(item)

            except Exception as e:
                print(f"\n[ERROR] 응답 처리 중 에러 발생 (ID: {item['id']}): {e}")
                error_count += 1
                continue
        
        # 주기적 저장 로직
        if len(results) - last_save_count >= SAVE_INTERVAL:
            print(f"\n[INFO] 중간 저장: {len(results)}개의 누적 결과를 파일에 저장합니다.")
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_save_count = len(results)
            
        # API 속도 제한을 피하기 위한 짧은 대기
        await asyncio.sleep(random.uniform(0, 4))

    # 모든 배치가 끝난 후 최종 저장
    print(f"\n[SUCCESS] ToW 데이터셋 생성이 완료되었습니다.")
    print(f"  - 성공적으로 처리된 신규 항목: {len(tasks_to_run) - error_count}")
    print(f"  - 오류 또는 건너뛴 항목: {error_count}")
    print(f"  - 총 저장된 항목 수: {len(results)}")
    
    print(f"  - '{OUTPUT_JSON_PATH}' 파일에 최종 결과를 저장합니다.")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 스크립트 실행 전, 터미널에서 GCP 인증이 필요합니다:
    # gcloud auth application-default login
    
    # asyncio.run()을 사용하여 비동기 함수 실행
    asyncio.run(generate_tow_dataset_async())