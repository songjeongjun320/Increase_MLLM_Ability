#!/usr/bin/env python3
"""
generate_gold_word_with_gemini.py

GCP Vertex AI (Gemini 1.5 Pro)를 사용하여
문장에서 가장 예측하기 어려운 단어를 JSON 형식으로 생성하고,
그 결과를 파싱하여 최종 데이터셋을 구축합니다.

(수정) 배치(Batch) 처리와 주기적 저장을 통해 속도와 안정성을 개선합니다.
"""
import json
import os
import re
import time
import asyncio  # 비동기 처리를 위해 추가
from tqdm import tqdm
import random
import logging # 로깅 모듈 추가

# =================================================================
# 수정 1: Vertex AI SDK 임포트
# =================================================================
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold

# --- 설정 (Configuration) ---
# =================================================================
# 수정 2: GCP 및 Gemini 모델 설정으로 변경
# =================================================================
PROJECT_ID = "gen-lang-client-0996841973"
LOCATION = "us-central1"

GEMINI_MODEL_ID = "gemini-2.0-flash"

INPUT_JSON_PATH = "./extracted_data/extract_over_19token.json"
OUTPUT_JSON_PATH = "./gold_labels_extracted_data_kr_prompt/extract_over_19token_next_word_prediction_gemini_2.0-flash.json"

# =================================================================
# 수정 3: 배치 크기 및 저장 주기 설정 추가
# =================================================================
BATCH_SIZE = 10  # 한 번에 처리할 문장의 수 (병렬 요청 개수)
SAVE_INTERVAL = 1000 # 몇 개의 문장을 처리할 때마다 저장할지 결정

# =================================================================
# 프롬프트는 기존 형식을 그대로 유지합니다.
# =================================================================
def create_prompt(sentence: str) -> str:
    """
    모델이 예측하기 가장 어려운 단어를 JSON 형식으로 출력하도록 유도하는
    상세한 Few-shot 프롬프트를 생성합니다.
    """
    return f"""
당신의 임무는 주어진 한국어 문장 내에서 가장 **"핵심적인(key)" 단어**들을 식별하는 것입니다.
"핵심적인" 단어란, 주변 문맥만으로는 쉽게 추측할 수 없는 단어를 의미합니다. 이런 단어는 종종 새롭고 놀라운 정보를 도입하거나, 문장의 의미나 방향에 있어 중요한 전환점을 만듭니다.
---

## 엄격한 규칙
단어를 선택할 때는 아래의 규칙들을 반드시 따라야 합니다.
1.  **핵심 원칙:**
    *   선택된 단어는 놀라움을 주어야 하지만, **일단 읽고 나면 말이 되어야 합니다.**
    *   즉, 앞선 문맥이 그 단어를 추측하는 데 필요한 단서의 일부는 제공하지만, 전부는 아니어야 합니다.
2.  **선택 기준:**
    *   문맥적으로 놀라움을 주는 단어를 선택해야 합니다.
3.  **제외 규칙:**
    *   **고유명사** (예: 이름, 장소, 지역, 브랜드, 날짜, 시간, 요일, 월, 년)는 선택하면 안 됩니다.
    *   문장의 **첫 번째 단어**는 선택하면 안 됩니다.
    *   **숫자**는 선택하면 안 됩니다.
4.  **인접성 규칙:**
    *   선택된 단어들은 문장 내에서 서로 바로 옆에 붙어 있으면 안 됩니다.
    *   두 개의 선택된 단어 사이에는 최소 한 개 이상의 다른 단어가 있어야 합니다.
5.  **수량:**
    *   선택하는 단어의 개수는 문장의 길이나 복잡성에 따라 달라질 수 있습니다.
---

## 출력 형식
*   당신의 출력물은 반드시 **`"key_word"`**라는 단일 키를 가진 **JSON 형식**이어야 합니다.
*   **단일 단어 선택 시:** 값(value)은 **문자열(string)**이어야 합니다.
---

Example 1:
Sentence: "숙소 예약한 시간까지 택시 타고 숙소에 도착하고 싶은데 삼청각에서 출발할 겁니다."
JSON Output:
{{
  "key_word": "도착하고"
}}

---
Example 2:
Sentence: "심청효행대상은 가천문화재단 설립자인 이길여 가천길재단 회장이 지난 1999년에 고전소설 ‘심청전’의 배경인 인천광역시 옹진군 백령면에 심청동상을 제작, 기증한 것을 계기로 제정되었다."
JSON Output:
```json
{{
  "key_word": ["회장이", "기증한"]
}}
```
---

Example 3:
Sentence: "C 여학교에서 교원 겸 기숙사 사감 노릇을 하는 B 여사라면 딱장대요 독신주의자요 찰진 야소군으로 유명하다.\n사십에 가까운 노처녀인 그는 주근깨투성이 얼굴이 처녀다운 맛이란 약에 쓰려도 찾을 수 없을 뿐인가, 시들고 거칠고 마르고 누렇게 뜬 품이 곰팡 슬은 굴비를 생각나게 한다.\n여러 겹주름이 잡힌 훨렁 벗겨진 이마라든지, 숱이 적어서 법대로 쪽지거나 틀어 올리지를 못하고 엉성하게 그냥 빗어넘긴 머리꼬리가 뒤통수에 염소 똥만 하게 붙은 것이라든지, 벌써 늙어가는 자취를 감출 길이 없었다.\n
JSON Output:
```json
{{
  "key_word": ["독신주의자요", "얼굴이", "적어서"]
}}
```

---

Example 4:
Sentence: "웬 영문인지 알지 못하면서도 선생의 기색을 살피고 겁부터 집어먹은 학생은 한동안 어쩔 줄 모르다가 간신히 모기만 한 소리로,\n"저를 부르셨어요?"\n하고 묻는다.\n"그래 불렀다. 왜!"\n팍 무는 듯이 한마디 하고 나서 매우 못마땅한 것처럼 교의를 우당퉁탕 당겨서 철썩 주저앉았다가 그저 서 있는 걸 보면,\n"장승이냐? 왜 앉지를 못해!"\n하고 또 소리를 빽 지르는 법이었다.\n스승과 제자는 조그마한 책상 하나를 새에 두고 마주 앉는다.\n앉은 뒤에도,\n"네 죄상을 네가 알지!"
JSON Output:
```json
{{{{
"key_word": ["집어먹은", "부르셨어요", "못마땅한", "못해", "제자는"]
}}}}
```
---

Now, analyze this sentence:
Sentence: "{sentence}"
JSON Output:"""

# =================================================================
# 수정: 지수 백오프(Exponential Backoff)를 적용한 API 호출 래퍼
# =================================================================
async def generate_with_backoff(model, prompt, generation_config):
    """지수 백오프를 사용하여 API를 호출하고, 실패 시 재시도합니다."""
    max_retries = 5
    base_delay = 1.5  # 초

    for i in range(max_retries):
        try:
            # model.generate_content_async를 사용하여 비동기 태스크 생성
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            return response # 성공 시 결과 반환
        except Exception as e:
            error_type = type(e).__name__
            if i == max_retries - 1:
                # 마지막 재시도도 실패하면 예외를 다시 발생시킴
                print(f"\n[ERROR] API 호출 최종 실패. Error Type: {error_type}, Details: {e}")
                raise e
            
            # 지수적으로 대기 시간 증가 (+ 약간의 무작위성 추가)
            delay = base_delay * (3 ** i) + random.uniform(0, 1)
            print(f"\n[Warning] API 오류 발생 (Type: {error_type}). {delay:.2f}초 후 재시도합니다... (시도 {i+1}/{max_retries})")
            await asyncio.sleep(delay)

# =================================================================
# 수정 4: 비동기 배치 처리 및 주기적 저장을 위한 함수로 전면 수정
# =================================================================
async def generate_prediction_dataset_async():
    """Gemini API를 비동기 배치로 호출하고 주기적으로 저장하여 데이터셋을 생성합니다."""

    # =================================================================
    # 추가: 파일 로거 설정
    # =================================================================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='generation.log',
        filemode='a' # 'w'는 덮어쓰기, 'a'는 이어쓰기
    )
    # 콘솔에도 로그를 출력하기 위한 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


    # Vertex AI 초기화
    logging.info(f"Vertex AI를 초기화합니다. (Project: {PROJECT_ID}, Location: {LOCATION})")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # =================================================================
    # 추가: 안전 필터 비활성화 설정
    # =================================================================
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Gemini 모델 로드
    logging.info(f"Gemini 모델 '{GEMINI_MODEL_ID}'을(를) 로드합니다.")
    model = GenerativeModel(
        GEMINI_MODEL_ID,
        safety_settings=safety_settings
    )
    
    generation_config = GenerationConfig(
        temperature=0.0,
        max_output_tokens=2048
    )

    # =================================================================
    # 추가: 이어서 작업을 시작하기 위한 로직
    # =================================================================
    results = []
    processed_ids = set()
    if os.path.exists(OUTPUT_JSON_PATH):
        logging.info(f"기존 출력 파일 '{OUTPUT_JSON_PATH}'을(를) 확인합니다.")
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content and content.strip(): # 파일이 비어있지 않은지 확인
                    results = json.loads(content)
                    processed_ids = {item['id'] for item in results if 'id' in item}
                    logging.info(f"{len(results)}개의 기존 결과를 불러왔습니다. 이어서 작업을 시작합니다.")
                else:
                    logging.info("기존 출력 파일이 비어있어 처음부터 시작합니다.")
        except (json.JSONDecodeError, FileNotFoundError):
            logging.warning(f"기존 출력 파일을 읽는 데 실패했습니다. 처음부터 다시 시작합니다.")
            results = []
            processed_ids = set()

    # 입력 데이터 로드
    logging.info(f"'{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"입력 파일을 찾을 수 없습니다: {INPUT_JSON_PATH}")
        return


    # 이미 처리된 데이터를 제외
    data_to_process = [item for item in all_data if item.get('id') not in processed_ids]

    # 데이터를 BATCH_SIZE 크기의 묶음(배치)으로 나눔
    batches = [data_to_process[i:i + BATCH_SIZE] for i in range(0, len(data_to_process), BATCH_SIZE)]
    
    error_count = 0
    processed_count_this_run = 0
    last_save_count = len(results)

    logging.info(f"총 {len(all_data)}개 문장 중, 이미 처리된 {len(processed_ids)}개를 제외하고 {len(data_to_process)}개를 {len(batches)}개 배치로 나누어 처리를 시작합니다. (배치 크기: {BATCH_SIZE})")

    # =================================================================
    # 수정: tqdm 진행 표시줄에 상세 정보 추가
    # =================================================================
    progress_bar = tqdm(batches, desc="Generating with Gemini (Batch)")
    try:
        for batch in progress_bar:
            tasks = []
            for item in batch:
                prompt = create_prompt(item['sentence'])
                # 수정: 백오프 래퍼 함수를 사용하여 태스크 생성
                task = generate_with_backoff(model, prompt, generation_config)
                tasks.append(task)
            
            # asyncio.gather를 사용해 현재 배치의 모든 API 요청을 병렬로 실행
            # return_exceptions=True: 하나의 요청이 실패해도 전체가 멈추지 않음
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logging.error(f"배치 처리 중 심각한 오류 발생: {e}")
                error_count += len(batch)
                continue # 다음 배치로 넘어감

            # 응답 처리
            for item, response in zip(batch, responses):
                processed_count_this_run += 1
                if isinstance(response, Exception):
                    # API 호출 중 예외가 발생한 경우
                    error_type = type(response).__name__
                    logging.warning(f"API 요청 실패. ID: {item.get('id')}, Error Type: {error_type}, Details: {response}")
                    error_count += 1
                    continue

                try:
                    raw_output = response.text
                    predicted_words = None  # 단일 단어 또는 리스트를 받을 변수
                    
                    # 수정: 배열과 문자열을 모두 포함할 수 있는 더 유연한 정규식
                    json_match = re.search(r'\{\s*"key_word":\s*(\[.*?\]|".*?")\s*\}', raw_output, re.DOTALL)
                    
                    if json_match:
                        # 그룹 1에 있는 값 (배열 또는 문자열)을 포함하여 전체 JSON을 재구성
                        json_str = f'{{"key_word": {json_match.group(1)}}}'
                        try:
                            parsed_json = json.loads(json_str)
                            predicted_words = parsed_json.get("key_word")
                        except json.JSONDecodeError:
                            logging.warning(f"JSON 파싱 실패. ID: {item.get('id')}, Matched string: {json_str}")
                            error_count += 1
                            continue

                    if not predicted_words:
                        logging.warning(f"파싱 실패. ID: {item.get('id')}, 응답에서 'key_word'를 찾을 수 없음. Raw Output: {raw_output}")
                        error_count += 1
                        continue

                    original_sentence = item['sentence']
                    
                    # 예측된 단어가 리스트가 아니면 리스트로 변환하여 일관성 유지
                    if not isinstance(predicted_words, list):
                        predicted_words = [predicted_words]

                    # 문장 안에 실제로 존재하는 유효한 단어만 필터링
                    valid_words = [word for word in predicted_words if isinstance(word, str) and word in original_sentence]

                    if valid_words:
                        # 첫 번째 유효한 단어를 기준으로 컨텍스트 생성
                        first_word = valid_words[0]
                        index = original_sentence.find(first_word)
                        context = original_sentence[:index].strip()

                        if not context:
                            logging.warning(f"컨텍스트 없음. ID: {item.get('id')}, First Word: '{first_word}', Sentence: '{original_sentence}'")
                            error_count += 1
                            continue

                        # 하나의 객체로 결과 저장
                        new_item = {
                            'id': item['id'],
                            'original_sentence': original_sentence,
                            'context': context,
                            'gold_label': valid_words # gold_label에 단어 리스트 저장
                        }
                        results.append(new_item)
                    else:
                        logging.warning(f"유효한 예측 단어 없음. ID: {item.get('id')}, Predicted: '{predicted_words}', Sentence: '{original_sentence}'")
                        error_count += 1
                        continue
                
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    error_type = type(e).__name__
                    # 수정: raw_output이 없을 수 있으므로 예외 'e'를 직접 로깅
                    logging.warning(f"응답 처리/디코딩 오류. ID: {item.get('id')}, Error: {error_type}, Details: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    error_type = type(e).__name__
                    logging.error(f"응답 처리 중 예외 발생. ID: {item.get('id')}, Error: {error_type}, Details: {e}")
                    error_count += 1
                    continue
            
            # =================================================================
            # 수정: tqdm 진행 표시줄 업데이트
            # =================================================================
            progress_bar.set_postfix(success=len(results), errors=error_count, refresh=True)

            # 주기적 저장 로직
            if len(results) - last_save_count >= SAVE_INTERVAL:
                logging.info(f"중간 저장: 총 {len(results)}개의 결과를 파일에 저장합니다. (이번 실행에서 {processed_count_this_run}개 처리)")
                # 추가: 저장하기 전에 디렉터리가 있는지 확인하고 없으면 생성
                os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                last_save_count = len(results)
                
            await asyncio.sleep(random.uniform(0, 4))
        
        # for 루프가 정상적으로 완료되었을 때 실행
        logging.info(f"데이터셋 생성이 완료되었습니다.")

    finally:
        # =================================================================
        # 추가: 강제 종료 시에도 저장을 보장하는 로직
        # =================================================================
        logging.info(f"프로그램을 종료합니다. 현재까지의 결과를 저장합니다.")
        logging.info(f"  - 총 성공적으로 처리된 문장: {len(results)}")
        logging.info(f"  - 이번 실행에서 오류 또는 건너뛴 문장: {error_count}")
        
        logging.info(f"  - '{OUTPUT_JSON_PATH}' 파일에 최종 결과를 저장합니다.")
        # 추가: 저장하기 전에 디렉터리가 있는지 확인하고 없으면 생성
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("저장 완료.")


if __name__ == "__main__":
    # 스크립트 실행 전, 터미널에서 GCP 인증이 필요합니다:
    # gcloud auth application-default login
    
    # asyncio.run()을 사용하여 비동기 함수 실행
    asyncio.run(generate_prediction_dataset_async())