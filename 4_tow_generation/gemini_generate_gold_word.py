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

# =================================================================
# 수정 1: Vertex AI SDK 임포트
# =================================================================
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- 설정 (Configuration) ---
# =================================================================
# 수정 2: GCP 및 Gemini 모델 설정으로 변경
# =================================================================
PROJECT_ID = "apt-summer-468801-i9"
LOCATION = "us-central1"

GEMINI_MODEL_ID = "gemini-2.0-flash-lite"

INPUT_JSON_PATH = "./koconovel.json"
OUTPUT_JSON_PATH = "koconovel_next_word_prediction_gemini_2.0-flash-lite.json"

# =================================================================
# 수정 3: 배치 크기 및 저장 주기 설정 추가
# =================================================================
BATCH_SIZE = 3  # 한 번에 처리할 문장의 수 (병렬 요청 개수)
SAVE_INTERVAL = 100 # 몇 개의 문장을 처리할 때마다 저장할지 결정

# =================================================================
# 프롬프트는 기존 형식을 그대로 유지합니다.
# =================================================================
def create_prompt(sentence: str) -> str:
    """
    모델이 예측하기 가장 어려운 단어를 JSON 형식으로 출력하도록 유도하는
    상세한 Few-shot 프롬프트를 생성합니다.
    """
    # (프롬프트 내용은 질문과 동일하므로 생략)
    return f"""You are a language prediction expert. Your task is to find the single most unpredictable or surprising word in a given Korean sentence. This word is often a proper noun, a specific number, or a key piece of information that cannot be easily guessed.

Analyze the sentence and output your answer in a JSON format with a single key "unpredictable_word". Don't choose proper noun such as name, date, time and number.

---
Example 1:
Sentence: "심청효행대상은 가천문화재단 설립자인 이길여 가천길재단 회장이 지난 1999년에 고전소설 ‘심청전’의 배경인 인천광역시 옹진군 백령면에 심청동상을 제작, 기증한 것을 계기로 제정되었다."
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
Sentence: "C 여학교에서 교원 겸 기숙사 사감 노릇을 하는 B 여사라면 딱장대요 독신주의자요 찰진 야소군으로 유명하다.\n사십에 가까운 노처녀인 그는 주근깨투성이 얼굴이 처녀다운 맛이란 약에 쓰려도 찾을 수 없을 뿐인가, 시들고 거칠고 마르고 누렇게 뜬 품이 곰팡 슬은 굴비를 생각나게 한다.\n여러 겹주름이 잡힌 훨렁 벗겨진 이마라든지, 숱이 적어서 법대로 쪽지거나 틀어 올리지를 못하고 엉성하게 그냥 빗어넘긴 머리꼬리가 뒤통수에 염소 똥만 하게 붙은 것이라든지, 벌써 늙어가는 자취를 감출 길이 없었다.\n뾰족한 입을 앙다물고 돋보기 너머로 쌀쌀한 눈이 노릴 때엔 기숙생들이 오싹하고 몸서리를 치리만큼 그는 엄격하고 매서웠다.\n이 B 여사가 질겁을 하다시피 싫어하고 미워하는 것은 소위 '러브레터'였다.\n여학교 기숙사라면 으레 그런 편지가 많이 오는 것이지만 학교로도 유명하고 또 아름다운 여학생이 많은 탓인지 모르되 하루에도 몇 장씩 죽느니 사느니 하는 사랑 타령이 날아들어 왔었다.\n기숙생에게 오는 사신을 일일이 검토하는 터이니까 그따위 편지도 물론 B 여사의 손에 떨어진다.\n달짝지근한 사연을 보는 족족 그는 더할 수 없이 흥분되어서 얼굴이 붉으락푸르락, 편지 든 손이 발발 떨리도록 성을 낸다.\n아무 까닭 없이 그런 편지를 받은 학생이야말로 큰 재변이었다.\n하학하기가 무섭게 그 학생은 사감실로 불리어 간다.\n분해서 못 견디겠다는 사람 모양으로 쌔근쌔근하며 방안을 왔다 갔다 하던 그는, 들어오는 학생을 잡아먹을 듯이 노리면서 한 걸음 두 걸음 코가 맞닿을 만큼 바싹 다가들어서 딱 마주 선다."
JSON Output:
{{
"unpredictable_word": "얼굴이"
}}

---

Example 5:
Sentence: "웬 영문인지 알지 못하면서도 선생의 기색을 살피고 겁부터 집어먹은 학생은 한동안 어쩔 줄 모르다가 간신히 모기만 한 소리로,\n"저를 부르셨어요?"\n하고 묻는다.\n"그래 불렀다. 왜!"\n팍 무는 듯이 한마디 하고 나서 매우 못마땅한 것처럼 교의를 우당퉁탕 당겨서 철썩 주저앉았다가 그저 서 있는 걸 보면,\n"장승이냐? 왜 앉지를 못해!"\n하고 또 소리를 빽 지르는 법이었다.\n스승과 제자는 조그마한 책상 하나를 새에 두고 마주 앉는다.\n앉은 뒤에도,\n"네 죄상을 네가 알지!"\n하는 것처럼 아무 말 없이 눈살로 쏘기만 하다가 한참 만에야 그 편지를 끄집어내어 학생의 코앞에 동댕이치며,\n"이건 누구한테 오는 거냐?"\n하고, 문초를 시작한다.\n앞장에 제 이름이 쓰였는지라,\n"저한테 온 것이에요."\n하고, 대답하지 않을 수 없다.\n그러면 발신인이 누구인 것을 재차 묻는다.\n그런 편지의 항용으로 발신인의 성명이 똑똑지 않기 때문에 주저주저하다가 자세히 알 수 없다고 내 대일 양이면,\n"너한테 오는 것을 네가 모른단 말이냐?"\n고, 불호령을 내린 뒤에 또 사연을 읽어 보라 하여 무심한 학생이 나직나직하나마 꿀 같은 구절을 입술에 올리면, B 여사의 역정은 더욱 심해져서 어느 놈의 소위인 것을 기어이 알려 한다.\n기실 보지도 듣지도 못한 남성의 한 노릇이요, 자기에게는 아무 죄도 없는 것을 변명하여도 곧이듣지를 않는다.\n바른대로 아뢰어야 망정이지 그렇지 않으면 퇴학을 시킨다는 둥, 제 이름도 모르는 여자에게 편지할 리가 만무하다는 둥, 필연 행실이 부정한 일이 있으리라는 둥…"
JSON Output:
{{
"unpredictable_word": "집어먹은"
}}
---

Now, analyze this sentence:
Sentence: "{sentence}"
JSON Output:"""

# =================================================================
# 수정 4: 비동기 배치 처리 및 주기적 저장을 위한 함수로 전면 수정
# =================================================================
async def generate_prediction_dataset_async():
    """Gemini API를 비동기 배치로 호출하고 주기적으로 저장하여 데이터셋을 생성합니다."""

    # Vertex AI 초기화
    print(f"[INFO] Vertex AI를 초기화합니다. (Project: {PROJECT_ID}, Location: {LOCATION})")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Gemini 모델 로드
    print(f"[INFO] Gemini 모델 '{GEMINI_MODEL_ID}'을(를) 로드합니다.")
    model = GenerativeModel(GEMINI_MODEL_ID)
    
    generation_config = GenerationConfig(
        temperature=0.0,
        max_output_tokens=50
    )

    # 입력 데이터 로드
    print(f"[INFO] '{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {INPUT_JSON_PATH}")
        return
    
    # 데이터를 BATCH_SIZE 크기의 묶음(배치)으로 나눔
    batches = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    
    results = []
    error_count = 0
    processed_count = 0
    last_save_count = 0

    print(f"[INFO] 총 {len(data)}개의 문장을 {len(batches)}개의 배치로 나누어 처리를 시작합니다. (배치 크기: {BATCH_SIZE})")

    for batch in tqdm(batches, desc="Generating with Gemini (Batch)"):
        tasks = []
        for item in batch:
            prompt = create_prompt(item['sentence'])
            # model.generate_content_async를 사용하여 비동기 태스크 생성
            task = model.generate_content_async(prompt, generation_config=generation_config)
            tasks.append(task)
        
        # asyncio.gather를 사용해 현재 배치의 모든 API 요청을 병렬로 실행
        # return_exceptions=True: 하나의 요청이 실패해도 전체가 멈추지 않음
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"\n[ERROR] 배치 처리 중 심각한 오류 발생: {e}")
            error_count += len(batch)
            continue # 다음 배치로 넘어감

        # 응답 처리
        for item, response in zip(batch, responses):
            processed_count += 1
            if isinstance(response, Exception):
                # API 호출 중 예외가 발생한 경우
                print(f"\n[Warning] API 요청 실패. Error: {response}")
                error_count += 1
                continue

            try:
                raw_output = response.text
                predicted_word = None
                
                json_match = re.search(r'{\s*"unpredictable_word":\s*".*?"\s*}', raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    predicted_word = parsed_json.get("unpredictable_word")

                if not predicted_word:
                    error_count += 1
                    continue

                original_sentence = item['sentence']
                if predicted_word in original_sentence:
                    index = original_sentence.find(predicted_word) # find가 더 안전
                    context = original_sentence[:index].strip()
                    gold_label = predicted_word

                    if not context:
                        error_count += 1
                        continue

                    new_item = {
                        'id': item['id'],
                        'original_sentence': original_sentence,
                        'context': context,
                        'gold_label': gold_label
                    }
                    results.append(new_item)
                else:
                    error_count += 1
                    continue

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                error_count += 1
                continue
            except Exception as e:
                print(f"\n[ERROR] 응답 처리 중 에러 발생: {e}")
                error_count += 1
                continue
        
        # 주기적 저장 로직
        if len(results) - last_save_count >= SAVE_INTERVAL:
            print(f"\n[INFO] 중간 저장: {len(results)}개의 결과를 파일에 저장합니다. (총 {processed_count}개 처리)")
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_save_count = len(results)

    # 모든 배치가 끝난 후 최종 저장
    print(f"\n[SUCCESS] 데이터셋 생성이 완료되었습니다.")
    print(f"  - 성공적으로 처리된 문장: {len(results)}")
    print(f"  - 오류 또는 건너뛴 문장: {error_count}")
    
    print(f"  - '{OUTPUT_JSON_PATH}' 파일에 최종 결과를 저장합니다.")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 스크립트 실행 전, 터미널에서 GCP 인증이 필요합니다:
    # gcloud auth application-default login
    
    # asyncio.run()을 사용하여 비동기 함수 실행
    asyncio.run(generate_prediction_dataset_async())