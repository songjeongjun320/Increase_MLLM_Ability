#!/usr/bin/env python3
"""
generate_gold_word_with_gemini.py

GCP Vertex AI (Gemini 1.5 Pro)를 사용하여
문장에서 가장 예측하기 어려운 단어를 JSON 형식으로 생성하고,
그 결과를 파싱하여 최종 데이터셋을 구축합니다.
"""
import json
import os
import re
import time  # API 요청 간 딜레이를 위해 추가
from tqdm import tqdm

# =================================================================
# 수정 1: Vertex AI SDK 임포트
# =================================================================
# (주의) 라이브러리가 설치되어 있어야 합니다: pip install google-cloud-aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- 설정 (Configuration) ---
# =================================================================
# 수정 2: GCP 및 Gemini 모델 설정으로 변경
# =================================================================
# TODO: 본인의 Google Cloud 프로젝트 ID와 리전을 입력하세요.
# 터미널에서 `gcloud config get-value project` 명령으로 프로젝트 ID를 확인할 수 있습니다.
PROJECT_ID = "your-gcp-project-id"  # <<< 자신의 GCP 프로젝트 ID로 수정
LOCATION = "us-central1"           # <<< 모델을 사용할 리전으로 수정

# 사용할 Gemini 모델 설정
GEMINI_MODEL_ID = "gemini-1.5-pro-001" # Gemini 1.5 Pro 최신 모델 (Vertex AI Model Garden 참고)

# 입/출력 파일 경로는 그대로 사용
INPUT_JSON_PATH = "./klue_all.json"
OUTPUT_JSON_PATH = "/scratch/jsong132/Increase_MLLM_Ability/4_tow_generation/klue_next_word_prediction_gemini_1.5_pro.json"

# =================================================================
# 프롬프트는 기존 형식을 그대로 유지합니다.
# =================================================================
def create_prompt(sentence: str) -> str:
    """
    모델이 예측하기 가장 어려운 단어를 JSON 형식으로 출력하도록 유도하는
    상세한 Few-shot 프롬프트를 생성합니다.
    """
    # (기존 프롬프트 내용과 동일하므로 생략)
    return f"""You are a language prediction expert. Your task is to find the single most unpredictable or surprising word in a given Korean sentence. This word is often a proper noun, a specific number, or a key piece of information that cannot be easily guessed.

Analyze the sentence and output your answer in a JSON format with a single key "unpredictable_word".

---
Example 1:
Sentence: "심청효행대상은 가천문화재단 설립자인 이길여 가천길재단 회장이 지난 1999년에 고전소설 ‘심청전’의 배경인 인천광역시 옹진군 백령면에 심청동상을 제작, 기증한 것을 계기로 제정되었다."
JSON Output:
{{
"unpredictable_word": "회장이"
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
# 수정 3: 함수 이름을 명확하게 바꾸고, 로직을 Gemini API 호출 방식으로 전면 수정
# =================================================================
def generate_prediction_dataset_with_gemini():
    """Gemini API를 사용하여 데이터셋을 생성합니다."""

    # Vertex AI 초기화
    print(f"[INFO] Vertex AI를 초기화합니다. (Project: {PROJECT_ID}, Location: {LOCATION})")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Gemini 모델 로드
    print(f"[INFO] Gemini 모델 '{GEMINI_MODEL_ID}'을(를) 로드합니다.")
    model = GenerativeModel(GEMINI_MODEL_ID)
    
    # 모델 생성 옵션 (JSON 출력을 안정적으로 받기 위함)
    generation_config = GenerationConfig(
        temperature=0.0,  # 항상 동일한 결과를 얻기 위해 0으로 설정
        max_output_tokens=50 # JSON 객체를 받기에 충분한 토큰
    )

    # 입력 데이터 로드
    print(f"[INFO] '{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {INPUT_JSON_PATH}")
        return

    data = data[:5]  # 테스트를 위해 100개만 사용
    results = []
    error_count = 0
    print(f"[INFO] 총 {len(data)}개의 문장 처리를 시작합니다...")

    # 데이터를 순회하며 API 호출 (tqdm으로 진행률 표시)
    for item in tqdm(data, desc="Generating with Gemini"):
        original_sentence = item['sentence']
        prompt = create_prompt(original_sentence)

        try:
            # Gemini API 호출
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            raw_output = response.text
            
            predicted_word = None

            # 모델이 생성한 텍스트에서 JSON 부분만 추출
            json_match = re.search(r'{\s*"unpredictable_word":\s*".*?"\s*}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                predicted_word = parsed_json.get("unpredictable_word")
            
            if not predicted_word:
                # print(f"\n[Warning] JSON에서 단어를 추출하지 못했습니다. Raw output: '{raw_output}'")
                error_count += 1
                continue

            # 원본 문장에서 단어 위치 찾기
            if predicted_word in original_sentence:
                index = original_sentence.index(predicted_word)
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
                # print(f"\n[Warning] 예측된 단어 '{predicted_word}'가 원본 문장에 없습니다.")
                error_count += 1
                continue

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # JSON 파싱 실패 또는 단어를 찾지 못한 경우
            # print(f"\n[Warning] 처리 실패. Error: {e}\nRaw output: '{raw_output}'")
            error_count += 1
            continue
        except Exception as e:
            # Vertex AI API 관련 에러 등
            print(f"\n[ERROR] API 요청 중 에러 발생: {e}")
            error_count += 1
            # API 할당량 초과 등의 문제일 수 있으므로 잠시 대기
            time.sleep(1)
            continue
            
    print(f"\n[SUCCESS] 데이터셋 생성이 완료되었습니다.")
    print(f"  - 성공적으로 처리된 문장: {len(results)}")
    print(f"  - 오류 또는 건너뛴 문장: {error_count}")
    
    # 결과 저장
    print(f"  - '{OUTPUT_JSON_PATH}' 파일에 결과를 저장합니다.")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 스크립트 실행 전, 터미널에서 GCP 인증이 필요합니다:
    # gcloud auth application-default login
    generate_prediction_dataset_with_gemini()