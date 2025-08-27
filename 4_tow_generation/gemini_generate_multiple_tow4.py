#!/usr/bin/env python3
"""
gemini_generate_multiple_tow.py

GCP Vertex AI (Gemini)를 사용하여 여러 개의 gold_label에 대해 순차적으로 ToW(Thought-of-Word) 설명을 생성합니다.
각 gold_label마다 ToW를 생성하고, 이전 ToW가 다음 context에 포함되도록 처리합니다.

(수정) 비동기 배치 처리와 주기적 저장을 통해 속도와 안정성을 개선합니다.
"""
import json
import os
import asyncio
import time
import signal
import atexit
from tqdm import tqdm
import random
import re

# =================================================================
# Vertex AI SDK 임포트
# =================================================================
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- 설정 (Configuration) ---
# =================================================================
# GCP 프로젝트 및 Gemini 모델 설정
# =================================================================
PROJECT_ID = "data-media-470315-k7"
LOCATION = "us-central1"

GEMINI_MODEL_ID = "gemini-2.5-flash"

INPUT_JSON_PATH = "./multiple_gold_labels_extracted_data_kr_prompt/extract_over_19token_next_word_prediction_gemini_2.0-flash_part4_of_4.json"
OUTPUT_JSON_PATH = "./multiple_tow_data/extract_over_19token_multiple_tow_gemini_2.5-flash_part4_of_4.json"

# Gemini API 비용 (2024년 기준, 1M 토큰당 USD)
GEMINI_COST_PER_1M_INPUT = 0.3   # gemini-flash 입력 비용
GEMINI_COST_PER_1M_OUTPUT = 2.5  # gemini-flash 출력 비용

# =================================================================
# 배치 크기 및 저장 주기 설정
# =================================================================
BATCH_SIZE = 5 # 한 번에 처리할 요청의 수 (다중 ToW 생성으로 인해 감소)
PARALLEL_REQUESTS = 5  # 동시 API 호출 수 (레이트 리밋 고려)
SAVE_INTERVAL = 100 # 몇 개의 결과를 처리할 때마다 저장할지 결정
MAX_TOW_RETRIES = 5  # 각 ToW 생성 최대 재시도 횟수

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

# =================================================================
# 수정: 지수 백오프(Exponential Backoff)를 적용한 API 호출 래퍼
# =================================================================

# API 레이트 리밋 제어용 세마포어
api_semaphore = None

async def generate_with_backoff(model, prompt, generation_config):
    """세마포어와 지수 백오프를 사용하여 API를 호출하고, 실패 시 재시도합니다."""
    max_retries = 5
    base_delay = 2  # 초

    async with api_semaphore:  # 세마포어로 동시 호출 수 제한
        for i in range(max_retries):
            try:
                # model.generate_content_async를 사용하여 비동기 태스크 생성
                response = await model.generate_content_async(prompt, generation_config=generation_config)
                return response # 성공 시 결과 반환
            except Exception as e:
                if i == max_retries - 1:
                    # 마지막 재시도도 실패하면 예외를 다시 발생시킴
                    print(f"\n[ERROR] API 호출 최종 실패: {e}")
                    raise e
                
                # 지수적으로 대기 시간 증가 (+ 약간의 무작위성 추가)
                delay = base_delay * (3 ** i) + random.uniform(0, 1)
                print(f"\n[Warning] API 오류 발생. {delay:.2f}초 후 재시도합니다... (시도 {i+1}/{max_retries})")
                await asyncio.sleep(delay)

def estimate_tokens(text):
    """텍스트의 대략적인 토큰 수를 추정합니다."""
    if not text:
        return 0
    
    # 한글 문자 수 계산 (유니코드 범위: 0xAC00-0xD7AF)
    korean_chars = len(re.findall(r'[가-힣]', text))
    
    # 영문 단어 수 계산
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    
    # 숫자와 특수문자
    others = len(re.findall(r'[0-9\W]', text))
    
    # 토큰 추정: 한글 1.5토큰/문자, 영문 1토큰/단어, 기타 0.5토큰/문자
    estimated_tokens = int(korean_chars * 1.5 + english_words * 1.0 + others * 0.5)
    
    return estimated_tokens

def calculate_cost(input_tokens, output_tokens):
    """토큰 수를 기반으로 Gemini API 비용을 계산합니다 (USD)."""
    input_cost = (input_tokens / 1_000_000) * GEMINI_COST_PER_1M_INPUT
    output_cost = (output_tokens / 1_000_000) * GEMINI_COST_PER_1M_OUTPUT
    return input_cost + output_cost

def find_text_until_gold_label(original_sentence, current_gold_label, next_gold_label):
    """원본 문장에서 current_gold_label 이후부터 next_gold_label 이전까지의 텍스트를 반환합니다."""
    # current_gold_label이 나오는 위치 찾기
    current_pos = original_sentence.find(current_gold_label)
    if current_pos == -1:
        return ""
    
    # current_gold_label 이후 위치
    after_current = current_pos + len(current_gold_label)
    remaining_text = original_sentence[after_current:]
    
    # next_gold_label이 나오는 위치 찾기
    next_pos = remaining_text.find(next_gold_label)
    if next_pos == -1:
        # next_gold_label을 찾을 수 없는 경우 끝까지 반환
        return remaining_text
    
    # current_gold_label 이후부터 next_gold_label 이전까지의 텍스트
    return remaining_text[:next_pos]

# =================================================================
# 응급 저장 시스템 (Emergency Save System)
# =================================================================

# 전역 변수로 저장할 결과 데이터 관리
current_results = []
emergency_save_enabled = False

def emergency_save(results=None):
    """예기치 못한 종료 시 현재까지의 결과를 저장하는 함수"""
    global current_results
    
    if results is None:
        results = current_results
    
    if not results:
        print("[EMERGENCY] 저장할 데이터가 없습니다.")
        return
    
    try:
        # 응급 저장 파일명 생성 (타임스탬프 포함)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        emergency_filename = OUTPUT_JSON_PATH.replace('.json', f'_emergency_{timestamp}.json')
        
        print(f"\n[EMERGENCY] 예기치 못한 종료 감지. 현재까지의 결과를 저장합니다...")
        print(f"[EMERGENCY] 저장 위치: {emergency_filename}")
        print(f"[EMERGENCY] 저장할 항목 수: {len(results)}")
        
        # 응급 저장 실행
        with open(emergency_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 통계 정보 출력
        total_tows = sum(r.get('total_count', 0) for r in results)
        completed_tows = sum(r.get('completed_count', 0) for r in results)
        completion_rate = (completed_tows / total_tows * 100) if total_tows > 0 else 0
        
        print(f"[EMERGENCY] 응급 저장 완료!")
        print(f"[EMERGENCY] - 저장된 항목 수: {len(results)}")
        print(f"[EMERGENCY] - 완성된 ToW: {completed_tows:,} / {total_tows:,} ({completion_rate:.1f}%)")
        print(f"[EMERGENCY] 재시작 시 이 파일을 사용하여 작업을 이어갈 수 있습니다.")
        
    except Exception as e:
        print(f"[EMERGENCY ERROR] 응급 저장 실패: {e}")
        # 최후의 수단으로 pickle로 시도
        try:
            import pickle
            pickle_filename = OUTPUT_JSON_PATH.replace('.json', f'_emergency_{timestamp}.pkl')
            with open(pickle_filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"[EMERGENCY] 백업: pickle 형태로 저장됨 - {pickle_filename}")
        except Exception as pickle_error:
            print(f"[EMERGENCY ERROR] 백업 저장도 실패: {pickle_error}")

def setup_emergency_handlers():
    """시그널 핸들러와 종료 핸들러를 설정합니다"""
    global emergency_save_enabled
    emergency_save_enabled = True
    
    # 일반적인 종료 시그널 핸들러
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        print(f"\n[EMERGENCY] 시그널 {signal_name}({signum}) 수신됨. 응급 저장을 시작합니다...")
        emergency_save()
        print(f"[EMERGENCY] 프로그램을 종료합니다.")
        exit(0)
    
    # 프로그램 종료 시 자동 저장
    def exit_handler():
        if emergency_save_enabled and current_results:
            print(f"\n[EMERGENCY] 프로그램 종료 감지. 마지막 저장을 수행합니다...")
            emergency_save()
    
    # 시그널 핸들러 등록 (Windows/Linux 호환)
    try:
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 종료 시그널
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)  # Ctrl+Break
    except Exception as e:
        print(f"[WARNING] 일부 시그널 핸들러 등록 실패: {e}")
    
    # atexit 핸들러 등록
    atexit.register(exit_handler)
    
    print("[INFO] 응급 저장 시스템이 활성화되었습니다. (Ctrl+C로 안전하게 중단 가능)")

def update_emergency_data(results):
    """현재 결과를 전역 응급 저장 데이터에 업데이트"""
    global current_results
    current_results = results[:]  # 복사본 저장

# =================================================================
# 진짜 병렬 배치 처리 함수들
# =================================================================

async def process_single_tow(model, prompt, generation_config, context_info):
    """단일 ToW 생성을 처리하는 함수"""
    try:
        response = await generate_with_backoff(model, prompt, generation_config)
        tow_content = response.text.strip()
        
        return {
            'success': True,
            'content': tow_content,
            'input_tokens': estimate_tokens(prompt),
            'output_tokens': estimate_tokens(tow_content),
            'context_info': context_info
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'context_info': context_info
        }

# --- 개선된 부분: 예외 처리 강화 ---
async def process_item_parallel(item, model, generation_config, processed_data):
    """단일 아이템의 모든 ToW를 병렬로 처리"""
    original_sentence = item['original_sentence']
    context = item['context']
    gold_labels = item['gold_label']
    item_id = item['id']
    
    # 기존 처리 결과 확인
    if item_id in processed_data:
        proc_data = processed_data[item_id]
        tows = proc_data['tows'][:]
        completed_indices = set(proc_data['completed_indices'])
    else:
        tows = [''] * len(gold_labels)
        completed_indices = set()

    # 병렬 처리할 태스크들 생성
    tasks = []
    task_info = []
    current_context = context
    
    for i, gold_label in enumerate(gold_labels):
        if i in completed_indices:
            # 이미 완료된 ToW는 context 업데이트만
            if i < len(gold_labels) - 1:
                next_gold_label = gold_labels[i + 1]
                between_text = find_text_until_gold_label(original_sentence, gold_label, next_gold_label)
                current_context = current_context + " " + tows[i] + " " + gold_label + between_text
            continue
        
        # 프롬프트 생성
        prompt = FEW_SHOT_PROMPT_TEMPLATE.format(context=current_context, gold_label=gold_label)
        
        # 태스크 생성
        context_info = {
            'item_id': item_id,
            'gold_label_index': i,
            'gold_label': gold_label,
            'current_context': current_context
        }
        
        task = process_single_tow(model, prompt, generation_config, context_info)
        tasks.append(task)
        task_info.append((i, gold_label))
        
        # 다음 context 예상 업데이트 (병렬이므로 실제로는 나중에 순서대로 업데이트)
        if i < len(gold_labels) - 1:
            next_gold_label = gold_labels[i + 1]
            between_text = find_text_until_gold_label(original_sentence, gold_label, next_gold_label)
            current_context = current_context + " [ToW_PLACEHOLDER] " + gold_label + between_text
    
    # 병렬 실행
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리 (순서대로)
        for (task_idx, (gold_idx, gold_label)), result in zip(enumerate(task_info), results):
            if isinstance(result, Exception):
                tows[gold_idx] = f"[ERROR] ToW generation failed for '{gold_label}': {str(result)}"
            elif result['success']:
                tows[gold_idx] = result['content']
                completed_indices.add(gold_idx)
            else:
                tows[gold_idx] = f"[ERROR] ToW generation failed for '{gold_label}': {result['error']}"
    
    return {
        'id': item_id,
        'original_sentence': original_sentence,
        'context': context,
        'gold_label': gold_labels,
        'tows': tows,
        'completed_count': len([t for t in tows if not t.startswith('[ERROR]')]),
        'total_count': len(gold_labels)
    }


# --- 메인 실행 로직 (비동기 배치 처리 및 주기적 저장) ---
async def generate_multiple_tow_dataset_async():
    """Gemini API를 진짜 비동기 배치로 호출하고 병렬로 다중 ToW 데이터셋을 생성합니다."""
    global api_semaphore
    
    # 응급 저장 시스템 활성화
    setup_emergency_handlers()

    # API 세마포어 초기화
    api_semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)
    print(f"[INFO] API 세마포어 초기화: 동시 요청 수 {PARALLEL_REQUESTS}")

    # Vertex AI 초기화
    print(f"[INFO] Vertex AI를 초기화합니다. (Project: {PROJECT_ID}, Location: {LOCATION})")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Gemini 모델 로드
    print(f"[INFO] Gemini 모델 '{GEMINI_MODEL_ID}'을(를) 로드합니다.")
    model = GenerativeModel(GEMINI_MODEL_ID)
    
    # 생성 설정: ToW 설명이 길 수 있으므로 max_output_tokens를 충분히 확보
    generation_config = GenerationConfig(
        temperature=0.2, # 약간의 창의성을 허용하되 일관성을 유지
        max_output_tokens=2048
    )

    # 입력 데이터 로드
    print(f"[INFO] '{INPUT_JSON_PATH}' 파일에서 데이터를 로드합니다.")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {INPUT_JSON_PATH}")
        return
    

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(OUTPUT_JSON_PATH)
    if output_dir and not os.path.exists(output_dir):
        print(f"[INFO] 출력 디렉토리 '{output_dir}'을(를) 생성합니다.")
        os.makedirs(output_dir, exist_ok=True)

    # 이미 처리된 결과를 불러와서 이어하기
    results = []
    processed_data = {}  # id -> {완료된 gold_label 인덱스들, tows 리스트}
    if os.path.exists(OUTPUT_JSON_PATH):
        print(f"[INFO] 기존 출력 파일 '{OUTPUT_JSON_PATH}'을(를) 발견했습니다. 이어서 작업을 시작합니다.")
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 부분 완성된 항목들 분석
            for item in results:
                item_id = item['id']
                tows = item.get('tows', [])
                gold_labels = item.get('gold_label', [])
                
                # 성공적으로 생성된 ToW의 인덱스 찾기
                completed_indices = []
                for i, tow in enumerate(tows):
                    if not tow.startswith('[ERROR]') and i < len(gold_labels):
                        completed_indices.append(i)
                
                processed_data[item_id] = {
                    'completed_indices': completed_indices,
                    'tows': tows[:],
                    'original_item': item
                }
            
            print(f"[INFO] {len(results)}개의 항목을 발견했습니다 (완전/부분 처리 포함).")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] 기존 출력 파일에 문제가 있습니다. 새로 시작합니다: {e}")
            results = []
            processed_data = {}
    else:
        print(f"[INFO] 새로운 출력 파일 '{OUTPUT_JSON_PATH}'을(를) 생성합니다.")

    # 처리해야 할 데이터 분석 (완전히 새로운 것 + 부분 완성된 것)
    tasks_to_run = []
    for item in data:
        item_id = item['id']
        if item_id not in processed_data:
            # 완전히 새로운 항목
            tasks_to_run.append(item)
        else:
            # 부분 완성된 항목 - 실패한 ToW가 있는지 확인
            proc_data = processed_data[item_id]
            gold_labels = item['gold_label']
            if len(proc_data['completed_indices']) < len(gold_labels):
                # 아직 완성되지 않은 ToW가 있음
                tasks_to_run.append(item)
    if not tasks_to_run:
        print("[SUCCESS] 모든 항목이 이미 처리되었습니다. 프로그램을 종료합니다.")
        return

    error_count = 0
    last_save_count = len(results)
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"[INFO] 총 {len(tasks_to_run)}개의 신규 항목을 {BATCH_SIZE}개씩 배치로 처리를 시작합니다.")

    # 진짜 병렬 배치 처리
    print(f"[INFO] 진짜 병렬 배치 처리 시작: 배치당 {BATCH_SIZE}개 항목, 동시 API 요청 {PARALLEL_REQUESTS}개")
    
    for batch_start in tqdm(range(0, len(tasks_to_run), BATCH_SIZE), desc="Processing batches in parallel"):
        batch_end = min(batch_start + BATCH_SIZE, len(tasks_to_run))
        batch_items = tasks_to_run[batch_start:batch_end]
        
        print(f"\n[BATCH] 배치 {batch_start//BATCH_SIZE + 1}: {len(batch_items)}개 항목 병렬 처리 중...")
        
        # 배치 내 모든 아이템을 병렬로 처리
        batch_tasks = []
        for item in batch_items:
            task = process_item_parallel(item, model, generation_config, processed_data)
            batch_tasks.append(task)
        
        try:
            # 배치 내 모든 아이템을 병렬 실행
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 결과 처리
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"[ERROR] 배치 항목 {i} 처리 실패: {result}")
                    error_count += 1
                    continue
                
                item_id = result['id']
                
                # 토큰 카운팅 (간단화)
                for tow in result['tows']:
                    if not tow.startswith('[ERROR]'):
                        total_output_tokens += estimate_tokens(tow)
                        total_input_tokens += 1000  # 추정값
                
                # 기존 results에서 같은 id 항목 제거하고 새로운 결과 추가
                results = [r for r in results if r['id'] != item_id]
                results.append(result)
                
                # 진행 상황 출력
                session_cost = calculate_cost(total_input_tokens, total_output_tokens)
                print(f"[PARALLEL COMPLETED] ID {item_id}: {result['completed_count']}/{result['total_count']} ToW 완료")
            
            # 응급 저장용 데이터 업데이트
            update_emergency_data(results)
            
            # 배치 완료 후 누적 정보 표시
            session_cost = calculate_cost(total_input_tokens, total_output_tokens)
            print(f"[BATCH COMPLETED] 배치 {batch_start//BATCH_SIZE + 1}: {len(batch_items)}개 항목 병렬 처리 완료")
            print(f"                  세션 누적: {total_input_tokens:,} 입력, {total_output_tokens:,} 출력 토큰")
            print(f"                  예상 비용: ${session_cost:.4f} USD")
            
        except Exception as e:
            print(f"[ERROR] 배치 {batch_start//BATCH_SIZE + 1} 처리 실패: {e}")
            error_count += len(batch_items)
        
        if len(results) - last_save_count >= SAVE_INTERVAL:
            print(f"\n[INFO] 중간 저장: {len(results)}개의 누적 결과를 파일에 저장합니다.")
            try:
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                last_save_count = len(results)
                print(f"[INFO] 중간 저장 완료: {OUTPUT_JSON_PATH}")
            except Exception as e:
                print(f"[ERROR] 중간 저장 실패: {e}")
        
        # 배치 간 쿨다운
        print(f"[BATCH] 배치 간 쿨다운...")
        await asyncio.sleep(random.uniform(0.5, 1.5))  # 병렬 처리로 더 짧은 대기

    # 모든 배치가 끝난 후 최종 저장
    print(f"\n[SUCCESS] 다중 ToW 데이터셋 생성이 완료되었습니다.")
    print(f"  - 처리된 항목 수: {len(tasks_to_run)}")
    print(f"  - 실패한 개별 ToW 개수: {error_count}")
    print(f"  - 총 저장된 항목 수: {len(results)}")
    print(f"  - 배치 크기: {BATCH_SIZE}개, 총 배치 수: {(len(tasks_to_run) + BATCH_SIZE - 1) // BATCH_SIZE}개")
    
    # 완성도 통계
    total_tows = sum(r['total_count'] for r in results)
    completed_tows = sum(r['completed_count'] for r in results)
    completion_rate = (completed_tows / total_tows * 100) if total_tows > 0 else 0
    
    print(f"\n[COMPLETION STATISTICS]")
    print(f"  - 전체 ToW 목표 개수: {total_tows:,} 개")
    print(f"  - 성공적으로 생성된 ToW: {completed_tows:,} 개")
    print(f"  - ToW 생성 성공률: {completion_rate:.1f}%")
    
    # 토큰 사용량 및 비용 통계
    final_cost = calculate_cost(total_input_tokens, total_output_tokens)
    print(f"\n[TOKEN USAGE & COST STATISTICS]")
    print(f"  - 총 입력 토큰 수: {total_input_tokens:,} 토큰")
    print(f"  - 총 출력 토큰 수: {total_output_tokens:,} 토큰")
    print(f"  - 전체 토큰 사용량: {total_input_tokens + total_output_tokens:,} 토큰")
    print(f"  - 예상 총 비용: ${final_cost:.4f} USD")
    if len(tasks_to_run) > 0:
        avg_input_per_item = total_input_tokens / len(tasks_to_run)
        avg_output_per_item = total_output_tokens / len(tasks_to_run)
        avg_cost_per_item = final_cost / len(tasks_to_run)
        print(f"  - 항목당 평균 입력 토큰: {avg_input_per_item:.1f} 토큰")
        print(f"  - 항목당 평균 출력 토큰: {avg_output_per_item:.1f} 토큰")
        print(f"  - 항목당 평균 비용: ${avg_cost_per_item:.4f} USD")
    
    print(f"\n[SAVE] '{OUTPUT_JSON_PATH}' 파일에 최종 결과를 저장합니다.")
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] 최종 저장 완료: {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"[ERROR] 최종 저장 실패: {e}")
        # 백업 파일로 저장 시도
        backup_path = OUTPUT_JSON_PATH.replace('.json', '_backup.json')
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 백업 파일로 저장: {backup_path}")
        except Exception as backup_e:
            print(f"[ERROR] 백업 저장도 실패: {backup_e}")
    
    # 정상 완료 시 응급 저장 시스템 비활성화
    global emergency_save_enabled
    emergency_save_enabled = False
    print(f"[INFO] 작업이 정상적으로 완료되어 응급 저장 시스템을 비활성화합니다.")


if __name__ == "__main__":
    # 스크립트 실행 전, 터미널에서 GCP 인증이 필요합니다:
    # gcloud auth application-default login
    
    # asyncio.run()을 사용하여 비동기 함수 실행
    try:
        asyncio.run(generate_multiple_tow_dataset_async())
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 예기치 못한 오류로 프로그램이 종료됩니다: {e}")
        emergency_save()
        print(f"[CRITICAL ERROR] 응급 저장 완료. 프로그램을 종료합니다.")
        raise