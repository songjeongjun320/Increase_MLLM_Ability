import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
import os
import traceback

# --- 설정 ---
INPUT_JSON_PATH = "/scratch/jsong132/Increase_MLLM_Ability/DB/C4/c4_200_kr.json"
OUTPUT_JSON_PATH = "c4_context_detailed_log.json" # 출력 파일명 변경
JSON_KEY = "c4_first_200_sentences"
MODEL_NAME = "google/gemma-3-12b-it" # 실제 사용하는 모델명으로
MIN_PREVIOUS_WORDS_FOR_CANDIDATE = 1

# --- 모델 및 토크나이저 로드 ---
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

print(f"Loading model: {MODEL_NAME}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
except Exception as e:
    print(f"Model loading failed with device_map='auto' and specified dtype: {e}")
    print("Trying to load with default settings or on CPU...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    except Exception as e2:
        print(f"Model loading with device_map='auto' also failed: {e2}")
        print("Loading model on CPU explicitly.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()
print(f"Model loaded. Main device (can be misleading with device_map): {model.device if hasattr(model, 'device') else 'N/A'}")
if hasattr(model, 'hf_device_map'):
    print(f"Model device map: {model.hf_device_map}")


def load_sentences_from_json(filepath, key):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(key, [])
    except FileNotFoundError:
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        print(f"오류: {filepath} 파일이 올바른 JSON 형식이 아닙니다.")
        return []
    except Exception as e:
        print(f"파일 로드 중 오류 발생 ({filepath}): {e}")
        return []

def get_word_boundary_from_token_indices(sentence_text, token_indices_in_word, all_offsets_for_sentence):
    if not token_indices_in_word: # 단어를 구성하는 토큰 인덱스가 없으면 빈 문자열 반환
        return 0, 0, ""

    valid_offsets_for_word = [all_offsets_for_sentence[i] for i in token_indices_in_word
                              if i < len(all_offsets_for_sentence) and \
                                 all_offsets_for_sentence[i] is not None and \
                                 all_offsets_for_sentence[i] != (0,0)]
    
    if not valid_offsets_for_word:
        # (0,0) 이거나 None인 오프셋만 있는 경우, 첫번째 토큰의 오프셋이라도 사용 시도 (주로 특수 토큰)
        first_token_idx = token_indices_in_word[0]
        if first_token_idx < len(all_offsets_for_sentence):
            offset = all_offsets_for_sentence[first_token_idx]
            if offset is not None:
                start_char, end_char = offset
                # sentence_text 범위를 벗어나는 경우 방지
                end_char = min(end_char, len(sentence_text))
                start_char = min(start_char, end_char)
                return start_char, end_char, sentence_text[start_char:end_char]
        return 0, 0, ""
    
    word_start_char = min(off[0] for off in valid_offsets_for_word)
    word_end_char = max(off[1] for off in valid_offsets_for_word)
    
    # sentence_text 범위를 벗어나는 경우 방지
    word_end_char = min(word_end_char, len(sentence_text))
    word_start_char = min(word_start_char, word_end_char)

    return word_start_char, word_end_char, sentence_text[word_start_char:word_end_char]


def calculate_surprisal_at_token(model, input_ids_prefix, actual_next_token_id):
    """
    주어진 prefix 뒤에 actual_next_token_id가 나올 때의 (surprisal, probability)를 계산.
    """
    if input_ids_prefix.size(1) == 0:
        return float('inf'), 0.0

    target_device = next(model.parameters()).device
    input_ids_prefix = input_ids_prefix.to(target_device)

    with torch.no_grad():
        outputs = model(input_ids_prefix)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        probabilities_all = torch.softmax(next_token_logits, dim=-1)
        
        if not (0 <= actual_next_token_id < probabilities_all.shape[1]):
            # print(f"경고: actual_next_token_id ({actual_next_token_id})가 어휘 크기 ({probabilities_all.shape[1]})를 벗어납니다. Prefix: {tokenizer.decode(input_ids_prefix.squeeze().tolist())}")
            return float('inf'), 0.0
            
        actual_next_token_prob = probabilities_all[0, actual_next_token_id].item()
        
        surprisal = -math.log2(actual_next_token_prob) if actual_next_token_prob > 1e-9 else float('inf')
        
    return surprisal, actual_next_token_prob

def get_context_by_min_surprisal(
    sentence_idx, sentence, model, tokenizer, min_prev_words_for_candidate=1
):
    original_sentence_stripped = sentence.strip()
    print(f"\n--- 문장 {sentence_idx + 1} 처리 시작: '{original_sentence_stripped[:50]}...' ---")

    if not original_sentence_stripped:
        print("  [정보] 빈 문장이므로 스킵합니다.")
        return "", "", "", None # org, context, gold_next, stats

    try:
        sentence_to_tokenize = original_sentence_stripped
        if tokenizer.bos_token:
            if not original_sentence_stripped.startswith(tokenizer.bos_token):
                sentence_to_tokenize = tokenizer.bos_token + original_sentence_stripped
        
        inputs_with_offset = tokenizer(
            sentence_to_tokenize,
            return_tensors='pt',
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        
        all_token_ids_tensor = inputs_with_offset['input_ids']
        all_token_ids_list = all_token_ids_tensor.squeeze().tolist()
        offsets_in_tokenized_sentence = inputs_with_offset['offset_mapping'].squeeze().tolist()
        word_indices_in_tokenized_sentence = inputs_with_offset.word_ids(batch_index=0)

        if not isinstance(all_token_ids_list, list):
            all_token_ids_list = [all_token_ids_list]
            offsets_in_tokenized_sentence = [offsets_in_tokenized_sentence]
            word_indices_in_tokenized_sentence = [word_indices_in_tokenized_sentence]
        
        # 디버깅: 토큰화 결과 출력
        # print(f"  [디버그] 토큰화된 문장 ({len(all_token_ids_list)} 토큰): '{sentence_to_tokenize}'")
        # for tok_idx, tok_id in enumerate(all_token_ids_list):
        #     tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        #     offset = offsets_in_tokenized_sentence[tok_idx]
        #     word_id = word_indices_in_tokenized_sentence[tok_idx]
        #     print(f"    토큰 {tok_idx}: ID={tok_id}, Str='{tok_str}', Offset={offset}, WordID={word_id}")


        min_tokens_required = 2
        if tokenizer.bos_token_id is not None and all_token_ids_list and \
           all_token_ids_list[0] == tokenizer.bos_token_id:
            min_tokens_required = 3

        if len(all_token_ids_list) < min_tokens_required:
            print(f"  [정보] 문장이 너무 짧아 (토큰 수 {len(all_token_ids_list)} < {min_tokens_required}) surprisal 기반 분리 불가.")
            return original_sentence_stripped, original_sentence_stripped, "", {"status": "too_short", "reason": f"Tokens < {min_tokens_required}"}

        candidate_points = []
        print("  [정보] 후보 분리 지점 탐색 중...")

        first_content_token_index = 0
        if tokenizer.bos_token_id is not None and all_token_ids_list and \
           all_token_ids_list[0] == tokenizer.bos_token_id:
            first_content_token_index = 1
        
        for i in range(first_content_token_index, len(all_token_ids_list) - 1):
            current_prefix_ids_tensor = all_token_ids_tensor[:, :i+1]
            actual_next_token_id = all_token_ids_list[i+1]
            
            # 디코딩 시 에러 방지 위해 리스트로 변환
            prefix_token_ids_list = current_prefix_ids_tensor.squeeze().tolist()
            if not isinstance(prefix_token_ids_list, list): # 단일 토큰 prefix인 경우
                prefix_token_ids_list = [prefix_token_ids_list]

            prefix_str = tokenizer.decode(prefix_token_ids_list, skip_special_tokens=False)
            next_token_str_decoded = tokenizer.decode([actual_next_token_id], skip_special_tokens=False)

            surprisal, prob = calculate_surprisal_at_token(
                model, current_prefix_ids_tensor, actual_next_token_id
            )
            
            word_id_of_next_token = word_indices_in_tokenized_sentence[i+1]
            word_id_of_current_token = word_indices_in_tokenized_sentence[i]

            is_actually_new_word_start = \
                (word_id_of_next_token is not None) and \
                (word_id_of_current_token is None or word_id_of_next_token != word_id_of_current_token)

            if is_actually_new_word_start:
                unique_preceding_word_ids = set()
                for k in range(i + 1):
                    wid = word_indices_in_tokenized_sentence[k]
                    if wid is not None:
                        unique_preceding_word_ids.add(wid)
                
                num_prev_words_in_prefix = len(unique_preceding_word_ids)

                if num_prev_words_in_prefix >= min_prev_words_for_candidate:
                    ppl_token = math.pow(2, surprisal) if surprisal != float('inf') else float('inf')
                    print(f"    [후보] Prefix='{prefix_str}', NextTok='{next_token_str_decoded}' (ID:{actual_next_token_id}), "
                          f"Surp={surprisal:.3f}, Prob={prob:.4e}, PPL={ppl_token:.2f}, PrevWords={num_prev_words_in_prefix}")
                    candidate_points.append({
                        "surprisal": surprisal, 
                        "probability": prob,
                        "ppl_token": ppl_token,
                        "next_token_idx": i + 1, 
                        "num_prev_words": num_prev_words_in_prefix,
                        "prefix_str": prefix_str,
                        "next_token_str": next_token_str_decoded
                    })
        
        if not candidate_points:
            print(f"  [정보] 적합한 후보 지점을 찾지 못함 (min_prev_words={min_prev_words_for_candidate}).")
            return original_sentence_stripped, original_sentence_stripped, "", {"status": "no_candidates_found"}

        # "가장 예측하기 어려운" (surprisal이 가장 높은) 다음 단어를 선택
        candidate_points.sort(key=lambda x: x["surprisal"], reverse=True)
        
        best_candidate = candidate_points[0]
        best_surprisal = best_candidate["surprisal"]
        best_prob = best_candidate["probability"]
        best_ppl_token = best_candidate["ppl_token"]
        best_next_token_idx = best_candidate["next_token_idx"]
        
        print(f"  [선택됨] 가장 높은 Surprisal 후보:")
        print(f"    Prefix='{best_candidate['prefix_str']}', NextTok='{best_candidate['next_token_str']}'")
        print(f"    Surprisal={best_surprisal:.3f}, Prob={best_prob:.4e}, PPL (token)={best_ppl_token:.2f}, "
              f"Index={best_next_token_idx}, PrevWords={best_candidate['num_prev_words']}")

        target_word_id = word_indices_in_tokenized_sentence[best_next_token_idx]
        
        tokens_indices_for_target_word = [
            k for k, wid in enumerate(word_indices_in_tokenized_sentence) if wid == target_word_id
        ]

        word_start_char_in_tokenized_s, word_end_char_in_tokenized_s, gold_next_word_str = \
            get_word_boundary_from_token_indices(
                sentence_to_tokenize, tokens_indices_for_target_word, offsets_in_tokenized_sentence
            )

        bos_len = 0
        if tokenizer.bos_token and sentence_to_tokenize.startswith(tokenizer.bos_token):
            bos_len = len(tokenizer.bos_token)

        context_end_char_in_original = max(0, word_start_char_in_tokenized_s - bos_len)
        context_str = original_sentence_stripped[:context_end_char_in_original].strip()
        gold_next_s = gold_next_word_str.strip()

        if not context_str and gold_next_s == original_sentence_stripped and min_prev_words_for_candidate > 0 :
             print(f"  [주의] Context가 비었고 gold_next_word가 원본 문장과 동일. Fallback.")
             return original_sentence_stripped, original_sentence_stripped, "", {"status": "context_empty_fallback", "surprisal": best_surprisal}
        
        print(f"  [결과] Context: '{context_str}'")
        print(f"  [결과] Gold Next Word: '{gold_next_s}' (첫 토큰 Surprisal: {best_surprisal:.3f})")

        stats = {
            "status": "success",
            "best_surprisal": best_surprisal,
            "best_probability": best_prob,
            "best_ppl_token": best_ppl_token,
            "num_candidates": len(candidate_points),
            "selected_candidate_prefix": best_candidate['prefix_str'],
            "selected_candidate_next_token": best_candidate['next_token_str'],
        }
        return original_sentence_stripped, context_str, gold_next_s, stats

    except Exception as e:
        print(f"오류 발생 (get_context_by_min_surprisal): {original_sentence_stripped[:50]}..., 오류: {e}")
        traceback.print_exc()
        return original_sentence_stripped, original_sentence_stripped, "", {"status": "exception", "error_message": str(e)}

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    source_sentences = load_sentences_from_json(INPUT_JSON_PATH, JSON_KEY)

    if not source_sentences:
        print("처리할 문장이 없습니다. 프로그램을 종료합니다.")
        exit()

    results_with_stats = [] # 통계 정보 포함할 리스트
    print(f"\n총 {len(source_sentences)}개의 문장 처리 시작 (Gemma Surprisal 기반, 후보 최소 이전 단어 수: {MIN_PREVIOUS_WORDS_FOR_CANDIDATE})...")

    for i, org_sentence in enumerate(source_sentences):
        if (i + 1) % 1 == 0: # 로그가 많아졌으므로, 처리 중 표시는 함수 내부로 옮겨도 됨.
             print(f"\n===== 문장 {i+1}/{len(source_sentences)} 진행 =====")


        if not org_sentence or not org_sentence.strip():
            print(f"경고: 빈 문장 스킵 (인덱스 {i})")
            results_with_stats.append({
                "org_context": org_sentence, "context": "", "gold_next_word": "",
                "stats": {"status": "empty_input"}
            })
            continue

        processed_org_s, context_s, gold_next_s, stats = get_context_by_min_surprisal(
            i, # 문장 인덱스 전달
            org_sentence,
            model,
            tokenizer,
            min_prev_words_for_candidate=MIN_PREVIOUS_WORDS_FOR_CANDIDATE
        )
            
        results_with_stats.append({
            "org_context": processed_org_s,
            "context": context_s,
            "gold_next_word": gold_next_s,
            "stats": stats # 통계 정보 추가
        })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as outfile:
        json.dump(results_with_stats, outfile, ensure_ascii=False, indent=4)

    print(f"\n처리가 완료되었습니다. 결과가 {OUTPUT_JSON_PATH} 에 저장되었습니다.")
    
    print("\n결과 샘플 (처음 2개, 통계 포함):")
    for i_res in range(min(2, len(results_with_stats))): # 샘플 수 줄임
        print(json.dumps(results_with_stats[i_res], ensure_ascii=False, indent=2))
    
    # 통계 요약
    successful_processing = sum(1 for res in results_with_stats if res.get("stats", {}).get("status") == "success")
    no_candidates_found = sum(1 for res in results_with_stats if res.get("stats", {}).get("status") == "no_candidates_found")
    too_short_count = sum(1 for res in results_with_stats if res.get("stats", {}).get("status") == "too_short")
    exception_count = sum(1 for res in results_with_stats if res.get("stats", {}).get("status") == "exception")
    
    print(f"\n--- 처리 요약 ---")
    print(f"총 문장 수: {len(source_sentences)}")
    print(f"성공적으로 처리됨 (context/gold_next_word 분리): {successful_processing}")
    print(f"적합 후보 없음: {no_candidates_found}")
    print(f"문장 너무 짧음: {too_short_count}")
    print(f"예외 발생: {exception_count}")

    empty_context_count = sum(1 for res in results_with_stats if not res["context"].strip() and res["gold_next_word"].strip())
    print(f"Context가 비고 gold_next_word가 있는 경우 (주의 필요): {empty_context_count}")
    