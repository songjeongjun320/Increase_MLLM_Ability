import json
import time
import os
import traceback # traceback 모듈 임포트
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import timedelta # 시간 표시를 위해 추가

MODEL_NAME = "google/gemma-3-12b-it" # VRAM 고려하여 모델 크기 조절
# --- 모델 및 토크나이저 로드 ---
print(f"Loading model, processor, and tokenizer for {MODEL_NAME}...")
model = None
tokenizer = None
model_device = None
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if hasattr(model, 'device'):
        model_device = model.device
    else:
        model_device = next(model.parameters()).device
    print(f"Model loaded successfully on device: {model_device}")
    print(f"Tokenizer loaded: {type(tokenizer)}")
except Exception as e:
    print(f"Error loading model or tokenizer {MODEL_NAME}: {e}")
    traceback.print_exc()
    exit()

# --- 번역 관련 상수 ---
MAX_RETRIES_ON_ERROR = 3
WAIT_TIME_ON_ERROR = 10


def translate_text_gemma(text_to_translate, target_language="Korean"):
    """Gemma 모델을 사용하여 텍스트를 번역합니다."""
    if not text_to_translate or not text_to_translate.strip():
        return ""

    messages = [
        {"role": "user", "content": f"Translate this English sentence to Korean sentence. Just give translation result.: \"{text_to_translate}\""}
    ]
    for retry in range(MAX_RETRIES_ON_ERROR):
        try:
            inputs_dict = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            if isinstance(inputs_dict, torch.Tensor):
                 inputs_on_device = {"input_ids": inputs_dict.to(model_device)}
                 input_len = inputs_dict.shape[1]
            elif isinstance(inputs_dict, dict):
                 inputs_on_device = {k: v.to(model_device) for k, v in inputs_dict.items()}
                 input_len = inputs_on_device["input_ids"].shape[1]
            else:
                 raise ValueError(f"Unexpected output type: {type(inputs_dict)}")

            outputs = model.generate(
                **inputs_on_device, max_new_tokens=250, num_beams=1,
                do_sample=False, temperature=0.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            if outputs is None or not isinstance(outputs, torch.Tensor) or outputs.ndim == 0 or outputs.shape[0] != 1:
                 raise ValueError(f"Invalid model.generate output.")

            generated_tokens = outputs[0, input_len:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            translated_text = decoded_output.strip()

            if not translated_text:
                 return "[Translation Warning: Empty output]"
            
            if "Korean sentence:" in translated_text:
                translated_text = translated_text.split("Korean sentence:", 1)[-1].strip()
            if "한국어 문장:" in translated_text:
                translated_text = translated_text.split("한국어 문장:", 1)[-1].strip()
            if translated_text.startswith("\"") and translated_text.endswith("\""):
                translated_text = translated_text[1:-1]
            return translated_text
        except Exception as e:
            # print(f"Error translating (attempt {retry+1}): {e}") # 로그 간소화
            if retry < MAX_RETRIES_ON_ERROR - 1:
                time.sleep(WAIT_TIME_ON_ERROR)
                continue
            else:
                return f"[Translation Error: Failed after {MAX_RETRIES_ON_ERROR} retries]"
    return "[Translation Error: Unknown failure]"


def format_time(seconds):
    """초를 HH:MM:SS 형태로 변환"""
    if seconds < 0: return "N/A"
    return str(timedelta(seconds=int(seconds)))


def process_translation_for_file(input_filename_sentences: str):
    base_name = os.path.splitext(input_filename_sentences)[0]
    output_filename_translated = f"{base_name}_translated_ko.json"
    progress_filename = f"{base_name}_translation_progress.json"

    print(f"\n--- Starting translation for: {input_filename_sentences} ---")
    # ... (기존 파일 로드 및 진행 상황 로드 로직은 동일) ...
    try:
        with open(input_filename_sentences, 'r', encoding='utf-8') as f:
            original_sentences = json.load(f)
            if not isinstance(original_sentences, list):
                print(f"Error: Input file '{input_filename_sentences}' does not contain a list of sentences.")
                return
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename_sentences}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filename_sentences}'.")
        return
    except Exception as e:
         print(f"An unexpected error occurred while reading input file '{input_filename_sentences}': {e}")
         traceback.print_exc()
         return

    translated_sentences_list = []
    start_sentence_index = 0

    if os.path.exists(progress_filename):
        try:
            with open(progress_filename, 'r', encoding='utf-8') as pf:
                progress = json.load(pf)
                translated_sentences_list = progress.get("translated_sentences", [])
                start_sentence_index = progress.get("last_sentence_index", 0)
                if not isinstance(translated_sentences_list, list): # 타입 체크
                    print(f"Warning: Corrupted progress for '{progress_filename}'. Resetting.")
                    translated_sentences_list = []
                    start_sentence_index = 0
                elif len(translated_sentences_list) != start_sentence_index :
                    print(f"Warning: Mismatch in progress for '{progress_filename}'. Adjusting. Index: {start_sentence_index}, Loaded: {len(translated_sentences_list)}.")
                    translated_sentences_list = translated_sentences_list[:start_sentence_index]

                if start_sentence_index < len(original_sentences):
                     print(f"Resuming translation for '{input_filename_sentences}' from sentence index {start_sentence_index}")
                elif len(translated_sentences_list) >= len(original_sentences) and len(original_sentences) > 0 : # and len(original_sentences) > 0 추가
                     print(f"Progress file indicates all sentences in '{input_filename_sentences}' were translated. Finalizing.")
                else:
                     start_sentence_index = len(translated_sentences_list) # 안전장치

        except Exception as e:
            print(f"Could not load progress file '{progress_filename}', starting from scratch. Error: {e}")
            translated_sentences_list = []
            start_sentence_index = 0
    # --- 예상 시간 계산 로직 추가 ---
    total_sentences = len(original_sentences)
    s_idx = start_sentence_index
    
    file_processing_start_time = time.time() # 파일 전체 처리 시작 시간
    sentences_processed_since_last_eta_calc = 0
    time_spent_since_last_eta_calc = 0
    
    # 초기 ETA 계산을 위한 샘플 수
    INITIAL_ETA_SAMPLE_COUNT = min(10, total_sentences - s_idx if total_sentences - s_idx > 0 else 1)


    if s_idx >= total_sentences and total_sentences > 0:
        print(f"All {total_sentences} sentences in '{input_filename_sentences}' appear to be processed. Skipping translation loop.")
    elif total_sentences == 0:
        print(f"Input file '{input_filename_sentences}' contains no sentences.")
    else:
        print(f"Translating {total_sentences - s_idx} sentences for '{input_filename_sentences}'. Starting from index {s_idx}/{total_sentences}.")
        
        temp_sentence_times = [] # 초기 ETA 계산용 시간 저장

        while s_idx < total_sentences:
            sentence_start_time = time.time() # 개별 문장 처리 시작 시간
            sentence = original_sentences[s_idx]
            # 로그에 예상 시간 추가 안 함 (너무 자주 바뀜)
            # print(f"  Translating sentence {s_idx + 1}/{total_sentences} from '{input_filename_sentences}': \"{str(sentence)[:60]}...\"")

            translated_sentence = translate_text_gemma(sentence)
            
            sentence_end_time = time.time()
            time_taken_for_sentence = sentence_end_time - sentence_start_time
            
            # 초기 ETA 계산용 데이터 수집
            if s_idx < start_sentence_index + INITIAL_ETA_SAMPLE_COUNT:
                 temp_sentence_times.append(time_taken_for_sentence)

            # translated_sentences_list 크기 s_idx에 맞게 확장
            while len(translated_sentences_list) <= s_idx:
                 translated_sentences_list.append(None)
            translated_sentences_list[s_idx] = translated_sentence

            sentences_processed_since_last_eta_calc += 1
            time_spent_since_last_eta_calc += time_taken_for_sentence

            # 10문장마다 또는 특정 조건에서 진행 상황 저장 및 ETA 업데이트
            if (s_idx + 1) % 10 == 0 or (s_idx + 1) == total_sentences or \
               (isinstance(translated_sentence, str) and translated_sentence.startswith("[Translation Error:")):
                
                progress_to_save = {
                    "translated_sentences": translated_sentences_list,
                    "last_sentence_index": s_idx + 1
                }
                try:
                    with open(progress_filename, 'w', encoding='utf-8') as pf:
                        json.dump(progress_to_save, pf, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Error saving progress file '{progress_filename}': {e}")

                # ETA 계산 및 로깅
                sentences_remaining = total_sentences - (s_idx + 1)
                estimated_time_remaining_str = "Calculating..."
                
                if sentences_processed_since_last_eta_calc > 0:
                    avg_time_per_sentence_current_batch = time_spent_since_last_eta_calc / sentences_processed_since_last_eta_calc
                    if avg_time_per_sentence_current_batch > 0: # 0으로 나누기 방지
                        estimated_time_remaining_seconds = sentences_remaining * avg_time_per_sentence_current_batch
                        estimated_time_remaining_str = format_time(estimated_time_remaining_seconds)
                elif temp_sentence_times: # 초기 샘플 기반 ETA
                    avg_time_initial_samples = sum(temp_sentence_times) / len(temp_sentence_times)
                    if avg_time_initial_samples > 0:
                        estimated_time_remaining_seconds = sentences_remaining * avg_time_initial_samples
                        estimated_time_remaining_str = format_time(estimated_time_remaining_seconds)


                if (s_idx + 1) < total_sentences:
                    print(f"  Processed {s_idx + 1}/{total_sentences} for '{input_filename_sentences}'. Progress saved. ETA: {estimated_time_remaining_str}")
                    # ETA 계산을 위한 변수 리셋 (다음 10개 배치에 대해 새로 계산)
                    # sentences_processed_since_last_eta_calc = 0
                    # time_spent_since_last_eta_calc = 0
                    # time.sleep(1) # 짧은 대기 제거 또는 조절
                elif (s_idx + 1) == total_sentences:
                     print(f"  Processed {s_idx + 1}/{total_sentences} for '{input_filename_sentences}'. All sentences in this file processed.")


            s_idx += 1
    
    file_processing_end_time = time.time()
    total_time_for_file = file_processing_end_time - file_processing_start_time
    print(f"\nFinished processing '{input_filename_sentences}'. Total time taken: {format_time(total_time_for_file)}")
    print(f"Saving final translated sentences to '{output_filename_translated}'...")

    # ... (기존 최종 저장 및 progress 파일 삭제 로직은 동일) ...
    try:
        with open(output_filename_translated, 'w', encoding='utf-8') as f:
            json.dump(translated_sentences_list, f, ensure_ascii=False, indent=2)

        if os.path.exists(progress_filename) and len(translated_sentences_list) >= total_sentences and total_sentences > 0:
            print(f"All translations for '{input_filename_sentences}' completed. Removing progress file '{progress_filename}'.")
            try:
                os.remove(progress_filename)
                print("Progress file removed.")
            except OSError as e:
                print(f"Error removing progress file '{progress_filename}': {e}")
        elif not (len(translated_sentences_list) >= total_sentences and total_sentences > 0) and total_sentences > 0:
             print(f"Warning: Translation for '{input_filename_sentences}' might be incomplete. Progress file '{progress_filename}' retained.")
        print(f"Successfully saved translated data for '{input_filename_sentences}' to '{output_filename_translated}'.")
    except Exception as e:
        print(f"Error saving final output file '{output_filename_translated}': {e}")
        traceback.print_exc()


def main():
    input_files_to_translate = [
        "openwebmath_sentences_from_first_3000_docs.json",
    ]
    if model is None or tokenizer is None or model_device is None:
        print("Model, tokenizer, or device not loaded. Exiting.")
        return
    for input_file in input_files_to_translate:
        if not os.path.exists(input_file):
            print(f"Input file '{input_file}' not found. Skipping.")
            continue
        process_translation_for_file(input_file)
    print("\nAll specified files have been processed.")

if __name__ == "__main__":
    main()