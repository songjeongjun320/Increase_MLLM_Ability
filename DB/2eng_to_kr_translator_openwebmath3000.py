import json
import time
import os
import traceback # traceback 모듈 임포트
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch 

MODEL_NAME = "google/gemma-3-12b-it" 
# --- 모델 및 토크나이저 로드 ---
print(f"Loading model, processor, and tokenizer for {MODEL_NAME}...")
model = None
tokenizer = None
model_device = None
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True) # Qwen 등 일부 모델은 trust_remote_code 필요
    tokenizer = processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Gemma는 bfloat16 잘 지원
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
    print("Please ensure you have enough VRAM (GPU memory) and disk space.")
    traceback.print_exc()
    exit()

# --- 번역 관련 상수 ---
MAX_RETRIES_ON_ERROR = 3
WAIT_TIME_ON_ERROR = 10


def translate_text_gemma(text_to_translate, target_language="Korean"):
    """Gemma 모델을 사용하여 텍스트를 번역합니다."""
    if not text_to_translate or not text_to_translate.strip():
        return "" # 빈 텍스트는 빈 텍스트 반환

    messages = [
        {"role": "user", "content": f"Translate this English sentence to Korean sentence: \"{text_to_translate}\""}
    ]
    
    prompt_for_debug = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for retry in range(MAX_RETRIES_ON_ERROR):
        try:
            inputs_dict = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            if isinstance(inputs_dict, torch.Tensor):
                 inputs_on_device = {"input_ids": inputs_dict.to(model_device)}
                 input_len = inputs_dict.shape[1]
            elif isinstance(inputs_dict, dict):
                 inputs_on_device = {k: v.to(model_device) for k, v in inputs_dict.items()}
                 input_len = inputs_on_device["input_ids"].shape[1]
            else:
                 raise ValueError(f"Unexpected output type from tokenizer.apply_chat_template: {type(inputs_dict)}")

            outputs = model.generate(
                **inputs_on_device,
                max_new_tokens=250, # 번역 결과가 길어질 수 있으므로 약간 늘림
                num_beams=1, # 더 나은 품질을 위해 2~4 정도로 설정 가능 (속도 저하)
                do_sample=False,
                temperature=0.1, # 번역에는 낮은 온도가 좋음
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

            if outputs is None or not isinstance(outputs, torch.Tensor) or outputs.ndim == 0 or outputs.shape[0] != 1:
                 print(f"Warning: model.generate returned unexpected output for text: '{text_to_translate[:50]}...'")
                 # ... (기존 오류 로깅 유지)
                 raise ValueError(f"Invalid model.generate output.")

            generated_tokens = outputs[0, input_len:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            translated_text = decoded_output.strip()

            if not translated_text:
                 print(f"Warning: Translation resulted in empty string for text: '{text_to_translate[:50]}...'")
                 # ... (기존 오류 로깅 유지)
                 return "[Translation Warning: Empty output]"
            
            # 추가: Gemma 모델이 가끔 프롬프트를 반복하거나 이상한 접두사를 붙이는 경우가 있어,
            # 간단한 후처리로 "Korean sentence:" 와 같은 부분을 제거할 수 있습니다.
            # 예시: 만약 "Korean sentence: 안녕하세요." 와 같이 나오면 "안녕하세요."만 남김
            if "Korean sentence:" in translated_text:
                translated_text = translated_text.split("Korean sentence:", 1)[-1].strip()
            if "한국어 문장:" in translated_text: # 한국어로 나올 경우도 대비
                translated_text = translated_text.split("한국어 문장:", 1)[-1].strip()
            if translated_text.startswith("\"") and translated_text.endswith("\""):
                translated_text = translated_text[1:-1]


            return translated_text

        except Exception as e:
            print(f"Error during translation (attempt {retry+1}/{MAX_RETRIES_ON_ERROR}) for text: '{text_to_translate[:50]}...': {e}")
            # traceback.print_exc() # 너무 많은 트레이스백 방지, 필요시 활성화
            if retry < MAX_RETRIES_ON_ERROR - 1:
                print(f"Retrying in {WAIT_TIME_ON_ERROR} seconds...")
                time.sleep(WAIT_TIME_ON_ERROR)
                continue
            else:
                return f"[Translation Error: Failed after {MAX_RETRIES_ON_ERROR} retries]" # 오류 메시지 간소화

    return "[Translation Error: Unknown failure outside retry loop]"


def process_translation_for_file(input_filename_sentences: str):
    """ 단일 입력 파일 (문장 리스트 포함)을 번역하고 결과를 저장합니다. """
    base_name = os.path.splitext(input_filename_sentences)[0]
    output_filename_translated = f"{base_name}_translated_ko.json"
    progress_filename = f"{base_name}_translation_progress.json"

    print(f"\n--- Starting translation for: {input_filename_sentences} ---")
    print(f"Output will be saved to: {output_filename_translated}")
    print(f"Progress will be saved to: {progress_filename}")

    try:
        with open(input_filename_sentences, 'r', encoding='utf-8') as f:
            # 입력 파일은 문장 리스트를 직접 포함한다고 가정
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

    translated_sentences_list = [] # 번역된 문장들을 저장할 리스트
    start_sentence_index = 0

    if os.path.exists(progress_filename):
        try:
            with open(progress_filename, 'r', encoding='utf-8') as pf:
                progress = json.load(pf)
                # 진행 상황 파일은 번역된 문장 리스트와 마지막 인덱스를 포함
                translated_sentences_list = progress.get("translated_sentences", [])
                start_sentence_index = progress.get("last_sentence_index", 0)

                # 로드된 번역 리스트와 시작 인덱스 일관성 확인
                if not isinstance(translated_sentences_list, list):
                    print(f"Warning: Corrupted progress file '{progress_filename}'. Expected list. Resetting.")
                    translated_sentences_list = []
                    start_sentence_index = 0
                elif len(translated_sentences_list) != start_sentence_index :
                    print(f"Warning: Mismatch in progress file '{progress_filename}'. Index: {start_sentence_index}, Loaded translations: {len(translated_sentences_list)}. Adjusting to progress index.")
                    translated_sentences_list = translated_sentences_list[:start_sentence_index]

                if start_sentence_index < len(original_sentences):
                     print(f"Resuming translation for '{input_filename_sentences}' from sentence index {start_sentence_index}")
                elif len(translated_sentences_list) >= len(original_sentences):
                     print(f"Progress file indicates all sentences in '{input_filename_sentences}' were translated. Finalizing.")
                else: # start_sentence_index가 original_sentences 길이를 넘지만, translated_sentences_list가 더 짧은 경우 (이론상 발생 안해야함)
                     print(f"Inconsistent progress for '{input_filename_sentences}'. Resetting to last valid point or start.")
                     # translated_sentences_list 길이에 맞춰 start_sentence_index 조정
                     start_sentence_index = len(translated_sentences_list)


        except Exception as e:
            print(f"Could not load progress file '{progress_filename}', starting from scratch for this file. Error: {e}")
            # traceback.print_exc() # 디버깅 시 활성화
            translated_sentences_list = []
            start_sentence_index = 0

    total_sentences = len(original_sentences)
    s_idx = start_sentence_index
    
    if s_idx >= total_sentences and total_sentences > 0: # 이미 모든 문장이 처리된 경우
        print(f"All {total_sentences} sentences in '{input_filename_sentences}' appear to be processed based on progress. Skipping direct translation loop.")
    elif total_sentences == 0:
        print(f"Input file '{input_filename_sentences}' contains no sentences to translate.")
    else:
        print(f"Translating sentences for '{input_filename_sentences}'. Starting from sentence index {s_idx}/{total_sentences}.")
        while s_idx < total_sentences:
            sentence = original_sentences[s_idx]
            print(f"  Translating sentence {s_idx + 1}/{total_sentences} from '{input_filename_sentences}': \"{str(sentence)[:60]}...\"")

            translated_sentence = translate_text_gemma(sentence)

            # translated_sentences_list 크기 s_idx에 맞게 확장 (이미 로드된 부분은 덮어쓰지 않음)
            # 현재 s_idx에 해당하는 요소가 없으면 None으로 채우고, 그 다음 해당 위치에 번역 결과 삽입
            while len(translated_sentences_list) <= s_idx:
                 translated_sentences_list.append(None)
            translated_sentences_list[s_idx] = translated_sentence # 현재 인덱스에 번역된 문장 저장

            # 10문장마다 또는 마지막 문장 처리 후 또는 오류 발생 시 진행 상황 저장
            if (s_idx + 1) % 10 == 0 or (s_idx + 1) == total_sentences or \
               (isinstance(translated_sentence, str) and translated_sentence.startswith("[Translation Error:")):
                progress_to_save = {
                    "translated_sentences": translated_sentences_list,
                    "last_sentence_index": s_idx + 1 # 다음 시작할 인덱스
                }
                try:
                    with open(progress_filename, 'w', encoding='utf-8') as pf:
                        json.dump(progress_to_save, pf, ensure_ascii=False, indent=2) # indent 줄여서 파일 크기 작게
                except Exception as e:
                    print(f"Error saving progress file '{progress_filename}': {e}")
                    # traceback.print_exc()

                if (s_idx + 1) % 10 == 0 and (s_idx + 1) < total_sentences:
                    print(f"  Processed {s_idx + 1} sentences for '{input_filename_sentences}'. Progress saved. Pausing briefly...")
                    time.sleep(1) # 짧은 대기
            s_idx += 1

    print(f"\nSaving final translated sentences for '{input_filename_sentences}' to '{output_filename_translated}'...")
    try:
        # 최종 결과는 번역된 문장 리스트 자체를 저장
        with open(output_filename_translated, 'w', encoding='utf-8') as f:
            json.dump(translated_sentences_list, f, ensure_ascii=False, indent=2)

        if os.path.exists(progress_filename) and len(translated_sentences_list) >= total_sentences and total_sentences > 0:
            print(f"All translations for '{input_filename_sentences}' completed. Removing progress file {progress_filename}...")
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
        print(f"Translation completed but final save failed. Translated data for '{input_filename_sentences}' may be in {progress_filename}.")


def main():
    # 처리할 입력 파일 리스트 - 이전 스크립트에서 생성된 파일들
    input_files_to_translate = [
        "openwebmath_sentences_from_first_3000_docs.json",
        # 필요에 따라 다른 파일 추가
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