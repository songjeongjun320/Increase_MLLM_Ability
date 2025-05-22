import json
import time
import os
import traceback # traceback 모듈 임포트
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch 

MODEL_NAME = "google/gemma-3-12b-it" 

# 모델 로드 (로컬 자원 사용!)
print(f"Loading model, processor, and tokenizer for {MODEL_NAME}...")
try:
    # processor와 tokenizer를 모두 로드합니다.
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto", # 사용 가능한 GPU에 자동으로 모델 분산
        # torch_dtype=torch.bfloat16 # 메모리가 충분하다면 정밀도 낮춰 메모리 절약 (torch 필요)
                                     # Gemma 3 2B 모델은 bfloat16 없이도 잘 동작할 수 있음
    )

    # device_map="auto" 사용 시 model.device 속성이 없을 수 있으므로,
    # 입력 텐서를 모델과 같은 장치로 옮기기 위해 모델의 첫 번째 파라미터 장치 확인
    if hasattr(model, 'device'):
        model_device = model.device
    else: # device_map="auto" 사용 시
        model_device = next(model.parameters()).device
    print(f"Model loaded successfully on device: {model_device}")
    print(f"Tokenizer loaded: {type(tokenizer)}")


except Exception as e:
    print(f"Error loading model or tokenizer {MODEL_NAME}: {e}")
    print("Please ensure you have enough VRAM (GPU memory) and disk space.")
    traceback.print_exc()
    exit()


# 재시도 관련 상수 (일반 오류에 대한 짧은 재시도)
MAX_RETRIES_ON_ERROR = 3  # 오류 시 최대 재시도 횟수
WAIT_TIME_ON_ERROR = 10  # 오류 시 대기 시간 (초)


def translate_text_gemma(text_to_translate, target_language="Korean"):
    """Gemma 모델을 사용하여 텍스트를 번역합니다."""
    if not text_to_translate or not text_to_translate.strip():
        return ""

    # Gemma instruction-tuned 모델 프롬프트 형식 (ChatML 스타일)
    # tokenizer.apply_chat_template을 사용하기 위한 메시지 리스트 형태
    messages = [
        {"role": "user", "content": f"Translate this English sentence to Korean sentence: \"{text_to_translate}\""}
    ]
    
    # 프롬프트 준비 (디버깅용, apply_chat_template이 내부적으로 처리)
    prompt_for_debug = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    for retry in range(MAX_RETRIES_ON_ERROR):
        try:
            # --- tokenizer.apply_chat_template 사용 ---
            # add_generation_prompt=True: 모델 응답의 시작을 알리는 토큰(들)을 자동으로 추가
            inputs_dict = tokenizer.apply_chat_template(
                messages,
                tokenize=True,           # 토큰 ID로 변환
                add_generation_prompt=True, # 모델의 응답 시작 토큰 추가
                return_tensors="pt"      # PyTorch 텐서 반환
            )
            # --- 수정 끝 ---

            # 모델이 로드된 장치로 입력을 이동
            # apply_chat_template은 딕셔너리가 아닌 단일 텐서를 반환할 수 있음 (tokenize=True, return_tensors='pt'일 때 보통 딕셔너리)
            # Gemma는 input_ids만 필요할 수 있으나, attention_mask도 종종 사용됨. apply_chat_template이 반환하는 것을 그대로 사용.
            if isinstance(inputs_dict, torch.Tensor): # 만약 텐서 하나만 반환되면 (Gemma의 경우 input_ids)
                 inputs_on_device = {"input_ids": inputs_dict.to(model_device)}
                 input_len = inputs_dict.shape[1]
            elif isinstance(inputs_dict, dict): # 딕셔너리 형태로 반환되면 (input_ids, attention_mask 등)
                 inputs_on_device = {k: v.to(model_device) for k, v in inputs_dict.items()}
                 input_len = inputs_on_device["input_ids"].shape[1]
            else:
                 raise ValueError(f"Unexpected output type from tokenizer.apply_chat_template: {type(inputs_dict)}")


            # 텍스트 생성 (번역)
            # generate 함수에는 pad_token_id를 설정하는 것이 좋습니다.
            outputs = model.generate(
                **inputs_on_device,
                max_new_tokens=200, # 생성할 최대 토큰 수 (300 수정)
                num_beams=1,        # 빔 서치 사용 여부 (품질 위해 1보다 크게 설정 가능)
                do_sample=False,    # 샘플링 비활성화 (일관된 번역 결과 위해 False 권장)
                temperature=0.1,    # 낮을수록 예측 가능하고 일관적 (번역에 적합)
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id # 패딩 토큰 ID 설정
            )

            # --- 결과 유효성 검사 ---
            if outputs is None or not isinstance(outputs, torch.Tensor) or outputs.ndim == 0 or outputs.shape[0] != 1:
                 print(f"Warning: model.generate returned unexpected output (None, empty, or wrong batch size) for text: '{text_to_translate[:50]}...'")
                 print(f"Received output: {outputs}, Type: {type(outputs)}")
                 if isinstance(outputs, torch.Tensor):
                     print(f"Shape: {outputs.shape}")
                 raise ValueError(f"Invalid model.generate output.")

            # 생성된 토큰 디코딩
            # outputs 텐서에는 입력 프롬프트 토큰과 생성된 토큰이 모두 포함되어 있습니다.
            # 생성된 부분만 디코딩해야 합니다.
            generated_tokens = outputs[0, input_len:]

            # 생성된 토큰들만 디코딩
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            translated_text = decoded_output.strip()

            if not translated_text:
                 print(f"Warning: Translation resulted in empty string for text: '{text_to_translate[:50]}...'")
                 print(f"Prompt used (debug): {prompt_for_debug}")
                 print(f"Full model output (including input tokens): {tokenizer.decode(outputs[0], skip_special_tokens=False)}")
                 return "[Translation Warning: Empty output]"


            return translated_text

        except Exception as e:
            print(f"Error during translation (attempt {retry+1}/{MAX_RETRIES_ON_ERROR}) for text: '{text_to_translate[:50]}...': {e}")
            traceback.print_exc()
            if retry < MAX_RETRIES_ON_ERROR - 1:
                print(f"Retrying in {WAIT_TIME_ON_ERROR} seconds...")
                time.sleep(WAIT_TIME_ON_ERROR)
                continue
            else:
                return f"[Translation Error: Failed after {MAX_RETRIES_ON_ERROR} retries - {e}]"

    return "[Translation Error: Unknown failure outside retry loop]"


def main():
    input_filename = "first_200_sentences.json"
    output_filename = "first_200_sentences_translated_ko.json"
    progress_filename = "translation_progress.json"

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filename}'.")
        return
    except Exception as e:
         print(f"An unexpected error occurred while reading input file: {e}")
         traceback.print_exc()
         return


    translated_data = {}
    start_key_index = 0
    start_sentence_index = 0

    if os.path.exists(progress_filename):
        try:
            with open(progress_filename, 'r', encoding='utf-8') as pf:
                progress = json.load(pf)
                translated_data = progress.get("translated_data", {})
                start_key_index = progress.get("last_key_index", 0)
                start_sentence_index = progress.get("last_sentence_index", 0)

                keys_list_for_progress = list(data.keys())
                if start_key_index < len(keys_list_for_progress):
                    current_key_during_load = keys_list_for_progress[start_key_index]
                    # data[current_key_during_load]가 리스트인지 먼저 확인
                    if current_key_during_load in data and isinstance(data[current_key_during_load], list):
                        current_key_sentences_during_load = data[current_key_during_load]
                        if start_sentence_index >= len(current_key_sentences_during_load):
                            start_key_index += 1
                            start_sentence_index = 0
                            if start_key_index < len(keys_list_for_progress):
                                print(f"Last key ('{current_key_during_load}') was fully translated according to progress. Starting next key.")
                            else:
                                print("Progress file indicates all keys were translated.")
                    else: # data에 해당 키가 없거나 리스트가 아닌 경우 (데이터 파일 손상 등)
                        print(f"Warning: Key '{current_key_during_load}' not found in data or is not a list. Resetting progress for this key.")
                        start_sentence_index = 0 # 해당 키에 대한 진행상황은 처음부터

                if start_key_index < len(keys_list_for_progress):
                     print(f"Resuming from key index {start_key_index} ('{keys_list_for_progress[start_key_index]}'), sentence index {start_sentence_index}")
                elif len(translated_data) > 0: # 모든 키가 완료되었을 때
                     print("Progress file indicates all keys were translated. Finalizing.")
                else:
                     print("No valid progress found or could not load, starting from scratch.")
                     start_key_index = 0
                     start_sentence_index = 0
                     translated_data = {}

        except Exception as e:
            print(f"Could not load progress file, starting from scratch. Error: {e}")
            traceback.print_exc()
            translated_data = {}
            start_key_index = 0
            start_sentence_index = 0


    keys = list(data.keys())
    total_keys = len(keys)

    for k_idx in range(start_key_index, total_keys):
        key = keys[k_idx]
        
        # 원본 데이터에 해당 키가 있는지, 그리고 리스트인지 확인
        if key not in data or not isinstance(data[key], list):
            print(f"Warning: Key '{key}' not found in input data or is not a list of sentences. Skipping this key.")
            # 진행 상황 업데이트 (이 키를 건너뛰고 다음 키로)
            translated_data.pop(key.replace("_sentences", "_sentences_ko"), None) # 혹시라도 있을 수 있는 불완전한 데이터 제거
            progress_to_save_skipped_key = {
                "translated_data": translated_data,
                "last_key_index": k_idx + 1,
                "last_sentence_index": 0
            }
            try:
                with open(progress_filename, 'w', encoding='utf-8') as pf:
                    json.dump(progress_to_save_skipped_key, pf, ensure_ascii=False, indent=4)
                print(f"Progress updated after skipping problematic key '{key}'.")
            except Exception as e:
                print(f"Error saving progress file after skipping key '{key}': {e}")
                traceback.print_exc()
            start_sentence_index = 0 # 다음 키는 처음부터
            continue # 다음 키로 이동

        sentences = data[key]
        current_key_output_name = key.replace("_sentences", "_sentences_ko")

        if k_idx == start_key_index:
            current_sentence_start_index = start_sentence_index
        else:
            current_sentence_start_index = 0

        if current_key_output_name in translated_data:
            translated_sentences = translated_data[current_key_output_name]
            if not isinstance(translated_sentences, list): # 저장된 데이터가 리스트가 아니면 초기화
                print(f"Warning: Corrupted progress for key '{key}'. Expected list, got {type(translated_sentences)}. Resetting for this key.")
                translated_sentences = []
                current_sentence_start_index = 0

            if len(translated_sentences) != current_sentence_start_index:
                 print(f"Warning: Mismatch for key '{key}'. Progress index: {current_sentence_start_index}, Loaded translations: {len(translated_sentences)}. Adjusting to progress index.")
                 translated_sentences = translated_sentences[:current_sentence_start_index]
            
            if len(translated_sentences) >= len(sentences):
                print(f"Key '{key}' ({current_key_output_name}) already has {len(translated_sentences)} translations (>= total {len(sentences)}), skipping.")
                progress_to_save_completed_key = {
                    "translated_data": translated_data,
                    "last_key_index": k_idx + 1,
                    "last_sentence_index": 0
                }
                try:
                    with open(progress_filename, 'w', encoding='utf-8') as pf:
                        json.dump(progress_to_save_completed_key, pf, ensure_ascii=False, indent=4)
                except Exception as e:
                     print(f"Error saving progress file after skipping key: {e}")
                     traceback.print_exc()
                start_sentence_index = 0
                continue
        else:
            translated_sentences = []
            if k_idx == start_key_index and start_sentence_index != 0:
                 print(f"Warning: Resuming key '{key}' from index {start_sentence_index}, but no prior translations found for this key. Starting from index 0 for this key.")
                 current_sentence_start_index = 0


        print(f"\nTranslating sentences for key: '{key}' (Index: {k_idx+1}/{total_keys}). Starting from sentence index {current_sentence_start_index}/{len(sentences)}.")
        
        s_idx = current_sentence_start_index
        while s_idx < len(sentences):
            sentence = sentences[s_idx]
            print(f"  Translating sentence {s_idx + 1}/{len(sentences)} for key '{key}': \"{str(sentence)[:50]}...\"")

            translated_sentence = translate_text_gemma(sentence)

            # translated_sentences 리스트 크기 s_idx에 맞게 확장
            while len(translated_sentences) <= s_idx:
                 translated_sentences.append(None) 
            translated_sentences[s_idx] = translated_sentence


            if (s_idx + 1) % 10 == 0 or (s_idx + 1) == len(sentences) or (isinstance(translated_sentence, str) and translated_sentence.startswith("[Translation Error:")):
                 translated_data[current_key_output_name] = translated_sentences
                 progress_to_save = {
                     "translated_data": translated_data,
                     "last_key_index": k_idx,
                     "last_sentence_index": s_idx + 1
                 }
                 try:
                     with open(progress_filename, 'w', encoding='utf-8') as pf:
                         json.dump(progress_to_save, pf, ensure_ascii=False, indent=4)
                 except Exception as e:
                      print(f"Error saving progress file: {e}")
                      traceback.print_exc()

                 if (s_idx + 1) % 10 == 0 and s_idx + 1 < len(sentences):
                     print("  Processed 10 sentences. Pausing briefly...")
                     time.sleep(1)
            s_idx += 1

        translated_data[current_key_output_name] = translated_sentences
        progress_to_save_completed_key = {
            "translated_data": translated_data,
            "last_key_index": k_idx + 1,
            "last_sentence_index": 0
        }
        print(f"Finished translating all sentences for key: '{key}'. Saving final progress for this key...")
        try:
            with open(progress_filename, 'w', encoding='utf-8') as pf:
                json.dump(progress_to_save_completed_key, pf, ensure_ascii=False, indent=4)
            print(f"Progress saved to {progress_filename}. Ready for next key.")
        except Exception as e:
             print(f"Error saving progress file after completing key: {e}")
             traceback.print_exc()
        start_sentence_index = 0


    print(f"\nSaving final translated sentences to {output_filename}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)

        if os.path.exists(progress_filename):
            print(f"All translations completed. Removing progress file {progress_filename}...")
            os.remove(progress_filename)
            print("Progress file removed.")
        print(f"Successfully saved translated data to {output_filename}.")

    except Exception as e:
        print(f"Error saving final output file {output_filename}: {e}")
        traceback.print_exc()
        print(f"Translation completed but final save failed. Translated data may be in {progress_filename}.")


if __name__ == "__main__":
    main()