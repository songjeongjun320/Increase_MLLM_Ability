from datasets import load_dataset
import json

# 분석할 샘플 수 (예: 200개)
# 스트리밍 시에는 한 번에 너무 많이 가져오려 하면 중간에 오류날 가능성도 있으므로,
# 처음에는 작은 수 (예: 5-10개)로 테스트하고 점차 늘려보는 것이 좋습니다.
NUM_SAMPLES_TO_ANALYZE = 200 # 또는 처음 테스트 시에는 5 정도로 작게 시작

# (이전에 보여드린 clean_and_split_text 함수 정의는 여기에 있어야 합니다)
import re
MAX_TEXT_PREVIEW_LEN = 300
def clean_and_split_text(text_content):
    match = re.search(r"(\n\d+\s*Responses to|\nComments\s*:|\nDiscussion\s*:)", text_content, re.IGNORECASE)
    if match:
        text_content = text_content[:match.start()]
    text_content = re.sub(r'<[^>]+>', '', text_content)
    text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
    paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
    return paragraphs

print(f"Attempting to load and analyze first {NUM_SAMPLES_TO_ANALYZE} samples using streaming.")

try:
    # streaming=True 로 설정하여 실시간으로 데이터 가져오기
    dataset_stream = load_dataset(
        "open-web-math/open-web-math",
        split='train',
        streaming=True,
        trust_remote_code=True
    )

    print("Dataset stream object created. Now iterating to get samples...")

    collected_samples_data = [] # 분석 결과를 담을 리스트
    samples_processed_count = 0

    # dataset_stream.take(N) 을 사용하면 더 간결합니다.
    # for sample in dataset_stream.take(NUM_SAMPLES_TO_ANALYZE):
    # 위 take() 메소드가 Abort를 유발했다면, 수동으로 반복해볼 수 있습니다.

    stream_iterator = iter(dataset_stream) # 반복자 생성

    for i in range(NUM_SAMPLES_TO_ANALYZE):
        try:
            sample = next(stream_iterator) # 스트림에서 다음 샘플 가져오기
            samples_processed_count += 1

            print(f"\n--- Analyzing Sample {samples_processed_count} (from stream) ---")
            print(f"  URL: {sample.get('url', 'N/A')}")

            current_sample_analysis = {"url": sample.get('url', 'N/A'), "paragraphs": []}

            if 'text' in sample and sample['text']:
                original_text = sample['text']
                # print(f"  Original 'text' field (first {MAX_TEXT_PREVIEW_LEN} chars):")
                # print(original_text[:MAX_TEXT_PREVIEW_LEN] + ("..." if len(original_text) > MAX_TEXT_PREVIEW_LEN else ""))

                # print("  Attempting to split into paragraphs (and basic cleaning):")
                paragraphs = clean_and_split_text(original_text)

                if not paragraphs:
                    # print("    No paragraphs found after cleaning/splitting.")
                    current_sample_analysis["paragraphs"].append("No paragraphs found or text too short.")
                else:
                    valid_paragraphs_count = 0
                    for j, para in enumerate(paragraphs):
                        if len(para.split()) < 3 and len(para) < 20: # 짧은 단락 건너뛰기
                            continue
                        # print(f"    Paragraph {j + 1} (length: {len(para)} chars):")
                        # print(f"      \"{para[:MAX_TEXT_PREVIEW_LEN] + ('...' if len(para) > MAX_TEXT_PREVIEW_LEN else '')}\"")
                        current_sample_analysis["paragraphs"].append(para) # 실제 단락 저장
                        valid_paragraphs_count +=1
                    if valid_paragraphs_count == 0:
                         current_sample_analysis["paragraphs"].append("All paragraphs were too short after filtering.")

            else:
                # print("\n  'text' field is missing or empty.")
                current_sample_analysis["paragraphs"].append("Text field missing or empty.")
            
            collected_samples_data.append(current_sample_analysis)

            if samples_processed_count % 10 == 0: # 진행 상황 표시
                print(f"  ... processed {samples_processed_count} samples ...")


        except StopIteration:
            print(f"Stream ended after {samples_processed_count} samples (requested {NUM_SAMPLES_TO_ANALYZE}).")
            break # 루프 종료
        except Exception as e_iter:
            print(f"Error processing sample {samples_processed_count + 1} from stream: {e_iter}")
            # 오류 발생 시 해당 샘플은 건너뛰고 계속하거나, 아니면 중단할 수 있습니다.
            # 여기서는 루프를 중단합니다.
            break

    print(f"\n--- Finished processing. Collected analysis for {len(collected_samples_data)} samples ---")

    # 결과 저장 (예시)
    output_file_stream_analysis = f"openwebmath_first_{len(collected_samples_data)}_samples_analysis_streamed.json"
    if collected_samples_data:
        print(f"Saving analysis to {output_file_stream_analysis}...")
        with open(output_file_stream_analysis, 'w', encoding='utf-8') as f:
            json.dump(collected_samples_data, f, ensure_ascii=False, indent=2)
        print("Save complete.")
    else:
        print("No data collected to save.")


except Exception as e_load:
    print(f"Error setting up dataset stream: {e_load}")