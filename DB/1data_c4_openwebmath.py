import json
from datasets import load_dataset
import nltk
import os

def download_nltk_resource(resource_name, resource_path):
    """NLTK 리소스를 확인하고 없으면 다운로드하는 헬퍼 함수"""
    try:
        nltk.data.find(resource_path)
        print(f"NLTK '{resource_name}' resource found.")
    except LookupError:
        print(f"NLTK '{resource_name}' resource not found. Downloading...")
        try:
            nltk.download(resource_name, quiet=True)
            print(f"NLTK '{resource_name}' resource downloaded successfully.")
            # 다운로드 후 다시 확인
            nltk.data.find(resource_path)
        except Exception as e:
            print(f"Error downloading '{resource_name}': {e}")
            print(f"Please try downloading '{resource_name}' manually in a Python interpreter: import nltk; nltk.download('{resource_name}')")
            exit() # 다운로드 실패 시 스크립트 종료

# NLTK 리소스 다운로드 (punkt 및 punkt_tab)
download_nltk_resource('punkt', 'tokenizers/punkt')
# 'punkt_tab'은 특정 경로 지정 없이 이름만으로도 다운로드 및 검색이 잘 되는 경우가 많습니다.
# 만약 'punkt_tab'도 특정 경로로 찾아야 한다면, 오류 메시지에서 'Attempted to load tokenizers/punkt_tab/english/' 와 같은 경로를 참고하여
# resource_path를 'tokenizers/punkt_tab' 또는 'tokenizers/punkt_tab/english' 등으로 지정해볼 수 있습니다.
# 우선은 이름만으로 시도합니다.
download_nltk_resource('punkt_tab', 'tokenizers/punkt_tab')


def get_first_n_sentences(dataset_name, config_name=None, split='train', text_column='text', n=200):
    """
    주어진 Hugging Face 데이터셋에서 처음 n개의 문장을 추출합니다.
    데이터셋은 스트리밍 모드로 로드하여 메모리 부담을 줄입니다.
    """
    print(f"Loading dataset: {dataset_name} (config: {config_name}, split: {split})")
    # 스트리밍 모드로 데이터셋 로드
    # C4 데이터셋의 경우 trust_remote_code=True 추가 (경고 메시지 및 사용자 입력 방지)
    # 그리고 deprecated 경고에 따라 'allenai/c4' 사용 권장
    if dataset_name == 'c4':
        dataset_name_to_load = 'allenai/c4' # 추천 이름 사용
        print(f"Using recommended dataset name for C4: {dataset_name_to_load}")
    else:
        dataset_name_to_load = dataset_name

    if config_name:
        dataset = load_dataset(dataset_name_to_load, config_name, split=split, streaming=True, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name_to_load, split=split, streaming=True, trust_remote_code=True)

    sentences_collected = []
    documents_processed = 0
    print(f"Extracting first {n} sentences from {dataset_name_to_load}...")

    for doc in dataset:
        if len(sentences_collected) >= n:
            break

        documents_processed += 1
        if documents_processed % 10 == 0 and documents_processed > 0 :
             print(f"  Processed {documents_processed} documents from {dataset_name_to_load}, collected {len(sentences_collected)} sentences...")

        if text_column not in doc or not doc[text_column]:
            print(f"  Warning: Document found without '{text_column}' field or empty content. Skipping.")
            continue

        try:
            document_text = str(doc[text_column])
            # 문장 토큰화 시 발생할 수 있는 다양한 예외를 포괄적으로 처리
            doc_sentences = nltk.sent_tokenize(document_text)

            for sent in doc_sentences:
                if len(sentences_collected) < n:
                    sentences_collected.append(sent.strip())
                else:
                    break
        except LookupError as le: # NLTK 리소스 관련 오류 재발생 시
            print(f"  NLTK LookupError while tokenizing: {le}. This might indicate a missing resource.")
            print(f"  Skipping document. Please ensure all NLTK resources (like 'punkt', 'punkt_tab') are downloaded.")
            # 여기서 프로그램을 종료하거나, 특정 리소스 다운로드를 다시 시도하는 로직을 넣을 수도 있습니다.
            # 우선은 해당 문서만 스킵하도록 둡니다.
            continue
        except Exception as e: # 그 외 예외
            print(f"  Error tokenizing document: {e}. Skipping document.")
            continue


    print(f"Finished extracting from {dataset_name_to_load}. Collected {len(sentences_collected)} sentences from {documents_processed} documents.")
    return sentences_collected

if __name__ == "__main__":
    num_sentences_to_extract = 200
    output_filename = "first_200_sentences.json"

    # C4 데이터셋 (allenai/c4 로 변경하고 trust_remote_code=True 사용)
    c4_sentences = get_first_n_sentences(
        dataset_name='c4', # 내부적으로 'allenai/c4'로 변경됨
        config_name='en',
        split='train',
        text_column='text',
        n=num_sentences_to_extract
    )

    openwebmath_sentences = get_first_n_sentences(
        dataset_name='open-web-math/open-web-math',
        split='train',
        text_column='text',
        n=num_sentences_to_extract
    )

    data_to_save = {
        "c4_first_200_sentences": c4_sentences,
        "openwebmath_first_200_sentences": openwebmath_sentences
    }

    print(f"\nSaving extracted sentences to {output_filename}...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"Successfully saved data to {output_filename}")
    print(f"Number of sentences from C4: {len(c4_sentences)}")
    print(f"Number of sentences from OpenWebMath: {len(openwebmath_sentences)}")