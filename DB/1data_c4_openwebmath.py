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
            nltk.data.find(resource_path)
        except Exception as e:
            print(f"Error downloading '{resource_name}': {e}")
            print(f"Please try downloading '{resource_name}' manually: import nltk; nltk.download('{resource_name}')")
            exit()

download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('punkt_tab', 'tokenizers/punkt_tab')


def get_sentences_from_first_n_documents(dataset_name, num_documents, config_name=None, split='train', text_column='text'):
    """
    주어진 Hugging Face 데이터셋에서 처음 N개의 문서에서 모든 문장을 추출합니다.
    데이터셋은 스트리밍 모드로 로드합니다.
    """
    print(f"Loading dataset: {dataset_name} (config: {config_name}, split: {split}) to get sentences from first {num_documents} documents.")

    if dataset_name == 'c4':
        dataset_name_to_load = 'allenai/c4'
        print(f"Using recommended dataset name for C4: {dataset_name_to_load}")
    else:
        dataset_name_to_load = dataset_name

    try:
        if config_name:
            dataset_iterable = load_dataset(dataset_name_to_load, config_name, split=split, streaming=True, trust_remote_code=True)
        else:
            dataset_iterable = load_dataset(dataset_name_to_load, split=split, streaming=True, trust_remote_code=True)
        
        # 스트리밍 데이터셋에서 처음 N개의 문서를 가져옵니다.
        dataset = dataset_iterable.take(num_documents)

    except Exception as e:
        print(f"Error loading or taking first {num_documents} documents from {dataset_name_to_load}: {e}")
        return [] # 데이터셋 로드/처리 실패 시 빈 리스트 반환

    sentences_collected = []
    documents_processed = 0
    total_sentences_extracted = 0
    print(f"Extracting all sentences from the first {num_documents} documents of {dataset_name_to_load}...")

    for doc in dataset: # 이제 dataset은 num_documents 만큼의 문서만 포함
        documents_processed += 1
        if documents_processed % 500 == 0 and documents_processed > 0 : # 로그 빈도 조절
             print(f"  Processed {documents_processed}/{num_documents} documents from {dataset_name_to_load}, collected {total_sentences_extracted} sentences so far...")

        if text_column not in doc or not doc[text_column]:
            # print(f"  Warning: Document {documents_processed} found without '{text_column}' field or empty content. Skipping.")
            continue

        try:
            document_text = str(doc[text_column])
            doc_sentences = nltk.sent_tokenize(document_text)

            for sent in doc_sentences:
                stripped_sent = sent.strip()
                if stripped_sent: # 빈 문장이 아닌 경우에만 추가
                    sentences_collected.append(stripped_sent)
                    total_sentences_extracted += 1
        
        except LookupError as le:
            print(f"  NLTK LookupError while tokenizing document {documents_processed}: {le}. This might indicate a missing resource.")
            continue
        except Exception as e:
            # print(f"  Error tokenizing document {documents_processed}: {e}. Skipping document.")
            continue
    
    # 루프 후 최종 문서 처리 수 확인 (take(N)이 정확히 N개를 반환하지 않을 수도 있는 엣지 케이스 대비)
    print(f"Finished extracting sentences. Processed {documents_processed} documents (target: {num_documents}) from {dataset_name_to_load}.")
    print(f"Collected a total of {total_sentences_extracted} sentences.")
    return sentences_collected

if __name__ == "__main__":
    num_docs_to_process = 3000 # 논문에서 언급된 문서 수

    # 출력 파일 이름에 문서 수를 명시하는 것이 좋음
    output_filename_c4 = f"c4_sentences_from_first_{num_docs_to_process}_docs.json"
    output_filename_owm = f"openwebmath_sentences_from_first_{num_docs_to_process}_docs.json"

    print(f"--- Processing C4 dataset (first {num_docs_to_process} documents) ---")
    c4_sentences = get_sentences_from_first_n_documents(
        dataset_name='c4',
        num_documents=num_docs_to_process,
        config_name='en',
        split='train',
        text_column='text'
    )
    if c4_sentences:
        print(f"\nSaving sentences from first {num_docs_to_process} C4 documents to {output_filename_c4}...")
        with open(output_filename_c4, 'w', encoding='utf-8') as f:
            json.dump(c4_sentences, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved C4 data to {output_filename_c4}")
        print(f"Total number of sentences extracted from first {num_docs_to_process} C4 documents: {len(c4_sentences)}")
    else:
        print(f"No sentences extracted from the first {num_docs_to_process} C4 documents or an error occurred.")


    print(f"\n--- Processing OpenWebMath dataset (first {num_docs_to_process} documents) ---")
    openwebmath_sentences = get_sentences_from_first_n_documents(
        dataset_name='open-web-math/open-web-math',
        num_documents=num_docs_to_process,
        split='train',
        text_column='text'
    )
    if openwebmath_sentences:
        print(f"\nSaving sentences from first {num_docs_to_process} OpenWebMath documents to {output_filename_owm}...")
        with open(output_filename_owm, 'w', encoding='utf-8') as f:
            json.dump(openwebmath_sentences, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved OpenWebMath data to {output_filename_owm}")
        print(f"Total number of sentences extracted from first {num_docs_to_process} OpenWebMath documents: {len(openwebmath_sentences)}")
    else:
        print(f"No sentences extracted from the first {num_docs_to_process} OpenWebMath documents or an error occurred.")

    print("\nScript finished.")