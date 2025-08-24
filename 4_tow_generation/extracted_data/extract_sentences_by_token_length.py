import json
import os
import argparse
from tqdm import tqdm

def count_tokens_simple(text):
    """간단한 토큰 카운팅 (공백 기준)"""
    return len(text.split()) if text else 0

def extract_sentences(input_dir, output_file, min_token_count):
    """
    지정된 디렉토리의 모든 JSON 파일에서
    문장의 토큰 수가 임계값을 넘는 데이터를 추출하여 저장합니다.
    """
    if not os.path.isdir(input_dir):
        print(f"오류: 입력 디렉토리 '{input_dir}'를 찾을 수 없습니다.")
        return

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not json_files:
        print(f"'{input_dir}' 디렉토리에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"'{input_dir}' 디렉토리에서 총 {len(json_files)}개의 JSON 파일을 처리합니다.")
    print(f"최소 토큰 수 기준: {min_token_count}개")
    
    extracted_data = []
    total_items_processed = 0

    # tqdm을 사용하여 파일 처리 진행 상황 표시
    for filename in tqdm(json_files, desc="파일 처리 중"):
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    print(f"\n경고: '{filename}' 파일의 형식이 리스트가 아닙니다. 건너뜁니다.")
                    continue
                
                total_items_processed += len(data)
                
                for item in data:
                    # 'sentence' 키가 없는 경우를 대비
                    sentence = item.get('sentence')
                    if sentence and isinstance(sentence, str):
                        token_count = count_tokens_simple(sentence)
                        if token_count >= min_token_count:
                            extracted_data.append(item)
        except json.JSONDecodeError:
            print(f"\n오류: '{filename}' 파일이 올바른 JSON 형식이 아닙니다.")
        except Exception as e:
            print(f"\n오류: '{filename}' 파일을 처리하는 중 예외가 발생했습니다: {e}")

    print(f"\n총 {total_items_processed:,}개의 문장 중 {len(extracted_data):,}개를 추출했습니다.")

    try:
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        print(f"추출된 데이터를 '{output_file}' 파일에 성공적으로 저장했습니다.")
    except Exception as e:
        print(f"오류: 결과를 '{output_file}' 파일에 저장하는 중 예외가 발생했습니다: {e}")


def main():
    parser = argparse.ArgumentParser(description='문장 토큰 길이를 기준으로 JSON 데이터를 필터링합니다.')
    parser.add_argument(
        '--input-dir', 
        default='org_data', 
        help='입력 JSON 파일들이 있는 디렉토리 (기본값: org_data)'
    )
    parser.add_argument(
        '--output-file', 
        default='extract_over_20token.json', 
        help='추출된 데이터를 저장할 파일 경로 (기본값: extract_over_20token.json)'
    )
    parser.add_argument(
        '--min-tokens', 
        type=int, 
        default=20, 
        help='문장의 최소 토큰 수 (기본값: 20)'
    )
    
    args = parser.parse_args()
    
    extract_sentences(args.input_dir, args.output_file, args.min_tokens)

if __name__ == "__main__":
    main()
