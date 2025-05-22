import json
import os
from datasets import load_dataset
import logging
import sys
import datasets as ds_lib # datasets 버전 확인용

# --- Configuration ---
DATASET_NAME = "cais/mmlu"
# 'all'을 로드하거나 특정 과목(예: 'abstract_algebra')을 지정할 수 있습니다.
# 'all'을 로드하면 모든 과목 데이터가 포함됩니다.
DATASET_CONFIG = "all"
DATASET_SPLIT = "test"
# 출력 파일 이름을 원하는 대로 지정하세요.
OUTPUT_FILENAME = f"MMLU_{DATASET_CONFIG}_{DATASET_SPLIT}_origin.json"
HUGGINGFACE_TOKEN_PATH = os.path.expanduser("~/.huggingface/token")
CACHE_DIR = "/scratch/jsong132/.cache/huggingface" # 명시적 캐시 디렉토리 지정 (권장)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 버전 확인 ---
logger.info(f"Python version: {sys.version}")
logger.info(f"Datasets library version: {ds_lib.__version__}") # 업데이트된 버전 확인

# --- Main Function ---
def download_and_save_mmlu(output_filepath: str):
    """
    Hugging Face Hub에서 MMLU 데이터셋을 로드하고 JSON 파일로 저장합니다.
    """
    # 1. Hugging Face 토큰 로드
    token = None
    try:
        with open(HUGGINGFACE_TOKEN_PATH, "r") as f:
            token = f.read().strip()
        if not token:
             logger.warning(f"Hugging Face 토큰 파일({HUGGINGFACE_TOKEN_PATH})이 비어있습니다.")
        else:
            logger.info("Hugging Face 토큰 로드 완료.")
    except FileNotFoundError:
        logger.warning(f"Hugging Face 토큰 파일({HUGGINGFACE_TOKEN_PATH})을 찾을 수 없습니다. 토큰 없이 진행합니다.")
    except Exception as e:
        logger.error(f"Hugging Face 토큰 읽기 오류: {e}")
        # return # 필요시 여기서 중단

    # 2. 데이터셋 로드 (업데이트된 라이브러리 사용)
    logger.info(f"데이터셋 로딩 중: '{DATASET_NAME}' (구성: '{DATASET_CONFIG}', 스플릿: '{DATASET_SPLIT}')...")
    try:
        # 최신 라이브러리는 use_auth_token 대신 token 인수를 사용할 수 있습니다.
        # cache_dir을 명시적으로 지정하는 것이 좋습니다.
        ds = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=DATASET_SPLIT, # split 인수를 직접 사용
            token=token,         # use_auth_token 대신 token 사용 (최신 버전)
            cache_dir=CACHE_DIR  # 캐시 디렉토리 지정
        )
        # split 인수를 사용하지 않고 나중에 접근해도 됩니다:
        # ds_full = load_dataset(DATASET_NAME, DATASET_CONFIG, token=token, cache_dir=CACHE_DIR)
        # ds = ds_full[DATASET_SPLIT]

    except Exception as e:
        logger.error(f"데이터셋 로딩 실패: {e}")
        logger.error("네트워크, 데이터셋 이름/구성, 토큰, 캐시 디렉토리 권한/공간을 확인하세요.")
        # 추가 디버깅: 어떤 인수가 문제인지 확인
        import inspect
        sig = inspect.signature(load_dataset)
        logger.debug(f"load_dataset 인자: {sig}")
        return

    logger.info(f"'{DATASET_SPLIT}' 스플릿 로드 완료 ({len(ds)} 개 예제).")

    # 3. 데이터 예시 출력
    if len(ds) > 0:
         logger.info(f"데이터 예시: {ds[0]}")
    else:
         logger.warning(f"'{DATASET_SPLIT}' 스플릿에 데이터가 없습니다.")

    # 4. 데이터셋을 Python 리스트로 변환
    logger.info("데이터를 리스트 형태로 변환 중...")
    # Parquet 등 다른 형식이더라도 라이브러리가 딕셔너리로 잘 변환해 줍니다.
    data_list = [item for item in ds]
    logger.info("리스트 변환 완료.")

    # 5. JSON 파일로 저장
    logger.info(f"데이터를 '{output_filepath}' 파일로 저장 중...")
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
             logger.info(f"출력 디렉토리 생성됨: {output_dir}")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        logger.info(f"데이터가 성공적으로 저장되었습니다: {os.path.abspath(output_filepath)}")
    except IOError as e:
        logger.error(f"파일 쓰기 오류 {output_filepath}: {e}")
    except Exception as e:
        logger.error(f"JSON 저장 중 예기치 않은 오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    download_and_save_mmlu(OUTPUT_FILENAME)