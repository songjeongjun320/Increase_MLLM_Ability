from huggingface_hub import hf_hub_download
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 (스크린샷 기반, 실제 존재 여부 불확실) ---
# repo_id는 여전히 정확한 값으로 찾아야 합니다.
# 스크린샷 상단에 "patrickvonplaten"의 커밋이 보이므로, repo_id를 다음과 같이 가정합니다.
repo_id = "mistralai/Ministral-8B-Instruct-2410"
# 또는 repo_id = "mistralai/Mistral-8B-Instruct-2410" # 시도해볼 수 있는 다른 옵션

# 모델을 저장할 로컬 디렉토리 경로 (현재 작업 디렉토리 아래 'Mistral-8B-Instruct-2410')
# os.getcwd()는 현재 스크립트가 실행되는 디렉토리 경로를 반환합니다.
local_target_dir = os.path.join(os.getcwd(), "Mistral-8B-Instruct-2410")

# 스크린샷의 커밋 해시
revision = "4847e87" # 스크린샷 상단의 "4847e87 VERIFIED" 부분

# 다운로드할 파일 목록 (스크린샷에 보이는 그대로)
files_to_download_from_screenshot = [
    ".gitattributes",
    "README.md",
    "config.json",
    "consolidated.safetensors",       # 16 GB LFS
    "generation_config.json",
    "model-00001-of-00004.safetensors", # 4.98 GB LFS
    "model-00002-of-00004.safetensors", # 5 GB LFS
    "model-00003-of-00004.safetensors", # 4.98 GB LFS
    "model-00004-of-00004.safetensors", # 1.07 GB LFS
    "model.safetensors.index.json",
    "params.json",
    "passkey_example.json",
    "special_tokens_map.json",
    "tekken.json",                    # 14.8 MB LFS
    "tokenizer.json",                 # 17.1 MB LFS
    "tokenizer_config.json",
]

# --- 다운로드 함수 ---
def download_specific_files_to_current_subdir(
    repo_id: str,
    filenames: list,
    target_dir: str, # 파일을 저장할 최종 디렉토리
    revision: str = None,
):
    logger.info(f"스크린샷 기반으로 '{repo_id}' (리비전: {revision})에서 파일 다운로드를 시작합니다...")
    logger.info(f"저장 위치: '{target_dir}'")
    os.makedirs(target_dir, exist_ok=True) # 저장 디렉토리 생성

    downloaded_files_paths = []
    failed_files = []

    for filename in filenames:
        logger.info(f"'{filename}' 다운로드 시도 중...")
        try:
            # hf_hub_download는 local_dir 내부에 파일을 저장합니다.
            # target_dir을 local_dir로 직접 사용합니다.
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_dir, # 파일을 이 디렉토리 바로 아래에 저장
                local_dir_use_symlinks=False,
                revision=revision,
            )
            logger.info(f"'{filename}' 다운로드 완료! 저장 위치: {file_path}")
            downloaded_files_paths.append(file_path)
        except Exception as e:
            logger.error(f"'{filename}' 다운로드 중 오류 발생: {e}")
            failed_files.append(filename)

    logger.info("-" * 30)
    if downloaded_files_paths:
        logger.info(f"{len(downloaded_files_paths)}개 파일 다운로드 성공.")
    if failed_files:
        logger.warning(f"{len(failed_files)}개 파일 다운로드 실패: {', '.join(failed_files)}")
    logger.info("-" * 30)
    return downloaded_files_paths, failed_files

# --- 스크립트 실행 ---
if __name__ == "__main__":
    logger.warning("=" * 50)
    logger.warning("경고: 이 스크립트는 제공된 스크린샷의 파일 목록을 기반으로 합니다.")
    logger.warning(f"설정된 repo_id ('{repo_id}')와 revision ('{revision}')이 Hugging Face Hub에 실제로 존재하고,")
    logger.warning("나열된 파일들이 해당 repo/revision에 정확히 존재해야 합니다.")
    logger.warning("그렇지 않으면 대부분의 또는 모든 파일 다운로드가 404 오류로 실패할 것입니다.")
    logger.warning(f"파일은 현재 디렉토리의 '{os.path.basename(local_target_dir)}' 폴더에 저장될 예정입니다.")
    logger.warning("=" * 50)

    downloaded, failed = download_specific_files_to_current_subdir(
        repo_id=repo_id,
        filenames=files_to_download_from_screenshot,
        target_dir=local_target_dir, # 수정된 저장 경로 전달
        revision=revision,
    )

    if not failed:
        logger.info("모든 지정된 파일이 성공적으로 다운로드되었습니다.")
    else:
        logger.error("일부 파일 다운로드에 실패했습니다. 로그를 확인하고 repo_id, revision 및 파일명을 점검해주세요.")