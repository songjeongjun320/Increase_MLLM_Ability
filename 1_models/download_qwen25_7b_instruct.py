from huggingface_hub import hf_hub_download, snapshot_download
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
# 새 모델 정보로 업데이트
repo_id = "Qwen/Qwen2.5-7B-Instruct"
model_folder_name = "Qwen2.5-7B-Instruct_downloaded" # 저장될 폴더 이름

# 모델을 저장할 로컬 디렉토리 경로
local_target_dir = os.path.join(os.getcwd(), model_folder_name)

# 스크린샷의 커밋 해시 또는 브랜치명
# 스크린샷에는 "a09a354 VERIFIED" 커밋이 보입니다.
revision = "a09a354" # 또는 None으로 두어 기본 브랜치(main) 시도 가능

# 다운로드할 파일 목록 (스크린샷 기반 - snapshot_download 사용 시 필요 없음)
# 이 목록은 download_specific_files 함수를 사용할 경우에만 필요합니다.
files_to_download_qwen = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt", # Qwen 모델의 토크나이저 관련 파일
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json", # Qwen 모델의 토크나이저 관련 파일
]

# --- 스냅샷 다운로드 함수 (권장) ---
def download_model_snapshot(
    repo_id: str,
    target_dir: str,
    revision: str = None,
):
    logger.info(f"'{repo_id}' (리비전: {revision or 'default branch'}) 전체 스냅샷 다운로드를 시작합니다...")
    logger.info(f"저장 위치: '{target_dir}'")

    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False, # 실제 파일 다운로드
            revision=revision,
            # token=None, # 환경 변수 또는 로그인된 토큰 자동 사용
            # resume_download=True, # 다운로드 이어받기
        )
        logger.info(f"스냅샷 다운로드 완료! 파일이 저장된 경로: {snapshot_path}")
        return snapshot_path
    except Exception as e:
        logger.error(f"'{repo_id}' 스냅샷 다운로드 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# --- 개별 파일 다운로드 함수 (대안) ---
def download_specific_files(
    repo_id: str,
    filenames: list,
    target_dir: str,
    revision: str = None,
):
    logger.info(f"'{repo_id}' (리비전: {revision or 'default branch'})에서 개별 파일 다운로드를 시작합니다...")
    logger.info(f"저장 위치: '{target_dir}'")
    os.makedirs(target_dir, exist_ok=True)

    downloaded_files_paths = []
    failed_files = []

    for filename in filenames:
        logger.info(f"'{filename}' 다운로드 시도 중...")
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                revision=revision,
            )
            logger.info(f"'{filename}' 다운로드 완료! 저장 위치: {file_path}")
            downloaded_files_paths.append(file_path)
        except Exception as e:
            logger.error(f"'{filename}' (from repo '{repo_id}') 다운로드 중 오류 발생: {e}")
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
    logger.warning(f"경고: 모델 '{repo_id}' 다운로드를 시도합니다.")
    logger.warning(f"설정된 repo_id ('{repo_id}')와 revision ('{revision or 'default branch'}')이")
    logger.warning("Hugging Face Hub에 실제로 존재하고 접근 가능해야 합니다.")
    logger.warning(f"파일은 현재 디렉토리의 '{model_folder_name}' 폴더에 저장될 예정입니다.")
    logger.warning("Qwen 모델은 사용 전 동의(Accept)가 필요할 수 있습니다. Hub에서 확인하세요.")
    logger.warning("=" * 50)

    # --- 방법 1: 스냅샷 다운로드 (권장) ---
    logger.info("방법 1: 전체 스냅샷 다운로드를 시도합니다 (권장).")
    snapshot_result_path = download_model_snapshot(
        repo_id=repo_id,
        target_dir=local_target_dir,
        revision=revision,
    )

    if snapshot_result_path:
        logger.info(f"모델 스냅샷이 '{snapshot_result_path}'에 성공적으로 다운로드되었습니다.")
    else:
        logger.error("모델 스냅샷 다운로드에 실패했습니다. 로그를 확인하고 다음을 점검해주세요:")
        logger.error(f"  1. repo_id ('{repo_id}') 및 revision ('{revision or 'default branch'}')이 정확한지.")
        logger.error("  2. 해당 모델에 대한 접근 권한이 있는지 (Gated repo의 경우 Hub에서 동의 필요).")
        logger.error("  3. 인터넷 연결 상태.")

        # # --- 방법 2: 개별 파일 다운로드 (스냅샷 실패 시 대안으로 시도) ---
        # logger.info("-" * 50)
        # logger.info("스냅샷 다운로드 실패. 방법 2: 개별 파일 다운로드를 시도합니다.")
        # downloaded, failed = download_specific_files(
        #     repo_id=repo_id,
        #     filenames=files_to_download_qwen, # Qwen 모델 파일 목록 사용
        #     target_dir=local_target_dir,
        #     revision=revision,
        # )
        # if not failed:
        #     logger.info("지정된 모든 파일이 성공적으로 다운로드되었습니다 (개별 다운로드 방식).")
        # else:
        #     logger.error("개별 파일 다운로드 방식도 일부 파일 실패했습니다.")