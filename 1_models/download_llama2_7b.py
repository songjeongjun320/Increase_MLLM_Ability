from huggingface_hub import hf_hub_download, snapshot_download
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
repo_id = "meta-llama/Llama-2-7b"
model_folder_name = "../Base_Models/Llama-2-7b_pretrained"

# 모델을 저장할 로컬 디렉토리 경로
local_target_dir = os.path.join(os.getcwd(), model_folder_name)

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
            local_dir_use_symlinks=False,
            revision=revision,
        )
        logger.info(f"스냅샷 다운로드 완료! 파일이 저장된 경로: {snapshot_path}")
        return snapshot_path
    except Exception as e:
        logger.error(f"'{repo_id}' 스냅샷 다운로드 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# --- 스크립트 실행 ---
if __name__ == "__main__":
    logger.warning("=" * 50)
    logger.warning(f"경고: 모델 '{repo_id}' 다운로드를 시도합니다.")
    logger.warning(f"설정된 repo_id ('{repo_id}')가")
    logger.warning("Hugging Face Hub에 실제로 존재하고 접근 가능해야 합니다.")
    logger.warning(f"파일은 현재 디렉토리의 '{model_folder_name}' 폴더에 저장될 예정입니다.")
    logger.warning("=" * 50)
    logger.warning("")
    logger.warning("⚠️  중요: meta-llama 모델은 라이센스 동의가 필요합니다!")
    logger.warning("1. https://huggingface.co/meta-llama/Llama-2-7b 방문")
    logger.warning("2. 라이센스 동의 후 Access Token 발급")
    logger.warning("3. 'huggingface-cli login' 명령어로 로그인")
    logger.warning("=" * 50)

    # 스냅샷 다운로드
    logger.info("전체 스냅샷 다운로드를 시도합니다.")
    snapshot_result_path = download_model_snapshot(
        repo_id=repo_id,
        target_dir=local_target_dir,
        revision=None,
    )

    if snapshot_result_path:
        logger.info(f"모델 스냅샷이 '{snapshot_result_path}'에 성공적으로 다운로드되었습니다.")
    else:
        logger.error("모델 스냅샷 다운로드에 실패했습니다. 로그를 확인하고 다음을 점검해주세요:")
        logger.error(f"  1. repo_id ('{repo_id}')가 정확한지.")
        logger.error("  2. 해당 모델에 대한 접근 권한이 있는지 (라이센스 동의 및 로그인 필요).")
        logger.error("  3. 인터넷 연결 상태.")