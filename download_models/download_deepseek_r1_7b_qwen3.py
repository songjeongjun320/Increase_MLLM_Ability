from huggingface_hub import snapshot_download
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
repo_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
model_folder_name = "DeepSeek-R1-0528-Qwen3-8B_downloaded" # 저장될 폴더 이름

# 모델을 저장할 로컬 디렉토리 경로
local_target_dir = os.path.join(os.getcwd(), model_folder_name)

# 스크린샷의 커밋 해시 또는 브랜치명
# 스크린샷에는 "6e8885a" 커밋이 보입니다.
revision = "6e8885a" # 또는 None으로 두어 기본 브랜치(main) 시도 가능

# --- 스냅샷 다운로드 함수 (권장) ---
def download_model_snapshot(
    repo_id: str,
    target_dir: str,
    revision: str = None,
    # token: str = None, # 환경 변수 또는 로그인된 토큰 자동 사용
):
    logger.info(f"'{repo_id}' (리비전: {revision or 'default branch'}) 전체 스냅샷 다운로드를 시작합니다...")
    logger.info(f"저장 위치: '{target_dir}'")

    try:
        # snapshot_download는 target_dir이 없으면 생성합니다.
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False, # 실제 파일 다운로드
            revision=revision,
            # token=token, # 필요한 경우 명시적 토큰 전달
            # resume_download=True, # 다운로드 이어받기
            # ignore_patterns=["*.txt", "*.md"], # 특정 파일 제외 (선택 사항)
        )
        logger.info(f"스냅샷 다운로드 완료! 파일이 저장된 경로: {snapshot_path}")
        return snapshot_path
    except Exception as e:
        logger.error(f"'{repo_id}' 스냅샷 다운로드 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc()) # 전체 오류 스택 추적
        return None

# --- 스크립트 실행 ---
if __name__ == "__main__":
    logger.warning("=" * 50)
    logger.warning(f"경고: 모델 '{repo_id}' 다운로드를 시도합니다.")
    logger.warning(f"설정된 repo_id ('{repo_id}')와 revision ('{revision or 'default branch'}')이")
    logger.warning("Hugging Face Hub에 실제로 존재하고 접근 가능해야 합니다.")
    logger.warning(f"파일은 현재 디렉토리의 '{model_folder_name}' 폴더에 저장될 예정입니다.")
    logger.warning("DeepSeek 모델은 사용 전 Hugging Face Hub에서 라이선스 동의가 필요할 수 있습니다. 확인하세요.")
    logger.warning("=" * 50)

    # 스냅샷 다운로드 시도
    snapshot_result_path = download_model_snapshot(
        repo_id=repo_id,
        target_dir=local_target_dir,
        revision=revision,
    )

    if snapshot_result_path:
        logger.info(f"모델 스냅샷이 '{snapshot_result_path}'에 성공적으로 다운로드되었습니다.")
        # 스크린샷에 'figures' 폴더가 있었으므로, 해당 폴더도 다운로드되었는지 확인 가능
        figures_dir = os.path.join(snapshot_result_path, "figures")
        if os.path.isdir(figures_dir):
            logger.info(f"'figures' 폴더도 성공적으로 다운로드되었습니다: {figures_dir}")
        else:
            logger.info("'figures' 폴더는 다운로드되지 않았거나 스냅샷에 포함되지 않았습니다.")
    else:
        logger.error("모델 스냅샷 다운로드에 실패했습니다. 로그를 확인하고 다음을 점검해주세요:")
        logger.error(f"  1. repo_id ('{repo_id}') 및 revision ('{revision or 'default branch'}')이 정확한지.")
        logger.error("  2. 해당 모델에 대한 접근 권한이 있는지 (Gated repo 또는 라이선스 동의 필요).")
        logger.error("  3. 인터넷 연결 상태 및 디스크 공간.")