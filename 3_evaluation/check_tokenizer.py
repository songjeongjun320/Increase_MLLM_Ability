#!/usr/bin/env python3
"""
ToW Token Checker for Evaluation Framework
==========================================

This module provides ToW token verification functionality for model evaluation scripts.
Checks if <ToW> and </ToW> tokens exist in the model's tokenizer.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Any


@dataclass
class TokenizerStatus:
    """Status result from tokenizer verification"""
    is_valid: bool
    has_tow_start: bool
    has_tow_end: bool
    tow_start_id: Optional[int]
    tow_end_id: Optional[int]
    issues: List[str]
    warnings: List[str]
    vocab_size: int
    model_path: str
    model_name: str


def check_tow_tokens_for_eval(
    tokenizer: Any,
    model_path: str,
    model_name: str,
    logger: Optional[logging.Logger] = None
) -> TokenizerStatus:
    """
    주어진 토크나이저에서 ToW 토큰의 존재 여부를 상세히 확인 (Evaluation용)

    Args:
        tokenizer: 로드된 HuggingFace 토크나이저
        model_path: 모델 경로 (ModelConfig에서 전달)
        model_name: 모델 이름 (ModelConfig.name에서 전달)
        logger: 로깅용 logger (선택사항)

    Returns:
        TokenizerStatus: 검증 결과
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 강조된 헤더 출력
    logger.info("=" * 80)
    logger.info(f"TOKENIZER VERIFICATION FOR EVALUATION")
    logger.info(f"Model: {model_name}")
    logger.info(f"Path: {model_path}")
    logger.info("=" * 80)

    issues = []
    warnings = []

    try:
        # 1. 기본 토크나이저 정보
        vocab_size = len(tokenizer)
        logger.info(f"Basic tokenizer info:")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Model max length: {tokenizer.model_max_length}")
        logger.info(f"  Tokenizer class: {tokenizer.__class__.__name__}")

        # 2. 특수 토큰 확인
        logger.info(f"Special tokens:")
        special_tokens = {
            'bos_token': tokenizer.bos_token,
            'eos_token': tokenizer.eos_token,
            'unk_token': tokenizer.unk_token,
            'pad_token': tokenizer.pad_token,
        }

        for name, token in special_tokens.items():
            token_id = tokenizer.convert_tokens_to_ids(token) if token else None
            logger.info(f"  {name}: {token} (ID: {token_id})")

        # Additional special tokens 확인
        additional_tokens = getattr(tokenizer, 'additional_special_tokens', []) or []
        logger.info(f"  additional_special_tokens: {additional_tokens}")

        # 3. ToW 토큰 존재 확인
        logger.info(f"ToW Token Analysis:")

        # Vocab에서 ToW 토큰 찾기
        vocab = tokenizer.get_vocab()
        tow_start_in_vocab = '<ToW>' in vocab
        tow_end_in_vocab = '</ToW>' in vocab

        tow_start_id = vocab.get('<ToW>', None)
        tow_end_id = vocab.get('</ToW>', None)

        if tow_start_in_vocab:
            logger.info(f"  <ToW> found in vocab (ID: {tow_start_id})")
        else:
            logger.warning(f"  <ToW> NOT found in vocab")
            issues.append("<ToW> token missing from vocabulary")

        if tow_end_in_vocab:
            logger.info(f"  </ToW> found in vocab (ID: {tow_end_id})")
        else:
            logger.warning(f"  </ToW> NOT found in vocab")
            issues.append("</ToW> token missing from vocabulary")

        # Additional special tokens에서 확인
        tow_start_in_additional = '<ToW>' in additional_tokens
        tow_end_in_additional = '</ToW>' in additional_tokens

        logger.info(f"  <ToW> in additional_special_tokens: {tow_start_in_additional}")
        logger.info(f"  </ToW> in additional_special_tokens: {tow_end_in_additional}")

        # 4. 토큰화 테스트
        logger.info(f"Tokenization test:")
        test_text = "Question: What is 2+2? <ToW> Let me think </ToW> Answer: 4"

        try:
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.encode(test_text, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)

            logger.info(f"  Input: {test_text}")
            logger.info(f"  Tokens: {tokens}")
            logger.info(f"  Token IDs: {token_ids}")
            logger.info(f"  Decoded: {decoded}")

            # ToW 토큰이 올바르게 토큰화되는지 확인
            tow_start_found = any('<ToW>' in str(token) for token in tokens)
            tow_end_found = any('</ToW>' in str(token) for token in tokens)

            logger.info(f"  <ToW> found in tokens: {tow_start_found}")
            logger.info(f"  </ToW> found in tokens: {tow_end_found}")

            if not tow_start_found and tow_start_in_vocab:
                warnings.append("ToW start token exists but not properly tokenized")
            if not tow_end_found and tow_end_in_vocab:
                warnings.append("ToW end token exists but not properly tokenized")

        except Exception as e:
            logger.error(f"  Tokenization test failed: {e}")
            issues.append(f"Tokenization test failed: {e}")

        # 5. 개별 토큰 테스트
        logger.info(f"Individual token tests:")
        for token in ['<ToW>', '</ToW>']:
            try:
                direct_tokens = tokenizer.tokenize(token)
                direct_ids = tokenizer.encode(token, add_special_tokens=False)
                convert_id = tokenizer.convert_tokens_to_ids(token)

                logger.info(f"  {token}:")
                logger.info(f"    tokenize(): {direct_tokens}")
                logger.info(f"    encode(): {direct_ids}")
                logger.info(f"    convert_tokens_to_ids(): {convert_id}")

            except Exception as e:
                logger.warning(f"    Error: {e}")
                warnings.append(f"Individual token test failed for {token}: {e}")

        # 6. 전체 상태 평가
        is_valid = tow_start_in_vocab and tow_end_in_vocab

        # 7. 요약
        logger.info(f"EVALUATION SUMMARY:")
        if is_valid:
            logger.info(f"  Status: ToW tokens are properly configured")
            logger.info(f"  Model is ready for ToW-based evaluation")
        else:
            logger.warning(f"  Status: ToW tokens are missing or incomplete")
            logger.warning(f"  This may affect evaluation results")

        if issues:
            logger.error(f"  Critical Issues:")
            for issue in issues:
                logger.error(f"    - {issue}")

        if warnings:
            logger.warning(f"  Warnings:")
            for warning in warnings:
                logger.warning(f"    - {warning}")

        logger.info("=" * 80)

        return TokenizerStatus(
            is_valid=is_valid,
            has_tow_start=tow_start_in_vocab,
            has_tow_end=tow_end_in_vocab,
            tow_start_id=tow_start_id,
            tow_end_id=tow_end_id,
            issues=issues,
            warnings=warnings,
            vocab_size=vocab_size,
            model_path=model_path,
            model_name=model_name
        )

    except Exception as e:
        logger.error(f"Tokenizer verification failed: {e}")
        logger.info("=" * 80)
        return TokenizerStatus(
            is_valid=False,
            has_tow_start=False,
            has_tow_end=False,
            tow_start_id=None,
            tow_end_id=None,
            issues=[f"Verification failed: {e}"],
            warnings=[],
            vocab_size=0,
            model_path=model_path,
            model_name=model_name
        )


def quick_tow_check(tokenizer: Any, model_name: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    빠른 ToW 토큰 존재 확인 (간단한 True/False 반환)

    Args:
        tokenizer: 로드된 토크나이저
        model_name: 모델 이름
        logger: 선택적 로거

    Returns:
        bool: ToW 토큰이 모두 존재하면 True
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        vocab = tokenizer.get_vocab()
        has_both = '<ToW>' in vocab and '</ToW>' in vocab

        if has_both:
            logger.info(f"{model_name}: ToW tokens verified")
        else:
            logger.warning(f"{model_name}: ToW tokens missing")

        return has_both
    except Exception as e:
        logger.error(f"{model_name}: Token check failed - {e}")
        return False


if __name__ == "__main__":
    # 테스트용 코드
    print("ToW Token Checker for Evaluation Framework")
    print("Use this module by importing check_tow_tokens_for_eval function")