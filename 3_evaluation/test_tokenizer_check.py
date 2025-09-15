#!/usr/bin/env python3
"""
Test script for tokenizer verification in evaluation framework
"""

import os
import sys
import logging
from check_tokenizer import check_tow_tokens_for_eval, quick_tow_check

def test_tokenizer_verification():
    """
    Test the tokenizer verification functionality
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Test with a sample model path (change this to an actual model path you have)
    # You can test with any HuggingFace model or local model
    test_models = [
        {
            "name": "test-model-1",
            "path": "microsoft/DialoGPT-medium",  # Example model without ToW tokens
            "description": "Standard model (should not have ToW tokens)"
        }
    ]

    print("="*80)
    print("TESTING TOKENIZER VERIFICATION FUNCTIONALITY")
    print("="*80)

    for i, model_info in enumerate(test_models, 1):
        print(f"\nTest {i}: {model_info['description']}")
        print("-" * 60)

        try:
            # Try to load tokenizer and test
            from transformers import AutoTokenizer

            print(f"Loading tokenizer from: {model_info['path']}")
            tokenizer = AutoTokenizer.from_pretrained(model_info['path'])

            # Test the detailed verification
            print(f"\nRunning detailed verification...")
            status = check_tow_tokens_for_eval(
                tokenizer=tokenizer,
                model_path=model_info['path'],
                model_name=model_info['name'],
                logger=logger
            )

            # Test the quick check
            print(f"\nRunning quick verification...")
            quick_result = quick_tow_check(tokenizer, model_info['name'], logger)

            print(f"\nTest Results Summary:")
            print(f"  Detailed check valid: {status.is_valid}")
            print(f"  Quick check result: {quick_result}")
            print(f"  Has <ToW>: {status.has_tow_start}")
            print(f"  Has </ToW>: {status.has_tow_end}")
            if status.issues:
                print(f"  Issues: {status.issues}")
            if status.warnings:
                print(f"  Warnings: {status.warnings}")

        except Exception as e:
            logger.error(f"Test failed for {model_info['name']}: {e}")
            print(f"Error details: {e}")

    print("\n" + "="*80)
    print("TOKENIZER VERIFICATION TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    print("Starting tokenizer verification test...")
    test_tokenizer_verification()