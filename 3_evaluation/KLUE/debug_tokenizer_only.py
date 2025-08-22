#!/usr/bin/env python3
"""
Debug tokenizer loading specifically
"""

import os
import sys
import time
import logging
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIGS
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tokenizer_loading():
    """Test tokenizer loading for each model"""
    
    logger.info(f"🖥️ GPU info: {torch.cuda.device_count()} GPUs available")
    if torch.cuda.is_available():
        logger.info(f"🖥️ Current device: {torch.cuda.current_device()}")
        logger.info(f"🖥️ Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    for i, model_config in enumerate(MODEL_CONFIGS):
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing tokenizer {i+1}/{len(MODEL_CONFIGS)}: {model_config.name}")
        logger.info(f"📍 Path: {model_config.model_id}")
        
        # Check if path exists
        if not os.path.exists(model_config.model_id):
            logger.error(f"❌ Path does not exist: {model_config.model_id}")
            continue
            
        try:
            start_time = time.time()
            
            logger.info("🔤 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_id,
                trust_remote_code=True,
                padding_side='left'
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Tokenizer loaded in {load_time:.2f} seconds")
            logger.info(f"📊 Vocab size: {len(tokenizer)}")
            
            # Test tokenization
            test_text = "안녕하세요. 테스트입니다."
            tokens = tokenizer.tokenize(test_text)
            logger.info(f"🔍 Test tokenization: '{test_text}' -> {len(tokens)} tokens")
            
            # Clean up
            del tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {str(e)}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 Tokenizer testing completed")

if __name__ == "__main__":
    test_tokenizer_loading()