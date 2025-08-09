#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Option 2 TOW Dataset Generator
=============================

Updated TOW generation script for Option 2: Pure Original TOW Implementation.
This script generates TOW datasets with:

- Token Classification (trivial/exact/soft/unpredictable)
- Cross-lingual support (English thoughts for Korean/other languages)
- Data augmentation pipeline integration
- Proper <ToW> formatting compliance
- Enhanced training dataset generation

Usage:
    python DB/option2_tow_generator.py --input_file data.jsonl --output_file tow_dataset.jsonl
"""

import sys
import os
import json
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tow_architecture.core.thought_processor import ThoughtTokenProcessor
from tow_architecture.core.token_classifier import TokenClassifier, ClassificationContext, TOWCategory
from tow_architecture.core.cross_lingual_tow import CrossLingualTOWSystem, CrossLingualContext
from tow_architecture.data_augmentation.pipeline import TOWDataAugmentationPipeline, TOWEntry
from tow_architecture.utils.config import TOWConfig, get_development_config, get_production_config
from tow_architecture.utils.logger import get_logger
from tow_architecture.utils.text_utils import sanitize_tow_token, enforce_english_text

logger = get_logger(__name__)


class DeepSeekTOWAdapter:
    """
    Adapter for DeepSeek model with TOW generation capabilities.
    This replaces the old generation approach with Option 2 methodology.
    """
    
    def __init__(self, model_path: str = None, language: str = "ko"):
        """
        Initialize DeepSeek TOW adapter.
        
        Args:
            model_path: Path to DeepSeek model (optional for mock)
            language: Source language for processing
        """
        self.model_path = model_path
        self.language = language
        self.model = None
        self.tokenizer = None
        
        # Option 2 components
        self.token_classifier = TokenClassifier(language=language)
        self.cross_lingual_system = CrossLingualTOWSystem(self, language)
        self.thought_processor = ThoughtTokenProcessor(self, language=language)
        
        logger.info(f"DeepSeekTOWAdapter initialized for Option 2 (language: {language})")
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate text using DeepSeek model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated text
        """
        # Mock implementation - replace with actual DeepSeek model call
        if "다음 단어" in prompt or "next word" in prompt:
            # Simple next word prediction simulation
            if "수학" in prompt:
                return "방정식"
            elif "프로그래밍" in prompt:
                return "코드"
            elif "날씨" in prompt:
                return "좋다"
            else:
                return "것이다"
        
        return "생성된 텍스트"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": "DeepSeek-R1-Distill-Llama-70B",
            "model_path": self.model_path,
            "option_type": "Option 2 - Pure Original TOW",
            "language": self.language,
            "features": [
                "token_classification",
                "cross_lingual_thoughts", 
                "data_augmentation",
                "english_only_tow"
            ]
        }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input data from file"""
    data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                for line in f:
                    data.append(json.loads(line.strip()))
            else:
                data = json.load(f)
        
        logger.info(f"Loaded {len(data)} entries from {input_file}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return []


def generate_option2_tow_dataset(
    input_data: List[Dict[str, Any]], 
    adapter: DeepSeekTOWAdapter,
    config: TOWConfig
) -> List[TOWEntry]:
    """
    Generate Option 2 TOW dataset with classification and cross-lingual thoughts.
    
    Args:
        input_data: Input data entries
        adapter: DeepSeek TOW adapter
        config: TOW configuration
        
    Returns:
        List of TOW entries
    """
    tow_entries = []
    
    for i, entry in enumerate(input_data):
        try:
            # Extract text from entry
            text = entry.get('text', '') or entry.get('content', '') or entry.get('sentence', '')
            if not text:
                continue
            
            logger.info(f"Processing entry {i+1}/{len(input_data)}: {text[:50]}...")
            
            # Tokenize for word-level processing
            words = text.split()
            if len(words) < 3:  # Need minimum context
                continue
            
            # Generate TOW entries for different positions
            for pos in range(1, min(len(words), 5)):  # Limit positions for demo
                context_text = " ".join(words[:pos])
                actual_word = words[pos]
                
                # Generate prediction (mock for demo)
                predicted_word = adapter.generate(f"다음 단어 예측: {context_text}")
                predicted_word = predicted_word.split()[0] if predicted_word.split() else actual_word
                
                # Classify the token
                classification_context = ClassificationContext(
                    preceding_text=context_text,
                    predicted_token=predicted_word,
                    actual_token=actual_word,
                    language=adapter.language
                )
                
                classification_result = adapter.token_classifier.classify_token(classification_context)
                
                # Generate cross-lingual English thought
                cross_lingual_context = CrossLingualContext(
                    source_text=context_text,
                    source_language=adapter.language,
                    target_word=actual_word,
                    predicted_word=predicted_word,
                    reasoning_type=classification_result.category.value
                )
                
                english_tow = adapter.cross_lingual_system.generate_english_tow(cross_lingual_context)
                # Extra safety: enforce and sanitize
                english_tow = sanitize_tow_token(english_tow)
                
                # Create TOW entry
                tow_entry = TOWEntry(
                    original_text=text,
                    context=context_text,
                    predicted_token=predicted_word,
                    actual_token=actual_word,
                    thought_token=english_tow,  # Always English
                    category=classification_result.category.value,
                    confidence=classification_result.confidence,
                    language=adapter.language,
                    domain=_detect_domain(text),
                    metadata={
                        "position": pos,
                        "total_words": len(words),
                        "reasoning": classification_result.reasoning,
                        "similarity_score": classification_result.similarity_score,
                        "source_entry_id": i
                    }
                )
                
                tow_entries.append(tow_entry)
                
                logger.debug(f"Generated TOW: {english_tow} (category: {classification_result.category.value})")
        
        except Exception as e:
            logger.warning(f"Failed to process entry {i}: {e}")
            continue
    
    logger.info(f"Generated {len(tow_entries)} TOW entries")
    return tow_entries


def save_tow_dataset(tow_entries: List[TOWEntry], output_file: str):
    """Save TOW dataset to file"""
    try:
        # Convert to dictionaries
        data = []
        for entry in tow_entries:
            data.append({
                "original_text": entry.original_text,
                "context": entry.context,
                "predicted_token": entry.predicted_token,
                "actual_token": entry.actual_token,
                "thought_token": sanitize_tow_token(entry.thought_token),  # English TOW
                "category": entry.category,
                "confidence": entry.confidence,
                "language": entry.language,
                "domain": entry.domain,
                "metadata": entry.metadata
            })
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} TOW entries to {output_file}")
        
        # Generate summary statistics
        summary_file = output_file.replace('.jsonl', '_summary.json')
        category_counts = {}
        language_counts = {}
        domain_counts = {}
        
        for entry in data:
            category_counts[entry['category']] = category_counts.get(entry['category'], 0) + 1
            language_counts[entry['language']] = language_counts.get(entry['language'], 0) + 1
            domain_counts[entry['domain']] = domain_counts.get(entry['domain'], 0) + 1
        
        summary = {
            "total_entries": len(data),
            "option_type": "Option 2 - Pure Original TOW",
            "tow_format": "<ToW>English reasoning</ToW>",
            "features": [
                "token_classification",
                "cross_lingual_thoughts",
                "english_only_tow",
                "data_augmentation"
            ],
            "statistics": {
                "category_distribution": category_counts,
                "language_distribution": language_counts,
                "domain_distribution": domain_counts,
                "average_confidence": sum(entry['confidence'] for entry in data) / len(data) if data else 0
            },
            "sample_entries": data[:3] if data else []  # Show first 3 as examples
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save TOW dataset: {e}")


def _detect_domain(text: str) -> str:
    """Simple domain detection"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["수학", "방정식", "계산", "공식", "math"]):
        return "mathematics"
    elif any(word in text_lower for word in ["프로그래밍", "코드", "함수", "programming", "code"]):
        return "programming"
    elif any(word in text_lower for word in ["실험", "연구", "과학", "science", "research"]):
        return "science"
    elif any(word in text_lower for word in ["요리", "음식", "레시피", "cooking", "food"]):
        return "cooking"
    else:
        return "general"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Option 2 TOW Dataset Generator")
    parser.add_argument("--input_file", required=True, help="Input data file (JSON/JSONL)")
    parser.add_argument("--output_file", required=True, help="Output TOW dataset file")
    parser.add_argument("--language", default="ko", help="Source language (default: ko)")
    parser.add_argument("--model_path", help="Path to DeepSeek model (optional)")
    parser.add_argument("--config", choices=["development", "production"], 
                       default="development", help="Configuration preset")
    parser.add_argument("--max_entries", type=int, help="Maximum entries to process")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting Option 2 TOW Dataset Generation")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info(f"Language: {args.language}")
    
    try:
        # Load configuration
        if args.config == "production":
            config = get_production_config()
        else:
            config = get_development_config()
        
        config.language = args.language
        config.validate()
        
        logger.info(f"Using {args.config} configuration")
        
        # Initialize adapter
        adapter = DeepSeekTOWAdapter(
            model_path=args.model_path,
            language=args.language
        )
        
        logger.info(f"Model info: {adapter.get_model_info()}")
        
        # Load input data
        input_data = load_input_data(args.input_file)
        if not input_data:
            logger.error("No input data loaded")
            return 1
        
        # Limit entries if specified
        if args.max_entries:
            input_data = input_data[:args.max_entries]
            logger.info(f"Limited to {len(input_data)} entries")
        
        # Generate TOW dataset
        logger.info("🔄 Generating Option 2 TOW dataset...")
        tow_entries = generate_option2_tow_dataset(input_data, adapter, config)
        
        if not tow_entries:
            logger.error("No TOW entries generated")
            return 1
        
        # Save dataset
        logger.info("💾 Saving TOW dataset...")
        save_tow_dataset(tow_entries, args.output_file)
        
        # Print statistics
        logger.info("📊 Generation Statistics:")
        category_stats = {}
        for entry in tow_entries:
            category_stats[entry.category] = category_stats.get(entry.category, 0) + 1
        
        for category, count in category_stats.items():
            percentage = (count / len(tow_entries)) * 100
            logger.info(f"  • {category}: {count} ({percentage:.1f}%)")
        
        avg_confidence = sum(entry.confidence for entry in tow_entries) / len(tow_entries)
        logger.info(f"  • Average confidence: {avg_confidence:.3f}")
        
        logger.info("✅ Option 2 TOW dataset generation completed successfully!")
        logger.info(f"📁 Dataset saved to: {args.output_file}")
        logger.info(f"📊 Generated {len(tow_entries)} TOW entries with English-only thoughts")
        
        return 0
        
    except Exception as e:
        logger.error(f"TOW generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
