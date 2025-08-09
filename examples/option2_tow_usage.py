#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Option 2: Pure Original TOW Implementation - Usage Example
=========================================================

This script demonstrates how to use the Option 2 TOW system which implements:
- Data Augmentation Pipeline for training corpus enhancement
- Token Classification (trivial/exact/soft/unpredictable) 
- Cross-lingual TOW with English-only thoughts
- Training dataset generation with proper <ToW> formatting

Example Usage:
    python examples/option2_tow_usage.py --input_file data.txt --output_file tow_dataset.jsonl
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tow_architecture.core.thought_processor import ThoughtTokenProcessor
from tow_architecture.core.token_classifier import TokenClassifier, ClassificationContext
from tow_architecture.core.cross_lingual_tow import CrossLingualTOWSystem, CrossLingualContext
from tow_architecture.data_augmentation.pipeline import TOWDataAugmentationPipeline
from tow_architecture.models.base_adapter import BaseModelAdapter
from tow_architecture.utils.config import get_development_config, get_production_config
from tow_architecture.utils.text_utils import sanitize_tow_token
from tow_architecture.utils.logger import get_logger

logger = get_logger(__name__)


class MockModelAdapter(BaseModelAdapter):
    """
    Mock model adapter for demonstration purposes.
    In production, replace with actual model adapters (DeepSeek, Llama, Qwen).
    """
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, **kwargs) -> str:
        """Mock generation - in production use actual model"""
        # Simple mock responses for demonstration
        if "수학" in prompt or "math" in prompt:
            return "equation calculation formula"
        elif "프로그램" in prompt or "program" in prompt:
            return "code function algorithm"
        elif "요리" in prompt or "cooking" in prompt:
            return "ingredient recipe method"
        else:
            return "analysis context reasoning"
    
    def get_model_info(self) -> dict:
        return {"model_name": "MockModel", "version": "1.0"}


def demonstrate_token_classification():
    """Demonstrate token classification functionality"""
    print("\n" + "="*60)
    print("🏷️  Token Classification Demonstration")
    print("="*60)
    
    classifier = TokenClassifier(language="ko")
    
    # Test cases for different categories
    test_cases = [
        # (context, predicted, actual) tuples
        ("오늘 날씨가 정말", "좋다", "좋네요"),      # Soft consistent
        ("안녕하세요. 저", "는", "는"),              # Exact match
        ("파이썬에서 함수를 정의할 때", "def", "def"),  # Exact match
        ("그는 어제", "학교에", "병원에"),            # Unpredictable
        ("나는", "너를", "을"),                      # Trivial (particle)
    ]
    
    for context, predicted, actual in test_cases:
        classification_context = ClassificationContext(
            preceding_text=context,
            predicted_token=predicted,
            actual_token=actual,
            language="ko"
        )
        
        result = classifier.classify_token(classification_context)
        
        print(f"\n📝 Context: {context}")
        print(f"🎯 Predicted: '{predicted}' → Actual: '{actual}'")
        print(f"🏷️  Category: {result.category.value}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        print(f"💭 Reasoning: {result.reasoning}")


def demonstrate_cross_lingual_tow():
    """Demonstrate cross-lingual TOW generation"""
    print("\n" + "="*60)
    print("🌍 Cross-lingual TOW Demonstration")
    print("="*60)
    
    model_adapter = MockModelAdapter()
    cross_lingual_system = CrossLingualTOWSystem(model_adapter)
    
    # Test cases for different languages
    test_cases = [
        {
            "text": "오늘 날씨가 정말 좋아서 산책을",
            "language": "ko",
            "predicted": "했다",
            "actual": "하고싶다",
            "category": "soft_consistent"
        },
        {
            "text": "파이썬 프로그래밍에서 가장 중요한 것은",
            "language": "ko", 
            "predicted": "문법",
            "actual": "로직",
            "category": "unpredictable"
        },
        {
            "text": "今天天气很好，我想去",
            "language": "zh",
            "predicted": "公园",
            "actual": "散步",
            "category": "soft_consistent"
        }
    ]
    
    for case in test_cases:
        context = CrossLingualContext(
            source_text=case["text"],
            source_language=case["language"],
            target_word=case["actual"],
            predicted_word=case["predicted"],
            reasoning_type=case["category"]
        )
        
        tow_token = cross_lingual_system.generate_english_tow(context)
        
        print(f"\n📝 Source ({case['language']}): {case['text']}")
        print(f"🎯 Predicted: '{case['predicted']}' → Actual: '{case['actual']}'")
        print(f"🏷️  Category: {case['category']}")
        print(f"🧠 English TOW: {tow_token}")


def demonstrate_thought_processor():
    """Demonstrate enhanced thought processor"""
    print("\n" + "="*60)
    print("🧠 Thought Processor Demonstration (Option 2)")
    print("="*60)
    
    model_adapter = MockModelAdapter()
    processor = ThoughtTokenProcessor(model_adapter, language="ko")
    
    # Test texts
    test_texts = [
        "수학에서 방정식 2x + 5 = 17을 풀면",
        "파이썬에서 리스트를 생성하려면 대괄호를 사용하여",
        "오늘 날씨가 맑아서 공원에서 산책을"
    ]
    
    for text in test_texts:
        print(f"\n📝 Input Text: {text}")
        
        # Generate standard thoughts
        thoughts = processor.generate_thoughts(text, max_thoughts=3)
        print(f"🧠 Generated TOW Tokens:")
        for i, thought in enumerate(thoughts, 1):
            print(f"   {i}. {thought}")
        
        # Generate classified thoughts (simulation)
        words = text.split()
        if len(words) > 2:
            predicted_words = ["다음단어"]  # Mock prediction
            actual_words = [words[-1]]      # Last word as actual
            
            classified_results = processor.generate_classified_thoughts(
                text=" ".join(words[:-1]),
                predicted_words=predicted_words,
                actual_words=actual_words
            )
            
            if classified_results:
                result = classified_results[0]
                print(f"🏷️  Classification: {result['category']}")
                print(f"📊 Confidence: {result['confidence']:.3f}")
                print(f"🌍 Cross-lingual TOW: {result['tow_token']}")


def demonstrate_data_augmentation_pipeline():
    """Demonstrate data augmentation pipeline"""
    print("\n" + "="*60)
    print("🔄 Data Augmentation Pipeline Demonstration")
    print("="*60)
    
    model_adapter = MockModelAdapter()
    pipeline = TOWDataAugmentationPipeline(model_adapter, language="ko")
    
    # Sample input data
    input_texts = [
        "한국의 전통 음식 중 김치는 발효 음식으로 건강에 좋습니다.",
        "프로그래밍에서 알고리즘은 문제를 해결하는 절차적 방법입니다.",
        "기계학습은 데이터에서 패턴을 찾아 예측하는 기술입니다.",
        "오늘 날씨가 좋아서 친구들과 공원에서 피크닉을 했습니다.",
        "수학에서 미적분은 변화율을 다루는 중요한 분야입니다."
    ]
    
    print(f"📊 Processing {len(input_texts)} input texts...")
    
    # Process corpus (small batch for demo)
    output_path = "demo_tow_dataset.jsonl"
    stats = pipeline.process_corpus(
        input_data=input_texts,
        output_path=output_path,
        batch_size=5,
        max_workers=2
    )
    
    print(f"\n📈 Processing Statistics:")
    print(f"   • Total processed: {stats.total_processed}")
    print(f"   • Successful generations: {stats.successful_generations}")
    print(f"   • Failed generations: {stats.failed_generations}")
    print(f"   • Processing time: {stats.processing_time:.2f}s")
    print(f"   • Average confidence: {stats.average_confidence:.3f}")
    
    print(f"\n📊 Category Distribution:")
    for category, count in stats.category_counts.items():
        print(f"   • {category}: {count}")
    
    if os.path.exists(output_path):
        print(f"\n✅ Dataset saved to: {output_path}")
        
        # Show sample entries
        import json
        print(f"\n📋 Sample TOW Entries:")
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 entries
                    break
                entry = json.loads(line)
                print(f"\n   Entry {i+1}:")
                print(f"   • Context: {entry['context']}")
                print(f"   • Actual word: {entry['actual_token']}")
                print(f"   • TOW token: {sanitize_tow_token(entry['thought_token'])}")
                print(f"   • Category: {entry['category']}")
    
    # Cleanup demo file
    if os.path.exists(output_path):
        os.remove(output_path)


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="Option 2 TOW System Demonstration")
    parser.add_argument("--demo", choices=["all", "classification", "cross_lingual", "thoughts", "pipeline"],
                       default="all", help="Which demonstration to run")
    parser.add_argument("--input_file", type=str, help="Input file for processing")
    parser.add_argument("--output_file", type=str, help="Output file for TOW dataset")
    parser.add_argument("--language", type=str, default="ko", help="Source language")
    parser.add_argument("--config", choices=["development", "production"], default="development",
                       help="Configuration preset to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Option 2: Pure Original TOW Implementation")
    print("=" * 60)
    print("Features:")
    print("• Data Augmentation Pipeline for training corpus enhancement")
    print("• Token Classification (trivial/exact/soft/unpredictable)")
    print("• Cross-lingual TOW with English-only thoughts")
    print("• Training dataset generation with <ToW> formatting")
    print("• Token format consistency with original paper")
    
    # Get configuration
    if args.config == "production":
        config = get_production_config()
        print(f"\n🔧 Using production configuration")
    else:
        config = get_development_config()
        print(f"\n🔧 Using development configuration")
    
    config.language = args.language
    config.validate()
    
    try:
        # Run demonstrations
        if args.demo in ["all", "classification"]:
            demonstrate_token_classification()
        
        if args.demo in ["all", "cross_lingual"]:
            demonstrate_cross_lingual_tow()
        
        if args.demo in ["all", "thoughts"]:
            demonstrate_thought_processor()
        
        if args.demo in ["all", "pipeline"]:
            demonstrate_data_augmentation_pipeline()
        
        # Process input file if provided
        if args.input_file and args.output_file:
            print("\n" + "="*60)
            print("📁 Processing Input File")
            print("="*60)
            
            if not os.path.exists(args.input_file):
                print(f"❌ Input file not found: {args.input_file}")
                return
            
            model_adapter = MockModelAdapter()
            pipeline = TOWDataAugmentationPipeline(model_adapter, language=args.language)
            
            print(f"📂 Processing file: {args.input_file}")
            stats = pipeline.process_file(args.input_file, args.output_file)
            
            print(f"\n✅ Processing completed!")
            print(f"📊 Generated {stats.successful_generations} TOW entries")
            print(f"💾 Saved to: {args.output_file}")
        
        print("\n" + "="*60)
        print("🎉 Option 2 TOW Demonstration Completed Successfully!")
        print("="*60)
        print("\nNext Steps:")
        print("• Replace MockModelAdapter with actual model (DeepSeek/Llama/Qwen)")
        print("• Train models using generated TOW datasets")
        print("• Fine-tune classification thresholds based on your data")
        print("• Scale up processing for larger corpora")
        print("• Integrate with your MLOps pipeline")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
