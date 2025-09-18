#!/usr/bin/env python3
"""
Terminal-based Model Comparison Analysis

Simple script to compare two models without browser interface.

module load cuda-12.6.1-gcc-12.1.0

"""

import sys
import os
from pathlib import Path

# Add the platform to the Python path
platform_dir = Path(__file__).parent
sys.path.insert(0, str(platform_dir))

import numpy as np
from core.sentence_embedding import SentenceEmbeddingAnalyzer
from core.attention_analysis import AttentionAnalyzer
from core.confidence_analysis import ConfidenceAnalyzer
from utils.comparison_utils import ModelComparisonSuite

# ============================================================================
# 🔧 USER CONFIGURATION - 여기서 설정하세요!
# ============================================================================

# 모델 경로 설정
BASE_MODEL_PATH = "bert-base-multilingual-cased"  # 베이스 모델
TRAINING_MODEL_PATH = "/path/to/your/trained/model"  # 훈련된 모델 경로

# 분석할 문장들 (영어-한국어 쌍)
TEST_SENTENCES = [
    ("Hello world", "안녕하세요"),
    ("How are you?", "어떻게 지내세요?"),
    ("Thank you very much", "정말 감사합니다"),
    ("I love machine learning", "나는 기계학습을 좋아합니다"),
    ("The weather is beautiful today", "오늘 날씨가 정말 좋습니다"),
    ("Artificial intelligence is amazing", "인공지능은 놀랍습니다")
]

# 분석 옵션
ENABLE_ATTENTION_ANALYSIS = True
ENABLE_CONFIDENCE_ANALYSIS = True
SAVE_RESULTS = True

# ============================================================================

def print_header():
    """Print analysis header."""
    print("🌍 Multilingual Model Comparison Analysis")
    print("=" * 60)
    print(f"📊 Base Model: {BASE_MODEL_PATH}")
    print(f"🎯 Training Model: {TRAINING_MODEL_PATH}")
    print(f"📝 Test Sentences: {len(TEST_SENTENCES)} pairs")
    print("=" * 60)

def analyze_embedding_differences():
    """Analyze embedding differences between models."""
    print("\n📊 Running Sentence Embedding Analysis...")

    try:
        # Initialize comparison suite
        comparison_suite = ModelComparisonSuite()

        # Run comparison
        results = comparison_suite.compare_base_vs_training_models(
            base_model_path=BASE_MODEL_PATH,
            training_model_path=TRAINING_MODEL_PATH,
            test_sentences=TEST_SENTENCES
        )

        # Print results
        print("\n✅ Embedding Analysis Results:")
        print("-" * 40)

        if 'cross_language_performance' in results:
            perf = results['cross_language_performance']
            print(f"   Base Model Accuracy: {perf['base_accuracy']:.3f}")
            print(f"   Training Model Accuracy: {perf['training_accuracy']:.3f}")
            print(f"   Improvement: {perf['improvement']:.3f} ({perf['improvement_percentage']:.1f}%)")

        if 'pair_similarities' in results:
            print(f"\n📝 Sentence Pair Similarities:")
            pairs = results['pair_similarities']
            for i, (en, ko) in enumerate(TEST_SENTENCES):
                if i < len(pairs['base_similarities']) and i < len(pairs['training_similarities']):
                    base_sim = pairs['base_similarities'][i]
                    train_sim = pairs['training_similarities'][i]
                    improvement = train_sim - base_sim
                    print(f"   \"{en}\" ↔ \"{ko}\"")
                    print(f"     Base: {base_sim:.3f} → Training: {train_sim:.3f} ({improvement:+.3f})")

        return results

    except Exception as e:
        print(f"   ❌ Embedding analysis failed: {e}")
        return None

def analyze_attention_differences():
    """Analyze attention pattern differences."""
    if not ENABLE_ATTENTION_ANALYSIS:
        return None

    print("\n🔍 Running Attention Analysis...")

    try:
        attention_analyzer = AttentionAnalyzer()

        # Analyze first sentence as example
        sample_text = TEST_SENTENCES[0][0]  # First English sentence

        # Base model attention
        base_attention = attention_analyzer.extract_attention_weights(
            model_name_or_path=BASE_MODEL_PATH,
            text=sample_text,
            model_type='base'
        )

        print(f"   ✅ Base Model Attention: {base_attention['num_layers']} layers, {base_attention['num_heads']} heads")

        # Training model attention (if path exists)
        if os.path.exists(TRAINING_MODEL_PATH):
            train_attention = attention_analyzer.extract_attention_weights(
                model_name_or_path=TRAINING_MODEL_PATH,
                text=sample_text,
                model_type='trained'
            )
            print(f"   ✅ Training Model Attention: {train_attention['num_layers']} layers, {train_attention['num_heads']} heads")

            # Compare patterns
            comparison = attention_analyzer.compare_attention_patterns(
                base_attention, train_attention
            )
            print(f"   📊 Attention Similarity: {comparison.get('overall_similarity', 'N/A')}")
        else:
            print(f"   ⚠️ Training model path not found: {TRAINING_MODEL_PATH}")

        return base_attention

    except Exception as e:
        print(f"   ❌ Attention analysis failed: {e}")
        return None

def analyze_confidence_differences():
    """Analyze confidence differences."""
    if not ENABLE_CONFIDENCE_ANALYSIS:
        return None

    print("\n📈 Running Confidence Analysis...")

    try:
        confidence_analyzer = ConfidenceAnalyzer()

        # Analyze first sentence
        sample_text = TEST_SENTENCES[0][0]

        # Base model confidence
        base_confidence = confidence_analyzer.analyze_prediction_confidence(
            model_name_or_path=BASE_MODEL_PATH,
            text=sample_text,
            model_type='base'
        )

        print(f"   ✅ Base Model Confidence analyzed for {base_confidence['sequence_length']} tokens")

        if 'confidence_measures' in base_confidence:
            measures = base_confidence['confidence_measures']
            if 'entropy' in measures:
                entropy_data = measures['entropy']
                print(f"     Mean Entropy: {entropy_data['mean_entropy']:.3f}")
                print(f"     Uncertainty: {entropy_data['uncertainty_classification']}")

        # Training model confidence (if exists)
        if os.path.exists(TRAINING_MODEL_PATH):
            train_confidence = confidence_analyzer.analyze_prediction_confidence(
                model_name_or_path=TRAINING_MODEL_PATH,
                text=sample_text,
                model_type='trained'
            )
            print(f"   ✅ Training Model Confidence analyzed")

            if 'confidence_measures' in train_confidence:
                measures = train_confidence['confidence_measures']
                if 'entropy' in measures:
                    entropy_data = measures['entropy']
                    print(f"     Mean Entropy: {entropy_data['mean_entropy']:.3f}")
                    print(f"     Uncertainty: {entropy_data['uncertainty_classification']}")
        else:
            print(f"   ⚠️ Training model path not found: {TRAINING_MODEL_PATH}")

        return base_confidence

    except Exception as e:
        print(f"   ❌ Confidence analysis failed: {e}")
        return None

def save_results_to_file(embedding_results, attention_results, confidence_results):
    """Save analysis results to file."""
    if not SAVE_RESULTS:
        return

    print("\n💾 Saving Results...")

    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embedding results
        if embedding_results:
            import json
            with open(output_dir / "embedding_comparison.json", "w", encoding="utf-8") as f:
                # Convert numpy arrays to lists for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    return obj

                json.dump(embedding_results, f, ensure_ascii=False, indent=2, default=convert_numpy)
            print(f"   ✅ Embedding results saved to: {output_dir / 'embedding_comparison.json'}")

        # Save summary
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Multilingual Model Comparison Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Base Model: {BASE_MODEL_PATH}\n")
            f.write(f"Training Model: {TRAINING_MODEL_PATH}\n")
            f.write(f"Test Sentences: {len(TEST_SENTENCES)} pairs\n\n")

            if embedding_results and 'cross_language_performance' in embedding_results:
                perf = embedding_results['cross_language_performance']
                f.write(f"Cross-Language Performance:\n")
                f.write(f"  Base Accuracy: {perf['base_accuracy']:.3f}\n")
                f.write(f"  Training Accuracy: {perf['training_accuracy']:.3f}\n")
                f.write(f"  Improvement: {perf['improvement']:.3f} ({perf['improvement_percentage']:.1f}%)\n\n")

            f.write("Test Sentence Pairs:\n")
            for i, (en, ko) in enumerate(TEST_SENTENCES, 1):
                f.write(f"  {i}. \"{en}\" ↔ \"{ko}\"\n")

        print(f"   ✅ Summary saved to: {summary_file}")

    except Exception as e:
        print(f"   ❌ Failed to save results: {e}")

def main():
    """Main analysis function."""
    print_header()

    # Check if training model exists
    if not os.path.exists(TRAINING_MODEL_PATH) and TRAINING_MODEL_PATH != "/path/to/your/trained/model":
        print(f"\n⚠️ Warning: Training model path not found: {TRAINING_MODEL_PATH}")
        print("   Analysis will proceed with base model only.\n")

    # Run analyses
    embedding_results = analyze_embedding_differences()
    attention_results = analyze_attention_differences()
    confidence_results = analyze_confidence_differences()

    # Save results
    save_results_to_file(embedding_results, attention_results, confidence_results)

    # Summary
    print("\n📋 Analysis Summary")
    print("=" * 40)
    print("✅ Sentence embedding comparison completed")
    if ENABLE_ATTENTION_ANALYSIS:
        print("✅ Attention pattern analysis completed")
    if ENABLE_CONFIDENCE_ANALYSIS:
        print("✅ Confidence analysis completed")
    if SAVE_RESULTS:
        print("✅ Results saved to outputs/terminal_analysis/")

    print("\n🎉 Terminal analysis completed successfully!")

if __name__ == "__main__":
    main()