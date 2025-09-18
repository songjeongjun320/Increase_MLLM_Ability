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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import logging
import os

# Suppress warnings and logs
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# Suppress matplotlib font warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from core.sentence_embedding import SentenceEmbeddingAnalyzer
from core.attention_analysis import AttentionAnalyzer
from core.confidence_analysis import ConfidenceAnalyzer
from utils.comparison_utils import ModelComparisonSuite
from visualization.embedding_plots import EmbeddingVisualizer
from visualization.confidence_plots import ConfidenceVisualizer

# ============================================================================
# 🔧 USER CONFIGURATION - 여기서 설정하세요!
# ============================================================================

# 모델 경로 설정
BASE_MODEL_PATH = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt"  # 베이스 모델
TRAINING_MODEL_PATH = "/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-3.2-3b-pt-tow-09_11_2epoch_allenai-merged"

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

def setup_korean_font():
    """Setup Korean font for matplotlib to display Korean text properly."""
    try:
        # Suppress all font-related warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Try to find Korean fonts quietly
            korean_font_names = ['NanumGothic', 'Nanum Gothic', 'NanumBarunGothic', 'Malgun Gothic', 'Gulim', 'Dotum']
            available_fonts = [f.name for f in fm.fontManager.ttflist]

            found_font = None
            for font_name in korean_font_names:
                if font_name in available_fonts:
                    found_font = font_name
                    break

            if found_font:
                plt.rcParams['font.family'] = found_font
                print(f"🔤 Korean font: {found_font}")
            else:
                # Try manual download approach quietly
                try:
                    import urllib.request
                    font_dir = platform_dir / "fonts"
                    font_dir.mkdir(exist_ok=True)
                    font_path = font_dir / "NanumGothic.ttf"

                    if not font_path.exists():
                        print("🔤 Downloading Korean font...")
                        font_url = "https://github.com/naver/nanumfont/raw/master/fonts/NanumGothic.ttf"
                        urllib.request.urlretrieve(font_url, font_path)

                    # Add font to matplotlib quietly
                    fm.fontManager.addfont(str(font_path))
                    plt.rcParams['font.family'] = 'NanumGothic'
                    print("🔤 Korean font: NanumGothic (downloaded)")

                except:
                    # Final fallback - use DejaVu Sans
                    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
                    print("🔤 Korean font: Unicode fallback")

    except:
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

    plt.rcParams['axes.unicode_minus'] = False

def print_header():
    """Print analysis header."""
    print("🌍 Multilingual Model Comparison Analysis")
    print("=" * 60)
    print(f"📊 Base Model: {BASE_MODEL_PATH}")
    print(f"🎯 Training Model: {TRAINING_MODEL_PATH}")
    print(f"📝 Test Sentences: {len(TEST_SENTENCES)} pairs")
    print("=" * 60)
    print("\n📝 Sentence Index Reference:")
    for i, (en, ko) in enumerate(TEST_SENTENCES):
        print(f"   [{i*2}] {en}")
        print(f"   [{i*2+1}] {ko}")
    print("=" * 60)

def save_embedding_visualizations(results):
    """Generate and save embedding visualizations."""
    try:
        # Create output directory
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding analyzer and visualizer
        embedding_analyzer = SentenceEmbeddingAnalyzer()
        visualizer = EmbeddingVisualizer()

        # Prepare text and language data
        texts = []
        languages = []
        for en, ko in TEST_SENTENCES:
            texts.extend([en, ko])
            languages.extend(['en', 'ko'])

        # Generate embeddings
        embedding_result = embedding_analyzer.generate_embeddings(texts, languages)
        embeddings = embedding_result['embeddings']

        # Generate visualizations quietly
        plots_saved = 0

        # PCA
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = visualizer.plot_embeddings_2d(
                    embeddings=embeddings, languages=languages, texts=texts,
                    method='pca', interactive=False, title="PCA - Sentence Embeddings"
                )
                plt.savefig(output_dir / "pca_embeddings.png", dpi=300, bbox_inches='tight')
                plt.close()
                plots_saved += 1
        except:
            pass

        # t-SNE
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = visualizer.plot_embeddings_2d(
                    embeddings=embeddings, languages=languages, texts=texts,
                    method='tsne', interactive=False, title="t-SNE - Sentence Embeddings"
                )
                plt.savefig(output_dir / "tsne_embeddings.png", dpi=300, bbox_inches='tight')
                plt.close()
                plots_saved += 1
        except:
            pass

        # UMAP
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = visualizer.plot_embeddings_2d(
                    embeddings=embeddings, languages=languages, texts=texts,
                    method='umap', interactive=False, title="UMAP - Sentence Embeddings"
                )
                plt.savefig(output_dir / "umap_embeddings.png", dpi=300, bbox_inches='tight')
                plt.close()
                plots_saved += 1
        except:
            pass

        # Similarity heatmap
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                similarity_matrix = embedding_result['similarity_matrix']
                fig = visualizer.plot_similarity_heatmap(
                    similarity_matrix=similarity_matrix, texts=texts, languages=languages,
                    interactive=False, title="Sentence Similarity Heatmap"
                )
                plt.savefig(output_dir / "similarity_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
                plots_saved += 1
        except:
            pass

        print(f"   ✅ Saved {plots_saved}/4 visualizations")

    except Exception as e:
        print(f"   ❌ Visualization generation failed: {e}")
        raise

def save_confidence_visualizations(confidence_result):
    """Generate and save confidence analysis visualizations."""
    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = ConfidenceVisualizer()

        # Generate entropy by position plot
        try:
            fig = visualizer.plot_entropy_by_position(
                confidence_result=confidence_result,
                interactive=False,
                title="Token-wise Confidence Analysis"
            )
            plt.savefig(output_dir / "confidence_entropy.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Confidence plot saved: {output_dir / 'confidence_entropy.png'}")
        except Exception as e:
            print(f"   ⚠️ Entropy plot failed: {e}")

        # Generate uncertainty heatmap if we have token-level data
        try:
            if 'confidence_measures' in confidence_result:
                measures = confidence_result['confidence_measures']
                if 'variance' in measures or 'entropy' in measures:
                    fig = visualizer.plot_confidence_heatmap(
                        confidence_result=confidence_result,
                        interactive=False,
                        title="Confidence Heatmap"
                    )
                    plt.savefig(output_dir / "confidence_heatmap.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"   ✅ Confidence heatmap saved: {output_dir / 'confidence_heatmap.png'}")
        except Exception as e:
            print(f"   ⚠️ Confidence heatmap failed: {e}")

    except Exception as e:
        print(f"   ❌ Confidence visualization failed: {e}")

def save_confidence_comparison(base_result, train_result):
    """Generate and save confidence comparison visualizations."""
    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = ConfidenceVisualizer()

        # Generate side-by-side confidence comparison
        fig = visualizer.plot_confidence_comparison(
            base_result=base_result,
            training_result=train_result,
            interactive=False,
            title="Base vs Training Model Confidence Comparison"
        )
        plt.savefig(output_dir / "confidence_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Confidence comparison saved: {output_dir / 'confidence_comparison.png'}")

    except Exception as e:
        print(f"   ❌ Confidence comparison visualization failed: {e}")

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
                    print(f"   [{i*2}] \"{en}\" ↔ [{i*2+1}] \"{ko}\"")
                    print(f"     Base: {base_sim:.3f} → Training: {train_sim:.3f} ({improvement:+.3f})")

        # Generate and save visualizations
        print(f"\n🎨 Generating Visualizations...")
        try:
            save_embedding_visualizations(results)
        except Exception as e:
            print(f"   ⚠️ Visualization generation failed: {e}")

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
        print(f"   📊 Method: {base_confidence.get('method', 'logit_based')}")

        if 'confidence_measures' in base_confidence:
            measures = base_confidence['confidence_measures']
            if 'entropy' in measures:
                entropy_data = measures['entropy']
                print(f"     Mean Entropy: {entropy_data['mean_entropy']:.3f}")
                print(f"     Uncertainty: {entropy_data['uncertainty_classification']}")
                print(f"     Method: {entropy_data.get('method', 'standard')}")

        # Check if we have embedding-based uncertainty summary
        if 'summary' in base_confidence:
            summary = base_confidence['summary']
            print(f"     Overall Uncertainty: {summary['uncertainty_level']}")
            if 'high_uncertainty_positions' in summary:
                high_positions = summary['high_uncertainty_positions']
                if high_positions:
                    print(f"     High Uncertainty Positions: {high_positions}")

        # Generate confidence visualization
        try:
            save_confidence_visualizations(base_confidence)
        except Exception as e:
            print(f"   ⚠️ Confidence visualization failed: {e}")

        # Training model confidence (if exists)
        train_confidence = None
        if os.path.exists(TRAINING_MODEL_PATH):
            try:
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

                # Generate comparison visualization
                try:
                    save_confidence_comparison(base_confidence, train_confidence)
                except Exception as e:
                    print(f"   ⚠️ Confidence comparison visualization failed: {e}")

            except Exception as e:
                print(f"   ❌ Training model confidence analysis failed: {e}")
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
                    elif isinstance(obj, (np.floating, float)):
                        return float(obj)
                    elif isinstance(obj, (np.integer, int)):
                        return int(obj)
                    elif hasattr(obj, '__dict__'):
                        return str(obj)  # Convert complex objects to string
                    return obj

                # Create a safe copy of results without circular references
                safe_results = {}
                for key, value in embedding_results.items():
                    try:
                        converted = convert_numpy(value)
                        safe_results[key] = converted
                    except (TypeError, ValueError):
                        safe_results[key] = str(value)

                json.dump(safe_results, f, ensure_ascii=False, indent=2, default=str)
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
    # Setup Korean font first
    setup_korean_font()

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