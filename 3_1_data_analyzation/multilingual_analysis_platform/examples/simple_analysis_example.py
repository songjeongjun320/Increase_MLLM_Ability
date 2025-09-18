"""
Simple Analysis Example

Demonstrates basic usage of the Multilingual Language Model Analysis Platform.
This example shows how to perform sentence embedding analysis, attention analysis,
and confidence analysis programmatically.
"""

import sys
import os
from pathlib import Path

# Add the platform to the Python path
platform_dir = Path(__file__).parent.parent
sys.path.insert(0, str(platform_dir))

import numpy as np
import matplotlib.pyplot as plt
from core.sentence_embedding import SentenceEmbeddingAnalyzer
from core.attention_analysis import AttentionAnalyzer
from core.confidence_analysis import ConfidenceAnalyzer
from visualization.embedding_plots import EmbeddingVisualizer
from visualization.attention_plots import AttentionVisualizer
from visualization.confidence_plots import ConfidenceVisualizer


def main():
    """Main example function."""
    print("🌍 Multilingual Language Model Analysis Platform - Example")
    print("=" * 60)

    # Sample data: English-Korean sentence pairs
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
        "Artificial intelligence is transforming the world.",
        "인공지능이 세상을 변화시키고 있습니다.",
        "Machine learning enables computers to learn from data.",
        "기계학습은 컴퓨터가 데이터로부터 학습할 수 있게 합니다.",
        "Natural language processing analyzes human language.",
        "자연어처리는 인간의 언어를 분석합니다."
    ]

    sample_languages = ['en', 'ko', 'en', 'ko', 'en', 'ko', 'en', 'ko']

    print(f"📝 Analyzing {len(sample_texts)} texts in {len(set(sample_languages))} languages")

    # Initialize analyzers
    print("\n🔧 Initializing analyzers...")
    embedding_analyzer = SentenceEmbeddingAnalyzer()
    attention_analyzer = AttentionAnalyzer()
    confidence_analyzer = ConfidenceAnalyzer()

    # Initialize visualizers
    embedding_visualizer = EmbeddingVisualizer()
    attention_visualizer = AttentionVisualizer()
    confidence_visualizer = ConfidenceVisualizer()

    # 1. Sentence Embedding Analysis
    print("\n📊 Running sentence embedding analysis...")
    try:
        embedding_result = embedding_analyzer.generate_embeddings(
            texts=sample_texts,
            languages=sample_languages
        )

        print(f"   ✅ Generated embeddings: {embedding_result['embedding_dim']}D")
        print(f"   ✅ Similarity matrix: {embedding_result['similarity_matrix'].shape}")

        # Analyze cross-language similarities
        english_texts = [sample_texts[i] for i in range(0, len(sample_texts), 2)]
        korean_texts = [sample_texts[i] for i in range(1, len(sample_texts), 2)]

        comparison_result = embedding_analyzer.compare_sentence_pairs(
            sentences_en=english_texts,
            sentences_target=korean_texts,
            target_language='ko'
        )

        print(f"   ✅ Cross-language accuracy: {comparison_result['accuracy']:.3f}")
        print(f"   ✅ Mean pair similarity: {comparison_result['mean_pair_similarity']:.3f}")

        # Generate visualization
        try:
            fig = embedding_visualizer.plot_embeddings_2d(
                embeddings=embedding_result['embeddings'],
                languages=sample_languages,
                texts=sample_texts,
                method='umap',
                interactive=False,
                title="Sample Embedding Analysis"
            )

            # Save the plot
            output_dir = platform_dir / "outputs" / "examples"
            output_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_dir / "embedding_analysis.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved embedding plot to: {output_dir / 'embedding_analysis.png'}")
            plt.close()

        except Exception as e:
            print(f"   ⚠️ Visualization creation failed: {e}")

    except Exception as e:
        print(f"   ❌ Embedding analysis failed: {e}")

    # 2. Attention Analysis (using first text as example)
    print("\n🔍 Running attention analysis...")
    try:
        # Use a simple model for demonstration
        model_name = "bert-base-multilingual-cased"
        sample_text = sample_texts[0]

        attention_result = attention_analyzer.extract_attention_weights(
            model_name_or_path=model_name,
            text=sample_text,
            model_type='base'
        )

        print(f"   ✅ Extracted attention weights for {attention_result['num_layers']} layers")
        print(f"   ✅ Number of attention heads: {attention_result['num_heads']}")
        print(f"   ✅ Sequence length: {attention_result['sequence_length']}")

        # Analyze attention patterns
        pattern_analysis = attention_analyzer.analyze_attention_patterns(
            attention_result
        )

        print(f"   ✅ Analyzed patterns: {list(pattern_analysis.keys())}")

        # Generate attention visualization
        try:
            fig = attention_visualizer.plot_attention_heatmap(
                attention_result=attention_result,
                layer_idx=6,  # Middle layer
                head_idx=None,  # Average across heads
                interactive=False,
                title="Sample Attention Analysis"
            )

            plt.savefig(output_dir / "attention_analysis.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved attention plot to: {output_dir / 'attention_analysis.png'}")
            plt.close()

        except Exception as e:
            print(f"   ⚠️ Attention visualization failed: {e}")

    except Exception as e:
        print(f"   ❌ Attention analysis failed: {e}")

    # 3. Confidence Analysis
    print("\n📈 Running confidence analysis...")
    try:
        confidence_result = confidence_analyzer.analyze_prediction_confidence(
            model_name_or_path=model_name,
            text=sample_text,
            model_type='base'
        )

        print(f"   ✅ Analyzed confidence for {confidence_result['sequence_length']} tokens")

        measures = confidence_result['confidence_measures']
        if 'entropy' in measures:
            entropy_data = measures['entropy']
            print(f"   ✅ Mean entropy: {entropy_data['mean_entropy']:.3f}")
            print(f"   ✅ Uncertainty level: {entropy_data['uncertainty_classification']}")

        if 'perplexity' in measures:
            perplexity_data = measures['perplexity']
            print(f"   ✅ Overall perplexity: {perplexity_data['overall_perplexity']:.2f}")

        # Generate confidence visualization
        try:
            fig = confidence_visualizer.plot_entropy_by_position(
                confidence_result=confidence_result,
                interactive=False,
                title="Sample Confidence Analysis"
            )

            plt.savefig(output_dir / "confidence_analysis.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved confidence plot to: {output_dir / 'confidence_analysis.png'}")
            plt.close()

        except Exception as e:
            print(f"   ⚠️ Confidence visualization failed: {e}")

    except Exception as e:
        print(f"   ❌ Confidence analysis failed: {e}")

    # 4. Language-specific Analysis
    print("\n🌐 Running language-specific analysis...")
    try:
        # Analyze similarities within and across languages
        similarity_analysis = embedding_analyzer.analyze_semantic_similarity(
            texts=sample_texts,
            languages=sample_languages,
            similarity_threshold=0.8
        )

        within_lang = similarity_analysis['within_language_similarities']
        cross_lang = similarity_analysis['cross_language_similarities']

        print("   ✅ Within-language similarities:")
        for lang, stats in within_lang.items():
            print(f"      {lang}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        print("   ✅ Cross-language similarities:")
        for pair, stats in cross_lang.items():
            print(f"      {pair}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        similar_pairs = similarity_analysis['similar_pairs']
        print(f"   ✅ Found {len(similar_pairs)} highly similar pairs (threshold: 0.8)")

    except Exception as e:
        print(f"   ❌ Language-specific analysis failed: {e}")

    # Summary
    print("\n📋 Analysis Summary")
    print("=" * 60)
    print("✅ Sentence embedding analysis completed")
    print("✅ Attention pattern analysis completed")
    print("✅ Prediction confidence analysis completed")
    print("✅ Cross-language comparison completed")
    print(f"📁 Results saved to: {output_dir}")
    print("\n💡 Next Steps:")
    print("   • Run the full dashboard: python run_dashboard.py")
    print("   • Explore interactive visualizations")
    print("   • Compare your own models")
    print("   • Upload your own datasets")

    # Clean up
    embedding_analyzer.clear_cache()
    attention_analyzer.clear_cache()
    confidence_analyzer.clear_cache()

    print("\n🎉 Example completed successfully!")


if __name__ == "__main__":
    main()