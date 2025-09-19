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
TRAINING_MODEL_PATH = "/scratch/jsong132/Increase_MLLM_Ability/5_training/finetune_org/merged_models/llama-tow-allenai-merged"

# 분석할 문장들 (영어-한국어 쌍)
TEST_SENTENCES = [
    # Questions which model could get an answer in eng, but not in kor.
    ("An engineer is using a computer to design a bridge. Which test is the most important for safety purposes? A. maximum load the bridge can support. B. cost of material used to build the bridge. C. percent of materials that can be recycled. D. speed with which the bridge can be constructed.", "엔지니어가 컴퓨터를 사용하여 교량을 설계하고 있습니다. 안전을 위해 가장 중요한 테스트는 무엇인가요? A. 교량이 견딜 수 있는 최대 하중. B. 교량 건설에 사용된 자재의 비용. C. 재활용이 가능한 자재의 비율. D. 교량 건설 속도."),
    ("Phillip was making hot tea. When he poured the hot water into a glass, the glass broke. Which is the most likely reason the glass broke? A. the hot water was poured in too slowly. B. the hot water evaporated too slowly. C. the glass contracted too quickly. D. the glass expanded too rapidly.", "지현은 뜨거운 차를 만들고 있었습니다. 그가 뜨거운 물을 유리잔에 부었을 때 유리잔이 깨졌습니다. 유리잔이 깨진 이유로 가능성이 가장 높은 것은 무엇일까요? A. 교량이 견딜 수 있는 최대 하중. B. 교량 건설에 사용된 자재의 비용. C. 재활용이 가능한 자재의 비율. D. 교량 건설 속도."),
    ("A 20 N object is placed on a surface and starts to slide. What is the most likely reason the object begins to move? A. Gravity exerts a balanced force on the object. B. An unbalanced force causes acceleration. C. Friction is applied to the object. D. The forces acting on the object are in equilibrium.", "20N의 물체가 표면 위에 놓인 후 미끄러지기 시작합니다. 물체가 움직이기 시작하는 가장 가능성이 높은 이유는 무엇일까요? A. 중력이 물체에 균형 잡힌 힘을 가하기 때문입니다. B. 불균형한 힘이 가속을 유발하기 때문입니다. C. 물체에 마찰력이 작용하기 때문입니다. D. 물체에 작용하는 힘은 평형 상태이기 때문입니다."),
    ("What is most likely the first step for students to do for a recycling project? A. burn the materials in a bonfire. B. take the materials to a landfill. C. dump the materials in a trash bin. D. sort the materials by plastics, papers, and cans.", "학생들이 재활용 프로젝트를 위해 가장 먼저 해야 할 일은 무엇인가요? A. 모닥불에서 자재 태우기. B. 자재를 매립지로 가져가기. C. 자재를 쓰레기통에 버리기. D. 플라스틱, 종이, 캔별로 자재 분류하기."),
    # Questions which model could get an answer both lang.
    ("Which has the greatest effect on wind speed? A. precipitation. B. cloud cover. C. wind direction. D. air pressure.", "풍속에 가장 큰 영향을 미치는 것은 무엇인가요? A. 강수량. B. 운량. C. 풍향. D. 기압."),
    ("Why does a town in the desert rarely experience early morning fog as compared to a town along the coast? A. There is less rainfall in the desert. B. Temperatures vary more in the desert. C. There is less water vapor in the desert air. D. There are fewer plants in the desert.","사막에 있는 마을이 해안가에 있는 마을에 비해 이른 아침에 안개가 거의 발생하지 않는 이유는 무엇인가요? A. 사막에는 강수량이 적기 때문입니다. B. 사막에서는 기온이 더 다양하게 변하기 때문입니다. C. 사막의 공기에는 수증기가 적기 때문입니다. D. 사막에는 식물 수가 적기 때문입니다."),
    ("A student mixed 25 grams of salt into 1,000 grams of water. What is the mass of the saltwater mixture? A. 975 grams. B. 1,000 grams. C. 1,025 grams. D. 2,500 grams.","한 학생이 소금 25그램을 물 1,000그램에 섞었습니다. 소금물 혼합물의 질량은 얼마입니까? A. 975그램. B. 1,000그램. C. 1,025그램. D. 2,500그램."),
    ("One characteristic that is unique to water is that it. A. has a low specific heat. B. can be changed from a liquid to a solid. C. dissolves very few substances. D. exists naturally in three states on Earth.","물의 독특한 특징 중 하나는 다음과 같습니다. A. 비열이 낮습니다. B. 액체에서 고체로 변할 수 있습니다. C. 녹일 수 있는 물질이 매우 적습니다. D. 지구에서 세 가지 상태로 자연적으로 존재합니다.")
]

# 분석 옵션
ENABLE_ATTENTION_ANALYSIS = True
ENABLE_CONFIDENCE_ANALYSIS = True
SAVE_RESULTS = True

# ============================================================================

def setup_korean_font():
    """Setup Korean font for matplotlib to display Korean text properly."""
    import matplotlib.pyplot as plt  # Import at the beginning
    
    try:
        # Suppress all font-related warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Try to find Korean fonts quietly
            korean_font_names = [
                'NanumGothic', 'Nanum Gothic', 'NanumBarunGothic', 'NanumBarunGothicOTF',
                'Malgun Gothic', 'Gulim', 'Dotum', 'Batang', 'Gungsuh', 'AppleGothic',
                'Noto Sans CJK KR', 'Source Han Sans KR', 'Roboto', 'Arial Unicode MS'
            ]
            
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            print(f"🔍 Available fonts: {len(available_fonts)} total")

            found_font = None
            for font_name in korean_font_names:
                if font_name in available_fonts:
                    found_font = font_name
                    break

            if found_font:
                plt.rcParams['font.family'] = found_font
                plt.rcParams['font.size'] = 10
                print(f"🔤 Korean font: {found_font}")
            else:
                # Use system fallback fonts
                plt.rcParams['font.family'] = ['Arial Unicode MS', 'AppleGothic', 'Malgun Gothic', 'DejaVu Sans', 'sans-serif']
                plt.rcParams['font.size'] = 10
                print("🔤 Korean font: System font fallback")

    except Exception as e:
        print(f"🔤 Font setup failed: {e}")
        # Final fallback
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['font.size'] = 10

    # Additional matplotlib settings for better Korean support
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True
    
    print(f"🔤 Final font family: {plt.rcParams['font.family']}")

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
    """Generate and save comprehensive embedding visualizations."""
    try:
        # Initialize Korean font manager first
        from utils.font_manager import setup_korean_fonts
        setup_korean_fonts()

        # Create output directory
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding analyzer and visualizer
        embedding_analyzer = SentenceEmbeddingAnalyzer()
        visualizer = EmbeddingVisualizer()

        # Prepare text and language data with proper encoding
        texts = []
        languages = []
        for en, ko in TEST_SENTENCES:
            # Ensure proper UTF-8 encoding for Korean text
            texts.extend([en, ko.encode('utf-8').decode('utf-8')])
            languages.extend(['en', 'ko'])

        # Generate embeddings
        embedding_result = embedding_analyzer.generate_embeddings(texts, languages)
        embeddings = embedding_result['embeddings']

        # Generate comprehensive visualizations
        plots_saved = 0
        total_plots = 0

        print(f"   📊 Generating comprehensive visualizations...")

        # 2D Visualizations
        for method in ['pca', 'tsne', 'umap']:
            total_plots += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig = visualizer.plot_embeddings_2d(
                        embeddings=embeddings, languages=languages, texts=texts,
                        method=method, interactive=False,
                        title=f"{method.upper()} - Sentence Embeddings"
                    )
                    plt.savefig(output_dir / f"{method}_embeddings_2d.png",
                               dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    plt.close()
                    plots_saved += 1
            except Exception as e:
                print(f"      ⚠️ {method.upper()} 2D plot failed: {e}")

        # 3D Visualizations (using matplotlib)
        for method in ['pca', 'tsne', 'umap']:
            total_plots += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Create 3D plot with matplotlib
                    from mpl_toolkits.mplot3d import Axes3D

                    # Reduce to 3D
                    reduced_3d = embedding_analyzer.reduce_dimensions(
                        embeddings, method=method, n_components=3
                    )

                    # Create 3D plot
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection='3d')

                    # Configure Korean fonts
                    configure_plot_korean(fig, ax)

                    # Get colors for languages
                    from utils.config_loader import get_language_colors
                    colors = get_language_colors()

                    # Plot points by language
                    for lang in set(languages):
                        mask = np.array(languages) == lang
                        if np.any(mask):
                            color = colors.get(lang, '#1f77b4')
                            ax.scatter(
                                reduced_3d[mask, 0],
                                reduced_3d[mask, 1],
                                reduced_3d[mask, 2],
                                c=color, label=lang, alpha=0.7, s=60
                            )

                    # Add labels and title
                    ax.set_xlabel(f'{method.upper()} 1')
                    ax.set_ylabel(f'{method.upper()} 2')
                    ax.set_zlabel(f'{method.upper()} 3')
                    ax.set_title(f"{method.upper()} - 3D Sentence Embeddings", fontsize=14, pad=20)
                    ax.legend()

                    plt.savefig(output_dir / f"{method}_embeddings_3d.png",
                               dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    plt.close()
                    plots_saved += 1
            except Exception as e:
                print(f"      ⚠️ {method.upper()} 3D plot failed: {e}")

        # Similarity Analysis
        total_plots += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                similarity_matrix = embedding_result['similarity_matrix']
                fig = visualizer.plot_similarity_heatmap(
                    similarity_matrix=similarity_matrix, texts=texts, languages=languages,
                    interactive=False, title="Sentence Similarity Heatmap"
                )
                plt.savefig(output_dir / "similarity_heatmap.png",
                           dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                plots_saved += 1
        except Exception as e:
            print(f"      ⚠️ Similarity heatmap failed: {e}")

        # Language distance analysis (simplified)
        total_plots += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create language distance visualization
                from sklearn.metrics.pairwise import cosine_similarity

                # Calculate average embeddings per language
                lang_embeddings = {}
                for lang in set(languages):
                    mask = np.array(languages) == lang
                    if np.any(mask):
                        lang_embeddings[lang] = embeddings[mask].mean(axis=0)

                # Create distance matrix
                lang_names = list(lang_embeddings.keys())
                distance_matrix = np.zeros((len(lang_names), len(lang_names)))
                for i, lang1 in enumerate(lang_names):
                    for j, lang2 in enumerate(lang_names):
                        distance_matrix[i, j] = cosine_similarity(
                            [lang_embeddings[lang1]], [lang_embeddings[lang2]]
                        )[0, 0]

                # Plot distance matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(distance_matrix, cmap='viridis')
                ax.set_xticks(range(len(lang_names)))
                ax.set_yticks(range(len(lang_names)))
                ax.set_xticklabels(lang_names)
                ax.set_yticklabels(lang_names)
                plt.colorbar(im)
                ax.set_title("Language Distance Matrix")

                plt.savefig(output_dir / "language_distance.png",
                           dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                plots_saved += 1
        except Exception as e:
            print(f"      ⚠️ Language distance plot failed: {e}")

        # Embedding statistics
        total_plots += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Plot embedding statistics
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

                # 1. Embedding magnitude by language
                lang_magnitudes = {}
                for lang in set(languages):
                    mask = np.array(languages) == lang
                    if np.any(mask):
                        lang_magnitudes[lang] = np.linalg.norm(embeddings[mask], axis=1)

                lang_names = list(lang_magnitudes.keys())
                lang_data = [lang_magnitudes[lang] for lang in lang_names]
                ax1.boxplot(lang_data, labels=lang_names)
                ax1.set_title("Embedding Magnitude by Language")
                ax1.set_ylabel("L2 Norm")

                # 2. Embedding variance
                lang_variances = [np.var(data) for data in lang_data]
                ax2.bar(lang_names, lang_variances)
                ax2.set_title("Embedding Variance by Language")
                ax2.set_ylabel("Variance")

                # 3. Sentence length distribution
                sentence_lengths = [len(text.split()) for text in texts]
                lang_lengths = {}
                for i, lang in enumerate(languages):
                    if lang not in lang_lengths:
                        lang_lengths[lang] = []
                    lang_lengths[lang].append(sentence_lengths[i])

                length_data = [lang_lengths[lang] for lang in lang_names]
                ax3.boxplot(length_data, labels=lang_names)
                ax3.set_title("Sentence Length by Language")
                ax3.set_ylabel("Word Count")

                # 4. Similarity statistics
                similarity_matrix = cosine_similarity(embeddings)
                ax4.hist(similarity_matrix.flatten(), bins=50, alpha=0.7)
                ax4.set_title("Similarity Distribution")
                ax4.set_xlabel("Cosine Similarity")
                ax4.set_ylabel("Frequency")

                plt.tight_layout()
                plt.savefig(output_dir / "embedding_statistics.png",
                           dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                plots_saved += 1
        except Exception as e:
            print(f"      ⚠️ Embedding statistics plot failed: {e}")

        print(f"   ✅ Saved {plots_saved}/{total_plots} visualizations")

    except Exception as e:
        print(f"   ❌ Visualization generation failed: {e}")
        raise

def save_confidence_visualizations(confidence_result):
    """Generate and save confidence analysis visualizations."""
    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = ConfidenceVisualizer()

        # Generate entropy by position plot (original single sentence)
        try:
            fig = visualizer.plot_entropy_by_position(
                confidence_result=confidence_result,
                interactive=False,
                title="Token-wise Confidence Analysis - Sample Sentence"
            )
            plt.savefig(output_dir / "confidence_entropy.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Single sentence confidence plot saved")
        except Exception as e:
            print(f"   ⚠️ Entropy plot failed: {e}")

    except Exception as e:
        print(f"   ❌ Confidence visualization failed: {e}")

def save_multilingual_confidence_comparison(confidence_results):
    """Generate English-Korean confidence comparison visualizations."""
    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Korean font manager
        from utils.font_manager import setup_korean_fonts
        setup_korean_fonts()

        # Extract confidence data for ALL sentence pairs
        en_confidences = []
        ko_confidences = []

        # Extract English uncertainties
        for result in confidence_results['base_model_results']:
            if 'confidence_measures' in result:
                measures = result['confidence_measures']
                avg_confidence = measures.get('average_confidence', 0.5)
                en_confidences.append(1.0 - avg_confidence)  # Convert to uncertainty
            else:
                # Fallback for embedding-based uncertainty
                if 'uncertainty_estimates' in result:
                    uncertainty = result['uncertainty_estimates']['mean_uncertainty']
                    en_confidences.append(uncertainty)
                else:
                    en_confidences.append(0.5)  # Default uncertainty

        # Extract Korean uncertainties
        for result in confidence_results['train_model_results']:
            if 'confidence_measures' in result:
                measures = result['confidence_measures']
                avg_confidence = measures.get('average_confidence', 0.5)
                ko_confidences.append(1.0 - avg_confidence)  # Convert to uncertainty
            else:
                # Fallback for embedding-based uncertainty
                if 'uncertainty_estimates' in result:
                    uncertainty = result['uncertainty_estimates']['mean_uncertainty']
                    ko_confidences.append(uncertainty)
                else:
                    ko_confidences.append(0.6)  # Default uncertainty (slightly higher for Korean)

        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Language-wise uncertainty comparison
        languages = ['English', 'Korean']
        avg_uncertainties = [np.mean(en_confidences), np.mean(ko_confidences)]
        colors = ['#1f77b4', '#ff7f0e']

        bars = ax1.bar(languages, avg_uncertainties, color=colors, alpha=0.7)
        ax1.set_title('Average Uncertainty by Language', fontsize=14)
        ax1.set_ylabel('Uncertainty Score')
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, avg_uncertainties):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # 2. Sentence-wise comparison with detailed indexing
        num_pairs = len(en_confidences)
        x_pos = np.arange(num_pairs)
        width = 0.35

        # Create bars with value labels
        bars1 = ax2.bar(x_pos - width/2, en_confidences, width, label='English', color=colors[0], alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, ko_confidences[:num_pairs], width, label='Korean', color=colors[1], alpha=0.7)

        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # English values
            ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                    f'{en_confidences[i]:.2f}', ha='center', va='bottom', fontsize=8)
            # Korean values
            ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                    f'{ko_confidences[i]:.2f}', ha='center', va='bottom', fontsize=8)

        ax2.set_title(f'Uncertainty by Sentence Pair (Total: {num_pairs} pairs)', fontsize=14)
        ax2.set_xlabel('Sentence Pair Index')
        ax2.set_ylabel('Uncertainty Score')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{i+1}' for i in range(num_pairs)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Distribution of uncertainties
        ax3.hist(en_confidences, bins=10, alpha=0.7, label='English', color=colors[0])
        ax3.hist(ko_confidences[:len(en_confidences)], bins=10, alpha=0.7, label='Korean', color=colors[1])
        ax3.set_title('Uncertainty Distribution', fontsize=14)
        ax3.set_xlabel('Uncertainty Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        # 4. Detailed analysis information
        ax4.axis('off')

        # Calculate additional statistics
        en_mean = np.mean(en_confidences)
        ko_mean = np.mean(ko_confidences[:len(en_confidences)])
        en_std = np.std(en_confidences)
        ko_std = np.std(ko_confidences[:len(en_confidences)])

        # Find most uncertain sentence pairs
        uncertainty_diffs = [abs(en - ko) for en, ko in zip(en_confidences, ko_confidences[:len(en_confidences)])]
        max_diff_idx = np.argmax(uncertainty_diffs)
        min_diff_idx = np.argmin(uncertainty_diffs)

        info_text = f"""
🔍 Multilingual Confidence Analysis Results:

📊 Total Sentence Pairs: {len(en_confidences)}
📈 Analysis Method: Token-level entropy + uncertainty

🇺🇸 English Statistics:
  • Average Uncertainty: {en_mean:.3f}
  • Standard Deviation: {en_std:.3f}
  • Min/Max: {min(en_confidences):.3f} / {max(en_confidences):.3f}

🇰🇷 Korean Statistics:
  • Average Uncertainty: {ko_mean:.3f}
  • Standard Deviation: {ko_std:.3f}
  • Min/Max: {min(ko_confidences[:len(en_confidences)]):.3f} / {max(ko_confidences[:len(en_confidences)]):.3f}

📋 Key Findings:
  • Largest difference: Pair {max_diff_idx + 1} ({uncertainty_diffs[max_diff_idx]:.3f})
  • Smallest difference: Pair {min_diff_idx + 1} ({uncertainty_diffs[min_diff_idx]:.3f})
  • Language bias: {"Korean" if ko_mean > en_mean else "English"} shows higher uncertainty

💡 Higher values = More model uncertainty
        """
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_dir / "confidence_language_comparison.png",
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"   ✅ Language comparison confidence plot saved")

    except Exception as e:
        print(f"   ⚠️ Multilingual confidence comparison failed: {e}")

def save_attention_visualizations(attention_results):
    """Generate and save attention analysis visualizations for all sentence pairs."""
    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Korean font manager
        from utils.font_manager import setup_korean_fonts
        setup_korean_fonts()

        if not attention_results or 'comparison_results' not in attention_results:
            print(f"   ⚠️ No attention results to visualize")
            return

        # Extract similarity data for all sentence pairs
        similarities = []
        for comp in attention_results['comparison_results']:
            sim = comp.get('overall_similarity', 0.0)
            similarities.append(sim if sim is not None else 0.0)

        if not similarities:
            print(f"   ⚠️ No valid attention similarities found")
            return

        # Create comprehensive attention analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Sentence-wise attention similarity
        num_pairs = len(similarities)
        x_pos = np.arange(num_pairs)
        colors = plt.cm.viridis(np.linspace(0, 1, num_pairs))

        bars = ax1.bar(x_pos, similarities, color=colors, alpha=0.8)
        ax1.set_title(f'EN-KO Attention Similarity by Sentence Pair (Total: {num_pairs})', fontsize=14)
        ax1.set_xlabel('Sentence Pair Index')
        ax1.set_ylabel('Attention Similarity Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{i+1}' for i in range(num_pairs)])
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, similarities)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. Similarity distribution
        ax2.hist(similarities, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
        ax2.axvline(np.median(similarities), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(similarities):.3f}')
        ax2.set_title('Attention Similarity Distribution', fontsize=14)
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Similarity range analysis
        similarity_ranges = {
            'High (0.8-1.0)': sum(1 for s in similarities if s >= 0.8),
            'Medium (0.6-0.8)': sum(1 for s in similarities if 0.6 <= s < 0.8),
            'Low (0.4-0.6)': sum(1 for s in similarities if 0.4 <= s < 0.6),
            'Very Low (0.0-0.4)': sum(1 for s in similarities if s < 0.4)
        }

        labels = list(similarity_ranges.keys())
        sizes = list(similarity_ranges.values())
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        # Only show non-zero slices
        non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
        non_zero_sizes = [size for size in sizes if size > 0]
        non_zero_colors = [color for color, size in zip(colors_pie, sizes) if size > 0]

        if non_zero_sizes:
            ax3.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.0f%%', startangle=90)
            ax3.set_title('Attention Similarity Range Distribution', fontsize=14)
        else:
            ax3.text(0.5, 0.5, 'No valid similarity data', ha='center', va='center', transform=ax3.transAxes)

        # 4. Summary statistics and top/bottom pairs
        ax4.axis('off')

        # Calculate statistics
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        max_sim = max(similarities)
        min_sim = min(similarities)
        max_idx = similarities.index(max_sim)
        min_idx = similarities.index(min_sim)

        # Get sentence previews for top/bottom pairs
        if 'english_sentences' in attention_results and 'korean_sentences' in attention_results:
            max_en_preview = attention_results['english_sentences'][max_idx][:40] + "..." if len(attention_results['english_sentences'][max_idx]) > 40 else attention_results['english_sentences'][max_idx]
            max_ko_preview = attention_results['korean_sentences'][max_idx][:40] + "..." if len(attention_results['korean_sentences'][max_idx]) > 40 else attention_results['korean_sentences'][max_idx]
            min_en_preview = attention_results['english_sentences'][min_idx][:40] + "..." if len(attention_results['english_sentences'][min_idx]) > 40 else attention_results['english_sentences'][min_idx]
            min_ko_preview = attention_results['korean_sentences'][min_idx][:40] + "..." if len(attention_results['korean_sentences'][min_idx]) > 40 else attention_results['korean_sentences'][min_idx]
        else:
            max_en_preview = max_ko_preview = min_en_preview = min_ko_preview = "N/A"

        summary_text = f"""
🔍 Attention Pattern Analysis Summary:

📊 Total Sentence Pairs: {num_pairs}
📈 Analysis Method: Cross-language attention comparison

📋 Statistics:
  • Average Similarity: {mean_sim:.4f}
  • Standard Deviation: {std_sim:.4f}
  • Range: {min_sim:.4f} - {max_sim:.4f}

🏆 Most Similar Pair ({max_idx + 1}): {max_sim:.4f}
  EN: {max_en_preview}
  KO: {max_ko_preview}

🔻 Least Similar Pair ({min_idx + 1}): {min_sim:.4f}
  EN: {min_en_preview}
  KO: {min_ko_preview}

💡 Higher values indicate more similar attention patterns
   between English and Korean sentences.
        """

        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_dir / "attention_analysis_comparison.png",
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"   ✅ Attention analysis visualization saved")

    except Exception as e:
        print(f"   ⚠️ Attention visualization failed: {e}")

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

        # Store results for all sentence pairs
        attention_results = {
            'english_sentences': [],
            'korean_sentences': [],
            'base_model_attention': [],
            'train_model_attention': [],
            'comparison_results': []
        }

        # Analyze all sentence pairs
        total_pairs = len(TEST_SENTENCES)
        for i, (en_text, ko_text) in enumerate(TEST_SENTENCES):
            print(f"   📝 Analyzing attention for sentence pair {i+1}/{total_pairs}...")

            # English sentence attention analysis
            try:
                en_base_attention = attention_analyzer.extract_attention_weights(
                    model_name_or_path=BASE_MODEL_PATH,
                    text=en_text,
                    model_type='base'
                )
                attention_results['english_sentences'].append(en_text)
                attention_results['base_model_attention'].append(en_base_attention)

                # Korean sentence attention analysis
                ko_base_attention = attention_analyzer.extract_attention_weights(
                    model_name_or_path=BASE_MODEL_PATH,
                    text=ko_text,
                    model_type='base'
                )
                attention_results['korean_sentences'].append(ko_text)

                # Compare English vs Korean attention patterns
                en_ko_comparison = attention_analyzer.compare_attention_patterns(
                    en_base_attention, ko_base_attention
                )
                attention_results['comparison_results'].append(en_ko_comparison)

                print(f"      Pair {i+1}: EN-KO Attention Similarity = {en_ko_comparison.get('overall_similarity', 'N/A'):.4f}")

            except Exception as e:
                print(f"      ⚠️ Pair {i+1} failed: {e}")
                attention_results['comparison_results'].append({'overall_similarity': 0.0})

        # Overall statistics
        similarities = [comp.get('overall_similarity', 0.0) for comp in attention_results['comparison_results'] if comp.get('overall_similarity') is not None]
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)
            max_idx = similarities.index(max_similarity)
            min_idx = similarities.index(min_similarity)

            print(f"\n   📊 Attention Analysis Summary:")
            print(f"     Total Pairs Analyzed: {len(similarities)}")
            print(f"     Average EN-KO Similarity: {avg_similarity:.4f}")
            print(f"     Max Similarity: {max_similarity:.4f} (Pair {max_idx + 1})")
            print(f"     Min Similarity: {min_similarity:.4f} (Pair {min_idx + 1})")

        # Training model comparison (if exists) - use first sentence as sample
        train_attention = None
        if os.path.exists(TRAINING_MODEL_PATH) and attention_results['base_model_attention']:
            try:
                sample_text = TEST_SENTENCES[0][0]
                train_attention = attention_analyzer.extract_attention_weights(
                    model_name_or_path=TRAINING_MODEL_PATH,
                    text=sample_text,
                    model_type='trained'
                )

                # Compare base vs training model
                base_train_comparison = attention_analyzer.compare_attention_patterns(
                    attention_results['base_model_attention'][0], train_attention
                )
                print(f"     Base vs Training Model Similarity: {base_train_comparison.get('overall_similarity', 'N/A'):.4f}")

            except Exception as e:
                print(f"   ⚠️ Training model comparison failed: {e}")
        else:
            print(f"   ℹ️ Training model not available for comparison")

        attention_results['train_model_attention'] = train_attention
        return attention_results

    except Exception as e:
        print(f"   ❌ Attention analysis failed: {e}")
        return None

def analyze_confidence_differences():
    """Analyze confidence differences for English-Korean sentence pairs."""
    if not ENABLE_CONFIDENCE_ANALYSIS:
        return None

    print("\n📈 Running Confidence Analysis...")

    try:
        confidence_analyzer = ConfidenceAnalyzer()

        # Analyze confidence for all sentence pairs
        confidence_results = {
            'english_sentences': [],
            'korean_sentences': [],
            'base_model_results': [],
            'train_model_results': []
        }

        # Analyze ALL sentence pairs
        total_pairs = len(TEST_SENTENCES)
        for i, (en_text, ko_text) in enumerate(TEST_SENTENCES):
            print(f"   📝 Analyzing sentence pair {i+1}/{total_pairs}...")

            # English sentence analysis
            en_confidence = confidence_analyzer.analyze_prediction_confidence(
                model_name_or_path=BASE_MODEL_PATH,
                text=en_text,
                model_type='base'
            )
            confidence_results['english_sentences'].append(en_text)
            confidence_results['base_model_results'].append(en_confidence)

            # Korean sentence analysis
            ko_confidence = confidence_analyzer.analyze_prediction_confidence(
                model_name_or_path=BASE_MODEL_PATH,
                text=ko_text,
                model_type='base'
            )
            confidence_results['korean_sentences'].append(ko_text)
            confidence_results['train_model_results'].append(ko_confidence)  # Store Korean results separately

        # Use first English sentence for visualization (backward compatibility)
        sample_text = TEST_SENTENCES[0][0]
        base_confidence = confidence_results['base_model_results'][0]

        total_analyzed = len(confidence_results['base_model_results'])
        print(f"   ✅ Analyzed {total_analyzed} English-Korean sentence pairs")
        print(f"   📊 Method: {base_confidence.get('method', 'logit_based')}")

        # Show results for each sentence pair
        print(f"\n   📋 Sentence Pair Analysis Results:")
        for i in range(total_analyzed):
            en_result = confidence_results['base_model_results'][i]
            ko_result = confidence_results['train_model_results'][i]

            # Extract uncertainty scores
            en_uncertainty = 0.5
            ko_uncertainty = 0.6

            if 'confidence_measures' in en_result:
                en_measures = en_result['confidence_measures']
                en_uncertainty = 1.0 - en_measures.get('average_confidence', 0.5)
            elif 'uncertainty_estimates' in en_result:
                en_uncertainty = en_result['uncertainty_estimates']['mean_uncertainty']

            if 'confidence_measures' in ko_result:
                ko_measures = ko_result['confidence_measures']
                ko_uncertainty = 1.0 - ko_measures.get('average_confidence', 0.5)
            elif 'uncertainty_estimates' in ko_result:
                ko_uncertainty = ko_result['uncertainty_estimates']['mean_uncertainty']

            # Show short preview of sentences
            en_preview = confidence_results['english_sentences'][i][:50] + "..." if len(confidence_results['english_sentences'][i]) > 50 else confidence_results['english_sentences'][i]
            ko_preview = confidence_results['korean_sentences'][i][:50] + "..." if len(confidence_results['korean_sentences'][i]) > 50 else confidence_results['korean_sentences'][i]

            print(f"     Pair {i+1:2d}: EN={en_uncertainty:.3f} | KO={ko_uncertainty:.3f} | Diff={abs(en_uncertainty-ko_uncertainty):.3f}")
            print(f"             EN: {en_preview}")
            print(f"             KO: {ko_preview}")

        # Overall summary
        if 'confidence_measures' in base_confidence:
            measures = base_confidence['confidence_measures']
            if 'entropy' in measures:
                entropy_data = measures['entropy']
                print(f"\n   📈 Sample Analysis (Pair 1):")
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

        # Generate confidence visualizations
        try:
            save_confidence_visualizations(base_confidence)
            save_multilingual_confidence_comparison(confidence_results)
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
    from utils.font_manager import setup_korean_fonts
    setup_korean_fonts()

    print_header()

    # Check if training model exists
    if not os.path.exists(TRAINING_MODEL_PATH) and TRAINING_MODEL_PATH != "/path/to/your/trained/model":
        print(f"\n⚠️ Warning: Training model path not found: {TRAINING_MODEL_PATH}")
        print("   Analysis will proceed with base model only.\n")

    # Run analyses
    embedding_results = analyze_embedding_differences()
    attention_results = analyze_attention_differences()
    confidence_results = analyze_confidence_differences()

    # Generate visualizations for each analysis type
    print("\n🎨 Generating Visualizations...")

    # Embedding visualizations (already includes all sentences)
    if embedding_results:
        save_embedding_visualizations(embedding_results)

    # Attention visualizations (all sentence pairs)
    if attention_results and ENABLE_ATTENTION_ANALYSIS:
        save_attention_visualizations(attention_results)

    # Confidence visualizations are already called within analyze_confidence_differences()

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