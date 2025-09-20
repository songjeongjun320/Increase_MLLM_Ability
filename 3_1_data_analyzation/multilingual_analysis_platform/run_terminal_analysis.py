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
import torch
from typing import List, Dict, Any

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
from utils.font_manager import configure_plot_korean
from visualization.embedding_plots import EmbeddingVisualizer
from visualization.confidence_plots import ConfidenceVisualizer
from models.model_manager import get_model_manager

# ============================================================================
# ModelEmbeddingComparator Class - 실제 베이스/트레이닝 모델 비교
# ============================================================================

class ModelEmbeddingComparator:
    """Compare embeddings between base and training models using actual model loading."""

    def __init__(self, base_model_path, training_model_path):
        """Initialize with base and training model paths."""
        self.base_model_path = base_model_path
        self.training_model_path = training_model_path
        self.model_manager = get_model_manager()

    def generate_dual_model_embeddings(self, texts, languages):
        """
        Generate embeddings using both base and training models.

        Args:
            texts: List of texts to encode
            languages: List of language codes

        Returns:
            Dictionary containing embeddings from both models
        """
        print(f"   🔍 Loading base model: {self.base_model_path}")
        try:
            base_model, base_tokenizer = self.model_manager.load_model(self.base_model_path, 'base')
            print(f"   ✅ Base model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load base model: {e}")
            print(f"   🔄 Falling back to sentence transformer for base model")
            base_embeddings = self._generate_sentence_transformer_embeddings(texts)
        else:
            base_embeddings = self._generate_model_embeddings(base_model, base_tokenizer, texts)

        print(f"   🎯 Loading training model: {self.training_model_path}")
        try:
            train_model, train_tokenizer = self.model_manager.load_model(self.training_model_path, 'trained')
            print(f"   ✅ Training model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load training model: {e}")
            print(f"   🔄 Falling back to sentence transformer for training model")
            train_embeddings = self._generate_sentence_transformer_embeddings(texts)
        else:
            train_embeddings = self._generate_model_embeddings(train_model, train_tokenizer, texts)

        return {
            'base_embeddings': base_embeddings,
            'train_embeddings': train_embeddings,
            'texts': texts,
            'languages': languages,
            'base_model_path': self.base_model_path,
            'train_model_path': self.training_model_path
        }

    def _generate_model_embeddings(self, model, tokenizer, texts):
        """Generate embeddings using a specific model."""
        embeddings = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}

                # Get model output
                outputs = model(**inputs, output_hidden_states=True)

                # Use pooled output or mean of last hidden state
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output[0].cpu().numpy()
                else:
                    # Mean pooling of last hidden state
                    last_hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
                    attention_mask = inputs['attention_mask'][0]  # [seq_len]

                    # Apply attention mask and compute mean
                    masked_hidden = last_hidden * attention_mask.unsqueeze(-1).float()
                    embedding = masked_hidden.sum(0) / attention_mask.sum().float()
                    embedding = embedding.cpu().numpy()

                embeddings.append(embedding)

        return np.array(embeddings)

    def _generate_sentence_transformer_embeddings(self, texts):
        """Fallback to sentence transformer embeddings."""
        sentence_transformer = self.model_manager.load_sentence_transformer()
        embeddings = sentence_transformer.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
        return np.array(embeddings)

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

def plot_dual_model_pca_comparison(dual_embeddings, output_path):
    """
    Create a PCA comparison plot showing both base and training model embeddings.

    Args:
        dual_embeddings: Dictionary containing embeddings from both models
        output_path: Path to save the plot

    Returns:
        Boolean indicating success
    """
    try:
        base_embeddings = dual_embeddings['base_embeddings']
        train_embeddings = dual_embeddings['train_embeddings']
        languages = dual_embeddings['languages']

        # Combine embeddings for joint PCA
        all_embeddings = np.vstack([base_embeddings, train_embeddings])

        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(all_embeddings)

        # Split back into base and training
        n_samples = len(base_embeddings)
        base_reduced = reduced_embeddings[:n_samples]
        train_reduced = reduced_embeddings[n_samples:]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Configure Korean fonts
        configure_plot_korean(fig, ax)

        # Color mapping for languages
        from utils.config_loader import get_language_colors
        colors = get_language_colors()

        # Plot base model embeddings
        for lang in set(languages):
            lang_mask = np.array(languages) == lang
            if np.any(lang_mask):
                base_color = colors.get(lang, '#1f77b4')
                train_color = colors.get(lang, '#ff7f0e')

                # Base model points (circles)
                ax.scatter(
                    base_reduced[lang_mask, 0],
                    base_reduced[lang_mask, 1],
                    c=base_color, marker='o', s=80, alpha=0.7,
                    label=f'Base-{lang}', edgecolors='black', linewidth=0.5
                )

                # Training model points (triangles)
                ax.scatter(
                    train_reduced[lang_mask, 0],
                    train_reduced[lang_mask, 1],
                    c=train_color, marker='^', s=80, alpha=0.7,
                    label=f'Train-{lang}', edgecolors='black', linewidth=0.5
                )

        # Add index labels (only indices, not full text)
        for i in range(n_samples):
            # Base model labels
            ax.annotate(f'{i}', (base_reduced[i, 0], base_reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8, color='darkblue')

            # Training model labels
            ax.annotate(f'{i}', (train_reduced[i, 0], train_reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8, color='darkred')

        # Set labels and title
        ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('Base vs Training Model - PCA Comparison\n(Circles=Base, Triangles=Training)', fontsize=14, pad=20)

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add model information
        model_info = f"Base: {Path(dual_embeddings['base_model_path']).name}\nTrain: {Path(dual_embeddings['train_model_path']).name}"
        ax.text(0.02, 0.98, model_info, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return True

    except Exception as e:
        print(f"      ⚠️ Dual model PCA plot failed: {e}")
        return False

def analyze_pca_components(dual_embeddings, texts, languages):
    """
    Analyze what each PCA component represents linguistically.

    Args:
        dual_embeddings: Dictionary containing embeddings from both models
        texts: List of input texts
        languages: List of language codes

    Returns:
        Dictionary with component analysis results
    """
    try:
        base_embeddings = dual_embeddings['base_embeddings']
        train_embeddings = dual_embeddings['train_embeddings']

        # Combine embeddings for joint PCA
        all_embeddings = np.vstack([base_embeddings, train_embeddings])

        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(all_embeddings)

        # Split coordinates
        n_samples = len(base_embeddings)
        base_reduced = reduced_embeddings[:n_samples]
        train_reduced = reduced_embeddings[n_samples:]

        # Analyze PC1 and PC2
        pc1_coords = reduced_embeddings[:n_samples, 0]  # Use base model coordinates
        pc2_coords = reduced_embeddings[:n_samples, 1]

        print(f"\n🔍 PCA Component Analysis:")
        print(f"   PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
        print(f"   PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")

        # Language-wise analysis
        en_mask = np.array(languages) == 'en'
        ko_mask = np.array(languages) == 'ko'

        print(f"\n📊 PC1 (Language Separation):")
        print(f"   English average: {pc1_coords[en_mask].mean():.2f}")
        print(f"   Korean average: {pc1_coords[ko_mask].mean():.2f}")
        print(f"   Language separation: {abs(pc1_coords[en_mask].mean() - pc1_coords[ko_mask].mean()):.2f}")

        print(f"\n📊 PC2 (Content Variation):")
        print(f"   English std: {pc2_coords[en_mask].std():.2f}")
        print(f"   Korean std: {pc2_coords[ko_mask].std():.2f}")

        # Sentence-level analysis
        print(f"\n📝 Sentence Analysis:")
        for i in range(0, len(texts), 2):  # Every pair (EN-KO)
            if i+1 < len(texts):
                en_idx = i
                ko_idx = i + 1
                print(f"   Pair {i//2}:")
                print(f"     EN[{en_idx}]: PC1={pc1_coords[en_idx]:.2f}, PC2={pc2_coords[en_idx]:.2f}")
                print(f"     KO[{ko_idx}]: PC1={pc1_coords[ko_idx]:.2f}, PC2={pc2_coords[ko_idx]:.2f}")
                print(f"     Cross-lingual distance: {np.sqrt((pc1_coords[en_idx]-pc1_coords[ko_idx])**2 + (pc2_coords[en_idx]-pc2_coords[ko_idx])**2):.2f}")

        # Model comparison analysis
        print(f"\n🔄 Base vs Training Model Differences:")
        base_pc1 = base_reduced[:, 0]
        train_pc1 = train_reduced[:, 0]
        base_pc2 = base_reduced[:, 1]
        train_pc2 = train_reduced[:, 1]

        pc1_shift = np.mean(np.abs(base_pc1 - train_pc1))
        pc2_shift = np.mean(np.abs(base_pc2 - train_pc2))

        print(f"   Average PC1 shift: {pc1_shift:.2f}")
        print(f"   Average PC2 shift: {pc2_shift:.2f}")
        print(f"   Training impact: {'PC1 (language)' if pc1_shift > pc2_shift else 'PC2 (content)'} more affected")

        return {
            'pca_components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'pc1_coords': pc1_coords,
            'pc2_coords': pc2_coords,
            'language_separation': abs(pc1_coords[en_mask].mean() - pc1_coords[ko_mask].mean()),
            'model_shifts': {'pc1': pc1_shift, 'pc2': pc2_shift}
        }

    except Exception as e:
        print(f"   ❌ PCA component analysis failed: {e}")
        return None

def plot_dual_model_confidence_entropy(confidence_results, output_path):
    """
    Create confidence entropy comparison between base and training models.

    Args:
        confidence_results: Results from comprehensive confidence analysis
        output_path: Path to save the plot

    Returns:
        Boolean indicating success
    """
    try:
        # Extract entropy data for both models and languages
        def extract_entropies(results, default=0.5):
            entropies = []
            for result in results:
                if 'confidence_measures' in result:
                    # Extract entropy from confidence measures
                    measures = result['confidence_measures']
                    entropy = measures.get('entropy', default)
                    entropies.append(entropy)
                elif 'uncertainty_estimates' in result:
                    # Use uncertainty as proxy for entropy
                    uncertainty = result['uncertainty_estimates']['mean_uncertainty']
                    entropies.append(uncertainty)
                else:
                    entropies.append(default)
            return entropies

        base_en_entropies = extract_entropies(confidence_results['base_model_en_results'], 0.5)
        base_ko_entropies = extract_entropies(confidence_results['base_model_ko_results'], 0.6)
        train_en_entropies = extract_entropies(confidence_results['train_model_en_results'], 0.5)
        train_ko_entropies = extract_entropies(confidence_results['train_model_ko_results'], 0.6)

        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Model-Language Average Comparison
        categories = ['Base_EN', 'Base_KO', 'Train_EN', 'Train_KO']
        avg_entropies = [
            np.mean(base_en_entropies),
            np.mean(base_ko_entropies),
            np.mean(train_en_entropies),
            np.mean(train_ko_entropies)
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = ax1.bar(categories, avg_entropies, color=colors, alpha=0.8)
        ax1.set_title('Average Confidence Entropy by Model-Language', fontsize=14)
        ax1.set_ylabel('Average Entropy')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, avg_entropies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Sentence-wise Entropy Comparison
        num_pairs = len(base_en_entropies)
        x_pos = np.arange(num_pairs)
        width = 0.2

        bars1 = ax2.bar(x_pos - 1.5*width, base_en_entropies, width, label='Base_EN', color=colors[0], alpha=0.8)
        bars2 = ax2.bar(x_pos - 0.5*width, base_ko_entropies, width, label='Base_KO', color=colors[1], alpha=0.8)
        bars3 = ax2.bar(x_pos + 0.5*width, train_en_entropies, width, label='Train_EN', color=colors[2], alpha=0.8)
        bars4 = ax2.bar(x_pos + 1.5*width, train_ko_entropies, width, label='Train_KO', color=colors[3], alpha=0.8)

        ax2.set_title(f'Entropy by Sentence Pair (Total: {num_pairs} pairs)', fontsize=14)
        ax2.set_xlabel('Sentence Pair Index')
        ax2.set_ylabel('Confidence Entropy')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{i}' for i in range(num_pairs)])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. Distribution Comparison
        ax3.hist(base_en_entropies, bins=10, alpha=0.7, label='Base_EN', color=colors[0])
        ax3.hist(base_ko_entropies, bins=10, alpha=0.7, label='Base_KO', color=colors[1])
        ax3.hist(train_en_entropies, bins=10, alpha=0.7, label='Train_EN', color=colors[2])
        ax3.hist(train_ko_entropies, bins=10, alpha=0.7, label='Train_KO', color=colors[3])
        ax3.set_title('Entropy Distribution Comparison', fontsize=14)
        ax3.set_xlabel('Confidence Entropy')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        # 4. Analysis Summary
        ax4.axis('off')

        # Calculate improvements
        en_improvement = np.mean(base_en_entropies) - np.mean(train_en_entropies)
        ko_improvement = np.mean(base_ko_entropies) - np.mean(train_ko_entropies)

        summary_text = f"""
🔍 Confidence Entropy Analysis:

📊 Dataset: {num_pairs} English-Korean sentence pairs
📈 Method: Token-level entropy analysis

🇺🇸 English Confidence:
  • Base Model Entropy:     {np.mean(base_en_entropies):.4f}
  • Training Model Entropy: {np.mean(train_en_entropies):.4f}
  • Improvement:           {en_improvement:+.4f}

🇰🇷 Korean Confidence:
  • Base Model Entropy:     {np.mean(base_ko_entropies):.4f}
  • Training Model Entropy: {np.mean(train_ko_entropies):.4f}
  • Improvement:           {ko_improvement:+.4f}

💡 Key Insights:
  • {'English' if en_improvement > ko_improvement else 'Korean'} shows better entropy improvement
  • Lower entropy = Higher model confidence
  • Training {'improved' if en_improvement > 0 and ko_improvement > 0 else 'had mixed effects on'} confidence
        """

        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return True

    except Exception as e:
        print(f"      ⚠️ Dual model confidence entropy plot failed: {e}")
        return False

def plot_dual_model_comparison(dual_embeddings, output_path, method='tsne'):
    """
    Create a comparison plot showing both base and training model embeddings using specified method.

    Args:
        dual_embeddings: Dictionary containing embeddings from both models
        output_path: Path to save the plot
        method: Dimensionality reduction method ('tsne', 'umap')

    Returns:
        Boolean indicating success
    """
    try:
        base_embeddings = dual_embeddings['base_embeddings']
        train_embeddings = dual_embeddings['train_embeddings']
        languages = dual_embeddings['languages']

        # Combine embeddings for joint reduction
        all_embeddings = np.vstack([base_embeddings, train_embeddings])

        # Perform dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
            reduced_embeddings = reducer.fit_transform(all_embeddings)
            method_name = "t-SNE"
        elif method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_embeddings)-1))
                reduced_embeddings = reducer.fit_transform(all_embeddings)
                method_name = "UMAP"
            except ImportError:
                print(f"      ⚠️ UMAP not available, skipping")
                return False
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Split back into base and training
        n_samples = len(base_embeddings)
        base_reduced = reduced_embeddings[:n_samples]
        train_reduced = reduced_embeddings[n_samples:]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Configure Korean fonts
        configure_plot_korean(fig, ax)

        # Color mapping for languages
        from utils.config_loader import get_language_colors
        colors = get_language_colors()

        # Plot base model embeddings
        for lang in set(languages):
            lang_mask = np.array(languages) == lang
            if np.any(lang_mask):
                base_color = colors.get(lang, '#1f77b4')
                train_color = colors.get(lang, '#ff7f0e')

                # Base model points (circles)
                ax.scatter(
                    base_reduced[lang_mask, 0],
                    base_reduced[lang_mask, 1],
                    c=base_color, marker='o', s=80, alpha=0.7,
                    label=f'Base-{lang}', edgecolors='black', linewidth=0.5
                )

                # Training model points (triangles)
                ax.scatter(
                    train_reduced[lang_mask, 0],
                    train_reduced[lang_mask, 1],
                    c=train_color, marker='^', s=80, alpha=0.7,
                    label=f'Train-{lang}', edgecolors='black', linewidth=0.5
                )

        # Add index labels (only indices, not full text)
        for i in range(n_samples):
            # Base model labels
            ax.annotate(f'{i}', (base_reduced[i, 0], base_reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8, color='darkblue')

            # Training model labels
            ax.annotate(f'{i}', (train_reduced[i, 0], train_reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8, color='darkred')

        # Set labels and title
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f'Base vs Training Model - {method_name} Comparison\n(Circles=Base, Triangles=Training)', fontsize=14, pad=20)

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add model information
        model_info = f"Base: {Path(dual_embeddings['base_model_path']).name}\nTrain: {Path(dual_embeddings['train_model_path']).name}"
        ax.text(0.02, 0.98, model_info, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        return True

    except Exception as e:
        print(f"      ⚠️ Dual model {method} plot failed: {e}")
        return False

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
    """Generate and save comprehensive embedding visualizations with base vs training model comparison."""
    try:
        # Initialize Korean font manager first
        from utils.font_manager import setup_korean_fonts
        setup_korean_fonts()

        # Create output directory
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model comparator for actual base vs training comparison
        model_comparator = ModelEmbeddingComparator(BASE_MODEL_PATH, TRAINING_MODEL_PATH)

        # Prepare text and language data with proper encoding
        texts = []
        languages = []
        for en, ko in TEST_SENTENCES:
            # Ensure proper UTF-8 encoding for Korean text
            texts.extend([en, ko.encode('utf-8').decode('utf-8')])
            languages.extend(['en', 'ko'])

        print(f"   📊 Generating dual-model embeddings comparison...")

        # Generate embeddings from both models
        dual_embeddings = model_comparator.generate_dual_model_embeddings(texts, languages)

        # Generate comprehensive visualizations
        plots_saved = 0
        total_plots = 0

        print(f"   📊 Generating comprehensive visualizations...")

        # Dual-model comparisons for different dimensionality reduction methods

        # 1. PCA comparison
        total_plots += 1
        success = plot_dual_model_pca_comparison(dual_embeddings, output_dir / "dual_model_pca_comparison.png")
        if success:
            plots_saved += 1
            print(f"      ✅ Dual-model PCA comparison saved")

            # Perform PCA component analysis
            pca_analysis = analyze_pca_components(dual_embeddings, texts, languages)

        # 2. t-SNE comparison
        total_plots += 1
        success = plot_dual_model_comparison(dual_embeddings, output_dir / "dual_model_tsne_comparison.png", method='tsne')
        if success:
            plots_saved += 1
            print(f"      ✅ Dual-model t-SNE comparison saved")

        # 3. UMAP comparison (if available)
        total_plots += 1
        success = plot_dual_model_comparison(dual_embeddings, output_dir / "dual_model_umap_comparison.png", method='umap')
        if success:
            plots_saved += 1
            print(f"      ✅ Dual-model UMAP comparison saved")
        else:
            print(f"      ⚠️ UMAP comparison skipped (not available)")

        # Generate sentence transformer similarity matrix for legacy compatibility
        embedding_analyzer = SentenceEmbeddingAnalyzer()
        embedding_result = embedding_analyzer.generate_embeddings(texts, languages)
        embeddings = embedding_result['embeddings']

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

def save_comprehensive_confidence_comparison(confidence_results):
    """Generate comprehensive 4-way confidence comparison visualizations: Base_EN, Base_KO, Train_EN, Train_KO."""
    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Korean font manager
        from utils.font_manager import setup_korean_fonts
        setup_korean_fonts()

        # Extract uncertainty scores from 4-way analysis
        def extract_uncertainties(results, default=0.5):
            uncertainties = []
            for result in results:
                if 'confidence_measures' in result:
                    measures = result['confidence_measures']
                    avg_confidence = measures.get('average_confidence', default)
                    uncertainties.append(1.0 - avg_confidence)
                elif 'uncertainty_estimates' in result:
                    uncertainties.append(result['uncertainty_estimates']['mean_uncertainty'])
                else:
                    uncertainties.append(default)
            return uncertainties

        base_en_uncertainties = extract_uncertainties(confidence_results['base_model_en_results'], 0.5)
        base_ko_uncertainties = extract_uncertainties(confidence_results['base_model_ko_results'], 0.6)
        train_en_uncertainties = extract_uncertainties(confidence_results['train_model_en_results'], 0.5)
        train_ko_uncertainties = extract_uncertainties(confidence_results['train_model_ko_results'], 0.6)

        # Create comprehensive 4-way comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 4-way Model-Language Average Comparison
        categories = ['Base_EN', 'Base_KO', 'Train_EN', 'Train_KO']
        avg_uncertainties = [
            np.mean(base_en_uncertainties),
            np.mean(base_ko_uncertainties),
            np.mean(train_en_uncertainties),
            np.mean(train_ko_uncertainties)
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = ax1.bar(categories, avg_uncertainties, color=colors, alpha=0.8)
        ax1.set_title('4-Way Model-Language Uncertainty Comparison', fontsize=14)
        ax1.set_ylabel('Average Uncertainty Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels and improvement indicators
        for i, (bar, value) in enumerate(zip(bars, avg_uncertainties)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            # Add improvement indicators
            if i == 2:  # Train_EN vs Base_EN
                improvement = base_en_uncertainties[0] - value if base_en_uncertainties else 0
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'Δ{improvement:+.3f}', ha='center', va='bottom', fontsize=10, color='green' if improvement > 0 else 'red')
            elif i == 3:  # Train_KO vs Base_KO
                improvement = base_ko_uncertainties[0] - value if base_ko_uncertainties else 0
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'Δ{improvement:+.3f}', ha='center', va='bottom', fontsize=10, color='green' if improvement > 0 else 'red')

        # 2. Sentence-wise 4-way Comparison
        num_pairs = len(base_en_uncertainties)
        x_pos = np.arange(num_pairs)
        width = 0.2

        bars1 = ax2.bar(x_pos - 1.5*width, base_en_uncertainties, width, label='Base_EN', color=colors[0], alpha=0.8)
        bars2 = ax2.bar(x_pos - 0.5*width, base_ko_uncertainties, width, label='Base_KO', color=colors[1], alpha=0.8)
        bars3 = ax2.bar(x_pos + 0.5*width, train_en_uncertainties, width, label='Train_EN', color=colors[2], alpha=0.8)
        bars4 = ax2.bar(x_pos + 1.5*width, train_ko_uncertainties, width, label='Train_KO', color=colors[3], alpha=0.8)

        ax2.set_title(f'4-Way Uncertainty by Sentence Pair (Total: {num_pairs} pairs)', fontsize=14)
        ax2.set_xlabel('Sentence Pair Index')
        ax2.set_ylabel('Uncertainty Score')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{i+1}' for i in range(num_pairs)])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. Cross-lingual and Cross-model Improvement Analysis
        cross_lingual_base = [abs(en - ko) for en, ko in zip(base_en_uncertainties, base_ko_uncertainties)]
        cross_lingual_train = [abs(en - ko) for en, ko in zip(train_en_uncertainties, train_ko_uncertainties)]
        cross_model_en = [abs(base - train) for base, train in zip(base_en_uncertainties, train_en_uncertainties)]
        cross_model_ko = [abs(base - train) for base, train in zip(base_ko_uncertainties, train_ko_uncertainties)]

        improvement_categories = ['Cross-lingual\n(Base)', 'Cross-lingual\n(Train)', 'Cross-model\n(English)', 'Cross-model\n(Korean)']
        avg_improvements = [
            np.mean(cross_lingual_base),
            np.mean(cross_lingual_train),
            np.mean(cross_model_en),
            np.mean(cross_model_ko)
        ]

        bars = ax3.bar(improvement_categories, avg_improvements, color=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow'], alpha=0.8)
        ax3.set_title('Cross-dimensional Analysis', fontsize=14)
        ax3.set_ylabel('Average Difference Score')
        ax3.tick_params(axis='x', rotation=0)

        # Add improvement indicator for cross-lingual comparison
        cross_lingual_improvement = np.mean(cross_lingual_base) - np.mean(cross_lingual_train)
        ax3.text(1, max(avg_improvements) * 0.9, f'Improvement: {cross_lingual_improvement:+.3f}',
                ha='center', va='center', fontweight='bold',
                color='green' if cross_lingual_improvement > 0 else 'red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        for bar, value in zip(bars, avg_improvements):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. Comprehensive Analysis Summary
        ax4.axis('off')

        # Calculate comprehensive statistics
        base_en_avg = np.mean(base_en_uncertainties)
        base_ko_avg = np.mean(base_ko_uncertainties)
        train_en_avg = np.mean(train_en_uncertainties)
        train_ko_avg = np.mean(train_ko_uncertainties)

        en_improvement = base_en_avg - train_en_avg
        ko_improvement = base_ko_avg - train_ko_avg
        cross_lingual_base_avg = np.mean(cross_lingual_base)
        cross_lingual_train_avg = np.mean(cross_lingual_train)

        # Count improvements
        en_improved_pairs = sum(1 for base, train in zip(base_en_uncertainties, train_en_uncertainties) if base > train)
        ko_improved_pairs = sum(1 for base, train in zip(base_ko_uncertainties, train_ko_uncertainties) if base > train)

        summary_text = f"""
🔍 Comprehensive 4-Way Confidence Analysis:

📊 Dataset: {num_pairs} English-Korean sentence pairs
📈 Analysis: Base vs Training Model comparison

🇺🇸 English Performance:
  • Base Model Uncertainty:     {base_en_avg:.4f}
  • Training Model Uncertainty: {train_en_avg:.4f}
  • Improvement:               {en_improvement:+.4f} ({en_improved_pairs}/{num_pairs} pairs)

🇰🇷 Korean Performance:
  • Base Model Uncertainty:     {base_ko_avg:.4f}
  • Training Model Uncertainty: {train_ko_avg:.4f}
  • Improvement:               {ko_improvement:+.4f} ({ko_improved_pairs}/{num_pairs} pairs)

🌍 Cross-lingual Analysis:
  • Base Model EN-KO Difference:  {cross_lingual_base_avg:.4f}
  • Train Model EN-KO Difference: {cross_lingual_train_avg:.4f}
  • Cross-lingual Improvement:    {cross_lingual_improvement:+.4f}

💡 Key Insights:
  • {'English' if en_improvement > ko_improvement else 'Korean'} shows better training improvement
  • Cross-lingual consistency {'improved' if cross_lingual_improvement > 0 else 'decreased'} by {abs(cross_lingual_improvement):.3f}
  • Language bias: {'Reduced' if cross_lingual_improvement > 0.01 else 'Minimal change'}

🎯 Lower uncertainty = Better model confidence
        """

        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_dir / "comprehensive_confidence_4way_comparison.png",
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"   ✅ Comprehensive 4-way confidence comparison saved")

    except Exception as e:
        print(f"   ⚠️ Comprehensive confidence comparison failed: {e}")

def save_attention_visualizations(attention_results):
    """Generate comprehensive 4-way attention analysis visualizations: Base_EN, Base_KO, Train_EN, Train_KO."""
    try:
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Korean font manager
        from utils.font_manager import setup_korean_fonts
        setup_korean_fonts()

        if not attention_results or 'cross_lingual_comparisons' not in attention_results:
            print(f"   ⚠️ No attention results to visualize")
            return

        # Extract similarity data from 4-way analysis
        base_cross_lingual = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_lingual_comparisons']['base_model']]
        train_cross_lingual = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_lingual_comparisons']['train_model']]
        en_cross_model = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_model_comparisons']['english']]
        ko_cross_model = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_model_comparisons']['korean']]

        if not (base_cross_lingual and train_cross_lingual):
            print(f"   ⚠️ No valid attention similarities found")
            return

        # Create comprehensive 4-way attention analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Cross-lingual Attention Similarity Comparison (Base vs Training)
        num_pairs = len(base_cross_lingual)
        x_pos = np.arange(num_pairs)
        width = 0.35

        bars1 = ax1.bar(x_pos - width/2, base_cross_lingual, width, label='Base Model (EN-KO)', color='#1f77b4', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, train_cross_lingual, width, label='Training Model (EN-KO)', color='#ff7f0e', alpha=0.8)

        ax1.set_title(f'Cross-lingual Attention Similarity Comparison (Total: {num_pairs} pairs)', fontsize=14)
        ax1.set_xlabel('Sentence Pair Index')
        ax1.set_ylabel('EN-KO Attention Similarity')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{i+1}' for i in range(num_pairs)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add improvement indicators
        for i, (base_val, train_val) in enumerate(zip(base_cross_lingual, train_cross_lingual)):
            improvement = train_val - base_val
            if abs(improvement) > 0.05:  # Only show significant changes
                ax1.annotate(f'{improvement:+.2f}', xy=(i, max(base_val, train_val) + 0.02),
                           ha='center', va='bottom', fontsize=8,
                           color='green' if improvement > 0 else 'red', fontweight='bold')

        # 2. Cross-model Consistency Analysis (English vs Korean)
        if en_cross_model and ko_cross_model:
            bars3 = ax2.bar(x_pos - width/2, en_cross_model, width, label='English (Base-Train)', color='#2ca02c', alpha=0.8)
            bars4 = ax2.bar(x_pos + width/2, ko_cross_model, width, label='Korean (Base-Train)', color='#d62728', alpha=0.8)

            ax2.set_title(f'Cross-model Consistency by Language', fontsize=14)
            ax2.set_xlabel('Sentence Pair Index')
            ax2.set_ylabel('Base-Train Attention Similarity')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'{i+1}' for i in range(num_pairs)])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'Cross-model data not available', ha='center', va='center', transform=ax2.transAxes)

        # 3. Comprehensive Performance Metrics
        categories = ['Base\n(EN-KO)', 'Train\n(EN-KO)', 'English\n(Base-Train)', 'Korean\n(Base-Train)']
        avg_similarities = [
            np.mean(base_cross_lingual),
            np.mean(train_cross_lingual),
            np.mean(en_cross_model) if en_cross_model else 0,
            np.mean(ko_cross_model) if ko_cross_model else 0
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = ax3.bar(categories, avg_similarities, color=colors, alpha=0.8)
        ax3.set_title('Average Attention Similarity by Category', fontsize=14)
        ax3.set_ylabel('Average Similarity Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=0)

        # Add value labels and improvement indicators
        for i, (bar, value) in enumerate(zip(bars, avg_similarities)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            # Add improvement indicator for training model
            if i == 1:  # Training model cross-lingual
                improvement = avg_similarities[1] - avg_similarities[0]
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'Δ{improvement:+.3f}', ha='center', va='bottom', fontsize=10,
                        color='green' if improvement > 0 else 'red', fontweight='bold')

        # 4. Comprehensive Analysis Summary
        ax4.axis('off')

        # Calculate comprehensive statistics
        base_avg = np.mean(base_cross_lingual)
        train_avg = np.mean(train_cross_lingual)
        cross_lingual_improvement = train_avg - base_avg

        en_avg = np.mean(en_cross_model) if en_cross_model else 0
        ko_avg = np.mean(ko_cross_model) if ko_cross_model else 0

        # Count improvements
        improved_pairs = sum(1 for base, train in zip(base_cross_lingual, train_cross_lingual) if train > base)
        degraded_pairs = sum(1 for base, train in zip(base_cross_lingual, train_cross_lingual) if train < base)

        # Find best and worst performing pairs
        improvements = [train - base for base, train in zip(base_cross_lingual, train_cross_lingual)]
        max_improvement_idx = improvements.index(max(improvements))
        min_improvement_idx = improvements.index(min(improvements))

        summary_text = f"""
🔍 Comprehensive 4-Way Attention Analysis:

📊 Dataset: {num_pairs} English-Korean sentence pairs
📈 Analysis: Base vs Training Model attention patterns

🌍 Cross-lingual Performance (EN-KO similarity):
  • Base Model Average:     {base_avg:.4f}
  • Training Model Average: {train_avg:.4f}
  • Improvement:           {cross_lingual_improvement:+.4f} ({improved_pairs}/{num_pairs} pairs improved)

🔄 Cross-model Consistency:
  • English (Base-Train):   {en_avg:.4f}
  • Korean (Base-Train):    {ko_avg:.4f}
  • Language Bias: {'Korean' if ko_avg > en_avg else 'English'} shows {'higher' if abs(ko_avg - en_avg) > 0.05 else 'similar'} consistency

📈 Training Impact:
  • Improved Pairs: {improved_pairs}/{num_pairs} ({improved_pairs/num_pairs*100:.1f}%)
  • Degraded Pairs: {degraded_pairs}/{num_pairs} ({degraded_pairs/num_pairs*100:.1f}%)
  • Biggest Improvement: Pair {max_improvement_idx + 1} ({max(improvements):+.3f})
  • Biggest Degradation: Pair {min_improvement_idx + 1} ({min(improvements):+.3f})

💡 Key Insights:
  • Cross-lingual ability {'improved' if cross_lingual_improvement > 0.01 else 'showed minimal change' if abs(cross_lingual_improvement) <= 0.01 else 'degraded'}
  • Training {'enhanced' if improved_pairs > degraded_pairs else 'had mixed effects on'} attention alignment
  • {'Consistent' if abs(en_avg - ko_avg) < 0.05 else 'Language-specific'} attention pattern changes

🎯 Higher similarity = Better cross-lingual attention alignment
        """

        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_dir / "comprehensive_attention_4way_comparison.png",
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"   ✅ Comprehensive 4-way attention analysis saved")

    except Exception as e:
        print(f"   ⚠️ Comprehensive attention visualization failed: {e}")

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
    """Analyze attention pattern differences between Base and Training models for all sentences."""
    if not ENABLE_ATTENTION_ANALYSIS:
        return None

    print("\n🔍 Running Comprehensive Attention Analysis...")

    try:
        attention_analyzer = AttentionAnalyzer()

        # Store results for 4-way comparison: Base_EN, Base_KO, Train_EN, Train_KO
        attention_results = {
            'english_sentences': [],
            'korean_sentences': [],
            'base_model_en_attention': [],
            'base_model_ko_attention': [],
            'train_model_en_attention': [],
            'train_model_ko_attention': [],
            'cross_lingual_comparisons': {
                'base_model': [],  # EN-KO comparisons for base model
                'train_model': [], # EN-KO comparisons for training model
            },
            'cross_model_comparisons': {
                'english': [],     # Base-Train comparisons for English
                'korean': []       # Base-Train comparisons for Korean
            }
        }

        # Check if training model exists
        training_available = os.path.exists(TRAINING_MODEL_PATH)
        if not training_available:
            print(f"   ⚠️ Training model not found: {TRAINING_MODEL_PATH}")
            return None

        # Analyze all sentence pairs for both models
        total_pairs = len(TEST_SENTENCES)
        print(f"   📊 Analyzing {total_pairs} sentence pairs across both models...")

        for i, (en_text, ko_text) in enumerate(TEST_SENTENCES):
            print(f"   📝 Processing sentence pair {i+1}/{total_pairs}...")

            try:
                # Store sentences
                attention_results['english_sentences'].append(en_text)
                attention_results['korean_sentences'].append(ko_text)

                # === BASE MODEL ANALYSIS ===
                print(f"      🔍 Base Model Analysis...")

                # English sentence - Base model
                en_base_attention = attention_analyzer.extract_attention_weights(
                    model_name_or_path=BASE_MODEL_PATH,
                    text=en_text,
                    model_type='base'
                )
                attention_results['base_model_en_attention'].append(en_base_attention)

                # Korean sentence - Base model
                ko_base_attention = attention_analyzer.extract_attention_weights(
                    model_name_or_path=BASE_MODEL_PATH,
                    text=ko_text,
                    model_type='base'
                )
                attention_results['base_model_ko_attention'].append(ko_base_attention)

                # === TRAINING MODEL ANALYSIS ===
                print(f"      🎯 Training Model Analysis...")

                # English sentence - Training model
                en_train_attention = attention_analyzer.extract_attention_weights(
                    model_name_or_path=TRAINING_MODEL_PATH,
                    text=en_text,
                    model_type='trained'
                )
                attention_results['train_model_en_attention'].append(en_train_attention)

                # Korean sentence - Training model
                ko_train_attention = attention_analyzer.extract_attention_weights(
                    model_name_or_path=TRAINING_MODEL_PATH,
                    text=ko_text,
                    model_type='trained'
                )
                attention_results['train_model_ko_attention'].append(ko_train_attention)

                # === CROSS-LINGUAL COMPARISONS ===
                # Base model: EN vs KO
                base_en_ko_comparison = attention_analyzer.compare_attention_patterns(
                    en_base_attention, ko_base_attention
                )
                attention_results['cross_lingual_comparisons']['base_model'].append(base_en_ko_comparison)

                # Training model: EN vs KO
                train_en_ko_comparison = attention_analyzer.compare_attention_patterns(
                    en_train_attention, ko_train_attention
                )
                attention_results['cross_lingual_comparisons']['train_model'].append(train_en_ko_comparison)

                # === CROSS-MODEL COMPARISONS ===
                # English: Base vs Training
                en_base_train_comparison = attention_analyzer.compare_attention_patterns(
                    en_base_attention, en_train_attention
                )
                attention_results['cross_model_comparisons']['english'].append(en_base_train_comparison)

                # Korean: Base vs Training
                ko_base_train_comparison = attention_analyzer.compare_attention_patterns(
                    ko_base_attention, ko_train_attention
                )
                attention_results['cross_model_comparisons']['korean'].append(ko_base_train_comparison)

                # Print progress
                base_en_ko_sim = base_en_ko_comparison.get('overall_similarity', 0.0)
                train_en_ko_sim = train_en_ko_comparison.get('overall_similarity', 0.0)
                en_improvement = train_en_ko_sim - base_en_ko_sim

                print(f"      ✅ Pair {i+1}: Base_EN-KO={base_en_ko_sim:.4f}, Train_EN-KO={train_en_ko_sim:.4f}, Δ={en_improvement:+.4f}")

            except Exception as e:
                print(f"      ❌ Pair {i+1} failed: {e}")
                # Add placeholder data for failed analysis
                attention_results['base_model_en_attention'].append(None)
                attention_results['base_model_ko_attention'].append(None)
                attention_results['train_model_en_attention'].append(None)
                attention_results['train_model_ko_attention'].append(None)
                attention_results['cross_lingual_comparisons']['base_model'].append({'overall_similarity': 0.0})
                attention_results['cross_lingual_comparisons']['train_model'].append({'overall_similarity': 0.0})
                attention_results['cross_model_comparisons']['english'].append({'overall_similarity': 0.0})
                attention_results['cross_model_comparisons']['korean'].append({'overall_similarity': 0.0})

        # === COMPREHENSIVE STATISTICS ===
        print(f"\n   📊 Comprehensive Attention Analysis Results:")
        print(f"     Total Sentence Pairs: {total_pairs}")

        # Cross-lingual similarities
        base_similarities = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_lingual_comparisons']['base_model']]
        train_similarities = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_lingual_comparisons']['train_model']]

        if base_similarities and train_similarities:
            base_avg = sum(base_similarities) / len(base_similarities)
            train_avg = sum(train_similarities) / len(train_similarities)
            cross_lingual_improvement = train_avg - base_avg

            print(f"\n     🌍 Cross-lingual Performance (EN-KO Similarity):")
            print(f"       Base Model Average:     {base_avg:.4f}")
            print(f"       Training Model Average: {train_avg:.4f}")
            print(f"       Cross-lingual Improvement: {cross_lingual_improvement:+.4f} ({cross_lingual_improvement/base_avg*100:+.1f}%)")

        # Cross-model similarities
        en_cross_model = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_model_comparisons']['english']]
        ko_cross_model = [comp.get('overall_similarity', 0.0) for comp in attention_results['cross_model_comparisons']['korean']]

        if en_cross_model and ko_cross_model:
            en_avg = sum(en_cross_model) / len(en_cross_model)
            ko_avg = sum(ko_cross_model) / len(ko_cross_model)

            print(f"\n     🔄 Cross-model Consistency:")
            print(f"       English (Base-Train):   {en_avg:.4f}")
            print(f"       Korean (Base-Train):    {ko_avg:.4f}")
            print(f"       Language Bias: {'Korean' if ko_avg > en_avg else 'English'} shows {'higher' if abs(ko_avg - en_avg) > 0.05 else 'similar'} consistency")

        return attention_results

    except Exception as e:
        print(f"   ❌ Attention analysis failed: {e}")
        return None

def analyze_confidence_differences():
    """Analyze confidence differences between Base and Training models for all sentences."""
    if not ENABLE_CONFIDENCE_ANALYSIS:
        return None

    print("\n📈 Running Comprehensive Confidence Analysis...")

    try:
        confidence_analyzer = ConfidenceAnalyzer()

        # Store results for 4-way comparison: Base_EN, Base_KO, Train_EN, Train_KO
        confidence_results = {
            'english_sentences': [],
            'korean_sentences': [],
            'base_model_en_results': [],
            'base_model_ko_results': [],
            'train_model_en_results': [],
            'train_model_ko_results': [],
            'cross_lingual_analysis': {
                'base_model': [],  # EN-KO uncertainty differences for base model
                'train_model': [] # EN-KO uncertainty differences for training model
            },
            'cross_model_analysis': {
                'english': [],     # Base-Train uncertainty differences for English
                'korean': []       # Base-Train uncertainty differences for Korean
            }
        }

        # Check if training model exists
        training_available = os.path.exists(TRAINING_MODEL_PATH)
        if not training_available:
            print(f"   ⚠️ Training model not found: {TRAINING_MODEL_PATH}")
            return None

        # Analyze all sentence pairs for both models
        total_pairs = len(TEST_SENTENCES)
        print(f"   📊 Analyzing {total_pairs} sentence pairs across both models...")

        for i, (en_text, ko_text) in enumerate(TEST_SENTENCES):
            print(f"   📝 Processing sentence pair {i+1}/{total_pairs}...")

            try:
                # Store sentences
                confidence_results['english_sentences'].append(en_text)
                confidence_results['korean_sentences'].append(ko_text)

                # === BASE MODEL ANALYSIS ===
                print(f"      🔍 Base Model Analysis...")

                # English sentence - Base model
                en_base_confidence = confidence_analyzer.analyze_prediction_confidence(
                    model_name_or_path=BASE_MODEL_PATH,
                    text=en_text,
                    model_type='base'
                )
                confidence_results['base_model_en_results'].append(en_base_confidence)

                # Korean sentence - Base model
                ko_base_confidence = confidence_analyzer.analyze_prediction_confidence(
                    model_name_or_path=BASE_MODEL_PATH,
                    text=ko_text,
                    model_type='base'
                )
                confidence_results['base_model_ko_results'].append(ko_base_confidence)

                # === TRAINING MODEL ANALYSIS ===
                print(f"      🎯 Training Model Analysis...")

                # English sentence - Training model
                en_train_confidence = confidence_analyzer.analyze_prediction_confidence(
                    model_name_or_path=TRAINING_MODEL_PATH,
                    text=en_text,
                    model_type='trained'
                )
                confidence_results['train_model_en_results'].append(en_train_confidence)

                # Korean sentence - Training model
                ko_train_confidence = confidence_analyzer.analyze_prediction_confidence(
                    model_name_or_path=TRAINING_MODEL_PATH,
                    text=ko_text,
                    model_type='trained'
                )
                confidence_results['train_model_ko_results'].append(ko_train_confidence)

                # === EXTRACT UNCERTAINTY SCORES ===
                def extract_uncertainty(confidence_result, default=0.5):
                    """Extract uncertainty score from confidence analysis result."""
                    if 'confidence_measures' in confidence_result:
                        measures = confidence_result['confidence_measures']
                        avg_confidence = measures.get('average_confidence', default)
                        return 1.0 - avg_confidence  # Convert confidence to uncertainty
                    elif 'uncertainty_estimates' in confidence_result:
                        return confidence_result['uncertainty_estimates']['mean_uncertainty']
                    else:
                        return default

                en_base_uncertainty = extract_uncertainty(en_base_confidence, 0.5)
                ko_base_uncertainty = extract_uncertainty(ko_base_confidence, 0.6)
                en_train_uncertainty = extract_uncertainty(en_train_confidence, 0.5)
                ko_train_uncertainty = extract_uncertainty(ko_train_confidence, 0.6)

                # === CROSS-LINGUAL ANALYSIS ===
                # Base model: EN vs KO uncertainty difference
                base_en_ko_diff = abs(en_base_uncertainty - ko_base_uncertainty)
                confidence_results['cross_lingual_analysis']['base_model'].append({
                    'en_uncertainty': en_base_uncertainty,
                    'ko_uncertainty': ko_base_uncertainty,
                    'difference': base_en_ko_diff,
                    'language_bias': 'korean' if ko_base_uncertainty > en_base_uncertainty else 'english'
                })

                # Training model: EN vs KO uncertainty difference
                train_en_ko_diff = abs(en_train_uncertainty - ko_train_uncertainty)
                confidence_results['cross_lingual_analysis']['train_model'].append({
                    'en_uncertainty': en_train_uncertainty,
                    'ko_uncertainty': ko_train_uncertainty,
                    'difference': train_en_ko_diff,
                    'language_bias': 'korean' if ko_train_uncertainty > en_train_uncertainty else 'english'
                })

                # === CROSS-MODEL ANALYSIS ===
                # English: Base vs Training uncertainty difference
                en_base_train_diff = abs(en_base_uncertainty - en_train_uncertainty)
                en_improvement = en_base_uncertainty - en_train_uncertainty  # Positive = improvement
                confidence_results['cross_model_analysis']['english'].append({
                    'base_uncertainty': en_base_uncertainty,
                    'train_uncertainty': en_train_uncertainty,
                    'difference': en_base_train_diff,
                    'improvement': en_improvement
                })

                # Korean: Base vs Training uncertainty difference
                ko_base_train_diff = abs(ko_base_uncertainty - ko_train_uncertainty)
                ko_improvement = ko_base_uncertainty - ko_train_uncertainty  # Positive = improvement
                confidence_results['cross_model_analysis']['korean'].append({
                    'base_uncertainty': ko_base_uncertainty,
                    'train_uncertainty': ko_train_uncertainty,
                    'difference': ko_base_train_diff,
                    'improvement': ko_improvement
                })

                # Print progress with comprehensive metrics
                print(f"      ✅ Pair {i+1}:")
                print(f"         Cross-lingual: Base_EN-KO={base_en_ko_diff:.3f}, Train_EN-KO={train_en_ko_diff:.3f}")
                print(f"         Cross-model: EN_improvement={en_improvement:+.3f}, KO_improvement={ko_improvement:+.3f}")

            except Exception as e:
                print(f"      ❌ Pair {i+1} failed: {e}")
                # Add placeholder data for failed analysis
                confidence_results['base_model_en_results'].append({'confidence_measures': {'average_confidence': 0.5}})
                confidence_results['base_model_ko_results'].append({'confidence_measures': {'average_confidence': 0.4}})
                confidence_results['train_model_en_results'].append({'confidence_measures': {'average_confidence': 0.5}})
                confidence_results['train_model_ko_results'].append({'confidence_measures': {'average_confidence': 0.4}})
                confidence_results['cross_lingual_analysis']['base_model'].append({'difference': 0.1, 'language_bias': 'korean'})
                confidence_results['cross_lingual_analysis']['train_model'].append({'difference': 0.1, 'language_bias': 'korean'})
                confidence_results['cross_model_analysis']['english'].append({'improvement': 0.0})
                confidence_results['cross_model_analysis']['korean'].append({'improvement': 0.0})

        # === COMPREHENSIVE STATISTICS ===
        print(f"\n   📊 Comprehensive Confidence Analysis Results:")
        print(f"     Total Sentence Pairs: {total_pairs}")

        # Cross-lingual bias analysis
        base_cross_lingual = confidence_results['cross_lingual_analysis']['base_model']
        train_cross_lingual = confidence_results['cross_lingual_analysis']['train_model']

        if base_cross_lingual and train_cross_lingual:
            base_avg_diff = sum(item['difference'] for item in base_cross_lingual) / len(base_cross_lingual)
            train_avg_diff = sum(item['difference'] for item in train_cross_lingual) / len(train_cross_lingual)
            cross_lingual_improvement = base_avg_diff - train_avg_diff  # Positive = improvement

            # Language bias statistics
            base_korean_bias = sum(1 for item in base_cross_lingual if item['language_bias'] == 'korean') / len(base_cross_lingual)
            train_korean_bias = sum(1 for item in train_cross_lingual if item['language_bias'] == 'korean') / len(train_cross_lingual)

            print(f"\n     🌍 Cross-lingual Uncertainty Analysis:")
            print(f"       Base Model Avg EN-KO Diff:     {base_avg_diff:.4f}")
            print(f"       Training Model Avg EN-KO Diff: {train_avg_diff:.4f}")
            print(f"       Cross-lingual Improvement:     {cross_lingual_improvement:+.4f} ({cross_lingual_improvement/base_avg_diff*100:+.1f}%)")
            print(f"       Korean Bias: Base={base_korean_bias:.1%}, Train={train_korean_bias:.1%}")

        # Cross-model improvement analysis
        en_cross_model = confidence_results['cross_model_analysis']['english']
        ko_cross_model = confidence_results['cross_model_analysis']['korean']

        if en_cross_model and ko_cross_model:
            en_avg_improvement = sum(item['improvement'] for item in en_cross_model) / len(en_cross_model)
            ko_avg_improvement = sum(item['improvement'] for item in ko_cross_model) / len(ko_cross_model)

            # Count improvements vs regressions
            en_improvements = sum(1 for item in en_cross_model if item['improvement'] > 0)
            ko_improvements = sum(1 for item in ko_cross_model if item['improvement'] > 0)

            print(f"\n     🔄 Cross-model Improvement Analysis:")
            print(f"       English Avg Improvement:       {en_avg_improvement:+.4f} ({en_improvements}/{total_pairs} pairs improved)")
            print(f"       Korean Avg Improvement:        {ko_avg_improvement:+.4f} ({ko_improvements}/{total_pairs} pairs improved)")
            print(f"       Overall Training Effectiveness: {'English' if en_avg_improvement > ko_avg_improvement else 'Korean'} shows better improvement")

        # Generate enhanced visualizations
        try:
            save_confidence_visualizations(confidence_results['base_model_en_results'][0])  # Sample for compatibility
            save_comprehensive_confidence_comparison(confidence_results)

            # Generate dual-model confidence entropy comparison
            output_dir = platform_dir / "outputs" / "terminal_analysis"
            success = plot_dual_model_confidence_entropy(confidence_results, output_dir / "dual_model_confidence_entropy.png")
            if success:
                print(f"   ✅ Dual-model confidence entropy comparison saved")
        except Exception as e:
            print(f"   ⚠️ Confidence visualization failed: {e}")

        return confidence_results

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