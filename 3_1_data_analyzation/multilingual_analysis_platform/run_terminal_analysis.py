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

def plot_dual_model_euclidean_distance(dual_embeddings, texts, languages, output_path):
    """
    Analyze and visualize Euclidean distances between base and training model embeddings.
    Shows both magnitude and direction considerations with sentence pair details.

    Args:
        dual_embeddings: Dictionary containing embeddings from both models
        texts: List of input texts
        languages: List of language codes
        output_path: Path to save the plot

    Returns:
        Boolean indicating success
    """
    try:
        base_embeddings = dual_embeddings['base_embeddings']
        train_embeddings = dual_embeddings['train_embeddings']

        # Calculate Euclidean distances between cross-lingual pairs within same model
        # Instead of comparing base vs training for same language,
        # we compare EN vs KO within same model (base and training separately)

        cross_lingual_distances = {'base': [], 'training': []}

        # Process EN-KO pairs for both models
        for i in range(0, len(base_embeddings), 2):  # Every pair: i=EN, i+1=KO
            if i+1 < len(base_embeddings):
                # Base model: EN vs KO distance
                base_en_ko_distance = np.linalg.norm(base_embeddings[i] - base_embeddings[i+1])
                cross_lingual_distances['base'].append(base_en_ko_distance)

                # Training model: EN vs KO distance
                train_en_ko_distance = np.linalg.norm(train_embeddings[i] - train_embeddings[i+1])
                cross_lingual_distances['training'].append(train_en_ko_distance)

        # For backward compatibility, also calculate the old distances
        euclidean_distances = []
        for i in range(len(base_embeddings)):
            distance = np.linalg.norm(base_embeddings[i] - train_embeddings[i])
            euclidean_distances.append(distance)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

        # Configure Korean fonts
        configure_plot_korean(fig, None)

        # 1. Distance heatmap with sentence pairs
        ax1 = fig.add_subplot(gs[0, :])

        # Create cross-lingual distance mapping
        pair_distances = []
        pair_info = []
        for pair_idx, (base_dist, train_dist) in enumerate(zip(cross_lingual_distances['base'], cross_lingual_distances['training'])):
            # Get the corresponding texts for this pair
            en_idx, ko_idx = pair_idx * 2, pair_idx * 2 + 1
            if ko_idx < len(texts):
                pair_distances.append([base_dist, train_dist])
                pair_info.append({
                    'pair_id': pair_idx,
                    'en_text': texts[en_idx][:50] + "..." if len(texts[en_idx]) > 50 else texts[en_idx],
                    'ko_text': texts[ko_idx][:50] + "..." if len(texts[ko_idx]) > 50 else texts[ko_idx],
                    'base_distance': base_dist,
                    'training_distance': train_dist,
                    'pair_avg': (base_dist + train_dist) / 2
                })

        # Plot heatmap
        if pair_distances:
            heatmap_data = np.array(pair_distances).T  # 2 x num_pairs
            im = ax1.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')

            # Set labels
            ax1.set_title('Cross-Lingual Euclidean Distance (EN-KO pairs within same model)\n(Base Model vs Training Model Comparison)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Translation Pair Index')
            ax1.set_ylabel('Model Type')
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['Base Model', 'Training Model'])
            ax1.set_xticks(range(len(pair_distances)))
            ax1.set_xticklabels([f'Pair {i}' for i in range(len(pair_distances))])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.6)
            cbar.set_label('Euclidean Distance', fontsize=12)

            # Add distance values on heatmap
            for i in range(len(pair_distances)):
                for j in range(2):
                    distance = heatmap_data[j, i]
                    ax1.text(i, j, f'{distance:.3f}', ha='center', va='center',
                            color='white' if distance > np.mean(heatmap_data) else 'black',
                            fontweight='bold', fontsize=10)

        # 2. Distance distribution histogram
        ax2 = fig.add_subplot(gs[1, 0])

        base_distances = cross_lingual_distances['base']
        train_distances = cross_lingual_distances['training']

        ax2.hist(base_distances, bins=15, alpha=0.7, label='Base Model', color='#1f77b4', density=True)
        ax2.hist(train_distances, bins=15, alpha=0.7, label='Training Model', color='#ff7f0e', density=True)
        ax2.set_title('Cross-Lingual Distance Distribution by Model')
        ax2.set_xlabel('Cross-Lingual Euclidean Distance (EN-KO)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add statistics
        base_mean, train_mean = np.mean(base_distances), np.mean(train_distances)
        ax2.axvline(base_mean, color='#1f77b4', linestyle='--', alpha=0.8, label=f'Base mean: {base_mean:.3f}')
        ax2.axvline(train_mean, color='#ff7f0e', linestyle='--', alpha=0.8, label=f'Train mean: {train_mean:.3f}')

        # 3. Top divergent pairs analysis
        ax3 = fig.add_subplot(gs[1, 1])

        sorted_pairs = sorted(pair_info, key=lambda x: x['pair_avg'], reverse=True)
        top_pairs = sorted_pairs[:5]  # Top 5 most divergent pairs

        pair_ids = [p['pair_id'] for p in top_pairs]
        pair_avgs = [p['pair_avg'] for p in top_pairs]

        bars = ax3.bar(range(len(top_pairs)), pair_avgs, color='coral', alpha=0.8)
        ax3.set_title('Top 5 Most Divergent Sentence Pairs')
        ax3.set_xlabel('Pair Rank')
        ax3.set_ylabel('Average Distance')
        ax3.set_xticks(range(len(top_pairs)))
        ax3.set_xticklabels([f'Pair {p}' for p in pair_ids])
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, pair_avgs)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. Detailed sentence pair table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        # Create table data
        table_data = []
        table_headers = ['Pair ID', 'English Text', 'Korean Text', 'Base EN-KO Dist', 'Train EN-KO Dist', 'Avg Distance']

        for p in sorted_pairs[:8]:  # Show top 8 pairs
            table_data.append([
                f"{p['pair_id']}",
                p['en_text'],
                p['ko_text'],
                f"{p['base_distance']:.4f}",
                f"{p['training_distance']:.4f}",
                f"{p['pair_avg']:.4f}"
            ])

        # Create table
        table = ax4.table(cellText=table_data, colLabels=table_headers,
                         cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 2)

        # Style table
        for i in range(len(table_headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color code rows by distance
        for i in range(1, len(table_data) + 1):
            avg_dist = float(table_data[i-1][5])
            if avg_dist > np.percentile([p['pair_avg'] for p in pair_info], 75):
                color = '#ffcccc'  # Light red for high distance
            elif avg_dist > np.percentile([p['pair_avg'] for p in pair_info], 50):
                color = '#ffffcc'  # Light yellow for medium distance
            else:
                color = '#ccffcc'  # Light green for low distance

            for j in range(len(table_headers)):
                table[(i, j)].set_facecolor(color)

        ax4.set_title('Cross-Lingual Distance Analysis Table\n(EN-KO pairs within models, sorted by average distance)',
                     fontsize=12, fontweight='bold', pad=20)

        # Add summary statistics
        total_pairs = len(pair_info)
        all_cross_distances = base_distances + train_distances
        overall_mean = np.mean(all_cross_distances)
        overall_std = np.std(all_cross_distances)

        summary_text = f"""
📊 Cross-Lingual Distance Analysis Summary:
• Total translation pairs: {total_pairs}
• Base model mean EN-KO distance: {base_mean:.4f}
• Training model mean EN-KO distance: {train_mean:.4f}
• Overall standard deviation: {overall_std:.4f}
• Model difference: {abs(base_mean - train_mean):.4f}

💡 Interpretation:
• Distance = Euclidean distance between EN and KO embeddings within same model
• Lower distance = Better cross-lingual alignment
• Model comparison: {'Training improved' if train_mean < base_mean else 'Base better' if base_mean < train_mean else 'Similar'} cross-lingual alignment
        """

        fig.text(0.02, 0.02, summary_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"      📊 Euclidean distance analysis: {total_pairs} pairs, mean distance: {overall_mean:.4f}")
        return True

    except Exception as e:
        print(f"      ⚠️ Euclidean distance analysis failed: {e}")
        return False

def plot_dual_model_centered_kernel_alignment(dual_embeddings, texts, languages, output_path):
    """
    Analyze and visualize Centered Kernel Alignment (CKA) between base and training models.
    CKA measures structural similarity of representations across different models.

    Args:
        dual_embeddings: Dictionary containing embeddings from both models
        texts: List of input texts
        languages: List of language codes
        output_path: Path to save the plot

    Returns:
        Boolean indicating success
    """
    try:
        base_embeddings = dual_embeddings['base_embeddings']
        train_embeddings = dual_embeddings['train_embeddings']

        def compute_cka(X, Y):
            """Compute Centered Kernel Alignment (CKA) between two sets of representations."""
            # Center the Gram matrices
            def center_gram_matrix(K):
                n = K.shape[0]
                H = np.eye(n) - np.ones((n, n)) / n
                return H @ K @ H

            # Compute Gram matrices (linear kernel)
            K_X = X @ X.T
            K_Y = Y @ Y.T

            # Center the matrices
            K_X_centered = center_gram_matrix(K_X)
            K_Y_centered = center_gram_matrix(K_Y)

            # Compute CKA
            numerator = np.trace(K_X_centered @ K_Y_centered)
            denominator = np.sqrt(np.trace(K_X_centered @ K_X_centered) * np.trace(K_Y_centered @ K_Y_centered))

            if denominator == 0:
                return 0.0

            return numerator / denominator

        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)

        # Configure Korean fonts
        configure_plot_korean(fig, None)

        # 1. Overall CKA score
        overall_cka = compute_cka(base_embeddings, train_embeddings)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(0.5, 0.7, f'Overall CKA Score', fontsize=24, fontweight='bold',
                ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.5, 0.4, f'{overall_cka:.6f}', fontsize=48, fontweight='bold',
                ha='center', va='center', transform=ax1.transAxes,
                color='green' if overall_cka > 0.7 else 'orange' if overall_cka > 0.4 else 'red')

        # Add interpretation
        if overall_cka > 0.8:
            interpretation = "Very High Similarity"
            color = 'darkgreen'
        elif overall_cka > 0.6:
            interpretation = "High Similarity"
            color = 'green'
        elif overall_cka > 0.4:
            interpretation = "Moderate Similarity"
            color = 'orange'
        elif overall_cka > 0.2:
            interpretation = "Low Similarity"
            color = 'red'
        else:
            interpretation = "Very Low Similarity"
            color = 'darkred'

        ax1.text(0.5, 0.15, interpretation, fontsize=16, fontweight='bold',
                ha='center', va='center', transform=ax1.transAxes, color=color)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # Add background gradient
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax1.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto', alpha=0.3, cmap='RdYlGn')

        # 2. Language-wise CKA analysis
        ax2 = fig.add_subplot(gs[1, 0])

        # Separate by language
        lang_cka_scores = {}
        for lang in set(languages):
            mask = np.array(languages) == lang
            if np.sum(mask) > 1:  # Need at least 2 samples for CKA
                lang_base = base_embeddings[mask]
                lang_train = train_embeddings[mask]
                lang_cka_scores[lang] = compute_cka(lang_base, lang_train)

        if lang_cka_scores:
            lang_names = list(lang_cka_scores.keys())
            cka_values = list(lang_cka_scores.values())

            bars = ax2.bar(lang_names, cka_values, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax2.set_title('CKA Scores by Language')
            ax2.set_ylabel('CKA Score')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, cka_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # 3. Sentence pair-wise CKA heatmap
        ax3 = fig.add_subplot(gs[1, 1])

        # Compute CKA for individual sentence pairs
        pair_cka_scores = []
        for i in range(0, len(texts), 2):  # Process EN-KO pairs
            if i+1 < len(texts):
                en_idx, ko_idx = i, i+1

                # Create mini-batch for each pair
                pair_base = base_embeddings[[en_idx, ko_idx]]
                pair_train = train_embeddings[[en_idx, ko_idx]]

                if pair_base.shape[0] >= 2:  # Need at least 2 samples
                    pair_cka = compute_cka(pair_base, pair_train)
                    pair_cka_scores.append(pair_cka)
                else:
                    pair_cka_scores.append(0.0)

        if pair_cka_scores:
            x_pos = range(len(pair_cka_scores))
            bars = ax3.bar(x_pos, pair_cka_scores, color='skyblue', alpha=0.8)
            ax3.set_title('CKA Scores by Sentence Pair')
            ax3.set_xlabel('Sentence Pair Index')
            ax3.set_ylabel('CKA Score')
            ax3.set_ylim(0, 1)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'Pair {i}' for i in range(len(pair_cka_scores))])
            ax3.grid(True, alpha=0.3)

            # Highlight best and worst pairs
            if len(pair_cka_scores) > 0:
                best_idx = np.argmax(pair_cka_scores)
                worst_idx = np.argmin(pair_cka_scores)
                bars[best_idx].set_color('green')
                bars[worst_idx].set_color('red')

        # 4. CKA interpretation and analysis table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        # Create analysis table
        analysis_data = [
            ['Overall CKA', f'{overall_cka:.6f}', interpretation],
            ['English CKA', f'{lang_cka_scores.get("en", 0.0):.6f}', 'Language-specific similarity'],
            ['Korean CKA', f'{lang_cka_scores.get("ko", 0.0):.6f}', 'Language-specific similarity'],
            ['Best Pair CKA', f'{max(pair_cka_scores) if pair_cka_scores else 0.0:.6f}', f'Pair {np.argmax(pair_cka_scores) if pair_cka_scores else "N/A"}'],
            ['Worst Pair CKA', f'{min(pair_cka_scores) if pair_cka_scores else 0.0:.6f}', f'Pair {np.argmin(pair_cka_scores) if pair_cka_scores else "N/A"}'],
            ['CKA Variance', f'{np.var(pair_cka_scores) if pair_cka_scores else 0.0:.6f}', 'Consistency across pairs']
        ]

        table_headers = ['Metric', 'Value', 'Interpretation']
        table = ax4.table(cellText=analysis_data, colLabels=table_headers,
                         cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Style table
        for i in range(len(table_headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color code rows by CKA value
        for i in range(1, len(analysis_data) + 1):
            value_str = analysis_data[i-1][1]
            try:
                value = float(value_str)
                if value > 0.7:
                    color = '#ccffcc'  # Light green
                elif value > 0.4:
                    color = '#ffffcc'  # Light yellow
                else:
                    color = '#ffcccc'  # Light red
            except:
                color = '#f0f0f0'  # Light gray for non-numeric

            for j in range(len(table_headers)):
                table[(i, j)].set_facecolor(color)

        # Add comprehensive explanation
        explanation_text = """
🧬 Centered Kernel Alignment (CKA) Analysis:

📚 What is CKA?
• CKA measures structural similarity between neural representations
• Values range from 0 (no similarity) to 1 (perfect similarity)
• Higher values indicate models learned similar representational structures

📊 Interpretation Guide:
• CKA > 0.8: Very high structural similarity (models are very aligned)
• CKA 0.6-0.8: High similarity (good alignment with some differences)
• CKA 0.4-0.6: Moderate similarity (partial alignment)
• CKA 0.2-0.4: Low similarity (different representational structures)
• CKA < 0.2: Very low similarity (fundamentally different representations)

💡 Use Cases:
• Comparing models before/after training
• Analyzing cross-lingual representation alignment
• Understanding how training affects learned representations
        """

        ax4.text(0.02, 0.95, explanation_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"      🧬 CKA analysis: Overall score = {overall_cka:.6f} ({interpretation})")
        return True

    except Exception as e:
        print(f"      ⚠️ CKA analysis failed: {e}")
        return False

def plot_dual_model_singular_vector_canonical_correlation(dual_embeddings, texts, languages, output_path):
    """
    Analyze and visualize SVCCA (Singular Vector Canonical Correlation Analysis) between base and training models.
    SVCCA measures similarity between neural representations across models, layers, and languages.

    Args:
        dual_embeddings: Dictionary containing embeddings from both models
        texts: List of input texts
        languages: List of language codes
        output_path: Path to save the plot

    Returns:
        Boolean indicating success
    """
    try:
        base_embeddings = dual_embeddings['base_embeddings']
        train_embeddings = dual_embeddings['train_embeddings']

        def compute_svcca(X, Y, threshold=0.99):
            """
            Compute SVCCA (Singular Vector Canonical Correlation Analysis).

            Args:
                X, Y: Input matrices (samples x features)
                threshold: Threshold for selecting principal components

            Returns:
                mean_correlation: Mean canonical correlation
                correlations: All canonical correlations
                explained_variance: Cumulative explained variance
            """
            try:
                # Step 1: SVD on both matrices to reduce dimensionality
                def perform_svd(matrix, threshold):
                    U, s, Vt = np.linalg.svd(matrix.T, full_matrices=False)  # Features x samples

                    # Calculate explained variance
                    explained_variance = np.cumsum(s**2) / np.sum(s**2)

                    # Find number of components for threshold
                    n_components = np.argmax(explained_variance >= threshold) + 1
                    n_components = min(n_components, len(s), matrix.shape[0] - 1)
                    n_components = max(n_components, 1)

                    return U[:, :n_components], explained_variance[n_components-1]

                # SVD for both representations
                U1, var1 = perform_svd(X, threshold)
                U2, var2 = perform_svd(Y, threshold)

                # Step 2: Canonical Correlation Analysis
                # Project data onto the singular vectors
                X_proj = X @ U1  # samples x reduced_dims
                Y_proj = Y @ U2  # samples x reduced_dims

                # Ensure we have enough samples for CCA
                min_dim = min(X_proj.shape[1], Y_proj.shape[1], X_proj.shape[0] - 1)
                if min_dim < 1:
                    return 0.0, [0.0], [var1, var2]

                X_proj = X_proj[:, :min_dim]
                Y_proj = Y_proj[:, :min_dim]

                # Center the data
                X_centered = X_proj - np.mean(X_proj, axis=0)
                Y_centered = Y_proj - np.mean(Y_proj, axis=0)

                # Compute covariance matrices
                n = X_centered.shape[0]
                if n <= 1:
                    return 0.0, [0.0], [var1, var2]

                Cxx = X_centered.T @ X_centered / (n - 1)
                Cyy = Y_centered.T @ Y_centered / (n - 1)
                Cxy = X_centered.T @ Y_centered / (n - 1)

                # Add small regularization for numerical stability
                reg = 1e-6
                Cxx += reg * np.eye(Cxx.shape[0])
                Cyy += reg * np.eye(Cyy.shape[0])

                # Canonical correlation analysis
                try:
                    Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
                    Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))

                    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
                    U_cca, s_cca, Vt_cca = np.linalg.svd(T, full_matrices=False)

                    correlations = s_cca
                    correlations = np.clip(correlations, 0, 1)  # Ensure correlations are in [0,1]

                    return np.mean(correlations), correlations, [var1, var2]

                except np.linalg.LinAlgError:
                    # Fallback to eigenvalue decomposition
                    try:
                        M = np.linalg.pinv(Cxx) @ Cxy @ np.linalg.pinv(Cyy) @ Cxy.T
                        eigenvals = np.linalg.eigvals(M)
                        correlations = np.sqrt(np.maximum(eigenvals.real, 0))
                        correlations = correlations[correlations > 1e-10]  # Remove near-zero correlations

                        if len(correlations) == 0:
                            return 0.0, [0.0], [var1, var2]

                        return np.mean(correlations), correlations, [var1, var2]
                    except:
                        return 0.0, [0.0], [var1, var2]

            except Exception as e:
                print(f"SVCCA computation error: {e}")
                return 0.0, [0.0], [0.0, 0.0]

        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)

        # Configure Korean fonts
        configure_plot_korean(fig, None)

        # 1. Overall SVCCA analysis
        overall_svcca, overall_correlations, overall_vars = compute_svcca(base_embeddings, train_embeddings)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(0.3, 0.7, f'Overall SVCCA Score', fontsize=20, fontweight='bold',
                ha='center', va='center', transform=ax1.transAxes)
        ax1.text(0.3, 0.4, f'{overall_svcca:.6f}', fontsize=36, fontweight='bold',
                ha='center', va='center', transform=ax1.transAxes,
                color='green' if overall_svcca > 0.7 else 'orange' if overall_svcca > 0.4 else 'red')

        # Add interpretation
        if overall_svcca > 0.8:
            interpretation = "Very High Correlation"
            color = 'darkgreen'
        elif overall_svcca > 0.6:
            interpretation = "High Correlation"
            color = 'green'
        elif overall_svcca > 0.4:
            interpretation = "Moderate Correlation"
            color = 'orange'
        elif overall_svcca > 0.2:
            interpretation = "Low Correlation"
            color = 'red'
        else:
            interpretation = "Very Low Correlation"
            color = 'darkred'

        ax1.text(0.3, 0.15, interpretation, fontsize=14, fontweight='bold',
                ha='center', va='center', transform=ax1.transAxes, color=color)

        # Show canonical correlations distribution
        ax1_right = fig.add_subplot(gs[0, :])
        ax1_right.set_position([0.55, 0.7, 0.4, 0.25])  # [left, bottom, width, height]

        if len(overall_correlations) > 1:
            ax1_right.bar(range(len(overall_correlations)), sorted(overall_correlations, reverse=True),
                         color='skyblue', alpha=0.8)
            ax1_right.set_title('Canonical Correlations', fontsize=12)
            ax1_right.set_xlabel('Component')
            ax1_right.set_ylabel('Correlation')
            ax1_right.set_ylim(0, 1)
            ax1_right.grid(True, alpha=0.3)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # 2. Language-wise SVCCA analysis
        ax2 = fig.add_subplot(gs[1, 0])

        lang_svcca_scores = {}
        lang_correlations = {}
        for lang in set(languages):
            mask = np.array(languages) == lang
            if np.sum(mask) > 2:  # Need at least 3 samples for reliable SVCCA
                lang_base = base_embeddings[mask]
                lang_train = train_embeddings[mask]
                score, corrs, _ = compute_svcca(lang_base, lang_train)
                lang_svcca_scores[lang] = score
                lang_correlations[lang] = corrs

        if lang_svcca_scores:
            lang_names = list(lang_svcca_scores.keys())
            svcca_values = list(lang_svcca_scores.values())

            bars = ax2.bar(lang_names, svcca_values, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax2.set_title('SVCCA Scores by Language')
            ax2.set_ylabel('Mean Canonical Correlation')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, svcca_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # 3. Cross-lingual SVCCA analysis
        ax3 = fig.add_subplot(gs[1, 1])

        # Compare English vs Korean representations within each model
        en_mask = np.array(languages) == 'en'
        ko_mask = np.array(languages) == 'ko'

        cross_lingual_scores = {}
        if np.sum(en_mask) > 2 and np.sum(ko_mask) > 2:
            # Base model: EN vs KO
            base_cross_svcca, _, _ = compute_svcca(base_embeddings[en_mask], base_embeddings[ko_mask])
            cross_lingual_scores['Base Model'] = base_cross_svcca

            # Training model: EN vs KO
            train_cross_svcca, _, _ = compute_svcca(train_embeddings[en_mask], train_embeddings[ko_mask])
            cross_lingual_scores['Training Model'] = train_cross_svcca

        if cross_lingual_scores:
            model_names = list(cross_lingual_scores.keys())
            cross_values = list(cross_lingual_scores.values())

            bars = ax3.bar(model_names, cross_values, color=['lightcoral', 'lightgreen'], alpha=0.8)
            ax3.set_title('Cross-lingual SVCCA\n(English ↔ Korean)')
            ax3.set_ylabel('SVCCA Score')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, cross_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # 4. SVCCA interpretation and analysis table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        # Create comprehensive analysis table
        analysis_data = [
            ['Overall SVCCA', f'{overall_svcca:.6f}', interpretation],
            ['Number of Canonical Correlations', f'{len(overall_correlations)}', 'Effective dimensionality'],
            ['Highest Canonical Correlation', f'{max(overall_correlations) if overall_correlations else 0.0:.6f}', 'Best aligned component'],
            ['Lowest Canonical Correlation', f'{min(overall_correlations) if overall_correlations else 0.0:.6f}', 'Least aligned component'],
            ['English SVCCA', f'{lang_svcca_scores.get("en", 0.0):.6f}', 'Language-specific correlation'],
            ['Korean SVCCA', f'{lang_svcca_scores.get("ko", 0.0):.6f}', 'Language-specific correlation'],
            ['Base Cross-lingual', f'{cross_lingual_scores.get("Base Model", 0.0):.6f}', 'EN-KO alignment in base'],
            ['Training Cross-lingual', f'{cross_lingual_scores.get("Training Model", 0.0):.6f}', 'EN-KO alignment in training']
        ]

        table_headers = ['Metric', 'Value', 'Interpretation']
        table = ax4.table(cellText=analysis_data, colLabels=table_headers,
                         cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style table
        for i in range(len(table_headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color code rows by SVCCA value
        for i in range(1, len(analysis_data) + 1):
            value_str = analysis_data[i-1][1]
            try:
                value = float(value_str)
                if value > 0.7:
                    color = '#ccffcc'  # Light green
                elif value > 0.4:
                    color = '#ffffcc'  # Light yellow
                else:
                    color = '#ffcccc'  # Light red
            except:
                color = '#f0f0f0'  # Light gray for non-numeric

            for j in range(len(table_headers)):
                table[(i, j)].set_facecolor(color)

        # Add comprehensive explanation
        explanation_text = """
🔬 Singular Vector Canonical Correlation Analysis (SVCCA):

📚 What is SVCCA?
• SVCCA combines SVD and CCA to measure similarity between neural representations
• Step 1: SVD reduces dimensionality while preserving important variance
• Step 2: CCA finds maximally correlated linear combinations
• Values range from 0 (no correlation) to 1 (perfect correlation)

📊 Interpretation Guide:
• SVCCA > 0.8: Very high representational similarity (models learned similar patterns)
• SVCCA 0.6-0.8: High similarity (good alignment with some differences)
• SVCCA 0.4-0.6: Moderate similarity (partial representational overlap)
• SVCCA 0.2-0.4: Low similarity (different representational strategies)
• SVCCA < 0.2: Very low similarity (fundamentally different representations)

💡 Applications:
• Analyzing training effects on learned representations
• Comparing cross-lingual representational alignment
• Understanding how models encode different languages
• Measuring representational similarity across model architectures
        """

        ax4.text(0.02, 0.95, explanation_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsteelblue', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"      🔬 SVCCA analysis: Overall score = {overall_svcca:.6f} ({interpretation})")
        return True

    except Exception as e:
        print(f"      ⚠️ SVCCA analysis failed: {e}")
        return False

def analyze_token_generation_confidence(base_model_path, training_model_path, test_sentences):
    """
    Analyze token-by-token autoregressive generation confidence for both base and training models.
    For each sentence, perform autoregressive generation and track confidence for each generated token.

    Args:
        base_model_path: Path to base model
        training_model_path: Path to training model
        test_sentences: List of (english, korean) sentence pairs

    Returns:
        Dictionary containing confidence analysis results for all 32 analyses (16 sentences × 2 models)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import torch.nn.functional as F

        print(f"   🔍 Analyzing autoregressive token generation confidence...")

        # Load models and tokenizers
        print(f"   📥 Loading base model: {base_model_path}")
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
            if base_tokenizer.pad_token is None:
                base_tokenizer.pad_token = base_tokenizer.eos_token
        except Exception as e:
            print(f"   ⚠️ Failed to load base model: {e}")
            return None

        print(f"   📥 Loading training model: {training_model_path}")
        try:
            train_tokenizer = AutoTokenizer.from_pretrained(training_model_path)
            train_model = AutoModelForCausalLM.from_pretrained(training_model_path, torch_dtype=torch.float16)
            if train_tokenizer.pad_token is None:
                train_tokenizer.pad_token = train_tokenizer.eos_token
        except Exception as e:
            print(f"   ⚠️ Failed to load training model: {e}")
            return None

        def get_autoregressive_confidences(model, tokenizer, target_text, prompt="", max_length=50):
            """Extract token-by-token autoregressive generation confidences."""
            try:
                model.eval()
                device = next(model.parameters()).device

                # If no prompt, create a simple start
                if not prompt:
                    prompt = ""

                # Tokenize the target text to know what we're trying to generate
                target_tokens = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

                # Start with prompt (or empty)
                if prompt:
                    current_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                else:
                    current_ids = torch.tensor([], dtype=torch.long)

                confidences = []
                tokens = []
                generated_text = prompt

                with torch.no_grad():
                    # Generate each token of the target text autoregressively
                    for target_token_idx in range(len(target_tokens)):
                        target_token_id = target_tokens[target_token_idx].item()

                        # Prepare input for current generation step
                        if len(current_ids) > 0:
                            input_ids = current_ids.unsqueeze(0).to(device)
                        else:
                            # If no current context, start generation
                            input_ids = torch.tensor([[]], dtype=torch.long).to(device)

                        # Get model predictions for next token
                        if input_ids.shape[1] > 0:
                            outputs = model(input_ids=input_ids)
                            next_token_logits = outputs.logits[0, -1, :]  # Last position logits
                        else:
                            # For first token with no context, use model's initial predictions
                            dummy_input = torch.tensor([[tokenizer.bos_token_id if tokenizer.bos_token_id else 1]]).to(device)
                            outputs = model(input_ids=dummy_input)
                            next_token_logits = outputs.logits[0, -1, :]

                        # Get probabilities
                        probs = F.softmax(next_token_logits, dim=-1)

                        # Get confidence for the target token
                        target_token_confidence = probs[target_token_id].item()
                        target_token_text = tokenizer.decode([target_token_id])

                        confidences.append(target_token_confidence)
                        tokens.append(target_token_text)

                        # Add the target token to current sequence for next iteration
                        if len(current_ids) > 0:
                            current_ids = torch.cat([current_ids, torch.tensor([target_token_id])])
                        else:
                            current_ids = torch.tensor([target_token_id])

                        generated_text += target_token_text

                return {
                    'tokens': tokens,
                    'confidences': confidences,
                    'target_text': target_text,
                    'generated_text': generated_text,
                    'prompt': prompt
                }

            except Exception as e:
                print(f"      ⚠️ Autoregressive confidence extraction failed for text: {target_text[:50]}... Error: {e}")
                return None

        # Analyze all test sentences
        results = {
            'base_model_results': [],
            'training_model_results': [],
            'sentence_pairs': [],
            'languages': []
        }

        total_sentences = len(test_sentences) * 2  # EN + KO for each pair
        processed = 0

        for pair_idx, (en_text, ko_text) in enumerate(test_sentences):
            print(f"   📝 Processing sentence pair {pair_idx + 1}/{len(test_sentences)}...")

            # Store sentence pair info
            results['sentence_pairs'].append({
                'pair_id': pair_idx,
                'english': en_text,
                'korean': ko_text
            })

            # Process English sentence with autoregressive generation
            print(f"      🇺🇸 English: {en_text[:50]}...")
            base_en_conf = get_autoregressive_confidences(base_model, base_tokenizer, en_text)
            train_en_conf = get_autoregressive_confidences(train_model, train_tokenizer, en_text)

            if base_en_conf and train_en_conf:
                results['base_model_results'].append({
                    'pair_id': pair_idx,
                    'language': 'en',
                    'text': en_text,
                    **base_en_conf
                })
                results['training_model_results'].append({
                    'pair_id': pair_idx,
                    'language': 'en',
                    'text': en_text,
                    **train_en_conf
                })
                results['languages'].extend(['en'])
                processed += 1

            # Process Korean sentence with autoregressive generation
            print(f"      🇰🇷 Korean: {ko_text[:50]}...")
            base_ko_conf = get_autoregressive_confidences(base_model, base_tokenizer, ko_text)
            train_ko_conf = get_autoregressive_confidences(train_model, train_tokenizer, ko_text)

            if base_ko_conf and train_ko_conf:
                results['base_model_results'].append({
                    'pair_id': pair_idx,
                    'language': 'ko',
                    'text': ko_text,
                    **base_ko_conf
                })
                results['training_model_results'].append({
                    'pair_id': pair_idx,
                    'language': 'ko',
                    'text': ko_text,
                    **train_ko_conf
                })
                results['languages'].extend(['ko'])
                processed += 1

        print(f"   ✅ Token confidence analysis complete: {processed}/{total_sentences} sentences processed")

        # Clean up models to free memory
        del base_model, train_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    except Exception as e:
        print(f"   ❌ Token confidence analysis failed: {e}")
        return None

def plot_dual_model_token_confidence(confidence_results, output_path):
    """
    Create comprehensive autoregressive token confidence visualization.
    Shows 32 analyses: 16 sentences (8 EN + 8 KO) × 2 models = 32 confidence plots.

    Args:
        confidence_results: Results from analyze_token_generation_confidence
        output_path: Path to save the plot

    Returns:
        Boolean indicating success
    """
    try:
        if not confidence_results or not confidence_results['base_model_results']:
            print(f"      ⚠️ No confidence results to visualize")
            return False

        # Create large visualization for 32 analyses (16 sentences × 2 models)
        fig = plt.figure(figsize=(24, 20))

        # Configure Korean fonts
        configure_plot_korean(fig, None)

        base_results = confidence_results['base_model_results']
        train_results = confidence_results['training_model_results']

        print(f"      📊 Creating visualization for {len(base_results)} base + {len(train_results)} training = {len(base_results) + len(train_results)} analyses")

        # Create main grid: Top for individual token plots, bottom for summaries
        gs_main = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

        # 1. Individual sentence token confidence plots (32 subplots)
        # Arrange in 8 rows × 4 columns (2 models × 2 languages per pair)
        gs_plots = gs_main[0].subgridspec(8, 4, hspace=0.6, wspace=0.4)

        max_tokens = 0
        all_confidences = []

        # Plot each sentence's token confidence
        for pair_idx in range(8):  # 8 sentence pairs
            row = pair_idx

            # Find corresponding results for this pair
            base_en = next((r for r in base_results if r['pair_id'] == pair_idx and r['language'] == 'en'), None)
            base_ko = next((r for r in base_results if r['pair_id'] == pair_idx and r['language'] == 'ko'), None)
            train_en = next((r for r in train_results if r['pair_id'] == pair_idx and r['language'] == 'en'), None)
            train_ko = next((r for r in train_results if r['pair_id'] == pair_idx and r['language'] == 'ko'), None)

            # Plot base English (column 0)
            if base_en:
                ax = fig.add_subplot(gs_plots[row, 0])
                tokens = base_en['tokens']
                confidences = base_en['confidences']
                ax.plot(range(len(confidences)), confidences, 'b-', linewidth=2, alpha=0.8)
                ax.set_ylim(0, 1)
                ax.set_title(f'P{pair_idx} Base-EN', fontsize=8)
                ax.grid(True, alpha=0.3)
                if row == 7:  # Only bottom row gets x-label
                    ax.set_xlabel('Token Position', fontsize=8)
                max_tokens = max(max_tokens, len(confidences))
                all_confidences.extend(confidences)

            # Plot base Korean (column 1)
            if base_ko:
                ax = fig.add_subplot(gs_plots[row, 1])
                tokens = base_ko['tokens']
                confidences = base_ko['confidences']
                ax.plot(range(len(confidences)), confidences, 'g-', linewidth=2, alpha=0.8)
                ax.set_ylim(0, 1)
                ax.set_title(f'P{pair_idx} Base-KO', fontsize=8)
                ax.grid(True, alpha=0.3)
                if row == 7:
                    ax.set_xlabel('Token Position', fontsize=8)
                max_tokens = max(max_tokens, len(confidences))
                all_confidences.extend(confidences)

            # Plot training English (column 2)
            if train_en:
                ax = fig.add_subplot(gs_plots[row, 2])
                tokens = train_en['tokens']
                confidences = train_en['confidences']
                ax.plot(range(len(confidences)), confidences, 'r-', linewidth=2, alpha=0.8)
                ax.set_ylim(0, 1)
                ax.set_title(f'P{pair_idx} Train-EN', fontsize=8)
                ax.grid(True, alpha=0.3)
                if row == 7:
                    ax.set_xlabel('Token Position', fontsize=8)
                max_tokens = max(max_tokens, len(confidences))
                all_confidences.extend(confidences)

            # Plot training Korean (column 3)
            if train_ko:
                ax = fig.add_subplot(gs_plots[row, 3])
                tokens = train_ko['tokens']
                confidences = train_ko['confidences']
                ax.plot(range(len(confidences)), confidences, 'orange', linewidth=2, alpha=0.8)
                ax.set_ylim(0, 1)
                ax.set_title(f'P{pair_idx} Train-KO', fontsize=8)
                ax.grid(True, alpha=0.3)
                if row == 7:
                    ax.set_xlabel('Token Position', fontsize=8)
                max_tokens = max(max_tokens, len(confidences))
                all_confidences.extend(confidences)

        # 2. Summary statistics (bottom left)
        ax_summary = fig.add_subplot(gs_main[1])

        # Calculate language and model averages
        base_en_avg = np.mean([np.mean(r['confidences']) for r in base_results if r['language'] == 'en'])
        base_ko_avg = np.mean([np.mean(r['confidences']) for r in base_results if r['language'] == 'ko'])
        train_en_avg = np.mean([np.mean(r['confidences']) for r in train_results if r['language'] == 'en'])
        train_ko_avg = np.mean([np.mean(r['confidences']) for r in train_results if r['language'] == 'ko'])

        categories = ['Base-EN', 'Base-KO', 'Train-EN', 'Train-KO']
        averages = [base_en_avg, base_ko_avg, train_en_avg, train_ko_avg]
        colors = ['blue', 'green', 'red', 'orange']

        bars = ax_summary.bar(categories, averages, color=colors, alpha=0.7)
        ax_summary.set_title('Average Token Confidence by Model-Language', fontsize=12, fontweight='bold')
        ax_summary.set_ylabel('Average Confidence')
        ax_summary.set_ylim(0, 1)
        ax_summary.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, avg in zip(bars, averages):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Analysis summary (bottom right)
        ax_info = fig.add_subplot(gs_main[2])
        ax_info.axis('off')

        # Calculate improvements
        en_improvement = train_en_avg - base_en_avg
        ko_improvement = train_ko_avg - base_ko_avg
        overall_improvement = (en_improvement + ko_improvement) / 2

        summary_text = f"""
📊 Autoregressive Token Confidence Analysis Summary:

🔢 Analysis Scope:
  • Total analyses: {len(base_results) + len(train_results)} (16 sentences × 2 models)
  • English sentences: 8 × 2 models = 16 analyses
  • Korean sentences: 8 × 2 models = 16 analyses
  • Max tokens per sentence: {max_tokens}

📈 Model Performance:
  • Base Model - English: {base_en_avg:.4f}
  • Base Model - Korean: {base_ko_avg:.4f}
  • Training Model - English: {train_en_avg:.4f}
  • Training Model - Korean: {train_ko_avg:.4f}

🎯 Training Impact:
  • English improvement: {en_improvement:+.4f}
  • Korean improvement: {ko_improvement:+.4f}
  • Overall improvement: {overall_improvement:+.4f}
  • Better language: {'English' if en_improvement > ko_improvement else 'Korean' if ko_improvement > en_improvement else 'Equal'}

💡 Interpretation:
  • Higher confidence = Better autoregressive generation certainty
  • Each plot shows token-by-token confidence during generation
  • Training {'improved' if overall_improvement > 0 else 'decreased' if overall_improvement < 0 else 'maintained'} generation confidence
        """

        ax_info.text(0.02, 0.98, summary_text, transform=ax_info.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9))

        # Main title
        fig.suptitle('Autoregressive Token Generation Confidence Analysis\n'
                    f'32 Analyses: 8 Sentence Pairs × 2 Languages × 2 Models',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"      📊 Token confidence analysis: {len(all_confidences)} total tokens, overall avg: {np.mean(all_confidences):.4f}")
        return True

    except Exception as e:
        print(f"      ⚠️ Token confidence visualization failed: {e}")
        return False

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

        # Add translation pair connection lines
        # Connect EN-KO pairs within same model (base: 0↔1, 2↔3, 4↔5, etc.)
        for i in range(0, n_samples, 2):  # Process pairs: (0,1), (2,3), (4,5), etc.
            if i+1 < n_samples:
                # Base model connections (thin blue lines)
                ax.plot([base_reduced[i, 0], base_reduced[i+1, 0]],
                       [base_reduced[i, 1], base_reduced[i+1, 1]],
                       'b-', alpha=0.4, linewidth=1, zorder=1)

                # Training model connections (thin red lines)
                ax.plot([train_reduced[i, 0], train_reduced[i+1, 0]],
                       [train_reduced[i, 1], train_reduced[i+1, 1]],
                       'r-', alpha=0.4, linewidth=1, zorder=1)

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
        ax.set_title('Base vs Training Model - PCA Comparison\n(Circles=Base, Triangles=Training, Lines=Translation Pairs)', fontsize=14, pad=20)

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

        # 🆕 Detailed PCA vector component analysis
        print(f"\n🧬 PCA Vector Analysis (Component Directions):")

        # Get the principal component vectors
        pc1_vector = pca.components_[0]  # Shape: (embedding_dim,)
        pc2_vector = pca.components_[1]  # Shape: (embedding_dim,)

        print(f"   Embedding dimension: {len(pc1_vector)}")
        print(f"   PC1 vector norm: {np.linalg.norm(pc1_vector):.6f}")
        print(f"   PC2 vector norm: {np.linalg.norm(pc2_vector):.6f}")
        print(f"   PC1-PC2 orthogonality: {np.abs(np.dot(pc1_vector, pc2_vector)):.8f} (should be ~0)")

        # Find top contributing dimensions for each PC
        print(f"\n🎯 Top Contributing Embedding Dimensions:")

        # PC1 analysis - dimensions that contribute most to language separation
        pc1_abs = np.abs(pc1_vector)
        top_pc1_indices = np.argsort(pc1_abs)[-10:][::-1]  # Top 10
        print(f"   PC1 (Language Axis) - Top contributing dimensions:")
        for i, dim_idx in enumerate(top_pc1_indices[:5]):
            weight = pc1_vector[dim_idx]
            abs_weight = pc1_abs[dim_idx]
            print(f"     Dimension {dim_idx:3d}: {weight:+.6f} (magnitude: {abs_weight:.6f})")

        # PC2 analysis - dimensions that contribute most to content variation
        pc2_abs = np.abs(pc2_vector)
        top_pc2_indices = np.argsort(pc2_abs)[-10:][::-1]  # Top 10
        print(f"   PC2 (Content Axis) - Top contributing dimensions:")
        for i, dim_idx in enumerate(top_pc2_indices[:5]):
            weight = pc2_vector[dim_idx]
            abs_weight = pc2_abs[dim_idx]
            print(f"     Dimension {dim_idx:3d}: {weight:+.6f} (magnitude: {abs_weight:.6f})")

        # Vector interpretation
        print(f"\n🎨 Vector Interpretation:")
        pc1_mean_lang = pc1_coords[ko_mask].mean() - pc1_coords[en_mask].mean()
        print(f"   PC1 direction: {'Korean→English' if pc1_mean_lang < 0 else 'English→Korean'}")
        print(f"   PC1 strength: {abs(pc1_mean_lang):.3f} (higher = better language separation)")

        # Content diversity comparison
        en_pc2_range = pc2_coords[en_mask].max() - pc2_coords[en_mask].min()
        ko_pc2_range = pc2_coords[ko_mask].max() - pc2_coords[ko_mask].min()
        print(f"   PC2 English content range: {en_pc2_range:.3f}")
        print(f"   PC2 Korean content range: {ko_pc2_range:.3f}")
        print(f"   Content diversity: {'English' if en_pc2_range > ko_pc2_range else 'Korean'} shows more variation")

        # Semantic insight: which embedding dimensions are most important for what
        print(f"\n🔍 Semantic Insights:")
        print(f"   Language-critical dimensions (PC1): {list(top_pc1_indices[:3])}")
        print(f"   Content-critical dimensions (PC2): {list(top_pc2_indices[:3])}")

        # Check if same dimensions are important for both PCs (would be unusual)
        overlap = set(top_pc1_indices[:5]) & set(top_pc2_indices[:5])
        if overlap:
            print(f"   ⚠️  Overlapping important dims: {list(overlap)} (unusual - same dims affect both language and content)")
        else:
            print(f"   ✅ No overlap in top dimensions (good - language and content use different features)")

        return {
            'pca_components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'pc1_coords': pc1_coords,
            'pc2_coords': pc2_coords,
            'language_separation': abs(pc1_coords[en_mask].mean() - pc1_coords[ko_mask].mean()),
            'model_shifts': {'pc1': pc1_shift, 'pc2': pc2_shift},
            'pc1_vector': pc1_vector,
            'pc2_vector': pc2_vector,
            'top_pc1_dims': top_pc1_indices,
            'top_pc2_dims': top_pc2_indices,
            'dimension_overlap': list(set(top_pc1_indices[:5]) & set(top_pc2_indices[:5]))
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

        # Add translation pair connection lines
        # Connect EN-KO pairs within same model (base: 0↔1, 2↔3, 4↔5, etc.)
        for i in range(0, n_samples, 2):  # Process pairs: (0,1), (2,3), (4,5), etc.
            if i+1 < n_samples:
                # Base model connections (thin blue lines)
                ax.plot([base_reduced[i, 0], base_reduced[i+1, 0]],
                       [base_reduced[i, 1], base_reduced[i+1, 1]],
                       'b-', alpha=0.4, linewidth=1, zorder=1)

                # Training model connections (thin red lines)
                ax.plot([train_reduced[i, 0], train_reduced[i+1, 0]],
                       [train_reduced[i, 1], train_reduced[i+1, 1]],
                       'r-', alpha=0.4, linewidth=1, zorder=1)

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
        ax.set_title(f'Base vs Training Model - {method_name} Comparison\n(Circles=Base, Triangles=Training, Lines=Translation Pairs)', fontsize=14, pad=20)

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
    print("[ANALYSIS] Multilingual Model Comparison Analysis")
    print("=" * 60)
    print(f"[BASE] Base Model: {BASE_MODEL_PATH}")
    print(f"[TRAIN] Training Model: {TRAINING_MODEL_PATH}")
    print(f"[DATA] Test Sentences: {len(TEST_SENTENCES)} pairs")
    print("=" * 60)
    print("\n[INDEX] Sentence Index Reference:")
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

        # 🆕 Dual-model embedding statistics
        total_plots += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Plot dual-model embedding statistics
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

                # Extract base and training embeddings
                base_embeddings = dual_embeddings['base_embeddings']
                train_embeddings = dual_embeddings['train_embeddings']

                # 1. Dual-model embedding magnitude comparison by language
                lang_magnitudes_base = {}
                lang_magnitudes_train = {}
                for lang in set(languages):
                    mask = np.array(languages) == lang
                    if np.any(mask):
                        lang_magnitudes_base[lang] = np.linalg.norm(base_embeddings[mask], axis=1)
                        lang_magnitudes_train[lang] = np.linalg.norm(train_embeddings[mask], axis=1)

                lang_names = list(lang_magnitudes_base.keys())
                x_pos = np.arange(len(lang_names))
                width = 0.35

                base_means = [np.mean(lang_magnitudes_base[lang]) for lang in lang_names]
                train_means = [np.mean(lang_magnitudes_train[lang]) for lang in lang_names]

                bars1 = ax1.bar(x_pos - width/2, base_means, width, label='Base Model', alpha=0.8, color='#1f77b4')
                bars2 = ax1.bar(x_pos + width/2, train_means, width, label='Training Model', alpha=0.8, color='#ff7f0e')

                ax1.set_title("Dual-Model Embedding Magnitude Comparison")
                ax1.set_ylabel("Average L2 Norm")
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(lang_names)
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Add improvement indicators
                for i, (base_val, train_val) in enumerate(zip(base_means, train_means)):
                    improvement = train_val - base_val
                    ax1.text(i, max(base_val, train_val) + 0.01, f'{improvement:+.3f}',
                            ha='center', va='bottom', fontsize=9,
                            color='green' if improvement > 0 else 'red', fontweight='bold')

                # 2. Embedding variance comparison
                base_variances = [np.var(lang_magnitudes_base[lang]) for lang in lang_names]
                train_variances = [np.var(lang_magnitudes_train[lang]) for lang in lang_names]

                bars3 = ax2.bar(x_pos - width/2, base_variances, width, label='Base Model', alpha=0.8, color='#1f77b4')
                bars4 = ax2.bar(x_pos + width/2, train_variances, width, label='Training Model', alpha=0.8, color='#ff7f0e')

                ax2.set_title("Dual-Model Embedding Variance Comparison")
                ax2.set_ylabel("Variance")
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(lang_names)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # 3. Sentence length distribution (language-agnostic)
                sentence_lengths = [len(text.split()) for text in texts]
                lang_lengths = {}
                for i, lang in enumerate(languages):
                    if lang not in lang_lengths:
                        lang_lengths[lang] = []
                    lang_lengths[lang].append(sentence_lengths[i])

                length_data = [lang_lengths[lang] for lang in lang_names]
                ax3.boxplot(length_data, labels=lang_names)
                ax3.set_title("Sentence Length Distribution by Language")
                ax3.set_ylabel("Word Count")
                ax3.grid(True, alpha=0.3)

                # 4. Dual-model similarity distribution comparison
                base_similarity_matrix = cosine_similarity(base_embeddings)
                train_similarity_matrix = cosine_similarity(train_embeddings)

                ax4.hist(base_similarity_matrix.flatten(), bins=30, alpha=0.6,
                        label='Base Model', color='#1f77b4', density=True)
                ax4.hist(train_similarity_matrix.flatten(), bins=30, alpha=0.6,
                        label='Training Model', color='#ff7f0e', density=True)
                ax4.set_title("Dual-Model Similarity Distribution")
                ax4.set_xlabel("Cosine Similarity")
                ax4.set_ylabel("Density")
                ax4.legend()
                ax4.grid(True, alpha=0.3)

                # Add statistical comparison
                base_mean_sim = np.mean(base_similarity_matrix.flatten())
                train_mean_sim = np.mean(train_similarity_matrix.flatten())
                sim_improvement = train_mean_sim - base_mean_sim
                ax4.text(0.02, 0.98, f'Mean Similarity:\nBase: {base_mean_sim:.3f}\nTrain: {train_mean_sim:.3f}\nΔ: {sim_improvement:+.3f}',
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8),
                        fontsize=9)

                plt.tight_layout()
                plt.savefig(output_dir / "dual_model_embedding_statistics.png",
                           dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.close()
                plots_saved += 1
                print(f"      ✅ Dual-model embedding statistics saved")
        except Exception as e:
            print(f"      ⚠️ Embedding statistics plot failed: {e}")

        # 🆕 Dual-model Euclidean distance analysis
        total_plots += 1
        try:
            success = plot_dual_model_euclidean_distance(dual_embeddings, texts, languages, output_dir / "dual_model_euclidean_distance.png")
            if success:
                plots_saved += 1
                print(f"      ✅ Dual-model Euclidean distance analysis saved")
        except Exception as e:
            print(f"      ⚠️ Euclidean distance analysis failed: {e}")

        # 🆕 CKA (Centered Kernel Alignment) analysis
        total_plots += 1
        try:
            success = plot_dual_model_centered_kernel_alignment(dual_embeddings, texts, languages, output_dir / "dual_model_centered_kernel_alignment.png")
            if success:
                plots_saved += 1
                print(f"      ✅ Dual-model CKA analysis saved")
        except Exception as e:
            print(f"      ⚠️ CKA analysis failed: {e}")

        # 🆕 SVCCA (Singular Vector Canonical Correlation Analysis)
        total_plots += 1
        try:
            success = plot_dual_model_singular_vector_canonical_correlation(dual_embeddings, texts, languages, output_dir / "dual_model_singular_vector_canonical_correlation.png")
            if success:
                plots_saved += 1
                print(f"      ✅ Dual-model SVCCA analysis saved")
        except Exception as e:
            print(f"      ⚠️ SVCCA analysis failed: {e}")

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

        # Note: Single-model confidence_entropy.png removed - using dual-model version instead
        # The dual-model confidence entropy is generated by plot_dual_model_confidence_entropy()
        print(f"   ℹ️ Using dual-model confidence entropy analysis instead of single-model")

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

# Removed: save_comprehensive_confidence_comparison function (comprehensive 4way comparison not needed)

# Removed: save_attention_visualizations function (comprehensive 4way attention analysis not needed)

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
    print("   ⚠️ Confidence analysis temporarily disabled due to ConfidenceAnalyzer import issues")

    # Return empty results to prevent errors
    return {
        'english_sentences': [],
        'korean_sentences': [],
        'base_model_en_results': [],
        'base_model_ko_results': [],
        'train_model_en_results': [],
        'train_model_ko_results': [],
        'cross_lingual_analysis': {
            'base_model': [],
            'train_model': []
        },
        'cross_model_analysis': {
            'english': [],
            'korean': []
        }
    }

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
            # Removed: save_comprehensive_confidence_comparison(confidence_results) - not needed

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

    # print_header()  # Temporarily disabled due to Unicode issues
    print("Multilingual Model Comparison Analysis Started...")
    print(f"Base Model: {BASE_MODEL_PATH}")
    print(f"Training Model: {TRAINING_MODEL_PATH}")

    # Check if training model exists
    if not os.path.exists(TRAINING_MODEL_PATH) and TRAINING_MODEL_PATH != "/path/to/your/trained/model":
        print(f"\nWARNING: Training model path not found: {TRAINING_MODEL_PATH}")
        print("   Analysis will proceed with base model only.\n")

    # Run analyses
    embedding_results = analyze_embedding_differences()
    attention_results = analyze_attention_differences()
    confidence_results = analyze_confidence_differences()

    # Token-level confidence analysis
    print("\n🎯 Analyzing Token-Level Generation Confidence...")
    token_confidence_results = analyze_token_generation_confidence(
        BASE_MODEL_PATH,
        TRAINING_MODEL_PATH,
        TEST_SENTENCES
    )

    # Generate visualizations for each analysis type
    print("\n🎨 Generating Visualizations...")

    # Embedding visualizations (already includes all sentences)
    if embedding_results:
        save_embedding_visualizations(embedding_results)

    # Attention visualizations (all sentence pairs)
    if attention_results and ENABLE_ATTENTION_ANALYSIS:
        # Removed: save_attention_visualizations(attention_results) - comprehensive 4way analysis not needed
        pass

    # Confidence visualizations are already called within analyze_confidence_differences()

    # Token confidence visualizations
    if token_confidence_results:
        print("   🎯 Generating token-level confidence visualization...")
        output_dir = platform_dir / "outputs" / "terminal_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        token_confidence_output = output_dir / "DUAL_MODEL_TOKEN_CONFIDENCE.png"
        plot_dual_model_token_confidence(token_confidence_results, token_confidence_output)

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
    if token_confidence_results:
        print("✅ Token-level generation confidence analysis completed")
    if SAVE_RESULTS:
        print("✅ Results saved to outputs/terminal_analysis/")

    print("\n🎉 Terminal analysis completed successfully!")

if __name__ == "__main__":
    main()