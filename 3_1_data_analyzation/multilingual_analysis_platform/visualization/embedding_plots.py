"""
Embedding Visualization Module

Provides visualization functions for sentence embeddings including 2D/3D scatter plots,
similarity heatmaps, and interactive visualizations using matplotlib and Plotly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path

from utils.config_loader import get_config
from utils.font_manager import get_font_manager, configure_plot_korean
from core.sentence_embedding import SentenceEmbeddingAnalyzer

logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """Visualizes sentence embeddings and analysis results."""

    def __init__(self):
        """Initialize the embedding visualizer."""
        self.config = get_config()
        self.analyzer = SentenceEmbeddingAnalyzer()
        self.font_manager = get_font_manager()

        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette(self.config.get('visualization.plotting.color_palette', 'husl'))

    def plot_embeddings_2d(self, embeddings: np.ndarray,
                          languages: List[str],
                          texts: Optional[List[str]] = None,
                          method: str = 'umap',
                          title: str = "Sentence Embeddings 2D Visualization",
                          save_path: Optional[str] = None,
                          interactive: bool = False,
                          **reduction_kwargs) -> Union[plt.Figure, go.Figure]:
        """
        Create 2D visualization of sentence embeddings.

        Args:
            embeddings: High-dimensional embeddings
            languages: Language codes for each embedding
            texts: Optional text content for hover information
            method: Dimensionality reduction method
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive Plotly visualization
            **reduction_kwargs: Additional arguments for dimensionality reduction

        Returns:
            Matplotlib or Plotly figure
        """
        # Reduce dimensions to 2D
        reduced_embeddings = self.analyzer.reduce_dimensions(
            embeddings, method=method, n_components=2, **reduction_kwargs
        )

        # Get language colors
        unique_languages = list(set(languages))
        colors = {}
        for i, lang in enumerate(unique_languages):
            colors[lang] = self.config.get_language_color(lang)

        if interactive:
            return self._plot_interactive_2d(
                reduced_embeddings, languages, texts, colors, title, save_path
            )
        else:
            return self._plot_static_2d(
                reduced_embeddings, languages, texts, colors, title, save_path, method
            )

    def _plot_static_2d(self, reduced_embeddings: np.ndarray,
                       languages: List[str],
                       texts: Optional[List[str]],
                       colors: Dict[str, str],
                       title: str,
                       save_path: Optional[str],
                       method: str) -> plt.Figure:
        """Create static 2D plot with matplotlib."""
        fig, ax = plt.subplots(figsize=self.config.get('visualization.plotting.figure_size', [12, 8]))

        # Configure Korean fonts using font manager
        configure_plot_korean(fig, ax)

        # Plot points by language
        for lang in colors.keys():
            mask = np.array(languages) == lang
            if np.any(mask):
                ax.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c=colors[lang],
                    label=lang,
                    alpha=0.7,
                    s=80
                )

        # Add sentence index annotations
        if texts and len(texts) <= 50:
            for i, (x, y) in enumerate(reduced_embeddings):
                # Show sentence index and truncated text
                display_text = f"{i}: {texts[i][:25]}{'...' if len(texts[i]) > 25 else ''}"
                ax.annotate(
                    display_text,
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7),
                    fontfamily='monospace'
                )
        else:
            # If too many texts, just show indices
            for i, (x, y) in enumerate(reduced_embeddings):
                ax.annotate(
                    str(i),
                    (x, y),
                    xytext=(3, 3),
                    textcoords='offset points',
                    fontsize=10,
                    alpha=0.9,
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor='white', alpha=0.8),
                    ha='center',
                    va='center'
                )

        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.get('visualization.plotting.dpi', 300),
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def _plot_interactive_2d(self, reduced_embeddings: np.ndarray,
                           languages: List[str],
                           texts: Optional[List[str]],
                           colors: Dict[str, str],
                           title: str,
                           save_path: Optional[str]) -> go.Figure:
        """Create interactive 2D plot with Plotly."""
        # Create DataFrame for Plotly
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'language': languages
        })

        if texts:
            df['text'] = texts
            df['hover_text'] = [f"Language: {lang}<br>Text: {text[:100]}..."
                              for lang, text in zip(languages, texts)]
        else:
            df['hover_text'] = [f"Language: {lang}" for lang in languages]

        # Create color mapping
        color_discrete_map = colors

        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='language',
            hover_data=['hover_text'],
            color_discrete_map=color_discrete_map,
            title=title
        )

        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            width=1000,
            height=700,
            hovermode='closest'
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive plot saved to {save_path}")

        return fig

    def plot_embeddings_3d(self, embeddings: np.ndarray,
                          languages: List[str],
                          texts: Optional[List[str]] = None,
                          method: str = 'umap',
                          title: str = "Sentence Embeddings 3D Visualization",
                          save_path: Optional[str] = None,
                          **reduction_kwargs) -> go.Figure:
        """
        Create 3D visualization of sentence embeddings.

        Args:
            embeddings: High-dimensional embeddings
            languages: Language codes for each embedding
            texts: Optional text content for hover information
            method: Dimensionality reduction method
            title: Plot title
            save_path: Path to save the plot
            **reduction_kwargs: Additional arguments for dimensionality reduction

        Returns:
            Plotly 3D figure
        """
        # Reduce dimensions to 3D
        reduced_embeddings = self.analyzer.reduce_dimensions(
            embeddings, method=method, n_components=3, **reduction_kwargs
        )

        # Get language colors
        unique_languages = list(set(languages))
        colors = {}
        for lang in unique_languages:
            colors[lang] = self.config.get_language_color(lang)

        # Create DataFrame for Plotly
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'z': reduced_embeddings[:, 2],
            'language': languages
        })

        if texts:
            df['text'] = texts
            df['hover_text'] = [f"Language: {lang}<br>Text: {text[:100]}..."
                              for lang, text in zip(languages, texts)]
        else:
            df['hover_text'] = [f"Language: {lang}" for lang in languages]

        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='language',
            hover_data=['hover_text'],
            color_discrete_map=colors,
            title=title
        )

        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            scene=dict(
                xaxis_title=f'{method.upper()} 1',
                yaxis_title=f'{method.upper()} 2',
                zaxis_title=f'{method.upper()} 3'
            ),
            width=1000,
            height=700,
            title=title
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"3D plot saved to {save_path}")

        return fig

    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray,
                              languages: List[str],
                              texts: Optional[List[str]] = None,
                              title: str = "Similarity Heatmap",
                              save_path: Optional[str] = None,
                              interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Create similarity heatmap visualization.

        Args:
            similarity_matrix: Pairwise similarity matrix
            languages: Language codes
            texts: Optional text labels
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive visualization

        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            return self._plot_interactive_heatmap(
                similarity_matrix, languages, texts, title, save_path
            )
        else:
            return self._plot_static_heatmap(
                similarity_matrix, languages, texts, title, save_path
            )

    def _plot_static_heatmap(self, similarity_matrix: np.ndarray,
                           languages: List[str],
                           texts: Optional[List[str]],
                           title: str,
                           save_path: Optional[str]) -> plt.Figure:
        """Create static heatmap with matplotlib."""
        # Font should already be set globally, but ensure it's set
        if 'font.family' not in plt.rcParams or 'DejaVu Sans' in plt.rcParams['font.family']:
            plt.rcParams['font.family'] = ['NanumGothic', 'Malgun Gothic', 'Gulim', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # Create labels with index and text
        if texts and len(texts) <= 20:
            labels = []
            for i, (lang, text) in enumerate(zip(languages, texts)):
                # Truncate text and add index
                truncated_text = text[:20] + ('...' if len(text) > 20 else '')
                labels.append(f"{i}:{lang} {truncated_text}")
        else:
            labels = [f"{i}:{lang}" for i, lang in enumerate(languages)]

        fig, ax = plt.subplots(figsize=(14, 12))

        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)

        # Set ticks and labels
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        # Add text annotations for small matrices
        if similarity_matrix.shape[0] <= 20:
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)

        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")

        return fig

    def _plot_interactive_heatmap(self, similarity_matrix: np.ndarray,
                                languages: List[str],
                                texts: Optional[List[str]],
                                title: str,
                                save_path: Optional[str]) -> go.Figure:
        """Create interactive heatmap with Plotly."""
        # Create labels
        if texts:
            labels = [f"{lang}: {text[:50]}..." for lang, text in zip(languages, texts)]
        else:
            labels = [f"{lang}_{i}" for i, lang in enumerate(languages)]

        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            colorbar=dict(title="Cosine Similarity"),
            text=np.round(similarity_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Sentences",
            yaxis_title="Sentences",
            width=800,
            height=800
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive heatmap saved to {save_path}")

        return fig

    def plot_language_comparison(self, comparison_result: Dict[str, Any],
                               title: str = "Cross-Language Sentence Comparison",
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize cross-language sentence comparison results.

        Args:
            comparison_result: Result from compare_sentence_pairs
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sentence Pair Similarities',
                'Cross-Language Similarity Matrix',
                'Best Match Analysis',
                'Similarity Distribution'
            ),
            specs=[[{"secondary_y": False}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )

        # Plot 1: Sentence pair similarities
        pair_similarities = comparison_result['pair_similarities']
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pair_similarities))),
                y=pair_similarities,
                mode='markers+lines',
                name='Pair Similarity'
            ),
            row=1, col=1
        )

        # Plot 2: Cross-language similarity matrix
        cross_sim = comparison_result['cross_similarity_matrix']
        fig.add_trace(
            go.Heatmap(
                z=cross_sim,
                colorscale='Viridis',
                showscale=False
            ),
            row=1, col=2
        )

        # Plot 3: Best match analysis
        best_matches = comparison_result['best_matches']
        correct_count = sum(1 for match in best_matches if match['is_correct_pair'])
        incorrect_count = len(best_matches) - correct_count

        fig.add_trace(
            go.Bar(
                x=['Correct Matches', 'Incorrect Matches'],
                y=[correct_count, incorrect_count],
                name='Match Accuracy'
            ),
            row=2, col=1
        )

        # Plot 4: Similarity distribution
        fig.add_trace(
            go.Histogram(
                x=pair_similarities,
                nbinsx=20,
                name='Similarity Distribution'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title=title,
            showlegend=False
        )

        # Update axis titles
        fig.update_xaxes(title_text="Sentence Index", row=1, col=1)
        fig.update_yaxes(title_text="Similarity", row=1, col=1)
        fig.update_xaxes(title_text="Target Language Sentences", row=1, col=2)
        fig.update_yaxes(title_text="English Sentences", row=1, col=2)
        fig.update_xaxes(title_text="Match Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Similarity Score", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Language comparison plot saved to {save_path}")

        return fig

    def plot_clustering_analysis(self, embeddings: np.ndarray,
                                languages: List[str],
                                cluster_result: Dict[str, Any],
                                reduction_method: str = 'umap',
                                title: str = "Language Clustering Analysis",
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize clustering analysis results.

        Args:
            embeddings: Original embeddings
            languages: Language codes
            cluster_result: Result from analyze_language_clusters
            reduction_method: Method for dimensionality reduction
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Plotly figure
        """
        # Reduce dimensions for visualization
        reduced_embeddings = self.analyzer.reduce_dimensions(
            embeddings, method=reduction_method, n_components=2
        )

        # Create DataFrame
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'language': languages,
            'cluster': cluster_result['cluster_labels']
        })

        # Create scatter plot colored by both language and cluster
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Colored by Language', 'Colored by Cluster')
        )

        # Plot colored by language
        for lang in set(languages):
            mask = df['language'] == lang
            fig.add_trace(
                go.Scatter(
                    x=df[mask]['x'],
                    y=df[mask]['y'],
                    mode='markers',
                    name=f'Lang: {lang}',
                    marker=dict(color=self.config.get_language_color(lang), size=8)
                ),
                row=1, col=1
            )

        # Plot colored by cluster
        for cluster in set(cluster_result['cluster_labels']):
            mask = df['cluster'] == cluster
            fig.add_trace(
                go.Scatter(
                    x=df[mask]['x'],
                    y=df[mask]['y'],
                    mode='markers',
                    name=f'Cluster: {cluster}',
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=1, col=2
            )

        fig.update_layout(
            height=500,
            title=title,
            showlegend=True
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Clustering analysis plot saved to {save_path}")

        return fig