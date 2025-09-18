"""
Attention Visualization Module

Provides visualization functions for attention patterns including heatmaps,
head comparisons, and interactive visualizations compatible with BertViz style.
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
from core.attention_analysis import AttentionAnalyzer

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualizes attention patterns and analysis results."""

    def __init__(self):
        """Initialize the attention visualizer."""
        self.config = get_config()
        self.analyzer = AttentionAnalyzer()

        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    def plot_attention_heatmap(self, attention_result: Dict[str, Any],
                             layer_idx: int = 0,
                             head_idx: Optional[int] = None,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None,
                             interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Create attention heatmap visualization.

        Args:
            attention_result: Result from AttentionAnalyzer.extract_attention_weights
            layer_idx: Layer index to visualize
            head_idx: Head index to visualize (None for average across heads)
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive Plotly visualization

        Returns:
            Matplotlib or Plotly figure
        """
        tokens = attention_result['tokens']
        attention_weights = attention_result['attention_weights']

        layer_name = f'layer_{layer_idx}'
        if layer_name not in attention_weights:
            raise ValueError(f"Layer {layer_idx} not found in attention weights")

        layer_attention = attention_weights[layer_name]['attention_weights']

        # Select specific head or average across heads
        if head_idx is not None:
            if head_idx >= layer_attention.shape[0]:
                raise ValueError(f"Head {head_idx} not found (max: {layer_attention.shape[0] - 1})")
            attention_matrix = layer_attention[head_idx]
            title_suffix = f" - Layer {layer_idx}, Head {head_idx}"
        else:
            attention_matrix = np.mean(layer_attention, axis=0)
            title_suffix = f" - Layer {layer_idx}, Average Heads"

        if title is None:
            title = f"Attention Heatmap{title_suffix}"

        if interactive:
            return self._plot_interactive_attention_heatmap(
                attention_matrix, tokens, title, save_path
            )
        else:
            return self._plot_static_attention_heatmap(
                attention_matrix, tokens, title, save_path
            )

    def _plot_static_attention_heatmap(self, attention_matrix: np.ndarray,
                                     tokens: List[str],
                                     title: str,
                                     save_path: Optional[str]) -> plt.Figure:
        """Create static attention heatmap with matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')

        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        # Add text annotations for small matrices
        if attention_matrix.shape[0] <= 20:
            for i in range(attention_matrix.shape[0]):
                for j in range(attention_matrix.shape[1]):
                    text = ax.text(j, i, f'{attention_matrix[i, j]:.3f}',
                                 ha="center", va="center",
                                 color="white" if attention_matrix[i, j] > 0.5 else "black",
                                 fontsize=8)

        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {save_path}")

        return fig

    def _plot_interactive_attention_heatmap(self, attention_matrix: np.ndarray,
                                          tokens: List[str],
                                          title: str,
                                          save_path: Optional[str]) -> go.Figure:
        """Create interactive attention heatmap with Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale='Blues',
            colorbar=dict(title="Attention Weight"),
            text=np.round(attention_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=800,
            height=800
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive attention heatmap saved to {save_path}")

        return fig

    def plot_multi_head_attention(self, attention_result: Dict[str, Any],
                                layer_idx: int = 0,
                                max_heads: int = 8,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot attention patterns for multiple heads in a layer.

        Args:
            attention_result: Result from AttentionAnalyzer.extract_attention_weights
            layer_idx: Layer index to visualize
            max_heads: Maximum number of heads to display
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        tokens = attention_result['tokens']
        attention_weights = attention_result['attention_weights']

        layer_name = f'layer_{layer_idx}'
        if layer_name not in attention_weights:
            raise ValueError(f"Layer {layer_idx} not found in attention weights")

        layer_attention = attention_weights[layer_name]['attention_weights']
        num_heads = min(layer_attention.shape[0], max_heads)

        # Calculate subplot dimensions
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            attention_matrix = layer_attention[head_idx]

            # Create heatmap
            im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')

            # Set labels for smaller matrices
            if len(tokens) <= 15:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            ax.set_title(f'Head {head_idx}', fontsize=12)

        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)

        if title is None:
            title = f'Multi-Head Attention - Layer {layer_idx}'

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Multi-head attention plot saved to {save_path}")

        return fig

    def plot_attention_patterns(self, pattern_analysis: Dict[str, Any],
                              pattern_type: str = 'head_entropy',
                              title: Optional[str] = None,
                              save_path: Optional[str] = None,
                              interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Visualize attention pattern analysis results.

        Args:
            pattern_analysis: Result from AttentionAnalyzer.analyze_attention_patterns
            pattern_type: Type of pattern to visualize
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive visualization

        Returns:
            Matplotlib or Plotly figure
        """
        if pattern_type not in pattern_analysis:
            raise ValueError(f"Pattern type '{pattern_type}' not found in analysis results")

        if pattern_type == 'head_entropy':
            return self._plot_head_entropy(
                pattern_analysis[pattern_type], title, save_path, interactive
            )
        elif pattern_type == 'layer_entropy':
            return self._plot_layer_entropy(
                pattern_analysis[pattern_type], title, save_path, interactive
            )
        elif pattern_type == 'attention_distance':
            return self._plot_attention_distance(
                pattern_analysis[pattern_type], title, save_path, interactive
            )
        elif pattern_type == 'token_importance':
            return self._plot_token_importance(
                pattern_analysis[pattern_type], title, save_path, interactive
            )
        else:
            raise ValueError(f"Visualization not implemented for pattern type: {pattern_type}")

    def _plot_head_entropy(self, head_entropy_data: Dict[str, Any],
                          title: Optional[str],
                          save_path: Optional[str],
                          interactive: bool) -> Union[plt.Figure, go.Figure]:
        """Plot head entropy analysis."""
        if title is None:
            title = "Attention Head Entropy Analysis"

        # Prepare data
        layers = []
        heads = []
        entropies = []

        for layer_name, layer_heads in head_entropy_data.items():
            layer_num = int(layer_name.split('_')[1])
            for head_data in layer_heads:
                layers.append(layer_num)
                heads.append(head_data['head_idx'])
                entropies.append(head_data['mean_entropy'])

        if interactive:
            # Create interactive scatter plot
            fig = go.Figure()

            # Group by layer for different traces
            unique_layers = sorted(set(layers))
            for layer in unique_layers:
                layer_mask = np.array(layers) == layer
                layer_heads = np.array(heads)[layer_mask]
                layer_entropies = np.array(entropies)[layer_mask]

                fig.add_trace(go.Scatter(
                    x=layer_heads,
                    y=layer_entropies,
                    mode='markers',
                    name=f'Layer {layer}',
                    marker=dict(size=10)
                ))

            fig.update_layout(
                title=title,
                xaxis_title='Head Index',
                yaxis_title='Mean Entropy',
                width=800,
                height=600
            )

        else:
            # Create static plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create scatter plot colored by layer
            unique_layers = sorted(set(layers))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))

            for i, layer in enumerate(unique_layers):
                layer_mask = np.array(layers) == layer
                layer_heads = np.array(heads)[layer_mask]
                layer_entropies = np.array(entropies)[layer_mask]

                ax.scatter(layer_heads, layer_entropies,
                          color=colors[i], label=f'Layer {layer}', s=60, alpha=0.7)

            ax.set_xlabel('Head Index')
            ax.set_ylabel('Mean Entropy')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        if save_path:
            if interactive:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                else:
                    fig.write_image(save_path)
            else:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Head entropy plot saved to {save_path}")

        return fig

    def _plot_layer_entropy(self, layer_entropy_data: Dict[str, Any],
                           title: Optional[str],
                           save_path: Optional[str],
                           interactive: bool) -> Union[plt.Figure, go.Figure]:
        """Plot layer entropy analysis."""
        if title is None:
            title = "Layer Entropy Analysis"

        # Prepare data
        layer_nums = []
        mean_entropies = []

        for layer_name, layer_data in layer_entropy_data.items():
            layer_num = int(layer_name.split('_')[1])
            layer_nums.append(layer_num)
            mean_entropies.append(layer_data['mean_entropy'])

        # Sort by layer number
        sorted_data = sorted(zip(layer_nums, mean_entropies))
        layer_nums, mean_entropies = zip(*sorted_data)

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=layer_nums,
                y=mean_entropies,
                mode='markers+lines',
                name='Layer Entropy',
                marker=dict(size=10)
            ))

            fig.update_layout(
                title=title,
                xaxis_title='Layer Index',
                yaxis_title='Mean Entropy',
                width=800,
                height=600
            )

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(layer_nums, mean_entropies, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Mean Entropy')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        if save_path:
            if interactive:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                else:
                    fig.write_image(save_path)
            else:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Layer entropy plot saved to {save_path}")

        return fig

    def _plot_attention_distance(self, distance_data: Dict[str, Any],
                               title: Optional[str],
                               save_path: Optional[str],
                               interactive: bool) -> Union[plt.Figure, go.Figure]:
        """Plot attention distance analysis."""
        if title is None:
            title = "Attention Distance Analysis"

        # Prepare data
        layer_nums = []
        mean_distances = []

        for layer_name, layer_data in distance_data.items():
            layer_num = int(layer_name.split('_')[1])
            layer_nums.append(layer_num)
            mean_distances.append(layer_data['mean_distance'])

        # Sort by layer number
        sorted_data = sorted(zip(layer_nums, mean_distances))
        layer_nums, mean_distances = zip(*sorted_data)

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=layer_nums,
                y=mean_distances,
                mode='markers+lines',
                name='Mean Distance',
                marker=dict(size=10)
            ))

            fig.update_layout(
                title=title,
                xaxis_title='Layer Index',
                yaxis_title='Mean Attention Distance',
                width=800,
                height=600
            )

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(layer_nums, mean_distances, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Mean Attention Distance')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        if save_path:
            if interactive:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                else:
                    fig.write_image(save_path)
            else:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention distance plot saved to {save_path}")

        return fig

    def _plot_token_importance(self, token_importance_data: Dict[str, Any],
                             title: Optional[str],
                             save_path: Optional[str],
                             interactive: bool) -> Union[plt.Figure, go.Figure]:
        """Plot token importance analysis."""
        if title is None:
            title = "Token Importance Analysis"

        # Use first layer for visualization (can be extended to show multiple layers)
        first_layer = list(token_importance_data.keys())[0]
        token_scores = token_importance_data[first_layer]['token_scores']

        # Get top 15 tokens for better visualization
        top_tokens = token_scores[:15]
        tokens = [ts['token'] for ts in top_tokens]
        scores = [ts['importance_score'] for ts in top_tokens]

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tokens,
                y=scores,
                name='Importance Score'
            ))

            fig.update_layout(
                title=f"{title} - {first_layer}",
                xaxis_title='Tokens',
                yaxis_title='Importance Score',
                width=800,
                height=600,
                xaxis_tickangle=-45
            )

        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(tokens)), scores)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_xlabel('Tokens')
            ax.set_ylabel('Importance Score')
            ax.set_title(f"{title} - {first_layer}")

            # Color bars by importance
            normalized_scores = np.array(scores) / max(scores)
            for bar, norm_score in zip(bars, normalized_scores):
                bar.set_color(plt.cm.viridis(norm_score))

            plt.tight_layout()

        if save_path:
            if interactive:
                if save_path.endswith('.html'):
                    fig.write_html(save_path)
                else:
                    fig.write_image(save_path)
            else:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Token importance plot saved to {save_path}")

        return fig

    def plot_attention_comparison(self, comparison_result: Dict[str, Any],
                                title: str = "Attention Pattern Comparison",
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize attention pattern comparison between models.

        Args:
            comparison_result: Result from AttentionAnalyzer.compare_attention_patterns
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Plotly figure
        """
        comparison_metrics = comparison_result['comparison_metrics']

        # Create subplots for different metrics
        metrics = ['cosine_similarity', 'mse', 'correlation']
        available_metrics = [m for m in metrics if any(m in layer_data for layer_data in comparison_metrics.values())]

        fig = make_subplots(
            rows=1, cols=len(available_metrics),
            subplot_titles=available_metrics
        )

        layers = sorted(comparison_metrics.keys(), key=lambda x: int(x.split('_')[1]))

        for i, metric in enumerate(available_metrics):
            layer_nums = []
            metric_values = []

            for layer in layers:
                if metric in comparison_metrics[layer]:
                    layer_num = int(layer.split('_')[1])
                    layer_nums.append(layer_num)
                    metric_values.append(comparison_metrics[layer][metric])

            fig.add_trace(
                go.Scatter(
                    x=layer_nums,
                    y=metric_values,
                    mode='markers+lines',
                    name=metric,
                    showlegend=False
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            title=title,
            height=400,
            width=300 * len(available_metrics)
        )

        # Update axis titles
        for i, metric in enumerate(available_metrics):
            fig.update_xaxes(title_text="Layer Index", row=1, col=i+1)
            fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=1, col=i+1)

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Attention comparison plot saved to {save_path}")

        return fig