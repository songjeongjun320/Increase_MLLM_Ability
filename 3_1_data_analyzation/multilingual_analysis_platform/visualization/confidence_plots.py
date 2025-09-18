"""
Confidence Visualization Module

Provides visualization functions for token prediction confidence including entropy plots,
uncertainty heatmaps, and comparative confidence analysis visualizations.
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

from ..utils.config_loader import get_config
from ..core.confidence_analysis import ConfidenceAnalyzer

logger = logging.getLogger(__name__)


class ConfidenceVisualizer:
    """Visualizes confidence analysis results and uncertainty patterns."""

    def __init__(self):
        """Initialize the confidence visualizer."""
        self.config = get_config()
        self.analyzer = ConfidenceAnalyzer()

        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    def plot_entropy_by_position(self, confidence_result: Dict[str, Any],
                                title: str = "Entropy by Token Position",
                                save_path: Optional[str] = None,
                                interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot entropy values by token position.

        Args:
            confidence_result: Result from ConfidenceAnalyzer.analyze_prediction_confidence
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive visualization

        Returns:
            Matplotlib or Plotly figure
        """
        if 'entropy' not in confidence_result['confidence_measures']:
            raise ValueError("Entropy analysis not found in confidence result")

        entropy_data = confidence_result['confidence_measures']['entropy']
        tokens = confidence_result['tokens']
        position_entropies = entropy_data['position_entropies']

        if interactive:
            return self._plot_interactive_entropy(
                position_entropies, tokens, title, save_path
            )
        else:
            return self._plot_static_entropy(
                position_entropies, tokens, entropy_data, title, save_path
            )

    def _plot_static_entropy(self, position_entropies: List[float],
                           tokens: List[str],
                           entropy_data: Dict[str, Any],
                           title: str,
                           save_path: Optional[str]) -> plt.Figure:
        """Create static entropy plot with matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Entropy by position
        positions = range(len(position_entropies))
        ax1.plot(positions, position_entropies, 'o-', linewidth=2, markersize=6)

        # Highlight high and low uncertainty positions
        mean_entropy = entropy_data['mean_entropy']
        std_entropy = entropy_data['std_entropy']

        # Threshold lines
        ax1.axhline(y=mean_entropy, color='gray', linestyle='--', alpha=0.7, label='Mean')
        ax1.axhline(y=mean_entropy + std_entropy, color='red', linestyle='--', alpha=0.7, label='High Uncertainty')
        ax1.axhline(y=mean_entropy - std_entropy, color='green', linestyle='--', alpha=0.7, label='Low Uncertainty')

        # Color code points based on uncertainty level
        colors = []
        for entropy_val in position_entropies:
            if entropy_val > mean_entropy + std_entropy:
                colors.append('red')
            elif entropy_val < mean_entropy - std_entropy:
                colors.append('green')
            else:
                colors.append('blue')

        ax1.scatter(positions, position_entropies, c=colors, s=60, alpha=0.7, zorder=5)

        # Set token labels if not too many
        if len(tokens) <= 20:
            ax1.set_xticks(positions)
            ax1.set_xticklabels(tokens, rotation=45, ha='right')
        else:
            ax1.set_xlabel('Token Position')

        ax1.set_ylabel('Entropy')
        ax1.set_title(f'{title} - Sequence View')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Entropy distribution
        ax2.hist(position_entropies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=mean_entropy, color='gray', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(x=mean_entropy + std_entropy, color='red', linestyle='--', linewidth=2, label='High Threshold')
        ax2.axvline(x=mean_entropy - std_entropy, color='green', linestyle='--', linewidth=2, label='Low Threshold')

        ax2.set_xlabel('Entropy Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Entropy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entropy plot saved to {save_path}")

        return fig

    def _plot_interactive_entropy(self, position_entropies: List[float],
                                tokens: List[str],
                                title: str,
                                save_path: Optional[str]) -> go.Figure:
        """Create interactive entropy plot with Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Entropy by Position', 'Entropy Distribution'),
            vertical_spacing=0.12
        )

        # Plot 1: Entropy by position
        positions = list(range(len(position_entropies)))

        fig.add_trace(
            go.Scatter(
                x=positions,
                y=position_entropies,
                mode='markers+lines',
                name='Entropy',
                text=tokens,
                hovertemplate='Position: %{x}<br>Token: %{text}<br>Entropy: %{y:.3f}<extra></extra>',
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        # Add threshold lines
        mean_entropy = np.mean(position_entropies)
        std_entropy = np.std(position_entropies)

        fig.add_hline(y=mean_entropy, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=mean_entropy + std_entropy, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=mean_entropy - std_entropy, line_dash="dash", line_color="green", row=1, col=1)

        # Plot 2: Entropy distribution
        fig.add_trace(
            go.Histogram(
                x=position_entropies,
                nbinsx=20,
                name='Distribution',
                showlegend=False
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=title,
            height=700,
            width=900
        )

        fig.update_xaxes(title_text="Token Position", row=1, col=1)
        fig.update_yaxes(title_text="Entropy", row=1, col=1)
        fig.update_xaxes(title_text="Entropy Value", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive entropy plot saved to {save_path}")

        return fig

    def plot_confidence_heatmap(self, confidence_result: Dict[str, Any],
                              metric: str = 'entropy',
                              title: Optional[str] = None,
                              save_path: Optional[str] = None,
                              interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Create a heatmap showing confidence values across token positions.

        Args:
            confidence_result: Result from confidence analysis
            metric: Confidence metric to visualize
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive visualization

        Returns:
            Matplotlib or Plotly figure
        """
        if metric not in confidence_result['confidence_measures']:
            raise ValueError(f"Metric '{metric}' not found in confidence measures")

        metric_data = confidence_result['confidence_measures'][metric]
        tokens = confidence_result['tokens']

        if title is None:
            title = f"Confidence Heatmap - {metric.title()}"

        # Get position-wise values
        if metric == 'entropy':
            values = metric_data['position_entropies']
            colorscale = 'Reds'  # Higher entropy = less confident = red
        elif metric == 'variance':
            values = metric_data['position_variances']
            colorscale = 'Reds'
        elif metric == 'top_k_probability':
            values = metric_data['position_top_k_confidence']
            colorscale = 'Greens'  # Higher probability = more confident = green
        elif metric == 'perplexity':
            values = metric_data['position_perplexities']
            colorscale = 'Reds'  # Higher perplexity = less confident = red
        else:
            raise ValueError(f"Visualization not implemented for metric: {metric}")

        # Reshape for heatmap (single row)
        heatmap_data = np.array(values).reshape(1, -1)

        if interactive:
            return self._plot_interactive_confidence_heatmap(
                heatmap_data, tokens, values, title, colorscale, save_path
            )
        else:
            return self._plot_static_confidence_heatmap(
                heatmap_data, tokens, values, title, colorscale, save_path
            )

    def _plot_static_confidence_heatmap(self, heatmap_data: np.ndarray,
                                      tokens: List[str],
                                      values: List[float],
                                      title: str,
                                      colorscale: str,
                                      save_path: Optional[str]) -> plt.Figure:
        """Create static confidence heatmap."""
        fig, ax = plt.subplots(figsize=(16, 4))

        # Create heatmap
        cmap = plt.cm.get_cmap(colorscale.lower())
        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Confidence Value')

        # Set token labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([])

        # Add value annotations
        for i, (token, value) in enumerate(zip(tokens, values)):
            ax.text(i, 0, f'{value:.2f}', ha='center', va='center',
                   color='white' if value > np.mean(values) else 'black',
                   fontsize=10, weight='bold')

        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence heatmap saved to {save_path}")

        return fig

    def _plot_interactive_confidence_heatmap(self, heatmap_data: np.ndarray,
                                           tokens: List[str],
                                           values: List[float],
                                           title: str,
                                           colorscale: str,
                                           save_path: Optional[str]) -> go.Figure:
        """Create interactive confidence heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=tokens,
            y=['Confidence'],
            colorscale=colorscale,
            text=[[f'{v:.3f}' for v in values]],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate='Token: %{x}<br>Value: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            width=max(800, len(tokens) * 40),
            height=200,
            xaxis_title="Tokens",
            yaxis_title=""
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive confidence heatmap saved to {save_path}")

        return fig

    def plot_confidence_comparison(self, comparison_result: Dict[str, Any],
                                 title: str = "Model Confidence Comparison",
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize confidence comparison between models.

        Args:
            comparison_result: Result from ConfidenceAnalyzer.compare_model_confidence
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Plotly figure
        """
        comparisons = comparison_result['comparisons']

        # Create subplots for different metrics
        metrics = list(comparisons.keys())
        cols = min(3, len(metrics))
        rows = (len(metrics) + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=metrics,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        for i, metric in enumerate(metrics):
            row = i // cols + 1
            col = i % cols + 1

            metric_comparison = comparisons[metric]

            if metric == 'entropy':
                # Plot entropy differences
                if 'position_entropy_correlation' in metric_comparison:
                    correlation = metric_comparison['position_entropy_correlation']
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[correlation, correlation],
                            mode='lines+markers',
                            name=f'Correlation: {correlation:.3f}',
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                # Add mean difference as bar
                mean_diff = metric_comparison['mean_entropy_diff']
                fig.add_trace(
                    go.Bar(
                        x=['Mean Difference'],
                        y=[mean_diff],
                        name=f'Diff: {mean_diff:.3f}',
                        showlegend=False
                    ),
                    row=row, col=col
                )

            elif metric == 'top_k_probability':
                # Plot top-k confidence differences
                mean_diff = metric_comparison['mean_top_k_diff']
                correlation = metric_comparison.get('top_k_correlation', 0)

                fig.add_trace(
                    go.Bar(
                        x=['Mean Diff', 'Correlation'],
                        y=[mean_diff, correlation],
                        name=f'Top-K Analysis',
                        showlegend=False
                    ),
                    row=row, col=col
                )

            elif metric == 'perplexity':
                # Plot perplexity differences
                overall_diff = metric_comparison['overall_perplexity_diff']
                position_diff = metric_comparison['mean_position_perplexity_diff']

                fig.add_trace(
                    go.Bar(
                        x=['Overall', 'Position Mean'],
                        y=[overall_diff, position_diff],
                        name=f'Perplexity Diff',
                        showlegend=False
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title=title,
            height=300 * rows,
            width=400 * cols
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Confidence comparison plot saved to {save_path}")

        return fig

    def plot_multilingual_confidence(self, multilingual_result: Dict[str, Any],
                                    title: str = "Multilingual Confidence Analysis",
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        Visualize confidence patterns across multiple languages.

        Args:
            multilingual_result: Result from analyze_multilingual_confidence
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Plotly figure
        """
        language_analyses = multilingual_result['language_analyses']
        cross_lang_comparison = multilingual_result['cross_language_comparison']

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Entropy by Language', 'Perplexity by Language',
                          'Confidence Distribution', 'Language Ranking'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )

        # Plot 1: Entropy by language
        if 'entropy_by_language' in cross_lang_comparison:
            entropy_data = cross_lang_comparison['entropy_by_language']
            languages = list(entropy_data.keys())
            entropies = list(entropy_data.values())

            fig.add_trace(
                go.Bar(
                    x=languages,
                    y=entropies,
                    name='Entropy',
                    showlegend=False
                ),
                row=1, col=1
            )

        # Plot 2: Perplexity by language
        if 'perplexity_by_language' in cross_lang_comparison:
            perplexity_data = cross_lang_comparison['perplexity_by_language']
            languages = list(perplexity_data.keys())
            perplexities = list(perplexity_data.values())

            fig.add_trace(
                go.Bar(
                    x=languages,
                    y=perplexities,
                    name='Perplexity',
                    showlegend=False
                ),
                row=1, col=2
            )

        # Plot 3: Box plots of confidence distributions
        for lang, analysis in language_analyses.items():
            if 'entropy' in analysis['confidence_measures']:
                position_entropies = analysis['confidence_measures']['entropy']['position_entropies']
                fig.add_trace(
                    go.Box(
                        y=position_entropies,
                        name=lang,
                        showlegend=False
                    ),
                    row=2, col=1
                )

        # Plot 4: Language ranking
        if 'entropy_by_language' in cross_lang_comparison:
            entropy_data = cross_lang_comparison['entropy_by_language']
            # Sort languages by entropy (ascending = more confident)
            sorted_langs = sorted(entropy_data.items(), key=lambda x: x[1])
            langs, entropies = zip(*sorted_langs)

            fig.add_trace(
                go.Bar(
                    x=list(langs),
                    y=list(range(1, len(langs) + 1)),
                    name='Ranking',
                    text=[f'Rank {i+1}' for i in range(len(langs))],
                    showlegend=False,
                    orientation='v'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title=title,
            height=800,
            width=1000
        )

        # Update axis titles
        fig.update_xaxes(title_text="Language", row=1, col=1)
        fig.update_yaxes(title_text="Entropy", row=1, col=1)
        fig.update_xaxes(title_text="Language", row=1, col=2)
        fig.update_yaxes(title_text="Perplexity", row=1, col=2)
        fig.update_xaxes(title_text="Language", row=2, col=1)
        fig.update_yaxes(title_text="Entropy Distribution", row=2, col=1)
        fig.update_xaxes(title_text="Language", row=2, col=2)
        fig.update_yaxes(title_text="Confidence Rank", row=2, col=2)

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Multilingual confidence plot saved to {save_path}")

        return fig

    def plot_uncertainty_timeline(self, confidence_result: Dict[str, Any],
                                title: str = "Uncertainty Timeline",
                                save_path: Optional[str] = None,
                                interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot uncertainty as a timeline showing high/low confidence regions.

        Args:
            confidence_result: Result from confidence analysis
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive visualization

        Returns:
            Matplotlib or Plotly figure
        """
        if 'entropy' not in confidence_result['confidence_measures']:
            raise ValueError("Entropy analysis required for uncertainty timeline")

        entropy_data = confidence_result['confidence_measures']['entropy']
        tokens = confidence_result['tokens']
        position_entropies = entropy_data['position_entropies']

        high_uncertainty_positions = entropy_data['high_uncertainty_positions']
        low_uncertainty_positions = entropy_data['low_uncertainty_positions']

        if interactive:
            return self._plot_interactive_uncertainty_timeline(
                position_entropies, tokens, high_uncertainty_positions,
                low_uncertainty_positions, title, save_path
            )
        else:
            return self._plot_static_uncertainty_timeline(
                position_entropies, tokens, high_uncertainty_positions,
                low_uncertainty_positions, title, save_path
            )

    def _plot_static_uncertainty_timeline(self, position_entropies: List[float],
                                        tokens: List[str],
                                        high_uncertainty_positions: List[Dict],
                                        low_uncertainty_positions: List[Dict],
                                        title: str,
                                        save_path: Optional[str]) -> plt.Figure:
        """Create static uncertainty timeline."""
        fig, ax = plt.subplots(figsize=(16, 6))

        positions = range(len(position_entropies))

        # Plot entropy line
        ax.plot(positions, position_entropies, 'b-', linewidth=2, alpha=0.7)

        # Highlight high uncertainty regions
        for pos_info in high_uncertainty_positions:
            pos = pos_info['position']
            ax.axvspan(pos - 0.4, pos + 0.4, alpha=0.3, color='red', label='High Uncertainty')

        # Highlight low uncertainty regions
        for pos_info in low_uncertainty_positions:
            pos = pos_info['position']
            ax.axvspan(pos - 0.4, pos + 0.4, alpha=0.3, color='green', label='Low Uncertainty')

        # Set token labels
        if len(tokens) <= 25:
            ax.set_xticks(positions)
            ax.set_xticklabels(tokens, rotation=45, ha='right')
        else:
            ax.set_xlabel('Token Position')

        ax.set_ylabel('Entropy (Uncertainty)')
        ax.set_title(title)

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Uncertainty timeline saved to {save_path}")

        return fig

    def _plot_interactive_uncertainty_timeline(self, position_entropies: List[float],
                                             tokens: List[str],
                                             high_uncertainty_positions: List[Dict],
                                             low_uncertainty_positions: List[Dict],
                                             title: str,
                                             save_path: Optional[str]) -> go.Figure:
        """Create interactive uncertainty timeline."""
        fig = go.Figure()

        positions = list(range(len(position_entropies)))

        # Main entropy line
        fig.add_trace(go.Scatter(
            x=positions,
            y=position_entropies,
            mode='lines+markers',
            name='Entropy',
            text=tokens,
            hovertemplate='Position: %{x}<br>Token: %{text}<br>Entropy: %{y:.3f}<extra></extra>',
            line=dict(width=3)
        ))

        # Add uncertainty regions
        for pos_info in high_uncertainty_positions:
            pos = pos_info['position']
            entropy_val = pos_info['entropy']
            fig.add_shape(
                type="rect",
                x0=pos - 0.4, x1=pos + 0.4,
                y0=0, y1=max(position_entropies),
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0
            )

        for pos_info in low_uncertainty_positions:
            pos = pos_info['position']
            entropy_val = pos_info['entropy']
            fig.add_shape(
                type="rect",
                x0=pos - 0.4, x1=pos + 0.4,
                y0=0, y1=max(position_entropies),
                fillcolor="green",
                opacity=0.2,
                layer="below",
                line_width=0
            )

        fig.update_layout(
            title=title,
            xaxis_title="Token Position",
            yaxis_title="Entropy (Uncertainty)",
            width=1200,
            height=500,
            hovermode='x unified'
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Interactive uncertainty timeline saved to {save_path}")

        return fig