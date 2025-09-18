"""
Attention Analysis Module

Provides functionality for extracting and analyzing attention patterns from transformer models.
Supports attention visualization, pattern analysis, and cross-language comparison.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings

from models.model_manager import get_model_manager
from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """Analyzes attention patterns in transformer models."""

    def __init__(self):
        """Initialize the attention analyzer."""
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.attention_cache = {}

    def extract_attention_weights(self, model_name_or_path: str,
                                text: str,
                                model_type: str = 'base',
                                layers: Optional[Union[str, List[int]]] = None,
                                heads: Optional[Union[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Extract attention weights from a model for given text.

        Args:
            model_name_or_path: Model name or path
            text: Input text
            model_type: Type of model ('base' or 'trained')
            layers: Specific layers to analyze ('all' or list of layer indices)
            heads: Specific heads to analyze ('all' or list of head indices)

        Returns:
            Dictionary containing attention weights and metadata
        """
        cache_key = f"{model_type}_{model_name_or_path}_{hash(text)}_{str(layers)}_{str(heads)}"

        if cache_key in self.attention_cache:
            logger.info("Using cached attention weights")
            return self.attention_cache[cache_key]

        # Get attention weights from model manager
        attention_data = self.model_manager.get_attention_weights(
            model_name_or_path, text, model_type
        )

        attention_weights = attention_data['attention_weights']
        tokens = attention_data['tokens']
        input_ids = attention_data['input_ids']
        attention_mask = attention_data['attention_mask']

        # Process layer selection
        num_layers = len(attention_weights)
        if layers is None or layers == 'all':
            layer_indices = list(range(num_layers))
        elif isinstance(layers, list):
            layer_indices = [i for i in layers if 0 <= i < num_layers]
        else:
            raise ValueError("layers must be 'all' or a list of layer indices")

        # Process head selection
        num_heads = attention_weights[0].size(1)  # assuming all layers have same number of heads
        if heads is None or heads == 'all':
            head_indices = list(range(num_heads))
        elif isinstance(heads, list):
            head_indices = [i for i in heads if 0 <= i < num_heads]
        else:
            raise ValueError("heads must be 'all' or a list of head indices")

        # Extract and process attention weights
        processed_attention = {}
        for layer_idx in layer_indices:
            layer_attention = attention_weights[layer_idx].squeeze(0)  # Remove batch dimension

            # Filter heads if specified
            if heads != 'all':
                layer_attention = layer_attention[head_indices]

            # Convert to numpy for easier manipulation
            layer_attention_np = layer_attention.detach().cpu().numpy()

            processed_attention[f'layer_{layer_idx}'] = {
                'attention_weights': layer_attention_np,
                'shape': layer_attention_np.shape,
                'num_heads': layer_attention_np.shape[0] if len(layer_attention_np.shape) > 2 else 1
            }

        result = {
            'attention_weights': processed_attention,
            'tokens': tokens,
            'input_ids': input_ids.cpu().numpy(),
            'attention_mask': attention_mask.cpu().numpy(),
            'model_name': model_name_or_path,
            'model_type': model_type,
            'text': text,
            'num_layers': len(layer_indices),
            'num_heads': len(head_indices),
            'sequence_length': len(tokens)
        }

        # Cache the result
        self.attention_cache[cache_key] = result

        logger.info(f"Extracted attention weights for {len(layer_indices)} layers and {len(head_indices)} heads")
        return result

    def analyze_attention_patterns(self, attention_result: Dict[str, Any],
                                 pattern_types: List[str] = None) -> Dict[str, Any]:
        """
        Analyze various attention patterns in the extracted attention weights.

        Args:
            attention_result: Result from extract_attention_weights
            pattern_types: Types of patterns to analyze

        Returns:
            Dictionary with pattern analysis results
        """
        if pattern_types is None:
            pattern_types = ['head_entropy', 'layer_entropy', 'attention_distance',
                           'self_attention', 'token_importance']

        tokens = attention_result['tokens']
        attention_weights = attention_result['attention_weights']
        sequence_length = attention_result['sequence_length']

        analysis_results = {}

        # Analyze each pattern type
        if 'head_entropy' in pattern_types:
            analysis_results['head_entropy'] = self._analyze_head_entropy(attention_weights)

        if 'layer_entropy' in pattern_types:
            analysis_results['layer_entropy'] = self._analyze_layer_entropy(attention_weights)

        if 'attention_distance' in pattern_types:
            analysis_results['attention_distance'] = self._analyze_attention_distance(attention_weights)

        if 'self_attention' in pattern_types:
            analysis_results['self_attention'] = self._analyze_self_attention(attention_weights)

        if 'token_importance' in pattern_types:
            analysis_results['token_importance'] = self._analyze_token_importance(
                attention_weights, tokens
            )

        if 'attention_flow' in pattern_types:
            analysis_results['attention_flow'] = self._analyze_attention_flow(attention_weights)

        return analysis_results

    def _analyze_head_entropy(self, attention_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entropy of attention heads."""
        head_entropies = {}

        for layer_name, layer_data in attention_weights.items():
            layer_attention = layer_data['attention_weights']

            if len(layer_attention.shape) == 3:  # [heads, seq_len, seq_len]
                head_entropies[layer_name] = []

                for head_idx in range(layer_attention.shape[0]):
                    head_attention = layer_attention[head_idx]
                    # Calculate entropy for each position
                    entropies = []
                    for i in range(head_attention.shape[0]):
                        attention_dist = head_attention[i]
                        # Avoid log(0) by adding small epsilon
                        entropy = -np.sum(attention_dist * np.log(attention_dist + 1e-8))
                        entropies.append(entropy)

                    head_entropies[layer_name].append({
                        'head_idx': head_idx,
                        'entropies': entropies,
                        'mean_entropy': np.mean(entropies),
                        'std_entropy': np.std(entropies)
                    })

        return head_entropies

    def _analyze_layer_entropy(self, attention_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entropy across layers."""
        layer_entropies = {}

        for layer_name, layer_data in attention_weights.items():
            layer_attention = layer_data['attention_weights']

            # Average across heads if multiple heads
            if len(layer_attention.shape) == 3:
                avg_attention = np.mean(layer_attention, axis=0)
            else:
                avg_attention = layer_attention

            # Calculate entropy for each position
            entropies = []
            for i in range(avg_attention.shape[0]):
                attention_dist = avg_attention[i]
                entropy = -np.sum(attention_dist * np.log(attention_dist + 1e-8))
                entropies.append(entropy)

            layer_entropies[layer_name] = {
                'entropies': entropies,
                'mean_entropy': np.mean(entropies),
                'std_entropy': np.std(entropies),
                'total_entropy': np.sum(entropies)
            }

        return layer_entropies

    def _analyze_attention_distance(self, attention_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention distance patterns."""
        distance_analysis = {}

        for layer_name, layer_data in attention_weights.items():
            layer_attention = layer_data['attention_weights']

            # Average across heads if multiple heads
            if len(layer_attention.shape) == 3:
                avg_attention = np.mean(layer_attention, axis=0)
            else:
                avg_attention = layer_attention

            # Calculate average attention distance for each position
            distances = []
            for i in range(avg_attention.shape[0]):
                attention_dist = avg_attention[i]
                # Weighted average of distances
                positions = np.arange(len(attention_dist))
                avg_distance = np.sum(attention_dist * np.abs(positions - i))
                distances.append(avg_distance)

            distance_analysis[layer_name] = {
                'distances': distances,
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'max_distance': np.max(distances),
                'min_distance': np.min(distances)
            }

        return distance_analysis

    def _analyze_self_attention(self, attention_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze self-attention patterns."""
        self_attention_analysis = {}

        for layer_name, layer_data in attention_weights.items():
            layer_attention = layer_data['attention_weights']

            # Average across heads if multiple heads
            if len(layer_attention.shape) == 3:
                avg_attention = np.mean(layer_attention, axis=0)
            else:
                avg_attention = layer_attention

            # Extract diagonal (self-attention values)
            self_attention_values = np.diag(avg_attention)

            self_attention_analysis[layer_name] = {
                'self_attention_values': self_attention_values,
                'mean_self_attention': np.mean(self_attention_values),
                'std_self_attention': np.std(self_attention_values),
                'max_self_attention': np.max(self_attention_values),
                'min_self_attention': np.min(self_attention_values)
            }

        return self_attention_analysis

    def _analyze_token_importance(self, attention_weights: Dict[str, Any],
                                tokens: List[str]) -> Dict[str, Any]:
        """Analyze importance of each token based on attention received."""
        token_importance = {}

        for layer_name, layer_data in attention_weights.items():
            layer_attention = layer_data['attention_weights']

            # Average across heads if multiple heads
            if len(layer_attention.shape) == 3:
                avg_attention = np.mean(layer_attention, axis=0)
            else:
                avg_attention = layer_attention

            # Sum attention received by each token (column sums)
            importance_scores = np.sum(avg_attention, axis=0)

            # Create token importance mapping
            token_scores = []
            for i, token in enumerate(tokens):
                if i < len(importance_scores):
                    token_scores.append({
                        'token': token,
                        'position': i,
                        'importance_score': importance_scores[i],
                        'normalized_score': importance_scores[i] / np.sum(importance_scores)
                    })

            # Sort by importance
            token_scores.sort(key=lambda x: x['importance_score'], reverse=True)

            token_importance[layer_name] = {
                'token_scores': token_scores,
                'importance_distribution': importance_scores,
                'most_important_token': token_scores[0] if token_scores else None,
                'least_important_token': token_scores[-1] if token_scores else None
            }

        return token_importance

    def _analyze_attention_flow(self, attention_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention flow patterns across the sequence."""
        flow_analysis = {}

        for layer_name, layer_data in attention_weights.items():
            layer_attention = layer_data['attention_weights']

            # Average across heads if multiple heads
            if len(layer_attention.shape) == 3:
                avg_attention = np.mean(layer_attention, axis=0)
            else:
                avg_attention = layer_attention

            # Analyze flow patterns
            forward_flow = 0  # Attention to future tokens
            backward_flow = 0  # Attention to past tokens
            local_flow = 0  # Attention to nearby tokens (±2)

            seq_len = avg_attention.shape[0]
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        attention_value = avg_attention[i, j]
                        if j > i:  # Future token
                            forward_flow += attention_value
                        else:  # Past token
                            backward_flow += attention_value

                        # Local attention (within ±2 positions)
                        if abs(i - j) <= 2:
                            local_flow += attention_value

            total_flow = forward_flow + backward_flow

            flow_analysis[layer_name] = {
                'forward_flow': forward_flow,
                'backward_flow': backward_flow,
                'local_flow': local_flow,
                'total_flow': total_flow,
                'forward_ratio': forward_flow / total_flow if total_flow > 0 else 0,
                'backward_ratio': backward_flow / total_flow if total_flow > 0 else 0,
                'local_ratio': local_flow / np.sum(avg_attention) if np.sum(avg_attention) > 0 else 0
            }

        return flow_analysis

    def compare_attention_patterns(self, attention_result1: Dict[str, Any],
                                 attention_result2: Dict[str, Any],
                                 comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare attention patterns between two models or texts.

        Args:
            attention_result1: First attention result
            attention_result2: Second attention result
            comparison_metrics: Metrics to use for comparison

        Returns:
            Dictionary with comparison results
        """
        if comparison_metrics is None:
            comparison_metrics = ['cosine_similarity', 'mse', 'correlation']

        # Analyze patterns for both results
        patterns1 = self.analyze_attention_patterns(attention_result1)
        patterns2 = self.analyze_attention_patterns(attention_result2)

        comparison_results = {
            'model1': attention_result1['model_name'],
            'model2': attention_result2['model_name'],
            'text1': attention_result1['text'],
            'text2': attention_result2['text'],
            'comparison_metrics': {}
        }

        # Compare attention weights layer by layer
        attention1 = attention_result1['attention_weights']
        attention2 = attention_result2['attention_weights']

        common_layers = set(attention1.keys()).intersection(set(attention2.keys()))

        for layer in common_layers:
            layer_data1 = attention1[layer]['attention_weights']
            layer_data2 = attention2[layer]['attention_weights']

            # Ensure same shape for comparison
            if layer_data1.shape != layer_data2.shape:
                logger.warning(f"Layer {layer} has different shapes: {layer_data1.shape} vs {layer_data2.shape}")
                continue

            # Average across heads if multiple heads
            if len(layer_data1.shape) == 3:
                avg_attention1 = np.mean(layer_data1, axis=0)
                avg_attention2 = np.mean(layer_data2, axis=0)
            else:
                avg_attention1 = layer_data1
                avg_attention2 = layer_data2

            layer_comparison = {}

            if 'cosine_similarity' in comparison_metrics:
                # Flatten arrays for cosine similarity
                flat1 = avg_attention1.flatten()
                flat2 = avg_attention2.flatten()
                cosine_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
                layer_comparison['cosine_similarity'] = cosine_sim

            if 'mse' in comparison_metrics:
                mse = np.mean((avg_attention1 - avg_attention2) ** 2)
                layer_comparison['mse'] = mse

            if 'correlation' in comparison_metrics:
                correlation = np.corrcoef(avg_attention1.flatten(), avg_attention2.flatten())[0, 1]
                layer_comparison['correlation'] = correlation

            comparison_results['comparison_metrics'][layer] = layer_comparison

        # Compare pattern analysis results
        if 'head_entropy' in patterns1 and 'head_entropy' in patterns2:
            comparison_results['entropy_comparison'] = self._compare_entropies(
                patterns1['head_entropy'], patterns2['head_entropy']
            )

        # Calculate overall similarity score
        if comparison_results['comparison_metrics']:
            similarities = []
            for layer_data in comparison_results['comparison_metrics'].values():
                if 'cosine_similarity' in layer_data:
                    similarities.append(layer_data['cosine_similarity'])
                elif 'correlation' in layer_data:
                    similarities.append(abs(layer_data['correlation']))  # Use absolute correlation

            if similarities:
                comparison_results['overall_similarity'] = np.mean(similarities)
            else:
                comparison_results['overall_similarity'] = 0.0
        else:
            comparison_results['overall_similarity'] = 0.0

        return comparison_results

    def _compare_entropies(self, entropy1: Dict[str, Any], entropy2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare entropy patterns between two models."""
        entropy_comparison = {}

        common_layers = set(entropy1.keys()).intersection(set(entropy2.keys()))

        for layer in common_layers:
            layer_entropies1 = [head['mean_entropy'] for head in entropy1[layer]]
            layer_entropies2 = [head['mean_entropy'] for head in entropy2[layer]]

            # Ensure same number of heads
            min_heads = min(len(layer_entropies1), len(layer_entropies2))
            layer_entropies1 = layer_entropies1[:min_heads]
            layer_entropies2 = layer_entropies2[:min_heads]

            if min_heads > 0:
                correlation = np.corrcoef(layer_entropies1, layer_entropies2)[0, 1]
                mse = np.mean((np.array(layer_entropies1) - np.array(layer_entropies2)) ** 2)

                entropy_comparison[layer] = {
                    'correlation': correlation,
                    'mse': mse,
                    'mean_entropy1': np.mean(layer_entropies1),
                    'mean_entropy2': np.mean(layer_entropies2),
                    'entropy_difference': np.mean(layer_entropies1) - np.mean(layer_entropies2)
                }

        return entropy_comparison

    def export_attention_data(self, attention_result: Dict[str, Any],
                            output_path: str,
                            format: str = 'npz',
                            include_analysis: bool = True) -> str:
        """
        Export attention data to file.

        Args:
            attention_result: Result from extract_attention_weights
            output_path: Output file path
            format: Export format ('npz', 'json')
            include_analysis: Whether to include pattern analysis

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = attention_result.copy()

        if include_analysis:
            analysis_results = self.analyze_attention_patterns(attention_result)
            export_data['pattern_analysis'] = analysis_results

        if format == 'npz':
            # Convert nested dictionaries to flat structure for npz
            flat_data = {}
            for layer_name, layer_data in attention_result['attention_weights'].items():
                flat_data[f'{layer_name}_attention'] = layer_data['attention_weights']

            flat_data.update({
                'tokens': np.array(attention_result['tokens'], dtype=object),
                'input_ids': attention_result['input_ids'],
                'attention_mask': attention_result['attention_mask'],
                'text': attention_result['text'],
                'model_name': attention_result['model_name']
            })

            np.savez_compressed(output_path, **flat_data)

        elif format == 'json':
            import json

            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in export_data.items():
                if key == 'attention_weights':
                    json_data[key] = {}
                    for layer_name, layer_data in value.items():
                        json_data[key][layer_name] = {
                            'attention_weights': layer_data['attention_weights'].tolist(),
                            'shape': layer_data['shape'],
                            'num_heads': layer_data['num_heads']
                        }
                elif isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Attention data exported to {output_path}")
        return str(output_path)

    def clear_cache(self):
        """Clear attention cache to free memory."""
        self.attention_cache.clear()
        logger.info("Attention cache cleared")