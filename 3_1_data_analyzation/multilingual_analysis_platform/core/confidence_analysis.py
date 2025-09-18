"""
Token Prediction Confidence Analysis Module

Provides functionality for analyzing model confidence in token predictions using
logits, entropy, variance, and other uncertainty measures.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
import logging
from pathlib import Path

from ..models.model_manager import get_model_manager
from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ConfidenceAnalyzer:
    """Analyzes prediction confidence and uncertainty in language models."""

    def __init__(self):
        """Initialize the confidence analyzer."""
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.confidence_cache = {}

    def analyze_prediction_confidence(self, model_name_or_path: str,
                                    text: str,
                                    model_type: str = 'base',
                                    temperature: float = 1.0,
                                    top_k: int = 50,
                                    analysis_methods: List[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction confidence for given text.

        Args:
            model_name_or_path: Model name or path
            text: Input text
            model_type: Type of model ('base' or 'trained')
            temperature: Temperature scaling for softmax
            top_k: Number of top predictions to analyze
            analysis_methods: Methods to use for confidence analysis

        Returns:
            Dictionary containing confidence analysis results
        """
        if analysis_methods is None:
            analysis_methods = ['entropy', 'variance', 'top_k_probability', 'perplexity']

        cache_key = f"{model_type}_{model_name_or_path}_{hash(text)}_{temperature}_{top_k}"

        if cache_key in self.confidence_cache:
            logger.info("Using cached confidence analysis")
            return self.confidence_cache[cache_key]

        # Get prediction probabilities from model manager
        prediction_data = self.model_manager.get_prediction_probabilities(
            model_name_or_path, text, model_type
        )

        if 'logits' not in prediction_data:
            raise ValueError("Model does not support probability extraction")

        logits = prediction_data['logits']
        tokens = prediction_data['tokens']

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Analyze confidence using various methods
        analysis_results = {
            'model_name': model_name_or_path,
            'model_type': model_type,
            'text': text,
            'tokens': tokens,
            'temperature': temperature,
            'sequence_length': len(tokens),
            'confidence_measures': {}
        }

        if 'entropy' in analysis_methods:
            analysis_results['confidence_measures']['entropy'] = self._calculate_entropy(
                probabilities, tokens
            )

        if 'variance' in analysis_methods:
            analysis_results['confidence_measures']['variance'] = self._calculate_variance(
                probabilities, tokens
            )

        if 'top_k_probability' in analysis_methods:
            analysis_results['confidence_measures']['top_k_probability'] = self._calculate_top_k_confidence(
                probabilities, tokens, top_k
            )

        if 'perplexity' in analysis_methods:
            analysis_results['confidence_measures']['perplexity'] = self._calculate_perplexity(
                probabilities, tokens
            )

        if 'mutual_information' in analysis_methods:
            analysis_results['confidence_measures']['mutual_information'] = self._calculate_mutual_information(
                probabilities, tokens
            )

        if 'confidence_intervals' in analysis_methods:
            analysis_results['confidence_measures']['confidence_intervals'] = self._calculate_confidence_intervals(
                probabilities, tokens
            )

        # Cache the result
        self.confidence_cache[cache_key] = analysis_results

        logger.info(f"Analyzed prediction confidence using {len(analysis_methods)} methods")
        return analysis_results

    def _calculate_entropy(self, probabilities: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        """Calculate entropy of prediction distributions."""
        # Convert to numpy for easier manipulation
        probs_np = probabilities.detach().cpu().numpy()

        # Calculate entropy for each position
        position_entropies = []
        for i in range(probs_np.shape[1]):  # sequence length
            pos_probs = probs_np[0, i]  # Remove batch dimension
            # Use scipy entropy (natural log)
            entropy_val = scipy_entropy(pos_probs)
            position_entropies.append(entropy_val)

        # Calculate statistics
        entropy_stats = {
            'position_entropies': position_entropies,
            'mean_entropy': np.mean(position_entropies),
            'std_entropy': np.std(position_entropies),
            'max_entropy': np.max(position_entropies),
            'min_entropy': np.min(position_entropies),
            'entropy_range': np.max(position_entropies) - np.min(position_entropies)
        }

        # Identify high/low confidence positions
        mean_entropy = entropy_stats['mean_entropy']
        std_entropy = entropy_stats['std_entropy']

        high_uncertainty_positions = []
        low_uncertainty_positions = []

        for i, (entropy_val, token) in enumerate(zip(position_entropies, tokens)):
            if entropy_val > mean_entropy + std_entropy:
                high_uncertainty_positions.append({
                    'position': i,
                    'token': token,
                    'entropy': entropy_val,
                    'uncertainty_level': 'high'
                })
            elif entropy_val < mean_entropy - std_entropy:
                low_uncertainty_positions.append({
                    'position': i,
                    'token': token,
                    'entropy': entropy_val,
                    'uncertainty_level': 'low'
                })

        entropy_stats.update({
            'high_uncertainty_positions': high_uncertainty_positions,
            'low_uncertainty_positions': low_uncertainty_positions,
            'uncertainty_classification': self._classify_uncertainty_level(mean_entropy)
        })

        return entropy_stats

    def _calculate_variance(self, probabilities: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        """Calculate variance of prediction distributions."""
        probs_np = probabilities.detach().cpu().numpy()

        position_variances = []
        for i in range(probs_np.shape[1]):
            pos_probs = probs_np[0, i]
            # Calculate variance
            variance = np.var(pos_probs)
            position_variances.append(variance)

        variance_stats = {
            'position_variances': position_variances,
            'mean_variance': np.mean(position_variances),
            'std_variance': np.std(position_variances),
            'max_variance': np.max(position_variances),
            'min_variance': np.min(position_variances)
        }

        return variance_stats

    def _calculate_top_k_confidence(self, probabilities: torch.Tensor,
                                  tokens: List[str], top_k: int) -> Dict[str, Any]:
        """Calculate confidence based on top-k predictions."""
        probs_np = probabilities.detach().cpu().numpy()

        position_top_k_confidence = []
        position_top_k_predictions = []

        for i in range(probs_np.shape[1]):
            pos_probs = probs_np[0, i]

            # Get top-k probabilities and indices
            top_k_indices = np.argsort(pos_probs)[-top_k:][::-1]
            top_k_probs = pos_probs[top_k_indices]

            # Calculate top-k confidence (sum of top-k probabilities)
            top_k_confidence = np.sum(top_k_probs)
            position_top_k_confidence.append(top_k_confidence)

            # Store top predictions for this position
            position_predictions = []
            for j, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                position_predictions.append({
                    'rank': j + 1,
                    'token_id': int(idx),
                    'probability': float(prob),
                    'confidence_contribution': float(prob / top_k_confidence)
                })

            position_top_k_predictions.append(position_predictions)

        top_k_stats = {
            'k': top_k,
            'position_top_k_confidence': position_top_k_confidence,
            'position_top_k_predictions': position_top_k_predictions,
            'mean_top_k_confidence': np.mean(position_top_k_confidence),
            'std_top_k_confidence': np.std(position_top_k_confidence),
            'min_top_k_confidence': np.min(position_top_k_confidence),
            'max_top_k_confidence': np.max(position_top_k_confidence)
        }

        return top_k_stats

    def _calculate_perplexity(self, probabilities: torch.Tensor, tokens: List[str]) -> Dict[str, Any]:
        """Calculate perplexity of the model predictions."""
        probs_np = probabilities.detach().cpu().numpy()

        # Calculate perplexity for each position
        position_perplexities = []
        for i in range(probs_np.shape[1]):
            pos_probs = probs_np[0, i]
            # Perplexity = 2^(-log2(probability)) = 2^entropy_base2
            entropy_base2 = scipy_entropy(pos_probs, base=2)
            perplexity = 2 ** entropy_base2
            position_perplexities.append(perplexity)

        # Calculate overall perplexity (geometric mean)
        log_probs = np.log(probs_np[0] + 1e-8)  # Add small epsilon to avoid log(0)
        cross_entropy = -np.mean(log_probs)
        overall_perplexity = np.exp(cross_entropy)

        perplexity_stats = {
            'position_perplexities': position_perplexities,
            'overall_perplexity': float(overall_perplexity),
            'mean_position_perplexity': np.mean(position_perplexities),
            'std_position_perplexity': np.std(position_perplexities),
            'max_perplexity': np.max(position_perplexities),
            'min_perplexity': np.min(position_perplexities)
        }

        return perplexity_stats

    def _calculate_mutual_information(self, probabilities: torch.Tensor,
                                    tokens: List[str]) -> Dict[str, Any]:
        """Calculate mutual information between positions."""
        probs_np = probabilities.detach().cpu().numpy()[0]  # Remove batch dimension

        # Calculate pairwise mutual information between positions
        seq_len = probs_np.shape[0]
        mutual_info_matrix = np.zeros((seq_len, seq_len))

        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    # Discretize probabilities for MI calculation
                    bins = 10
                    hist_i, _ = np.histogram(probs_np[i], bins=bins, density=True)
                    hist_j, _ = np.histogram(probs_np[j], bins=bins, density=True)

                    # Calculate mutual information
                    mi = mutual_info_score(hist_i, hist_j)
                    mutual_info_matrix[i, j] = mi

        mutual_info_stats = {
            'mutual_info_matrix': mutual_info_matrix,
            'mean_mutual_info': np.mean(mutual_info_matrix),
            'max_mutual_info': np.max(mutual_info_matrix),
            'min_mutual_info': np.min(mutual_info_matrix)
        }

        return mutual_info_stats

    def _calculate_confidence_intervals(self, probabilities: torch.Tensor,
                                      tokens: List[str],
                                      confidence_level: float = 0.95) -> Dict[str, Any]:
        """Calculate confidence intervals for predictions."""
        probs_np = probabilities.detach().cpu().numpy()

        # Calculate confidence intervals for each position
        position_confidence_intervals = []
        alpha = 1 - confidence_level

        for i in range(probs_np.shape[1]):
            pos_probs = probs_np[0, i]

            # Sort probabilities
            sorted_probs = np.sort(pos_probs)[::-1]
            cumsum_probs = np.cumsum(sorted_probs)

            # Find the number of tokens needed for confidence level
            num_tokens_for_confidence = np.where(cumsum_probs >= confidence_level)[0]
            if len(num_tokens_for_confidence) > 0:
                tokens_needed = num_tokens_for_confidence[0] + 1
                interval_probability = cumsum_probs[num_tokens_for_confidence[0]]
            else:
                tokens_needed = len(sorted_probs)
                interval_probability = 1.0

            position_confidence_intervals.append({
                'tokens_needed': int(tokens_needed),
                'interval_probability': float(interval_probability),
                'coverage_ratio': float(tokens_needed / len(sorted_probs))
            })

        confidence_interval_stats = {
            'confidence_level': confidence_level,
            'position_confidence_intervals': position_confidence_intervals,
            'mean_tokens_needed': np.mean([ci['tokens_needed'] for ci in position_confidence_intervals]),
            'mean_coverage_ratio': np.mean([ci['coverage_ratio'] for ci in position_confidence_intervals])
        }

        return confidence_interval_stats

    def _classify_uncertainty_level(self, entropy_value: float) -> str:
        """Classify uncertainty level based on entropy value."""
        # These thresholds can be adjusted based on empirical observations
        if entropy_value < 2.0:
            return 'low'
        elif entropy_value < 4.0:
            return 'medium'
        elif entropy_value < 6.0:
            return 'high'
        else:
            return 'very_high'

    def compare_model_confidence(self, model1_result: Dict[str, Any],
                               model2_result: Dict[str, Any],
                               comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare confidence between two models.

        Args:
            model1_result: Confidence analysis result from first model
            model2_result: Confidence analysis result from second model
            comparison_metrics: Metrics to compare

        Returns:
            Dictionary with comparison results
        """
        if comparison_metrics is None:
            comparison_metrics = ['entropy', 'variance', 'top_k_probability', 'perplexity']

        comparison_result = {
            'model1': model1_result['model_name'],
            'model2': model2_result['model_name'],
            'text1': model1_result['text'],
            'text2': model2_result['text'],
            'comparisons': {}
        }

        for metric in comparison_metrics:
            if (metric in model1_result['confidence_measures'] and
                metric in model2_result['confidence_measures']):

                metric_data1 = model1_result['confidence_measures'][metric]
                metric_data2 = model2_result['confidence_measures'][metric]

                comparison_result['comparisons'][metric] = self._compare_confidence_metric(
                    metric_data1, metric_data2, metric
                )

        return comparison_result

    def _compare_confidence_metric(self, metric_data1: Dict[str, Any],
                                 metric_data2: Dict[str, Any],
                                 metric_name: str) -> Dict[str, Any]:
        """Compare a specific confidence metric between two models."""
        comparison = {
            'metric': metric_name
        }

        if metric_name == 'entropy':
            comparison.update({
                'mean_entropy_diff': metric_data1['mean_entropy'] - metric_data2['mean_entropy'],
                'std_entropy_diff': metric_data1['std_entropy'] - metric_data2['std_entropy'],
                'model1_uncertainty': metric_data1['uncertainty_classification'],
                'model2_uncertainty': metric_data2['uncertainty_classification'],
                'position_entropy_correlation': np.corrcoef(
                    metric_data1['position_entropies'],
                    metric_data2['position_entropies']
                )[0, 1] if len(metric_data1['position_entropies']) == len(metric_data2['position_entropies']) else None
            })

        elif metric_name == 'variance':
            comparison.update({
                'mean_variance_diff': metric_data1['mean_variance'] - metric_data2['mean_variance'],
                'variance_correlation': np.corrcoef(
                    metric_data1['position_variances'],
                    metric_data2['position_variances']
                )[0, 1] if len(metric_data1['position_variances']) == len(metric_data2['position_variances']) else None
            })

        elif metric_name == 'top_k_probability':
            comparison.update({
                'mean_top_k_diff': metric_data1['mean_top_k_confidence'] - metric_data2['mean_top_k_confidence'],
                'top_k_correlation': np.corrcoef(
                    metric_data1['position_top_k_confidence'],
                    metric_data2['position_top_k_confidence']
                )[0, 1] if len(metric_data1['position_top_k_confidence']) == len(metric_data2['position_top_k_confidence']) else None
            })

        elif metric_name == 'perplexity':
            comparison.update({
                'overall_perplexity_diff': metric_data1['overall_perplexity'] - metric_data2['overall_perplexity'],
                'mean_position_perplexity_diff': metric_data1['mean_position_perplexity'] - metric_data2['mean_position_perplexity'],
                'perplexity_correlation': np.corrcoef(
                    metric_data1['position_perplexities'],
                    metric_data2['position_perplexities']
                )[0, 1] if len(metric_data1['position_perplexities']) == len(metric_data2['position_perplexities']) else None
            })

        return comparison

    def analyze_multilingual_confidence(self, model_name_or_path: str,
                                      text_pairs: List[Tuple[str, str]],
                                      languages: List[str],
                                      model_type: str = 'base') -> Dict[str, Any]:
        """
        Analyze confidence patterns across multiple languages.

        Args:
            model_name_or_path: Model name or path
            text_pairs: List of (text, language_code) pairs
            languages: List of language codes
            model_type: Type of model

        Returns:
            Dictionary with multilingual confidence analysis
        """
        language_analyses = {}

        for (text, lang) in text_pairs:
            if lang in languages:
                analysis = self.analyze_prediction_confidence(
                    model_name_or_path, text, model_type
                )
                analysis['language'] = lang
                language_analyses[lang] = analysis

        # Compare confidence across languages
        multilingual_comparison = {
            'model_name': model_name_or_path,
            'model_type': model_type,
            'languages': languages,
            'language_analyses': language_analyses,
            'cross_language_comparison': {}
        }

        # Calculate cross-language statistics
        if len(language_analyses) > 1:
            entropy_by_lang = {}
            perplexity_by_lang = {}

            for lang, analysis in language_analyses.items():
                if 'entropy' in analysis['confidence_measures']:
                    entropy_by_lang[lang] = analysis['confidence_measures']['entropy']['mean_entropy']
                if 'perplexity' in analysis['confidence_measures']:
                    perplexity_by_lang[lang] = analysis['confidence_measures']['perplexity']['overall_perplexity']

            multilingual_comparison['cross_language_comparison'] = {
                'entropy_by_language': entropy_by_lang,
                'perplexity_by_language': perplexity_by_lang,
                'most_confident_language': min(entropy_by_lang, key=entropy_by_lang.get) if entropy_by_lang else None,
                'least_confident_language': max(entropy_by_lang, key=entropy_by_lang.get) if entropy_by_lang else None,
                'entropy_variance_across_languages': np.var(list(entropy_by_lang.values())) if entropy_by_lang else None
            }

        return multilingual_comparison

    def export_confidence_analysis(self, confidence_result: Dict[str, Any],
                                 output_path: str,
                                 format: str = 'json') -> str:
        """
        Export confidence analysis results to file.

        Args:
            confidence_result: Result from analyze_prediction_confidence
            output_path: Output file path
            format: Export format ('json', 'csv')

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            import json

            # Convert numpy arrays to lists for JSON serialization
            export_data = {}
            for key, value in confidence_result.items():
                if isinstance(value, dict):
                    export_data[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            export_data[key][subkey] = subvalue.tolist()
                        elif isinstance(subvalue, dict):
                            export_data[key][subkey] = {}
                            for subsubkey, subsubvalue in subvalue.items():
                                if isinstance(subsubvalue, np.ndarray):
                                    export_data[key][subkey][subsubkey] = subsubvalue.tolist()
                                else:
                                    export_data[key][subkey][subsubkey] = subsubvalue
                        else:
                            export_data[key][subkey] = subvalue
                else:
                    export_data[key] = value

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

        elif format == 'csv':
            # Export summary statistics to CSV
            summary_data = []

            for measure_name, measure_data in confidence_result['confidence_measures'].items():
                if isinstance(measure_data, dict):
                    for stat_name, stat_value in measure_data.items():
                        if isinstance(stat_value, (int, float)):
                            summary_data.append({
                                'measure': measure_name,
                                'statistic': stat_name,
                                'value': stat_value
                            })

            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Confidence analysis exported to {output_path}")
        return str(output_path)

    def clear_cache(self):
        """Clear confidence analysis cache."""
        self.confidence_cache.clear()
        logger.info("Confidence analysis cache cleared")