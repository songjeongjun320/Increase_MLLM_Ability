"""
Comparison Utilities

Specialized utilities for comparing models and languages, particularly for
English-Korean comparisons and base vs training model analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

from .config_loader import get_config
from ..core.sentence_embedding import SentenceEmbeddingAnalyzer
from ..core.attention_analysis import AttentionAnalyzer
from ..core.confidence_analysis import ConfidenceAnalyzer

logger = logging.getLogger(__name__)


class ModelComparisonSuite:
    """Comprehensive model comparison utilities."""

    def __init__(self):
        """Initialize the comparison suite."""
        self.config = get_config()
        self.embedding_analyzer = SentenceEmbeddingAnalyzer()
        self.attention_analyzer = AttentionAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()

    def compare_base_vs_training_models(self, base_model_path: str,
                                      training_model_path: str,
                                      test_sentences: List[Tuple[str, str]],
                                      languages: List[str] = ['en', 'ko']) -> Dict[str, Any]:
        """
        Comprehensive comparison between base and training models.

        Args:
            base_model_path: Path to base model
            training_model_path: Path to training model
            test_sentences: List of (english_sentence, korean_sentence) pairs
            languages: Language codes to analyze

        Returns:
            Dictionary with comprehensive comparison results
        """
        logger.info(f"Comparing base model ({base_model_path}) vs training model ({training_model_path})")

        comparison_results = {
            'base_model': base_model_path,
            'training_model': training_model_path,
            'test_sentences': test_sentences,
            'languages': languages,
            'analysis_results': {}
        }

        # Extract English and Korean sentences
        english_sentences = [pair[0] for pair in test_sentences]
        korean_sentences = [pair[1] for pair in test_sentences]
        all_sentences = english_sentences + korean_sentences
        all_languages = ['en'] * len(english_sentences) + ['ko'] * len(korean_sentences)

        try:
            # 1. Embedding Analysis Comparison
            embedding_comparison = self._compare_embeddings(
                base_model_path, training_model_path, all_sentences, all_languages
            )
            comparison_results['analysis_results']['embeddings'] = embedding_comparison

            # 2. Attention Analysis Comparison
            attention_comparison = self._compare_attention_patterns(
                base_model_path, training_model_path, test_sentences
            )
            comparison_results['analysis_results']['attention'] = attention_comparison

            # 3. Confidence Analysis Comparison
            confidence_comparison = self._compare_confidence_patterns(
                base_model_path, training_model_path, test_sentences
            )
            comparison_results['analysis_results']['confidence'] = confidence_comparison

            # 4. Cross-Language Performance
            cross_lang_comparison = self._compare_cross_language_performance(
                base_model_path, training_model_path, test_sentences
            )
            comparison_results['analysis_results']['cross_language'] = cross_lang_comparison

            # 5. Summary Statistics
            summary = self._generate_comparison_summary(comparison_results)
            comparison_results['summary'] = summary

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            comparison_results['error'] = str(e)

        return comparison_results

    def _compare_embeddings(self, base_model: str, training_model: str,
                          sentences: List[str], languages: List[str]) -> Dict[str, Any]:
        """Compare embedding spaces between models."""
        logger.info("Comparing embedding spaces")

        # Generate embeddings for both models (using sentence transformers)
        base_embeddings = self.embedding_analyzer.generate_embeddings(
            texts=sentences,
            languages=languages,
            model_name=None  # Use default sentence transformer
        )

        # For training model, we might need to use a different approach
        # This is a simplified version - in practice, you'd need to load
        # and use the actual training model for embedding generation
        training_embeddings = self.embedding_analyzer.generate_embeddings(
            texts=sentences,
            languages=languages,
            model_name=None  # This would be replaced with training model logic
        )

        # Compare embedding spaces
        comparison = {
            'base_embeddings': base_embeddings,
            'training_embeddings': training_embeddings,
            'similarity_comparison': self._compare_similarity_matrices(
                base_embeddings['similarity_matrix'],
                training_embeddings['similarity_matrix']
            ),
            'language_clustering_comparison': self._compare_language_clustering(
                base_embeddings, training_embeddings, languages
            )
        }

        return comparison

    def _compare_attention_patterns(self, base_model: str, training_model: str,
                                  test_sentences: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Compare attention patterns between models."""
        logger.info("Comparing attention patterns")

        attention_comparisons = []

        for i, (en_sentence, ko_sentence) in enumerate(test_sentences[:5]):  # Limit to first 5 for performance
            # Analyze English sentence
            en_comparison = self._compare_single_attention(
                base_model, training_model, en_sentence, 'en'
            )

            # Analyze Korean sentence
            ko_comparison = self._compare_single_attention(
                base_model, training_model, ko_sentence, 'ko'
            )

            attention_comparisons.append({
                'sentence_pair_idx': i,
                'english_analysis': en_comparison,
                'korean_analysis': ko_comparison,
                'cross_language_similarity': self._calculate_cross_language_attention_similarity(
                    en_comparison, ko_comparison
                )
            })

        return {
            'sentence_comparisons': attention_comparisons,
            'aggregated_statistics': self._aggregate_attention_statistics(attention_comparisons)
        }

    def _compare_single_attention(self, base_model: str, training_model: str,
                                sentence: str, language: str) -> Dict[str, Any]:
        """Compare attention for a single sentence between models."""
        try:
            # Base model attention
            base_attention = self.attention_analyzer.extract_attention_weights(
                model_name_or_path=base_model,
                text=sentence,
                model_type='base'
            )

            # Training model attention
            training_attention = self.attention_analyzer.extract_attention_weights(
                model_name_or_path=training_model,
                text=sentence,
                model_type='trained'
            )

            # Compare patterns
            comparison = self.attention_analyzer.compare_attention_patterns(
                base_attention, training_attention
            )

            return {
                'base_attention': base_attention,
                'training_attention': training_attention,
                'comparison': comparison,
                'language': language,
                'sentence': sentence
            }

        except Exception as e:
            logger.error(f"Single attention comparison failed for {language}: {e}")
            return {'error': str(e), 'language': language, 'sentence': sentence}

    def _compare_confidence_patterns(self, base_model: str, training_model: str,
                                   test_sentences: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Compare confidence patterns between models."""
        logger.info("Comparing confidence patterns")

        confidence_comparisons = []

        for i, (en_sentence, ko_sentence) in enumerate(test_sentences[:5]):  # Limit for performance
            # English sentence confidence
            en_comparison = self._compare_single_confidence(
                base_model, training_model, en_sentence, 'en'
            )

            # Korean sentence confidence
            ko_comparison = self._compare_single_confidence(
                base_model, training_model, ko_sentence, 'ko'
            )

            confidence_comparisons.append({
                'sentence_pair_idx': i,
                'english_analysis': en_comparison,
                'korean_analysis': ko_comparison
            })

        return {
            'sentence_comparisons': confidence_comparisons,
            'aggregated_statistics': self._aggregate_confidence_statistics(confidence_comparisons)
        }

    def _compare_single_confidence(self, base_model: str, training_model: str,
                                 sentence: str, language: str) -> Dict[str, Any]:
        """Compare confidence for a single sentence between models."""
        try:
            # Base model confidence
            base_confidence = self.confidence_analyzer.analyze_prediction_confidence(
                model_name_or_path=base_model,
                text=sentence,
                model_type='base'
            )

            # Training model confidence
            training_confidence = self.confidence_analyzer.analyze_prediction_confidence(
                model_name_or_path=training_model,
                text=sentence,
                model_type='trained'
            )

            # Compare confidence
            comparison = self.confidence_analyzer.compare_model_confidence(
                base_confidence, training_confidence
            )

            return {
                'base_confidence': base_confidence,
                'training_confidence': training_confidence,
                'comparison': comparison,
                'language': language,
                'sentence': sentence
            }

        except Exception as e:
            logger.error(f"Single confidence comparison failed for {language}: {e}")
            return {'error': str(e), 'language': language, 'sentence': sentence}

    def _compare_cross_language_performance(self, base_model: str, training_model: str,
                                          test_sentences: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Compare cross-language performance between models."""
        logger.info("Comparing cross-language performance")

        english_sentences = [pair[0] for pair in test_sentences]
        korean_sentences = [pair[1] for pair in test_sentences]

        # Base model cross-language analysis
        base_cross_lang = self.embedding_analyzer.compare_sentence_pairs(
            sentences_en=english_sentences,
            sentences_target=korean_sentences,
            target_language='ko',
            model_name=None  # Would use base model embedding approach
        )

        # Training model cross-language analysis
        training_cross_lang = self.embedding_analyzer.compare_sentence_pairs(
            sentences_en=english_sentences,
            sentences_target=korean_sentences,
            target_language='ko',
            model_name=None  # Would use training model embedding approach
        )

        # Compare the results
        cross_lang_comparison = {
            'base_model_results': base_cross_lang,
            'training_model_results': training_cross_lang,
            'accuracy_improvement': training_cross_lang['accuracy'] - base_cross_lang['accuracy'],
            'similarity_improvement': (
                training_cross_lang['mean_pair_similarity'] - base_cross_lang['mean_pair_similarity']
            ),
            'performance_summary': {
                'base_accuracy': base_cross_lang['accuracy'],
                'training_accuracy': training_cross_lang['accuracy'],
                'improvement_ratio': (
                    training_cross_lang['accuracy'] / base_cross_lang['accuracy']
                    if base_cross_lang['accuracy'] > 0 else float('inf')
                )
            }
        }

        return cross_lang_comparison

    def _compare_similarity_matrices(self, matrix1: np.ndarray, matrix2: np.ndarray) -> Dict[str, Any]:
        """Compare two similarity matrices."""
        if matrix1.shape != matrix2.shape:
            return {'error': 'Matrix shapes do not match'}

        # Calculate various comparison metrics
        correlation = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
        mse = np.mean((matrix1 - matrix2) ** 2)
        mae = np.mean(np.abs(matrix1 - matrix2))

        # Element-wise differences
        diff_matrix = matrix1 - matrix2

        return {
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'max_difference': np.max(np.abs(diff_matrix)),
            'mean_difference': np.mean(diff_matrix),
            'std_difference': np.std(diff_matrix),
            'difference_matrix': diff_matrix
        }

    def _compare_language_clustering(self, embeddings1: Dict[str, Any],
                                   embeddings2: Dict[str, Any],
                                   languages: List[str]) -> Dict[str, Any]:
        """Compare language clustering between two embedding sets."""
        # Analyze clustering for both embedding sets
        clustering1 = self.embedding_analyzer.analyze_language_clusters(
            embeddings1['embeddings'], languages
        )

        clustering2 = self.embedding_analyzer.analyze_language_clusters(
            embeddings2['embeddings'], languages
        )

        # Compare clustering quality
        comparison = {
            'clustering1': clustering1,
            'clustering2': clustering2,
            'silhouette_improvement': (
                clustering2.get('silhouette_score', 0) - clustering1.get('silhouette_score', 0)
            ),
            'purity_comparison': {}
        }

        # Compare language purity
        for lang in set(languages):
            if lang in clustering1['language_purity'] and lang in clustering2['language_purity']:
                comparison['purity_comparison'][lang] = {
                    'base_purity': clustering1['language_purity'][lang],
                    'training_purity': clustering2['language_purity'][lang],
                    'improvement': clustering2['language_purity'][lang] - clustering1['language_purity'][lang]
                }

        return comparison

    def _calculate_cross_language_attention_similarity(self, en_analysis: Dict[str, Any],
                                                     ko_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate similarity between English and Korean attention patterns."""
        if 'error' in en_analysis or 'error' in ko_analysis:
            return {'error': 'Cannot compare due to analysis errors'}

        try:
            en_comparison = en_analysis['comparison']
            ko_comparison = ko_analysis['comparison']

            # Compare attention similarities between languages
            similarity_metrics = {}

            if 'comparison_metrics' in en_comparison and 'comparison_metrics' in ko_comparison:
                # Calculate average similarities across layers
                en_similarities = []
                ko_similarities = []

                for layer in en_comparison['comparison_metrics']:
                    if 'cosine_similarity' in en_comparison['comparison_metrics'][layer]:
                        en_similarities.append(en_comparison['comparison_metrics'][layer]['cosine_similarity'])

                for layer in ko_comparison['comparison_metrics']:
                    if 'cosine_similarity' in ko_comparison['comparison_metrics'][layer]:
                        ko_similarities.append(ko_comparison['comparison_metrics'][layer]['cosine_similarity'])

                if en_similarities and ko_similarities:
                    similarity_metrics = {
                        'english_avg_similarity': np.mean(en_similarities),
                        'korean_avg_similarity': np.mean(ko_similarities),
                        'cross_language_difference': np.mean(en_similarities) - np.mean(ko_similarities)
                    }

            return similarity_metrics

        except Exception as e:
            return {'error': f'Cross-language similarity calculation failed: {e}'}

    def _aggregate_attention_statistics(self, attention_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate attention comparison statistics."""
        english_similarities = []
        korean_similarities = []
        errors = []

        for comparison in attention_comparisons:
            if 'error' not in comparison['english_analysis']:
                en_comp = comparison['english_analysis'].get('comparison', {})
                if 'comparison_metrics' in en_comp:
                    # Extract average similarity across layers
                    layer_similarities = []
                    for layer_data in en_comp['comparison_metrics'].values():
                        if 'cosine_similarity' in layer_data:
                            layer_similarities.append(layer_data['cosine_similarity'])
                    if layer_similarities:
                        english_similarities.append(np.mean(layer_similarities))

            if 'error' not in comparison['korean_analysis']:
                ko_comp = comparison['korean_analysis'].get('comparison', {})
                if 'comparison_metrics' in ko_comp:
                    # Extract average similarity across layers
                    layer_similarities = []
                    for layer_data in ko_comp['comparison_metrics'].values():
                        if 'cosine_similarity' in layer_data:
                            layer_similarities.append(layer_data['cosine_similarity'])
                    if layer_similarities:
                        korean_similarities.append(np.mean(layer_similarities))

            if 'error' in comparison['english_analysis'] or 'error' in comparison['korean_analysis']:
                errors.append(comparison)

        aggregated = {
            'english_statistics': {
                'mean_similarity': np.mean(english_similarities) if english_similarities else 0,
                'std_similarity': np.std(english_similarities) if english_similarities else 0,
                'sample_count': len(english_similarities)
            },
            'korean_statistics': {
                'mean_similarity': np.mean(korean_similarities) if korean_similarities else 0,
                'std_similarity': np.std(korean_similarities) if korean_similarities else 0,
                'sample_count': len(korean_similarities)
            },
            'error_count': len(errors),
            'total_comparisons': len(attention_comparisons)
        }

        # Language comparison
        if english_similarities and korean_similarities:
            aggregated['cross_language_comparison'] = {
                'english_vs_korean_difference': np.mean(english_similarities) - np.mean(korean_similarities),
                'similarity_correlation': np.corrcoef(english_similarities, korean_similarities)[0, 1]
                if len(english_similarities) == len(korean_similarities) else None
            }

        return aggregated

    def _aggregate_confidence_statistics(self, confidence_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate confidence comparison statistics."""
        english_entropy_diffs = []
        korean_entropy_diffs = []
        errors = []

        for comparison in confidence_comparisons:
            if 'error' not in comparison['english_analysis']:
                en_comp = comparison['english_analysis'].get('comparison', {})
                if 'entropy' in en_comp.get('comparisons', {}):
                    entropy_data = en_comp['comparisons']['entropy']
                    if 'mean_entropy_diff' in entropy_data:
                        english_entropy_diffs.append(entropy_data['mean_entropy_diff'])

            if 'error' not in comparison['korean_analysis']:
                ko_comp = comparison['korean_analysis'].get('comparison', {})
                if 'entropy' in ko_comp.get('comparisons', {}):
                    entropy_data = ko_comp['comparisons']['entropy']
                    if 'mean_entropy_diff' in entropy_data:
                        korean_entropy_diffs.append(entropy_data['mean_entropy_diff'])

            if 'error' in comparison['english_analysis'] or 'error' in comparison['korean_analysis']:
                errors.append(comparison)

        aggregated = {
            'english_confidence_changes': {
                'mean_entropy_change': np.mean(english_entropy_diffs) if english_entropy_diffs else 0,
                'std_entropy_change': np.std(english_entropy_diffs) if english_entropy_diffs else 0,
                'sample_count': len(english_entropy_diffs)
            },
            'korean_confidence_changes': {
                'mean_entropy_change': np.mean(korean_entropy_diffs) if korean_entropy_diffs else 0,
                'std_entropy_change': np.std(korean_entropy_diffs) if korean_entropy_diffs else 0,
                'sample_count': len(korean_entropy_diffs)
            },
            'error_count': len(errors),
            'total_comparisons': len(confidence_comparisons)
        }

        return aggregated

    def _generate_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of the comparison results."""
        summary = {
            'overall_performance': {},
            'language_specific_improvements': {},
            'attention_pattern_changes': {},
            'confidence_improvements': {},
            'cross_language_performance': {},
            'recommendations': []
        }

        analysis_results = comparison_results.get('analysis_results', {})

        # Cross-language performance summary
        if 'cross_language' in analysis_results:
            cross_lang = analysis_results['cross_language']
            perf_summary = cross_lang.get('performance_summary', {})

            summary['cross_language_performance'] = {
                'accuracy_improvement': cross_lang.get('accuracy_improvement', 0),
                'base_accuracy': perf_summary.get('base_accuracy', 0),
                'training_accuracy': perf_summary.get('training_accuracy', 0),
                'improvement_ratio': perf_summary.get('improvement_ratio', 1)
            }

        # Generate recommendations based on results
        if 'cross_language' in analysis_results:
            accuracy_improvement = analysis_results['cross_language'].get('accuracy_improvement', 0)
            if accuracy_improvement > 0.1:
                summary['recommendations'].append(
                    "Significant improvement in cross-language accuracy detected. "
                    "Training model shows better multilingual understanding."
                )
            elif accuracy_improvement < -0.05:
                summary['recommendations'].append(
                    "Training model shows decreased cross-language accuracy. "
                    "Consider reviewing training data quality and multilingual coverage."
                )

        # Add more detailed recommendations based on other metrics...

        return summary

    def export_comparison_results(self, comparison_results: Dict[str, Any],
                                output_path: str,
                                format: str = 'json') -> str:
        """Export comparison results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            import json

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            json_data = convert_numpy(comparison_results)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

        elif format == 'csv':
            # Export summary to CSV
            summary_data = []
            if 'summary' in comparison_results:
                summary = comparison_results['summary']
                for section, data in summary.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            summary_data.append({
                                'section': section,
                                'metric': key,
                                'value': value
                            })

            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(output_path, index=False)

        logger.info(f"Comparison results exported to {output_path}")
        return str(output_path)