"""
Sentence Embedding Analysis Module

Provides functionality for generating and analyzing sentence embeddings using Sentence-BERT
and other multilingual models. Includes dimensionality reduction and similarity analysis.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
import umap
import logging
from pathlib import Path

from models.model_manager import get_model_manager
from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class SentenceEmbeddingAnalyzer:
    """Analyzes sentence embeddings for multilingual comparison."""

    def __init__(self):
        """Initialize the sentence embedding analyzer."""
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.embeddings_cache = {}
        self.reduced_embeddings_cache = {}

    def generate_embeddings(self, texts: List[str], languages: List[str],
                          model_name: Optional[str] = None,
                          normalize: bool = True) -> Dict[str, Any]:
        """
        Generate sentence embeddings for given texts.

        Args:
            texts: List of texts to encode
            languages: List of language codes corresponding to each text
            model_name: Sentence transformer model name
            normalize: Whether to normalize embeddings

        Returns:
            Dictionary containing embeddings and metadata
        """
        if len(texts) != len(languages):
            raise ValueError("Number of texts must match number of language codes")

        # Generate embeddings
        embeddings = self.model_manager.generate_embeddings(
            texts=texts,
            model_name=model_name,
            normalize=normalize
        )

        # Convert to numpy for easier manipulation
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Create metadata dataframe
        metadata = pd.DataFrame({
            'text': texts,
            'language': languages,
            'text_length': [len(text) for text in texts],
            'word_count': [len(text.split()) for text in texts]
        })

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        result = {
            'embeddings': embeddings,
            'metadata': metadata,
            'similarity_matrix': similarity_matrix,
            'model_name': model_name or self.config.get('models.sentence_transformer.default_model'),
            'embedding_dim': embeddings.shape[1],
            'num_samples': len(texts)
        }

        # Cache for future use
        cache_key = f"{model_name}_{hash(tuple(texts))}"
        self.embeddings_cache[cache_key] = result

        logger.info(f"Generated embeddings for {len(texts)} texts using {result['model_name']}")
        return result

    def reduce_dimensions(self, embeddings: np.ndarray,
                        method: str = 'umap',
                        n_components: int = 2,
                        **kwargs) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.

        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of dimensions to reduce to
            **kwargs: Additional parameters for the reduction method

        Returns:
            Reduced embeddings
        """
        method = method.lower()
        cache_key = f"{method}_{n_components}_{hash(embeddings.tobytes())}"

        if cache_key in self.reduced_embeddings_cache:
            logger.info(f"Using cached {method} reduction")
            return self.reduced_embeddings_cache[cache_key]

        logger.info(f"Reducing dimensions using {method} to {n_components}D")

        if method == 'pca':
            reducer = PCA(
                n_components=n_components,
                random_state=kwargs.get('random_state', 42)
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=kwargs.get('perplexity', 30),
                random_state=kwargs.get('random_state', 42),
                n_iter=kwargs.get('n_iter', 1000)
            )
        elif method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported reduction method: {method}")

        try:
            reduced_embeddings = reducer.fit_transform(embeddings)
            self.reduced_embeddings_cache[cache_key] = reduced_embeddings
            logger.info(f"Successfully reduced dimensions from {embeddings.shape[1]} to {n_components}")
            return reduced_embeddings

        except Exception as e:
            logger.error(f"Dimension reduction failed: {e}")
            raise

    def analyze_language_clusters(self, embeddings: np.ndarray,
                                languages: List[str],
                                method: str = 'kmeans') -> Dict[str, Any]:
        """
        Analyze clustering patterns by language.

        Args:
            embeddings: Sentence embeddings
            languages: Language codes for each embedding
            method: Clustering method ('kmeans', 'dbscan')

        Returns:
            Dictionary with clustering analysis results
        """
        unique_languages = list(set(languages))
        n_languages = len(unique_languages)

        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_languages, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # Analyze cluster-language correspondence
        cluster_lang_matrix = pd.crosstab(
            pd.Series(languages, name='Language'),
            pd.Series(cluster_labels, name='Cluster')
        )

        # Calculate cluster purity for each language
        language_purity = {}
        for lang in unique_languages:
            lang_mask = np.array(languages) == lang
            lang_clusters = cluster_labels[lang_mask]
            if len(lang_clusters) > 0:
                # Most common cluster for this language
                most_common_cluster = pd.Series(lang_clusters).mode().iloc[0]
                purity = np.sum(lang_clusters == most_common_cluster) / len(lang_clusters)
                language_purity[lang] = purity

        # Calculate silhouette score if possible
        try:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(embeddings, cluster_labels)
        except Exception:
            silhouette = None

        return {
            'cluster_labels': cluster_labels,
            'cluster_language_matrix': cluster_lang_matrix,
            'language_purity': language_purity,
            'silhouette_score': silhouette,
            'num_clusters': len(set(cluster_labels)),
            'method': method
        }

    def compare_sentence_pairs(self, sentences_en: List[str],
                             sentences_target: List[str],
                             target_language: str = 'ko',
                             model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare sentence pairs between English and target language.

        Args:
            sentences_en: English sentences
            sentences_target: Target language sentences (parallel to English)
            target_language: Target language code
            model_name: Sentence transformer model name

        Returns:
            Dictionary with comparison results
        """
        if len(sentences_en) != len(sentences_target):
            raise ValueError("Number of English and target sentences must match")

        # Combine all sentences for embedding generation
        all_sentences = sentences_en + sentences_target
        all_languages = ['en'] * len(sentences_en) + [target_language] * len(sentences_target)

        # Generate embeddings
        embedding_result = self.generate_embeddings(
            texts=all_sentences,
            languages=all_languages,
            model_name=model_name
        )

        embeddings = embedding_result['embeddings']

        # Split embeddings back
        en_embeddings = embeddings[:len(sentences_en)]
        target_embeddings = embeddings[len(sentences_en):]

        # Calculate pair-wise similarities
        pair_similarities = []
        for i in range(len(sentences_en)):
            similarity = cosine_similarity([en_embeddings[i]], [target_embeddings[i]])[0, 0]
            pair_similarities.append(similarity)

        # Calculate cross-language similarity matrix
        cross_similarity = cosine_similarity(en_embeddings, target_embeddings)

        # Find best matches for each English sentence in target language
        best_matches = []
        for i, en_sentence in enumerate(sentences_en):
            similarities = cross_similarity[i]
            best_match_idx = np.argmax(similarities)
            best_matches.append({
                'en_sentence': en_sentence,
                'en_index': i,
                'best_match_sentence': sentences_target[best_match_idx],
                'best_match_index': best_match_idx,
                'similarity': similarities[best_match_idx],
                'is_correct_pair': best_match_idx == i,
                'correct_pair_similarity': pair_similarities[i]
            })

        # Calculate statistics
        correct_matches = sum(1 for match in best_matches if match['is_correct_pair'])
        accuracy = correct_matches / len(best_matches) if best_matches else 0

        return {
            'embedding_result': embedding_result,
            'en_embeddings': en_embeddings,
            'target_embeddings': target_embeddings,
            'pair_similarities': np.array(pair_similarities),
            'cross_similarity_matrix': cross_similarity,
            'best_matches': best_matches,
            'accuracy': accuracy,
            'mean_pair_similarity': np.mean(pair_similarities),
            'std_pair_similarity': np.std(pair_similarities),
            'target_language': target_language
        }

    def analyze_semantic_similarity(self, texts: List[str],
                                  languages: List[str],
                                  similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze semantic similarity patterns within and across languages.

        Args:
            texts: List of texts
            languages: Language codes
            similarity_threshold: Threshold for considering texts as similar

        Returns:
            Dictionary with similarity analysis
        """
        # Generate embeddings
        embedding_result = self.generate_embeddings(texts, languages)
        embeddings = embedding_result['embeddings']
        similarity_matrix = embedding_result['similarity_matrix']

        # Create language groups
        language_groups = {}
        for i, lang in enumerate(languages):
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(i)

        # Analyze within-language similarities
        within_language_similarities = {}
        for lang, indices in language_groups.items():
            if len(indices) > 1:
                lang_similarities = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        lang_similarities.append(similarity_matrix[idx1, idx2])
                within_language_similarities[lang] = {
                    'similarities': lang_similarities,
                    'mean': np.mean(lang_similarities),
                    'std': np.std(lang_similarities),
                    'max': np.max(lang_similarities),
                    'min': np.min(lang_similarities)
                }

        # Analyze cross-language similarities
        cross_language_similarities = {}
        unique_languages = list(language_groups.keys())
        for i, lang1 in enumerate(unique_languages):
            for lang2 in unique_languages[i + 1:]:
                pair_key = f"{lang1}-{lang2}"
                similarities = []
                for idx1 in language_groups[lang1]:
                    for idx2 in language_groups[lang2]:
                        similarities.append(similarity_matrix[idx1, idx2])

                cross_language_similarities[pair_key] = {
                    'similarities': similarities,
                    'mean': np.mean(similarities),
                    'std': np.std(similarities),
                    'max': np.max(similarities),
                    'min': np.min(similarities)
                }

        # Find highly similar text pairs
        similar_pairs = []
        n_texts = len(texts)
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    similar_pairs.append({
                        'text1': texts[i],
                        'text2': texts[j],
                        'language1': languages[i],
                        'language2': languages[j],
                        'similarity': similarity,
                        'is_cross_language': languages[i] != languages[j]
                    })

        return {
            'similarity_matrix': similarity_matrix,
            'within_language_similarities': within_language_similarities,
            'cross_language_similarities': cross_language_similarities,
            'similar_pairs': similar_pairs,
            'language_groups': language_groups,
            'similarity_threshold': similarity_threshold,
            'total_similar_pairs': len(similar_pairs),
            'cross_language_similar_pairs': sum(1 for pair in similar_pairs if pair['is_cross_language'])
        }

    def export_embeddings(self, embedding_result: Dict[str, Any],
                         output_path: str,
                         format: str = 'npz') -> str:
        """
        Export embeddings and metadata to file.

        Args:
            embedding_result: Result from generate_embeddings
            output_path: Output file path
            format: Export format ('npz', 'csv', 'json')

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npz':
            np.savez_compressed(
                output_path,
                embeddings=embedding_result['embeddings'],
                similarity_matrix=embedding_result['similarity_matrix'],
                texts=embedding_result['metadata']['text'].values,
                languages=embedding_result['metadata']['language'].values,
                model_name=embedding_result['model_name']
            )
        elif format == 'csv':
            # Export embeddings as CSV with metadata
            df = embedding_result['metadata'].copy()

            # Add embedding dimensions as columns
            embeddings = embedding_result['embeddings']
            for i in range(embeddings.shape[1]):
                df[f'emb_{i}'] = embeddings[:, i]

            df.to_csv(output_path, index=False)

        elif format == 'json':
            import json

            # Convert numpy arrays to lists for JSON serialization
            export_data = {
                'embeddings': embedding_result['embeddings'].tolist(),
                'similarity_matrix': embedding_result['similarity_matrix'].tolist(),
                'metadata': embedding_result['metadata'].to_dict('records'),
                'model_name': embedding_result['model_name'],
                'embedding_dim': embedding_result['embedding_dim'],
                'num_samples': embedding_result['num_samples']
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Embeddings exported to {output_path}")
        return str(output_path)

    def clear_cache(self):
        """Clear all cached embeddings and reduced embeddings."""
        self.embeddings_cache.clear()
        self.reduced_embeddings_cache.clear()
        logger.info("Embedding caches cleared")