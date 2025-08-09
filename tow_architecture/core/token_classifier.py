"""
Token Classifier - TOW Token Classification System
=================================================

Implements the four categories of TOW token classification:
1. Trivial: Frequently occurring words with no special meaning
2. Exact Match: Predicted word matches actual next word exactly
3. Soft Consistent: Predicted word is contextually similar but not exact
4. Unpredictable: Words that cannot be reasonably predicted by the model

This classifier helps categorize TOW tokens for training data augmentation.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from collections import Counter, defaultdict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TOWCategory(Enum):
    """TOW Token Categories based on original paper"""
    TRIVIAL = "trivial"
    EXACT_MATCH = "exact_match"
    SOFT_CONSISTENT = "soft_consistent"
    UNPREDICTABLE = "unpredictable"


@dataclass
class TokenClassificationResult:
    """Result of token classification"""
    token: str
    category: TOWCategory
    confidence: float
    reasoning: str
    context_words: List[str]
    similarity_score: float = 0.0
    frequency_score: float = 0.0


@dataclass
class ClassificationContext:
    """Context for token classification"""
    preceding_text: str
    predicted_token: str
    actual_token: str
    domain: str = "general"
    language: str = "ko"


class TokenClassifier:
    """
    Token Classifier for TOW token categorization.
    
    Classifies tokens into four categories according to the original TOW paper:
    - Trivial: Common function words, particles, etc.
    - Exact Match: Perfect prediction match
    - Soft Consistent: Contextually appropriate but not exact
    - Unpredictable: Cannot be reasonably predicted from context
    """
    
    def __init__(self, language: str = "ko"):
        """
        Initialize Token Classifier.
        
        Args:
            language: Primary language for classification (default: Korean)
        """
        self.language = language
        
        # Load language-specific resources
        self.trivial_words = self._load_trivial_words()
        self.function_words = self._load_function_words()
        self.domain_vocabularies = self._load_domain_vocabularies()
        
        # Classification thresholds
        self.thresholds = {
            "trivial_frequency": 0.8,    # High frequency threshold for trivial
            "exact_match": 1.0,          # Perfect match
            "soft_consistency": 0.6,     # Semantic similarity threshold
            "unpredictable": 0.3         # Low predictability threshold
        }
        
        # Statistics tracking
        self.classification_stats = Counter()
        self.word_frequencies = Counter()
        
        logger.info(f"TokenClassifier initialized for language: {language}")
    
    def classify_token(
        self, 
        context: ClassificationContext
    ) -> TokenClassificationResult:
        """
        Classify a token based on context and prediction.
        
        Args:
            context: Classification context with text, predicted and actual tokens
            
        Returns:
            TokenClassificationResult with category and metadata
        """
        predicted = context.predicted_token.strip()
        actual = context.actual_token.strip()
        
        # Update frequency statistics
        self.word_frequencies[actual] += 1
        
        # 1. Check for exact match first
        if self._is_exact_match(predicted, actual):
            result = TokenClassificationResult(
                token=actual,
                category=TOWCategory.EXACT_MATCH,
                confidence=1.0,
                reasoning="Predicted token exactly matches actual token",
                context_words=self._extract_context_words(context.preceding_text)
            )
            self.classification_stats[TOWCategory.EXACT_MATCH] += 1
            return result
        
        # 2. Check for trivial words
        if self._is_trivial(actual, context):
            result = TokenClassificationResult(
                token=actual,
                category=TOWCategory.TRIVIAL,
                confidence=self._calculate_trivial_confidence(actual, context),
                reasoning=self._get_trivial_reasoning(actual),
                context_words=self._extract_context_words(context.preceding_text),
                frequency_score=self._get_frequency_score(actual)
            )
            self.classification_stats[TOWCategory.TRIVIAL] += 1
            return result
        
        # 3. Check for soft consistency
        consistency_score = self._calculate_consistency(predicted, actual, context)
        if consistency_score >= self.thresholds["soft_consistency"]:
            result = TokenClassificationResult(
                token=actual,
                category=TOWCategory.SOFT_CONSISTENT,
                confidence=consistency_score,
                reasoning=self._get_consistency_reasoning(predicted, actual, consistency_score),
                context_words=self._extract_context_words(context.preceding_text),
                similarity_score=consistency_score
            )
            self.classification_stats[TOWCategory.SOFT_CONSISTENT] += 1
            return result
        
        # 4. Default to unpredictable
        result = TokenClassificationResult(
            token=actual,
            category=TOWCategory.UNPREDICTABLE,
            confidence=1.0 - consistency_score,
            reasoning=self._get_unpredictable_reasoning(predicted, actual, context),
            context_words=self._extract_context_words(context.preceding_text)
        )
        self.classification_stats[TOWCategory.UNPREDICTABLE] += 1
        return result
    
    def classify_batch(
        self, 
        contexts: List[ClassificationContext]
    ) -> List[TokenClassificationResult]:
        """
        Classify multiple tokens in batch.
        
        Args:
            contexts: List of classification contexts
            
        Returns:
            List of TokenClassificationResult objects
        """
        results = []
        for context in contexts:
            result = self.classify_token(context)
            results.append(result)
        
        logger.info(f"Batch classification completed: {len(results)} tokens classified")
        return results
    
    def _is_exact_match(self, predicted: str, actual: str) -> bool:
        """Check if predicted token exactly matches actual token"""
        # Normalize whitespace and case for comparison
        pred_normalized = re.sub(r'\s+', ' ', predicted.lower().strip())
        actual_normalized = re.sub(r'\s+', ' ', actual.lower().strip())
        return pred_normalized == actual_normalized
    
    def _is_trivial(self, token: str, context: ClassificationContext) -> bool:
        """
        Check if token is trivial (common function word, particle, etc.)
        
        Args:
            token: Token to check
            context: Classification context
            
        Returns:
            True if token is trivial
        """
        token_lower = token.lower().strip()
        
        # Check against trivial word lists
        if token_lower in self.trivial_words:
            return True
        
        # Check function words
        if token_lower in self.function_words:
            return True
        
        # Check for single characters (often particles in Korean)
        if len(token.strip()) == 1 and self.language == "ko":
            korean_particles = {"은", "는", "이", "가", "을", "를", "에", "서", "로", "와", "과", "도", "만"}
            if token.strip() in korean_particles:
                return True
        
        # Check frequency-based triviality
        frequency_score = self._get_frequency_score(token)
        if frequency_score > self.thresholds["trivial_frequency"]:
            return True
        
        return False
    
    def _calculate_consistency(
        self, 
        predicted: str, 
        actual: str, 
        context: ClassificationContext
    ) -> float:
        """
        Calculate consistency score between predicted and actual tokens.
        
        Args:
            predicted: Predicted token
            actual: Actual token
            context: Classification context
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        # Semantic similarity based on multiple factors
        scores = []
        
        # 1. String similarity (edit distance)
        string_sim = self._calculate_string_similarity(predicted, actual)
        scores.append(string_sim * 0.3)
        
        # 2. Contextual appropriateness
        context_sim = self._calculate_contextual_similarity(predicted, actual, context)
        scores.append(context_sim * 0.4)
        
        # 3. Domain relevance
        domain_sim = self._calculate_domain_similarity(predicted, actual, context.domain)
        scores.append(domain_sim * 0.3)
        
        return sum(scores)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using edit distance"""
        if not str1 or not str2:
            return 0.0
        
        # Simple Levenshtein-based similarity
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        distance = levenshtein(str1.lower(), str2.lower())
        return 1.0 - (distance / max_len)
    
    def _calculate_contextual_similarity(
        self, 
        predicted: str, 
        actual: str, 
        context: ClassificationContext
    ) -> float:
        """Calculate how well both tokens fit the context"""
        # Extract context words
        context_words = self._extract_context_words(context.preceding_text)
        
        if not context_words:
            return 0.5  # Neutral score if no context
        
        # Simple context matching based on common patterns
        predicted_score = self._calculate_context_fit(predicted, context_words)
        actual_score = self._calculate_context_fit(actual, context_words)
        
        # If both fit well, they're consistent
        if predicted_score > 0.5 and actual_score > 0.5:
            return min(predicted_score, actual_score)
        
        # If neither fits well, still somewhat consistent
        if predicted_score <= 0.5 and actual_score <= 0.5:
            return 0.4
        
        # If one fits much better than the other, less consistent
        return 0.3
    
    def _calculate_context_fit(self, token: str, context_words: List[str]) -> float:
        """Calculate how well a token fits with context words"""
        if not context_words:
            return 0.5
        
        # Simple word co-occurrence based scoring
        # In a real implementation, this could use word embeddings
        token_lower = token.lower()
        context_lower = [w.lower() for w in context_words]
        
        # Check for direct mentions
        if token_lower in context_lower:
            return 0.8
        
        # Check for partial matches
        partial_matches = sum(1 for w in context_lower if token_lower in w or w in token_lower)
        if partial_matches > 0:
            return 0.6
        
        # Domain-specific patterns could be added here
        return 0.3
    
    def _calculate_domain_similarity(
        self, 
        predicted: str, 
        actual: str, 
        domain: str
    ) -> float:
        """Calculate domain-specific similarity"""
        domain_vocab = self.domain_vocabularies.get(domain, set())
        
        if not domain_vocab:
            return 0.5  # Neutral if no domain vocabulary
        
        predicted_in_domain = predicted.lower() in domain_vocab
        actual_in_domain = actual.lower() in domain_vocab
        
        if predicted_in_domain and actual_in_domain:
            return 0.8
        elif predicted_in_domain or actual_in_domain:
            return 0.6
        else:
            return 0.4
    
    def _extract_context_words(self, text: str) -> List[str]:
        """Extract meaningful words from context"""
        if not text:
            return []
        
        # Simple tokenization and filtering
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out trivial words
        meaningful_words = [w for w in words if w not in self.trivial_words and len(w) > 1]
        
        # Return last N words for context
        return meaningful_words[-10:] if meaningful_words else []
    
    def _get_frequency_score(self, token: str) -> float:
        """Get frequency-based score for token"""
        if not self.word_frequencies:
            return 0.0
        
        total_words = sum(self.word_frequencies.values())
        token_freq = self.word_frequencies.get(token, 0)
        
        if total_words == 0:
            return 0.0
        
        return token_freq / total_words
    
    def _calculate_trivial_confidence(
        self, 
        token: str, 
        context: ClassificationContext
    ) -> float:
        """Calculate confidence for trivial classification"""
        confidence_factors = []
        
        # Frequency factor
        freq_score = self._get_frequency_score(token)
        confidence_factors.append(freq_score)
        
        # Function word factor
        if token.lower() in self.function_words:
            confidence_factors.append(0.9)
        
        # Length factor (shorter words more likely to be trivial)
        if len(token.strip()) <= 2:
            confidence_factors.append(0.8)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _get_trivial_reasoning(self, token: str) -> str:
        """Get reasoning for trivial classification"""
        reasons = []
        
        if token.lower() in self.trivial_words:
            reasons.append("listed in trivial words")
        
        if token.lower() in self.function_words:
            reasons.append("function word")
        
        freq_score = self._get_frequency_score(token)
        if freq_score > self.thresholds["trivial_frequency"]:
            reasons.append(f"high frequency ({freq_score:.3f})")
        
        if len(token.strip()) == 1:
            reasons.append("single character (likely particle)")
        
        if reasons:
            return f"Classified as trivial: {', '.join(reasons)}"
        else:
            return "Classified as trivial based on general patterns"
    
    def _get_consistency_reasoning(
        self, 
        predicted: str, 
        actual: str, 
        score: float
    ) -> str:
        """Get reasoning for soft consistency classification"""
        return f"Soft consistent: predicted '{predicted}' is contextually similar to actual '{actual}' (similarity: {score:.3f})"
    
    def _get_unpredictable_reasoning(
        self, 
        predicted: str, 
        actual: str, 
        context: ClassificationContext
    ) -> str:
        """Get reasoning for unpredictable classification"""
        return f"Unpredictable: actual token '{actual}' could not be reasonably predicted from context, predicted '{predicted}'"
    
    def _load_trivial_words(self) -> Set[str]:
        """Load language-specific trivial words"""
        if self.language == "ko":
            return {
                # Korean particles
                "은", "는", "이", "가", "을", "를", "에", "서", "로", "와", "과", "도", "만", "부터", "까지",
                # Common function words
                "그", "이", "저", "그런", "이런", "저런", "것", "수", "때", "곳", "분", "년", "월", "일",
                # Conjunctions
                "그리고", "하지만", "그러나", "또한", "따라서", "그래서", "즉", "또는", "혹은",
                # Common verbs
                "이다", "있다", "없다", "되다", "하다", "가다", "오다", "보다", "듣다", "말하다",
                # Numbers
                "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열"
            }
        elif self.language == "en":
            return {
                # Articles
                "a", "an", "the",
                # Prepositions
                "of", "in", "to", "for", "with", "on", "by", "from", "up", "about", "into", "over", "after",
                # Conjunctions
                "and", "or", "but", "so", "yet", "for", "nor",
                # Common pronouns
                "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                # Common verbs
                "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did"
            }
        else:
            return set()
    
    def _load_function_words(self) -> Set[str]:
        """Load language-specific function words"""
        if self.language == "ko":
            return {
                "그", "이", "저", "여기", "거기", "저기", "이것", "그것", "저것",
                "누구", "무엇", "언제", "어디", "어떻게", "왜", "어느", "몇",
                "모든", "각", "어떤", "다른", "같은", "새로운", "오래된"
            }
        elif self.language == "en":
            return {
                "this", "that", "these", "those", "here", "there", "where", "when", "how", "why",
                "who", "what", "which", "whose", "whom",
                "all", "any", "each", "every", "some", "many", "much", "few", "little",
                "other", "another", "same", "different", "new", "old"
            }
        else:
            return set()
    
    def _load_domain_vocabularies(self) -> Dict[str, Set[str]]:
        """Load domain-specific vocabularies"""
        return {
            "programming": {
                "function", "variable", "class", "method", "object", "array", "list", "string",
                "integer", "boolean", "loop", "condition", "algorithm", "data", "structure",
                "함수", "변수", "클래스", "메서드", "객체", "배열", "리스트", "문자열", 
                "정수", "불린", "반복", "조건", "알고리즘", "데이터", "구조"
            },
            "mathematics": {
                "equation", "formula", "calculate", "solve", "proof", "theorem", "variable",
                "function", "derivative", "integral", "limit", "matrix", "vector",
                "방정식", "공식", "계산", "해결", "증명", "정리", "변수",
                "함수", "도함수", "적분", "극한", "행렬", "벡터"
            },
            "science": {
                "experiment", "hypothesis", "theory", "observation", "data", "result",
                "analysis", "conclusion", "research", "study", "method", "process",
                "실험", "가설", "이론", "관찰", "데이터", "결과",
                "분석", "결론", "연구", "학습", "방법", "과정"
            }
        }
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification statistics"""
        total_classifications = sum(self.classification_stats.values())
        
        if total_classifications == 0:
            return {"total": 0, "categories": {}}
        
        stats = {
            "total": total_classifications,
            "categories": {}
        }
        
        for category in TOWCategory:
            count = self.classification_stats[category]
            percentage = (count / total_classifications) * 100
            stats["categories"][category.value] = {
                "count": count,
                "percentage": percentage
            }
        
        return stats
    
    def export_classifications(self, filepath: str, results: List[TokenClassificationResult]):
        """Export classification results to file"""
        export_data = []
        
        for result in results:
            export_data.append({
                "token": result.token,
                "category": result.category.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "context_words": result.context_words,
                "similarity_score": result.similarity_score,
                "frequency_score": result.frequency_score
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Classification results exported to {filepath}")
