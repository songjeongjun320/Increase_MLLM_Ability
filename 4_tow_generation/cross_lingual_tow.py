"""
Cross-lingual TOW System - English-Only Thoughts for Multilingual Text
=====================================================================

This system implements the core cross-lingual TOW approach where:
- Source text can be in any language (Korean, Chinese, Japanese, etc.)
- TOW tokens inside <ToW> tags are ALWAYS in English
- This enforces consistent reasoning patterns across languages
- Maintains the cognitive bridging benefits of TOW

Key principle: TOW thoughts serve as universal reasoning tokens in English,
regardless of the source language context.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .utils.text_processing import enforce_english_text, sanitize_tow_token

logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """Supported language codes"""
    KOREAN = "ko"
    ENGLISH = "en" 
    CHINESE = "zh"
    JAPANESE = "ja"
    AUTO = "auto"


@dataclass
class CrossLingualContext:
    """Context for cross-lingual TOW generation"""
    source_text: str
    source_language: str
    target_word: str
    predicted_word: str
    reasoning_type: str
    domain: str = "general"


@dataclass
class EnglishThought:
    """English thought token with metadata"""
    content: str  # English thought content
    confidence: float
    reasoning_type: str
    source_context: str
    target_word: str
    linguistic_pattern: str


class CrossLingualTOWSystem:
    """
    Cross-lingual TOW system that generates English thoughts for any language.
    
    This system ensures that all TOW tokens contain English reasoning,
    creating a universal cognitive bridge regardless of source language.
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        default_language: str = "ko"
    ):
        """
        Initialize Cross-lingual TOW System.
        
        Args:
            model_adapter: Model adapter for thought generation
            default_language: Default source language
        """
        self.model_adapter = model_adapter
        self.default_language = default_language
        
        # Language detection patterns
        self.language_patterns = self._initialize_language_patterns()
        
        # English thought generation templates
        self.english_templates = self._initialize_english_templates()
        
        # Cross-lingual reasoning patterns
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        logger.info(f"CrossLingualTOWSystem initialized with default language: {default_language}")
    
    def generate_english_tow(
        self,
        context: CrossLingualContext
    ) -> str:
        """
        Generate English TOW token for any source language.
        
        This is the core function that ensures TOW tokens are ALWAYS in English,
        regardless of the source language.
        
        Args:
            context: Cross-lingual context for generation
            
        Returns:
            English TOW token in <ToW>...</ToW> format
        """
        try:
            # Detect source language if not specified
            if context.source_language == "auto":
                context.source_language = self.detect_language(context.source_text)
            
            # Generate English reasoning based on context
            english_reasoning = self._generate_english_reasoning(context)
            
            # Enforce English-only and format as TOW token
            english_reasoning = enforce_english_text(english_reasoning)
            tow_token = sanitize_tow_token(f"<ToW>{english_reasoning}</ToW>")
            
            logger.debug(f"Generated English TOW for {context.source_language}: {tow_token}")
            return tow_token
            
        except Exception as e:
            logger.warning(f"Failed to generate English TOW: {e}")
            # Fallback to simple English thought
            return sanitize_tow_token(f"<ToW>The context suggests the word '{context.target_word}'.</ToW>")
    
    def generate_enhanced_tow(
        self,
        source_text: str,
        predicted_word: str,
        actual_word: str,
        category: str,
        source_language: Optional[str] = None
    ) -> str:
        """
        Generate enhanced English TOW with full analysis.
        
        Args:
            source_text: Source context text (any language)
            predicted_word: Model's predicted word
            actual_word: Actual next word
            category: TOW category (trivial/exact/soft/unpredictable)
            source_language: Source language code
            
        Returns:
            Enhanced English TOW token
        """
        # Create context
        context = CrossLingualContext(
            source_text=source_text,
            source_language=source_language or self.detect_language(source_text),
            target_word=actual_word,
            predicted_word=predicted_word,
            reasoning_type=category,
            domain=self._detect_domain(source_text)
        )
        
        # Generate with category-specific enhancement
        enhanced_reasoning = self._generate_category_specific_reasoning(context, category)
        
        enhanced_reasoning = enforce_english_text(enhanced_reasoning)
        return sanitize_tow_token(f"<ToW>{enhanced_reasoning}</ToW>")
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (ko/en/zh/ja)
        """
        if not text:
            return self.default_language
        
        # Check for Korean characters
        if any(0x1100 <= ord(char) <= 0x11FF or  # Hangul Jamo
               0x3130 <= ord(char) <= 0x318F or  # Hangul Compatibility Jamo  
               0xAC00 <= ord(char) <= 0xD7AF     # Hangul Syllables
               for char in text):
            return "ko"
        
        # Check for Chinese characters
        if any(0x4E00 <= ord(char) <= 0x9FFF for char in text):
            # Further distinguish Chinese vs Japanese
            if any(0x3040 <= ord(char) <= 0x309F or  # Hiragana
                   0x30A0 <= ord(char) <= 0x30FF     # Katakana
                   for char in text):
                return "ja"
            return "zh"
        
        # Check for Japanese characters (Hiragana/Katakana without Kanji)
        if any(0x3040 <= ord(char) <= 0x309F or  # Hiragana
               0x30A0 <= ord(char) <= 0x30FF     # Katakana
               for char in text):
            return "ja"
        
        # Default to English for Latin scripts
        return "en"
    
    def _generate_english_reasoning(
        self, 
        context: CrossLingualContext
    ) -> str:
        """
        Generate English reasoning for the given context.
        
        This function creates context-aware English explanations
        regardless of the source language.
        """
        # Get base reasoning template
        template = self._get_reasoning_template(context)
        
        # Generate context-specific reasoning
        reasoning_parts = []
        
        # 1. Context analysis
        context_analysis = self._analyze_context_in_english(context)
        if context_analysis:
            reasoning_parts.append(context_analysis)
        
        # 2. Linguistic pattern analysis
        pattern_analysis = self._analyze_linguistic_patterns(context)
        if pattern_analysis:
            reasoning_parts.append(pattern_analysis)
        
        # 3. Prediction reasoning
        prediction_reasoning = self._generate_prediction_reasoning(context)
        if prediction_reasoning:
            reasoning_parts.append(prediction_reasoning)
        
        # Combine reasoning parts
        if reasoning_parts:
            full_reasoning = " ".join(reasoning_parts)
        else:
            full_reasoning = template.format(
                target_word=context.target_word,
                predicted_word=context.predicted_word,
                language=context.source_language.upper()
            )
        
        # Ensure reasonable length
        if len(full_reasoning) > 200:
            full_reasoning = full_reasoning[:197] + "..."
        
        return full_reasoning
    
    def _generate_category_specific_reasoning(
        self,
        context: CrossLingualContext,
        category: str
    ) -> str:
        """Generate reasoning specific to TOW category"""
        
        if category == "exact_match":
            return self._generate_exact_match_reasoning(context)
        elif category == "soft_consistent":
            return self._generate_soft_consistent_reasoning(context)
        elif category == "trivial":
            return self._generate_trivial_reasoning(context)
        elif category == "unpredictable":
            return self._generate_unpredictable_reasoning(context)
        else:
            return self._generate_english_reasoning(context)
    
    def _generate_exact_match_reasoning(self, context: CrossLingualContext) -> str:
        """Generate reasoning for exact match category"""
        patterns = [
            f"The {context.source_language.upper()} context clearly indicates '{context.target_word}' as the logical continuation.",
            f"Based on the linguistic pattern, '{context.target_word}' is the precise next word.",
            f"The contextual flow in {context.source_language.upper()} strongly suggests '{context.target_word}'."
        ]
        
        base_reasoning = patterns[hash(context.source_text) % len(patterns)]
        
        # Add language-specific enhancement
        if context.source_language == "ko":
            base_reasoning += " This follows Korean grammatical structure."
        elif context.source_language == "zh":
            base_reasoning += " This aligns with Chinese syntactic patterns."
        elif context.source_language == "ja":
            base_reasoning += " This matches Japanese linguistic conventions."
        
        return base_reasoning
    
    def _generate_soft_consistent_reasoning(self, context: CrossLingualContext) -> str:
        """Generate reasoning for soft consistent category"""
        consistency_indicators = []
        
        # Analyze semantic consistency
        if self._has_semantic_similarity(context.predicted_word, context.target_word):
            consistency_indicators.append("semantic similarity")
        
        # Analyze contextual appropriateness
        if self._is_contextually_appropriate(context.target_word, context.source_text):
            consistency_indicators.append("contextual appropriateness")
        
        # Analyze domain relevance
        if context.domain != "general":
            consistency_indicators.append(f"{context.domain} domain relevance")
        
        if consistency_indicators:
            indicators_text = ", ".join(consistency_indicators)
            return f"The word '{context.target_word}' shows {indicators_text} with the {context.source_language.upper()} context."
        else:
            return f"The word '{context.target_word}' is contextually consistent with the {context.source_language.upper()} text pattern."
    
    def _generate_trivial_reasoning(self, context: CrossLingualContext) -> str:
        """Generate reasoning for trivial category"""
        trivial_types = []
        
        # Language-specific trivial patterns
        if context.source_language == "ko":
            if context.target_word in ["은", "는", "이", "가", "을", "를"]:
                trivial_types.append("Korean grammatical particle")
            elif context.target_word in ["그", "이", "저"]:
                trivial_types.append("Korean demonstrative")
            elif len(context.target_word) == 1:
                trivial_types.append("Korean single-character function word")
        elif context.source_language == "en":
            if context.target_word.lower() in ["the", "a", "an", "and", "or", "but"]:
                trivial_types.append("English function word")
        elif context.source_language == "zh":
            if context.target_word in ["的", "了", "在", "是"]:
                trivial_types.append("Chinese function word")
        
        if trivial_types:
            type_text = trivial_types[0]
            return f"'{context.target_word}' is a common {type_text} that frequently appears in this context."
        else:
            return f"'{context.target_word}' is a frequent function word in {context.source_language.upper()} text."
    
    def _generate_unpredictable_reasoning(self, context: CrossLingualContext) -> str:
        """Generate reasoning for unpredictable category"""
        unpredictable_factors = []
        
        # Analyze why it's unpredictable
        if self._requires_world_knowledge(context.target_word):
            unpredictable_factors.append("requires specific domain knowledge")
        
        if self._is_proper_noun(context.target_word):
            unpredictable_factors.append("involves proper nouns")
        
        if self._has_ambiguous_context(context.source_text):
            unpredictable_factors.append("context provides multiple possibilities")
        
        if unpredictable_factors:
            factors_text = " and ".join(unpredictable_factors)
            return f"'{context.target_word}' is unpredictable as it {factors_text}."
        else:
            return f"'{context.target_word}' cannot be reliably predicted from the given {context.source_language.upper()} context alone."
    
    def _analyze_context_in_english(self, context: CrossLingualContext) -> str:
        """Analyze context and provide English description"""
        analysis_parts = []
        
        # Identify key context elements
        if context.domain != "general":
            analysis_parts.append(f"In {context.domain} context")
        
        # Analyze text length and structure
        word_count = len(context.source_text.split())
        if word_count > 20:
            analysis_parts.append("with extensive contextual information")
        elif word_count < 5:
            analysis_parts.append("with limited context")
        
        if analysis_parts:
            return ", ".join(analysis_parts) + ","
        return ""
    
    def _analyze_linguistic_patterns(self, context: CrossLingualContext) -> str:
        """Analyze linguistic patterns specific to source language"""
        patterns = []
        
        if context.source_language == "ko":
            # Korean-specific patterns
            if "습니다" in context.source_text or "입니다" in context.source_text:
                patterns.append("formal Korean ending pattern")
            if any(particle in context.source_text for particle in ["은", "는", "이", "가"]):
                patterns.append("Korean subject marking pattern")
                
        elif context.source_language == "zh":
            # Chinese-specific patterns
            if "的" in context.source_text:
                patterns.append("Chinese possessive/attributive pattern")
            if "了" in context.source_text:
                patterns.append("Chinese completion aspect pattern")
                
        elif context.source_language == "ja":
            # Japanese-specific patterns
            if any(particle in context.source_text for particle in ["は", "が", "を"]):
                patterns.append("Japanese particle pattern")
        
        if patterns:
            return f"following {' and '.join(patterns)}"
        return ""
    
    def _generate_prediction_reasoning(self, context: CrossLingualContext) -> str:
        """Generate reasoning about the prediction process"""
        if context.predicted_word and context.predicted_word != context.target_word:
            return f"while the model predicted '{context.predicted_word}', the actual word '{context.target_word}' better fits the context."
        else:
            return f"leading to the word '{context.target_word}'."
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain of text (same as in pipeline.py)"""
        text_lower = text.lower()
        
        # Programming keywords
        if any(word in text_lower for word in ["코드", "프로그램", "함수", "변수", "code", "function", "algorithm"]):
            return "programming"
        
        # Mathematics keywords  
        if any(word in text_lower for word in ["수학", "방정식", "계산", "공식", "math", "equation", "formula"]):
            return "mathematics"
        
        # Science keywords
        if any(word in text_lower for word in ["실험", "연구", "이론", "experiment", "research", "theory"]):
            return "science"
        
        return "general"
    
    def _get_reasoning_template(self, context: CrossLingualContext) -> str:
        """Get appropriate reasoning template"""
        templates = self.english_templates.get(context.reasoning_type, [])
        if templates:
            return templates[hash(context.source_text) % len(templates)]
        
        # Default template
        return "The {language} context suggests the word '{target_word}'."
    
    def _has_semantic_similarity(self, word1: str, word2: str) -> bool:
        """Check if two words are semantically similar (simplified)"""
        if not word1 or not word2:
            return False
        
        # Simple similarity checks
        if word1.lower() == word2.lower():
            return True
        
        # Check if one is contained in the other
        if word1.lower() in word2.lower() or word2.lower() in word1.lower():
            return True
        
        # Could be enhanced with word embeddings in production
        return False
    
    def _is_contextually_appropriate(self, word: str, context: str) -> bool:
        """Check if word is contextually appropriate"""
        # Simple context matching
        return word.lower() in context.lower()
    
    def _requires_world_knowledge(self, word: str) -> bool:
        """Check if word requires world knowledge"""
        # Simple heuristics
        if word.isupper() and len(word) > 1:  # Likely acronym
            return True
        if word[0].isupper() and len(word) > 3:  # Likely proper noun
            return True
        return False
    
    def _is_proper_noun(self, word: str) -> bool:
        """Check if word is a proper noun"""
        return word[0].isupper() and len(word) > 1
    
    def _has_ambiguous_context(self, context: str) -> bool:
        """Check if context is ambiguous"""
        # Simple heuristics for ambiguity
        ambiguity_markers = ["또는", "혹은", "maybe", "perhaps", "either", "or"]
        return any(marker in context.lower() for marker in ambiguity_markers)
    
    def _initialize_language_patterns(self) -> Dict[str, List[str]]:
        """Initialize language detection patterns"""
        return {
            "ko": ["은", "는", "이", "가", "을", "를", "습니다", "입니다", "해요", "이에요"],
            "zh": ["的", "了", "在", "是", "有", "我", "你", "他", "她", "它"],
            "ja": ["は", "が", "を", "に", "で", "から", "まで", "と", "です", "ます"],
            "en": ["the", "and", "or", "but", "in", "on", "at", "to", "for", "with"]
        }
    
    def _initialize_english_templates(self) -> Dict[str, List[str]]:
        """Initialize English reasoning templates"""
        return {
            "exact_match": [
                "The {language} context clearly indicates '{target_word}' as the next word.",
                "Based on the linguistic pattern, '{target_word}' is the logical continuation.",
                "The contextual flow strongly suggests '{target_word}'."
            ],
            "soft_consistent": [
                "The {language} text pattern suggests '{target_word}' or a similar word.",
                "Given the context, '{target_word}' is contextually appropriate.",
                "The linguistic structure indicates '{target_word}' fits the pattern."
            ],
            "trivial": [
                "'{target_word}' is a common function word in {language} text.",
                "This word serves a grammatical role in the {language} sentence structure.",
                "'{target_word}' is a frequent structural element."
            ],
            "unpredictable": [
                "'{target_word}' is difficult to predict from the {language} context alone.",
                "The next word requires additional domain knowledge beyond the immediate context.",
                "'{target_word}' represents an unexpected but valid continuation."
            ]
        }
    
    def _initialize_reasoning_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize cross-lingual reasoning patterns"""
        return {
            "ko": {
                "grammatical": ["particle usage", "honorific system", "verb ending patterns"],
                "structural": ["SOV word order", "topic-comment structure", "case marking"],
                "semantic": ["context-dependent meaning", "pragmatic inference", "cultural context"]
            },
            "zh": {
                "grammatical": ["aspect marking", "classifier usage", "tone patterns"],
                "structural": ["SVO word order", "topic-prominence", "serial verb construction"],
                "semantic": ["character-based meaning", "contextual disambiguation", "idiomatic expressions"]
            },
            "ja": {
                "grammatical": ["particle system", "honorific levels", "verb conjugation"],
                "structural": ["SOV word order", "topic-comment structure", "agglutination"],
                "semantic": ["context sensitivity", "implicit subjects", "social register"]
            },
            "en": {
                "grammatical": ["article usage", "auxiliary verbs", "modal constructions"],
                "structural": ["SVO word order", "prepositional phrases", "relative clauses"],
                "semantic": ["compositional meaning", "phrasal verbs", "idiomatic usage"]
            }
        }
    
    def format_cross_lingual_dataset(
        self,
        source_texts: List[str],
        tow_tokens: List[str],
        languages: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Format cross-lingual dataset with English TOW tokens.
        
        Args:
            source_texts: List of source texts in various languages
            tow_tokens: List of English TOW tokens
            languages: List of language codes
            
        Returns:
            Formatted dataset entries
        """
        formatted_entries = []
        
        for i, (text, tow, lang) in enumerate(zip(source_texts, tow_tokens, languages)):
            entry = {
                "id": i,
                "source_text": text,
                "source_language": lang,
                "tow_token": tow,  # Always in English
                "reasoning_language": "en",  # Always English
                "cross_lingual": lang != "en",  # True if source is not English
                "metadata": {
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "domain": self._detect_domain(text)
                }
            }
            formatted_entries.append(entry)
        
        return formatted_entries
