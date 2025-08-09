"""
Data Augmentation Pipeline - TOW Training Corpus Enhancement
===========================================================

This pipeline implements the core data augmentation system for generating
TOW (Thoughts of Words) training datasets. It processes raw text corpora
and generates enhanced datasets with proper <ToW> token formatting.

Key Features:
- Batch processing of large corpora
- Token classification (trivial/exact/soft/unpredictable)
- Cross-lingual TOW generation (English thoughts for non-English text)
- Quality filtering and validation
- Multiple output formats for training
"""

import logging
import asyncio
import json
import os
import random
from typing import Dict, List, Optional, Any, Tuple, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import time

from ..core.token_classifier import TokenClassifier, TOWCategory, ClassificationContext
from ..models.base_adapter import BaseModelAdapter
from ..utils.logger import get_logger
from ..utils.text_utils import enforce_english_text, sanitize_tow_token
from ..utils.config import DataAugmentationConfig

logger = get_logger(__name__)


@dataclass
class TOWEntry:
    """Single TOW training entry"""
    original_text: str
    context: str
    predicted_token: str
    actual_token: str
    thought_token: str  # English thought in <ToW> format
    category: str
    confidence: float
    language: str
    domain: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingStats:
    """Statistics for processing run"""
    total_processed: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    category_counts: Dict[str, int] = None
    processing_time: float = 0.0
    average_confidence: float = 0.0
    
    def __post_init__(self):
        if self.category_counts is None:
            self.category_counts = {cat.value: 0 for cat in TOWCategory}


class TOWDataAugmentationPipeline:
    """
    Main pipeline for TOW data augmentation.
    
    Processes raw text corpora and generates training datasets with
    proper TOW token classification and English thought generation.
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        config: Optional[DataAugmentationConfig] = None,
        language: str = "ko"
    ):
        """
        Initialize TOW Data Augmentation Pipeline.
        
        Args:
            model_adapter: Model adapter for thought generation
            config: Pipeline configuration
            language: Target language for processing
        """
        self.model_adapter = model_adapter
        self.config = config or DataAugmentationConfig()
        self.language = language
        
        # Initialize components
        self.token_classifier = TokenClassifier(language=language)
        
        # Processing queues and threading
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.processing_stats = ProcessingStats()
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # English thought generation templates
        self.thought_templates = self._load_thought_templates()
        
        logger.info(f"TOWDataAugmentationPipeline initialized for language: {language}")
    
    def process_corpus(
        self,
        input_data: List[str],
        output_path: str,
        batch_size: int = 100,
        max_workers: int = 4
    ) -> ProcessingStats:
        """
        Process a corpus to generate TOW training data.
        
        Args:
            input_data: List of text strings to process
            output_path: Path to save generated TOW dataset
            batch_size: Batch size for processing
            max_workers: Number of worker threads
            
        Returns:
            ProcessingStats with processing results
        """
        logger.info(f"Starting corpus processing: {len(input_data)} texts")
        start_time = time.time()
        
        # Reset statistics
        self.processing_stats = ProcessingStats()
        
        # Process in batches
        all_tow_entries = []
        total_batches = (len(input_data) + batch_size - 1) // batch_size
        
        for i in range(0, len(input_data), batch_size):
            batch_num = (i // batch_size) + 1
            batch_data = input_data[i:i + batch_size]
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_data)} texts)")
            
            # Process batch
            batch_results = self._process_batch(batch_data, max_workers)
            all_tow_entries.extend(batch_results)
            
            # Update statistics
            with self.stats_lock:
                self.processing_stats.total_processed += len(batch_data)
                self.processing_stats.successful_generations += len(batch_results)
                self.processing_stats.failed_generations += len(batch_data) - len(batch_results)
        
        # Calculate final statistics
        self.processing_stats.processing_time = time.time() - start_time
        if all_tow_entries:
            self.processing_stats.average_confidence = sum(
                entry.confidence for entry in all_tow_entries
            ) / len(all_tow_entries)
        
        # Count categories
        for entry in all_tow_entries:
            self.processing_stats.category_counts[entry.category] += 1
        
        # Save results
        self._save_tow_dataset(all_tow_entries, output_path)
        
        logger.info(
            f"Corpus processing completed: {len(all_tow_entries)} TOW entries generated "
            f"in {self.processing_stats.processing_time:.1f}s"
        )
        
        return self.processing_stats
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        text_column: str = "text",
        **kwargs
    ) -> ProcessingStats:
        """
        Process a file to generate TOW training data.
        
        Args:
            input_path: Path to input file (JSON lines or text)
            output_path: Path to save TOW dataset
            text_column: Column name for text data (for JSON)
            **kwargs: Additional arguments for process_corpus
            
        Returns:
            ProcessingStats with processing results
        """
        logger.info(f"Processing file: {input_path}")
        
        # Load data based on file extension
        input_data = self._load_input_file(input_path, text_column)
        
        if not input_data:
            logger.error(f"No data loaded from {input_path}")
            return ProcessingStats()
        
        return self.process_corpus(input_data, output_path, **kwargs)
    
    def _process_batch(
        self, 
        batch_data: List[str], 
        max_workers: int
    ) -> List[TOWEntry]:
        """Process a batch of texts with multiple workers"""
        
        if max_workers == 1:
            # Single-threaded processing
            results = []
            for text in batch_data:
                entries = self._process_single_text(text)
                results.extend(entries)
            return results
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._process_single_text, text) for text in batch_data]
                
                results = []
                for future in futures:
                    try:
                        entries = future.result(timeout=self.config.processing_timeout)
                        results.extend(entries)
                    except Exception as e:
                        logger.warning(f"Failed to process text: {e}")
                        continue
                
                return results
    
    def _process_single_text(self, text: str) -> List[TOWEntry]:
        """
        Process a single text to generate TOW entries.
        
        Args:
            text: Input text to process
            
        Returns:
            List of TOWEntry objects
        """
        if not text or len(text.strip()) < self.config.min_text_length:
            return []
        
        entries = []
        
        try:
            # Split text into sentences for processing
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if len(sentence.strip()) < self.config.min_sentence_length:
                    continue
                
                # Generate TOW entries for this sentence
                sentence_entries = self._process_sentence(sentence)
                entries.extend(sentence_entries)
                
                # Limit entries per text to avoid overwhelming
                if len(entries) >= self.config.max_entries_per_text:
                    break
        
        except Exception as e:
            logger.warning(f"Failed to process text: {e}")
            return []
        
        return entries
    
    def _process_sentence(self, sentence: str) -> List[TOWEntry]:
        """
        Process a single sentence to generate TOW entries.
        
        This is where the core TOW generation happens:
        1. Split sentence to get context and target word
        2. Generate prediction using model
        3. Classify the prediction
        4. Generate English thought token
        5. Create TOW entry
        """
        entries = []
        
        # Tokenize sentence (simple word-based tokenization)
        words = self._tokenize_sentence(sentence)
        
        if len(words) < 3:  # Need at least some context
            return []
        
        # Generate TOW entries for different positions
        num_positions = min(len(words) - 1, self.config.max_positions_per_sentence)
        positions = self._select_positions(words, num_positions)
        
        for pos in positions:
            try:
                entry = self._generate_tow_entry(words, pos)
                if entry:
                    entries.append(entry)
            except Exception as e:
                logger.debug(f"Failed to generate TOW entry at position {pos}: {e}")
                continue
        
        return entries
    
    def _generate_tow_entry(self, words: List[str], position: int) -> Optional[TOWEntry]:
        """
        Generate a single TOW entry for a specific position.
        
        Args:
            words: List of words in sentence
            position: Position of target word
            
        Returns:
            TOWEntry if successful, None otherwise
        """
        if position >= len(words) or position < 1:
            return None
        
        # Extract context and target
        context_words = words[:position]
        target_word = words[position]
        context_text = " ".join(context_words)
        
        # Generate prediction using model
        predicted_word = self._predict_next_word(context_text)
        
        if not predicted_word:
            return None
        
        # Classify the prediction
        classification_context = ClassificationContext(
            preceding_text=context_text,
            predicted_token=predicted_word,
            actual_token=target_word,
            domain=self._detect_domain(context_text),
            language=self.language
        )
        
        classification_result = self.token_classifier.classify_token(classification_context)
        
        # Generate English thought token
        thought_token = self._generate_english_thought(
            context_text, predicted_word, target_word, classification_result
        )
        
        if not thought_token:
            return None
        
        # Create TOW entry
        return TOWEntry(
            original_text=" ".join(words),
            context=context_text,
            predicted_token=predicted_word,
            actual_token=target_word,
            thought_token=thought_token,
            category=classification_result.category.value,
            confidence=classification_result.confidence,
            language=self.language,
            domain=classification_context.domain,
            metadata={
                "position": position,
                "context_length": len(context_words),
                "reasoning": classification_result.reasoning,
                "similarity_score": classification_result.similarity_score,
                "frequency_score": classification_result.frequency_score
            }
        )
    
    def _predict_next_word(self, context: str) -> Optional[str]:
        """
        Use model to predict next word given context.
        
        Args:
            context: Context text for prediction
            
        Returns:
            Predicted word or None if prediction fails
        """
        try:
            # Build prediction prompt
            prompt = f"다음 문장을 완성하세요: {context}"
            
            # Generate with model
            response = self.model_adapter.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3,  # Lower temperature for more consistent predictions
                top_p=0.8
            )
            
            if not response:
                return None
            
            # Extract first word from response
            predicted_words = response.strip().split()
            if predicted_words:
                return predicted_words[0]
            
            return None
            
        except Exception as e:
            logger.debug(f"Prediction failed for context '{context[:50]}...': {e}")
            return None
    
    def _generate_english_thought(
        self,
        context: str,
        predicted: str,
        actual: str,
        classification: Any
    ) -> str:
        """
        Generate English thought token for TOW format.
        
        This is the key function that ensures TOW tokens are ALWAYS in English,
        regardless of the source language.
        
        Args:
            context: Context text
            predicted: Predicted word
            actual: Actual word
            classification: Classification result
            
        Returns:
            English thought token in <ToW> format
        """
        try:
            # Select template based on category
            template = self._select_thought_template(classification.category)
            
            # Generate thought based on category
            if classification.category == TOWCategory.EXACT_MATCH:
                thought = f"The context clearly indicates the next word should be '{actual}', which matches the prediction exactly."
                
            elif classification.category == TOWCategory.SOFT_CONSISTENT:
                thought = f"The context suggests '{predicted}' which is contextually similar to the actual word '{actual}' (similarity: {classification.similarity_score:.2f})."
                
            elif classification.category == TOWCategory.TRIVIAL:
                thought = f"The word '{actual}' is a common function word that frequently appears in this context."
                
            elif classification.category == TOWCategory.UNPREDICTABLE:
                thought = "This word is difficult to predict from the given context alone."
            
            else:
                thought = f"The context leads to '{actual}' through contextual reasoning."
            
            # Enhance with context-specific reasoning
            enhanced_thought = self._enhance_thought_with_context(thought, context, actual)

            # Enforce English-only and format as TOW token
            enhanced_thought = enforce_english_text(enhanced_thought)
            return sanitize_tow_token(f"<ToW>{enhanced_thought}</ToW>")
            
        except Exception as e:
            logger.debug(f"Failed to generate thought token: {e}")
            # Fallback simple thought
            return sanitize_tow_token(f"<ToW>The context suggests the word '{actual}'.</ToW>")
    
    def _enhance_thought_with_context(self, base_thought: str, context: str, actual: str) -> str:
        """Enhance thought with context-specific information"""
        
        enhancements = []
        
        # Add domain context if detected
        domain = self._detect_domain(context)
        if domain != "general":
            enhancements.append(f"In {domain} context")
        
        # Add linguistic patterns
        if self.language == "ko":
            # Korean-specific patterns
            if actual.endswith(("다", "요", "니다")):
                enhancements.append("following Korean verb ending pattern")
            elif actual in ["은", "는", "이", "가", "을", "를"]:
                enhancements.append("as a Korean grammatical particle")
        
        # Combine enhancements
        if enhancements:
            enhanced = f"{base_thought} {' and '.join(enhancements)}."
        else:
            enhanced = base_thought
        
        # Ensure reasonable length
        if len(enhanced) > 200:
            enhanced = enhanced[:197] + "..."
        
        return enhanced
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain/topic of text"""
        text_lower = text.lower()
        
        # Programming keywords
        if any(word in text_lower for word in ["코드", "프로그램", "함수", "변수", "알고리즘", "code", "function"]):
            return "programming"
        
        # Mathematics keywords
        if any(word in text_lower for word in ["수학", "방정식", "계산", "공식", "math", "equation"]):
            return "mathematics"
        
        # Science keywords
        if any(word in text_lower for word in ["실험", "연구", "이론", "실험", "experiment", "research"]):
            return "science"
        
        # Business keywords
        if any(word in text_lower for word in ["비즈니스", "회사", "사업", "business", "company"]):
            return "business"
        
        return "general"
    
    def _select_thought_template(self, category: TOWCategory) -> str:
        """Select appropriate thought template for category"""
        templates = self.thought_templates.get(category.value, [])
        if templates:
            return random.choice(templates)
        return "The context suggests this word based on linguistic patterns."
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting for Korean and English
        sentences = re.split(r'[.!?]+\s+', text.strip())
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) >= self.config.min_sentence_length]
        
        return sentences
    
    def _tokenize_sentence(self, sentence: str) -> List[str]:
        """Simple word tokenization"""
        import re
        
        # Remove punctuation and split on whitespace
        cleaned = re.sub(r'[^\w\s]', ' ', sentence)
        words = cleaned.split()
        
        # Filter empty words
        return [w.strip() for w in words if w.strip()]
    
    def _select_positions(self, words: List[str], num_positions: int) -> List[int]:
        """Select positions for TOW generation"""
        if len(words) <= 2:
            return []
        
        # Available positions (not first word, not last word for complete context)
        available = list(range(1, len(words)))
        
        if len(available) <= num_positions:
            return available
        
        # Prefer positions that are not too early or too late
        middle_start = max(1, len(words) // 4)
        middle_end = min(len(words) - 1, 3 * len(words) // 4)
        
        middle_positions = [p for p in available if middle_start <= p <= middle_end]
        
        if len(middle_positions) >= num_positions:
            return random.sample(middle_positions, num_positions)
        else:
            # Take all middle positions plus some random ones
            remaining = num_positions - len(middle_positions)
            other_positions = [p for p in available if p not in middle_positions]
            if other_positions:
                additional = random.sample(other_positions, min(remaining, len(other_positions)))
                return middle_positions + additional
            else:
                return middle_positions
    
    def _load_input_file(self, filepath: str, text_column: str) -> List[str]:
        """Load input data from file"""
        texts = []
        
        try:
            path = Path(filepath)
            
            if path.suffix == '.txt':
                # Plain text file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts = [content]
            
            elif path.suffix in ['.jsonl', '.json']:
                # JSON lines or JSON file
                with open(filepath, 'r', encoding='utf-8') as f:
                    if path.suffix == '.jsonl':
                        for line in f:
                            data = json.loads(line.strip())
                            if text_column in data and data[text_column]:
                                texts.append(data[text_column])
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and text_column in item:
                                    texts.append(item[text_column])
                                elif isinstance(item, str):
                                    texts.append(item)
                        elif isinstance(data, dict) and text_column in data:
                            texts.append(data[text_column])
            
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return []
        
        except Exception as e:
            logger.error(f"Failed to load input file {filepath}: {e}")
            return []
        
        logger.info(f"Loaded {len(texts)} texts from {filepath}")
        return texts
    
    def _save_tow_dataset(self, entries: List[TOWEntry], output_path: str):
        """Save TOW dataset to file"""
        try:
            # Convert entries to dictionaries
            data = [asdict(entry) for entry in entries]
            
            # Save as JSON lines
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry_dict in data:
                    f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(entries)} TOW entries to {output_path}")
            
            # Also save a summary
            summary_path = output_path.replace('.jsonl', '_summary.json')
            summary = {
                "total_entries": len(entries),
                "categories": {},
                "languages": {},
                "domains": {},
                "average_confidence": sum(e.confidence for e in entries) / len(entries) if entries else 0
            }
            
            # Count categories, languages, domains
            for entry in entries:
                summary["categories"][entry.category] = summary["categories"].get(entry.category, 0) + 1
                summary["languages"][entry.language] = summary["languages"].get(entry.language, 0) + 1  
                summary["domains"][entry.domain] = summary["domains"].get(entry.domain, 0) + 1
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved dataset summary to {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save TOW dataset to {output_path}: {e}")
    
    def _load_thought_templates(self) -> Dict[str, List[str]]:
        """Load thought generation templates"""
        return {
            "exact_match": [
                "The context clearly indicates the next word should be '{word}'.",
                "Based on the preceding text, '{word}' is the logical continuation.",
                "The linguistic pattern strongly suggests '{word}' as the next word."
            ],
            "soft_consistent": [
                "The context suggests a word similar to '{word}' in meaning.",
                "Given the context, '{word}' is contextually appropriate.",
                "The text pattern indicates '{word}' or a semantically similar word."
            ],
            "trivial": [
                "This is a common function word that appears frequently.",
                "'{word}' is a typical grammatical element in this context.",
                "This word serves a structural role in the sentence."
            ],
            "unpredictable": [
                "This word is difficult to predict from the given context alone.",
                "The next word requires additional context or domain knowledge.",
                "'{word}' represents an unexpected continuation of the text."
            ]
        }
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics"""
        return self.processing_stats
