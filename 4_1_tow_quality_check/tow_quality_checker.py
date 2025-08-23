"""
ToW (Thought of Words) Dataset Quality Checker

This module provides quality assessment and filtering for ToW datasets
based on the methodology described in the ToW paper.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for ToW dataset evaluation"""
    total_samples: int = 0
    valid_tow_format: int = 0
    meaningful_explanations: int = 0
    appropriate_length: int = 0
    logical_coherence: int = 0
    contextual_relevance: int = 0
    linguistic_quality: int = 0
    
    def get_scores(self) -> Dict[str, float]:
        """Calculate quality scores as percentages"""
        if self.total_samples == 0:
            return {}
        
        return {
            'valid_tow_format': (self.valid_tow_format / self.total_samples) * 100,
            'meaningful_explanations': (self.meaningful_explanations / self.total_samples) * 100,
            'appropriate_length': (self.appropriate_length / self.total_samples) * 100,
            'logical_coherence': (self.logical_coherence / self.total_samples) * 100,
            'contextual_relevance': (self.contextual_relevance / self.total_samples) * 100,
            'linguistic_quality': (self.linguistic_quality / self.total_samples) * 100
        }

class ToWQualityChecker:
    """Quality checker for ToW datasets"""
    
    def __init__(self, min_tow_length: int = 50, max_tow_length: int = 500):
        """
        Initialize quality checker
        
        Args:
            min_tow_length: Minimum acceptable ToW explanation length
            max_tow_length: Maximum acceptable ToW explanation length
        """
        self.min_tow_length = min_tow_length
        self.max_tow_length = max_tow_length
        
        # Quality assessment patterns
        self.tow_pattern = re.compile(r'<ToW>(.*?)</ToW>', re.DOTALL)
        self.meaningless_patterns = [
            r'^The word.*is.*because.*\.$',  # Too formulaic
            r'^.*is the most logical.*$',    # Repetitive phrases
            r'^This is.*$',                  # Too simple
        ]
        
        # Keywords indicating good explanations
        self.quality_keywords = [
            'context', 'establishes', 'indicates', 'suggests', 'implies',
            'logical', 'coherent', 'relevant', 'appropriate', 'necessary',
            'maintains', 'progression', 'flow', 'continuation', 'expectation'
        ]
    
    def check_tow_format(self, tow_text: str) -> bool:
        """Check if ToW text has proper format"""
        if not tow_text or not isinstance(tow_text, str):
            return False
        
        match = self.tow_pattern.search(tow_text)
        return match is not None and len(match.group(1).strip()) > 0
    
    def check_length_appropriateness(self, tow_text: str) -> bool:
        """Check if ToW explanation has appropriate length"""
        if not self.check_tow_format(tow_text):
            return False
        
        match = self.tow_pattern.search(tow_text)
        if not match:
            return False
        
        content_length = len(match.group(1).strip())
        return self.min_tow_length <= content_length <= self.max_tow_length
    
    def check_meaningful_explanation(self, tow_text: str) -> bool:
        """Check if explanation is meaningful and not formulaic"""
        if not self.check_tow_format(tow_text):
            return False
        
        match = self.tow_pattern.search(tow_text)
        if not match:
            return False
        
        content = match.group(1).strip()
        
        # Check against meaningless patterns
        for pattern in self.meaningless_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        
        # Check for quality keywords
        quality_score = sum(1 for keyword in self.quality_keywords 
                          if keyword.lower() in content.lower())
        
        return quality_score >= 2  # At least 2 quality indicators
    
    def check_logical_coherence(self, context: str, gold_label: str, tow_text: str) -> bool:
        """Check logical coherence between context, label, and explanation"""
        if not all([context, gold_label, tow_text]):
            return False
        
        if not self.check_tow_format(tow_text):
            return False
        
        match = self.tow_pattern.search(tow_text)
        if not match:
            return False
        
        explanation = match.group(1).strip().lower()
        
        # Check if explanation mentions the context and target word
        mentions_context = any(word in explanation for word in context.lower().split()[-5:])
        mentions_target = gold_label.lower() in explanation
        
        # Check for logical connectives
        logical_connectives = ['because', 'since', 'as', 'therefore', 'thus', 'consequently']
        has_logical_structure = any(conn in explanation for conn in logical_connectives)
        
        return mentions_context and mentions_target and has_logical_structure
    
    def check_contextual_relevance(self, context: str, gold_label: str, tow_text: str) -> bool:
        """Check if explanation is contextually relevant"""
        if not all([context, gold_label, tow_text]):
            return False
        
        if not self.check_tow_format(tow_text):
            return False
        
        match = self.tow_pattern.search(tow_text)
        if not match:
            return False
        
        explanation = match.group(1).strip().lower()
        context_words = set(context.lower().split())
        
        # Count contextually relevant words in explanation
        relevant_words = sum(1 for word in explanation.split() if word in context_words)
        explanation_words = len(explanation.split())
        
        # At least 10% of explanation should be contextually relevant
        relevance_ratio = relevant_words / explanation_words if explanation_words > 0 else 0
        
        return relevance_ratio >= 0.1
    
    def check_linguistic_quality(self, tow_text: str) -> bool:
        """Check linguistic quality of the explanation"""
        if not self.check_tow_format(tow_text):
            return False
        
        match = self.tow_pattern.search(tow_text)
        if not match:
            return False
        
        content = match.group(1).strip()
        
        # Basic linguistic quality checks
        sentences = content.split('.')
        
        # Check for proper sentence structure
        has_proper_sentences = len(sentences) >= 2 and all(len(s.strip()) > 5 for s in sentences[:-1])
        
        # Check for repetition (avoid excessive repetition of words)
        words = content.lower().split()
        word_counts = Counter(words)
        max_repetition = max(word_counts.values()) if word_counts else 0
        reasonable_repetition = max_repetition <= 3
        
        # Check for diverse vocabulary
        unique_words = len(set(words))
        total_words = len(words)
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        return has_proper_sentences and reasonable_repetition and vocabulary_diversity >= 0.7
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
        """
        Evaluate a single sample for quality
        
        Returns:
            Tuple of (overall_quality, detailed_scores)
        """
        context = sample.get('context', '')
        gold_label = sample.get('gold_label', '')
        tow_text = sample.get('tow', '')
        
        scores = {
            'valid_tow_format': self.check_tow_format(tow_text),
            'meaningful_explanations': self.check_meaningful_explanation(tow_text),
            'appropriate_length': self.check_length_appropriateness(tow_text),
            'logical_coherence': self.check_logical_coherence(context, gold_label, tow_text),
            'contextual_relevance': self.check_contextual_relevance(context, gold_label, tow_text),
            'linguistic_quality': self.check_linguistic_quality(tow_text)
        }
        
        # Overall quality: all criteria must pass
        overall_quality = all(scores.values())
        
        return overall_quality, scores
    
    def evaluate_dataset(self, data: List[Dict[str, Any]]) -> QualityMetrics:
        """Evaluate entire dataset and return quality metrics"""
        metrics = QualityMetrics()
        metrics.total_samples = len(data)
        
        for sample in data:
            overall_quality, scores = self.evaluate_sample(sample)
            
            # Update metrics
            if scores['valid_tow_format']:
                metrics.valid_tow_format += 1
            if scores['meaningful_explanations']:
                metrics.meaningful_explanations += 1
            if scores['appropriate_length']:
                metrics.appropriate_length += 1
            if scores['logical_coherence']:
                metrics.logical_coherence += 1
            if scores['contextual_relevance']:
                metrics.contextual_relevance += 1
            if scores['linguistic_quality']:
                metrics.linguistic_quality += 1
        
        return metrics
    
    def filter_high_quality_samples(self, data: List[Dict[str, Any]], 
                                  min_criteria_met: int = 5) -> List[Dict[str, Any]]:
        """
        Filter dataset to keep only high-quality samples
        
        Args:
            data: List of samples
            min_criteria_met: Minimum number of quality criteria that must be met
        
        Returns:
            Filtered list of high-quality samples
        """
        high_quality_samples = []
        
        for sample in data:
            overall_quality, scores = self.evaluate_sample(sample)
            criteria_met = sum(scores.values())
            
            if criteria_met >= min_criteria_met:
                # Add quality score to sample
                sample['quality_score'] = criteria_met
                sample['quality_details'] = scores
                high_quality_samples.append(sample)
        
        logger.info(f"Filtered {len(high_quality_samples)} high-quality samples "
                   f"from {len(data)} total samples")
        
        return high_quality_samples
    
    def generate_quality_report(self, data: List[Dict[str, Any]], 
                              output_path: str = None) -> str:
        """Generate detailed quality assessment report"""
        metrics = self.evaluate_dataset(data)
        scores = metrics.get_scores()
        
        report = f"""
ToW Dataset Quality Assessment Report
=====================================

Dataset Overview:
- Total samples: {metrics.total_samples}

Quality Metrics:
- Valid ToW format: {scores.get('valid_tow_format', 0):.1f}% ({metrics.valid_tow_format}/{metrics.total_samples})
- Meaningful explanations: {scores.get('meaningful_explanations', 0):.1f}% ({metrics.meaningful_explanations}/{metrics.total_samples})
- Appropriate length: {scores.get('appropriate_length', 0):.1f}% ({metrics.appropriate_length}/{metrics.total_samples})
- Logical coherence: {scores.get('logical_coherence', 0):.1f}% ({metrics.logical_coherence}/{metrics.total_samples})
- Contextual relevance: {scores.get('contextual_relevance', 0):.1f}% ({metrics.contextual_relevance}/{metrics.total_samples})
- Linguistic quality: {scores.get('linguistic_quality', 0):.1f}% ({metrics.linguistic_quality}/{metrics.total_samples})

Overall Assessment:
- Average quality score: {sum(scores.values()) / len(scores):.1f}%

Recommendations:
"""
        
        # Add recommendations based on scores
        if scores.get('valid_tow_format', 0) < 95:
            report += "- Fix ToW formatting issues\n"
        if scores.get('meaningful_explanations', 0) < 80:
            report += "- Improve explanation meaningfulness\n"
        if scores.get('logical_coherence', 0) < 85:
            report += "- Enhance logical coherence in explanations\n"
        if scores.get('contextual_relevance', 0) < 90:
            report += "- Strengthen contextual relevance\n"
        if scores.get('linguistic_quality', 0) < 85:
            report += "- Improve linguistic quality\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Quality report saved to {output_path}")
        
        return report

def main():
    """Example usage of ToW quality checker"""
    # Initialize quality checker
    checker = ToWQualityChecker()
    
    # Data directory
    data_dir = Path("../4_tow_generation/tow_data")
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)
    
    # Process all JSON files in the directory
    for json_file in data_dir.glob("*.json"):
        logger.info(f"Processing {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Generate quality report
            report = checker.generate_quality_report(
                data, 
                output_path=output_dir / f"{json_file.stem}_quality_report.txt"
            )
            
            # Filter high-quality samples
            high_quality_data = checker.filter_high_quality_samples(data, min_criteria_met=5)
            
            # Save filtered data
            output_file = output_dir / f"{json_file.stem}_filtered.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(high_quality_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Filtered data saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {str(e)}")

if __name__ == "__main__":
    main()