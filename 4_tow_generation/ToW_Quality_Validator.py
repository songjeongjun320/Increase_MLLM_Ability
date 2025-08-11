#!/usr/bin/env python3
"""
ToW Dataset Quality Validation System
Comprehensive validation for augmented Korean ToW datasets
"""

import json
import re
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import logging
from pathlib import Path
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToWQualityValidator:
    """Comprehensive quality validator for ToW-augmented Korean datasets"""
    
    def __init__(self):
        self.validation_results = {}
        self.quality_metrics = {}
        
    def load_dataset(self, file_path: str) -> List[Dict]:
        """Load and return dataset"""
        logger.info(f"Loading dataset from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries")
        return data
    
    def extract_tow_tokens(self, text: str) -> List[str]:
        """Extract ToW token contents"""
        pattern = r'<ToW>(.*?)</ToW>'
        return re.findall(pattern, text, re.DOTALL)
    
    def validate_tow_token_integrity(self, dataset: List[Dict]) -> Dict:
        """Validate ToW token preservation and integrity"""
        logger.info("Validating ToW token integrity...")
        
        results = {
            'total_entries': len(dataset),
            'entries_with_tow': 0,
            'tow_token_preservation': 0,
            'semantic_coherence_issues': [],
            'malformed_tokens': [],
            'difficulty_marker_consistency': 0
        }
        
        for entry in tqdm(dataset, desc="Checking ToW tokens"):
            text = entry.get('augmented_text', '')
            tow_tokens = self.extract_tow_tokens(text)
            
            if tow_tokens:
                results['entries_with_tow'] += 1
                
                # Check token count consistency
                declared_count = entry.get('tow_count', 0)
                actual_count = len(tow_tokens)
                
                if declared_count == actual_count:
                    results['tow_token_preservation'] += 1
                
                # Check for malformed tokens
                for token in tow_tokens:
                    if len(token.strip()) < 20:  # Too short to be meaningful
                        results['malformed_tokens'].append({
                            'doc_id': entry.get('doc_id'),
                            'token': token[:100] + '...' if len(token) > 100 else token
                        })
                
                # Check semantic coherence (basic heuristics)
                for token in tow_tokens:
                    if not any(keyword in token.lower() for keyword in 
                              ['korean', 'korea', 'linguistic', 'context', 'understanding', 
                               'cultural', 'social', 'grammatical', 'honorific']):
                        results['semantic_coherence_issues'].append({
                            'doc_id': entry.get('doc_id'),
                            'issue': 'Missing Korean linguistic context'
                        })
                
                # Check difficulty marker consistency
                markers = entry.get('difficulty_markers', [])
                if markers:
                    # Check if ToW tokens mention relevant difficulty aspects
                    token_text = ' '.join(tow_tokens).lower()
                    marker_keywords = {
                        'honorific_system': ['honorific', 'respect', 'formal'],
                        'cultural_reference': ['cultural', 'tradition', 'historical'],
                        'religious_historical': ['religious', 'historical', 'context'],
                        'linguistic_nuance': ['linguistic', 'nuance', 'grammar']
                    }
                    
                    for marker in markers:
                        if marker in marker_keywords:
                            keywords = marker_keywords[marker]
                            if any(keyword in token_text for keyword in keywords):
                                results['difficulty_marker_consistency'] += 1
                                break
        
        # Calculate percentages
        if results['entries_with_tow'] > 0:
            results['preservation_rate'] = (results['tow_token_preservation'] / results['entries_with_tow']) * 100
            results['coherence_rate'] = ((results['entries_with_tow'] - len(results['semantic_coherence_issues'])) / results['entries_with_tow']) * 100
            results['consistency_rate'] = (results['difficulty_marker_consistency'] / results['entries_with_tow']) * 100
        
        return results
    
    def validate_korean_linguistic_quality(self, dataset: List[Dict]) -> Dict:
        """Validate Korean linguistic accuracy"""
        logger.info("Validating Korean linguistic quality...")
        
        results = {
            'total_augmented': 0,
            'honorific_consistency': 0,
            'particle_validity': 0,
            'grammatical_errors': [],
            'unnatural_constructions': []
        }
        
        # Korean linguistic patterns for validation
        honorific_patterns = {
            '습니다': 'formal',
            '어요': 'informal',
            '해요': 'informal',
            '입니다': 'formal'
        }
        
        particle_patterns = [
            '은', '는', '이', '가', '을', '를', '에', '에서', '와', '과', '한테', '에게'
        ]
        
        for entry in tqdm(dataset, desc="Checking linguistic quality"):
            if entry.get('augmentation_type'):
                results['total_augmented'] += 1
                text = entry.get('augmented_text', '')
                
                # Check honorific consistency
                honorific_levels = []
                for pattern, level in honorific_patterns.items():
                    if pattern in text:
                        honorific_levels.append(level)
                
                # Count consistent honorific usage
                if len(set(honorific_levels)) <= 1:  # All same level or none
                    results['honorific_consistency'] += 1
                
                # Check particle usage
                particle_count = sum(1 for particle in particle_patterns if particle in text)
                if particle_count > 0:
                    results['particle_validity'] += 1
                
                # Check for obvious grammatical errors (basic heuristics)
                if '은는' in text or '이가' in text or '을를' in text:
                    results['grammatical_errors'].append({
                        'doc_id': entry.get('doc_id'),
                        'error': 'Duplicate particles'
                    })
        
        # Calculate rates
        if results['total_augmented'] > 0:
            results['honorific_consistency_rate'] = (results['honorific_consistency'] / results['total_augmented']) * 100
            results['particle_validity_rate'] = (results['particle_validity'] / results['total_augmented']) * 100
        
        return results
    
    def analyze_augmentation_distribution(self, dataset: List[Dict]) -> Dict:
        """Analyze distribution of augmentation types"""
        logger.info("Analyzing augmentation distribution...")
        
        augmentation_stats = defaultdict(int)
        original_entries = 0
        difficulty_distribution = defaultdict(lambda: defaultdict(int))
        
        for entry in dataset:
            aug_type = entry.get('augmentation_type')
            if aug_type:
                augmentation_stats[aug_type] += 1
                
                # Analyze by difficulty markers
                markers = entry.get('difficulty_markers', ['unknown'])
                for marker in markers:
                    difficulty_distribution[aug_type][marker] += 1
            else:
                original_entries += 1
        
        return {
            'original_entries': original_entries,
            'augmentation_stats': dict(augmentation_stats),
            'difficulty_distribution': dict(difficulty_distribution),
            'total_augmented': sum(augmentation_stats.values()),
            'augmentation_ratio': sum(augmentation_stats.values()) / max(original_entries, 1)
        }
    
    def check_data_diversity(self, dataset: List[Dict]) -> Dict:
        """Check diversity of augmented data"""
        logger.info("Checking data diversity...")
        
        # Collect text lengths
        text_lengths = []
        tow_counts = []
        unique_texts = set()
        
        augmented_entries = [e for e in dataset if e.get('augmentation_type')]
        
        for entry in augmented_entries:
            text = entry.get('augmented_text', '')
            text_lengths.append(len(text))
            tow_counts.append(entry.get('tow_count', 0))
            unique_texts.add(text[:200])  # First 200 chars for uniqueness check
        
        diversity_score = len(unique_texts) / max(len(augmented_entries), 1) * 100
        
        return {
            'total_augmented_entries': len(augmented_entries),
            'unique_text_variants': len(unique_texts),
            'diversity_score': diversity_score,
            'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
            'text_length_std': np.std(text_lengths) if text_lengths else 0,
            'avg_tow_count': np.mean(tow_counts) if tow_counts else 0,
            'tow_count_distribution': Counter(tow_counts)
        }
    
    def generate_validation_report(self, dataset: List[Dict]) -> Dict:
        """Generate comprehensive validation report"""
        logger.info("Generating comprehensive validation report...")
        
        # Run all validation checks
        tow_integrity = self.validate_tow_token_integrity(dataset)
        linguistic_quality = self.validate_korean_linguistic_quality(dataset)
        augmentation_dist = self.analyze_augmentation_distribution(dataset)
        diversity_metrics = self.check_data_diversity(dataset)
        
        # Calculate overall quality score
        quality_components = [
            tow_integrity.get('preservation_rate', 0) * 0.3,
            tow_integrity.get('coherence_rate', 0) * 0.25,
            linguistic_quality.get('honorific_consistency_rate', 0) * 0.2,
            linguistic_quality.get('particle_validity_rate', 0) * 0.15,
            min(diversity_metrics.get('diversity_score', 0), 100) * 0.1
        ]
        
        overall_quality_score = sum(quality_components)
        
        report = {
            'dataset_summary': {
                'total_entries': len(dataset),
                'original_entries': augmentation_dist['original_entries'],
                'augmented_entries': augmentation_dist['total_augmented'],
                'augmentation_ratio': f"{augmentation_dist['augmentation_ratio']:.2f}x"
            },
            'tow_token_integrity': tow_integrity,
            'linguistic_quality': linguistic_quality,
            'augmentation_distribution': augmentation_dist,
            'diversity_metrics': diversity_metrics,
            'overall_quality_score': round(overall_quality_score, 2),
            'quality_grade': self.get_quality_grade(overall_quality_score),
            'recommendations': self.generate_recommendations(
                tow_integrity, linguistic_quality, diversity_metrics
            )
        }
        
        return report
    
    def get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 85:
            return "A (Very Good)"
        elif score >= 80:
            return "B+ (Good)"
        elif score >= 75:
            return "B (Acceptable)"
        elif score >= 70:
            return "C+ (Needs Improvement)"
        else:
            return "C (Poor Quality)"
    
    def generate_recommendations(self, tow_integrity: Dict, linguistic_quality: Dict, diversity_metrics: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # ToW token recommendations
        if tow_integrity.get('preservation_rate', 0) < 95:
            recommendations.append("Improve ToW token preservation during augmentation")
        
        if tow_integrity.get('coherence_rate', 0) < 90:
            recommendations.append("Enhance semantic coherence of ToW reasoning tokens")
        
        # Linguistic quality recommendations
        if linguistic_quality.get('honorific_consistency_rate', 0) < 85:
            recommendations.append("Improve honorific level consistency in augmented texts")
        
        if len(linguistic_quality.get('grammatical_errors', [])) > 0:
            recommendations.append("Address grammatical errors in particle usage")
        
        # Diversity recommendations
        if diversity_metrics.get('diversity_score', 0) < 80:
            recommendations.append("Increase diversity of augmented variations")
        
        if not recommendations:
            recommendations.append("Quality is excellent! Consider expanding to additional augmentation techniques.")
        
        return recommendations
    
    def save_validation_report(self, report: Dict, output_file: str):
        """Save validation report to file"""
        logger.info(f"Saving validation report to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Also save a human-readable summary
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ToW Dataset Quality Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset summary
            f.write("📊 DATASET SUMMARY\n")
            f.write("-" * 30 + "\n")
            summary = report['dataset_summary']
            f.write(f"Total entries: {summary['total_entries']:,}\n")
            f.write(f"Original entries: {summary['original_entries']:,}\n")
            f.write(f"Augmented entries: {summary['augmented_entries']:,}\n")
            f.write(f"Augmentation ratio: {summary['augmentation_ratio']}\n\n")
            
            # Overall quality
            f.write("🎯 OVERALL QUALITY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Quality Score: {report['overall_quality_score']}/100\n")
            f.write(f"Quality Grade: {report['quality_grade']}\n\n")
            
            # ToW token integrity
            f.write("🔍 ToW TOKEN INTEGRITY\n")
            f.write("-" * 30 + "\n")
            tow = report['tow_token_integrity']
            f.write(f"Entries with ToW tokens: {tow['entries_with_tow']:,}\n")
            f.write(f"Token preservation rate: {tow.get('preservation_rate', 0):.1f}%\n")
            f.write(f"Semantic coherence rate: {tow.get('coherence_rate', 0):.1f}%\n")
            f.write(f"Malformed tokens: {len(tow['malformed_tokens'])}\n\n")
            
            # Linguistic quality
            f.write("📝 LINGUISTIC QUALITY\n")
            f.write("-" * 30 + "\n")
            ling = report['linguistic_quality']
            f.write(f"Total augmented entries: {ling['total_augmented']:,}\n")
            f.write(f"Honorific consistency: {ling.get('honorific_consistency_rate', 0):.1f}%\n")
            f.write(f"Particle validity: {ling.get('particle_validity_rate', 0):.1f}%\n")
            f.write(f"Grammatical errors: {len(ling['grammatical_errors'])}\n\n")
            
            # Augmentation distribution
            f.write("📈 AUGMENTATION DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            aug_stats = report['augmentation_distribution']['augmentation_stats']
            for aug_type, count in aug_stats.items():
                f.write(f"{aug_type}: {count:,} variants\n")
            f.write("\n")
            
            # Diversity metrics
            f.write("🌈 DIVERSITY METRICS\n")
            f.write("-" * 30 + "\n")
            div = report['diversity_metrics']
            f.write(f"Unique text variants: {div['unique_text_variants']:,}\n")
            f.write(f"Diversity score: {div['diversity_score']:.1f}%\n")
            f.write(f"Average text length: {div['avg_text_length']:.0f} chars\n")
            f.write(f"Average ToW count: {div['avg_tow_count']:.1f}\n\n")
            
            # Recommendations
            f.write("💡 RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"✅ Validation report saved to {output_file}")
        logger.info(f"✅ Human-readable summary saved to {summary_file}")

def main():
    """Main validation function"""
    logger.info("🔍 ToW Dataset Quality Validation System")
    
    # File paths
    augmented_file = "ToW_koconovel_augmented.json"
    report_file = "ToW_quality_validation_report.json"
    
    # Check if augmented file exists
    if not Path(augmented_file).exists():
        logger.error(f"Augmented dataset not found: {augmented_file}")
        return
    
    # Initialize validator
    validator = ToWQualityValidator()
    
    try:
        # Load dataset
        dataset = validator.load_dataset(augmented_file)
        
        # Generate validation report
        report = validator.generate_validation_report(dataset)
        
        # Save report
        validator.save_validation_report(report, report_file)
        
        # Print summary
        logger.info("\n📊 VALIDATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Overall Quality Score: {report['overall_quality_score']}/100")
        logger.info(f"Quality Grade: {report['quality_grade']}")
        logger.info(f"Dataset Size: {len(dataset):,} total entries")
        logger.info(f"Augmentation Ratio: {report['dataset_summary']['augmentation_ratio']}")
        
        # Print key metrics
        tow_integrity = report['tow_token_integrity']
        logger.info(f"ToW Token Preservation: {tow_integrity.get('preservation_rate', 0):.1f}%")
        logger.info(f"Semantic Coherence: {tow_integrity.get('coherence_rate', 0):.1f}%")
        
        linguistic_quality = report['linguistic_quality']
        logger.info(f"Honorific Consistency: {linguistic_quality.get('honorific_consistency_rate', 0):.1f}%")
        
        diversity = report['diversity_metrics']
        logger.info(f"Text Diversity: {diversity['diversity_score']:.1f}%")
        
        logger.info(f"\n✅ Validation completed! Report saved as: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()