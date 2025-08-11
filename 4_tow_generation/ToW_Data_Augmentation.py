#!/usr/bin/env python3
"""
ToW-Aware Korean Data Augmentation System
Advanced data augmentation for Korean ToW datasets with preservation of reasoning tokens
"""

import json
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """Configuration for ToW-aware augmentation"""
    # Input/Output paths
    input_file: str = "ToW_koconovel_complete.json"
    output_file: str = "ToW_koconovel_augmented.json"
    
    # Augmentation ratios (how many variants per original)
    honorific_variations: int = 2      # 경어체 변형
    particle_substitutions: int = 2    # 조사 변형
    word_order_changes: int = 1        # 어순 변경
    synonym_replacements: int = 2      # 동의어 교체
    tense_variations: int = 1          # 시제 변형
    back_translation_aug: int = 1      # 역번역 증강
    
    # Quality thresholds
    semantic_similarity_threshold: float = 0.85
    preserve_tow_integrity: bool = True
    maintain_difficulty_markers: bool = True
    
    # Processing limits
    max_samples_per_category: int = 500  # 카테고리별 최대 샘플 수
    random_seed: int = 42

class ToWAugmentationEngine:
    """Main engine for ToW-aware Korean text augmentation"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Korean linguistic patterns for augmentation
        self.initialize_korean_patterns()
        
    def initialize_korean_patterns(self):
        """Initialize Korean language patterns for augmentation"""
        
        # Honorific transformations (formal <-> informal)
        self.honorific_patterns = {
            # Formal to informal endings
            '습니다': '어요',
            '습니까': '어요',
            '셨습니다': '셨어요', 
            '하십니다': '해요',
            '입니다': '예요',
            '였습니다': '였어요',
            '이십니다': '이에요',
            
            # Informal to formal (reverse mapping)
            '어요': '습니다',
            '해요': '합니다',
            '예요': '입니다',
            '이에요': '입니다',
            '셨어요': '셨습니다',
            '였어요': '였습니다'
        }
        
        # Particle substitutions (maintaining grammatical validity)
        self.particle_substitutions = {
            '은': '는',     # topic particles
            '는': '은',
            '이': '가',     # subject particles  
            '가': '이',
            '을': '를',     # object particles
            '를': '을',
            '에': '에서',   # location particles (context dependent)
            '에게': '한테', # to/for particles
            '한테': '에게',
            '와': '과',     # and/with particles
            '과': '와'
        }
        
        # Common Korean synonyms for replacement
        self.synonym_groups = {
            '사람': ['인간', '이', '자'],
            '여자': ['여성', '계집', '부인'],
            '남자': ['남성', '사내', '놈'],
            '집': ['가정', '댁', '가옥'],
            '학교': ['학당', '교육기관'],
            '선생': ['교사', '스승', '훈장'],
            '학생': ['제자', '생도'],
            '아이': ['어린이', '소아', '애'],
            '어머니': ['모친', '어미', '엄마'],
            '아버지': ['부친', '아비', '아빠'],
            '아름답다': ['곱다', '예쁘다', '아리따다'],
            '크다': ['대단하다', '웅대하다'],
            '작다': ['조그맣다', '소소하다'],
            '좋다': ['훌륭하다', '괜찮다', '멋지다'],
            '나쁘다': ['못되다', '악하다', '형편없다']
        }
        
        # Tense transformation patterns
        self.tense_patterns = {
            # Present to past
            '한다': '했다',
            '간다': '갔다', 
            '온다': '왔다',
            '본다': '봤다',
            '먹는다': '먹었다',
            '마신다': '마셨다',
            
            # Past to present (reverse)
            '했다': '한다',
            '갔다': '간다',
            '왔다': '온다', 
            '봤다': '본다',
            '먹었다': '먹는다',
            '마셨다': '마신다'
        }
        
        # Word order change patterns (Korean flexible word order)
        self.word_order_patterns = [
            # SOV -> OSV patterns
            (r'(\w+이?\s+)(\w+을?\s+)(\w+다)', r'\2\1\3'),
            # Topic fronting patterns
            (r'(\w+는?\s+)(.+?)(\s+\w+다)', r'\1\3 \2'),
        ]

    def extract_tow_tokens(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract ToW token positions and content"""
        tow_tokens = []
        pattern = r'<ToW>(.*?)</ToW>'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            start_pos = match.start()
            end_pos = match.end()
            content = match.group(1)
            tow_tokens.append((start_pos, end_pos, content))
            
        return tow_tokens
    
    def preserve_tow_tokens(self, original_text: str, modified_text: str, tow_tokens: List[Tuple[int, int, str]]) -> str:
        """Preserve ToW tokens in modified text"""
        if not tow_tokens:
            return modified_text
        
        # Simple preservation: re-insert ToW tokens at approximately same relative positions
        for start_pos, end_pos, content in reversed(tow_tokens):
            # Calculate relative position
            total_length = len(original_text)
            relative_pos = start_pos / total_length
            
            # Insert at relative position in modified text
            insert_pos = int(len(modified_text) * relative_pos)
            tow_token = f"<ToW>{content}</ToW>"
            
            # Insert the ToW token
            modified_text = modified_text[:insert_pos] + tow_token + modified_text[insert_pos:]
            
        return modified_text

    def apply_honorific_variation(self, text: str) -> List[str]:
        """Apply honorific level variations"""
        variants = []
        
        # Extract ToW tokens first
        tow_tokens = self.extract_tow_tokens(text)
        base_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
        
        for _ in range(self.config.honorific_variations):
            modified = base_text
            
            # Apply random honorific transformations
            available_patterns = list(self.honorific_patterns.items())
            random.shuffle(available_patterns)
            
            for original, replacement in available_patterns[:3]:  # Apply up to 3 changes
                if original in modified:
                    modified = modified.replace(original, replacement)
                    break
            
            # Preserve ToW tokens
            if self.config.preserve_tow_integrity:
                modified = self.preserve_tow_tokens(text, modified, tow_tokens)
            
            if modified != text:  # Only add if different
                variants.append(modified)
                
        return variants
    
    def apply_particle_substitution(self, text: str) -> List[str]:
        """Apply particle substitution variations"""
        variants = []
        tow_tokens = self.extract_tow_tokens(text)
        base_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
        
        for _ in range(self.config.particle_substitutions):
            modified = base_text
            
            # Apply random particle substitutions
            available_substitutions = list(self.particle_substitutions.items())
            random.shuffle(available_substitutions)
            
            for original, replacement in available_substitutions[:2]:  # Apply up to 2 changes
                if original in modified:
                    modified = modified.replace(original, replacement)
                    break
            
            if self.config.preserve_tow_integrity:
                modified = self.preserve_tow_tokens(text, modified, tow_tokens)
                
            if modified != text:
                variants.append(modified)
                
        return variants
    
    def apply_synonym_replacement(self, text: str) -> List[str]:
        """Apply synonym replacement with cultural context preservation"""
        variants = []
        tow_tokens = self.extract_tow_tokens(text)
        base_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
        
        for _ in range(self.config.synonym_replacements):
            modified = base_text
            
            # Find applicable synonyms
            applicable_synonyms = []
            for word, synonyms in self.synonym_groups.items():
                if word in modified:
                    applicable_synonyms.append((word, synonyms))
            
            # Apply random synonym replacement
            if applicable_synonyms:
                word, synonyms = random.choice(applicable_synonyms)
                replacement = random.choice(synonyms)
                modified = modified.replace(word, replacement, 1)  # Replace only first occurrence
                
                if self.config.preserve_tow_integrity:
                    modified = self.preserve_tow_tokens(text, modified, tow_tokens)
                    
                if modified != text:
                    variants.append(modified)
                    
        return variants
    
    def apply_word_order_variation(self, text: str) -> List[str]:
        """Apply word order variations (Korean flexible syntax)"""
        variants = []
        tow_tokens = self.extract_tow_tokens(text)
        base_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
        
        # Split into sentences for individual processing
        sentences = base_text.split('.')
        
        for _ in range(self.config.word_order_changes):
            modified_sentences = []
            
            for sentence in sentences:
                modified_sentence = sentence
                
                # Apply word order patterns
                for pattern, replacement in self.word_order_patterns:
                    if re.search(pattern, modified_sentence):
                        modified_sentence = re.sub(pattern, replacement, modified_sentence)
                        break
                
                modified_sentences.append(modified_sentence)
            
            modified = '.'.join(modified_sentences)
            
            if self.config.preserve_tow_integrity:
                modified = self.preserve_tow_tokens(text, modified, tow_tokens)
                
            if modified != text:
                variants.append(modified)
                
        return variants
    
    def apply_tense_variation(self, text: str) -> List[str]:
        """Apply tense variations"""
        variants = []
        tow_tokens = self.extract_tow_tokens(text)
        base_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
        
        for _ in range(self.config.tense_variations):
            modified = base_text
            
            # Apply random tense transformations
            available_tenses = list(self.tense_patterns.items())
            random.shuffle(available_tenses)
            
            for original, replacement in available_tenses[:2]:  # Apply up to 2 changes
                if original in modified:
                    modified = modified.replace(original, replacement)
                    break
            
            if self.config.preserve_tow_integrity:
                modified = self.preserve_tow_tokens(text, modified, tow_tokens)
                
            if modified != text:
                variants.append(modified)
                
        return variants
    
    def generate_back_translation_augmentation(self, text: str) -> List[str]:
        """Simulate back-translation augmentation (rule-based approach)"""
        variants = []
        tow_tokens = self.extract_tow_tokens(text)
        base_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
        
        # Simulate back-translation effects with Korean paraphrasing patterns
        paraphrase_patterns = [
            # Change word order slightly
            (r'(\w+)하고\s+(\w+)', r'\2하고 \1'),
            # Change connector words
            ('그리고', '또한'),
            ('하지만', '그러나'),
            ('때문에', '탓에'),
            ('같이', '처럼'),
            ('매우', '아주'),
            ('정말', '참으로'),
        ]
        
        for _ in range(self.config.back_translation_aug):
            modified = base_text
            
            # Apply paraphrasing patterns
            applied_changes = 0
            for pattern, replacement in paraphrase_patterns:
                if applied_changes >= 2:  # Limit changes
                    break
                    
                if isinstance(pattern, str) and pattern in modified:
                    modified = modified.replace(pattern, replacement)
                    applied_changes += 1
                elif hasattr(pattern, 'search') and pattern.search(modified):
                    modified = re.sub(pattern, replacement, modified)
                    applied_changes += 1
            
            if self.config.preserve_tow_integrity:
                modified = self.preserve_tow_tokens(text, modified, tow_tokens)
                
            if modified != text:
                variants.append(modified)
                
        return variants
    
    def create_augmentation_entry(self, original_entry: Dict, augmented_text: str, aug_type: str, variant_num: int) -> Dict:
        """Create new augmented entry maintaining ToW metadata"""
        new_entry = original_entry.copy()
        
        # Update identifiers
        new_entry['doc_id'] = f"{original_entry['doc_id']}_aug_{aug_type}_{variant_num}"
        new_entry['augmented_text'] = augmented_text
        
        # Recalculate ToW tokens
        tow_count = len(re.findall(r'<ToW>.*?</ToW>', augmented_text, re.DOTALL))
        new_entry['tow_count'] = tow_count
        
        # Extract ToW tokens for metadata
        tow_matches = re.findall(r'<ToW>(.*?)</ToW>', augmented_text, re.DOTALL)
        new_entry['tow_tokens'] = [f"<ToW>{match}</ToW>" for match in tow_matches]
        
        # Preserve other metadata
        if self.config.maintain_difficulty_markers:
            # Keep original difficulty markers as they should remain valid
            pass
        
        # Add augmentation metadata
        new_entry['augmentation_type'] = aug_type
        new_entry['original_doc_id'] = original_entry['doc_id']
        
        return new_entry
    
    def augment_dataset(self, input_file: str, output_file: str):
        """Main function to augment the entire dataset"""
        logger.info("Starting ToW-aware Korean data augmentation...")
        
        # Load original dataset
        logger.info(f"Loading dataset from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        logger.info(f"Loaded {len(original_data)} original entries")
        
        # Filter entries with ToW tokens and sample by category
        tow_entries = [entry for entry in original_data if entry.get('tow_count', 0) > 0]
        logger.info(f"Found {len(tow_entries)} entries with ToW tokens")
        
        # Group by difficulty markers for balanced sampling
        entries_by_category = defaultdict(list)
        for entry in tow_entries:
            markers = entry.get('difficulty_markers', ['unknown'])
            category = markers[0] if markers else 'unknown'
            entries_by_category[category].append(entry)
        
        logger.info(f"Categories found: {list(entries_by_category.keys())}")
        
        # Sample entries for augmentation
        selected_entries = []
        for category, entries in entries_by_category.items():
            sample_size = min(len(entries), self.config.max_samples_per_category)
            sampled = random.sample(entries, sample_size)
            selected_entries.extend(sampled)
            logger.info(f"Sampled {len(sampled)} entries from {category} category")
        
        logger.info(f"Total selected for augmentation: {len(selected_entries)}")
        
        # Apply augmentations
        augmented_entries = []
        augmentation_stats = defaultdict(int)
        
        for entry in tqdm(selected_entries, desc="Augmenting entries"):
            original_text = entry['augmented_text']
            
            # Apply all augmentation techniques
            augmentation_methods = [
                ('honorific', self.apply_honorific_variation),
                ('particle', self.apply_particle_substitution),
                ('synonym', self.apply_synonym_replacement),
                ('word_order', self.apply_word_order_variation),
                ('tense', self.apply_tense_variation),
                ('back_trans', self.generate_back_translation_augmentation)
            ]
            
            for aug_type, aug_method in augmentation_methods:
                try:
                    variants = aug_method(original_text)
                    
                    for i, variant in enumerate(variants):
                        augmented_entry = self.create_augmentation_entry(
                            entry, variant, aug_type, i
                        )
                        augmented_entries.append(augmented_entry)
                        augmentation_stats[aug_type] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to apply {aug_type} augmentation: {e}")
        
        # Combine original and augmented data
        final_dataset = original_data + augmented_entries
        
        logger.info(f"Generated {len(augmented_entries)} augmented entries")
        logger.info(f"Final dataset size: {len(final_dataset)} entries")
        
        # Log augmentation statistics
        logger.info("Augmentation statistics:")
        for aug_type, count in augmentation_stats.items():
            logger.info(f"  {aug_type}: {count} variants generated")
        
        # Save augmented dataset
        logger.info(f"Saving augmented dataset to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ Data augmentation completed successfully!")
        
        return {
            'original_entries': len(original_data),
            'augmented_entries': len(augmented_entries),
            'total_entries': len(final_dataset),
            'augmentation_stats': dict(augmentation_stats)
        }

def main():
    """Main function to run data augmentation"""
    logger.info("🚀 ToW-Aware Korean Data Augmentation System")
    
    # Create configuration
    config = AugmentationConfig()
    
    # Check input file exists
    if not Path(config.input_file).exists():
        logger.error(f"Input file not found: {config.input_file}")
        logger.error("Please ensure ToW_koconovel_complete.json exists")
        return
    
    # Initialize augmentation engine
    engine = ToWAugmentationEngine(config)
    
    # Run augmentation
    try:
        results = engine.augment_dataset(config.input_file, config.output_file)
        
        # Print summary
        logger.info("\n📊 AUGMENTATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Original entries: {results['original_entries']:,}")
        logger.info(f"Generated augmented entries: {results['augmented_entries']:,}")
        logger.info(f"Total final entries: {results['total_entries']:,}")
        logger.info(f"Augmentation ratio: {results['augmented_entries']/results['original_entries']:.2f}x")
        
        logger.info("\nAugmentation breakdown:")
        for aug_type, count in results['augmentation_stats'].items():
            logger.info(f"  📝 {aug_type}: {count:,} variants")
        
        logger.info(f"\n✅ Augmented dataset saved as: {config.output_file}")
        
    except Exception as e:
        logger.error(f"❌ Augmentation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()