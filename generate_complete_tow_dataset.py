#!/usr/bin/env python3
"""
Complete ToW Dataset Generator for Korean Text
Processes entire koconovel.json dataset with sophisticated linguistic analysis
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ToWEnhancedEntry:
    doc_id: str
    original_text: str
    augmented_text: str
    tow_tokens: List[str]
    tow_count: int
    difficulty_markers: List[str]
    word_category: str
    prediction_challenge: str
    source: str
    story_id: str
    sentence_id: int

class KoreanLinguisticAnalyzer:
    def __init__(self):
        self.cultural_patterns = {
            # Traditional Korean foods
            '굴비': 'dried_corvina_fish', '김치': 'fermented_vegetables', '된장': 'soybean_paste',
            '고추장': 'chili_paste', '젓갈': 'fermented_seafood', '막걸리': 'rice_wine',
            '떡': 'rice_cake', '나물': 'seasoned_vegetables', '미역국': 'seaweed_soup',
            
            # Traditional places and geography
            '한양': 'old_seoul', '개성': 'gaeseong_city', '평양': 'pyongyang', 
            '제주': 'jeju_island', '경주': 'gyeongju_city', '안동': 'andong_city',
            
            # Cultural customs and ceremonies
            '제사': 'ancestral_rites', '차례': 'memorial_service', '성묘': 'grave_visiting',
            '돌잔치': 'first_birthday', '혼례': 'wedding_ceremony', '상례': 'funeral_rites',
            
            # Traditional occupations
            '기생': 'traditional_entertainer', '서당': 'traditional_school', '훈장': 'village_teacher',
            '포수': 'hunter', '나무꾼': 'woodcutter', '농부': 'farmer',
            
            # Traditional objects and clothing
            '한복': 'traditional_clothing', '갓': 'traditional_hat', '비녀': 'hairpin',
            '부채': 'folding_fan', '댕기': 'hair_ribbon', '저고리': 'jacket',
        }
        
        self.honorific_patterns = {
            # Honorific particles
            '께서': 'honorific_subject_particle',
            '께': 'honorific_to_particle', 
            '에서': 'location_particle',
            
            # Humble verbs
            '드리다': 'humble_give', '여쭙다': 'humble_ask', '뵙다': 'humble_see',
            '모시다': 'humble_serve', '올리다': 'humble_offer',
            
            # Respectful verbs
            '주시다': 'respectful_give', '계시다': 'respectful_be', '하시다': 'respectful_do',
            '말씀하시다': 'respectful_speak', '잡수시다': 'respectful_eat',
            
            # Formal endings
            '습니다': 'formal_declarative', '십니다': 'formal_honorific', 
            '어요': 'polite_informal', '아요': 'polite_informal',
        }
        
        self.archaic_patterns = {
            '하오': 'archaic_formal_ending', '하게': 'archaic_casual_ending',
            '일세': 'archaic_emphatic_ending', '네': 'archaic_informal_ending',
            '아뢰다': 'archaic_to_report', '아니오니': 'archaic_negation',
            '아니하다': 'archaic_negation', '으리라': 'archaic_future_presumptive',
            '이로다': 'archaic_exclamation', '도다': 'archaic_emphasis',
        }
        
        self.religious_historical_patterns = {
            '야소군': 'christian_zealot', '불교': 'buddhism', '천주교': 'catholicism',
            '개신교': 'protestantism', '유교': 'confucianism', '도교': 'taoism',
            '조선': 'joseon_dynasty', '고려': 'goryeo_dynasty', '신라': 'silla_dynasty',
            '백제': 'baekje_dynasty', '고구려': 'goguryeo_dynasty',
            '임진왜란': 'japanese_invasion', '한국전쟁': 'korean_war',
            '일제강점기': 'japanese_occupation',
        }
        
        self.dialectal_patterns = {
            '잉': 'gyeongsang_dialect', '심마': 'jeolla_dialect_what',
            '머싱': 'jeolla_dialect_what', '데기': 'dialect_place',
            '맨시리': 'dialect_very', '질랑': 'dialect_very',
            '앵모': 'dialect_mute_person', '답답한게': 'dialect_frustrating',
        }
        
        self.metaphorical_patterns = {
            '구름': 'cloud_metaphor', '달': 'moon_metaphor', '별': 'star_metaphor',
            '꽃': 'flower_metaphor', '나무': 'tree_metaphor', '바람': 'wind_metaphor',
            '새': 'bird_metaphor', '물': 'water_metaphor', '불': 'fire_metaphor',
            '산': 'mountain_metaphor', '바다': 'sea_metaphor',
        }
        
        self.loanword_patterns = {
            '러브레터': 'english_love_letter', '크리스마스': 'english_christmas',
            '커피': 'english_coffee', '버터': 'english_butter', '빵': 'portuguese_bread',
            '담배': 'japanese_tobacco', '가방': 'bag', '구두': 'japanese_shoes',
            '와이셔츠': 'japanese_white_shirt', '후라이팬': 'japanese_frying_pan',
        }

    def analyze_text_challenges(self, text: str) -> List[Dict]:
        """Analyze text for linguistic challenges and difficulty points"""
        challenges = []
        
        # Cultural references
        for pattern, challenge_type in self.cultural_patterns.items():
            if pattern in text:
                challenges.append({
                    'word': pattern,
                    'type': 'cultural_reference',
                    'subtype': challenge_type,
                    'difficulty': 0.9,
                    'position': text.find(pattern)
                })
        
        # Honorific complexity
        for pattern, challenge_type in self.honorific_patterns.items():
            if pattern in text:
                challenges.append({
                    'word': pattern,
                    'type': 'honorific_system',
                    'subtype': challenge_type,
                    'difficulty': 0.8,
                    'position': text.find(pattern)
                })
        
        # Archaic language
        for pattern, challenge_type in self.archaic_patterns.items():
            if pattern in text:
                challenges.append({
                    'word': pattern,
                    'type': 'archaic_language',
                    'subtype': challenge_type,
                    'difficulty': 0.9,
                    'position': text.find(pattern)
                })
        
        # Religious/Historical terms
        for pattern, challenge_type in self.religious_historical_patterns.items():
            if pattern in text:
                challenges.append({
                    'word': pattern,
                    'type': 'religious_historical',
                    'subtype': challenge_type,
                    'difficulty': 0.8,
                    'position': text.find(pattern)
                })
        
        # Dialectal expressions
        for pattern, challenge_type in self.dialectal_patterns.items():
            if pattern in text:
                challenges.append({
                    'word': pattern,
                    'type': 'dialectal_expression',
                    'subtype': challenge_type,
                    'difficulty': 0.9,
                    'position': text.find(pattern)
                })
        
        # Metaphorical language
        for pattern, challenge_type in self.metaphorical_patterns.items():
            if pattern in text:
                challenges.append({
                    'word': pattern,
                    'type': 'metaphorical_language',
                    'subtype': challenge_type,
                    'difficulty': 0.7,
                    'position': text.find(pattern)
                })
        
        # Foreign loanwords
        for pattern, challenge_type in self.loanword_patterns.items():
            if pattern in text:
                challenges.append({
                    'word': pattern,
                    'type': 'foreign_loanword',
                    'subtype': challenge_type,
                    'difficulty': 0.6,
                    'position': text.find(pattern)
                })
        
        return sorted(challenges, key=lambda x: x['difficulty'], reverse=True)

    def select_optimal_points(self, challenges: List[Dict], max_points: int = 4) -> List[Dict]:
        """Select optimal insertion points for ToW tokens"""
        if not challenges:
            return []
        
        # Group by type to ensure diversity
        by_type = {}
        for challenge in challenges:
            challenge_type = challenge['type']
            if challenge_type not in by_type:
                by_type[challenge_type] = []
            by_type[challenge_type].append(challenge)
        
        # Select best from each type
        selected = []
        type_priorities = ['cultural_reference', 'religious_historical', 'archaic_language', 
                          'honorific_system', 'dialectal_expression', 'metaphorical_language', 'foreign_loanword']
        
        for challenge_type in type_priorities:
            if challenge_type in by_type and len(selected) < max_points:
                # Take highest difficulty from this type
                best = max(by_type[challenge_type], key=lambda x: x['difficulty'])
                selected.append(best)
        
        # If still need more, add highest remaining
        remaining = [c for c in challenges if c not in selected]
        while len(selected) < max_points and remaining:
            best_remaining = max(remaining, key=lambda x: x['difficulty'])
            selected.append(best_remaining)
            remaining.remove(best_remaining)
        
        return sorted(selected, key=lambda x: x['position'])

    def generate_tow_reasoning(self, challenge: Dict, context_before: str, context_after: str) -> str:
        """Generate specific English reasoning for the challenge"""
        word = challenge['word']
        challenge_type = challenge['type']
        subtype = challenge['subtype']
        
        reasoning_templates = {
            'cultural_reference': f"This requires specific Korean cultural knowledge. '{word}' is a traditional Korean cultural reference that requires understanding of historical Korean society, customs, and traditional practices. The word choice reflects deep cultural context that would be difficult to predict without specific cultural background knowledge.",
            
            'honorific_system': f"This involves Korean honorific complexity. '{word}' is part of Korea's intricate honorific system requiring understanding of social hierarchies, age relationships, and contextual formality levels. Correct prediction requires analyzing the social dynamics between speakers and appropriate linguistic respect levels.",
            
            'archaic_language': f"This uses archaic or literary Korean language. '{word}' represents classical Korean linguistic forms that are no longer commonly used in modern speech. This requires knowledge of historical Korean grammar patterns and literary language conventions from earlier periods.",
            
            'religious_historical': f"This requires historical and religious knowledge. '{word}' refers to specific religious or historical contexts in Korean culture that require specialized knowledge of Korean history, religious practices, and their linguistic representations in traditional Korean society.",
            
            'dialectal_expression': f"This involves regional Korean dialect. '{word}' is a dialectal expression specific to certain Korean regions, requiring knowledge of regional linguistic variations and local cultural expressions that differ from standard Korean.",
            
            'metaphorical_language': f"This uses Korean metaphorical expression. '{word}' functions as part of a culturally-specific metaphor that requires understanding of Korean symbolic associations and traditional imagery patterns in Korean literature and speech.",
            
            'foreign_loanword': f"This involves foreign loanword adaptation. '{word}' is a foreign loanword adapted into Korean context, requiring understanding of how foreign concepts were linguistically and culturally integrated into Korean society and language."
        }
        
        base_reasoning = reasoning_templates.get(challenge_type, f"This requires specific linguistic knowledge for '{word}'.")
        
        # Add context-specific enhancement
        if context_before:
            context_hint = f" Given the preceding context about {context_before[-50:].strip()}, this word choice becomes more specific to the narrative situation."
        else:
            context_hint = ""
        
        return f"<ToW>{base_reasoning}{context_hint}</ToW>"

def process_complete_dataset():
    """Process the complete koconovel.json dataset"""
    input_file = "C:\\Users\\songj\\OneDrive\\Desktop\\Increase_MLLM_Ability\\2_datasets\\koconovel\\koconovel.json"
    output_file = "C:\\Users\\songj\\OneDrive\\Desktop\\Increase_MLLM_Ability\\ToW_koconovel_complete.json"
    
    analyzer = KoreanLinguisticAnalyzer()
    enhanced_entries = []
    
    logger.info("Loading complete koconovel dataset...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Processing {len(data)} documents...")
        
        for idx, entry in enumerate(data):
            if idx % 100 == 0:
                logger.info(f"Processing document {idx+1}/{len(data)}")
            
            doc_id = entry['doc_id']
            text = entry['text']
            
            # Extract story and sentence IDs
            parts = doc_id.split('_')
            story_id = parts[0] if parts else "unknown"
            sentence_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            
            # Analyze text for challenges
            challenges = analyzer.analyze_text_challenges(text)
            
            if challenges:
                # Select 2-4 optimal insertion points
                selected_challenges = analyzer.select_optimal_points(challenges, max_points=4)
                
                for challenge_idx, challenge in enumerate(selected_challenges):
                    # Create enhanced entry
                    word = challenge['word']
                    position = challenge['position']
                    
                    # Get context
                    context_before = text[:position]
                    context_after = text[position + len(word):]
                    
                    # Generate ToW reasoning
                    tow_token = analyzer.generate_tow_reasoning(challenge, context_before, context_after)
                    
                    # Create augmented text
                    augmented_text = text[:position] + tow_token + text[position:]
                    
                    # Create enhanced entry
                    enhanced_entry = ToWEnhancedEntry(
                        doc_id=f"{doc_id}_enhanced_{challenge_idx+1}",
                        original_text=text,
                        augmented_text=augmented_text,
                        tow_tokens=[tow_token],
                        tow_count=1,
                        difficulty_markers=[challenge['type']],
                        word_category=classify_word_category(challenge),
                        prediction_challenge=challenge['subtype'],
                        source="koconovel",
                        story_id=story_id,
                        sentence_id=sentence_id
                    )
                    
                    enhanced_entries.append(enhanced_entry)
            else:
                # No challenges found, create basic entry
                enhanced_entry = ToWEnhancedEntry(
                    doc_id=f"{doc_id}_basic",
                    original_text=text,
                    augmented_text=text,  # No augmentation
                    tow_tokens=[],
                    tow_count=0,
                    difficulty_markers=[],
                    word_category="trivial",
                    prediction_challenge="none",
                    source="koconovel",
                    story_id=story_id,
                    sentence_id=sentence_id
                )
                
                enhanced_entries.append(enhanced_entry)
    
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return
    
    # Save enhanced dataset
    logger.info(f"Saving {len(enhanced_entries)} enhanced entries...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(entry) for entry in enhanced_entries], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully created enhanced dataset: {output_file}")
        
        # Print statistics
        total_entries = len(enhanced_entries)
        entries_with_tow = sum(1 for entry in enhanced_entries if entry.tow_count > 0)
        total_tow_tokens = sum(entry.tow_count for entry in enhanced_entries)
        
        logger.info(f"Statistics:")
        logger.info(f"- Total entries: {total_entries}")
        logger.info(f"- Entries with ToW tokens: {entries_with_tow}")
        logger.info(f"- Total ToW tokens: {total_tow_tokens}")
        logger.info(f"- Average ToW tokens per enhanced entry: {total_tow_tokens/entries_with_tow if entries_with_tow > 0 else 0:.2f}")
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")

def classify_word_category(challenge: Dict) -> str:
    """Classify word prediction difficulty category"""
    difficulty = challenge['difficulty']
    challenge_type = challenge['type']
    
    if difficulty >= 0.9 or challenge_type in ['cultural_reference', 'dialectal_expression']:
        return "unpredictable"
    elif difficulty >= 0.7 or challenge_type in ['honorific_system', 'archaic_language']:
        return "soft_consistent"
    else:
        return "exact_match"

if __name__ == "__main__":
    process_complete_dataset()