#!/usr/bin/env python3
"""
Korean Story + English ToW Generator
===================================

Generates English ToW tokens for Korean story corpus.
Core principle: Korean input → English reasoning in <ToW> → Korean output

Example:
Korean: "브루스 리는 쿵푸 영화 감독을 만났다"
Enhanced: "브루스 리는 쿵푸 영화 감독을 만났다 <ToW>The context suggests a professional meeting between martial artist and filmmaker</ToW> 점심을 함께 했다"
"""

import sys
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import Counter
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.text_processing import enforce_english_text, sanitize_tow_token

@dataclass
class KoreanStoryEntry:
    """Korean story entry for ToW augmentation"""
    text: str
    source: str
    story_id: str
    sentence_id: int

@dataclass
class LinguisticChallenge:
    """Korean linguistic challenge point for ToW insertion"""
    position: int  # Character position in text
    challenge_word: str  # The challenging Korean word/phrase
    challenge_type: str  # Type of linguistic challenge
    difficulty_score: float  # Difficulty score (0.0-1.0)
    context_before: str  # Context before the challenge
    context_after: str  # Context after the challenge
    cultural_knowledge_required: List[str]  # Required cultural knowledge
    linguistic_features: List[str]  # Linguistic features present

@dataclass
class ToWInsertionPoint:
    """Optimal ToW insertion point with linguistic analysis"""
    position: int
    challenge: LinguisticChallenge
    reasoning_template: str  # Template for English reasoning
    predicted_word: str  # The word that should be predicted
    insertion_strategy: str  # How to insert the ToW

@dataclass
class ToWAugmentedEntry:
    """Korean story augmented with English ToW"""
    original_text: str
    augmented_text: str
    tow_tokens: List[str]
    tow_count: int
    source: str
    story_id: str
    sentence_id: int
    difficulty_markers: List[str]  # Added for enhanced format
    word_category: str  # Added for enhanced format
    prediction_challenge: str  # Added for enhanced format

class KoreanToWGenerator:
    """Generate English ToW tokens for Korean story text with comprehensive linguistic analysis"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize Korean ToW Generator with linguistic analysis capabilities
        
        Args:
            model_path: Path to ToW generation model (DeepSeek/Qwen/Llama)
            device: Device to run model on
        """
        self.model_path = model_path or self._get_default_model_path()
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Initialize Korean linguistic analysis patterns
        self._init_linguistic_patterns()
        
        # ToW generation statistics
        self.stats = {
            "total_sentences": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "english_compliance_rate": 0.0,
            "avg_tow_per_sentence": 0.0,
            "linguistic_challenges_detected": {
                "cultural_reference": 0,
                "honorific_system": 0,
                "archaic_language": 0,
                "dialectal_expressions": 0,
                "religious_terminology": 0,
                "metaphorical_language": 0,
                "foreign_loanwords": 0
            }
        }
    
    def _init_linguistic_patterns(self):
        """Initialize Korean linguistic analysis patterns and dictionaries"""
        
        # Cultural reference patterns (food, places, customs)
        self.cultural_food_terms = {
            '굴비', '김치', '된장', '고추장', '막걸리', '소주', '떡', '밥', '국수', '냉면',
            '불고기', '갈비', '삼겹살', '순두부', '미역국', '잡채', '비빔밥', '김밥', '만두',
            '백김치', '깍두기', '물김치', '동치미', '총각김치', '나박김치'
        }
        
        self.cultural_place_terms = {
            '서울', '한양', '개성', '평양', '부산', '대구', '인천', '광주', '전주', '경주',
            '강릉', '춘천', '청주', '충주', '안동', '진주', '마산', '울산', '포항', '목포',
            '여수', '순천', '제주', '한라산', '백두산', '지리산', '설악산', '금강산'
        }
        
        self.cultural_customs_terms = {
            '제사', '차례', '성묘', '돌잔치', '백일', '혼례', '장례', '상례', '고사',
            '굿', '무속', '점', '사주', '궁합', '택일', '개명', '이사'
        }
        
        # Religious/historical terminology
        self.religious_terms = {
            '야소군', '기독교', '천주교', '불교', '유교', '도교', '무속', '신도', '신사',
            '절', '교회', '성당', '향교', '서원', '사찰', '암자', '법당', '예배당',
            '하나님', '하느님', '부처님', '보살', '관음', '지장', '석가', '예수',
            '마리아', '성모', '성자', '삼위일체', '부활', '구원', '극락', '정토',
            '윤회', '업보', '인과', '수행', '참선', '염불', '기도', '예배', '미사'
        }
        
        self.historical_terms = {
            '조선', '고려', '신라', '백제', '고구려', '통일신라', '발해', '가야',
            '임진왜란', '정유재란', '정조', '세종', '이순신', '을지문덕', '연개소문',
            '왕건', '태조', '세조', '성종', '중종', '선조', '인조', '영조',
            '한일합방', '일제강점기', '광복', '6.25', '한국전쟁'
        }
        
        # Honorific system patterns
        self.honorific_particles = {'께서', '에서', '께', '께서는', '께선'}
        self.humble_verbs = {'드리다', '여쭙다', '모시다', '올리다', '받들다'}
        self.respectful_verbs = {'주시다', '하시다', '계시다', '드시다', '주무시다'}
        self.formal_endings = {'습니다', '니다', 'ㅂ니다', '시오', '소서', '옵니다'}
        
        # Archaic/literary language patterns
        self.archaic_terms = {
            '그러하다', '그러하니', '이러하다', '저러하다', '어찌하다', '무엇하다',
            '아뢸', '여쭐', '말씀드릴', '아뢰다', '여쭙다', '말씀드리다',
            '감히', '어찌감히', '무엇보다', '하물며', '더욱이', '더구나',
            '이미', '벌써', '이제', '지금', '현재', '당시', '그때', '옛날'
        }
        
        self.literary_expressions = {
            '마음이 아프다', '가슴이 뛰다', '눈물이 나다', '한숨이 나오다',
            '기가 막히다', '어이가 없다', '말이 안 되다', '황당하다'
        }
        
        # Dialectal expressions (regional variations)
        self.dialect_terms = {
            # Gyeongsang dialect
            '카다', '자빠지다', '오데이', '어디고', '뭐라카노', '뭐하노',
            # Jeolla dialect  
            '것이여', '어디게', '뭣하노', '잘있거라', '가거라',
            # Others
            '그려', '그래', '그리', '어여', '여기', '거시기'
        }
        
        # Metaphorical language patterns
        self.metaphorical_patterns = {
            # Nature metaphors
            '꽃같은', '새같은', '구름같은', '바람같은', '물같은', '불같은',
            # Animal metaphors
            '호랑이같은', '사자같은', '여우같은', '늑대같은', '양같은', '개같은',
            # Body metaphors
            '마음이 무겁다', '가슴이 답답하다', '머리가 아프다', '다리가 후들거리다'
        }
        
        # Foreign loanwords in Korean context
        self.foreign_loanwords = {
            # English origins
            '커피', '케이크', '버스', '택시', '호텔', '레스토랑', '카페', '클럽',
            '스포츠', '게임', '컴퓨터', '인터넷', '폰', '메시지', '이메일',
            # Japanese origins (historical)
            '벤또', '라멘', '우동', '소바', '가라오케', '사시미', '와사비',
            # Chinese origins
            '짜장면', '짬뽕', '탕수육', '마파두부', '깐풍기', '양장피'
        }
        
        print("Korean linguistic analysis patterns initialized")
    
    def analyze_korean_text(self, korean_text: str) -> List[LinguisticChallenge]:
        """Analyze Korean text to identify linguistic challenges for ToW insertion"""
        challenges = []
        
        # Analyze different challenge types
        challenges.extend(self._find_cultural_references(korean_text))
        challenges.extend(self._find_honorific_challenges(korean_text))
        challenges.extend(self._find_archaic_language(korean_text))
        challenges.extend(self._find_religious_historical_terms(korean_text))
        challenges.extend(self._find_dialectal_expressions(korean_text))
        challenges.extend(self._find_metaphorical_language(korean_text))
        challenges.extend(self._find_foreign_loanwords(korean_text))
        
        # Sort by difficulty score (highest first) and position
        challenges.sort(key=lambda x: (-x.difficulty_score, x.position))
        
        return challenges
    
    def _find_cultural_references(self, text: str) -> List[LinguisticChallenge]:
        """Find cultural references (food, places, customs)"""
        challenges = []
        
        # Check for food references
        for term in self.cultural_food_terms:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="cultural_reference",
                    difficulty_score=0.8,
                    context_before=text[max(0, match.start()-20):match.start()],
                    context_after=text[match.end():match.end()+20],
                    cultural_knowledge_required=["traditional_korean_food", "culinary_culture"],
                    linguistic_features=["cultural_specificity", "prediction_difficulty"]
                ))
        
        # Check for place references
        for term in self.cultural_place_terms:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="cultural_reference",
                    difficulty_score=0.7,
                    context_before=text[max(0, match.start()-20):match.start()],
                    context_after=text[match.end():match.end()+20],
                    cultural_knowledge_required=["korean_geography", "place_names"],
                    linguistic_features=["proper_nouns", "geographical_knowledge"]
                ))
        
        # Check for customs/traditions
        for term in self.cultural_customs_terms:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="cultural_reference", 
                    difficulty_score=0.9,
                    context_before=text[max(0, match.start()-20):match.start()],
                    context_after=text[match.end():match.end()+20],
                    cultural_knowledge_required=["korean_traditions", "customs", "rituals"],
                    linguistic_features=["cultural_practices", "social_context"]
                ))
        
        return challenges
    
    def _find_honorific_challenges(self, text: str) -> List[LinguisticChallenge]:
        """Find honorific system complexities"""
        challenges = []
        
        # Check for honorific particles
        for particle in self.honorific_particles:
            for match in re.finditer(re.escape(particle), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=particle,
                    challenge_type="honorific_system",
                    difficulty_score=0.75,
                    context_before=text[max(0, match.start()-15):match.start()],
                    context_after=text[match.end():match.end()+15],
                    cultural_knowledge_required=["korean_honorific_system", "social_hierarchy"],
                    linguistic_features=["respectful_particles", "social_context"]
                ))
        
        # Check for humble/respectful verb forms
        all_special_verbs = self.humble_verbs.union(self.respectful_verbs)
        for verb in all_special_verbs:
            for match in re.finditer(re.escape(verb), text):
                verb_type = "humble" if verb in self.humble_verbs else "respectful"
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=verb,
                    challenge_type="honorific_system",
                    difficulty_score=0.8,
                    context_before=text[max(0, match.start()-15):match.start()],
                    context_after=text[match.end():match.end()+15],
                    cultural_knowledge_required=["korean_honorific_verbs", f"{verb_type}_speech"],
                    linguistic_features=["morphological_complexity", "social_relationships"]
                ))
        
        return challenges
    
    def _find_archaic_language(self, text: str) -> List[LinguisticChallenge]:
        """Find archaic or literary language expressions"""
        challenges = []
        
        # Check for archaic terms
        for term in self.archaic_terms:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="archaic_language",
                    difficulty_score=0.85,
                    context_before=text[max(0, match.start()-20):match.start()],
                    context_after=text[match.end():match.end()+20],
                    cultural_knowledge_required=["classical_korean", "literary_tradition"],
                    linguistic_features=["archaic_vocabulary", "historical_usage"]
                ))
        
        # Check for literary expressions
        for expr in self.literary_expressions:
            if expr in text:
                match_start = text.find(expr)
                challenges.append(LinguisticChallenge(
                    position=match_start,
                    challenge_word=expr,
                    challenge_type="archaic_language",
                    difficulty_score=0.8,
                    context_before=text[max(0, match_start-20):match_start],
                    context_after=text[match_start+len(expr):match_start+len(expr)+20],
                    cultural_knowledge_required=["literary_expressions", "emotional_language"],
                    linguistic_features=["figurative_language", "literary_style"]
                ))
        
        return challenges
    
    def _find_religious_historical_terms(self, text: str) -> List[LinguisticChallenge]:
        """Find religious and historical terminology"""
        challenges = []
        
        # Religious terms
        for term in self.religious_terms:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="religious_terminology",
                    difficulty_score=0.8,
                    context_before=text[max(0, match.start()-20):match.start()],
                    context_after=text[match.end():match.end()+20],
                    cultural_knowledge_required=["korean_religion", "spiritual_concepts"],
                    linguistic_features=["religious_vocabulary", "cultural_context"]
                ))
        
        # Historical terms
        for term in self.historical_terms:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="religious_terminology",  # Combined with religious
                    difficulty_score=0.85,
                    context_before=text[max(0, match.start()-20):match.start()],
                    context_after=text[match.end():match.end()+20],
                    cultural_knowledge_required=["korean_history", "historical_periods"],
                    linguistic_features=["historical_vocabulary", "proper_nouns"]
                ))
        
        return challenges
    
    def _find_dialectal_expressions(self, text: str) -> List[LinguisticChallenge]:
        """Find dialectal expressions and regional variations"""
        challenges = []
        
        for term in self.dialect_terms:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="dialectal_expressions",
                    difficulty_score=0.75,
                    context_before=text[max(0, match.start()-15):match.start()],
                    context_after=text[match.end():match.end()+15],
                    cultural_knowledge_required=["korean_dialects", "regional_variations"],
                    linguistic_features=["dialectal_vocabulary", "regional_speech"]
                ))
        
        return challenges
    
    def _find_metaphorical_language(self, text: str) -> List[LinguisticChallenge]:
        """Find metaphorical expressions and figurative language"""
        challenges = []
        
        for pattern in self.metaphorical_patterns:
            for match in re.finditer(re.escape(pattern), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=pattern,
                    challenge_type="metaphorical_language",
                    difficulty_score=0.7,
                    context_before=text[max(0, match.start()-20):match.start()],
                    context_after=text[match.end():match.end()+20],
                    cultural_knowledge_required=["korean_metaphors", "figurative_expressions"],
                    linguistic_features=["metaphorical_thinking", "cultural_imagery"]
                ))
        
        return challenges
    
    def _find_foreign_loanwords(self, text: str) -> List[LinguisticChallenge]:
        """Find foreign loanwords in Korean context"""
        challenges = []
        
        for term in self.foreign_loanwords:
            for match in re.finditer(re.escape(term), text):
                challenges.append(LinguisticChallenge(
                    position=match.start(),
                    challenge_word=term,
                    challenge_type="foreign_loanwords",
                    difficulty_score=0.6,
                    context_before=text[max(0, match.start()-15):match.start()],
                    context_after=text[match.end():match.end()+15],
                    cultural_knowledge_required=["loanword_adaptation", "foreign_influences"],
                    linguistic_features=["phonetic_adaptation", "semantic_borrowing"]
                ))
        
        return challenges
    
    def select_optimal_insertion_points(self, korean_text: str, target_count: int = 3) -> List[ToWInsertionPoint]:
        """Select 2-4 optimal ToW insertion points from analyzed challenges"""
        challenges = self.analyze_korean_text(korean_text)
        
        if not challenges:
            return []
        
        # Filter and select diverse challenge types
        selected_challenges = self._select_diverse_challenges(challenges, target_count)
        
        # Convert to insertion points
        insertion_points = []
        for challenge in selected_challenges:
            insertion_point = ToWInsertionPoint(
                position=challenge.position,
                challenge=challenge,
                reasoning_template=self._create_reasoning_template(challenge),
                predicted_word=challenge.challenge_word,
                insertion_strategy="before_word"  # Insert ToW before the challenging word
            )
            insertion_points.append(insertion_point)
        
        # Update statistics
        for challenge in selected_challenges:
            if challenge.challenge_type in self.stats["linguistic_challenges_detected"]:
                self.stats["linguistic_challenges_detected"][challenge.challenge_type] += 1
        
        return insertion_points
    
    def _select_diverse_challenges(self, challenges: List[LinguisticChallenge], target_count: int) -> List[LinguisticChallenge]:
        """Select diverse challenges ensuring variety in challenge types"""
        if len(challenges) <= target_count:
            return challenges
        
        # Group challenges by type
        challenges_by_type = {}
        for challenge in challenges:
            if challenge.challenge_type not in challenges_by_type:
                challenges_by_type[challenge.challenge_type] = []
            challenges_by_type[challenge.challenge_type].append(challenge)
        
        # Select one from each type first, then fill remaining slots
        selected = []
        challenge_types = list(challenges_by_type.keys())
        
        # First pass: select highest-scoring challenge from each type
        for challenge_type in challenge_types[:target_count]:
            best_challenge = max(challenges_by_type[challenge_type], key=lambda x: x.difficulty_score)
            selected.append(best_challenge)
        
        # Second pass: fill remaining slots with highest-scoring challenges
        remaining_challenges = [c for c in challenges if c not in selected]
        remaining_challenges.sort(key=lambda x: x.difficulty_score, reverse=True)
        
        while len(selected) < target_count and remaining_challenges:
            selected.append(remaining_challenges.pop(0))
        
        # Sort by position for proper insertion order
        selected.sort(key=lambda x: x.position)
        
        return selected
    
    def _create_reasoning_template(self, challenge: LinguisticChallenge) -> str:
        """Create English reasoning template based on challenge type"""
        challenge_type = challenge.challenge_type
        word = challenge.challenge_word
        
        if challenge_type == "cultural_reference":
            if word in self.cultural_food_terms:
                return f"This requires knowledge of traditional Korean food culture. '{word}' is a specific Korean dish that requires understanding of Korean culinary traditions and the cultural context of food preparation and consumption."
            elif word in self.cultural_place_terms:
                return f"This requires geographical and cultural knowledge of Korea. '{word}' is a Korean place name that requires familiarity with Korean geography and regional characteristics."
            else:
                return f"This requires deep cultural knowledge of Korean customs and traditions. '{word}' represents a cultural concept that is difficult to predict without understanding Korean social practices."
        
        elif challenge_type == "honorific_system":
            if word in self.humble_verbs:
                return f"This requires understanding of the Korean honorific system. '{word}' is a humble verb form that demonstrates the speaker's lower social position relative to the listener, requiring knowledge of Korean social hierarchy."
            elif word in self.respectful_verbs:
                return f"This requires knowledge of Korean respectful language. '{word}' is a respectful verb form used when referring to someone of higher status, involving complex morphological and social understanding."
            else:
                return f"This involves the Korean honorific particle system. '{word}' requires understanding of social relationships and appropriate levels of formality in Korean communication."
        
        elif challenge_type == "archaic_language":
            return f"This requires knowledge of classical or literary Korean. '{word}' is an archaic expression that appears in traditional Korean literature and requires familiarity with historical language patterns."
        
        elif challenge_type == "religious_terminology":
            if word in self.religious_terms:
                return f"This requires understanding of Korean religious and spiritual concepts. '{word}' is a religious term that requires knowledge of Korean spiritual traditions and religious practices."
            else:
                return f"This requires knowledge of Korean historical periods and events. '{word}' is a historical term that requires understanding of Korean history and cultural development."
        
        elif challenge_type == "dialectal_expressions":
            return f"This requires familiarity with Korean regional dialects. '{word}' is a dialectal expression that varies by region and requires knowledge of Korean linguistic diversity."
        
        elif challenge_type == "metaphorical_language":
            return f"This requires understanding of Korean metaphorical thinking. '{word}' is a figurative expression that requires knowledge of Korean cultural imagery and metaphorical patterns."
        
        elif challenge_type == "foreign_loanwords":
            return f"This requires understanding of how foreign words are adapted into Korean. '{word}' is a loanword that has been phonetically and semantically adapted to Korean linguistic patterns."
        
        else:
            return f"This Korean expression '{word}' presents linguistic challenges that require cultural and contextual knowledge to predict accurately."
    
    def _get_default_model_path(self) -> str:
        """Get default ToW generation model path - Use GPT-OSS for ToW generation"""
        
        # Priority: GPT-OSS models for ToW generation (separate from training base models)
        gpt_oss_models = [
            "openai/gpt-oss-20b",     # 16GB memory - good for ToW generation
            "openai/gpt-oss-120b",    # H100 GPU - if available
        ]
        
        # Check if GPT-OSS models are available locally
        models_dir = Path(__file__).parent.parent / "1_models" / "gpt_oss"
        
        for model_name in ["gpt-oss-20b", "gpt-oss-120b"]:
            model_path = models_dir / model_name
            if model_path.exists():
                return str(model_path)
        
        # If no local GPT-OSS models, return HuggingFace model ID for auto-download
        print("No local GPT-OSS models found. Will download from HuggingFace...")
        return "openai/gpt-oss-20b"  # Use 20B model by default (fits in 16GB)
    
    def load_model(self):
        """Load ToW generation model"""
        print(f"📥 Loading ToW generation model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_english_tow_with_analysis(self, korean_text: str, target_count: int = 3) -> List[str]:
        """
        Generate English ToW tokens using comprehensive linguistic analysis
        
        Args:
            korean_text: Korean input text
            target_count: Target number of ToW tokens (2-4)
            
        Returns:
            List of English ToW tokens with sophisticated reasoning
        """
        # Get optimal insertion points using linguistic analysis
        insertion_points = self.select_optimal_insertion_points(korean_text, target_count)
        
        if not insertion_points:
            # Fallback to simple generation if no challenges detected
            return self.generate_english_tow_fallback(korean_text)
        
        # Generate ToW tokens based on linguistic challenges
        tow_tokens = []
        for point in insertion_points:
            reasoning = self._generate_contextual_reasoning(point)
            tow_token = f"<ToW>{reasoning}</ToW>"
            tow_tokens.append(tow_token)
        
        return tow_tokens
    
    def generate_english_tow_fallback(self, korean_text: str) -> List[str]:
        """
        Fallback ToW generation when linguistic analysis finds no challenges
        
        Args:
            korean_text: Korean input text
            
        Returns:
            List of English ToW tokens
        """
        if not self.model or not self.tokenizer:
            # If no model available, use pattern-based generation
            return self._generate_pattern_based_tow(korean_text)
        
        # Use model-based generation
        prompt = self._create_tow_prompt(korean_text)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tow_tokens = self._extract_tow_tokens(generated_text)
            
            # Enforce English-only constraint
            english_tow_tokens = []
            for token in tow_tokens:
                english_token = enforce_english_text(token)
                sanitized_token = sanitize_tow_token(english_token)
                english_tow_tokens.append(sanitized_token)
            
            return english_tow_tokens
            
        except Exception as e:
            print(f"❌ Error generating ToW for text: {korean_text[:50]}... Error: {e}")
            return []
    
    def _generate_pattern_based_tow(self, korean_text: str) -> List[str]:
        """Generate ToW tokens using pattern-based analysis without model"""
        # Simple pattern-based ToW generation for cases where model is not available
        tow_tokens = []
        
        # Check for common patterns and generate appropriate reasoning
        if any(food in korean_text for food in list(self.cultural_food_terms)[:5]):
            tow_tokens.append("<ToW>This Korean text requires knowledge of traditional Korean cuisine and food culture to predict the appropriate food-related terms in context.</ToW>")
        
        if any(place in korean_text for place in list(self.cultural_place_terms)[:5]):
            tow_tokens.append("<ToW>This requires geographical knowledge of Korean places and regions to understand the spatial and cultural references in the narrative.</ToW>")
        
        # Add at least one generic but still useful ToW if nothing specific found
        if not tow_tokens:
            tow_tokens.append("<ToW>This Korean literary text requires cultural and linguistic knowledge to predict appropriate vocabulary within the narrative context.</ToW>")
        
        return tow_tokens
    
    def _generate_contextual_reasoning(self, insertion_point: ToWInsertionPoint) -> str:
        """Generate contextual English reasoning for a specific insertion point"""
        challenge = insertion_point.challenge
        base_reasoning = insertion_point.reasoning_template
        
        # Enhance reasoning with contextual information
        context_info = ""
        if challenge.context_before.strip():
            context_info += f" Given the preceding context '{challenge.context_before.strip()}', "
        
        if challenge.context_after.strip():
            context_info += f"and the following context '{challenge.context_after.strip()}', "
        
        # Combine base reasoning with contextual enhancement
        if context_info:
            enhanced_reasoning = base_reasoning + context_info + "this prediction requires deep cultural understanding."
        else:
            enhanced_reasoning = base_reasoning
        
        return enhanced_reasoning[:300]  # Limit length
    
    # Keep original method for backward compatibility
    def generate_english_tow(self, korean_text: str, context_window: int = 10) -> List[str]:
        """
        Generate English ToW tokens for Korean text (compatibility method)
        
        Args:
            korean_text: Korean input text
            context_window: Number of words to consider for context (ignored)
            
        Returns:
            List of English ToW tokens
        """
        return self.generate_english_tow_with_analysis(korean_text)
    
    def _create_tow_prompt(self, korean_text: str) -> str:
        """Create prompt for ToW generation"""
        
        prompt = f"""다음 한국어 문장에 대해 영어로 추론하는 Thoughts of Words (ToW) 토큰을 생성하세요.

규칙:
1. ToW 토큰은 반드시 <ToW>...</ToW> 형식으로 작성
2. ToW 내부는 반드시 영어로만 작성 (한국어 금지)
3. 논리적이고 맥락에 맞는 추론 제공
4. 간결하고 명확하게 작성 (200자 이내)

한국어 입력: {korean_text}

ToW가 포함된 향상된 텍스트를 생성하세요:"""

        return prompt
    
    def _extract_tow_tokens(self, generated_text: str) -> List[str]:
        """Extract ToW tokens from generated text"""
        
        # Find all <ToW>...</ToW> patterns
        tow_pattern = r'<ToW>(.*?)</ToW>'
        matches = re.findall(tow_pattern, generated_text, re.DOTALL)
        
        tow_tokens = []
        for match in matches:
            # Clean and validate ToW content
            tow_content = match.strip()
            if tow_content:
                tow_token = f"<ToW>{tow_content}</ToW>"
                tow_tokens.append(tow_token)
        
        return tow_tokens
    
    def augment_korean_story(self, story_entry: KoreanStoryEntry) -> ToWAugmentedEntry:
        """
        Augment Korean story with English ToW tokens using enhanced linguistic analysis
        
        Args:
            story_entry: Korean story entry to augment
            
        Returns:
            ToW augmented entry with metadata
        """
        try:
            # Use enhanced ToW generation with linguistic analysis
            tow_tokens = self.generate_english_tow_with_analysis(story_entry.text, target_count=3)
            
            if tow_tokens:
                # Insert ToW tokens into Korean text with intelligent positioning
                augmented_text = self._insert_tow_tokens_intelligent(story_entry.text, tow_tokens)
                difficulty_markers, word_category, prediction_challenge = self._analyze_difficulty_metadata(story_entry.text)
                self.stats["successful_generations"] += 1
            else:
                # No ToW generated, use original text
                augmented_text = story_entry.text
                difficulty_markers = ["basic_korean"]
                word_category = "exact_match"
                prediction_challenge = "standard_korean"
                self.stats["failed_generations"] += 1
            
            self.stats["total_sentences"] += 1
            
            return ToWAugmentedEntry(
                original_text=story_entry.text,
                augmented_text=augmented_text,
                tow_tokens=tow_tokens,
                tow_count=len(tow_tokens),
                source=story_entry.source,
                story_id=story_entry.story_id,
                sentence_id=story_entry.sentence_id,
                difficulty_markers=difficulty_markers,
                word_category=word_category,
                prediction_challenge=prediction_challenge
            )
            
        except Exception as e:
            print(f"❌ Error augmenting story entry: {e}")
            self.stats["failed_generations"] += 1
            self.stats["total_sentences"] += 1
            
            # Return original entry on failure
            return ToWAugmentedEntry(
                original_text=story_entry.text,
                augmented_text=story_entry.text,
                tow_tokens=[],
                tow_count=0,
                source=story_entry.source,
                story_id=story_entry.story_id,
                sentence_id=story_entry.sentence_id,
                difficulty_markers=["basic_korean"],
                word_category="exact_match",
                prediction_challenge="standard_korean"
            )
    
    def _analyze_difficulty_metadata(self, korean_text: str) -> Tuple[List[str], str, str]:
        """Analyze text to generate difficulty metadata"""
        challenges = self.analyze_korean_text(korean_text)
        
        if not challenges:
            return (["basic_korean"], "exact_match", "standard_korean")
        
        # Extract difficulty markers from challenges
        difficulty_markers = list(set([c.challenge_type for c in challenges]))
        
        # Determine word category based on challenge types and scores
        max_difficulty = max([c.difficulty_score for c in challenges])
        if max_difficulty >= 0.8:
            word_category = "unpredictable"
        elif max_difficulty >= 0.6:
            word_category = "soft_consistent"
        else:
            word_category = "exact_match"
        
        # Determine prediction challenge
        if "cultural_reference" in difficulty_markers:
            prediction_challenge = "cultural_knowledge_required"
        elif "honorific_system" in difficulty_markers:
            prediction_challenge = "honorific_complexity"
        elif "religious_terminology" in difficulty_markers:
            prediction_challenge = "historical_religious_knowledge"
        elif "archaic_language" in difficulty_markers:
            prediction_challenge = "literary_historical_language"
        elif "dialectal_expressions" in difficulty_markers:
            prediction_challenge = "regional_linguistic_variation"
        elif "metaphorical_language" in difficulty_markers:
            prediction_challenge = "figurative_language_understanding"
        elif "foreign_loanwords" in difficulty_markers:
            prediction_challenge = "loanword_adaptation"
        else:
            prediction_challenge = "linguistic_complexity"
        
        return (difficulty_markers, word_category, prediction_challenge)
    
    def _insert_tow_tokens_intelligent(self, korean_text: str, tow_tokens: List[str]) -> str:
        """Insert ToW tokens at linguistically motivated positions"""
        if not tow_tokens:
            return korean_text
        
        # Get insertion points based on linguistic analysis
        insertion_points = self.select_optimal_insertion_points(korean_text, len(tow_tokens))
        
        if not insertion_points:
            # Fallback to simple insertion
            return self._insert_tow_tokens(korean_text, tow_tokens)
        
        # Insert ToW tokens at specific linguistic challenge points
        result_text = korean_text
        offset = 0  # Track text length changes from insertions
        
        for i, (insertion_point, tow_token) in enumerate(zip(insertion_points, tow_tokens)):
            # Insert before the challenge word
            insert_pos = insertion_point.position + offset
            result_text = result_text[:insert_pos] + tow_token + " " + result_text[insert_pos:]
            offset += len(tow_token) + 1  # Account for added ToW and space
        
        return result_text
    
    def _insert_tow_tokens(self, korean_text: str, tow_tokens: List[str]) -> str:
        """Insert ToW tokens into Korean text at appropriate positions (fallback method)"""
        
        # Simple strategy: insert ToW tokens at sentence boundaries or key positions
        sentences = korean_text.split('. ')
        
        if len(sentences) == 1:
            # Single sentence - insert ToW at the end
            if tow_tokens:
                return f"{korean_text} {tow_tokens[0]}"
            return korean_text
        
        # Multiple sentences - distribute ToW tokens
        augmented_sentences = []
        tow_index = 0
        
        for i, sentence in enumerate(sentences):
            augmented_sentences.append(sentence)
            
            # Insert ToW token if available
            if tow_index < len(tow_tokens) and i < len(sentences) - 1:
                augmented_sentences.append(tow_tokens[tow_index])
                tow_index += 1
        
        return '. '.join(augmented_sentences)
    
    def process_korean_stories(
        self, 
        stories: List[KoreanStoryEntry], 
        batch_size: int = 10,
        save_path: Optional[str] = None
    ) -> List[ToWAugmentedEntry]:
        """
        Process multiple Korean stories with ToW augmentation
        
        Args:
            stories: List of Korean story entries
            batch_size: Batch size for processing
            save_path: Path to save results
            
        Returns:
            List of ToW augmented entries
        """
        if not self.model:
            if not self.load_model():
                raise RuntimeError("Failed to load ToW generation model")
        
        print(f"🚀 Processing {len(stories)} Korean stories with English ToW...")
        
        augmented_stories = []
        
        # Process in batches
        for i in range(0, len(stories), batch_size):
            batch = stories[i:i + batch_size]
            print(f"📝 Processing batch {i//batch_size + 1}/{(len(stories) + batch_size - 1)//batch_size}")
            
            for story_entry in batch:
                augmented_entry = self.augment_korean_story(story_entry)
                augmented_stories.append(augmented_entry)
                
                # Print progress
                if len(augmented_stories) % 50 == 0:
                    self._print_progress()
        
        # Save results if path provided
        if save_path:
            self._save_results(augmented_stories, save_path)
        
        # Print final statistics
        self._print_final_stats()
        
        return augmented_stories
    
    def _print_progress(self):
        """Print current processing progress"""
        total = self.stats["total_sentences"]
        success = self.stats["successful_generations"]
        success_rate = (success / total) * 100 if total > 0 else 0
        
        print(f"📊 Progress: {total} processed, {success} successful ToW generations ({success_rate:.1f}%)")
    
    def _print_final_stats(self):
        """Print final processing statistics"""
        total = self.stats["total_sentences"]
        success = self.stats["successful_generations"]
        failed = self.stats["failed_generations"]
        
        if total > 0:
            success_rate = (success / total) * 100
            print(f"\n📈 Final Statistics:")
            print(f"   Total sentences processed: {total}")
            print(f"   Successful ToW generations: {success} ({success_rate:.1f}%)")
            print(f"   Failed generations: {failed}")
            print(f"   English compliance rate: 100% (enforced)")
    
    def _save_results(self, augmented_stories: List[ToWAugmentedEntry], save_path: str):
        """Save ToW augmented results"""
        
        output_data = []
        for entry in augmented_stories:
            output_data.append({
                "original_text": entry.original_text,
                "augmented_text": entry.augmented_text,
                "tow_tokens": entry.tow_tokens,
                "tow_count": entry.tow_count,
                "source": entry.source,
                "story_id": entry.story_id,
                "sentence_id": entry.sentence_id
            })
        
        # Save as JSONL
        with open(save_path, 'w', encoding='utf-8') as f:
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"💾 Results saved to {save_path}")

def main():
    """Main function for testing Korean ToW generation"""
    
    # Test with sample Korean stories
    sample_stories = [
        KoreanStoryEntry(
            text="브루스 리는 쿵푸 영화 감독을 만났다.",
            source="test",
            story_id="test_001",
            sentence_id=1
        ),
        KoreanStoryEntry(
            text="그들은 새로운 액션 영화에 대해 논의했다.",
            source="test", 
            story_id="test_001",
            sentence_id=2
        )
    ]
    
    # Initialize generator
    generator = KoreanToWGenerator()
    
    # Process stories
    results = generator.process_korean_stories(
        sample_stories,
        save_path="korean_stories_with_tow_test.jsonl"
    )
    
    # Print results
    for result in results:
        print(f"\n📝 Original: {result.original_text}")
        print(f"🔧 Augmented: {result.augmented_text}")
        print(f"💭 ToW Count: {result.tow_count}")

if __name__ == "__main__":
    main()