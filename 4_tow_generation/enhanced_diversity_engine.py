#!/usr/bin/env python3
"""
Enhanced Diversity Engine for Korean ToW Data Augmentation
다양성 점수 39.5% → 65%+ 달성을 위한 고급 증강 시스템
"""

import json
import re
import random
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class DiversityMetrics:
    """다양성 측정 지표"""
    lexical_diversity: float = 0.0    # 어휘 다양성
    structural_diversity: float = 0.0  # 구조적 다양성
    semantic_diversity: float = 0.0    # 의미적 다양성
    length_diversity: float = 0.0      # 길이 다양성
    overall_diversity: float = 0.0     # 종합 다양성

class AdvancedDiversityEngine:
    """고급 다양성 증강 엔진"""
    
    def __init__(self):
        self.initialize_diversity_patterns()
        self.diversity_cache = {}
        
    def initialize_diversity_patterns(self):
        """다양성 증강 패턴 초기화"""
        
        # 1. 구조적 변형 패턴
        self.structural_patterns = {
            # 능동-수동 변환
            'active_passive': [
                (r'(\w+)가 (\w+)을 (\w+다)', r'\2이 \1에 의해 \3'),
                (r'(\w+)이 (\w+)를 (\w+다)', r'\2가 \1에 의해 \3'),
            ],
            
            # 문장 재구성 (주어-목적어 순서 변경)
            'sentence_restructure': [
                (r'(\w+가) (.+) (\w+을|를) (.+다)', r'\3 \1 \2 \4'),
                (r'(\w+은|는) (.+) (\w+을|를) (.+다)', r'\3 \1 \2 \4'),
            ],
            
            # 관형절 변환
            'relative_clause': [
                (r'(\w+)한 (\w+)', r'\2는 \1하다'),
                (r'(\w+)된 (\w+)', r'\2는 \1되다'),
            ]
        }
        
        # 2. 문체적 변형
        self.style_variations = {
            'formal_to_casual': {
                '습니다': ['어요', '아요', '에요'],
                '했습니다': ['했어요', '했네요', '했답니다'],
                '입니다': ['이에요', '예요', '이랍니다'],
                '것입니다': ['거예요', '것이에요', '겁니다']
            },
            
            'declarative_to_interrogative': {
                '다.': ['까?', '나?', '지?'],
                '어요.': ['어요?', '나요?'],
                '습니다.': ['습니까?', '나요?']
            },
            
            'add_discourse_markers': [
                '그런데', '그러니까', '사실', '물론', '하지만',
                '그래서', '따라서', '즉', '예를 들어', '한편'
            ]
        }
        
        # 3. 의미 보존 패러프레이즈
        self.paraphrase_patterns = {
            # 연결어구 다양화
            'connectors': {
                '그리고': ['또한', '더욱이', '게다가', '아울러', '한편'],
                '하지만': ['그러나', '다만', '그런데', '반면에', '그렇지만'],
                '그래서': ['따라서', '그러므로', '때문에', '그리하여', '이에'],
                '왜냐하면': ['이는', '그 이유는', '~기 때문에', '~으로 인해']
            },
            
            # 강조 표현 다양화
            'emphasis': {
                '매우': ['아주', '무척', '상당히', '대단히', '꽤', '제법'],
                '정말': ['참으로', '진짜로', '실로', '과연', '정녕'],
                '항상': ['언제나', '늘', '계속', '지속적으로', '끊임없이']
            },
            
            # 관용적 표현 변형
            'idiomatic': {
                '중요하다': ['핵심적이다', '필수적이다', '결정적이다', '주요하다'],
                '어렵다': ['힘들다', '곤란하다', '버겁다', '난해하다'],
                '쉽다': ['간단하다', '용이하다', '손쉽다', '수월하다']
            }
        }
        
        # 4. 길이 다양성을 위한 확장/축약 패턴
        self.length_patterns = {
            'expansion': {
                # 단문을 복문으로
                'simple_to_complex': [
                    (r'(\w+다)\. (\w+다)', r'\1고, \2'),
                    (r'(\w+다)\. (그\w+)', r'\1며, \2'),
                ],
                # 부연설명 추가
                'add_explanation': [
                    '즉, ', '다시 말해, ', '구체적으로, ', '예를 들어, '
                ]
            },
            
            'compression': {
                # 복문을 단문으로
                'complex_to_simple': [
                    (r'(\w+)하고, (\w+다)', r'\1하여 \2'),
                    (r'(\w+)며, (\w+다)', r'\1하고 \2'),
                ],
                # 불필요한 수식어 제거
                'remove_modifiers': ['매우', '정말', '아주', '꽤', '상당히']
            }
        }
    
    def calculate_text_signature(self, text: str) -> str:
        """텍스트 시그니처 생성 (다양성 측정용)"""
        # 의미 핵심 단어만 추출
        core_words = re.findall(r'\b\w{2,}\b', text)
        # 조사와 어미 제거된 핵심 단어들로 시그니처 생성
        signature = ''.join(sorted(core_words[:20]))  # 상위 20개 단어
        return hashlib.md5(signature.encode()).hexdigest()[:8]
    
    def measure_lexical_diversity(self, texts: List[str]) -> float:
        """어휘 다양성 측정"""
        all_words = []
        for text in texts:
            # ToW 토큰 제외하고 단어 추출
            clean_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
            words = re.findall(r'\b\w{2,}\b', clean_text)
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        return (unique_words / total_words) * 100
    
    def measure_structural_diversity(self, texts: List[str]) -> float:
        """구조적 다양성 측정"""
        structural_patterns = []
        
        for text in texts:
            clean_text = re.sub(r'<ToW>.*?</ToW>', '', text, flags=re.DOTALL)
            
            # 문장 구조 패턴 추출
            patterns = []
            
            # 주어-서술어 패턴
            if re.search(r'\w+이 \w+다', clean_text):
                patterns.append('subject_predicate')
            if re.search(r'\w+가 \w+다', clean_text):
                patterns.append('subject_predicate_casual')
            
            # 목적어 포함 패턴
            if re.search(r'\w+을 \w+다', clean_text):
                patterns.append('object_verb')
            if re.search(r'\w+를 \w+다', clean_text):
                patterns.append('object_verb_casual')
            
            # 연결어미 패턴
            if re.search(r'\w+고', clean_text):
                patterns.append('connective_go')
            if re.search(r'\w+며', clean_text):
                patterns.append('connective_myeo')
            
            structural_patterns.append(tuple(sorted(patterns)))
        
        unique_patterns = len(set(structural_patterns))
        total_patterns = len(structural_patterns)
        
        return (unique_patterns / max(total_patterns, 1)) * 100
    
    def measure_semantic_diversity(self, texts: List[str]) -> float:
        """의미적 다양성 측정 (텍스트 시그니처 기반)"""
        signatures = []
        for text in texts:
            sig = self.calculate_text_signature(text)
            signatures.append(sig)
        
        unique_signatures = len(set(signatures))
        total_signatures = len(signatures)
        
        return (unique_signatures / max(total_signatures, 1)) * 100
    
    def measure_length_diversity(self, texts: List[str]) -> float:
        """길이 다양성 측정"""
        lengths = [len(text) for text in texts]
        if not lengths:
            return 0.0
        
        # 표준편차 기반 다양성 점수
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # 상대적 표준편차를 백분율로 변환
        coefficient_of_variation = (std_length / max(mean_length, 1)) * 100
        
        # 0-100 범위로 정규화
        return min(coefficient_of_variation, 100.0)
    
    def calculate_overall_diversity(self, metrics: DiversityMetrics) -> float:
        """종합 다양성 점수 계산"""
        weights = {
            'lexical': 0.25,
            'structural': 0.30,
            'semantic': 0.35,
            'length': 0.10
        }
        
        weighted_score = (
            metrics.lexical_diversity * weights['lexical'] +
            metrics.structural_diversity * weights['structural'] +
            metrics.semantic_diversity * weights['semantic'] +
            metrics.length_diversity * weights['length']
        )
        
        return weighted_score
    
    def apply_structural_variations(self, text: str) -> List[str]:
        """구조적 변형 적용"""
        variants = []
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        # 각 구조적 패턴 적용
        for pattern_type, patterns in self.structural_patterns.items():
            for pattern, replacement in patterns:
                if re.search(pattern, base_text):
                    modified = re.sub(pattern, replacement, base_text, count=1)
                    
                    # ToW 토큰 재삽입
                    for start_pos, end_pos, content in reversed(tow_tokens):
                        relative_pos = start_pos / len(text)
                        insert_pos = int(len(modified) * relative_pos)
                        tow_token = f"<ToW>{content}</ToW>"
                        modified = modified[:insert_pos] + tow_token + modified[insert_pos:]
                    
                    if modified != text and modified not in variants:
                        variants.append(modified)
                        break
        
        return variants
    
    def apply_style_variations(self, text: str) -> List[str]:
        """문체적 변형 적용"""
        variants = []
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        # 격식체 → 비격식체
        modified = base_text
        for formal, casuals in self.style_variations['formal_to_casual'].items():
            if formal in modified:
                casual = random.choice(casuals)
                modified = modified.replace(formal, casual, 1)
                break
        
        # 평서문 → 의문문
        if modified == base_text:  # 이전에 변경되지 않았다면
            for decl, interr_list in self.style_variations['declarative_to_interrogative'].items():
                if decl in modified:
                    interr = random.choice(interr_list)
                    modified = modified.replace(decl, interr, 1)
                    break
        
        # 담화 표지어 추가
        if modified == base_text:  # 이전에 변경되지 않았다면
            marker = random.choice(self.style_variations['add_discourse_markers'])
            sentences = modified.split('.')
            if len(sentences) > 1:
                # 두 번째 문장 앞에 담화 표지어 추가
                sentences[1] = f" {marker} " + sentences[1].strip()
                modified = '.'.join(sentences)
        
        # ToW 토큰 재삽입
        if modified != base_text:
            for start_pos, end_pos, content in reversed(tow_tokens):
                relative_pos = start_pos / len(text)
                insert_pos = int(len(modified) * relative_pos)
                tow_token = f"<ToW>{content}</ToW>"
                modified = modified[:insert_pos] + tow_token + modified[insert_pos:]
            
            variants.append(modified)
        
        return variants
    
    def apply_semantic_paraphrasing(self, text: str) -> List[str]:
        """의미 보존 패러프레이즈 적용"""
        variants = []
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        # 연결어구 다양화
        modified = base_text
        for category, replacements in self.paraphrase_patterns.items():
            for original, alternatives in replacements.items():
                if original in modified:
                    alternative = random.choice(alternatives)
                    modified = modified.replace(original, alternative, 1)
                    break
            if modified != base_text:
                break
        
        # ToW 토큰 재삽입
        if modified != base_text:
            for start_pos, end_pos, content in reversed(tow_tokens):
                relative_pos = start_pos / len(text)
                insert_pos = int(len(modified) * relative_pos)
                tow_token = f"<ToW>{content}</ToW>"
                modified = modified[:insert_pos] + tow_token + modified[insert_pos:]
            
            variants.append(modified)
        
        return variants
    
    def apply_length_variations(self, text: str) -> List[str]:
        """길이 다양성 변형 적용"""
        variants = []
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        # 확장 변형 (50% 확률)
        if random.random() < 0.5:
            modified = base_text
            
            # 단문을 복문으로
            for pattern, replacement in self.length_patterns['expansion']['simple_to_complex']:
                if re.search(pattern, modified):
                    modified = re.sub(pattern, replacement, modified, count=1)
                    break
            
            # 부연설명 추가
            if modified == base_text:
                explanation = random.choice(self.length_patterns['expansion']['add_explanation'])
                sentences = modified.split('.')
                if len(sentences) > 1:
                    sentences[1] = f" {explanation}" + sentences[1].strip()
                    modified = '.'.join(sentences)
        
        # 축약 변형 (50% 확률)
        else:
            modified = base_text
            
            # 복문을 단문으로
            for pattern, replacement in self.length_patterns['compression']['complex_to_simple']:
                if re.search(pattern, modified):
                    modified = re.sub(pattern, replacement, modified, count=1)
                    break
            
            # 수식어 제거
            if modified == base_text:
                for modifier in self.length_patterns['compression']['remove_modifiers']:
                    if modifier in modified:
                        modified = modified.replace(f" {modifier} ", " ", 1)
                        break
        
        # ToW 토큰 재삽입
        if modified != base_text:
            for start_pos, end_pos, content in reversed(tow_tokens):
                relative_pos = start_pos / len(text)
                insert_pos = int(len(modified) * relative_pos)
                tow_token = f"<ToW>{content}</ToW>"
                modified = modified[:insert_pos] + tow_token + modified[insert_pos:]
            
            variants.append(modified)
        
        return variants
    
    def generate_high_diversity_variants(self, text: str, target_count: int = 5) -> List[str]:
        """고다양성 변형 생성"""
        all_variants = []
        
        # 각 변형 기법 적용
        structural_variants = self.apply_structural_variations(text)
        style_variants = self.apply_style_variations(text)
        semantic_variants = self.apply_semantic_paraphrasing(text)
        length_variants = self.apply_length_variations(text)
        
        # 모든 변형 수집
        all_variants.extend(structural_variants)
        all_variants.extend(style_variants)
        all_variants.extend(semantic_variants)
        all_variants.extend(length_variants)
        
        # 중복 제거 (의미 시그니처 기반)
        unique_variants = []
        seen_signatures = {self.calculate_text_signature(text)}
        
        for variant in all_variants:
            signature = self.calculate_text_signature(variant)
            if signature not in seen_signatures:
                unique_variants.append(variant)
                seen_signatures.add(signature)
        
        # 목표 개수만큼 반환
        return unique_variants[:target_count]
    
    def measure_diversity_improvement(self, original_variants: List[str], enhanced_variants: List[str]) -> Dict:
        """다양성 개선 효과 측정"""
        
        # 원본 다양성 측정
        original_metrics = DiversityMetrics(
            lexical_diversity=self.measure_lexical_diversity(original_variants),
            structural_diversity=self.measure_structural_diversity(original_variants),
            semantic_diversity=self.measure_semantic_diversity(original_variants),
            length_diversity=self.measure_length_diversity(original_variants)
        )
        original_metrics.overall_diversity = self.calculate_overall_diversity(original_metrics)
        
        # 개선된 다양성 측정
        enhanced_metrics = DiversityMetrics(
            lexical_diversity=self.measure_lexical_diversity(enhanced_variants),
            structural_diversity=self.measure_structural_diversity(enhanced_variants),
            semantic_diversity=self.measure_semantic_diversity(enhanced_variants),
            length_diversity=self.measure_length_diversity(enhanced_variants)
        )
        enhanced_metrics.overall_diversity = self.calculate_overall_diversity(enhanced_metrics)
        
        # 개선율 계산
        improvement = {
            'original_diversity': original_metrics.overall_diversity,
            'enhanced_diversity': enhanced_metrics.overall_diversity,
            'improvement_rate': enhanced_metrics.overall_diversity - original_metrics.overall_diversity,
            'detailed_improvements': {
                'lexical': enhanced_metrics.lexical_diversity - original_metrics.lexical_diversity,
                'structural': enhanced_metrics.structural_diversity - original_metrics.structural_diversity,
                'semantic': enhanced_metrics.semantic_diversity - original_metrics.semantic_diversity,
                'length': enhanced_metrics.length_diversity - original_metrics.length_diversity
            }
        }
        
        return improvement

def main():
    """테스트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터
    test_text = """학교에서 <ToW>This requires understanding Korean honorific complexity</ToW> 선생님이 학생들을 가르친다. 
    학생들은 매우 열심히 공부한다. 그리고 시험을 잘 본다."""
    
    engine = AdvancedDiversityEngine()
    
    print("🌈 Enhanced Diversity Engine 테스트")
    print("=" * 50)
    print(f"원본 텍스트: {test_text[:100]}...")
    
    # 고다양성 변형 생성
    variants = engine.generate_high_diversity_variants(test_text, 5)
    
    print(f"\n생성된 변형 {len(variants)}개:")
    for i, variant in enumerate(variants, 1):
        print(f"{i}. {variant[:100]}...")
    
    # 다양성 측정
    original_variants = [test_text] * 10  # 시뮬레이션
    enhanced_variants = variants + [test_text]
    
    improvement = engine.measure_diversity_improvement(original_variants, enhanced_variants)
    
    print(f"\n📊 다양성 개선 결과:")
    print(f"원본 다양성: {improvement['original_diversity']:.1f}%")
    print(f"개선된 다양성: {improvement['enhanced_diversity']:.1f}%")
    print(f"개선율: +{improvement['improvement_rate']:.1f}%")
    
    for aspect, improvement_val in improvement['detailed_improvements'].items():
        print(f"  {aspect}: +{improvement_val:.1f}%")

if __name__ == "__main__":
    main()