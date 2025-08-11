#!/usr/bin/env python3
"""
Advanced Augmentation Techniques for Korean ToW Data
추가 증강 기법: 구문 변환, 문체 변경, 도메인별 특화 증강
"""

import json
import re
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DomainConfig:
    """도메인별 증강 설정"""
    domain_name: str
    specialized_vocab: Dict[str, List[str]]
    syntax_patterns: List[Tuple[str, str]]
    style_preferences: Dict[str, float]  # 문체 선호도

class AdvancedAugmentationEngine:
    """고급 증강 기법 엔진"""
    
    def __init__(self):
        self.initialize_syntactic_patterns()
        self.initialize_stylistic_patterns()
        self.initialize_domain_configs()
        self.initialize_mathematical_patterns()
    
    def initialize_syntactic_patterns(self):
        """구문 변환 패턴 초기화"""
        
        # 1. 능동/수동태 변환
        self.voice_patterns = {
            'active_to_passive': [
                (r'(\w+)가 (\w+)을 (\w+)한다', r'\2이 \1에 의해 \3되다'),
                (r'(\w+)이 (\w+)를 (\w+)한다', r'\2가 \1에 의해 \3되다'),
                (r'(\w+)가 (\w+)을 (\w+)했다', r'\2이 \1에 의해 \3되었다'),
                (r'(\w+)이 (\w+)를 (\w+)했다', r'\2가 \1에 의해 \3되었다')
            ],
            'passive_to_active': [
                (r'(\w+)이 (\w+)에 의해 (\w+)되다', r'\2가 \1을 \3한다'),
                (r'(\w+)가 (\w+)에 의해 (\w+)되다', r'\2이 \1를 \3한다'),
                (r'(\w+)이 (\w+)에 의해 (\w+)되었다', r'\2가 \1을 \3했다'),
                (r'(\w+)가 (\w+)에 의해 (\w+)되었다', r'\2이 \1를 \3했다')
            ]
        }
        
        # 2. 문장 성분 재배열
        self.constituent_reordering = {
            'sov_to_osv': [
                (r'(\w+가|이) (\w+을|를) (\w+다)', r'\2 \1 \3'),
                (r'(\w+는|은) (\w+을|를) (\w+다)', r'\2 \1 \3')
            ],
            'topic_fronting': [
                (r'(\w+에서) (\w+가|이) (\w+다)', r'\1는 \2 \3'),
                (r'(\w+에게) (\w+가|이) (\w+다)', r'\1는 \2 \3')
            ],
            'temporal_fronting': [
                (r'(\w+가|이) (어제|오늘|내일) (\w+다)', r'\2 \1 \3'),
                (r'(\w+는|은) (어제|오늘|내일) (\w+다)', r'\2 \1 \3')
            ]
        }
        
        # 3. 절 변환 (단순절 ↔ 복절)
        self.clause_transformation = {
            'simple_to_complex': [
                (r'(\w+다)\. (\w+다)', r'\1고, \2'),
                (r'(\w+다)\. (\w+다)', r'\1어서 \2'),
                (r'(\w+다)\. (\w+다)', r'\1지만 \2')
            ],
            'complex_to_simple': [
                (r'(\w+)고, (\w+다)', r'\1다. \2'),
                (r'(\w+)어서 (\w+다)', r'\1다. \2'),
                (r'(\w+)지만 (\w+다)', r'\1다. 하지만 \2')
            ],
            'relative_clause': [
                (r'(\w+)는 (\w+)이다', r'\2인 \1'),
                (r'(\w+)은 (\w+)이다', r'\2인 \1')
            ]
        }
    
    def initialize_stylistic_patterns(self):
        """문체 변환 패턴 초기화"""
        
        # 1. 격식도 변환
        self.formality_patterns = {
            'formal_academic': {
                '이다': '이다', '한다': '한다', '된다': '된다',
                # 학술체 특징
                'transforms': {
                    '그리고': '그리고', '하지만': '그러나', '때문에': '으로 인해',
                    '매우': '상당히', '정말': '실로', '아주': '대단히'
                }
            },
            'semi_formal': {
                '이다': '입니다', '한다': '합니다', '된다': '됩니다',
                'transforms': {
                    '그리고': '그리고', '하지만': '그렇지만', '때문에': '때문입니다',
                    '매우': '매우', '정말': '정말', '아주': '아주'
                }
            },
            'informal_polite': {
                '이다': '이에요', '한다': '해요', '된다': '돼요',
                'transforms': {
                    '그리고': '그리고', '하지만': '그런데', '때문에': '때문이에요',
                    '매우': '아주', '정말': '진짜', '아주': '엄청'
                }
            },
            'casual': {
                '이다': '이야', '한다': '해', '된다': '돼',
                'transforms': {
                    '그리고': '그리고', '하지만': '근데', '때문에': '때문이야',
                    '매우': '엄청', '정말': '진짜', '아주': '완전'
                }
            }
        }
        
        # 2. 양상성 표현 (modality)
        self.modality_patterns = {
            'certainty': {
                'high': ['반드시', '확실히', '틀림없이', '분명히'],
                'medium': ['아마', '대개', '보통', '일반적으로'],
                'low': ['혹시', '아마도', '가능하면', '경우에 따라']
            },
            'possibility': {
                'high': ['할 수 있다', '가능하다', '될 수 있다'],
                'medium': ['할지도 모른다', '할 가능성이 있다'],
                'low': ['하기 어렵다', '힘들 것 같다', '어려울 것이다']
            },
            'necessity': {
                'strong': ['해야 한다', '하지 않으면 안 된다', '필수다'],
                'medium': ['하는 것이 좋다', '권장된다', '바람직하다'],
                'weak': ['해도 된다', '할 수도 있다', '고려해볼 만하다']
            }
        }
        
        # 3. 담화 표지어와 연결어구
        self.discourse_markers = {
            'sequence': ['먼저', '다음으로', '그 다음에', '마지막으로', '결론적으로'],
            'contrast': ['반면에', '한편', '그와 달리', '오히려', '역으로'],
            'addition': ['또한', '더욱이', '게다가', '뿐만 아니라', '아울러'],
            'explanation': ['즉', '다시 말해', '구체적으로', '예를 들어', '바꾸어 말하면'],
            'emphasis': ['특히', '무엇보다도', '중요한 것은', '주목할 점은', '강조하면']
        }
    
    def initialize_domain_configs(self):
        """도메인별 설정 초기화"""
        
        # 수학 도메인
        self.math_domain = DomainConfig(
            domain_name="mathematics",
            specialized_vocab={
                '계산': ['연산', '산출', '산정'],
                '방법': ['방식', '기법', '절차', '알고리즘'],
                '결과': ['답', '해', '값', '결론'],
                '문제': ['과제', '예제', '연습문제', '문항'],
                '공식': ['식', '정리', '법칙', '원리'],
                '증명': ['논증', '입증', '확인', '검증'],
                '그래프': ['도표', '도형', '차트', '함수'],
                '변수': ['미지수', '인수', '파라미터', '요소']
            },
            syntax_patterns=[
                (r'(\w+)을 계산하다', r'\1을 구하다'),
                (r'(\w+)을 구하다', r'\1을 계산하다'),
                (r'(\w+)는 (\w+)이다', r'\1 = \2'),
                (r'만약 (\w+)라면', r'\1일 때'),
                (r'따라서 (\w+)', r'그러므로 \1'),
                (r'(\w+)로부터', r'\1에서')
            ],
            style_preferences={
                'formal_academic': 0.7,
                'semi_formal': 0.2,
                'informal_polite': 0.1
            }
        )
        
        # 과학 도메인
        self.science_domain = DomainConfig(
            domain_name="science",
            specialized_vocab={
                '실험': ['시험', '검증', '테스트', '분석'],
                '관찰': ['관측', '확인', '목격', '발견'],
                '현상': ['사건', '상황', '경우', '일'],
                '원인': ['이유', '근거', '요인', '배경'],
                '결과': ['효과', '산물', '영향', '변화'],
                '가설': ['추정', '예상', '추측', '이론'],
                '법칙': ['원리', '규칙', '정리', '공식'],
                '물질': ['재료', '성분', '요소', '화합물']
            },
            syntax_patterns=[
                (r'(\w+)을 관찰하다', r'\1을 살펴보다'),
                (r'(\w+)가 발생하다', r'\1가 일어나다'),
                (r'(\w+)에 의해', r'\1로 인해'),
                (r'실험 결과', r'시험 결과'),
                (r'이론적으로', r'원리상'),
                (r'실제로', r'현실적으로')
            ],
            style_preferences={
                'formal_academic': 0.8,
                'semi_formal': 0.2
            }
        )
        
        # 역사 도메인
        self.history_domain = DomainConfig(
            domain_name="history",
            specialized_vocab={
                '시대': ['시기', '연대', '때', '시절'],
                '왕조': ['조', '왕가', '황실', '정권'],
                '전쟁': ['싸움', '전투', '분쟁', '충돌'],
                '문화': ['문명', '전통', '풍속', '관습'],
                '정치': ['통치', '행정', '권력', '정부'],
                '경제': ['재정', '상업', '무역', '산업'],
                '종교': ['신앙', '믿음', '사상', '교리'],
                '사회': ['공동체', '계층', '신분', '집단']
            },
            syntax_patterns=[
                (r'(\w+) 시대에', r'\1 때에'),
                (r'(\w+)가 일어났다', r'\1가 발생했다'),
                (r'그 당시', r'그 시대에'),
                (r'역사적으로', r'과거에'),
                (r'전통적으로', r'옛부터'),
                (r'고대부터', r'옛날부터')
            ],
            style_preferences={
                'formal_academic': 0.6,
                'semi_formal': 0.3,
                'informal_polite': 0.1
            }
        )
    
    def initialize_mathematical_patterns(self):
        """수학 문제 특화 패턴"""
        
        self.math_expression_patterns = {
            # 수식 표현 다양화
            'equation_variants': {
                r'(\w+) = (\w+)': [r'\1는 \2이다', r'\1와 \2는 같다', r'\1는 \2와 같다'],
                r'(\w+) > (\w+)': [r'\1는 \2보다 크다', r'\1가 \2를 초과한다'],
                r'(\w+) < (\w+)': [r'\1는 \2보다 작다', r'\1가 \2에 미치지 못한다'],
                r'(\w+) + (\w+)': [r'\1와 \2의 합', r'\1에 \2를 더한 것'],
                r'(\w+) - (\w+)': [r'\1에서 \2를 뺀 것', r'\1와 \2의 차']
            },
            
            # 수학적 논리 표현
            'logical_expressions': {
                '만약 ~라면': ['~일 때', '~인 경우', '~라고 가정하면', '~라고 하면'],
                '따라서': ['그러므로', '고로', '즉', '결국'],
                '반대로': ['거꾸로', '반면에', '한편', '다시 말해'],
                '예를 들어': ['가령', '실례로', '구체적으로', '즉']
            },
            
            # 문제 해결 과정 표현
            'solution_process': {
                '첫 번째로': ['먼저', '우선', '시작으로', '첫째'],
                '두 번째로': ['다음으로', '그 다음', '둘째', '이어서'],
                '마지막으로': ['끝으로', '결론적으로', '최종적으로', '셋째'],
                '단계별로': ['순서대로', '차례로', '과정별로', '순차적으로']
            }
        }
    
    def apply_syntactic_transformation(self, text: str, transformation_type: str = 'random') -> List[str]:
        """구문 변환 적용"""
        variants = []
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        if transformation_type == 'random':
            # 모든 변환 중 랜덤 선택
            all_patterns = []
            for pattern_group in [self.voice_patterns, self.constituent_reordering, self.clause_transformation]:
                for patterns in pattern_group.values():
                    all_patterns.extend(patterns)
            
            random.shuffle(all_patterns)
            
            for pattern, replacement in all_patterns[:3]:  # 최대 3개 시도
                if re.search(pattern, base_text):
                    modified = re.sub(pattern, replacement, base_text, count=1)
                    
                    # ToW 토큰 재삽입
                    for start_pos, end_pos, content in reversed(tow_tokens):
                        relative_pos = start_pos / len(text)
                        insert_pos = int(len(modified) * relative_pos)
                        tow_token = f"<ToW>{content}</ToW>"
                        modified = modified[:insert_pos] + tow_token + modified[insert_pos:]
                    
                    if modified != text:
                        variants.append(modified)
                        break
        
        return variants
    
    def apply_stylistic_transformation(self, text: str, target_style: str = 'random') -> List[str]:
        """문체 변환 적용"""
        variants = []
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        if target_style == 'random':
            target_style = random.choice(list(self.formality_patterns.keys()))
        
        if target_style in self.formality_patterns:
            style_config = self.formality_patterns[target_style]
            modified = base_text
            
            # 기본 어미 변환
            for original, transformed in style_config.items():
                if original != 'transforms' and original in modified:
                    modified = modified.replace(original, transformed)
            
            # 추가 변환 적용
            if 'transforms' in style_config:
                for original, transformed in style_config['transforms'].items():
                    if original in modified:
                        modified = modified.replace(original, transformed)
            
            # 담화 표지어 추가 (20% 확률)
            if random.random() < 0.2:
                marker_type = random.choice(list(self.discourse_markers.keys()))
                marker = random.choice(self.discourse_markers[marker_type])
                sentences = modified.split('.')
                if len(sentences) > 1:
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
    
    def apply_domain_specialization(self, text: str, domain: str = 'mathematics') -> List[str]:
        """도메인별 특화 증강"""
        variants = []
        
        # 도메인 설정 선택
        domain_config = None
        if domain == 'mathematics':
            domain_config = self.math_domain
        elif domain == 'science':
            domain_config = self.science_domain
        elif domain == 'history':
            domain_config = self.history_domain
        
        if not domain_config:
            return variants
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        # 전문 용어 교체
        modified = base_text
        for original, alternatives in domain_config.specialized_vocab.items():
            if original in modified:
                alternative = random.choice(alternatives)
                modified = modified.replace(original, alternative, 1)
                break
        
        # 도메인 특화 구문 패턴 적용
        if modified == base_text:  # 용어 교체가 없었다면 구문 변환
            for pattern, replacement in domain_config.syntax_patterns:
                if re.search(pattern, modified):
                    modified = re.sub(pattern, replacement, modified, count=1)
                    break
        
        # 수학 도메인 특화 처리
        if domain == 'mathematics' and modified == base_text:
            # 수식 표현 다양화
            for pattern, replacements in self.math_expression_patterns['equation_variants'].items():
                if re.search(pattern, modified):
                    replacement = random.choice(replacements)
                    modified = re.sub(pattern, replacement, modified, count=1)
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
    
    def apply_modality_variation(self, text: str) -> List[str]:
        """양상성 표현 변형"""
        variants = []
        
        # ToW 토큰 보존
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tow_pattern, text, re.DOTALL)]
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        # 양상성 표현 추가/변경
        modality_type = random.choice(['certainty', 'possibility', 'necessity'])
        level = random.choice(['high', 'medium', 'low'])
        
        if modality_type in self.modality_patterns and level in self.modality_patterns[modality_type]:
            expressions = self.modality_patterns[modality_type][level]
            expression = random.choice(expressions)
            
            # 문장 시작에 양상성 표현 추가
            sentences = base_text.split('.')
            if sentences:
                first_sentence = sentences[0].strip()
                if first_sentence:
                    sentences[0] = f" {expression} " + first_sentence
                    modified = '.'.join(sentences)
                    
                    # ToW 토큰 재삽입
                    for start_pos, end_pos, content in reversed(tow_tokens):
                        relative_pos = start_pos / len(text)
                        insert_pos = int(len(modified) * relative_pos)
                        tow_token = f"<ToW>{content}</ToW>"
                        modified = modified[:insert_pos] + tow_token + modified[insert_pos:]
                    
                    variants.append(modified)
        
        return variants
    
    def generate_advanced_variants(self, text: str, domain: str = 'general') -> List[str]:
        """고급 변형 종합 생성"""
        all_variants = []
        
        # 1. 구문 변환
        syntactic_variants = self.apply_syntactic_transformation(text)
        all_variants.extend(syntactic_variants)
        
        # 2. 문체 변환
        stylistic_variants = self.apply_stylistic_transformation(text)
        all_variants.extend(stylistic_variants)
        
        # 3. 도메인 특화
        if domain in ['mathematics', 'science', 'history']:
            domain_variants = self.apply_domain_specialization(text, domain)
            all_variants.extend(domain_variants)
        
        # 4. 양상성 변형
        modality_variants = self.apply_modality_variation(text)
        all_variants.extend(modality_variants)
        
        # 중복 제거
        unique_variants = []
        for variant in all_variants:
            if variant not in unique_variants and variant != text:
                unique_variants.append(variant)
        
        return unique_variants[:6]  # 최대 6개 변형 반환

def main():
    """테스트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 케이스
    test_cases = [
        {
            'text': "학생이 수학 문제를 <ToW>Mathematical reasoning required</ToW> 푼다. 이 방법은 매우 효과적이다.",
            'domain': 'mathematics'
        },
        {
            'text': "과학자가 실험을 <ToW>Scientific methodology needed</ToW> 진행한다. 결과를 관찰한다.",
            'domain': 'science'
        },
        {
            'text': "조선 시대에 <ToW>Historical context important</ToW> 문화가 발달했다. 전통이 중요했다.",
            'domain': 'history'
        }
    ]
    
    engine = AdvancedAugmentationEngine()
    
    print("🚀 Advanced Augmentation Techniques 테스트")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        text = case['text']
        domain = case['domain']
        
        print(f"\n{i}. {domain.upper()} 도메인")
        print(f"원본: {text}")
        
        variants = engine.generate_advanced_variants(text, domain)
        
        print(f"생성된 고급 변형 {len(variants)}개:")
        for j, variant in enumerate(variants, 1):
            print(f"  {j}. {variant}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()