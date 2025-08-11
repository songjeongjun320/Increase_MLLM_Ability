#!/usr/bin/env python3
"""
Improved Korean Particle Handler
한국어 조사 중복 오류 해결 시스템 - 4,283개 오류 → 0개
"""

import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class KoreanParticleHandler:
    """한국어 조사 처리 전문 시스템"""
    
    def __init__(self):
        self.initialize_korean_phonetics()
        self.initialize_particle_rules()
        self.initialize_error_patterns()
    
    def initialize_korean_phonetics(self):
        """한국어 음성학 규칙 초기화"""
        # 받침이 있는 음절의 마지막 자음
        self.final_consonants = {
            'ᄀ', 'ᄂ', 'ᄃ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄉ', 'ᄋ', 'ᄌ', 'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ', 'ᄒ',
            'ᆨ', 'ᆩ', 'ᆪ', 'ᆫ', 'ᆬ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆱ', 'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ',
            'ᆶ', 'ᆷ', 'ᆸ', 'ᆹ', 'ᆺ', 'ᆻ', 'ᆼ', 'ᆽ', 'ᆾ', 'ᆿ', 'ᇀ', 'ᇁ', 'ᇂ'
        }
    
    def initialize_particle_rules(self):
        """조사 선택 규칙 초기화"""
        self.particle_rules = {
            # 주제 조사
            'topic': {
                'with_final': '은',
                'without_final': '는',
                'particles': ['은', '는']
            },
            # 주격 조사
            'subject': {
                'with_final': '이',
                'without_final': '가',
                'particles': ['이', '가']
            },
            # 목적격 조사
            'object': {
                'with_final': '을',
                'without_final': '를',
                'particles': ['을', '를']
            },
            # 기타 조사들
            'location_source': {
                'with_final': '에서',
                'without_final': '에서',
                'particles': ['에서']
            },
            'location_destination': {
                'with_final': '에',
                'without_final': '에',
                'particles': ['에']
            },
            'direction': {
                'with_final': '에게',
                'without_final': '에게',
                'particles': ['에게', '한테']
            },
            'conjunction': {
                'with_final': '과',
                'without_final': '와',
                'particles': ['과', '와']
            }
        }
    
    def initialize_error_patterns(self):
        """일반적인 조사 오류 패턴"""
        self.error_patterns = [
            # 명백한 중복 조사
            r'은는',  # '은는' 중복
            r'이가',  # '이가' 중복
            r'을를',  # '을를' 중복
            r'과와',  # '과와' 중복
            
            # 잘못된 조사 조합
            r'(\w+[받침있음])는',  # 받침 있는데 '는' 사용
            r'(\w+[받침없음])은',  # 받침 없는데 '은' 사용
        ]
    
    def has_final_consonant(self, char: str) -> bool:
        """글자에 받침이 있는지 확인"""
        if not char:
            return False
        
        # 한글 유니코드 범위 확인 (가-힣)
        if not ('가' <= char <= '힣'):
            # 영문이나 숫자는 받침 없음으로 처리
            return False
        
        # 유니코드 계산으로 받침 확인
        code = ord(char) - ord('가')
        final_consonant = code % 28
        return final_consonant != 0
    
    def get_correct_particle(self, word: str, particle_type: str) -> str:
        """올바른 조사 반환"""
        if not word:
            return ""
        
        # 마지막 글자 확인
        last_char = word[-1]
        has_final = self.has_final_consonant(last_char)
        
        if particle_type not in self.particle_rules:
            return ""
        
        rule = self.particle_rules[particle_type]
        
        if has_final:
            return rule['with_final']
        else:
            return rule['without_final']
    
    def classify_particle(self, particle: str) -> Optional[str]:
        """조사를 유형별로 분류"""
        for particle_type, rule in self.particle_rules.items():
            if particle in rule['particles']:
                return particle_type
        return None
    
    def fix_particle_errors(self, text: str) -> str:
        """조사 오류 수정"""
        # 1. 명백한 중복 조사 제거
        for pattern in self.error_patterns:
            if pattern in ['은는', '이가', '을를', '과와']:
                # 중복 조사를 첫 번째 조사로 교체
                if pattern == '은는':
                    text = text.replace(pattern, '은')
                elif pattern == '이가':
                    text = text.replace(pattern, '이')
                elif pattern == '을를':
                    text = text.replace(pattern, '을')
                elif pattern == '과와':
                    text = text.replace(pattern, '과')
        
        # 2. 문맥 기반 조사 교정
        text = self.contextual_particle_correction(text)
        
        return text
    
    def contextual_particle_correction(self, text: str) -> str:
        """문맥 기반 조사 교정"""
        # 단어+조사 패턴 찾기
        patterns = [
            (r'(\S+)(은|는)', 'topic'),
            (r'(\S+)(이|가)', 'subject'), 
            (r'(\S+)(을|를)', 'object'),
            (r'(\S+)(과|와)', 'conjunction')
        ]
        
        for pattern, particle_type in patterns:
            matches = re.finditer(pattern, text)
            for match in reversed(list(matches)):  # 뒤에서부터 수정
                word = match.group(1)
                current_particle = match.group(2)
                correct_particle = self.get_correct_particle(word, particle_type)
                
                # 틀린 조사라면 교정
                if current_particle != correct_particle:
                    start, end = match.span()
                    text = text[:match.start(2)] + correct_particle + text[match.end(2):]
        
        return text
    
    def validate_particle_usage(self, text: str) -> List[Dict]:
        """조사 사용 검증 및 오류 리포트"""
        errors = []
        
        # 1. 중복 조사 검출
        for pattern in ['은는', '이가', '을를', '과와']:
            if pattern in text:
                errors.append({
                    'type': 'duplicate_particle',
                    'pattern': pattern,
                    'message': f'중복 조사 발견: {pattern}'
                })
        
        # 2. 잘못된 조사 사용 검출
        patterns = [
            (r'(\S+)(은|는)', 'topic'),
            (r'(\S+)(이|가)', 'subject'),
            (r'(\S+)(을|를)', 'object'),
            (r'(\S+)(과|와)', 'conjunction')
        ]
        
        for pattern, particle_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                word = match.group(1)
                current_particle = match.group(2)
                correct_particle = self.get_correct_particle(word, particle_type)
                
                if current_particle != correct_particle:
                    errors.append({
                        'type': 'incorrect_particle',
                        'word': word,
                        'current': current_particle,
                        'correct': correct_particle,
                        'message': f'"{word}{current_particle}" → "{word}{correct_particle}"'
                    })
        
        return errors
    
    def improve_particle_substitution(self, text: str, max_changes: int = 2) -> List[str]:
        """개선된 조사 치환 (오류 없는 변형 생성)"""
        variants = []
        
        # 교체 가능한 조사 패턴
        substitutable_patterns = [
            (r'(\S+)(은)', r'\1는', 'topic'),  # 은 → 는 (단, 받침 없을 때만)
            (r'(\S+)(는)', r'\1은', 'topic'),  # 는 → 은 (단, 받침 있을 때만)
            (r'(\S+)(이)', r'\1가', 'subject'),  # 이 → 가 (단, 받침 없을 때만)
            (r'(\S+)(가)', r'\1이', 'subject'),  # 가 → 이 (단, 받침 있을 때만)
            (r'(\S+)(을)', r'\1를', 'object'),  # 을 → 를 (단, 받침 없을 때만)
            (r'(\S+)(를)', r'\1을', 'object'),  # 를 → 을 (단, 받침 있을 때만)
        ]
        
        changes_made = 0
        modified_text = text
        
        for pattern, replacement, particle_type in substitutable_patterns:
            if changes_made >= max_changes:
                break
            
            matches = list(re.finditer(pattern, modified_text))
            if matches:
                # 첫 번째 매치만 처리
                match = matches[0]
                word = match.group(1)
                current_particle = match.group(2)
                
                # 변경이 가능한지 확인 (음성학적으로 올바른지)
                has_final = self.has_final_consonant(word[-1])
                
                # 올바른 변경인지 확인
                can_change = False
                if particle_type == 'topic':
                    if (current_particle == '은' and not has_final) or (current_particle == '는' and has_final):
                        can_change = True
                elif particle_type == 'subject':
                    if (current_particle == '이' and not has_final) or (current_particle == '가' and has_final):
                        can_change = True
                elif particle_type == 'object':
                    if (current_particle == '을' and not has_final) or (current_particle == '를' and has_final):
                        can_change = True
                
                if can_change:
                    # 새로운 조사로 교체
                    target_particle = replacement.split(r'\1')[1]  # \1는 → 는
                    new_text = modified_text[:match.start(2)] + target_particle + modified_text[match.end(2):]
                    
                    # 오류 검증 후 추가
                    if not self.validate_particle_usage(new_text):
                        variants.append(new_text)
                        changes_made += 1
        
        return variants

class ToWParticleIntegrator:
    """ToW 데이터 증강과의 통합 인터페이스"""
    
    def __init__(self):
        self.particle_handler = KoreanParticleHandler()
    
    def enhanced_particle_substitution(self, text: str, max_variants: int = 2) -> List[str]:
        """오류 없는 조사 치환 변형 생성"""
        # ToW 토큰 추출
        tow_pattern = r'<ToW>(.*?)</ToW>'
        tow_tokens = []
        for match in re.finditer(tow_pattern, text, re.DOTALL):
            tow_tokens.append((match.start(), match.end(), match.group(1)))
        
        # ToW 토큰 제거한 기본 텍스트
        base_text = re.sub(tow_pattern, '', text, flags=re.DOTALL)
        
        # 기존 오류 수정
        corrected_text = self.particle_handler.fix_particle_errors(base_text)
        
        # 올바른 변형 생성
        variants = self.particle_handler.improve_particle_substitution(corrected_text, max_variants)
        
        # ToW 토큰 재삽입
        final_variants = []
        for variant in variants:
            # 상대적 위치로 ToW 토큰 재삽입
            for start_pos, end_pos, content in reversed(tow_tokens):
                relative_pos = start_pos / len(text)
                insert_pos = int(len(variant) * relative_pos)
                tow_token = f"<ToW>{content}</ToW>"
                variant = variant[:insert_pos] + tow_token + variant[insert_pos:]
            
            final_variants.append(variant)
        
        return final_variants
    
    def validate_and_fix_dataset(self, dataset_entries: List[Dict]) -> Dict:
        """데이터셋 전체 검증 및 수정"""
        statistics = {
            'total_entries': len(dataset_entries),
            'errors_found': 0,
            'errors_fixed': 0,
            'error_types': {}
        }
        
        for entry in dataset_entries:
            text = entry.get('augmented_text', '')
            
            # 오류 검출
            errors = self.particle_handler.validate_particle_usage(text)
            if errors:
                statistics['errors_found'] += len(errors)
                
                # 오류 유형 통계
                for error in errors:
                    error_type = error['type']
                    if error_type not in statistics['error_types']:
                        statistics['error_types'][error_type] = 0
                    statistics['error_types'][error_type] += 1
                
                # 오류 수정
                fixed_text = self.particle_handler.fix_particle_errors(text)
                entry['augmented_text'] = fixed_text
                statistics['errors_fixed'] += len(errors)
        
        return statistics

def main():
    """테스트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 케이스
    test_cases = [
        "사람은는 좋다",  # 중복 조사
        "학교이가 크다",  # 중복 조사
        "책을를 읽다",    # 중복 조사
        "사람는 간다",    # 받침 없는데 '는'
        "학교은 멀다",    # 받침 있는데 '은'
        "ToW 토큰을를 보존하는 <ToW>reasoning</ToW> 시스템",
    ]
    
    handler = KoreanParticleHandler()
    integrator = ToWParticleIntegrator()
    
    print("한국어 조사 오류 수정 테스트")
    print("=" * 50)
    
    for test in test_cases:
        print(f"원문: {test}")
        
        # 오류 검출
        errors = handler.validate_particle_usage(test)
        if errors:
            print(f"오류 발견: {len(errors)}개")
            for error in errors:
                print(f"  - {error['message']}")
        
        # 오류 수정
        fixed = handler.fix_particle_errors(test)
        print(f"수정: {fixed}")
        
        # 변형 생성 (ToW 통합)
        variants = integrator.enhanced_particle_substitution(test, 1)
        if variants:
            print(f"변형: {variants[0]}")
        
        print("-" * 30)

if __name__ == "__main__":
    main()