#!/usr/bin/env python3
"""
Integrated ToW Augmentation System
통합 ToW 증강 시스템 - 모든 개선사항 통합 적용
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict

# 개발한 모듈들 import
from improved_particle_handler import ToWParticleIntegrator
from enhanced_diversity_engine import AdvancedDiversityEngine
from advanced_augmentation_techniques import AdvancedAugmentationEngine

logger = logging.getLogger(__name__)

class IntegratedToWAugmentationSystem:
    """통합 ToW 증강 시스템"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.get_default_config()
        
        # 각 엔진 초기화
        self.particle_integrator = ToWParticleIntegrator()
        self.diversity_engine = AdvancedDiversityEngine()
        self.advanced_engine = AdvancedAugmentationEngine()
        
        # 통계 정보
        self.processing_stats = defaultdict(int)
        
    def get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'input_file': 'ToW_koconovel_augmented.json',
            'output_file': 'ToW_koconovel_enhanced_final.json',
            'batch_size': 100,
            'max_variants_per_entry': 8,
            'enable_particle_fixing': True,
            'enable_diversity_enhancement': True,
            'enable_advanced_techniques': True,
            'target_diversity_score': 65.0,
            'quality_threshold': 0.9,
            'domain_detection': True,
            'preserve_tow_integrity': True,
            'validation_enabled': True
        }
    
    def detect_domain(self, text: str) -> str:
        """도메인 자동 감지"""
        # 키워드 기반 도메인 분류
        domain_keywords = {
            'mathematics': ['수학', '계산', '공식', '방정식', '함수', '그래프', '증명', '문제', '답', '해', '값', '변수'],
            'science': ['실험', '관찰', '가설', '이론', '법칙', '현상', '물질', '화학', '물리', '생물', '과학'],
            'history': ['시대', '왕조', '전쟁', '문화', '정치', '경제', '조선', '고려', '역사', '전통', '과거'],
            'literature': ['소설', '시', '문학', '작품', '작가', '등장인물', '주제', '표현', '문체', '서술'],
            'general': []
        }
        
        text_lower = text.lower()
        domain_scores = defaultdict(int)
        
        for domain, keywords in domain_keywords.items():
            if domain == 'general':
                continue
            for keyword in keywords:
                if keyword in text_lower:
                    domain_scores[domain] += 1
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general'
    
    def process_single_entry(self, entry: Dict) -> List[Dict]:
        """단일 엔트리 처리"""
        original_text = entry.get('augmented_text', '')
        if not original_text:
            return [entry]
        
        results = [entry]  # 원본 포함
        generated_variants = []
        
        try:
            # 1. 조사 오류 수정 (기본적으로 적용)
            if self.config['enable_particle_fixing']:
                corrected_variants = self.particle_integrator.enhanced_particle_substitution(
                    original_text, max_variants=2
                )
                generated_variants.extend(corrected_variants)
                self.processing_stats['particle_corrections'] += len(corrected_variants)
            
            # 2. 다양성 증강
            if self.config['enable_diversity_enhancement']:
                diversity_variants = self.diversity_engine.generate_high_diversity_variants(
                    original_text, target_count=3
                )
                generated_variants.extend(diversity_variants)
                self.processing_stats['diversity_enhancements'] += len(diversity_variants)
            
            # 3. 고급 증강 기법 적용
            if self.config['enable_advanced_techniques']:
                domain = 'general'
                if self.config['domain_detection']:
                    domain = self.detect_domain(original_text)
                
                advanced_variants = self.advanced_engine.generate_advanced_variants(
                    original_text, domain=domain
                )
                generated_variants.extend(advanced_variants)
                self.processing_stats['advanced_techniques'] += len(advanced_variants)
                self.processing_stats[f'domain_{domain}'] += 1
            
            # 4. 변형 품질 검증 및 선별
            validated_variants = self.validate_and_select_variants(
                original_text, generated_variants, entry
            )
            
            # 5. 새로운 엔트리 생성
            for i, variant in enumerate(validated_variants):
                if len(results) >= self.config['max_variants_per_entry']:
                    break
                
                new_entry = self.create_enhanced_entry(entry, variant, i)
                results.append(new_entry)
                self.processing_stats['total_variants_created'] += 1
        
        except Exception as e:
            logger.warning(f"Entry processing failed for {entry.get('doc_id', 'unknown')}: {e}")
            self.processing_stats['processing_errors'] += 1
        
        return results
    
    def validate_and_select_variants(self, original_text: str, variants: List[str], entry: Dict) -> List[str]:
        """변형 품질 검증 및 선별"""
        if not variants:
            return []
        
        validated = []
        original_signature = self.diversity_engine.calculate_text_signature(original_text)
        
        for variant in variants:
            # 1. 기본 품질 검사
            if len(variant.strip()) < 10:  # 너무 짧음
                continue
            
            if variant == original_text:  # 원본과 동일
                continue
            
            # 2. ToW 토큰 보존 확인
            if self.config.get('preserve_tow_integrity', True):
                original_tow_count = len(self.diversity_engine.extract_tow_tokens(original_text))
                variant_tow_count = len(self.diversity_engine.extract_tow_tokens(variant))
                
                if original_tow_count != variant_tow_count:
                    self.processing_stats['tow_integrity_failures'] += 1
                    continue
            
            # 3. 다양성 확인 (중복 방지)
            variant_signature = self.diversity_engine.calculate_text_signature(variant)
            if variant_signature == original_signature:
                continue
            
            # 4. 한국어 조사 오류 확인
            particle_errors = self.particle_integrator.particle_handler.validate_particle_usage(variant)
            if len(particle_errors) > 2:  # 2개 초과 오류는 제외
                self.processing_stats['particle_validation_failures'] += 1
                continue
            
            validated.append(variant)
        
        return validated
    
    def create_enhanced_entry(self, original_entry: Dict, variant_text: str, variant_index: int) -> Dict:
        """개선된 엔트리 생성"""
        new_entry = original_entry.copy()
        
        # ID 업데이트
        base_id = original_entry.get('doc_id', f'entry_{int(time.time())}')
        new_entry['doc_id'] = f"{base_id}_enhanced_{variant_index + 1}"
        
        # 텍스트 업데이트
        new_entry['augmented_text'] = variant_text
        
        # ToW 메타데이터 재계산
        tow_tokens = self.diversity_engine.extract_tow_tokens(variant_text)
        new_entry['tow_count'] = len(tow_tokens)
        new_entry['tow_tokens'] = [f"<ToW>{token}</ToW>" for token in tow_tokens]
        
        # 개선 메타데이터 추가
        new_entry['enhancement_type'] = 'integrated_system'
        new_entry['original_doc_id'] = original_entry.get('doc_id')
        new_entry['enhancement_timestamp'] = int(time.time())
        
        # 품질 지표
        diversity_score = self.calculate_entry_diversity_score(variant_text, original_entry.get('augmented_text', ''))
        new_entry['diversity_score'] = diversity_score
        
        return new_entry
    
    def calculate_entry_diversity_score(self, variant_text: str, original_text: str) -> float:
        """개별 엔트리 다양성 점수 계산"""
        variants = [original_text, variant_text]
        
        lexical = self.diversity_engine.measure_lexical_diversity(variants)
        structural = self.diversity_engine.measure_structural_diversity(variants)
        semantic = self.diversity_engine.measure_semantic_diversity(variants)
        
        # 가중 평균
        diversity_score = (lexical * 0.3 + structural * 0.4 + semantic * 0.3)
        return round(diversity_score, 2)
    
    def process_dataset(self, input_file: str, output_file: str) -> Dict:
        """전체 데이터셋 처리"""
        logger.info(f"🚀 Starting integrated ToW augmentation system")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        
        # 데이터 로드
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        logger.info(f"Loaded {len(original_data)} entries")
        
        # 배치 처리
        batch_size = self.config['batch_size']
        all_enhanced_entries = []
        
        for i in tqdm(range(0, len(original_data), batch_size), desc="Processing batches"):
            batch = original_data[i:i + batch_size]
            batch_results = []
            
            for entry in batch:
                processed_entries = self.process_single_entry(entry)
                batch_results.extend(processed_entries)
            
            all_enhanced_entries.extend(batch_results)
            
            # 중간 진행 상황 로깅
            if i % (batch_size * 10) == 0 and i > 0:
                logger.info(f"Processed {i + batch_size} entries, generated {len(all_enhanced_entries)} total entries")
        
        # 최종 다양성 평가
        if self.config['validation_enabled']:
            final_diversity = self.evaluate_final_diversity(all_enhanced_entries)
            logger.info(f"Final diversity score: {final_diversity:.2f}%")
        
        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_enhanced_entries, f, ensure_ascii=False, indent=2)
        
        # 통계 리포트 생성
        stats_report = self.generate_processing_report(
            len(original_data), len(all_enhanced_entries)
        )
        
        logger.info("✅ Integrated augmentation completed!")
        return stats_report
    
    def evaluate_final_diversity(self, entries: List[Dict]) -> float:
        """최종 다양성 평가"""
        texts = [entry.get('augmented_text', '') for entry in entries]
        texts = [t for t in texts if t.strip()]  # 빈 텍스트 제외
        
        if not texts:
            return 0.0
        
        # 샘플링해서 평가 (메모리 효율성)
        if len(texts) > 1000:
            import random
            texts = random.sample(texts, 1000)
        
        lexical = self.diversity_engine.measure_lexical_diversity(texts)
        structural = self.diversity_engine.measure_structural_diversity(texts)
        semantic = self.diversity_engine.measure_semantic_diversity(texts)
        length = self.diversity_engine.measure_length_diversity(texts)
        
        overall = (lexical * 0.25 + structural * 0.30 + semantic * 0.35 + length * 0.10)
        return overall
    
    def generate_processing_report(self, original_count: int, final_count: int) -> Dict:
        """처리 리포트 생성"""
        report = {
            'processing_summary': {
                'original_entries': original_count,
                'final_entries': final_count,
                'enhancement_ratio': f"{final_count / original_count:.2f}x",
                'processing_time': time.time()
            },
            'enhancement_statistics': dict(self.processing_stats),
            'quality_metrics': {
                'particle_error_prevention': self.processing_stats.get('particle_corrections', 0),
                'diversity_improvements': self.processing_stats.get('diversity_enhancements', 0),
                'advanced_technique_applications': self.processing_stats.get('advanced_techniques', 0),
                'tow_integrity_maintained': 100 - (self.processing_stats.get('tow_integrity_failures', 0) / max(final_count, 1)) * 100
            },
            'domain_distribution': {
                key: value for key, value in self.processing_stats.items() 
                if key.startswith('domain_')
            },
            'error_summary': {
                'processing_errors': self.processing_stats.get('processing_errors', 0),
                'validation_failures': self.processing_stats.get('particle_validation_failures', 0) + self.processing_stats.get('tow_integrity_failures', 0)
            }
        }
        
        # 리포트 파일로 저장
        report_file = self.config['output_file'].replace('.json', '_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Processing report saved to {report_file}")
        
        return report
    
    def print_summary_report(self, report: Dict):
        """요약 리포트 출력"""
        print("\n" + "="*60)
        print("🎯 INTEGRATED ToW AUGMENTATION SYSTEM RESULTS")
        print("="*60)
        
        summary = report['processing_summary']
        print(f"📊 Dataset Scale:")
        print(f"  Original entries: {summary['original_entries']:,}")
        print(f"  Enhanced entries: {summary['final_entries']:,}")
        print(f"  Enhancement ratio: {summary['enhancement_ratio']}")
        
        quality = report['quality_metrics']
        print(f"\n🔧 Quality Improvements:")
        print(f"  Particle corrections: {quality['particle_error_prevention']:,}")
        print(f"  Diversity enhancements: {quality['diversity_improvements']:,}")
        print(f"  Advanced techniques: {quality['advanced_technique_applications']:,}")
        print(f"  ToW integrity: {quality['tow_integrity_maintained']:.1f}%")
        
        if report['domain_distribution']:
            print(f"\n🏷️ Domain Distribution:")
            for domain, count in report['domain_distribution'].items():
                domain_name = domain.replace('domain_', '').capitalize()
                print(f"  {domain_name}: {count:,}")
        
        errors = report['error_summary']
        if errors['processing_errors'] > 0 or errors['validation_failures'] > 0:
            print(f"\n⚠️ Error Summary:")
            print(f"  Processing errors: {errors['processing_errors']}")
            print(f"  Validation failures: {errors['validation_failures']}")
        
        print("="*60)

def main():
    """메인 실행 함수"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 사용자 정의 설정
    config = {
        'input_file': 'ToW_koconovel_augmented.json',
        'output_file': 'ToW_koconovel_enhanced_final.json',
        'batch_size': 50,  # 메모리 효율성을 위해 축소
        'max_variants_per_entry': 6,
        'target_diversity_score': 65.0,
        'enable_particle_fixing': True,
        'enable_diversity_enhancement': True,
        'enable_advanced_techniques': True,
        'domain_detection': True,
        'validation_enabled': True
    }
    
    try:
        # 시스템 초기화 및 실행
        system = IntegratedToWAugmentationSystem(config)
        
        # 입력 파일 존재 확인
        if not Path(config['input_file']).exists():
            logger.error(f"Input file not found: {config['input_file']}")
            logger.info("Please ensure the augmented dataset exists")
            return
        
        # 처리 실행
        start_time = time.time()
        report = system.process_dataset(config['input_file'], config['output_file'])
        end_time = time.time()
        
        # 처리 시간 추가
        report['processing_summary']['processing_time'] = f"{end_time - start_time:.2f}s"
        
        # 결과 출력
        system.print_summary_report(report)
        
        logger.info(f"✅ All improvements applied successfully!")
        logger.info(f"📁 Enhanced dataset: {config['output_file']}")
        
    except Exception as e:
        logger.error(f"❌ System execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()