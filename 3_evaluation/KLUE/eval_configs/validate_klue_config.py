#!/usr/bin/env python3
"""
KLUE 설정 파일 검증 스크립트
데이터셋 로딩, 프롬프트 템플릿, 필터 동작을 테스트
"""

import yaml
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import load_dataset

class KLUEConfigValidator:
    """KLUE 설정 검증 클래스"""
    
    def __init__(self):
        self.tasks = ['ynat', 'sts', 'nli', 're', 'dp', 'mrc', 'wos']
        self.config_files = {
            'ynat': 'tc.yaml',
            'sts': 'sts.yaml', 
            'nli': 'nli.yaml',
            're': 're.yaml',
            'dp': 'dp.yaml',
            'mrc': 'mrc.yaml',
            'wos': 'dst.yaml'
        }
    
    def load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"설정 파일 로드 실패 {config_path}: {e}")
            return None
    
    def test_dataset_loading(self, dataset_name: str) -> bool:
        """데이터셋 로딩 테스트"""
        try:
            print(f"\n📊 {dataset_name} 데이터셋 로딩 테스트...")
            dataset = load_dataset('klue', dataset_name, split='validation[:10]')
            
            if len(dataset) == 0:
                print(f"❌ {dataset_name}: 데이터가 없습니다")
                return False
            
            # 첫 번째 샘플 출력
            sample = dataset[0]
            print(f"✅ {dataset_name}: {len(dataset)}개 샘플 로드 성공")
            print(f"📝 샘플 구조: {list(sample.keys())}")
            
            # 각 태스크별 중요 필드 확인
            self._check_task_fields(dataset_name, sample)
            return True
            
        except Exception as e:
            print(f"❌ {dataset_name} 로딩 실패: {e}")
            return False
    
    def _check_task_fields(self, dataset_name: str, sample: Dict[str, Any]):
        """태스크별 필수 필드 확인"""
        required_fields = {
            'ynat': ['title', 'label'],
            'sts': ['sentence1', 'sentence2', 'labels'],
            'nli': ['premise', 'hypothesis', 'label'],
            're': ['sentence', 'subject_entity', 'object_entity', 'label'],
            'dp': ['sentence', 'word_form', 'head'],
            'mrc': ['context', 'question', 'answers'],
            'wos': ['dialogue', 'domains']
        }
        
        if dataset_name in required_fields:
            missing_fields = []
            for field in required_fields[dataset_name]:
                if field not in sample:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"⚠️  누락된 필드: {missing_fields}")
            else:
                print(f"✅ 모든 필수 필드 존재")
    
    def test_prompt_template(self, config: Dict[str, Any], dataset_name: str) -> bool:
        """프롬프트 템플릿 테스트"""
        try:
            print(f"\n🔧 {dataset_name} 프롬프트 템플릿 테스트...")
            
            # 샘플 데이터 로드
            dataset = load_dataset('klue', dataset_name, split='validation[:1]')
            sample = dataset[0]
            
            # doc_to_text 템플릿 적용 테스트
            template = config.get('doc_to_text', '')
            if not template:
                print(f"❌ doc_to_text가 설정되지 않음")
                return False
            
            # Jinja2 템플릿 렌더링 시뮬레이션 (간단한 변수 치환)
            rendered = self._render_template(template, sample)
            
            print(f"✅ 템플릿 렌더링 성공")
            print(f"📝 렌더링 결과 (처음 200자):")
            print(rendered[:200] + "..." if len(rendered) > 200 else rendered)
            
            return True
            
        except Exception as e:
            print(f"❌ 프롬프트 템플릿 테스트 실패: {e}")
            return False
    
    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """간단한 템플릿 렌더링 (Jinja2 대용)"""
        rendered = template
        
        # 기본 변수 치환
        for key, value in data.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, str(value))
        
        # 중첩된 속성 처리 (예: subject_entity.word)
        nested_pattern = r'\{\{(\w+)\.(\w+)\}\}'
        matches = re.findall(nested_pattern, rendered)
        
        for obj_name, attr_name in matches:
            if obj_name in data and isinstance(data[obj_name], dict):
                if attr_name in data[obj_name]:
                    placeholder = f"{{{{{obj_name}.{attr_name}}}}}"
                    rendered = rendered.replace(placeholder, str(data[obj_name][attr_name]))
        
        return rendered
    
    def test_regex_filter(self, config: Dict[str, Any]) -> bool:
        """정규식 필터 테스트"""
        try:
            filter_list = config.get('filter_list', [])
            if not filter_list:
                print("ℹ️  필터가 설정되지 않음")
                return True
            
            print(f"\n🔍 정규식 필터 테스트...")
            
            # 테스트 케이스
            test_cases = {
                'sts': ['점수: 4.5', '평가: 3.2', '4', '점수는 2.1점입니다'],
                'dp': ['1 2 3 4 5', '10 20 30', '0 1 2'],
                're': ['A', 'B', 'C']
            }
            
            for filter_config in filter_list:
                filter_steps = filter_config.get('filter', [])
                
                for step in filter_steps:
                    if step.get('function') == 'regex':
                        pattern = step.get('regex_pattern', '')
                        print(f"🔧 패턴 테스트: {pattern}")
                        
                        # 해당하는 테스트 케이스 찾기
                        task_type = self._infer_task_type(pattern)
                        if task_type and task_type in test_cases:
                            for test_input in test_cases[task_type]:
                                match = re.search(pattern, test_input)
                                result = match.group(1) if match else None
                                print(f"  입력: '{test_input}' → 결과: '{result}'")
            
            print("✅ 정규식 필터 테스트 완료")
            return True
            
        except Exception as e:
            print(f"❌ 정규식 필터 테스트 실패: {e}")
            return False
    
    def _infer_task_type(self, pattern: str) -> Optional[str]:
        """정규식 패턴으로부터 태스크 타입 추론"""
        if '점수' in pattern or '[0-5]' in pattern:
            return 'sts'
        elif '[0-9]+' in pattern and 'space' in pattern:
            return 'dp'
        elif '[A-Z]' in pattern:
            return 're'
        return None
    
    def test_target_extraction(self, config: Dict[str, Any], dataset_name: str) -> bool:
        """타겟 추출 테스트"""
        try:
            print(f"\n🎯 {dataset_name} 타겟 추출 테스트...")
            
            # 샘플 데이터 로드
            dataset = load_dataset('klue', dataset_name, split='validation[:1]')
            sample = dataset[0]
            
            doc_to_target = config.get('doc_to_target', '')
            if not doc_to_target:
                print(f"❌ doc_to_target이 설정되지 않음")
                return False
            
            # 타겟 값 추출
            target_value = self._extract_target_value(sample, doc_to_target)
            
            print(f"✅ 타겟 추출 성공")
            print(f"📝 타겟 경로: {doc_to_target}")
            print(f"📝 타겟 값: {target_value}")
            print(f"📝 타겟 타입: {type(target_value)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 타겟 추출 테스트 실패: {e}")
            return False
    
    def _extract_target_value(self, data: Dict[str, Any], target_path: str) -> Any:
        """중첩된 경로에서 타겟 값 추출"""
        keys = target_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None
        
        return current
    
    def validate_all_configs(self, config_dir: str = '.') -> Dict[str, bool]:
        """모든 KLUE 설정 파일 검증"""
        results = {}
        config_path = Path(config_dir)
        
        print("🚀 KLUE 설정 검증 시작\n" + "="*50)
        
        for task_name, config_file in self.config_files.items():
            print(f"\n📋 {task_name.upper()} 태스크 검증 중...")
            
            # 설정 파일 로드
            full_path = config_path / config_file
            config = self.load_config(str(full_path))
            
            if not config:
                results[task_name] = False
                continue
            
            # 각종 테스트 수행
            tests_passed = []
            
            # 1. 데이터셋 로딩 테스트
            tests_passed.append(self.test_dataset_loading(task_name))
            
            # 2. 프롬프트 템플릿 테스트
            tests_passed.append(self.test_prompt_template(config, task_name))
            
            # 3. 정규식 필터 테스트
            tests_passed.append(self.test_regex_filter(config))
            
            # 4. 타겟 추출 테스트
            tests_passed.append(self.test_target_extraction(config, task_name))
            
            results[task_name] = all(tests_passed)
            
            status = "✅ 통과" if results[task_name] else "❌ 실패"
            print(f"📊 {task_name} 전체 결과: {status}")
        
        # 최종 결과 요약
        print("\n" + "="*50)
        print("📊 전체 검증 결과:")
        
        passed = sum(results.values())
        total = len(results)
        
        for task_name, passed_test in results.items():
            status = "✅" if passed_test else "❌"
            print(f"  {status} {task_name.upper()}")
        
        print(f"\n🎯 총 {total}개 태스크 중 {passed}개 통과 ({passed/total*100:.1f}%)")
        
        return results
    
    def generate_test_report(self, results: Dict[str, bool]) -> str:
        """테스트 결과 보고서 생성"""
        report = []
        report.append("# KLUE 설정 검증 보고서")
        report.append("")
        
        # 요약
        passed = sum(results.values())
        total = len(results)
        report.append(f"## 요약")
        report.append(f"- 전체 태스크: {total}개")
        report.append(f"- 통과한 태스크: {passed}개")
        report.append(f"- 실패한 태스크: {total-passed}개")
        report.append(f"- 성공률: {passed/total*100:.1f}%")
        report.append("")
        
        # 상세 결과
        report.append("## 상세 결과")
        for task_name, passed_test in results.items():
            status = "✅ 통과" if passed_test else "❌ 실패"
            report.append(f"- **{task_name.upper()}**: {status}")
        
        report.append("")
        report.append("## 권장 사항")
        
        failed_tasks = [task for task, passed in results.items() if not passed]
        if failed_tasks:
            report.append("실패한 태스크들을 위한 권장 사항:")
            for task in failed_tasks:
                report.append(f"- {task}: 설정 파일과 데이터 구조 재확인 필요")
        else:
            report.append("모든 태스크가 통과했습니다! 🎉")
        
        return "\n".join(report)

def main():
    """메인 실행 함수"""
    validator = KLUEConfigValidator()
    
    # 검증 실행
    results = validator.validate_all_configs()
    
    # 보고서 생성
    report = validator.generate_test_report(results)
    
    # 보고서 저장
    with open('klue_validation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 상세 보고서가 'klue_validation_report.md'에 저장되었습니다.")

if __name__ == "__main__":
    main()