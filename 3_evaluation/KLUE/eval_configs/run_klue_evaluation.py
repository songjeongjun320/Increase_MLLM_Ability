#!/usr/bin/env python3
"""
KLUE 벤치마크 모델 평가 자동 실행 스크립트
모든 모델에 대해 KLUE 태스크를 순차적으로 평가합니다.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import yaml

class KLUEEvaluationRunner:
    """KLUE 평가 자동 실행 클래스"""
    
    def __init__(self, config_dir: str = ".", results_dir: str = "./klue_evaluation_results"):
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # KLUE 태스크 정의
        self.klue_tasks = [
            "tc",      # Topic Classification (YNAT)
            "sts",     # Semantic Textual Similarity  
            "nli",     # Natural Language Inference
            "re",      # Relation Extraction
            "dp",      # Dependency Parsing
            "mrc",     # Machine Reading Comprehension
            "dst"      # Dialogue State Tracking (WoS)
        ]
        
        # 태스크별 few-shot 설정 (복잡도에 따라)
        self.task_fewshots = {
            "tc": 3,
            "sts": 3,
            "nli": 3,
            "re": 2,
            "dp": 1,
            "mrc": 2,
            "dst": 1
        }
    
    def load_model_configs(self) -> List[Dict[str, str]]:
        """모델 설정 로드"""
        models = []
        
        # 하드코딩된 모델 리스트 (사용자 환경에 맞게 수정)
        model_configs = [
            {
                "name": "DeepSeek-R1-Distill-Qwen-1.5B",
                "path": "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-Distill-Qwen-1.5B",
                "adapter": ""
            },
            {
                "name": "google_gemma-3-4b-it", 
                "path": "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
                "adapter": ""
            },
            {
                "name": "Qwen2.5-3B-Instruct",
                "path": "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-3B-Instruct", 
                "adapter": ""
            },
            {
                "name": "Llama-3.2-3B-Instruct",
                "path": "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct",
                "adapter": ""
            }
        ]
        
        # 모델 경로 존재 확인
        for model_config in model_configs:
            if os.path.exists(model_config["path"]):
                models.append(model_config)
                print(f"✅ 모델 발견: {model_config['name']}")
            else:
                print(f"⚠️  모델 경로 없음: {model_config['path']}")
        
        return models
    
    def load_model_configs_from_file(self, config_file: str = "model_configs.yaml") -> List[Dict[str, str]]:
        """YAML 파일에서 모델 설정 로드"""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            print(f"⚠️  모델 설정 파일을 찾을 수 없습니다: {config_path}")
            print("기본 모델 리스트를 사용합니다.")
            return self.load_model_configs()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            models = []
            for model_config in config.get('models', []):
                if os.path.exists(model_config["path"]):
                    models.append(model_config)
                    print(f"✅ 모델 발견: {model_config['name']}")
                else:
                    print(f"⚠️  모델 경로 없음: {model_config['path']}")
            
            return models
            
        except Exception as e:
            print(f"❌ 모델 설정 파일 로드 실패: {e}")
            return self.load_model_configs()
    
    def check_klue_configs(self) -> bool:
        """KLUE 설정 파일들이 모두 존재하는지 확인"""
        config_files = {
            "tc": "tc.yaml",
            "sts": "sts.yaml", 
            "nli": "nli.yaml",
            "re": "re.yaml",
            "dp": "dp.yaml",
            "mrc": "mrc.yaml",
            "dst": "dst.yaml"
        }
        
        missing_files = []
        for task, filename in config_files.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"❌ 누락된 KLUE 설정 파일들: {missing_files}")
            return False
        
        print("✅ 모든 KLUE 설정 파일 존재 확인")
        return True
    
    def run_single_evaluation(self, model: Dict[str, str], task: str) -> Dict[str, Any]:
        """단일 모델-태스크 평가 실행"""
        print(f"\n🚀 평가 시작: {model['name']} on {task.upper()}")
        
        # 모델 인자 구성
        model_args = f"pretrained={model['path']}"
        if model.get('adapter') and model['adapter']:
            model_args += f",peft={model['adapter']},tokenizer={model['adapter']}"
        
        # 출력 파일명
        output_file = self.results_dir / f"{model['name']}_{task}.json"
        
        # lm_eval 명령어 구성
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", task,
            "--num_fewshot", str(self.task_fewshots.get(task, 3)),
            "--batch_size", "auto",
            "--output_path", str(output_file),
            "--verbosity", "INFO"
        ]
        
        # 평가 실행
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1시간 타임아웃
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ 평가 완료: {model['name']} on {task.upper()} ({duration:.1f}초)")
                
                # 결과 파일에서 정확도 추출
                accuracy = self.extract_accuracy(output_file)
                
                return {
                    "model": model['name'],
                    "task": task,
                    "status": "success",
                    "accuracy": accuracy,
                    "duration": duration,
                    "output_file": str(output_file)
                }
            else:
                print(f"❌ 평가 실패: {model['name']} on {task.upper()}")
                print(f"오류: {result.stderr}")
                
                return {
                    "model": model['name'],
                    "task": task,
                    "status": "failed",
                    "error": result.stderr,
                    "duration": duration
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 평가 타임아웃: {model['name']} on {task.upper()}")
            return {
                "model": model['name'],
                "task": task,
                "status": "timeout",
                "duration": 3600
            }
        except Exception as e:
            print(f"❌ 평가 오류: {model['name']} on {task.upper()} - {e}")
            return {
                "model": model['name'],
                "task": task,
                "status": "error",
                "error": str(e)
            }
    
    def extract_accuracy(self, result_file: Path) -> Optional[float]:
        """결과 파일에서 정확도 추출"""
        try:
            if not result_file.exists():
                return None
                
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 결과에서 주요 메트릭 찾기
            if 'results' in data:
                for task_key, task_results in data['results'].items():
                    # 다양한 정확도 키 시도
                    for acc_key in ['acc', 'accuracy', 'exact_match', 'f1', 'pearsonr']:
                        if acc_key in task_results:
                            return float(task_results[acc_key])
            
            return None
            
        except Exception as e:
            print(f"⚠️  정확도 추출 실패 {result_file}: {e}")
            return None
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """모든 모델에 대해 모든 KLUE 태스크 평가 실행"""
        print("🎯 KLUE 벤치마크 평가 시작!")
        print("="*60)
        
        # 설정 확인
        if not self.check_klue_configs():
            return {"status": "failed", "reason": "Missing KLUE config files"}
        
        # 모델 로드
        models = self.load_model_configs_from_file()
        if not models:
            return {"status": "failed", "reason": "No valid models found"}
        
        print(f"📊 평가 대상: {len(models)}개 모델 × {len(self.klue_tasks)}개 태스크 = {len(models) * len(self.klue_tasks)}개 실험")
        print(f"💾 결과 저장 위치: {self.results_dir}")
        
        # 전체 결과 저장
        evaluation_results = []
        start_time = datetime.now()
        
        total_experiments = len(models) * len(self.klue_tasks)
        current_experiment = 0
        
        # 각 모델별로 평가 실행
        for model in models:
            print(f"\n{'='*60}")
            print(f"📋 모델 평가 중: {model['name']}")
            print(f"{'='*60}")
            
            model_results = []
            
            for task in self.klue_tasks:
                current_experiment += 1
                print(f"\n[{current_experiment}/{total_experiments}] {model['name']} → {task.upper()}")
                
                result = self.run_single_evaluation(model, task)
                evaluation_results.append(result)
                model_results.append(result)
            
            # 모델별 결과 요약 출력
            self.print_model_summary(model['name'], model_results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 전체 결과 정리
        summary = {
            "evaluation_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(), 
                "duration_seconds": duration,
                "total_models": len(models),
                "total_tasks": len(self.klue_tasks),
                "total_experiments": total_experiments
            },
            "models": [m['name'] for m in models],
            "tasks": self.klue_tasks,
            "results": evaluation_results
        }
        
        # 결과 저장
        summary_file = self.results_dir / f"klue_evaluation_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("🎉 모든 평가 완료!")
        print(f"⏱️  총 소요 시간: {duration/3600:.1f}시간")
        print(f"📊 전체 결과: {summary_file}")
        print(f"{'='*60}")
        
        # 최종 결과 테이블 출력
        self.print_final_summary(evaluation_results, models)
        
        return summary
    
    def print_model_summary(self, model_name: str, model_results: List[Dict[str, Any]]):
        """모델별 결과 요약 출력"""
        print(f"\n📊 {model_name} 결과 요약:")
        
        for result in model_results:
            task = result['task'].upper()
            status = result['status']
            
            if status == "success":
                acc = result.get('accuracy')
                acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                duration = result.get('duration', 0)
                print(f"  ✅ {task:6}: {acc_str} ({duration:.1f}초)")
            else:
                print(f"  ❌ {task:6}: {status}")
    
    def print_final_summary(self, results: List[Dict[str, Any]], models: List[Dict[str, str]]):
        """최종 결과 요약 테이블 출력"""
        print(f"\n📈 KLUE 벤치마크 최종 결과")
        print(f"{'='*80}")
        
        # 헤더 출력
        header = "모델명".ljust(25)
        for task in self.klue_tasks:
            header += task.upper().rjust(8)
        header += "평균".rjust(8)
        print(header)
        print("-" * 80)
        
        # 각 모델별 결과 출력
        for model in models:
            model_name = model['name']
            row = model_name[:24].ljust(25)
            
            model_results = [r for r in results if r['model'] == model_name]
            accuracies = []
            
            for task in self.klue_tasks:
                task_result = next((r for r in model_results if r['task'] == task), None)
                
                if task_result and task_result['status'] == 'success':
                    acc = task_result.get('accuracy')
                    if acc is not None:
                        row += f"{acc:.3f}".rjust(8)
                        accuracies.append(acc)
                    else:
                        row += "N/A".rjust(8)
                else:
                    row += "FAIL".rjust(8)
            
            # 평균 계산
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                row += f"{avg_acc:.3f}".rjust(8)
            else:
                row += "N/A".rjust(8)
            
            print(row)
        
        print("=" * 80)

def create_model_config_template():
    """모델 설정 템플릿 생성"""
    template = {
        "models": [
            {
                "name": "DeepSeek-R1-Distill-Qwen-1.5B",
                "path": "/path/to/your/model1",
                "adapter": ""  # LoRA 어댑터 경로 (있는 경우)
            },
            {
                "name": "Qwen2.5-3B-Instruct",
                "path": "/path/to/your/model2", 
                "adapter": ""
            }
        ]
    }
    
    with open("model_configs.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ model_configs.yaml 템플릿 생성됨")
    print("모델 경로를 실제 경로로 수정해주세요.")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KLUE 벤치마크 모델 평가 자동 실행")
    parser.add_argument("--config_dir", default=".", help="설정 파일 디렉토리")
    parser.add_argument("--results_dir", default="./klue_evaluation_results", help="결과 저장 디렉토리")
    parser.add_argument("--create_template", action="store_true", help="모델 설정 템플릿 생성")
    
    args = parser.parse_args()
    
    if args.create_template:
        create_model_config_template()
        return
    
    # 평가 실행
    runner = KLUEEvaluationRunner(args.config_dir, args.results_dir)
    results = runner.run_all_evaluations()
    
    if results.get("status") == "failed":
        print(f"❌ 평가 실패: {results.get('reason')}")
        exit(1)

if __name__ == "__main__":
    main()