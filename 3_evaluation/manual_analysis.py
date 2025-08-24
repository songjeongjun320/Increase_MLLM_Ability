#!/usr/bin/env python3
"""
Manual Performance Analysis for Existing Evaluation Results
기존 평가 결과에 대한 수동 성능 분석 스크립트

Usage:
    python manual_analysis.py --folder "mmlu_model1_5shot"
    python manual_analysis.py --folder "kmmlu_tow_model1_5shot" --type "kmmlu"
    python manual_analysis.py --folder "mmlu_prox_5shot" --type "mmlu_prox"
    python manual_analysis.py --folder "gsm8k_results" --type "gsm8k"
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import glob

# Import performance analyzer
from performance_analyzer import create_enhanced_summary, analyze_model_performance, analyze_subject_performance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """기존 평가 결과를 분석하는 클래스"""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.results = []
        
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
    def detect_evaluation_type(self) -> str:
        """폴더 구조와 파일명을 보고 평가 타입을 자동 감지"""
        folder_name = self.folder_path.name.lower()
        
        # Check for existing summary files
        if (self.folder_path / "SUMMARY.json").exists():
            return "mmlu_prox"
        elif (self.folder_path / "summary.json").exists():
            return "mmlu_kmmlu"
        elif (self.folder_path / "final_gsm8k_results.json").exists():
            return "gsm8k"
            
        # Check folder name patterns
        if "mmlu_prox" in folder_name or "mmluprox" in folder_name:
            return "mmlu_prox"
        elif "kmmlu" in folder_name or "ko_mmlu" in folder_name:
            return "kmmlu"
        elif "mmlu" in folder_name:
            return "mmlu"
        elif "gsm8k" in folder_name or "hrm8k" in folder_name:
            return "gsm8k"
        elif "arc" in folder_name:
            return "arc"
        elif "piqa" in folder_name:
            return "piqa"
        elif "klue" in folder_name:
            return "klue"
        else:
            return "auto"  # Try to auto-detect from files
    
    def load_mmlu_kmmlu_results(self) -> List[Dict[str, Any]]:
        """MMLU/KMMLU 결과 로드"""
        results = []
        
        # Look for model subdirectories
        for model_dir in self.folder_path.iterdir():
            if model_dir.is_dir():
                result_files = list(model_dir.glob("results_*.json"))
                
                for result_file in result_files:
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Extract model name from file or directory
                        model_name = model_dir.name
                        
                        result = {
                            "model_name": model_name,
                            "accuracy_standard": data.get("accuracy_standard (correct / valid_predictions)", 0),
                            "accuracy_strict": data.get("accuracy_strict (correct / total_test_items)", 0),
                            "correct_predictions": data.get("correct_predictions", 0),
                            "valid_predictions": data.get("valid_predictions", 0),
                            "total_items": data.get("test_items", 0),
                            "subject_wise_accuracy": data.get("subject_wise_accuracy", {})
                        }
                        results.append(result)
                        logger.info(f"Loaded results for {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading {result_file}: {e}")
        
        return results
    
    def load_mmlu_prox_results(self) -> List[Dict[str, Any]]:
        """MMLU_ProX 결과 로드"""
        results = []
        
        for model_dir in self.folder_path.iterdir():
            if model_dir.is_dir():
                result_files = list(model_dir.glob("results_*_5shot.json"))
                
                for result_file in result_files:
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        model_name = model_dir.name
                        
                        # Calculate combined accuracy
                        en_acc = data.get("mmlu_prox_en_results", {}).get("accuracy_strict", 0)
                        ko_acc = data.get("mmlu_prox_ko_results", {}).get("accuracy_strict", 0)
                        combined_acc = (en_acc + ko_acc) / 2
                        
                        result = {
                            "model_name": model_name,
                            "accuracy_strict": combined_acc,
                            "mmlu_prox_en_accuracy": en_acc,
                            "mmlu_prox_ko_accuracy": ko_acc,
                            "correct_predictions": (
                                data.get("mmlu_prox_en_results", {}).get("correct_predictions", 0) +
                                data.get("mmlu_prox_ko_results", {}).get("correct_predictions", 0)
                            ),
                            "total_items": (
                                data.get("mmlu_prox_en_results", {}).get("total_items", 0) +
                                data.get("mmlu_prox_ko_results", {}).get("total_items", 0)
                            )
                        }
                        results.append(result)
                        logger.info(f"Loaded MMLU_ProX results for {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading {result_file}: {e}")
        
        return results
    
    def load_gsm8k_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """GSM8K 결과 로드 (한국어/영어 분리)"""
        korean_results = []
        english_results = []
        
        # Look for individual model results
        for model_dir in self.folder_path.iterdir():
            if model_dir.is_dir():
                korean_file = model_dir / f"results_korean_{model_dir.name}.json"
                english_file = model_dir / f"results_english_{model_dir.name}.json"
                
                # Try Korean results
                if korean_file.exists():
                    try:
                        with open(korean_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        result = {
                            "model_name": model_dir.name,
                            "accuracy_strict": data.get("accuracy_strict", 0),
                            "accuracy_standard": data.get("accuracy_standard", 0),
                            "correct_predictions": data.get("correct_predictions", 0),
                            "valid_predictions": data.get("valid_predictions", 0),
                            "total_questions": data.get("total_questions", 0)
                        }
                        korean_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error loading Korean GSM8K results for {model_dir.name}: {e}")
                
                # Try English results
                if english_file.exists():
                    try:
                        with open(english_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        result = {
                            "model_name": model_dir.name,
                            "accuracy_strict": data.get("accuracy_strict", 0),
                            "accuracy_standard": data.get("accuracy_standard", 0),
                            "correct_predictions": data.get("correct_predictions", 0),
                            "valid_predictions": data.get("valid_predictions", 0),
                            "total_questions": data.get("total_questions", 0)
                        }
                        english_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error loading English GSM8K results for {model_dir.name}: {e}")
        
        return {"korean": korean_results, "english": english_results}
    
    def load_generic_results(self) -> List[Dict[str, Any]]:
        """일반적인 결과 파일 로드 (자동 감지)"""
        results = []
        
        # Look for any JSON files containing results
        json_files = list(self.folder_path.rglob("*.json"))
        result_files = [f for f in json_files if any(keyword in f.name.lower() 
                       for keyword in ['result', 'summary', 'evaluation'])]
        
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Try to extract model results from different formats
                if isinstance(data, list):
                    # List of model results
                    for item in data:
                        if isinstance(item, dict) and "model_name" in item:
                            results.append(item)
                elif isinstance(data, dict):
                    # Check for different structures
                    if "model_results" in data:
                        results.extend(data["model_results"])
                    elif "korean_results" in data:
                        results.extend(data["korean_results"])
                    elif "english_results" in data:
                        results.extend(data["english_results"])
                    elif "model_name" in data:
                        results.append(data)
                
            except Exception as e:
                logger.debug(f"Skipping {result_file}: {e}")
        
        return results
    
    def analyze_results(self, evaluation_type: str = "auto") -> Dict[str, Any]:
        """결과 분석 수행"""
        if evaluation_type == "auto":
            evaluation_type = self.detect_evaluation_type()
        
        logger.info(f"Analyzing results as type: {evaluation_type}")
        
        if evaluation_type in ["mmlu", "kmmlu", "mmlu_kmmlu"]:
            results = self.load_mmlu_kmmlu_results()
            return self._create_enhanced_summary(results, evaluation_type, "accuracy_strict", "subject_wise_accuracy")
            
        elif evaluation_type == "mmlu_prox":
            results = self.load_mmlu_prox_results()
            return self._create_enhanced_summary(results, evaluation_type, "accuracy_strict")
            
        elif evaluation_type == "gsm8k":
            gsm8k_results = self.load_gsm8k_results()
            
            if gsm8k_results["korean"] and gsm8k_results["english"]:
                # Analyze both languages
                korean_analysis = self._create_enhanced_summary(gsm8k_results["korean"], "gsm8k_korean", "accuracy_strict")
                english_analysis = self._create_enhanced_summary(gsm8k_results["english"], "gsm8k_english", "accuracy_strict")
                
                return {
                    "evaluation_type": "GSM8K (Korean and English)",
                    "korean_analysis": korean_analysis,
                    "english_analysis": english_analysis,
                    "language_comparison": {
                        "korean_avg_score": korean_analysis["performance_analysis"]["average_score"],
                        "english_avg_score": english_analysis["performance_analysis"]["average_score"],
                        "performance_difference": english_analysis["performance_analysis"]["average_score"] - korean_analysis["performance_analysis"]["average_score"]
                    }
                }
            else:
                # Single language
                results = gsm8k_results["korean"] or gsm8k_results["english"]
                lang = "korean" if gsm8k_results["korean"] else "english"
                return self._create_enhanced_summary(results, f"gsm8k_{lang}", "accuracy_strict")
        
        else:
            # Generic analysis
            results = self.load_generic_results()
            return self._create_enhanced_summary(results, evaluation_type, "accuracy_strict")
    
    def _create_enhanced_summary(self, results: List[Dict[str, Any]], eval_type: str, 
                                primary_metric: str, subject_metric: str = None) -> Dict[str, Any]:
        """향상된 summary 생성"""
        if not results:
            logger.error("No results found to analyze")
            return {"error": "No results found"}
        
        evaluation_info = {
            "evaluation_type": eval_type,
            "evaluation_date": datetime.now().isoformat(),
            "folder_path": str(self.folder_path),
            "total_models_evaluated": len(results),
            "analysis_method": "manual_analysis"
        }
        
        return create_enhanced_summary(
            model_results=results,
            evaluation_info=evaluation_info,
            primary_metric=primary_metric,
            subject_metric=subject_metric
        )

def main():
    parser = argparse.ArgumentParser(description="Manually analyze existing evaluation results")
    parser.add_argument("--folder", "-f", required=True, help="Path to evaluation results folder")
    parser.add_argument("--type", "-t", default="auto", 
                       choices=["auto", "mmlu", "kmmlu", "mmlu_prox", "gsm8k", "arc", "piqa"],
                       help="Type of evaluation (auto-detect if not specified)")
    parser.add_argument("--output", "-o", help="Output file path (default: enhanced_summary.json in the folder)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = ResultsAnalyzer(args.folder)
        
        # Perform analysis
        logger.info(f"Starting analysis of folder: {args.folder}")
        analysis_results = analyzer.analyze_results(args.type)
        
        if "error" in analysis_results:
            logger.error(f"Analysis failed: {analysis_results['error']}")
            return
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = analyzer.folder_path / "enhanced_summary.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced summary saved to: {output_path}")
        
        # Display key insights
        if "performance_analysis" in analysis_results:
            perf = analysis_results["performance_analysis"]
            logger.info("\n" + "="*50)
            logger.info("KEY INSIGHTS")
            logger.info("="*50)
            logger.info(f"🏆 Best Model: {perf['best_model']}")
            logger.info(f"📊 Average Score: {perf['average_score']:.2f}%")
            logger.info(f"📈 Performance Gap: {perf['performance_gap']:.2f}%p")
            logger.info(f"📋 Total Models: {len(analysis_results.get('model_results', []))}")
            
            if perf['top_3_models']:
                logger.info(f"\nTop 3 Models:")
                for i, model in enumerate(perf['top_3_models'], 1):
                    logger.info(f"  {i}. {model['model_name']}: {model['score']:.2f}%")
            
            if perf['worst_3_models']:
                logger.info(f"\nBottom 3 Models:")
                for i, model in enumerate(perf['worst_3_models'], 1):
                    logger.info(f"  {i}. {model['model_name']}: {model['score']:.2f}%")
        
        # Language comparison for GSM8K
        if "language_comparison" in analysis_results:
            lang_comp = analysis_results["language_comparison"]
            logger.info(f"\n🌐 Language Performance:")
            logger.info(f"   Korean: {lang_comp['korean_avg_score']:.2f}%")
            logger.info(f"   English: {lang_comp['english_avg_score']:.2f}%")
            logger.info(f"   Difference: {abs(lang_comp['performance_difference']):.2f}%p")
        
        logger.info("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()