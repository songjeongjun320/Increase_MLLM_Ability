"""
Performance Analysis Utilities for Evaluation Scripts
공통으로 사용할 성능 분석 함수들
"""

import json
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def analyze_model_performance(model_results: List[Dict], metric_key: str = "accuracy_strict") -> Dict[str, Any]:
    """
    모델별 성능 분석을 수행하여 Top3/Worst3를 반환합니다.
    
    Args:
        model_results: 모델별 결과 리스트
        metric_key: 비교할 메트릭 키 (예: "accuracy_strict")
    
    Returns:
        성능 분석 결과 딕셔너리
    """
    if not model_results:
        return {
            "top_3_models": [],
            "worst_3_models": [],
            "performance_gap": 0.0,
            "average_score": 0.0,
            "best_model": "N/A",
            "worst_model": "N/A"
        }
    
    # 유효한 결과만 필터링
    valid_results = []
    for result in model_results:
        if metric_key in result and isinstance(result[metric_key], (int, float)):
            valid_results.append(result)
        else:
            logger.warning(f"Missing or invalid {metric_key} for model {result.get('model_name', 'Unknown')}")
    
    if not valid_results:
        logger.error("No valid results found for performance analysis")
        return analyze_model_performance([], metric_key)
    
    # 성능에 따라 정렬
    sorted_results = sorted(valid_results, key=lambda x: x[metric_key], reverse=True)
    
    # Top 3과 Worst 3 추출
    top_3 = sorted_results[:3]
    worst_3 = sorted_results[-3:]
    worst_3.reverse()  # 낮은 순서대로 정렬
    
    # 통계 계산
    scores = [result[metric_key] for result in valid_results]
    average_score = sum(scores) / len(scores)
    performance_gap = max(scores) - min(scores)
    
    return {
        "top_3_models": [
            {
                "model_name": result["model_name"],
                "score": result[metric_key],
                "rank": idx + 1
            }
            for idx, result in enumerate(top_3)
        ],
        "worst_3_models": [
            {
                "model_name": result["model_name"], 
                "score": result[metric_key],
                "rank": len(valid_results) - idx
            }
            for idx, result in enumerate(worst_3)
        ],
        "performance_gap": performance_gap,
        "average_score": average_score,
        "best_model": sorted_results[0]["model_name"],
        "worst_model": sorted_results[-1]["model_name"],
        "total_models": len(valid_results)
    }

def analyze_subject_performance(model_results: List[Dict], subject_key: str = "subject_wise_accuracy") -> Dict[str, Any]:
    """
    과목별/카테고리별 성능 분석을 수행합니다.
    
    Args:
        model_results: 모델별 결과 리스트
        subject_key: 과목별 데이터 키
    
    Returns:
        과목별 성능 분석 결과
    """
    all_subjects = set()
    model_subject_scores = {}
    
    # 모든 과목 수집 및 모델별 과목 점수 추출
    for result in model_results:
        model_name = result.get("model_name", "Unknown")
        if subject_key in result and isinstance(result[subject_key], dict):
            model_subject_scores[model_name] = {}
            for subject, subject_data in result[subject_key].items():
                if isinstance(subject_data, dict) and "accuracy" in subject_data:
                    all_subjects.add(subject)
                    model_subject_scores[model_name][subject] = subject_data["accuracy"]
    
    if not all_subjects:
        return {
            "subject_analysis": {},
            "strongest_subjects": [],
            "weakest_subjects": [],
            "subject_performance_variance": {}
        }
    
    # 과목별 평균 점수 계산
    subject_averages = {}
    subject_performance_variance = {}
    
    for subject in all_subjects:
        scores = []
        for model_name, subjects_data in model_subject_scores.items():
            if subject in subjects_data:
                scores.append(subjects_data[subject])
        
        if scores:
            avg_score = sum(scores) / len(scores)
            variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
            subject_averages[subject] = avg_score
            subject_performance_variance[subject] = {
                "variance": variance,
                "std_dev": variance ** 0.5,
                "min_score": min(scores),
                "max_score": max(scores),
                "score_range": max(scores) - min(scores)
            }
    
    # 가장 강한/약한 과목 TOP 3
    sorted_subjects = sorted(subject_averages.items(), key=lambda x: x[1], reverse=True)
    strongest_subjects = sorted_subjects[:3]
    weakest_subjects = sorted_subjects[-3:]
    weakest_subjects.reverse()
    
    return {
        "subject_analysis": subject_averages,
        "strongest_subjects": [
            {"subject": subject, "average_score": score}
            for subject, score in strongest_subjects
        ],
        "weakest_subjects": [
            {"subject": subject, "average_score": score}
            for subject, score in weakest_subjects
        ],
        "subject_performance_variance": subject_performance_variance
    }

def generate_model_insights(model_result: Dict, all_results: List[Dict], metric_key: str = "accuracy_strict") -> Dict[str, Any]:
    """
    개별 모델에 대한 인사이트를 생성합니다.
    
    Args:
        model_result: 분석할 모델의 결과
        all_results: 모든 모델의 결과 리스트
        metric_key: 비교할 메트릭 키
    
    Returns:
        모델 인사이트 딕셔너리
    """
    if not model_result or metric_key not in model_result:
        return {"insights": [], "relative_performance": "N/A"}
    
    model_score = model_result[metric_key]
    model_name = model_result.get("model_name", "Unknown")
    
    # 모든 모델 점수 수집
    all_scores = []
    for result in all_results:
        if metric_key in result and isinstance(result[metric_key], (int, float)):
            all_scores.append(result[metric_key])
    
    if not all_scores:
        return {"insights": [], "relative_performance": "N/A"}
    
    # 상대적 성능 계산
    average_score = sum(all_scores) / len(all_scores)
    percentile = (sum(1 for score in all_scores if score < model_score) / len(all_scores)) * 100
    
    # 인사이트 생성
    insights = []
    
    if model_score >= max(all_scores):
        insights.append(f"🏆 최고 성능 모델 ({model_score:.2f}%)")
    elif model_score <= min(all_scores):
        insights.append(f"⚠️ 최저 성능 모델 ({model_score:.2f}%)")
    
    if model_score > average_score:
        diff = model_score - average_score
        insights.append(f"📈 평균보다 {diff:.2f}%p 높은 성능")
    elif model_score < average_score:
        diff = average_score - model_score
        insights.append(f"📉 평균보다 {diff:.2f}%p 낮은 성능")
    
    performance_level = "상위" if percentile >= 66.7 else "중위" if percentile >= 33.3 else "하위"
    insights.append(f"🎯 전체 모델 중 상위 {100-percentile:.1f}% ({performance_level} 그룹)")
    
    return {
        "insights": insights,
        "relative_performance": {
            "percentile": percentile,
            "performance_level": performance_level,
            "score": model_score,
            "average_score": average_score,
            "score_difference": model_score - average_score
        }
    }

def create_enhanced_summary(model_results: List[Dict], evaluation_info: Dict, 
                          primary_metric: str = "accuracy_strict",
                          subject_metric: str = "subject_wise_accuracy") -> Dict[str, Any]:
    """
    향상된 summary를 생성합니다.
    
    Args:
        model_results: 모델별 결과 리스트
        evaluation_info: 평가 정보
        primary_metric: 주요 메트릭 키
        subject_metric: 과목별 메트릭 키
    
    Returns:
        향상된 summary 딕셔너리
    """
    # 기본 성능 분석
    performance_analysis = analyze_model_performance(model_results, primary_metric)
    
    # 과목별 성능 분석 (있는 경우)
    subject_analysis = analyze_subject_performance(model_results, subject_metric)
    
    # 각 모델별 인사이트
    model_insights = {}
    for result in model_results:
        model_name = result.get("model_name", "Unknown")
        model_insights[model_name] = generate_model_insights(result, model_results, primary_metric)
    
    return {
        "evaluation_info": evaluation_info,
        "performance_analysis": performance_analysis,
        "subject_analysis": subject_analysis,
        "model_insights": model_insights,
        "model_results": model_results,
        "summary_statistics": {
            "total_models_evaluated": len(model_results),
            "evaluation_metric": primary_metric,
            "has_subject_breakdown": bool(subject_analysis["subject_analysis"]),
            "performance_range": {
                "highest": max((r.get(primary_metric, 0) for r in model_results), default=0),
                "lowest": min((r.get(primary_metric, 0) for r in model_results), default=0),
                "average": sum(r.get(primary_metric, 0) for r in model_results) / len(model_results) if model_results else 0
            }
        }
    }