#!/usr/bin/env python3
"""
KLUE Model-wise Benchmark Runner
Evaluates a single model on all KLUE tasks sequentially to optimize memory usage
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS, KLUE_TASKS, PROMPT_TEMPLATES, BASE_OUTPUT_DIR, DATA_DIR
from utils import ModelLoader, load_data, save_results, run_evaluation

logger = logging.getLogger(__name__)

# Import evaluation functions from task modules
import tc
import sts
import nli
import ner
import re as re_module
import dp
import mrc
import dst

TASK_EVALUATORS = {
    'tc': tc.evaluate_tc_task,
    'sts': sts.evaluate_sts_task,
    'nli': nli.evaluate_nli_task,
    'ner': ner.evaluate_ner_task,
    're': re_module.evaluate_re_task,
    'dp': dp.evaluate_dp_task,
    'mrc': mrc.evaluate_mrc_task,
    'dst': dst.evaluate_dst_task
}

def run_model_on_all_tasks(model_config, output_dir: str, tasks: list = None, max_samples: int = None):
    """Run a single model on all KLUE tasks"""
    if tasks is None:
        tasks = list(KLUE_TASKS.keys())
    
    logger.info(f"Starting evaluation for model: {model_config.name}")
    logger.info(f"Tasks to evaluate: {tasks}")
    
    model_results = {}
    model_output_dir = os.path.join(output_dir, f"model_{model_config.name}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    for task_key in tasks:
        if task_key not in TASK_EVALUATORS:
            logger.warning(f"No evaluator found for task: {task_key}")
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating {model_config.name} on {task_key.upper()}")
        logger.info(f"{'='*50}")
        
        try:
            # Run task evaluation
            evaluator = TASK_EVALUATORS[task_key]
            results = evaluator(model_config, model_output_dir, max_samples)
            
            if results:
                model_results[task_key] = results['metrics']
                logger.info(f"✅ {task_key.upper()} completed for {model_config.name}")
            else:
                logger.error(f"❌ {task_key.upper()} failed for {model_config.name}")
                model_results[task_key] = None
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {task_key.upper()} for {model_config.name}: {e}")
            model_results[task_key] = None
    
    # Save model summary
    model_summary = {
        'model_name': model_config.name,
        'model_config': {
            'model_id': model_config.model_id,
            'adapter_path': model_config.adapter_path,
            'use_quantization': model_config.use_quantization
        },
        'timestamp': datetime.now().isoformat(),
        'tasks_evaluated': tasks,
        'results': model_results
    }
    
    summary_file = os.path.join(model_output_dir, f"{model_config.name}_klue_summary.json")
    save_results(model_summary, summary_file)
    
    # Print model summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY FOR {model_config.name}")
    logger.info(f"{'='*60}")
    
    for task_key, metrics in model_results.items():
        if metrics:
            task_name = KLUE_TASKS[task_key]['name']
            primary_metric = KLUE_TASKS[task_key]['metric']
            
            if primary_metric in metrics:
                metric_value = metrics[primary_metric]
                logger.info(f"{task_key.upper():4} ({task_name:30}): {primary_metric} = {metric_value:.4f}")
            else:
                # Find first available metric
                first_metric = list(metrics.keys())[0] if metrics else None
                if first_metric:
                    metric_value = metrics[first_metric]
                    logger.info(f"{task_key.upper():4} ({task_name:30}): {first_metric} = {metric_value:.4f}")
        else:
            logger.info(f"{task_key.upper():4}: FAILED")
    
    return model_summary

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run KLUE benchmark on a specific model')
    parser.add_argument('--model', type=str, help='Model name to evaluate (if not specified, evaluates all)')
    parser.add_argument('--tasks', nargs='+', choices=list(KLUE_TASKS.keys()), 
                       default=list(KLUE_TASKS.keys()), help='Tasks to evaluate')
    parser.add_argument('--max-samples', type=int, help='Maximum samples per task (for testing)')
    
    args = parser.parse_args()
    
    # Filter models if specific model requested
    if args.model:
        models_to_evaluate = [config for config in MODEL_CONFIGS if config.name == args.model]
        if not models_to_evaluate:
            logger.error(f"Model '{args.model}' not found in MODEL_CONFIGS")
            logger.info(f"Available models: {[config.name for config in MODEL_CONFIGS]}")
            return
    else:
        models_to_evaluate = MODEL_CONFIGS
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"klue_model_wise_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting model-wise KLUE evaluation")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to evaluate: {[config.name for config in models_to_evaluate]}")
    logger.info(f"Tasks: {args.tasks}")
    if args.max_samples:
        logger.info(f"Max samples per task: {args.max_samples}")
    
    all_model_summaries = {}
    
    for model_config in models_to_evaluate:
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING MODEL: {model_config.name}")
        logger.info(f"{'='*80}")
        
        try:
            model_summary = run_model_on_all_tasks(
                model_config=model_config,
                output_dir=output_dir,
                tasks=args.tasks,
                max_samples=args.max_samples
            )
            all_model_summaries[model_config.name] = model_summary
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_config.name}: {e}")
            continue
    
    # Save overall summary
    overall_summary = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_type': 'model_wise',
        'tasks_evaluated': args.tasks,
        'max_samples': args.max_samples,
        'models_evaluated': len(all_model_summaries),
        'model_summaries': all_model_summaries
    }
    
    overall_file = os.path.join(output_dir, "overall_summary.json")
    save_results(overall_summary, overall_file)
    
    # Print final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total models evaluated: {len(all_model_summaries)}")
    logger.info(f"Total tasks per model: {len(args.tasks)}")
    logger.info(f"Results saved to: {output_dir}")
    
    # Print comparative results
    if len(all_model_summaries) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("COMPARATIVE RESULTS")
        logger.info(f"{'='*60}")
        
        for task_key in args.tasks:
            task_name = KLUE_TASKS[task_key]['name']
            primary_metric = KLUE_TASKS[task_key]['metric']
            
            logger.info(f"\n{task_key.upper()} - {task_name} ({primary_metric}):")
            logger.info("-" * 50)
            
            # Collect results for this task
            task_results = []
            for model_name, summary in all_model_summaries.items():
                results = summary.get('results', {}).get(task_key)
                if results and primary_metric in results:
                    task_results.append((model_name, results[primary_metric]))
                elif results:
                    # Use first available metric
                    first_metric = list(results.keys())[0]
                    task_results.append((model_name, results[first_metric]))
            
            # Sort by score (descending)
            task_results.sort(key=lambda x: x[1], reverse=True)
            
            for model_name, score in task_results:
                logger.info(f"  {model_name:40} {score:.4f}")

if __name__ == "__main__":
    main()