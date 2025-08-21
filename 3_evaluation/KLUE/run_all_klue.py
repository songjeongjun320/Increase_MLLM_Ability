#!/usr/bin/env python3
"""
KLUE Benchmark Runner - Run All Tasks
Sequentially evaluates all models on all KLUE tasks and generates comprehensive results
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS, KLUE_TASKS, BASE_OUTPUT_DIR

logger = logging.getLogger(__name__)

# Task configurations
TASK_CONFIGS = {
    'tc': {
        'script': 'tc.py',
        'name': 'Topic Classification',
        'primary_metric': 'f1_macro'
    },
    'sts': {
        'script': 'sts.py',
        'name': 'Sentence Textual Similarity',
        'primary_metric': 'pearson'
    },
    'nli': {
        'script': 'nli.py',
        'name': 'Natural Language Inference',
        'primary_metric': 'accuracy'
    },
    'ner': {
        'script': 'ner.py',
        'name': 'Named Entity Recognition',
        'primary_metric': 'f1'
    },
    're': {
        'script': 're.py',
        'name': 'Relation Extraction',
        'primary_metric': 'f1_micro'
    },
    'dp': {
        'script': 'dp.py',
        'name': 'Dependency Parsing',
        'primary_metric': 'las'
    },
    'mrc': {
        'script': 'mrc.py',
        'name': 'Machine Reading Comprehension',
        'primary_metric': 'f1'
    },
    'dst': {
        'script': 'dst.py',
        'name': 'Dialogue State Tracking',
        'primary_metric': 'joint_goal_accuracy'
    }
}

def run_task_evaluation(task_key: str, output_dir: str) -> bool:
    """Run evaluation for a specific task"""
    task_config = TASK_CONFIGS[task_key]
    script_path = os.path.join(os.path.dirname(__file__), task_config['script'])
    
    if not os.path.exists(script_path):
        logger.error(f"Task script not found: {script_path}")
        return False
    
    logger.info(f"Running {task_config['name']} evaluation...")
    
    try:
        # Run the task script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=3600*2)  # 2 hour timeout
        
        if result.returncode == 0:
            logger.info(f"✅ {task_config['name']} evaluation completed successfully")
            return True
        else:
            logger.error(f"❌ {task_config['name']} evaluation failed:")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {task_config['name']} evaluation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error running {task_config['name']} evaluation: {e}")
        return False

def collect_results(base_output_dir: str) -> dict:
    """Collect results from all task evaluations"""
    all_results = {}
    
    # Find all result directories
    if not os.path.exists(base_output_dir):
        logger.error(f"Output directory not found: {base_output_dir}")
        return all_results
    
    for task_key, task_config in TASK_CONFIGS.items():
        task_results = {}
        
        # Look for task evaluation directories
        task_pattern = f"{task_key}_evaluation_"
        
        for subdir in os.listdir(base_output_dir):
            if subdir.startswith(task_pattern):
                task_dir = os.path.join(base_output_dir, subdir)
                summary_file = os.path.join(task_dir, f"{task_key}_summary.json")
                
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary_data = json.load(f)
                        
                        results_summary = summary_data.get('results_summary', {})
                        for model_name, metrics in results_summary.items():
                            if model_name not in task_results:
                                task_results[model_name] = {}
                            task_results[model_name] = metrics
                            
                    except Exception as e:
                        logger.error(f"Error reading {summary_file}: {e}")
        
        if task_results:
            all_results[task_key] = task_results
    
    return all_results

def create_results_table(all_results: dict, output_dir: str):
    """Create comprehensive results table"""
    # Collect all model names
    all_models = set()
    for task_results in all_results.values():
        all_models.update(task_results.keys())
    all_models = sorted(list(all_models))
    
    # Create results matrix
    results_data = []
    
    for model_name in all_models:
        row = {'Model': model_name}
        
        for task_key, task_config in TASK_CONFIGS.items():
            primary_metric = task_config['primary_metric']
            task_results = all_results.get(task_key, {})
            model_results = task_results.get(model_name, {})
            
            # Get primary metric value
            metric_value = model_results.get(primary_metric, 0.0)
            row[f"{task_key.upper()}_{primary_metric}"] = f"{metric_value:.4f}"
            
            # Add secondary metrics if available
            if task_key == 'tc' and 'f1_macro' in model_results:
                row[f"{task_key.upper()}_f1_macro"] = f"{model_results['f1_macro']:.4f}"
            elif task_key == 'nli' and 'f1_macro' in model_results:
                row[f"{task_key.upper()}_f1_macro"] = f"{model_results['f1_macro']:.4f}"
            elif task_key == 'sts' and 'p_value' in model_results:
                row[f"{task_key.upper()}_p_value"] = f"{model_results['p_value']:.6f}"
            elif task_key == 'ner':
                if 'precision' in model_results:
                    row[f"{task_key.upper()}_precision"] = f"{model_results['precision']:.4f}"
                if 'recall' in model_results:
                    row[f"{task_key.upper()}_recall"] = f"{model_results['recall']:.4f}"
            elif task_key == 're' and 'accuracy' in model_results:
                row[f"{task_key.upper()}_accuracy"] = f"{model_results['accuracy']:.4f}"
            elif task_key == 'dp' and 'uas' in model_results:
                row[f"{task_key.upper()}_uas"] = f"{model_results['uas']:.4f}"
            elif task_key == 'mrc' and 'exact_match' in model_results:
                row[f"{task_key.upper()}_exact_match"] = f"{model_results['exact_match']:.4f}"
            elif task_key == 'dst' and 'slot_accuracy' in model_results:
                row[f"{task_key.upper()}_slot_accuracy"] = f"{model_results['slot_accuracy']:.4f}"
        
        results_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(results_data)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, "klue_benchmark_results.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Results table saved to: {csv_file}")
    
    # Save as JSON
    json_file = os.path.join(output_dir, "klue_benchmark_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'task_configs': TASK_CONFIGS,
            'results': all_results,
            'results_table': results_data
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Results JSON saved to: {json_file}")
    
    return df

def print_summary(all_results: dict):
    """Print evaluation summary"""
    logger.info(f"\n{'='*80}")
    logger.info("KLUE BENCHMARK EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    
    # Print task-wise results
    for task_key, task_config in TASK_CONFIGS.items():
        task_results = all_results.get(task_key, {})
        if not task_results:
            logger.info(f"\n{task_config['name']} ({task_key.upper()}): No results")
            continue
        
        logger.info(f"\n{task_config['name']} ({task_key.upper()}):")
        logger.info("-" * 50)
        
        primary_metric = task_config['primary_metric']
        
        # Sort models by primary metric
        sorted_models = sorted(
            task_results.items(),
            key=lambda x: x[1].get(primary_metric, 0.0),
            reverse=True
        )
        
        for model_name, metrics in sorted_models:
            metric_value = metrics.get(primary_metric, 0.0)
            logger.info(f"  {model_name:40} {primary_metric}: {metric_value:.4f}")
    
    # Print model-wise summary
    logger.info(f"\n{'='*50}")
    logger.info("MODEL PERFORMANCE OVERVIEW")
    logger.info(f"{'='*50}")
    
    all_models = set()
    for task_results in all_results.values():
        all_models.update(task_results.keys())
    
    for model_name in sorted(all_models):
        logger.info(f"\n{model_name}:")
        for task_key, task_config in TASK_CONFIGS.items():
            task_results = all_results.get(task_key, {})
            model_results = task_results.get(model_name, {})
            primary_metric = task_config['primary_metric']
            metric_value = model_results.get(primary_metric, 0.0)
            
            logger.info(f"  {task_key.upper():4} ({primary_metric:20}): {metric_value:.4f}")

def main():
    """Main function to run all KLUE evaluations"""
    parser = argparse.ArgumentParser(description='Run KLUE benchmark on all models')
    parser.add_argument('--tasks', nargs='+', choices=list(TASK_CONFIGS.keys()), 
                       default=list(TASK_CONFIGS.keys()), help='Tasks to evaluate')
    parser.add_argument('--skip-evaluation', action='store_true', 
                       help='Skip evaluation and only collect results')
    
    args = parser.parse_args()
    
    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(BASE_OUTPUT_DIR, f"klue_full_benchmark_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    logger.info(f"Starting KLUE benchmark evaluation")
    logger.info(f"Main output directory: {main_output_dir}")
    logger.info(f"Tasks to evaluate: {args.tasks}")
    logger.info(f"Models to evaluate: {len(MODEL_CONFIGS)}")
    
    # Run evaluations
    if not args.skip_evaluation:
        completed_tasks = []
        failed_tasks = []
        
        for task_key in args.tasks:
            if task_key not in TASK_CONFIGS:
                logger.warning(f"Unknown task: {task_key}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting task: {TASK_CONFIGS[task_key]['name']} ({task_key.upper()})")
            logger.info(f"{'='*60}")
            
            if run_task_evaluation(task_key, main_output_dir):
                completed_tasks.append(task_key)
            else:
                failed_tasks.append(task_key)
        
        logger.info(f"\n{'='*60}")
        logger.info("TASK COMPLETION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Completed tasks: {completed_tasks}")
        if failed_tasks:
            logger.info(f"Failed tasks: {failed_tasks}")
    
    # Collect and analyze results
    logger.info(f"\n{'='*60}")
    logger.info("COLLECTING RESULTS")
    logger.info(f"{'='*60}")
    
    all_results = collect_results(BASE_OUTPUT_DIR)
    
    if all_results:
        # Create comprehensive results table
        results_table = create_results_table(all_results, main_output_dir)
        
        # Print summary
        print_summary(all_results)
        
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Results saved to: {main_output_dir}")
        logger.info(f"Total tasks evaluated: {len(all_results)}")
        logger.info(f"Total models evaluated: {len(set().union(*[task_results.keys() for task_results in all_results.values()]))}")
    
    else:
        logger.warning("No results found to collect")

if __name__ == "__main__":
    main()