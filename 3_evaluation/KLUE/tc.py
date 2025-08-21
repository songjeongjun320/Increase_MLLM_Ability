#!/usr/bin/env python3
"""
KLUE Topic Classification (TC) Benchmark
Evaluates Korean language understanding through news topic classification
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS, KLUE_TASKS, PROMPT_TEMPLATES, BASE_OUTPUT_DIR, DATA_DIR
from utils import ModelLoader, load_data, save_results, run_evaluation

logger = logging.getLogger(__name__)

def evaluate_tc_task(model_config, output_dir: str, max_samples: int = None):
    """Evaluate Topic Classification task for a single model"""
    logger.info(f"Starting TC evaluation for {model_config.name}")
    
    # Load data
    validation_file = os.path.join(DATA_DIR, "klue_tc_validation.json")
    validation_data = load_data(validation_file)
    
    if not validation_data:
        logger.error(f"No validation data found for TC task")
        return None
    
    # Initialize model loader
    model_loader = ModelLoader(model_config)
    
    try:
        # Load model
        model_loader.load_model()
        
        # Run evaluation
        results = run_evaluation(
            model_loader=model_loader,
            data=validation_data,
            task_type='tc',
            prompt_template=PROMPT_TEMPLATES['tc'],
            max_samples=max_samples
        )
        
        # Add model info
        results['model_name'] = model_config.name
        results['model_config'] = {
            'model_id': model_config.model_id,
            'adapter_path': model_config.adapter_path,
            'use_quantization': model_config.use_quantization
        }
        results['timestamp'] = datetime.now().isoformat()
        
        # Save results
        output_file = os.path.join(output_dir, f"{model_config.name}_tc_results.json")
        save_results(results, output_file)
        
        # Log metrics
        metrics = results['metrics']
        logger.info(f"TC Results for {model_config.name}:")
        logger.info(f"  Macro F1 (Official): {metrics.get('f1_macro', 0.0):.4f}")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating TC for {model_config.name}: {e}")
        return None
    
    finally:
        # Always unload model
        model_loader.unload_model()

def main():
    """Main function to run TC evaluation on all models"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"tc_evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting KLUE TC evaluation for {len(MODEL_CONFIGS)} models")
    logger.info(f"Output directory: {output_dir}")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_config.name}")
        logger.info(f"{'='*50}")
        
        try:
            results = evaluate_tc_task(model_config, output_dir, max_samples=500)  # Limit for testing
            if results:
                all_results[model_config.name] = results['metrics']
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_config.name}: {e}")
            continue
    
    # Save summary results
    summary = {
        'task': 'Topic Classification (TC)',
        'timestamp': datetime.now().isoformat(),
        'results_summary': all_results
    }
    
    summary_file = os.path.join(output_dir, "tc_summary.json")
    save_results(summary, summary_file)
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TC EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_name, metrics in all_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Macro F1 (Official): {metrics.get('f1_macro', 0.0):.4f}")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        logger.info("")

if __name__ == "__main__":
    main()