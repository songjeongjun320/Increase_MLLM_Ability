#!/usr/bin/env python3
"""
KLUE Relation Extraction (RE) Benchmark
Evaluates relation extraction capabilities between entities in Korean text
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
from utils import ModelLoader, load_data, save_results, compute_metrics

logger = logging.getLogger(__name__)

def evaluate_re_sample(model_loader: ModelLoader, sample: dict) -> tuple:
    """Evaluate a single Relation Extraction sample"""
    # Extract information from sample
    sentence = sample.get('sentence', '')
    entity1 = sample.get('subject_entity', {}).get('word', '')
    entity2 = sample.get('object_entity', {}).get('word', '')
    
    # Create prompt
    prompt = PROMPT_TEMPLATES['re'].format(
        sentence=sentence,
        entity1=entity1,
        entity2=entity2
    )
    
    # Generate prediction
    try:
        prediction = model_loader.generate_text(prompt, max_new_tokens=128, temperature=0.1)
        # Extract first line as relation
        prediction = prediction.split('\n')[0].strip()
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        prediction = "no_relation"
    
    # Get reference label
    reference = sample.get('label', 'no_relation')
    
    return prediction, reference

def compute_re_metrics(predictions, references):
    """Compute RE-specific metrics"""
    # Basic accuracy
    correct = sum(1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip())
    accuracy = correct / len(predictions) if predictions else 0.0
    
    # Get unique labels
    unique_labels = list(set(references + predictions))
    
    # Compute per-class metrics
    per_class_metrics = {}
    for label in unique_labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for r in references if r == label)
        }
    
    # Macro averaged F1
    macro_f1 = sum(metrics['f1'] for metrics in per_class_metrics.values()) / len(per_class_metrics)
    
    # Micro averaged F1 (KLUE official metric for RE)
    total_tp = sum(sum(1 for p, r in zip(predictions, references) if p == label and r == label) for label in unique_labels)
    total_fp = sum(sum(1 for p, r in zip(predictions, references) if p == label and r != label) for label in unique_labels)
    total_fn = sum(sum(1 for p, r in zip(predictions, references) if p != label and r == label) for label in unique_labels)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'f1_micro': micro_f1,  # KLUE official primary metric
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_class_metrics': per_class_metrics,
        'num_classes': len(unique_labels)
    }

def evaluate_re_task(model_config, output_dir: str, max_samples: int = None):
    """Evaluate Relation Extraction task for a single model"""
    logger.info(f"Starting RE evaluation for {model_config.name}")
    
    # Load data
    validation_file = os.path.join(DATA_DIR, "klue_re_validation.json")
    validation_data = load_data(validation_file)
    
    if not validation_data:
        logger.error(f"No validation data found for RE task")
        return None
    
    if max_samples:
        validation_data = validation_data[:max_samples]
    
    # Initialize model loader
    model_loader = ModelLoader(model_config)
    
    try:
        # Load model
        model_loader.load_model()
        
        logger.info(f"Evaluating {len(validation_data)} samples for RE")
        
        predictions = []
        references = []
        
        for i, sample in enumerate(validation_data):
            if i % 50 == 0:
                logger.info(f"Processing sample {i+1}/{len(validation_data)}")
            
            pred, ref = evaluate_re_sample(model_loader, sample)
            predictions.append(pred)
            references.append(ref)
        
        # Compute metrics
        metrics = compute_re_metrics(predictions, references)
        
        # Prepare results
        results = {
            'task_type': 're',
            'model_name': model_config.name,
            'model_config': {
                'model_id': model_config.model_id,
                'adapter_path': model_config.adapter_path,
                'use_quantization': model_config.use_quantization
            },
            'num_samples': len(validation_data),
            'metrics': metrics,
            'predictions': predictions,
            'references': references,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        output_file = os.path.join(output_dir, f"{model_config.name}_re_results.json")
        save_results(results, output_file)
        
        # Log metrics
        logger.info(f"RE Results for {model_config.name}:")
        logger.info(f"  Micro F1 (Official): {metrics['f1_micro']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Number of classes: {metrics['num_classes']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating RE for {model_config.name}: {e}")
        return None
    
    finally:
        # Always unload model
        model_loader.unload_model()

def main():
    """Main function to run RE evaluation on all models"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"re_evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting KLUE RE evaluation for {len(MODEL_CONFIGS)} models")
    logger.info(f"Output directory: {output_dir}")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_config.name}")
        logger.info(f"{'='*50}")
        
        try:
            results = evaluate_re_task(model_config, output_dir, max_samples=200)  # Limit for testing
            if results:
                all_results[model_config.name] = results['metrics']
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_config.name}: {e}")
            continue
    
    # Save summary results
    summary = {
        'task': 'Relation Extraction (RE)',
        'timestamp': datetime.now().isoformat(),
        'results_summary': all_results
    }
    
    summary_file = os.path.join(output_dir, "re_summary.json")
    save_results(summary, summary_file)
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("RE EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_name, metrics in all_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Micro F1 (Official): {metrics['f1_micro']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info("")

if __name__ == "__main__":
    main()