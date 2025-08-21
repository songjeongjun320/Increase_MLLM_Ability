#!/usr/bin/env python3
"""
KLUE Machine Reading Comprehension (MRC) Benchmark
Evaluates reading comprehension and question answering capabilities in Korean
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS, KLUE_TASKS, PROMPT_TEMPLATES, BASE_OUTPUT_DIR, DATA_DIR
from utils import ModelLoader, load_data, save_results

logger = logging.getLogger(__name__)

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison"""
    # Remove extra whitespaces and punctuation
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def compute_f1_score(pred: str, ref: str) -> float:
    """Compute F1 score between prediction and reference"""
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Compute precision and recall
    common_tokens = set(pred_tokens) & set(ref_tokens)
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_exact_match(pred: str, ref: str) -> float:
    """Compute exact match score"""
    return 1.0 if normalize_answer(pred) == normalize_answer(ref) else 0.0

def compute_mrc_metrics(predictions, references):
    """Compute MRC-specific metrics (EM, F1)"""
    exact_matches = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # Handle multiple reference answers
        if isinstance(ref, list):
            max_em = max(compute_exact_match(pred, r) for r in ref)
            max_f1 = max(compute_f1_score(pred, r) for r in ref)
        else:
            max_em = compute_exact_match(pred, str(ref))
            max_f1 = compute_f1_score(pred, str(ref))
        
        exact_matches.append(max_em)
        f1_scores.append(max_f1)
    
    return {
        'exact_match': sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'num_samples': len(predictions)
    }

def evaluate_mrc_sample(model_loader: ModelLoader, sample: dict) -> tuple:
    """Evaluate a single MRC sample"""
    # Extract information
    context = sample.get('context', sample.get('passage', ''))
    question = sample.get('question', '')
    
    # Create prompt
    prompt = PROMPT_TEMPLATES['mrc'].format(context=context, question=question)
    
    # Generate prediction
    try:
        prediction = model_loader.generate_text(prompt, max_new_tokens=256, temperature=0.1)
        # Extract answer (usually first line or until first newline)
        prediction = prediction.split('\n')[0].strip()
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        prediction = ""
    
    # Get reference answer(s)
    reference = sample.get('answers', sample.get('answer', ''))
    if isinstance(reference, dict) and 'text' in reference:
        reference = reference['text']
    elif isinstance(reference, list) and len(reference) > 0 and isinstance(reference[0], dict):
        reference = [ans.get('text', str(ans)) for ans in reference]
    
    return prediction, reference

def evaluate_mrc_task(model_config, output_dir: str, max_samples: int = None):
    """Evaluate Machine Reading Comprehension task for a single model"""
    logger.info(f"Starting MRC evaluation for {model_config.name}")
    
    # Load data
    validation_file = os.path.join(DATA_DIR, "klue_mrc_validation.json")
    validation_data = load_data(validation_file)
    
    if not validation_data:
        logger.error(f"No validation data found for MRC task")
        return None
    
    if max_samples:
        validation_data = validation_data[:max_samples]
    
    # Initialize model loader
    model_loader = ModelLoader(model_config)
    
    try:
        # Load model
        model_loader.load_model()
        
        logger.info(f"Evaluating {len(validation_data)} samples for MRC")
        
        predictions = []
        references = []
        
        for i, sample in enumerate(validation_data):
            if i % 20 == 0:
                logger.info(f"Processing sample {i+1}/{len(validation_data)}")
            
            pred, ref = evaluate_mrc_sample(model_loader, sample)
            predictions.append(pred)
            references.append(ref)
        
        # Compute metrics
        metrics = compute_mrc_metrics(predictions, references)
        
        # Prepare results
        results = {
            'task_type': 'mrc',
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
        output_file = os.path.join(output_dir, f"{model_config.name}_mrc_results.json")
        save_results(results, output_file)
        
        # Log metrics
        logger.info(f"MRC Results for {model_config.name}:")
        logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating MRC for {model_config.name}: {e}")
        return None
    
    finally:
        # Always unload model
        model_loader.unload_model()

def main():
    """Main function to run MRC evaluation on all models"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"mrc_evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting KLUE MRC evaluation for {len(MODEL_CONFIGS)} models")
    logger.info(f"Output directory: {output_dir}")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_config.name}")
        logger.info(f"{'='*50}")
        
        try:
            results = evaluate_mrc_task(model_config, output_dir, max_samples=100)  # Limit for testing
            if results:
                all_results[model_config.name] = results['metrics']
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_config.name}: {e}")
            continue
    
    # Save summary results
    summary = {
        'task': 'Machine Reading Comprehension (MRC)',
        'timestamp': datetime.now().isoformat(),
        'results_summary': all_results
    }
    
    summary_file = os.path.join(output_dir, "mrc_summary.json")
    save_results(summary, summary_file)
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("MRC EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_name, metrics in all_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info("")

if __name__ == "__main__":
    main()