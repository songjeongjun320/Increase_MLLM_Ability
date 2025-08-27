#!/usr/bin/env python3
"""
KLUE Named Entity Recognition (NER) Benchmark
Evaluates named entity recognition capabilities in Korean text
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

def extract_ner_entities(text: str):
    """Extract named entities from text"""
    # Simple extraction logic - can be enhanced
    lines = text.split('\n')
    entities = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for patterns like "entity: type" or "entity (type)"
        patterns = [
            r'(.+?)\s*:\s*(.+)',  # entity: type
            r'(.+?)\s*\((.+?)\)',  # entity (type)
            r'(.+?)\s*-\s*(.+)',   # entity - type
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                entity = match.group(1).strip()
                entity_type = match.group(2).strip()
                entities.append((entity, entity_type))
                break
    
    return entities

def compute_ner_metrics(predictions, references):
    """Compute NER-specific metrics (F1, precision, recall)"""
    # Convert to sets of (entity, type) tuples for comparison
    pred_sets = []
    ref_sets = []
    
    for pred, ref in zip(predictions, references):
        # Extract entities from prediction
        if isinstance(pred, str):
            pred_entities = extract_ner_entities(pred)
        else:
            pred_entities = pred if isinstance(pred, list) else []
        
        # Process reference (assuming it's in some structured format)
        if isinstance(ref, list):
            ref_entities = ref
        elif isinstance(ref, str):
            ref_entities = extract_ner_entities(ref)
        else:
            ref_entities = []
        
        pred_sets.append(set(pred_entities))
        ref_sets.append(set(ref_entities))
    
    # Compute micro-averaged metrics
    total_pred = sum(len(s) for s in pred_sets)
    total_ref = sum(len(s) for s in ref_sets)
    total_correct = sum(len(p & r) for p, r in zip(pred_sets, ref_sets))
    
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_ref if total_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_predicted': total_pred,
        'total_reference': total_ref,
        'total_correct': total_correct
    }

def evaluate_ner_sample(model_loader: ModelLoader, sample: dict) -> tuple:
    """Evaluate a single NER sample"""
    # Extract sentence from sample
    sentence = sample.get('sentence', sample.get('text', ''))
    
    # Create prompt
    prompt = PROMPT_TEMPLATES['ner'].format(sentence=sentence)
    
    # Generate prediction
    try:
        prediction = model_loader.generate_text(prompt, max_new_tokens=256, temperature=0.1)
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        prediction = ""
    
    # Get reference entities
    reference = sample.get('entities', sample.get('ne', []))
    
    return prediction, reference

def evaluate_ner_task(model_config, output_dir: str, max_samples: int = None):
    """Evaluate Named Entity Recognition task for a single model"""
    logger.info(f"Starting NER evaluation for {model_config.name}")
    
    # Load data
    validation_file = os.path.join(DATA_DIR, "klue_ner_validation.json")
    validation_data = load_data(validation_file)
    
    if not validation_data:
        logger.error(f"No validation data found for NER task")
        return None
    
    if max_samples:
        validation_data = validation_data[:max_samples]
    
    # Initialize model loader
    model_loader = ModelLoader(model_config)
    
    try:
        # Load model
        model_loader.load_model()
        
        logger.info(f"Evaluating {len(validation_data)} samples for NER")
        
        predictions = []
        references = []
        
        error_count = 0
        pbar = tqdm(enumerate(validation_data), desc=f"Evaluating NER (errors: 0)", total=len(validation_data))
        for i, sample in pbar:
            try:
                pred, ref = evaluate_ner_sample(model_loader, sample)
                predictions.append(pred)
                references.append(ref)
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                error_count += 1
                predictions.append("")
                references.append([])
            
            # Update progress bar with current error count
            pbar.set_description(f"Evaluating NER (errors: {error_count})")
            
            if i % 50 == 0:
                logger.info(f"Processing sample {i+1}/{len(validation_data)}")
        
        # Compute metrics
        metrics = compute_ner_metrics(predictions, references)
        
        # Prepare results
        results = {
            'task_type': 'ner',
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
        output_file = os.path.join(output_dir, f"{model_config.name}_ner_results.json")
        save_results(results, output_file)
        
        # Log metrics
        logger.info(f"NER Results for {model_config.name}:")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating NER for {model_config.name}: {e}")
        return None
    
    finally:
        # Always unload model
        model_loader.unload_model()

def main():
    """Main function to run NER evaluation on all models"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"ner_evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting KLUE NER evaluation for {len(MODEL_CONFIGS)} models")
    logger.info(f"Output directory: {output_dir}")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_config.name}")
        logger.info(f"{'='*50}")
        
        try:
            results = evaluate_ner_task(model_config, output_dir, max_samples=200)  # Limit for testing
            if results:
                all_results[model_config.name] = results['metrics']
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_config.name}: {e}")
            continue
    
    # Save summary results
    summary = {
        'task': 'Named Entity Recognition (NER)',
        'timestamp': datetime.now().isoformat(),
        'results_summary': all_results
    }
    
    summary_file = os.path.join(output_dir, "ner_summary.json")
    save_results(summary, summary_file)
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("NER EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_name, metrics in all_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info("")

if __name__ == "__main__":
    main()