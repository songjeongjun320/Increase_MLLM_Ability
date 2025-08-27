#!/usr/bin/env python3
"""
KLUE Dependency Parsing (DP) Benchmark
Evaluates syntactic parsing capabilities for Korean sentences
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

def parse_dependency_output(text: str):
    """Parse dependency parsing output from model"""
    # Simple parsing - can be enhanced with more sophisticated parsing
    dependencies = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for patterns like "word1 -> word2 (relation)"
        patterns = [
            r'(\w+)\s*->\s*(\w+)\s*\((.+?)\)',  # word1 -> word2 (relation)
            r'(\w+)\s*:\s*(\w+)\s*\((.+?)\)',   # word1 : word2 (relation)
            r'(\d+)\s+(\w+)\s+(\d+)\s+(.+)',    # index word head_index relation
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                if len(match.groups()) == 3:
                    word1, word2, relation = match.groups()
                    dependencies.append((word1.strip(), word2.strip(), relation.strip()))
                elif len(match.groups()) == 4:
                    idx, word, head_idx, relation = match.groups()
                    dependencies.append((word.strip(), head_idx.strip(), relation.strip()))
                break
    
    return dependencies

def compute_dp_metrics(predictions, references):
    """Compute dependency parsing metrics (UAS, LAS)"""
    total_tokens = 0
    correct_unlabeled = 0
    correct_labeled = 0
    
    for pred, ref in zip(predictions, references):
        # Parse predictions and references
        if isinstance(pred, str):
            pred_deps = parse_dependency_output(pred)
        else:
            pred_deps = pred if isinstance(pred, list) else []
        
        if isinstance(ref, list):
            ref_deps = ref
        elif isinstance(ref, dict):
            # Handle different reference formats
            ref_deps = []
            if 'dependencies' in ref:
                ref_deps = ref['dependencies']
        else:
            ref_deps = []
        
        # Convert to standardized format for comparison
        pred_dict = {}
        ref_dict = {}
        
        for dep in pred_deps:
            if len(dep) >= 3:
                token, head, label = dep[0], dep[1], dep[2]
                pred_dict[token] = (head, label)
        
        for dep in ref_deps:
            if isinstance(dep, dict) and 'word' in dep:
                token = dep['word']
                head = dep.get('head', '')
                label = dep.get('label', '')
                ref_dict[token] = (head, label)
            elif len(dep) >= 3:
                token, head, label = dep[0], dep[1], dep[2]
                ref_dict[token] = (head, label)
        
        # Count tokens and matches
        all_tokens = set(pred_dict.keys()) | set(ref_dict.keys())
        total_tokens += len(all_tokens)
        
        for token in all_tokens:
            pred_head, pred_label = pred_dict.get(token, ('', ''))
            ref_head, ref_label = ref_dict.get(token, ('', ''))
            
            # Unlabeled attachment (UAS)
            if pred_head == ref_head:
                correct_unlabeled += 1
                
                # Labeled attachment (LAS)
                if pred_label == ref_label:
                    correct_labeled += 1
    
    uas = correct_unlabeled / total_tokens if total_tokens > 0 else 0.0
    las = correct_labeled / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'uas': uas,  # Unlabeled Attachment Score
        'las': las,  # Labeled Attachment Score
        'total_tokens': total_tokens,
        'correct_unlabeled': correct_unlabeled,
        'correct_labeled': correct_labeled
    }

def evaluate_dp_sample(model_loader: ModelLoader, sample: dict) -> tuple:
    """Evaluate a single dependency parsing sample"""
    # Extract sentence
    sentence = sample.get('sentence', '')
    
    # Create prompt
    prompt = PROMPT_TEMPLATES['dp'].format(sentence=sentence)
    
    # Generate prediction
    try:
        prediction = model_loader.generate_text(prompt, max_new_tokens=512, temperature=0.1)
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        prediction = ""
    
    # Get reference dependencies
    reference = sample.get('dependencies', sample.get('word_form', []))
    
    return prediction, reference

def evaluate_dp_task(model_config, output_dir: str, max_samples: int = None):
    """Evaluate Dependency Parsing task for a single model"""
    logger.info(f"Starting DP evaluation for {model_config.name}")
    
    # Load data
    validation_file = os.path.join(DATA_DIR, "klue_dp_validation.json")
    validation_data = load_data(validation_file)
    
    if not validation_data:
        logger.error(f"No validation data found for DP task")
        return None
    
    if max_samples:
        validation_data = validation_data[:max_samples]
    
    # Initialize model loader
    model_loader = ModelLoader(model_config)
    
    try:
        # Load model
        model_loader.load_model()
        
        logger.info(f"Evaluating {len(validation_data)} samples for DP")
        
        predictions = []
        references = []
        
        for i, sample in enumerate(validation_data):
            if i % 50 == 0:
                logger.info(f"Processing sample {i+1}/{len(validation_data)}")
            
            pred, ref = evaluate_dp_sample(model_loader, sample)
            predictions.append(pred)
            references.append(ref)
        
        # Compute metrics
        metrics = compute_dp_metrics(predictions, references)
        
        # Prepare results
        results = {
            'task_type': 'dp',
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
        output_file = os.path.join(output_dir, f"{model_config.name}_dp_results.json")
        save_results(results, output_file)
        
        # Log metrics
        logger.info(f"DP Results for {model_config.name}:")
        logger.info(f"  UAS (Unlabeled Attachment Score): {metrics['uas']:.4f}")
        logger.info(f"  LAS (Labeled Attachment Score): {metrics['las']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating DP for {model_config.name}: {e}")
        return None
    
    finally:
        # Always unload model
        model_loader.unload_model()

def main():
    """Main function to run DP evaluation on all models"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"dp_evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting KLUE DP evaluation for {len(MODEL_CONFIGS)} models")
    logger.info(f"Output directory: {output_dir}")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_config.name}")
        logger.info(f"{'='*50}")
        
        try:
            results = evaluate_dp_task(model_config, output_dir, max_samples=100)  # Limit for testing
            if results:
                all_results[model_config.name] = results['metrics']
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_config.name}: {e}")
            continue
    
    # Save summary results
    summary = {
        'task': 'Dependency Parsing (DP)',
        'timestamp': datetime.now().isoformat(),
        'results_summary': all_results
    }
    
    summary_file = os.path.join(output_dir, "dp_summary.json")
    save_results(summary, summary_file)
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("DP EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_name, metrics in all_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  UAS: {metrics['uas']:.4f}")
        logger.info(f"  LAS: {metrics['las']:.4f}")
        logger.info("")

if __name__ == "__main__":
    main()