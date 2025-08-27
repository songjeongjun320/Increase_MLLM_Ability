#!/usr/bin/env python3
"""
KLUE Dialogue State Tracking (DST) Benchmark
Evaluates dialogue state tracking capabilities for Korean task-oriented dialogue
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

def parse_dialogue_state(text: str):
    """Parse dialogue state from model output"""
    # Extract intent and slots from text
    intent = None
    slots = {}
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for intent
        intent_patterns = [
            r'의도\s*:\s*(.+)',
            r'intent\s*:\s*(.+)',
            r'목적\s*:\s*(.+)'
        ]
        
        for pattern in intent_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                intent = match.group(1).strip()
                break
        
        # Look for slots
        slot_patterns = [
            r'(.+?)\s*:\s*(.+)',  # key: value
            r'(.+?)\s*=\s*(.+)',  # key = value
        ]
        
        for pattern in slot_patterns:
            match = re.search(pattern, line)
            if match and match.group(1).strip().lower() not in ['의도', 'intent', '목적']:
                slot_name = match.group(1).strip()
                slot_value = match.group(2).strip()
                slots[slot_name] = slot_value
                break
    
    return {
        'intent': intent,
        'slots': slots
    }

def compute_joint_goal_accuracy(predictions, references):
    """Compute Joint Goal Accuracy for DST"""
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Parse prediction
        if isinstance(pred, str):
            pred_state = parse_dialogue_state(pred)
        else:
            pred_state = pred
        
        # Handle reference format
        if isinstance(ref, dict):
            ref_state = ref
        else:
            ref_state = {'intent': None, 'slots': {}}
        
        # Check if intent and all slots match
        intent_match = pred_state.get('intent') == ref_state.get('intent')
        
        pred_slots = pred_state.get('slots', {})
        ref_slots = ref_state.get('slots', {})
        
        # Normalize slot keys and values
        pred_slots_norm = {k.lower(): v.lower() for k, v in pred_slots.items()}
        ref_slots_norm = {k.lower(): v.lower() for k, v in ref_slots.items()}
        
        slots_match = pred_slots_norm == ref_slots_norm
        
        if intent_match and slots_match:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def compute_slot_accuracy(predictions, references):
    """Compute slot-level accuracy"""
    total_slots = 0
    correct_slots = 0
    
    for pred, ref in zip(predictions, references):
        # Parse prediction
        if isinstance(pred, str):
            pred_state = parse_dialogue_state(pred)
        else:
            pred_state = pred
        
        if isinstance(ref, dict):
            ref_state = ref
        else:
            ref_state = {'slots': {}}
        
        pred_slots = pred_state.get('slots', {})
        ref_slots = ref_state.get('slots', {})
        
        # Normalize
        pred_slots_norm = {k.lower(): v.lower() for k, v in pred_slots.items()}
        ref_slots_norm = {k.lower(): v.lower() for k, v in ref_slots.items()}
        
        # Count all slots mentioned in reference
        all_slot_keys = set(ref_slots_norm.keys()) | set(pred_slots_norm.keys())
        
        for slot_key in all_slot_keys:
            total_slots += 1
            pred_value = pred_slots_norm.get(slot_key, "")
            ref_value = ref_slots_norm.get(slot_key, "")
            
            if pred_value == ref_value:
                correct_slots += 1
    
    return correct_slots / total_slots if total_slots > 0 else 0.0

def evaluate_dst_sample(model_loader: ModelLoader, sample: dict) -> tuple:
    """Evaluate a single DST sample"""
    # Extract dialogue
    dialogue_turns = sample.get('dialogue', [])
    
    # Format dialogue context
    dialogue_text = ""
    for turn in dialogue_turns:
        speaker = turn.get('speaker', 'Unknown')
        utterance = turn.get('text', '')
        dialogue_text += f"{speaker}: {utterance}\n"
    
    # Create prompt
    prompt = PROMPT_TEMPLATES['dst'].format(dialogue=dialogue_text.strip())
    
    # Generate prediction
    try:
        prediction = model_loader.generate_text(prompt, max_new_tokens=256, temperature=0.1)
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        prediction = ""
    
    # Get reference state
    reference = sample.get('state', sample.get('belief_state', {}))
    
    return prediction, reference

def evaluate_dst_task(model_config, output_dir: str, max_samples: int = None):
    """Evaluate Dialogue State Tracking task for a single model"""
    logger.info(f"Starting DST evaluation for {model_config.name}")
    
    # Load data
    validation_file = os.path.join(DATA_DIR, "klue_dst_validation.json")
    validation_data = load_data(validation_file)
    
    if not validation_data:
        logger.error(f"No validation data found for DST task")
        return None
    
    if max_samples:
        validation_data = validation_data[:max_samples]
    
    # Initialize model loader
    model_loader = ModelLoader(model_config)
    
    try:
        # Load model
        model_loader.load_model()
        
        logger.info(f"Evaluating {len(validation_data)} samples for DST")
        
        predictions = []
        references = []
        
        for i, sample in enumerate(validation_data):
            if i % 20 == 0:
                logger.info(f"Processing sample {i+1}/{len(validation_data)}")
            
            pred, ref = evaluate_dst_sample(model_loader, sample)
            predictions.append(pred)
            references.append(ref)
        
        # Compute metrics
        jga = compute_joint_goal_accuracy(predictions, references)
        slot_acc = compute_slot_accuracy(predictions, references)
        
        metrics = {
            'joint_goal_accuracy': jga,
            'slot_accuracy': slot_acc
        }
        
        # Prepare results
        results = {
            'task_type': 'dst',
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
        output_file = os.path.join(output_dir, f"{model_config.name}_dst_results.json")
        save_results(results, output_file)
        
        # Log metrics
        logger.info(f"DST Results for {model_config.name}:")
        logger.info(f"  Joint Goal Accuracy: {metrics['joint_goal_accuracy']:.4f}")
        logger.info(f"  Slot Accuracy: {metrics['slot_accuracy']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating DST for {model_config.name}: {e}")
        return None
    
    finally:
        # Always unload model
        model_loader.unload_model()

def main():
    """Main function to run DST evaluation on all models"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"dst_evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting KLUE DST evaluation for {len(MODEL_CONFIGS)} models")
    logger.info(f"Output directory: {output_dir}")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model_config.name}")
        logger.info(f"{'='*50}")
        
        try:
            results = evaluate_dst_task(model_config, output_dir, max_samples=50)  # Limit for testing
            if results:
                all_results[model_config.name] = results['metrics']
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_config.name}: {e}")
            continue
    
    # Save summary results
    summary = {
        'task': 'Dialogue State Tracking (DST)',
        'timestamp': datetime.now().isoformat(),
        'results_summary': all_results
    }
    
    summary_file = os.path.join(output_dir, "dst_summary.json")
    save_results(summary, summary_file)
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("DST EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_name, metrics in all_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Joint Goal Accuracy: {metrics['joint_goal_accuracy']:.4f}")
        logger.info(f"  Slot Accuracy: {metrics['slot_accuracy']:.4f}")
        logger.info("")

if __name__ == "__main__":
    main()