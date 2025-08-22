#!/usr/bin/env python3
"""
KLUE Benchmark Utilities
Common utility functions for KLUE evaluation
"""

import os
import json
import torch
import gc
import logging
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
import re

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles model loading and management for KLUE evaluation"""
    
    def __init__(self, model_config, device="cuda"):
        self.config = model_config
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"🔄 Starting model load: {self.config.name}")
        logger.info(f"📍 Model path: {self.config.model_id}")
        logger.info(f"📍 Adapter path: {self.config.adapter_path}")
        logger.info(f"⚙️ Use quantization: {self.config.use_quantization}")
        logger.info(f"🖥️ Device: {self.device}")
        logger.info(f"🔧 Torch dtype: {self.config.torch_dtype}")
        
        # Check if model path exists
        if not os.path.exists(self.config.model_id):
            error_msg = f"❌ Model path does not exist: {self.config.model_id}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check if adapter path exists (if specified)
        if self.config.adapter_path and not os.path.exists(self.config.adapter_path):
            error_msg = f"❌ Adapter path does not exist: {self.config.adapter_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Setup quantization if needed
            quantization_config = None
            if self.config.use_quantization:
                logger.info("🔧 Setting up 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.config.torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                logger.info("✅ Quantization config created")
            
            # Load tokenizer
            logger.info("🔤 Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    padding_side='left'
                )
                logger.info(f"✅ Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
            except Exception as e:
                error_msg = f"❌ Failed to load tokenizer: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("🔧 Set pad_token to eos_token")
            
            # Check GPU memory before loading model
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_free = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"🖥️ GPU Memory - Total: {gpu_memory:.1f}GB, Free: {gpu_free:.1f}GB")
            
            # Load base model
            logger.info("🤖 Loading base model...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    quantization_config=quantization_config,
                    torch_dtype=self.config.torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
                )
                logger.info("✅ Base model loaded successfully")
                
                # Log model info
                if hasattr(self.model, 'config'):
                    logger.info(f"📊 Model config: {self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else 'N/A'}")
                    if hasattr(self.model.config, 'num_parameters'):
                        logger.info(f"📊 Parameters: {self.model.config.num_parameters}")
                
            except Exception as e:
                error_msg = f"❌ Failed to load base model: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # Load adapter if specified
            if self.config.adapter_path:
                logger.info(f"🔧 Loading adapter from: {self.config.adapter_path}")
                try:
                    self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)
                    logger.info("✅ Adapter loaded successfully")
                except Exception as e:
                    error_msg = f"❌ Failed to load adapter: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            
            # Final GPU memory check
            if torch.cuda.is_available():
                gpu_used = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"🖥️ GPU Memory used after model load: {gpu_used:.1f}GB")
            
            logger.info(f"🎉 Model {self.config.name} loaded successfully!")
            
        except Exception as e:
            logger.error(f"💥 CRITICAL ERROR loading model {self.config.name}: {str(e)}")
            logger.error(f"💥 Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                logger.error(f"💥 Traceback: {traceback.format_exc()}")
            raise
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Model {self.config.name} unloaded")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        """Generate text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return generated_text

def load_data(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return []

def save_results(results: Dict, output_file: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def extract_answer(text: str, task_type: str) -> str:
    """Extract answer from generated text based on task type"""
    text = text.strip()
    
    if task_type == 'tc':
        # Extract topic classification answer
        categories = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
        for cat in categories:
            if cat in text:
                return cat
        return text.split('\n')[0].strip()
    
    elif task_type == 'sts':
        # Extract similarity score (0-5)
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            try:
                score = float(numbers[0])
                return str(min(5.0, max(0.0, score)))  # Clamp to 0-5 range
            except:
                return "2.5"  # Default middle score
        return "2.5"
    
    elif task_type == 'nli':
        # Extract NLI answer
        text_lower = text.lower()
        if 'entailment' in text_lower or '함의' in text_lower:
            return 'entailment'
        elif 'contradiction' in text_lower or '모순' in text_lower:
            return 'contradiction'
        elif 'neutral' in text_lower or '중립' in text_lower:
            return 'neutral'
        return text.split('\n')[0].strip()
    
    else:
        # For other tasks, return first line or full text
        lines = text.split('\n')
        return lines[0].strip() if lines else text

def compute_metrics(predictions: List[str], references: List[str], task_type: str) -> Dict[str, float]:
    """Compute evaluation metrics based on task type"""
    metrics = {}
    
    if task_type == 'tc' or task_type == 'nli':
        # Accuracy for classification tasks
        correct = sum(1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip())
        metrics['accuracy'] = correct / len(predictions) if predictions else 0.0
        
        # Also compute F1 if there are multiple classes
        try:
            if task_type == 'tc':
                # Map categories to indices for F1
                label_map = {'IT과학': 0, '경제': 1, '사회': 2, '생활문화': 3, '세계': 4, '스포츠': 5, '정치': 6}
                pred_indices = [label_map.get(p.strip(), 0) for p in predictions]
                ref_indices = [label_map.get(str(r).strip(), 0) for r in references]
                metrics['f1_macro'] = f1_score(ref_indices, pred_indices, average='macro')
            elif task_type == 'nli':
                label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
                pred_indices = [label_map.get(p.strip(), 2) for p in predictions]
                ref_indices = [label_map.get(str(r).strip(), 2) for r in references]
                metrics['f1_macro'] = f1_score(ref_indices, pred_indices, average='macro')
        except Exception as e:
            logger.warning(f"Could not compute F1 score: {e}")
    
    elif task_type == 'sts':
        # Pearson correlation for STS
        try:
            pred_scores = [float(p) for p in predictions]
            ref_scores = [float(r) for r in references]
            correlation, p_value = pearsonr(pred_scores, ref_scores)
            metrics['pearson'] = correlation if not np.isnan(correlation) else 0.0
            metrics['p_value'] = p_value if not np.isnan(p_value) else 1.0
        except Exception as e:
            logger.error(f"Error computing Pearson correlation: {e}")
            metrics['pearson'] = 0.0
            metrics['p_value'] = 1.0
    
    else:
        # For other tasks (NER, RE, DP, MRC, DST), compute basic accuracy
        correct = sum(1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip())
        metrics['accuracy'] = correct / len(predictions) if predictions else 0.0
        
        # TODO: Implement task-specific metrics (F1 for NER/RE, UAS/LAS for DP, etc.)
    
    return metrics

def evaluate_sample(model_loader: ModelLoader, sample: Dict, prompt_template: str, task_type: str) -> Tuple[str, str]:
    """Evaluate a single sample"""
    # Format prompt based on task type and sample data
    if task_type == 'tc':
        prompt = prompt_template.format(title=sample.get('title', ''))
        true_answer = str(sample.get('label', ''))
    
    elif task_type == 'sts':
        # Handle both formats in STS data
        if 'input' in sample and 'output' in sample:
            # Parse the input to extract sentences
            input_text = sample['input']
            sentences = re.findall(r'Sentence \d+: (.+?)(?=Sentence \d+:|$)', input_text)
            if len(sentences) >= 2:
                prompt = prompt_template.format(sentence1=sentences[0].strip(), sentence2=sentences[1].strip())
            else:
                prompt = input_text
            true_answer = str(sample['output'])
        else:
            prompt = prompt_template.format(
                sentence1=sample.get('sentence1', ''),
                sentence2=sample.get('sentence2', '')
            )
            true_answer = str(sample.get('label', ''))
    
    elif task_type == 'nli':
        prompt = prompt_template.format(
            premise=sample.get('premise', ''),
            hypothesis=sample.get('hypothesis', '')
        )
        true_answer = sample.get('label', '')
        if isinstance(true_answer, int):
            label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
            true_answer = label_map.get(true_answer, 'neutral')
    
    else:
        # For other tasks, use the sample as is or adapt as needed
        prompt = str(sample)
        true_answer = str(sample.get('label', ''))
    
    # Generate prediction
    try:
        prediction = model_loader.generate_text(prompt, max_new_tokens=128, temperature=0.1)
        prediction = extract_answer(prediction, task_type)
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        prediction = ""
    
    return prediction, true_answer

def run_evaluation(model_loader: ModelLoader, data: List[Dict], task_type: str, prompt_template: str, max_samples: int = None) -> Dict:
    """Run evaluation on a dataset"""
    if max_samples:
        data = data[:max_samples]
    
    logger.info(f"🧪 Evaluating {len(data)} samples for task: {task_type}")
    logger.info(f"📋 Prompt template preview: {prompt_template[:100]}...")
    
    if not data:
        error_msg = f"❌ No data provided for evaluation"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if model is loaded
    if model_loader.model is None or model_loader.tokenizer is None:
        error_msg = f"❌ Model or tokenizer is not loaded"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    predictions = []
    references = []
    failed_samples = 0
    
    logger.info(f"🔄 Starting sample processing...")
    for i, sample in enumerate(data):
        if i % 50 == 0:
            logger.info(f"📊 Processing sample {i+1}/{len(data)} ({(i+1)/len(data)*100:.1f}%)")
        
        try:
            pred, ref = evaluate_sample(model_loader, sample, prompt_template, task_type)
            predictions.append(pred)
            references.append(ref)
            
            # Log first few samples for debugging
            if i < 3:
                logger.info(f"🔍 Sample {i+1} - Predicted: '{pred}' | Reference: '{ref}'")
                
        except Exception as e:
            logger.error(f"❌ Failed to process sample {i+1}: {str(e)}")
            failed_samples += 1
            # Use default values to keep arrays aligned
            predictions.append("")
            references.append("")
    
    if failed_samples > 0:
        logger.warning(f"⚠️ Failed to process {failed_samples}/{len(data)} samples ({failed_samples/len(data)*100:.1f}%)")
    
    logger.info(f"✅ Sample processing completed. Success: {len(data) - failed_samples}/{len(data)}")
    
    # Compute metrics
    logger.info(f"📈 Computing metrics for task: {task_type}")
    try:
        metrics = compute_metrics(predictions, references, task_type)
        logger.info(f"✅ Metrics computed successfully: {list(metrics.keys())}")
        
        # Log metric values
        for metric_name, value in metrics.items():
            logger.info(f"   📊 {metric_name}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"❌ Failed to compute metrics: {str(e)}")
        metrics = {'error': str(e)}
    
    result = {
        'task_type': task_type,
        'num_samples': len(data),
        'failed_samples': failed_samples,
        'success_rate': (len(data) - failed_samples) / len(data) if data else 0.0,
        'metrics': metrics,
        'predictions': predictions,
        'references': references
    }
    
    logger.info(f"🎉 Evaluation completed for task: {task_type}")
    return result