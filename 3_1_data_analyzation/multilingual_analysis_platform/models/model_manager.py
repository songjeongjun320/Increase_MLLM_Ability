"""
Model Manager

Handles loading and managing various types of models for multilingual analysis:
- Sentence Transformers for embedding generation
- Base models and trained models for comparison
- Model caching and efficient loading
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import gc
import warnings

from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of various models for analysis."""

    def __init__(self):
        """Initialize the model manager."""
        self.config = get_config()
        self.device = self._get_device()
        self.models = {}
        self.tokenizers = {}
        self.sentence_transformer = None

        logger.info(f"ModelManager initialized with device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device."""
        device_config = self.config.get('performance.device', 'auto')

        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("Using CPU (CUDA not available)")
        else:
            device = device_config

        return device

    def load_sentence_transformer(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        Load sentence transformer model for embedding generation.

        Args:
            model_name: Name of the sentence transformer model

        Returns:
            Loaded sentence transformer model
        """
        if model_name is None:
            model_name = self.config.get(
                'models.sentence_transformer.default_model',
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )

        if self.sentence_transformer is None or hasattr(self.sentence_transformer, '_model_name') and self.sentence_transformer._model_name != model_name:
            try:
                logger.info(f"Loading sentence transformer: {model_name}")
                self.sentence_transformer = SentenceTransformer(model_name, device=self.device)
                self.sentence_transformer._model_name = model_name
                logger.info(f"Successfully loaded sentence transformer: {model_name}")

            except Exception as e:
                logger.error(f"Failed to load sentence transformer {model_name}: {e}")
                # Try backup models
                backup_models = self.config.get('models.sentence_transformer.backup_models', [])
                for backup_model in backup_models:
                    try:
                        logger.info(f"Trying backup model: {backup_model}")
                        self.sentence_transformer = SentenceTransformer(backup_model, device=self.device)
                        self.sentence_transformer._model_name = backup_model
                        logger.info(f"Successfully loaded backup model: {backup_model}")
                        break
                    except Exception as backup_e:
                        logger.error(f"Backup model {backup_model} also failed: {backup_e}")
                        continue
                else:
                    raise RuntimeError("Failed to load any sentence transformer model")

        return self.sentence_transformer

    def load_model(self, model_name_or_path: str, model_type: str = 'base') -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load a transformer model and tokenizer.

        Args:
            model_name_or_path: Model name from Hugging Face or local path
            model_type: Type of model ('base' or 'trained')

        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = f"{model_type}_{model_name_or_path}"

        if cache_key not in self.models:
            try:
                logger.info(f"Loading {model_type} model: {model_name_or_path}")

                # Check if it's a local path
                if Path(model_name_or_path).exists():
                    model_path = model_name_or_path
                else:
                    model_path = model_name_or_path

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                # Fix padding token issue
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    elif tokenizer.unk_token is not None:
                        tokenizer.pad_token = tokenizer.unk_token
                    else:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

                # Load model
                model = AutoModel.from_pretrained(
                    model_path,
                    output_attentions=True,
                    output_hidden_states=True
                )
                model.to(self.device)
                model.eval()

                self.models[cache_key] = model
                self.tokenizers[cache_key] = tokenizer

                logger.info(f"Successfully loaded {model_type} model: {model_name_or_path}")

            except Exception as e:
                logger.error(f"Failed to load {model_type} model {model_name_or_path}: {e}")
                raise

        return self.models[cache_key], self.tokenizers[cache_key]

    def load_base_model(self, model_name: Optional[str] = None) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load base model for comparison.

        Args:
            model_name: Name of the base model

        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name is None:
            model_name = self.config.get('models.base_models.default', 'bert-base-multilingual-cased')

        return self.load_model(model_name, 'base')

    def load_trained_model(self, model_path: str) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load trained/fine-tuned model for comparison.

        Args:
            model_path: Path to the trained model

        Returns:
            Tuple of (model, tokenizer)
        """
        return self.load_model(model_path, 'trained')

    def get_model_info(self, model_name_or_path: str, model_type: str = 'base') -> Dict[str, Any]:
        """
        Get information about a loaded model.

        Args:
            model_name_or_path: Model name or path
            model_type: Type of model

        Returns:
            Dictionary with model information
        """
        cache_key = f"{model_type}_{model_name_or_path}"

        if cache_key in self.models:
            model = self.models[cache_key]
            config = model.config

            info = {
                'model_name': model_name_or_path,
                'model_type': model_type,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'num_layers': config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'unknown',
                'hidden_size': config.hidden_size if hasattr(config, 'hidden_size') else 'unknown',
                'num_attention_heads': config.num_attention_heads if hasattr(config, 'num_attention_heads') else 'unknown',
                'vocab_size': config.vocab_size if hasattr(config, 'vocab_size') else 'unknown',
                'device': next(model.parameters()).device.type,
                'dtype': next(model.parameters()).dtype,
            }

            return info

        return {}

    def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None,
                          normalize: bool = True, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Generate sentence embeddings for given texts.

        Args:
            texts: List of texts to encode
            model_name: Sentence transformer model name
            normalize: Whether to normalize embeddings
            batch_size: Batch size for encoding

        Returns:
            Tensor of embeddings
        """
        sentence_transformer = self.load_sentence_transformer(model_name)

        if batch_size is None:
            batch_size = self.config.get('analysis.embedding.batch_size', 32)

        # Generate embeddings
        embeddings = sentence_transformer.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=True
        )

        return embeddings

    def get_attention_weights(self, model_name_or_path: str, text: str, model_type: str = 'base') -> Dict[str, Any]:
        """
        Extract attention weights from a model for given text.

        Args:
            model_name_or_path: Model name or path
            text: Input text
            model_type: Type of model

        Returns:
            Dictionary containing attention weights and tokens
        """
        model, tokenizer = self.load_model(model_name_or_path, model_type)

        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs with attention
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Extract attention weights
        attention_weights = outputs.attentions  # Tuple of tensors, one for each layer

        # Convert tokens back to strings
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return {
            'attention_weights': attention_weights,
            'tokens': tokens,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }

    def get_prediction_probabilities(self, model_name_or_path: str, text: str,
                                   model_type: str = 'base', next_token_only: bool = True) -> Dict[str, Any]:
        """
        Get prediction probabilities for next token(s).

        Args:
            model_name_or_path: Model name or path
            text: Input text
            model_type: Type of model
            next_token_only: Whether to return only next token probabilities

        Returns:
            Dictionary containing prediction probabilities and uncertainty measures
        """
        model, tokenizer = self.load_model(model_name_or_path, model_type)

        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits (note: this assumes the model has a language modeling head)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            # If no LM head, we can't get token probabilities
            logger.warning("Model doesn't have language modeling head, returning hidden states only")
            return {
                'hidden_states': outputs.last_hidden_state,
                'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            }
        else:
            raise ValueError("Model output format not supported for probability extraction")

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Calculate uncertainty measures
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)

        # Get top-k predictions for each position
        top_k = 10
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

        # Convert tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        result = {
            'logits': logits,
            'probabilities': probabilities,
            'entropy': entropy,
            'top_k_probs': top_probs,
            'top_k_indices': top_indices,
            'tokens': tokens,
            'input_ids': inputs['input_ids']
        }

        if next_token_only:
            # Focus on the last position (next token prediction)
            last_pos = inputs['attention_mask'].sum(-1) - 1  # Last non-padded position
            result['next_token'] = {
                'logits': logits[0, last_pos],
                'probabilities': probabilities[0, last_pos],
                'entropy': entropy[0, last_pos],
                'top_k_probs': top_probs[0, last_pos],
                'top_k_indices': top_indices[0, last_pos],
                'top_k_tokens': tokenizer.convert_ids_to_tokens(top_indices[0, last_pos])
            }

        return result

    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear model cache to free memory.

        Args:
            model_type: Specific model type to clear, or None for all
        """
        if model_type is None:
            # Clear all models
            self.models.clear()
            self.tokenizers.clear()
            self.sentence_transformer = None
            logger.info("Cleared all model caches")
        else:
            # Clear specific model type
            keys_to_remove = [k for k in self.models.keys() if k.startswith(f"{model_type}_")]
            for key in keys_to_remove:
                del self.models[key]
                if key in self.tokenizers:
                    del self.tokenizers[key]
            logger.info(f"Cleared {model_type} model caches")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {
            'loaded_models': len(self.models),
            'loaded_tokenizers': len(self.tokenizers),
            'sentence_transformer_loaded': self.sentence_transformer is not None
        }

        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
                'gpu_memory_free': (torch.cuda.get_device_properties(0).total_memory -
                                  torch.cuda.memory_reserved()) / 1024**3  # GB
            })

        return memory_info


# Global model manager instance
_global_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager