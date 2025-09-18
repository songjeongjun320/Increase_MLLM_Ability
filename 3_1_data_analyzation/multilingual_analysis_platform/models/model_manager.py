"""
Model Manager

Handles loading and managing various types of models for multilingual analysis:
- Sentence Transformers for embedding generation
- Base models and trained models for comparison
- Model caching and efficient loading
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
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

        # Try to load as causal LM model for probability extraction
        try:
            if not hasattr(model, 'lm_head') and not hasattr(outputs, 'logits'):
                # Try to load the same model as CausalLM
                lm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
                lm_model.to(self.device)
                lm_model.eval()

                with torch.no_grad():
                    outputs = lm_model(**inputs)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    raise ValueError("Unable to extract logits from causal LM model")
            else:
                # Get model outputs
                with torch.no_grad():
                    outputs = model(**inputs)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    raise ValueError("Model output format not supported for probability extraction")

        except Exception as e:
            logger.warning(f"Failed to extract probabilities: {e}")
            logger.info("Attempting alternative approach using embedding similarity...")

            # Alternative: Use embedding similarity for uncertainty estimation
            return self._estimate_uncertainty_from_embeddings(model, tokenizer, inputs, text)

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

    def _estimate_uncertainty_from_embeddings(self, model, tokenizer, inputs, text):
        """
        Estimate uncertainty using embedding-based approaches when logits are unavailable.

        This method provides an alternative uncertainty estimation by analyzing:
        1. Token-level embedding variations
        2. Attention pattern entropy
        3. Hidden state magnitudes
        """
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        # Get hidden states and attention
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        attentions = outputs.attentions if hasattr(outputs, 'attentions') else None

        # Token information
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        seq_len = hidden_states.shape[1]

        # Calculate embedding-based uncertainty measures
        uncertainties = []
        attention_entropies = []

        for pos in range(seq_len):
            # 1. Hidden state magnitude (higher magnitude = more confident)
            hidden_vec = hidden_states[0, pos]  # [hidden_size]
            magnitude = torch.norm(hidden_vec).item()

            # 2. Attention entropy (higher entropy = more uncertain)
            if attentions is not None:
                # Average attention across all heads and layers
                avg_attention = torch.stack([att[0, :, pos, :].mean(0) for att in attentions]).mean(0)
                # Calculate entropy
                att_entropy = -(avg_attention * torch.log(avg_attention + 1e-8)).sum().item()
                attention_entropies.append(att_entropy)
            else:
                attention_entropies.append(0.0)

            # 3. Normalize magnitude to uncertainty (inverse relationship)
            # Higher magnitude = lower uncertainty
            uncertainty = 1.0 / (1.0 + magnitude / 100.0)  # Normalize to [0, 1]
            uncertainties.append(uncertainty)

        # Calculate statistics
        mean_uncertainty = sum(uncertainties) / len(uncertainties)
        mean_attention_entropy = sum(attention_entropies) / len(attention_entropies) if attention_entropies else 0.0

        # Classify uncertainty level
        if mean_uncertainty < 0.3:
            uncertainty_level = "low"
        elif mean_uncertainty < 0.6:
            uncertainty_level = "medium"
        else:
            uncertainty_level = "high"

        return {
            'logits': None,  # Not available
            'tokens': tokens,
            'uncertainty_estimates': {
                'token_uncertainties': uncertainties,
                'attention_entropies': attention_entropies,
                'mean_uncertainty': mean_uncertainty,
                'mean_attention_entropy': mean_attention_entropy,
                'uncertainty_classification': uncertainty_level
            },
            'method': 'embedding_based',
            'note': 'Probabilities unavailable - using embedding-based uncertainty estimation'
        }


# Global model manager instance
_global_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager