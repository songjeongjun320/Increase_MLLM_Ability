"""
Base Model Adapter - Abstract Interface for LLM Backends
========================================================

This module defines the abstract base class that all model adapters
must implement to ensure consistent behavior across different
model architectures (DeepSeek, Llama, Qwen, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import torch


class ModelType(Enum):
    """Supported model types"""
    DEEPSEEK = "deepseek"
    LLAMA = "llama" 
    QWEN = "qwen"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_path: str
    model_type: ModelType
    device_map: Optional[Dict[str, int]] = None
    torch_dtype: torch.dtype = torch.float16
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    max_memory: Optional[Dict[int, str]] = None
    low_cpu_mem_usage: bool = True
    use_flash_attention: bool = True


@dataclass 
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    model_name: str
    model_type: ModelType
    parameter_count: str
    context_length: int
    supported_languages: List[str]
    capabilities: List[str]
    memory_usage: Dict[str, float]


class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters.
    
    This class defines the interface that all model adapters must implement
    to ensure consistent behavior across different model architectures.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model adapter.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_config = None
        self._model_info = None
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            True if loading successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text strings
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the loaded model.
        
        Returns:
            ModelInfo object with model details
        """
        pass
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.tokenizer is not None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            memory_stats[f"gpu_{i}"] = {
                "allocated_gb": allocated,
                "cached_gb": cached,
                "total_gb": total,
                "usage_percent": (allocated / total) * 100
            }
        
        return memory_stats
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def set_generation_config(self, config: GenerationConfig):
        """
        Set generation configuration.
        
        Args:
            config: Generation configuration object
        """
        self.generation_config = config
    
    def validate_inputs(self, prompt: Union[str, List[str]]) -> bool:
        """
        Validate input prompt(s).
        
        Args:
            prompt: Single prompt or list of prompts
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if isinstance(prompt, str):
            return len(prompt.strip()) > 0
        elif isinstance(prompt, list):
            return all(isinstance(p, str) and len(p.strip()) > 0 for p in prompt)
        else:
            return False
    
    def preprocess_prompt(self, prompt: str) -> str:
        """
        Preprocess prompt before generation.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            Preprocessed prompt
        """
        # Basic preprocessing - can be overridden by subclasses
        return prompt.strip()
    
    def postprocess_output(self, output: str) -> str:
        """
        Postprocess generated output.
        
        Args:
            output: Raw generated output
            
        Returns:
            Postprocessed output
        """
        # Basic postprocessing - can be overridden by subclasses
        return output.strip()
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: ~4 chars per token
            return len(text) // 4
    
    def supports_language(self, language: str) -> bool:
        """
        Check if model supports a specific language.
        
        Args:
            language: Language code (e.g., 'en', 'ko', 'zh')
            
        Returns:
            True if language is supported, False otherwise
        """
        if self._model_info:
            return language in self._model_info.supported_languages
        
        # Default supported languages for multilingual models
        common_languages = ['en', 'ko', 'zh', 'ja', 'es', 'fr', 'de']
        return language in common_languages
    
    def get_context_length(self) -> int:
        """
        Get maximum context length for the model.
        
        Returns:
            Maximum context length in tokens
        """
        if self._model_info:
            return self._model_info.context_length
        
        # Default context length
        return 4096
    
    def __repr__(self) -> str:
        """String representation of the adapter"""
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}({self.config.model_type.value}, {status})"
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.clear_cache()