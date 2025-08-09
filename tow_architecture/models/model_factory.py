"""
Model Factory - Unified Model Adapter Creation
=============================================

The ModelAdapterFactory provides a unified interface for creating
appropriate model adapters based on model type detection and
configuration requirements.
"""

import logging
import os
from typing import Dict, Optional, Type, Union
from pathlib import Path

from .base_adapter import BaseModelAdapter, ModelConfig, ModelType
from .deepseek_adapter import DeepSeekAdapter
from .llama_adapter import LlamaAdapter
from .qwen_adapter import QwenAdapter
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelAdapterFactory:
    """
    Factory class for creating model adapters.
    
    Automatically detects model type and creates the appropriate adapter
    with optimized configuration for each model architecture.
    """
    
    # Registry of model adapters
    ADAPTER_REGISTRY: Dict[ModelType, Type[BaseModelAdapter]] = {
        ModelType.DEEPSEEK: DeepSeekAdapter,
        ModelType.LLAMA: LlamaAdapter,
        ModelType.QWEN: QwenAdapter
    }
    
    @classmethod
    def create_adapter(
        self,
        model_path: str,
        model_type: Optional[Union[str, ModelType]] = None,
        **config_kwargs
    ) -> BaseModelAdapter:
        """
        Create a model adapter for the specified model.
        
        Args:
            model_path: Path to the model directory or HuggingFace model ID
            model_type: Explicitly specify model type (optional)
            **config_kwargs: Additional configuration parameters
            
        Returns:
            Appropriate model adapter instance
            
        Raises:
            ValueError: If model type cannot be determined or is unsupported
            FileNotFoundError: If model path doesn't exist
        """
        logger.info(f"Creating adapter for model: {model_path}")
        
        # Validate model path
        if not self._validate_model_path(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Determine model type
        if model_type is None:
            detected_type = self._detect_model_type(model_path)
        else:
            detected_type = self._normalize_model_type(model_type)
        
        logger.info(f"Detected model type: {detected_type.value}")
        
        # Create configuration
        config = self._create_model_config(model_path, detected_type, **config_kwargs)
        
        # Get adapter class
        adapter_class = self.ADAPTER_REGISTRY.get(detected_type)
        if adapter_class is None:
            raise ValueError(f"Unsupported model type: {detected_type}")
        
        # Create and return adapter
        adapter = adapter_class(config)
        logger.info(f"Created {adapter_class.__name__}")
        
        return adapter
    
    @classmethod
    def create_and_load(
        self,
        model_path: str,
        model_type: Optional[Union[str, ModelType]] = None,
        **config_kwargs
    ) -> BaseModelAdapter:
        """
        Create and load a model adapter in one step.
        
        Args:
            model_path: Path to the model directory or HuggingFace model ID
            model_type: Explicitly specify model type (optional)
            **config_kwargs: Additional configuration parameters
            
        Returns:
            Loaded model adapter instance
            
        Raises:
            RuntimeError: If model loading fails
        """
        adapter = self.create_adapter(model_path, model_type, **config_kwargs)
        
        if not adapter.load_model():
            raise RuntimeError(f"Failed to load model from {model_path}")
        
        logger.info("Model adapter created and loaded successfully")
        return adapter
    
    @classmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported model types"""
        return [model_type.value for model_type in self.ADAPTER_REGISTRY.keys()]
    
    @classmethod
    def register_adapter(
        self, 
        model_type: ModelType, 
        adapter_class: Type[BaseModelAdapter]
    ):
        """
        Register a new model adapter.
        
        Args:
            model_type: Model type to register
            adapter_class: Adapter class to register
        """
        self.ADAPTER_REGISTRY[model_type] = adapter_class
        logger.info(f"Registered adapter {adapter_class.__name__} for {model_type.value}")
    
    @classmethod
    def _validate_model_path(self, model_path: str) -> bool:
        """Validate that model path exists or is a valid HuggingFace ID"""
        # Check if it's a local path
        if os.path.exists(model_path):
            return True
        
        # Check if it looks like a HuggingFace model ID
        if "/" in model_path and not os.path.isabs(model_path):
            # Assume it's a HuggingFace model ID - validation will happen during loading
            return True
        
        return False
    
    @classmethod
    def _detect_model_type(self, model_path: str) -> ModelType:
        """
        Detect model type from model path and configuration.
        
        Args:
            model_path: Path to model
            
        Returns:
            Detected ModelType
        """
        path_lower = model_path.lower()
        
        # Check for DeepSeek models
        deepseek_indicators = [
            "deepseek", "deep-seek", "deep_seek"
        ]
        if any(indicator in path_lower for indicator in deepseek_indicators):
            return ModelType.DEEPSEEK
        
        # Check for Llama models
        llama_indicators = [
            "llama", "llama2", "llama3", "llama-2", "llama-3",
            "meta-llama", "code-llama", "codellama"
        ]
        if any(indicator in path_lower for indicator in llama_indicators):
            return ModelType.LLAMA
        
        # Check for Qwen models
        qwen_indicators = [
            "qwen", "qwen2", "qwen2.5", "qwen-", "alibaba"
        ]
        if any(indicator in path_lower for indicator in qwen_indicators):
            return ModelType.QWEN
        
        # Try to detect from config.json if available
        config_type = self._detect_from_config(model_path)
        if config_type != ModelType.UNKNOWN:
            return config_type
        
        logger.warning(f"Could not detect model type from path: {model_path}")
        logger.info("Defaulting to DeepSeek adapter")
        return ModelType.DEEPSEEK  # Default fallback
    
    @classmethod
    def _detect_from_config(self, model_path: str) -> ModelType:
        """Detect model type from config.json file"""
        try:
            import json
            
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                return ModelType.UNKNOWN
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Check architecture type
            architectures = config.get("architectures", [])
            if not architectures:
                return ModelType.UNKNOWN
            
            arch = architectures[0].lower()
            
            # Map architectures to model types
            if "deepseek" in arch:
                return ModelType.DEEPSEEK
            elif "llama" in arch:
                return ModelType.LLAMA
            elif "qwen" in arch:
                return ModelType.QWEN
            
            # Check model name/ID
            model_name = config.get("name_or_path", "").lower()
            if "deepseek" in model_name:
                return ModelType.DEEPSEEK
            elif "llama" in model_name:
                return ModelType.LLAMA
            elif "qwen" in model_name:
                return ModelType.QWEN
            
        except Exception as e:
            logger.warning(f"Failed to read config.json: {e}")
        
        return ModelType.UNKNOWN
    
    @classmethod
    def _normalize_model_type(self, model_type: Union[str, ModelType]) -> ModelType:
        """Normalize model type to ModelType enum"""
        if isinstance(model_type, ModelType):
            return model_type
        
        # Convert string to ModelType
        type_str = model_type.lower().strip()
        
        type_mapping = {
            "deepseek": ModelType.DEEPSEEK,
            "deep-seek": ModelType.DEEPSEEK,
            "deep_seek": ModelType.DEEPSEEK,
            "llama": ModelType.LLAMA,
            "llama2": ModelType.LLAMA,
            "llama3": ModelType.LLAMA,
            "qwen": ModelType.QWEN,
            "qwen2": ModelType.QWEN,
            "qwen2.5": ModelType.QWEN
        }
        
        return type_mapping.get(type_str, ModelType.UNKNOWN)
    
    @classmethod
    def _create_model_config(
        self,
        model_path: str,
        model_type: ModelType,
        **kwargs
    ) -> ModelConfig:
        """Create model configuration with type-specific optimizations"""
        
        # Base configuration
        config = ModelConfig(
            model_path=model_path,
            model_type=model_type,
            **kwargs
        )
        
        # Apply type-specific optimizations
        if model_type == ModelType.DEEPSEEK:
            config = self._optimize_deepseek_config(config)
        elif model_type == ModelType.LLAMA:
            config = self._optimize_llama_config(config)
        elif model_type == ModelType.QWEN:
            config = self._optimize_qwen_config(config)
        
        return config
    
    @classmethod
    def _optimize_deepseek_config(self, config: ModelConfig) -> ModelConfig:
        """Apply DeepSeek-specific optimizations"""
        # DeepSeek models work well with bfloat16
        if config.torch_dtype is None:
            config.torch_dtype = torch.bfloat16
        
        # Enable flash attention for better performance
        if config.use_flash_attention is None:
            config.use_flash_attention = True
        
        return config
    
    @classmethod
    def _optimize_llama_config(self, config: ModelConfig) -> ModelConfig:
        """Apply Llama-specific optimizations"""
        # Llama models typically use float16
        if config.torch_dtype is None:
            config.torch_dtype = torch.float16
        
        return config
    
    @classmethod
    def _optimize_qwen_config(self, config: ModelConfig) -> ModelConfig:
        """Apply Qwen-specific optimizations"""
        # Qwen models work well with bfloat16
        if config.torch_dtype is None:
            config.torch_dtype = torch.bfloat16
        
        return config
    
    @classmethod
    def create_optimized_config(
        self,
        model_path: str,
        device_map: Optional[Dict[str, int]] = None,
        max_memory: Optional[Dict[int, str]] = None,
        quantization: Optional[str] = None,
        **kwargs
    ) -> ModelConfig:
        """
        Create optimized configuration for high-performance deployment.
        
        Args:
            model_path: Path to model
            device_map: GPU device mapping
            max_memory: Memory limits per device
            quantization: Quantization type ('4bit', '8bit', or None)
            **kwargs: Additional configuration parameters
            
        Returns:
            Optimized ModelConfig
        """
        # Detect model type for optimization
        model_type = self._detect_model_type(model_path)
        
        # Base optimized settings
        config_args = {
            "device_map": device_map or "auto",
            "max_memory": max_memory,
            "low_cpu_mem_usage": True,
            "use_flash_attention": True,
            "trust_remote_code": True
        }
        
        # Apply quantization settings
        if quantization == "4bit":
            config_args.update({
                "load_in_4bit": True,
                "load_in_8bit": False
            })
        elif quantization == "8bit":
            config_args.update({
                "load_in_4bit": False,
                "load_in_8bit": True
            })
        
        # Add user overrides
        config_args.update(kwargs)
        
        return self._create_model_config(model_path, model_type, **config_args)


# Import torch here to avoid circular imports
import torch