"""
DeepSeek Model Adapter - Specialized Interface for DeepSeek Models
=================================================================

This adapter provides specialized support for DeepSeek models,
including proper prompt formatting, thought token processing,
and DeepSeek-specific optimizations.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base_adapter import BaseModelAdapter, ModelConfig, ModelInfo, ModelType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekAdapter(BaseModelAdapter):
    """
    Specialized adapter for DeepSeek models.
    
    Supports DeepSeek-specific features like:
    - Proper chat template formatting
    - Thought token processing (<|thinking|> tags)
    - DeepSeek-R1 reasoning capabilities
    - Optimized generation parameters
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize DeepSeek adapter.
        
        Args:
            config: Model configuration with DeepSeek-specific settings
        """
        super().__init__(config)
        self.model_family = self._detect_model_family()
        self.supports_thinking = self._supports_thinking_mode()
        
        # DeepSeek-specific tokens
        self.thinking_start_token = "<｜thinking｜>"
        self.thinking_end_token = "<｜/thinking｜>"
        self.special_tokens = [
            "<｜start▁header▁id｜>",
            "<｜end▁header▁id｜>",
            "<｜eot▁id｜>",
            self.thinking_start_token,
            self.thinking_end_token
        ]
        
        logger.info(f"DeepSeek adapter initialized for {self.model_family}")
    
    def load_model(self) -> bool:
        """Load DeepSeek model and tokenizer"""
        try:
            logger.info(f"Loading DeepSeek model from {self.config.model_path}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=True,
                padding_side="left"
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare model loading arguments
            model_kwargs = self._prepare_model_kwargs()
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Apply optimizations
            self._apply_optimizations()
            
            # Set device
            if hasattr(self.model, 'hf_device_map'):
                self.device = torch.device("cuda:0")
            else:
                self.device = next(self.model.parameters()).device
            
            # Create model info
            self._model_info = self._create_model_info()
            
            logger.info("DeepSeek model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DeepSeek model: {str(e)}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text using DeepSeek model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text with DeepSeek-specific post-processing
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Preprocess prompt with DeepSeek formatting
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.get_context_length() - max_tokens
        )
        
        # Move to device
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Generation configuration
        generation_kwargs = self._prepare_generation_kwargs(
            max_tokens, temperature, top_p, **kwargs
        )
        
        # Generate
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs
                )
        
        # Extract new tokens
        new_tokens = generated_ids[0][input_ids.shape[-1]:]
        
        # Decode output
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Post-process DeepSeek output
        return self._postprocess_deepseek_output(output)
    
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
            List of generated texts
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Format all prompts
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        
        # Tokenize batch
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.get_context_length() - max_tokens
        )
        
        # Move to device
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Generation configuration
        generation_kwargs = self._prepare_generation_kwargs(
            max_tokens, temperature, top_p, **kwargs
        )
        
        # Generate
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs
                )
        
        # Process outputs
        outputs = []
        for i, gen_ids in enumerate(generated_ids):
            # Extract new tokens
            new_tokens = gen_ids[input_ids[i].shape[-1]:]
            # Decode
            output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Post-process
            outputs.append(self._postprocess_deepseek_output(output))
        
        return outputs
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_model_info(self) -> ModelInfo:
        """Get DeepSeek model information"""
        if self._model_info:
            return self._model_info
        
        return ModelInfo(
            model_name="DeepSeek (Not Loaded)",
            model_type=ModelType.DEEPSEEK,
            parameter_count="Unknown",
            context_length=8192,
            supported_languages=["en", "ko", "zh", "ja"],
            capabilities=["generation", "reasoning", "thinking"],
            memory_usage={}
        )
    
    def _detect_model_family(self) -> str:
        """Detect DeepSeek model family from path"""
        path_lower = self.config.model_path.lower()
        
        if "deepseek-r1" in path_lower:
            return "deepseek-r1"
        elif "deepseek-v3" in path_lower:
            return "deepseek-v3"
        elif "deepseek-v2" in path_lower:
            return "deepseek-v2"
        elif "deepseek-coder" in path_lower:
            return "deepseek-coder"
        else:
            return "deepseek-base"
    
    def _supports_thinking_mode(self) -> bool:
        """Check if model supports thinking mode"""
        # DeepSeek-R1 models support thinking mode
        return "r1" in self.model_family.lower()
    
    def _prepare_model_kwargs(self) -> Dict[str, Any]:
        """Prepare DeepSeek-specific model loading arguments"""
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": self.config.torch_dtype or torch.bfloat16,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "device_map": self.config.device_map or "auto"
        }
        
        # Add memory constraints if specified
        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory
        
        # Add quantization if enabled
        if self.config.load_in_4bit or self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = bnb_config
        
        # FlashAttention support
        if self.config.use_flash_attention:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using FlashAttention for DeepSeek model")
            except ImportError:
                logger.warning("FlashAttention not available, using default attention")
        
        return model_kwargs
    
    def _apply_optimizations(self):
        """Apply DeepSeek-specific optimizations"""
        if hasattr(torch, 'compile'):
            try:
                # PyTorch 2.0+ compilation for faster inference
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",
                    fullgraph=False,
                    dynamic=True
                )
                logger.info("Applied PyTorch compilation optimization")
            except Exception as e:
                logger.warning(f"Failed to apply compilation optimization: {e}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using DeepSeek chat template"""
        # Check if tokenizer has chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
        
        # Fallback to manual DeepSeek format
        return f"""<｜start▁header▁id｜>user<｜end▁header▁id｜>

{prompt}<｜eot▁id｜><｜start▁header▁id｜>assistant<｜end▁header▁id｜>

"""
    
    def _prepare_generation_kwargs(
        self, 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare generation arguments for DeepSeek"""
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1,
            "early_stopping": True
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in generation_kwargs:
                generation_kwargs[key] = value
        
        return generation_kwargs
    
    def _postprocess_deepseek_output(self, output: str) -> str:
        """Post-process DeepSeek output"""
        if not output:
            return output
        
        # Remove thinking tags if present
        if self.supports_thinking:
            output = self._extract_final_answer(output)
        
        # Remove special tokens
        for token in self.special_tokens:
            output = output.replace(token, "")
        
        # Remove common artifacts
        artifacts = ["Response:", "Assistant:", "Output:", "Answer:"]
        for artifact in artifacts:
            if output.strip().startswith(artifact):
                output = output[len(artifact):].strip()
        
        return output.strip()
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from DeepSeek-R1 thinking output"""
        # If response contains thinking tags, extract the final answer
        if self.thinking_start_token in response and self.thinking_end_token in response:
            parts = response.split(self.thinking_end_token)
            if len(parts) > 1:
                # Return content after thinking tags
                final_answer = parts[1].strip()
                return final_answer if final_answer else response.strip()
        
        return response.strip()
    
    def _create_model_info(self) -> ModelInfo:
        """Create model information object"""
        # Estimate parameter count from model name/path
        param_count = "Unknown"
        if "70b" in self.config.model_path.lower():
            param_count = "70B"
        elif "7b" in self.config.model_path.lower():
            param_count = "7B"
        elif "8b" in self.config.model_path.lower():
            param_count = "8B"
        
        # Determine context length
        context_length = 8192  # Default for DeepSeek
        if hasattr(self.tokenizer, 'model_max_length'):
            context_length = min(self.tokenizer.model_max_length, 32768)
        
        # Model capabilities
        capabilities = ["generation", "multilingual"]
        if self.supports_thinking:
            capabilities.extend(["reasoning", "thinking", "cot"])
        
        if "coder" in self.model_family:
            capabilities.append("code_generation")
        
        return ModelInfo(
            model_name=f"DeepSeek-{self.model_family}",
            model_type=ModelType.DEEPSEEK,
            parameter_count=param_count,
            context_length=context_length,
            supported_languages=["en", "ko", "zh", "ja", "es", "fr", "de"],
            capabilities=capabilities,
            memory_usage=self.get_memory_usage()
        )
    
    def generate_with_thinking(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate with explicit thinking process (DeepSeek-R1 feature).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with 'thinking' and 'answer' keys
        """
        if not self.supports_thinking:
            # Fallback to regular generation
            answer = self.generate(prompt, max_tokens, **kwargs)
            return {"thinking": "", "answer": answer}
        
        # Generate full response with thinking
        full_response = self.generate(prompt, max_tokens, **kwargs)
        
        # Parse thinking and answer
        if self.thinking_start_token in full_response and self.thinking_end_token in full_response:
            thinking_match = re.search(
                rf"{re.escape(self.thinking_start_token)}(.*?){re.escape(self.thinking_end_token)}",
                full_response,
                re.DOTALL
            )
            
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                answer = full_response.split(self.thinking_end_token)[-1].strip()
                return {"thinking": thinking, "answer": answer}
        
        # No thinking found, return as answer
        return {"thinking": "", "answer": full_response}
    
    def supports_language(self, language: str) -> bool:
        """Check if DeepSeek model supports a specific language"""
        # DeepSeek models are primarily multilingual
        supported = ["en", "ko", "zh", "ja", "es", "fr", "de", "it", "pt", "ru"]
        return language.lower() in supported