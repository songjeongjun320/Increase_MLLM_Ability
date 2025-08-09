"""
Llama Model Adapter - Specialized Interface for Llama Models
===========================================================

This adapter provides specialized support for Llama models,
including proper prompt formatting, optimization settings,
and Llama-specific features.
"""

import logging
from typing import Dict, List, Optional, Any, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base_adapter import BaseModelAdapter, ModelConfig, ModelInfo, ModelType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LlamaAdapter(BaseModelAdapter):
    """
    Specialized adapter for Llama models.
    
    Supports Llama-specific features:
    - Proper chat template formatting
    - Llama 2/3 optimizations
    - Code Llama support
    - Instruction following
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Llama adapter.
        
        Args:
            config: Model configuration with Llama-specific settings
        """
        super().__init__(config)
        self.model_family = self._detect_llama_family()
        self.is_chat_model = self._is_chat_model()
        self.is_code_model = self._is_code_model()
        
        # Llama-specific tokens
        self.system_token = "<s>"
        self.end_token = "</s>"
        self.user_token = "[INST]"
        self.assistant_token = "[/INST]"
        
        logger.info(f"Llama adapter initialized for {self.model_family}")
    
    def load_model(self) -> bool:
        """Load Llama model and tokenizer"""
        try:
            logger.info(f"Loading Llama model from {self.config.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=True,
                padding_side="left"
            )
            
            # Set pad token
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
            
            logger.info("Llama model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {str(e)}")
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
        Generate text using Llama model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text with Llama-specific post-processing
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Format prompt for Llama
        formatted_prompt = self._format_llama_prompt(prompt)
        
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
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs
                )
        
        # Extract new tokens
        new_tokens = generated_ids[0][input_ids.shape[-1]:]
        
        # Decode output
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Post-process Llama output
        return self._postprocess_llama_output(output)
    
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
        formatted_prompts = [self._format_llama_prompt(p) for p in prompts]
        
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
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
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
            outputs.append(self._postprocess_llama_output(output))
        
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
        """Get Llama model information"""
        if self._model_info:
            return self._model_info
        
        return ModelInfo(
            model_name="Llama (Not Loaded)",
            model_type=ModelType.LLAMA,
            parameter_count="Unknown",
            context_length=4096,
            supported_languages=["en", "es", "fr", "de", "it", "pt"],
            capabilities=["generation", "instruction_following"],
            memory_usage={}
        )
    
    def _detect_llama_family(self) -> str:
        """Detect specific Llama model family"""
        path_lower = self.config.model_path.lower()
        
        if "llama-3" in path_lower or "llama3" in path_lower:
            return "llama-3"
        elif "llama-2" in path_lower or "llama2" in path_lower:
            return "llama-2"
        elif "code-llama" in path_lower or "codellama" in path_lower:
            return "code-llama"
        else:
            return "llama-base"
    
    def _is_chat_model(self) -> bool:
        """Check if this is a chat/instruct model"""
        path_lower = self.config.model_path.lower()
        chat_indicators = ["chat", "instruct", "it", "inst"]
        return any(indicator in path_lower for indicator in chat_indicators)
    
    def _is_code_model(self) -> bool:
        """Check if this is a code model"""
        path_lower = self.config.model_path.lower()
        return "code" in path_lower
    
    def _prepare_model_kwargs(self) -> Dict[str, Any]:
        """Prepare Llama-specific model loading arguments"""
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": self.config.torch_dtype or torch.float16,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "device_map": self.config.device_map or "auto"
        }
        
        # Add memory constraints
        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory
        
        # Add quantization if enabled
        if self.config.load_in_4bit or self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = bnb_config
        
        # FlashAttention support
        if self.config.use_flash_attention:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using FlashAttention for Llama model")
            except ImportError:
                logger.warning("FlashAttention not available, using default attention")
        
        return model_kwargs
    
    def _apply_optimizations(self):
        """Apply Llama-specific optimizations"""
        if hasattr(torch, 'compile'):
            try:
                # PyTorch 2.0+ compilation
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",  # Good balance for Llama
                    fullgraph=False,
                    dynamic=True
                )
                logger.info("Applied PyTorch compilation optimization")
            except Exception as e:
                logger.warning(f"Failed to apply compilation optimization: {e}")
    
    def _format_llama_prompt(self, prompt: str) -> str:
        """Format prompt using Llama chat template"""
        if not self.is_chat_model:
            # For base models, use simple formatting
            return prompt
        
        # Check for chat template
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
        
        # Fallback to manual Llama format
        if "llama-2" in self.model_family:
            # Llama 2 format
            return f"<s>[INST] {prompt} [/INST]"
        elif "llama-3" in self.model_family:
            # Llama 3 format
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Generic format
            return f"[INST] {prompt} [/INST]"
    
    def _prepare_generation_kwargs(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare generation arguments for Llama"""
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
    
    def _postprocess_llama_output(self, output: str) -> str:
        """Post-process Llama output"""
        if not output:
            return output
        
        # Remove Llama-specific tokens
        tokens_to_remove = [
            self.system_token,
            self.end_token,
            self.user_token,
            self.assistant_token,
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>", 
            "<|eot_id|>"
        ]
        
        for token in tokens_to_remove:
            output = output.replace(token, "")
        
        # Remove common artifacts
        artifacts = ["Assistant:", "Response:", "Output:", "[/INST]"]
        for artifact in artifacts:
            if output.strip().startswith(artifact):
                output = output[len(artifact):].strip()
        
        return output.strip()
    
    def _create_model_info(self) -> ModelInfo:
        """Create model information object"""
        # Estimate parameter count
        param_count = "Unknown"
        if "70b" in self.config.model_path.lower():
            param_count = "70B"
        elif "13b" in self.config.model_path.lower():
            param_count = "13B"
        elif "7b" in self.config.model_path.lower():
            param_count = "7B"
        
        # Determine context length
        context_length = 4096  # Default for Llama 2
        if "llama-3" in self.model_family:
            context_length = 8192  # Llama 3 has longer context
        
        # Model capabilities
        capabilities = ["generation"]
        if self.is_chat_model:
            capabilities.extend(["instruction_following", "conversation"])
        if self.is_code_model:
            capabilities.append("code_generation")
        
        # Supported languages (Llama is primarily English but has some multilingual capability)
        supported_langs = ["en"]
        if not self.is_code_model:  # Code models are primarily English
            supported_langs.extend(["es", "fr", "de", "it", "pt", "nl"])
        
        return ModelInfo(
            model_name=f"{self.model_family.title()}",
            model_type=ModelType.LLAMA,
            parameter_count=param_count,
            context_length=context_length,
            supported_languages=supported_langs,
            capabilities=capabilities,
            memory_usage=self.get_memory_usage()
        )
    
    def supports_language(self, language: str) -> bool:
        """Check if Llama model supports a specific language"""
        # Llama models have good support for major European languages
        supported = ["en", "es", "fr", "de", "it", "pt", "nl"]
        
        # Code models are primarily English
        if self.is_code_model:
            supported = ["en"]
        
        return language.lower() in supported
    
    def generate_instruction_following(
        self,
        instruction: str,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate response following specific instructions (optimized for Llama).
        
        Args:
            instruction: The instruction to follow
            context: Optional context information
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response following the instruction
        """
        if context:
            prompt = f"Context: {context}\n\nInstruction: {instruction}"
        else:
            prompt = f"Instruction: {instruction}"
        
        return self.generate(prompt, max_tokens, **kwargs)