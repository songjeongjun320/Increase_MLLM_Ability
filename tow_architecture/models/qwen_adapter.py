"""
Qwen Model Adapter - Specialized Interface for Qwen Models
==========================================================

This adapter provides specialized support for Qwen models,
including proper prompt formatting, optimization settings,
and Qwen-specific features.
"""

import logging
from typing import Dict, List, Optional, Any, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base_adapter import BaseModelAdapter, ModelConfig, ModelInfo, ModelType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QwenAdapter(BaseModelAdapter):
    """
    Specialized adapter for Qwen models.
    
    Supports Qwen-specific features:
    - Proper chat template formatting
    - Qwen 2/2.5 optimizations
    - Multilingual capabilities
    - Chinese language specialization
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Qwen adapter.
        
        Args:
            config: Model configuration with Qwen-specific settings
        """
        super().__init__(config)
        self.model_family = self._detect_qwen_family()
        self.is_chat_model = self._is_chat_model()
        self.is_coder_model = self._is_coder_model()
        
        # Qwen-specific tokens
        self.system_start = "<|im_start|>"
        self.system_end = "<|im_end|>"
        self.user_role = "user"
        self.assistant_role = "assistant"
        self.system_role = "system"
        
        logger.info(f"Qwen adapter initialized for {self.model_family}")
    
    def load_model(self) -> bool:
        """Load Qwen model and tokenizer"""
        try:
            logger.info(f"Loading Qwen model from {self.config.model_path}")
            
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
            
            logger.info("Qwen model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {str(e)}")
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
        Generate text using Qwen model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text with Qwen-specific post-processing
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Format prompt for Qwen
        formatted_prompt = self._format_qwen_prompt(prompt)
        
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
        
        # Post-process Qwen output
        return self._postprocess_qwen_output(output)
    
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
        formatted_prompts = [self._format_qwen_prompt(p) for p in prompts]
        
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
            outputs.append(self._postprocess_qwen_output(output))
        
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
        """Get Qwen model information"""
        if self._model_info:
            return self._model_info
        
        return ModelInfo(
            model_name="Qwen (Not Loaded)",
            model_type=ModelType.QWEN,
            parameter_count="Unknown",
            context_length=8192,
            supported_languages=["zh", "en", "ja", "ko", "es", "fr", "de"],
            capabilities=["generation", "multilingual", "reasoning"],
            memory_usage={}
        )
    
    def _detect_qwen_family(self) -> str:
        """Detect specific Qwen model family"""
        path_lower = self.config.model_path.lower()
        
        if "qwen2.5" in path_lower:
            return "qwen-2.5"
        elif "qwen2" in path_lower:
            return "qwen-2"
        elif "qwen1.5" in path_lower:
            return "qwen-1.5"
        elif "qwen-coder" in path_lower:
            return "qwen-coder"
        else:
            return "qwen-base"
    
    def _is_chat_model(self) -> bool:
        """Check if this is a chat/instruct model"""
        path_lower = self.config.model_path.lower()
        chat_indicators = ["chat", "instruct", "it"]
        return any(indicator in path_lower for indicator in chat_indicators)
    
    def _is_coder_model(self) -> bool:
        """Check if this is a coder model"""
        path_lower = self.config.model_path.lower()
        return "coder" in path_lower
    
    def _prepare_model_kwargs(self) -> Dict[str, Any]:
        """Prepare Qwen-specific model loading arguments"""
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": self.config.torch_dtype or torch.bfloat16,
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
                logger.info("Using FlashAttention for Qwen model")
            except ImportError:
                logger.warning("FlashAttention not available, using default attention")
        
        return model_kwargs
    
    def _apply_optimizations(self):
        """Apply Qwen-specific optimizations"""
        if hasattr(torch, 'compile'):
            try:
                # PyTorch 2.0+ compilation
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True
                )
                logger.info("Applied PyTorch compilation optimization")
            except Exception as e:
                logger.warning(f"Failed to apply compilation optimization: {e}")
    
    def _format_qwen_prompt(self, prompt: str) -> str:
        """Format prompt using Qwen chat template"""
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
        
        # Fallback to manual Qwen format
        return f"{self.system_start}{self.user_role}\n{prompt}{self.system_end}\n{self.system_start}{self.assistant_role}\n"
    
    def _prepare_generation_kwargs(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare generation arguments for Qwen"""
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
    
    def _postprocess_qwen_output(self, output: str) -> str:
        """Post-process Qwen output"""
        if not output:
            return output
        
        # Remove Qwen-specific tokens
        tokens_to_remove = [
            self.system_start,
            self.system_end,
            f"{self.system_start}{self.user_role}",
            f"{self.system_start}{self.assistant_role}",
            f"{self.system_start}{self.system_role}"
        ]
        
        for token in tokens_to_remove:
            output = output.replace(token, "")
        
        # Remove common artifacts
        artifacts = ["Assistant:", "Response:", "Output:", "用户:", "助手:"]
        for artifact in artifacts:
            if output.strip().startswith(artifact):
                output = output[len(artifact):].strip()
        
        return output.strip()
    
    def _create_model_info(self) -> ModelInfo:
        """Create model information object"""
        # Estimate parameter count
        param_count = "Unknown"
        if "72b" in self.config.model_path.lower():
            param_count = "72B"
        elif "32b" in self.config.model_path.lower():
            param_count = "32B"
        elif "14b" in self.config.model_path.lower():
            param_count = "14B"
        elif "7b" in self.config.model_path.lower():
            param_count = "7B"
        elif "3b" in self.config.model_path.lower():
            param_count = "3B"
        elif "1.8b" in self.config.model_path.lower():
            param_count = "1.8B"
        elif "0.5b" in self.config.model_path.lower():
            param_count = "0.5B"
        
        # Determine context length
        context_length = 8192  # Default for Qwen 2
        if "qwen2.5" in self.model_family:
            context_length = 32768  # Qwen 2.5 has longer context
        
        # Model capabilities
        capabilities = ["generation", "multilingual"]
        if self.is_chat_model:
            capabilities.extend(["instruction_following", "conversation"])
        if self.is_coder_model:
            capabilities.append("code_generation")
        
        # Add reasoning capability
        capabilities.append("reasoning")
        
        # Comprehensive language support (Qwen is strongly multilingual)
        supported_langs = [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "ar", "hi"
        ]
        
        return ModelInfo(
            model_name=f"{self.model_family.title()}",
            model_type=ModelType.QWEN,
            parameter_count=param_count,
            context_length=context_length,
            supported_languages=supported_langs,
            capabilities=capabilities,
            memory_usage=self.get_memory_usage()
        )
    
    def supports_language(self, language: str) -> bool:
        """Check if Qwen model supports a specific language"""
        # Qwen models have excellent multilingual support
        supported = [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", 
            "ar", "hi", "th", "vi", "ms", "id", "tr", "pl", "nl", "sv"
        ]
        
        return language.lower() in supported
    
    def generate_multilingual(
        self,
        prompt: str,
        source_lang: str = "auto",
        target_lang: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate multilingual response (optimized for Qwen).
        
        Args:
            prompt: Input prompt
            source_lang: Source language code
            target_lang: Target language code (if translation needed)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated multilingual response
        """
        if target_lang and target_lang != source_lang:
            # Add translation instruction
            enhanced_prompt = f"Please translate the following from {source_lang} to {target_lang}:\n\n{prompt}"
        else:
            enhanced_prompt = prompt
        
        return self.generate(enhanced_prompt, max_tokens, **kwargs)
    
    def generate_chinese_specialized(
        self,
        prompt: str,
        style: str = "standard",  # standard, formal, colloquial
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate Chinese text with style specification (Qwen specialization).
        
        Args:
            prompt: Input prompt
            style: Chinese writing style
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated Chinese text in specified style
        """
        style_instructions = {
            "standard": "",
            "formal": "请用正式的书面语回答：",
            "colloquial": "请用日常口语的方式回答：",
            "classical": "请用文言文风格回答：",
            "technical": "请用专业技术术语回答："
        }
        
        instruction = style_instructions.get(style, "")
        enhanced_prompt = f"{instruction}\n{prompt}" if instruction else prompt
        
        return self.generate(enhanced_prompt, max_tokens, **kwargs)