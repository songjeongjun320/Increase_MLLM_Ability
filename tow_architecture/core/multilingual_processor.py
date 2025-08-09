"""
Multilingual Processor - Target Language Output Generation
=========================================================

The MultilingualProcessor generates final output in the target language
using the bridged English thoughts and cognitive mapping information
to ensure accurate and culturally appropriate multilingual generation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F

from ..models.base_adapter import BaseModelAdapter
from ..utils.config import MultilingualProcessorConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OutputMode(Enum):
    """Different output generation modes"""
    DIRECT_GENERATION = "direct_generation"
    THOUGHT_GUIDED = "thought_guided"
    BRIDGE_INTEGRATED = "bridge_integrated"
    CULTURALLY_ADAPTED = "culturally_adapted"


class LanguageStyle(Enum):
    """Language style preferences"""
    FORMAL = "formal"
    INFORMAL = "informal"
    NEUTRAL = "neutral"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"


@dataclass
class OutputRequest:
    """Request for multilingual output generation"""
    bridge_output: Dict[str, Any]
    target_language: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    style: LanguageStyle = LanguageStyle.NEUTRAL
    cultural_adaptation: bool = True


@dataclass
class OutputResult:
    """Result from multilingual output generation"""
    generated_text: str
    confidence_score: float
    language_quality: float
    cultural_appropriateness: float
    thought_alignment: float
    metadata: Dict[str, Any]


class MultilingualProcessor:
    """
    Multilingual Processor for target language output generation.
    
    This component takes the bridged English thoughts and generates
    final output in the target language, ensuring accuracy, cultural
    appropriateness, and alignment with the reasoning process.
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        config: Optional[MultilingualProcessorConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Multilingual Processor.
        
        Args:
            model_adapter: Model adapter for text generation
            config: Processor configuration
            device: PyTorch device for computation
        """
        self.model_adapter = model_adapter
        self.config = config or MultilingualProcessorConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Language-specific configurations
        self.language_configs = self._initialize_language_configs()
        
        # Output templates and patterns
        self.output_templates = self._load_output_templates()
        self.style_patterns = self._load_style_patterns()
        
        # Quality assessment components
        self.quality_assessor = self._initialize_quality_assessor()
        
        logger.info("MultilingualProcessor initialized")
    
    def generate_output(
        self,
        bridge_output: Dict[str, Any],
        target_language: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        style: Optional[LanguageStyle] = None
    ) -> str:
        """
        Generate output in target language using bridge information.
        
        Args:
            bridge_output: Results from cognitive bridging
            target_language: Target language code
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_p: Nucleus sampling parameter
            style: Language style preference
            
        Returns:
            Generated text in target language
        """
        request = OutputRequest(
            bridge_output=bridge_output,
            target_language=target_language,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            style=style or LanguageStyle.NEUTRAL
        )
        
        result = self.generate_detailed_output(request)
        return result.generated_text
    
    def generate_detailed_output(self, request: OutputRequest) -> OutputResult:
        """
        Generate detailed output with quality metrics.
        
        Args:
            request: Output generation request
            
        Returns:
            Detailed output result with quality metrics
        """
        logger.info(f"Generating output in {request.target_language}")
        
        # Extract bridge information
        bridge_info = self._extract_bridge_information(request.bridge_output)
        
        # Select output mode based on available information
        output_mode = self._select_output_mode(request, bridge_info)
        
        # Generate output using selected mode
        generated_text = self._generate_with_mode(request, bridge_info, output_mode)
        
        # Post-process output
        processed_text = self._post_process_output(
            generated_text, request.target_language, request.style
        )
        
        # Assess output quality
        quality_metrics = self._assess_output_quality(
            processed_text, request, bridge_info
        )
        
        return OutputResult(
            generated_text=processed_text,
            confidence_score=quality_metrics["confidence"],
            language_quality=quality_metrics["language_quality"],
            cultural_appropriateness=quality_metrics["cultural_appropriateness"],
            thought_alignment=quality_metrics["thought_alignment"],
            metadata={
                "output_mode": output_mode.value,
                "bridge_info": bridge_info,
                "quality_metrics": quality_metrics
            }
        )
    
    def _extract_bridge_information(self, bridge_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from bridge output"""
        return {
            "bridged_thoughts": bridge_output.get("bridged_thoughts", []),
            "semantic_mapping": bridge_output.get("semantic_mapping", {}),
            "cultural_adaptations": bridge_output.get("cultural_adaptations", []),
            "bridge_strategy": bridge_output.get("bridge_strategy", "contextual_bridging"),
            "quality_score": bridge_output.get("quality_score", 0.5),
            "context": bridge_output.get("context")
        }
    
    def _select_output_mode(
        self, 
        request: OutputRequest, 
        bridge_info: Dict[str, Any]
    ) -> OutputMode:
        """Select appropriate output generation mode"""
        
        # Consider bridge quality
        bridge_quality = bridge_info.get("quality_score", 0.5)
        
        # Consider cultural adaptations
        has_cultural_adaptations = bool(bridge_info.get("cultural_adaptations"))
        
        # Consider thought availability
        has_thoughts = bool(bridge_info.get("bridged_thoughts"))
        
        # Select mode based on available information and quality
        if bridge_quality > 0.8 and has_cultural_adaptations:
            return OutputMode.CULTURALLY_ADAPTED
        elif bridge_quality > 0.6 and has_thoughts:
            return OutputMode.BRIDGE_INTEGRATED
        elif has_thoughts:
            return OutputMode.THOUGHT_GUIDED
        else:
            return OutputMode.DIRECT_GENERATION
    
    def _generate_with_mode(
        self,
        request: OutputRequest,
        bridge_info: Dict[str, Any],
        output_mode: OutputMode
    ) -> str:
        """Generate output using the selected mode"""
        
        if output_mode == OutputMode.DIRECT_GENERATION:
            return self._direct_generation(request, bridge_info)
        elif output_mode == OutputMode.THOUGHT_GUIDED:
            return self._thought_guided_generation(request, bridge_info)
        elif output_mode == OutputMode.BRIDGE_INTEGRATED:
            return self._bridge_integrated_generation(request, bridge_info)
        elif output_mode == OutputMode.CULTURALLY_ADAPTED:
            return self._culturally_adapted_generation(request, bridge_info)
        else:
            logger.warning(f"Unknown output mode {output_mode}, falling back to direct")
            return self._direct_generation(request, bridge_info)
    
    def _direct_generation(
        self, 
        request: OutputRequest, 
        bridge_info: Dict[str, Any]
    ) -> str:
        """Direct generation without explicit thought integration"""
        
        # Get basic context
        context = bridge_info.get("context")
        source_text = context.source_text if context else ""
        
        # Build basic prompt
        prompt = self._build_direct_prompt(
            source_text, request.target_language, request.style
        )
        
        # Generate output
        return self.model_adapter.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
    
    def _thought_guided_generation(
        self, 
        request: OutputRequest, 
        bridge_info: Dict[str, Any]
    ) -> str:
        """Generation guided by thought tokens"""
        
        bridged_thoughts = bridge_info.get("bridged_thoughts", [])
        context = bridge_info.get("context")
        
        # Build thought-integrated prompt
        prompt = self._build_thought_guided_prompt(
            context, bridged_thoughts, request.target_language, request.style
        )
        
        # Generate output
        return self.model_adapter.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
    
    def _bridge_integrated_generation(
        self, 
        request: OutputRequest, 
        bridge_info: Dict[str, Any]
    ) -> str:
        """Generation fully integrated with bridge information"""
        
        bridged_thoughts = bridge_info.get("bridged_thoughts", [])
        semantic_mapping = bridge_info.get("semantic_mapping", {})
        context = bridge_info.get("context")
        
        # Build comprehensive prompt
        prompt = self._build_bridge_integrated_prompt(
            context, bridged_thoughts, semantic_mapping, 
            request.target_language, request.style
        )
        
        # Generate output
        return self.model_adapter.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
    
    def _culturally_adapted_generation(
        self, 
        request: OutputRequest, 
        bridge_info: Dict[str, Any]
    ) -> str:
        """Generation with full cultural adaptation"""
        
        bridged_thoughts = bridge_info.get("bridged_thoughts", [])
        cultural_adaptations = bridge_info.get("cultural_adaptations", [])
        context = bridge_info.get("context")
        
        # Build culturally adapted prompt
        prompt = self._build_culturally_adapted_prompt(
            context, bridged_thoughts, cultural_adaptations,
            request.target_language, request.style
        )
        
        # Generate output
        return self.model_adapter.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
    
    def _build_direct_prompt(
        self, 
        source_text: str, 
        target_language: str, 
        style: LanguageStyle
    ) -> str:
        """Build prompt for direct generation"""
        
        style_instruction = self._get_style_instruction(style, target_language)
        
        prompt = f"""Task: Generate a response in {target_language} for the following input.

Input: {source_text}

Style: {style_instruction}

Generate a natural, appropriate response in {target_language}:
Response: """
        
        return prompt
    
    def _build_thought_guided_prompt(
        self,
        context,
        bridged_thoughts: List[str],
        target_language: str,
        style: LanguageStyle
    ) -> str:
        """Build prompt guided by thought tokens"""
        
        source_text = context.source_text if context else ""
        style_instruction = self._get_style_instruction(style, target_language)
        
        thoughts_text = "\n".join([f"- {thought}" for thought in bridged_thoughts])
        
        prompt = f"""Task: Generate a response in {target_language} based on the following reasoning process.

Input: {source_text}

Reasoning thoughts:
{thoughts_text}

Style: {style_instruction}

Using the reasoning above, generate a response in {target_language}:
Response: """
        
        return prompt
    
    def _build_bridge_integrated_prompt(
        self,
        context,
        bridged_thoughts: List[str],
        semantic_mapping: Dict[str, str],
        target_language: str,
        style: LanguageStyle
    ) -> str:
        """Build prompt with full bridge integration"""
        
        source_text = context.source_text if context else ""
        style_instruction = self._get_style_instruction(style, target_language)
        
        thoughts_text = "\n".join([f"- {thought}" for thought in bridged_thoughts])
        
        # Include semantic mappings if available
        mappings_text = ""
        if semantic_mapping:
            mappings_text = "Semantic mappings:\n"
            for orig, mapped in list(semantic_mapping.items())[:3]:
                mappings_text += f"'{orig}' → '{mapped}'\n"
        
        prompt = f"""Task: Generate a response in {target_language} using the provided reasoning and semantic mappings.

Input: {source_text}

Reasoning thoughts:
{thoughts_text}

{mappings_text}

Style: {style_instruction}

Generate a culturally appropriate response in {target_language}:
Response: """
        
        return prompt
    
    def _build_culturally_adapted_prompt(
        self,
        context,
        bridged_thoughts: List[str],
        cultural_adaptations: List[str],
        target_language: str,
        style: LanguageStyle
    ) -> str:
        """Build prompt with cultural adaptations"""
        
        source_text = context.source_text if context else ""
        style_instruction = self._get_style_instruction(style, target_language)
        
        thoughts_text = "\n".join([f"- {thought}" for thought in bridged_thoughts])
        adaptations_text = "\n".join([f"- {adaptation}" for adaptation in cultural_adaptations])
        
        prompt = f"""Task: Generate a culturally appropriate response in {target_language}.

Input: {source_text}

Reasoning thoughts:
{thoughts_text}

Cultural adaptations to consider:
{adaptations_text}

Style: {style_instruction}

Generate a response that respects {target_language} cultural norms:
Response: """
        
        return prompt
    
    def _get_style_instruction(self, style: LanguageStyle, target_language: str) -> str:
        """Get style instruction for target language"""
        
        base_instructions = {
            LanguageStyle.FORMAL: "formal and respectful",
            LanguageStyle.INFORMAL: "casual and friendly",
            LanguageStyle.NEUTRAL: "balanced and natural",
            LanguageStyle.TECHNICAL: "precise and technical",
            LanguageStyle.CONVERSATIONAL: "conversational and engaging"
        }
        
        base = base_instructions.get(style, "natural")
        
        # Add language-specific style notes
        if target_language == "ko":
            if style == LanguageStyle.FORMAL:
                return f"{base}, using appropriate honorifics (존댓말)"
            elif style == LanguageStyle.INFORMAL:
                return f"{base}, using casual speech (반말) appropriately"
        elif target_language == "zh":
            if style == LanguageStyle.FORMAL:
                return f"{base}, using respectful language patterns"
        
        return base
    
    def _post_process_output(
        self, 
        text: str, 
        target_language: str, 
        style: LanguageStyle
    ) -> str:
        """Post-process generated output for quality and consistency"""
        
        if not text:
            return text
        
        # Clean up formatting
        processed = text.strip()
        
        # Remove common generation artifacts
        artifacts = ["Response:", "Output:", "Text:", target_language.upper() + ":"]
        for artifact in artifacts:
            if processed.startswith(artifact):
                processed = processed[len(artifact):].strip()
        
        # Apply language-specific post-processing
        processed = self._apply_language_specific_processing(
            processed, target_language, style
        )
        
        # Final cleanup
        processed = self._final_cleanup(processed)
        
        return processed
    
    def _apply_language_specific_processing(
        self, 
        text: str, 
        target_language: str, 
        style: LanguageStyle
    ) -> str:
        """Apply language-specific post-processing"""
        
        if target_language == "ko":
            return self._process_korean_text(text, style)
        elif target_language == "zh":
            return self._process_chinese_text(text, style)
        else:
            return text
    
    def _process_korean_text(self, text: str, style: LanguageStyle) -> str:
        """Apply Korean-specific processing"""
        # Basic Korean text processing
        # In practice, this could include:
        # - Honorific consistency checking
        # - Particle usage validation
        # - Sentence ending appropriateness
        
        return text
    
    def _process_chinese_text(self, text: str, style: LanguageStyle) -> str:
        """Apply Chinese-specific processing"""
        # Basic Chinese text processing
        # In practice, this could include:
        # - Character vs. word segmentation
        # - Formal language pattern checking
        # - Punctuation normalization
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup"""
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?', '。', '！', '？')):
            text += '.'
        
        return text
    
    def _assess_output_quality(
        self, 
        output_text: str, 
        request: OutputRequest, 
        bridge_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess quality of generated output"""
        
        metrics = {}
        
        # Overall confidence based on output length and completeness
        metrics["confidence"] = self._calculate_confidence(output_text, request)
        
        # Language quality assessment
        metrics["language_quality"] = self._assess_language_quality(
            output_text, request.target_language
        )
        
        # Cultural appropriateness
        metrics["cultural_appropriateness"] = self._assess_cultural_appropriateness(
            output_text, request.target_language, bridge_info
        )
        
        # Thought alignment
        metrics["thought_alignment"] = self._assess_thought_alignment(
            output_text, bridge_info.get("bridged_thoughts", [])
        )
        
        return metrics
    
    def _calculate_confidence(self, output_text: str, request: OutputRequest) -> float:
        """Calculate overall confidence in the output"""
        if not output_text:
            return 0.0
        
        # Length-based confidence
        length_score = min(len(output_text) / 100, 1.0)
        
        # Completeness check
        completeness_score = 1.0 if len(output_text) > 10 else 0.5
        
        # Request fulfillment
        fulfillment_score = 0.8  # Base score, could be enhanced
        
        return (length_score + completeness_score + fulfillment_score) / 3
    
    def _assess_language_quality(self, text: str, target_language: str) -> float:
        """Assess quality of language usage"""
        if not text:
            return 0.0
        
        # Basic quality indicators
        quality_score = 0.5  # Base score
        
        # Check for appropriate length
        if 10 <= len(text) <= 1000:
            quality_score += 0.2
        
        # Check for sentence structure (basic heuristic)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count > 0:
            quality_score += 0.2
        
        # Language-specific checks
        if target_language == "ko":
            # Check for Korean characters
            if any(0xAC00 <= ord(char) <= 0xD7AF for char in text):
                quality_score += 0.1
        elif target_language == "zh":
            # Check for Chinese characters
            if any(0x4E00 <= ord(char) <= 0x9FFF for char in text):
                quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _assess_cultural_appropriateness(
        self, 
        text: str, 
        target_language: str, 
        bridge_info: Dict[str, Any]
    ) -> float:
        """Assess cultural appropriateness of output"""
        if not text:
            return 0.0
        
        # Base appropriateness
        score = 0.7  # Default score
        
        # Check if cultural adaptations were applied
        cultural_adaptations = bridge_info.get("cultural_adaptations", [])
        if cultural_adaptations:
            score += 0.2
        
        # Language-specific cultural markers
        if target_language == "ko":
            # Check for appropriate formality markers (simplified)
            formal_markers = ["습니다", "입니다", "해요", "세요"]
            if any(marker in text for marker in formal_markers):
                score += 0.1
        
        return min(score, 1.0)
    
    def _assess_thought_alignment(
        self, 
        output_text: str, 
        bridged_thoughts: List[str]
    ) -> float:
        """Assess how well output aligns with thoughts"""
        if not bridged_thoughts:
            return 0.5  # Neutral if no thoughts
        
        if not output_text:
            return 0.0
        
        # Simple alignment based on concept overlap
        output_words = set(output_text.lower().split())
        thought_words = set()
        
        for thought in bridged_thoughts:
            thought_words.update(thought.lower().split())
        
        if not thought_words:
            return 0.5
        
        overlap = len(output_words.intersection(thought_words))
        alignment = min(overlap / len(thought_words), 1.0)
        
        return alignment
    
    def _initialize_language_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific configurations"""
        return {
            "ko": {
                "formality_levels": ["formal", "polite", "casual"],
                "special_features": ["honorifics", "particles", "sentence_endings"],
                "cultural_markers": ["hierarchy", "age_respect", "social_distance"]
            },
            "zh": {
                "formality_levels": ["formal", "standard", "informal"],
                "special_features": ["measure_words", "aspect_markers", "tone_consideration"],
                "cultural_markers": ["collectivism", "face_saving", "hierarchy"]
            },
            "en": {
                "formality_levels": ["formal", "neutral", "casual"],
                "special_features": ["contractions", "idioms", "phrasal_verbs"],
                "cultural_markers": ["directness", "individualism", "efficiency"]
            }
        }
    
    def _load_output_templates(self) -> Dict[str, List[str]]:
        """Load output generation templates"""
        return {
            "direct": [
                "Based on the input, {output}",
                "In response to your question, {output}",
                "The answer is {output}"
            ],
            "thought_guided": [
                "Considering the reasoning process, {output}",
                "Based on the analysis, {output}",
                "Following the thought process, {output}"
            ],
            "culturally_adapted": [
                "In the context of {culture}, {output}",
                "Respecting cultural norms, {output}",
                "Appropriately expressed, {output}"
            ]
        }
    
    def _load_style_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load style patterns for different languages"""
        return {
            "formal": {
                "ko": {"markers": ["습니다", "입니다"], "avoid": ["야", "거야"]},
                "zh": {"markers": ["请", "您"], "avoid": ["你好吗"]},
                "en": {"markers": ["please", "would"], "avoid": ["gonna", "wanna"]}
            },
            "informal": {
                "ko": {"markers": ["야", "어"], "avoid": ["습니다", "시겠습니다"]},
                "zh": {"markers": ["你", "吧"], "avoid": ["您", "敬请"]},
                "en": {"markers": ["hey", "gonna"], "avoid": ["therefore", "furthermore"]}
            }
        }
    
    def _initialize_quality_assessor(self) -> Dict[str, Any]:
        """Initialize quality assessment components"""
        return {
            "confidence_threshold": 0.7,
            "quality_threshold": 0.6,
            "alignment_threshold": 0.5
        }