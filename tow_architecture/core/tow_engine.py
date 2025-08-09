"""
ToW Engine - Main Orchestration System
=====================================

The ToWEngine coordinates the entire Thoughts of Words process, managing
the flow from input processing through English intermediary reasoning
to multilingual output generation.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .cognitive_bridge import CognitiveBridge
from .thought_processor import ThoughtTokenProcessor
from .multilingual_processor import MultilingualProcessor
from ..models.base_adapter import BaseModelAdapter
from ..utils.config import ToWConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingStage(Enum):
    """Processing stages in the ToW pipeline"""
    INPUT_ANALYSIS = "input_analysis"
    THOUGHT_GENERATION = "thought_generation" 
    COGNITIVE_BRIDGING = "cognitive_bridging"
    OUTPUT_GENERATION = "output_generation"
    VALIDATION = "validation"


@dataclass
class ToWRequest:
    """Request object for ToW processing"""
    text: str
    source_language: str = "auto"
    target_language: str = "ko"
    task_type: str = "generation"
    context: Optional[Dict[str, Any]] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass 
class ToWResponse:
    """Response object from ToW processing"""
    output_text: str
    thought_tokens: List[str]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class ToWEngine:
    """
    Main orchestration engine for Thoughts of Words processing.
    
    This class coordinates the entire ToW pipeline:
    1. Input analysis and language detection
    2. English thought token generation
    3. Cognitive bridging between languages
    4. Target language output generation
    5. Quality validation and scoring
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        config: Optional[ToWConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ToW Engine.
        
        Args:
            model_adapter: Model adapter instance (DeepSeek, Llama, Qwen)
            config: ToW configuration object
            device: PyTorch device for computation
        """
        self.model_adapter = model_adapter
        self.config = config or ToWConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize core components
        self._initialize_components()
        
        # Processing statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_processing_time": 0.0,
            "stage_times": {stage.value: [] for stage in ProcessingStage}
        }
        
        logger.info(f"ToWEngine initialized with {type(model_adapter).__name__}")
    
    def _initialize_components(self):
        """Initialize core ToW components"""
        # Cognitive bridge for cross-lingual reasoning
        self.cognitive_bridge = CognitiveBridge(
            model_adapter=self.model_adapter,
            config=self.config.cognitive_bridge,
            device=self.device
        )
        
        # Thought token processor for English reasoning
        self.thought_processor = ThoughtTokenProcessor(
            model_adapter=self.model_adapter,
            config=self.config.thought_processor,
            device=self.device
        )
        
        # Multilingual processor for target language output
        self.multilingual_processor = MultilingualProcessor(
            model_adapter=self.model_adapter,
            config=self.config.multilingual_processor,
            device=self.device
        )
        
        logger.info("Core ToW components initialized")
    
    def process(self, request: Union[str, ToWRequest]) -> ToWResponse:
        """
        Process a ToW request through the complete pipeline.
        
        Args:
            request: Text string or ToWRequest object
            
        Returns:
            ToWResponse with generated output and metadata
        """
        # Convert string to request object
        if isinstance(request, str):
            request = ToWRequest(text=request)
        
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Stage 1: Input Analysis
            stage_start = time.time()
            input_analysis = self._analyze_input(request)
            self._record_stage_time(ProcessingStage.INPUT_ANALYSIS, stage_start)
            
            # Stage 2: Thought Generation
            stage_start = time.time()
            thought_tokens = self._generate_thoughts(request, input_analysis)
            self._record_stage_time(ProcessingStage.THOUGHT_GENERATION, stage_start)
            
            # Stage 3: Cognitive Bridging
            stage_start = time.time()
            bridge_output = self._perform_cognitive_bridging(
                request, thought_tokens, input_analysis
            )
            self._record_stage_time(ProcessingStage.COGNITIVE_BRIDGING, stage_start)
            
            # Stage 4: Output Generation
            stage_start = time.time()
            output_text = self._generate_output(request, bridge_output)
            self._record_stage_time(ProcessingStage.OUTPUT_GENERATION, stage_start)
            
            # Stage 5: Validation
            stage_start = time.time()
            validation_result = self._validate_output(request, output_text, thought_tokens)
            self._record_stage_time(ProcessingStage.VALIDATION, stage_start)
            
            # Build response
            processing_time = time.time() - start_time
            response = ToWResponse(
                output_text=output_text,
                thought_tokens=thought_tokens,
                confidence_score=validation_result["confidence"],
                processing_time=processing_time,
                metadata={
                    "input_analysis": input_analysis,
                    "bridge_output": bridge_output,
                    "validation": validation_result,
                    "stage_times": {
                        stage.value: self.stats["stage_times"][stage.value][-1]
                        for stage in ProcessingStage
                    }
                }
            )
            
            self.stats["successful_requests"] += 1
            self._update_avg_processing_time(processing_time)
            
            logger.info(
                f"ToW processing completed in {processing_time:.2f}s "
                f"(confidence: {validation_result['confidence']:.3f})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"ToW processing failed: {str(e)}")
            # Return error response
            return ToWResponse(
                output_text="",
                thought_tokens=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _analyze_input(self, request: ToWRequest) -> Dict[str, Any]:
        """
        Analyze input text for language, complexity, and task requirements.
        
        Args:
            request: ToW request object
            
        Returns:
            Dictionary containing input analysis results
        """
        analysis = {
            "detected_language": self._detect_language(request.text),
            "text_length": len(request.text),
            "complexity_score": self._assess_complexity(request.text),
            "task_requirements": self._analyze_task_requirements(request),
            "preprocessing_needed": self._check_preprocessing_needs(request.text)
        }
        
        # Adjust processing strategy based on analysis
        if analysis["complexity_score"] > 0.8:
            logger.info("High complexity detected, enabling enhanced thought processing")
        
        return analysis
    
    def _generate_thoughts(
        self, 
        request: ToWRequest, 
        input_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate English thought tokens for the given input.
        
        Args:
            request: ToW request object
            input_analysis: Results from input analysis
            
        Returns:
            List of generated thought tokens
        """
        return self.thought_processor.generate_thoughts(
            text=request.text,
            context=request.context,
            analysis=input_analysis,
            max_thoughts=self.config.max_thought_tokens,
            temperature=request.temperature
        )
    
    def _perform_cognitive_bridging(
        self,
        request: ToWRequest,
        thought_tokens: List[str],
        input_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform cognitive bridging between English thoughts and target language.
        
        Args:
            request: ToW request object
            thought_tokens: Generated English thought tokens
            input_analysis: Results from input analysis
            
        Returns:
            Dictionary containing bridge processing results
        """
        return self.cognitive_bridge.bridge_languages(
            source_text=request.text,
            thought_tokens=thought_tokens,
            target_language=request.target_language,
            analysis=input_analysis,
            task_type=request.task_type
        )
    
    def _generate_output(
        self,
        request: ToWRequest,
        bridge_output: Dict[str, Any]
    ) -> str:
        """
        Generate final output in target language.
        
        Args:
            request: ToW request object
            bridge_output: Results from cognitive bridging
            
        Returns:
            Generated output text in target language
        """
        return self.multilingual_processor.generate_output(
            bridge_output=bridge_output,
            target_language=request.target_language,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
    
    def _validate_output(
        self,
        request: ToWRequest,
        output_text: str,
        thought_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Validate generated output quality and consistency.
        
        Args:
            request: ToW request object
            output_text: Generated output text
            thought_tokens: Generated thought tokens
            
        Returns:
            Dictionary containing validation results
        """
        validation = {
            "confidence": self._calculate_confidence(output_text, thought_tokens),
            "coherence_score": self._assess_coherence(request.text, output_text),
            "language_consistency": self._check_language_consistency(
                output_text, request.target_language
            ),
            "thought_alignment": self._assess_thought_alignment(
                output_text, thought_tokens
            )
        }
        
        # Overall quality score
        validation["quality_score"] = (
            validation["confidence"] * 0.3 +
            validation["coherence_score"] * 0.3 +
            validation["language_consistency"] * 0.2 +
            validation["thought_alignment"] * 0.2
        )
        
        return validation
    
    def _detect_language(self, text: str) -> str:
        """Detect the primary language of input text"""
        # Simplified language detection - in practice use langdetect or similar
        if any(ord(char) > 127 for char in text):
            if any(0x1100 <= ord(char) <= 0x11FF or 
                   0x3130 <= ord(char) <= 0x318F or
                   0xAC00 <= ord(char) <= 0xD7AF for char in text):
                return "ko"
            elif any(0x4E00 <= ord(char) <= 0x9FFF for char in text):
                return "zh"
            else:
                return "unknown"
        return "en"
    
    def _assess_complexity(self, text: str) -> float:
        """Assess text complexity on scale 0-1"""
        # Simple heuristic - could be enhanced with NLP analysis
        factors = [
            len(text.split()) / 100,  # Length factor
            len(set(text.split())) / len(text.split()) if text.split() else 0,  # Vocabulary diversity
            text.count(',') + text.count(';') + text.count(':'),  # Punctuation complexity
        ]
        return min(sum(factors) / 3, 1.0)
    
    def _analyze_task_requirements(self, request: ToWRequest) -> Dict[str, Any]:
        """Analyze specific task requirements"""
        return {
            "task_type": request.task_type,
            "requires_reasoning": request.task_type in ["reasoning", "analysis", "problem_solving"],
            "requires_creativity": request.task_type in ["generation", "writing", "creative"],
            "requires_accuracy": request.task_type in ["translation", "summarization", "qa"]
        }
    
    def _check_preprocessing_needs(self, text: str) -> List[str]:
        """Check if text needs preprocessing"""
        needs = []
        if len(text) > 2000:
            needs.append("chunking")
        if text.count('\n') > 10:
            needs.append("formatting")
        return needs
    
    def _calculate_confidence(self, output_text: str, thought_tokens: List[str]) -> float:
        """Calculate confidence score based on output and thought consistency"""
        # Simplified confidence calculation
        if not output_text or not thought_tokens:
            return 0.0
        
        # Factors affecting confidence
        output_length_score = min(len(output_text) / 100, 1.0)
        thought_coverage_score = min(len(thought_tokens) / 5, 1.0)
        
        return (output_length_score + thought_coverage_score) / 2
    
    def _assess_coherence(self, input_text: str, output_text: str) -> float:
        """Assess coherence between input and output"""
        # Simplified coherence assessment
        if not input_text or not output_text:
            return 0.0
        
        # Basic overlap scoring
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        
        if not input_words:
            return 0.0
        
        overlap = len(input_words.intersection(output_words))
        return min(overlap / len(input_words), 1.0)
    
    def _check_language_consistency(self, text: str, target_language: str) -> float:
        """Check if output matches target language"""
        detected = self._detect_language(text)
        return 1.0 if detected == target_language else 0.5
    
    def _assess_thought_alignment(self, output_text: str, thought_tokens: List[str]) -> float:
        """Assess how well output aligns with thought tokens"""
        if not thought_tokens:
            return 0.5  # Neutral score if no thoughts
        
        # Simple alignment based on keyword presence
        thought_keywords = set()
        for thought in thought_tokens:
            thought_keywords.update(thought.lower().split())
        
        output_words = set(output_text.lower().split())
        
        if not thought_keywords:
            return 0.5
        
        alignment = len(thought_keywords.intersection(output_words))
        return min(alignment / len(thought_keywords), 1.0)
    
    def _record_stage_time(self, stage: ProcessingStage, start_time: float):
        """Record processing time for a stage"""
        stage_time = time.time() - start_time
        self.stats["stage_times"][stage.value].append(stage_time)
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time"""
        total = self.stats["total_requests"]
        current_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        
        # Calculate average stage times
        for stage in ProcessingStage:
            stage_times = stats["stage_times"][stage.value]
            if stage_times:
                stats["stage_times"][stage.value] = {
                    "avg": sum(stage_times) / len(stage_times),
                    "min": min(stage_times),
                    "max": max(stage_times),
                    "count": len(stage_times)
                }
        
        # Calculate success rate
        stats["success_rate"] = (
            stats["successful_requests"] / stats["total_requests"]
            if stats["total_requests"] > 0 else 0.0
        )
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_processing_time": 0.0,
            "stage_times": {stage.value: [] for stage in ProcessingStage}
        }
        logger.info("Statistics reset")