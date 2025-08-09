"""
Thought Token Processor - English Intermediary Reasoning (Option 2 Implementation)
=================================================================================

The ThoughtTokenProcessor generates and manages English thought tokens
that serve as intermediary reasoning steps to improve multilingual
LLM accuracy through cognitive bridging.

Updated for Option 2: Pure Original TOW Implementation with:
- Token classification (trivial/exact/soft/unpredictable)
- Cross-lingual support (English thoughts for any language)
- Data augmentation integration
- Enhanced training dataset generation
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F

from ..models.base_adapter import BaseModelAdapter
from ..utils.config import ThoughtProcessorConfig
from ..utils.logger import get_logger
from ..utils.text_utils import enforce_english_text, sanitize_tow_token
from .token_classifier import TokenClassifier, TOWCategory, ClassificationContext
from .cross_lingual_tow import CrossLingualTOWSystem, CrossLingualContext

logger = get_logger(__name__)


class ThoughtType(Enum):
    """Types of thought tokens"""
    ANALYTICAL = "analytical"        # Breaking down concepts
    CONTEXTUAL = "contextual"       # Understanding context
    INFERENTIAL = "inferential"     # Making inferences
    COMPARATIVE = "comparative"     # Comparing options
    PREDICTIVE = "predictive"       # Predicting outcomes
    EXPLANATORY = "explanatory"     # Explaining reasoning


@dataclass
class ThoughtToken:
    """Individual thought token with metadata"""
    content: str
    thought_type: ThoughtType
    confidence: float
    reasoning_chain: List[str]
    source_span: Tuple[int, int] = None
    dependencies: List[str] = None


@dataclass
class ThoughtContext:
    """Context for thought generation"""
    text: str
    language: str
    task_type: str
    complexity_level: float
    domain: str = "general"
    previous_thoughts: List[ThoughtToken] = None


class ThoughtTokenProcessor:
    """
    Thought Token Processor for English intermediary reasoning (Option 2).
    
    This component generates English thought tokens that serve as
    cognitive bridges, providing explicit reasoning steps that can
    be used to improve accuracy in multilingual generation tasks.
    
    New Features for Option 2:
    - Integrated token classification system
    - Cross-lingual TOW generation (English thoughts for any language)
    - Enhanced data augmentation support
    - Original TOW format compliance
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        config: Optional[ThoughtProcessorConfig] = None,
        device: Optional[torch.device] = None,
        language: str = "ko"
    ):
        """
        Initialize Thought Token Processor.
        
        Args:
            model_adapter: Model adapter for thought generation
            config: Processor configuration
            device: PyTorch device for computation
            language: Source language for processing
        """
        self.model_adapter = model_adapter
        self.config = config or ThoughtProcessorConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        
        # Initialize new Option 2 components
        self.token_classifier = TokenClassifier(language=language)
        self.cross_lingual_system = CrossLingualTOWSystem(model_adapter, language)
        
        # Thought generation templates and patterns
        self.thought_templates = self._load_thought_templates()
        self.reasoning_patterns = self._load_reasoning_patterns()
        
        # Thought quality metrics
        self.quality_thresholds = self._initialize_quality_thresholds()
        
        logger.info(f"ThoughtTokenProcessor initialized for Option 2 (language: {language})")
    
    def generate_thoughts(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        max_thoughts: int = 5,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate English thought tokens for the given text (Option 2).
        
        Args:
            text: Input text to generate thoughts for
            context: Additional context information
            analysis: Input analysis results
            max_thoughts: Maximum number of thoughts to generate
            temperature: Temperature for generation
            
        Returns:
            List of generated thought token strings in <ToW> format
        """
        logger.info(f"Generating Option 2 thoughts for text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Create thought context
        thought_context = self._create_thought_context(text, context, analysis)
        
        # Generate initial thought candidates
        thought_candidates = self._generate_thought_candidates(
            thought_context, max_thoughts * 2, temperature
        )
        
        # Rank and select best thoughts
        selected_thoughts = self._select_best_thoughts(
            thought_candidates, max_thoughts
        )
        
        # Post-process and validate thoughts
        final_thoughts = self._post_process_thoughts(selected_thoughts, thought_context)
        
        # Extract thought strings in proper <ToW> format
        thought_strings = [self._format_as_tow_token(thought.content) for thought in final_thoughts]
        
        logger.info(f"Generated {len(thought_strings)} Option 2 TOW tokens")
        return thought_strings
    
    def generate_classified_thoughts(
        self,
        text: str,
        predicted_words: List[str],
        actual_words: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate classified thoughts with TOW categories (Option 2).
        
        Args:
            text: Input text context
            predicted_words: Model predictions for each position
            actual_words: Actual next words
            context: Additional context information
            
        Returns:
            List of dictionaries with thought, category, and metadata
        """
        results = []
        
        for predicted, actual in zip(predicted_words, actual_words):
            # Classify the prediction
            classification_context = ClassificationContext(
                preceding_text=text,
                predicted_token=predicted,
                actual_token=actual,
                domain=self._detect_domain(text),
                language=self.language
            )
            
            classification_result = self.token_classifier.classify_token(classification_context)
            
            # Generate cross-lingual thought
            cross_lingual_context = CrossLingualContext(
                source_text=text,
                source_language=self.language,
                target_word=actual,
                predicted_word=predicted,
                reasoning_type=classification_result.category.value
            )
            
            tow_token = self.cross_lingual_system.generate_english_tow(cross_lingual_context)
            
            result = {
                "text": text,
                "predicted_word": predicted,
                "actual_word": actual,
                "tow_token": tow_token,
                "category": classification_result.category.value,
                "confidence": classification_result.confidence,
                "reasoning": classification_result.reasoning,
                "language": self.language,
                "metadata": {
                    "similarity_score": classification_result.similarity_score,
                    "frequency_score": classification_result.frequency_score,
                    "context_words": classification_result.context_words
                }
            }
            
            results.append(result)
        
        return results
    
    def generate_structured_thoughts(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        max_thoughts: int = 5,
        temperature: float = 0.7
    ) -> List[ThoughtToken]:
        """
        Generate structured thought tokens with full metadata.
        
        Args:
            text: Input text to generate thoughts for
            context: Additional context information
            analysis: Input analysis results
            max_thoughts: Maximum number of thoughts to generate
            temperature: Temperature for generation
            
        Returns:
            List of ThoughtToken objects with metadata
        """
        thought_context = self._create_thought_context(text, context, analysis)
        thought_candidates = self._generate_thought_candidates(
            thought_context, max_thoughts * 2, temperature
        )
        selected_thoughts = self._select_best_thoughts(thought_candidates, max_thoughts)
        return self._post_process_thoughts(selected_thoughts, thought_context)
    
    def _create_thought_context(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]]
    ) -> ThoughtContext:
        """Create context object for thought generation"""
        # Extract information from analysis if available
        language = "auto"
        task_type = "generation"
        complexity_level = 0.5
        
        if analysis:
            language = analysis.get("detected_language", "auto")
            complexity_level = analysis.get("complexity_score", 0.5)
            task_requirements = analysis.get("task_requirements", {})
            if "task_type" in task_requirements:
                task_type = task_requirements["task_type"]
        
        if context:
            task_type = context.get("task_type", task_type)
        
        return ThoughtContext(
            text=text,
            language=language,
            task_type=task_type,
            complexity_level=complexity_level,
            domain=self._detect_domain(text),
            previous_thoughts=[]
        )
    
    def _detect_domain(self, text: str) -> str:
        """Detect the domain/topic of the input text"""
        # Simplified domain detection - could be enhanced with classification
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["code", "program", "function", "algorithm"]):
            return "programming"
        elif any(word in text_lower for word in ["equation", "formula", "calculate", "solve"]):
            return "mathematics"
        elif any(word in text_lower for word in ["patient", "medical", "treatment", "diagnosis"]):
            return "medical"
        elif any(word in text_lower for word in ["law", "legal", "court", "regulation"]):
            return "legal"
        else:
            return "general"
    
    def _generate_thought_candidates(
        self,
        context: ThoughtContext,
        num_candidates: int,
        temperature: float
    ) -> List[ThoughtToken]:
        """Generate candidate thought tokens"""
        candidates = []
        
        # Generate thoughts using different strategies
        for i in range(num_candidates):
            thought_type = self._select_thought_type(context, i)
            thought_token = self._generate_single_thought(
                context, thought_type, temperature
            )
            if thought_token:
                candidates.append(thought_token)
        
        return candidates
    
    def _select_thought_type(self, context: ThoughtContext, iteration: int) -> ThoughtType:
        """Select appropriate thought type based on context and iteration"""
        # Strategy: vary thought types to get diverse reasoning
        types = list(ThoughtType)
        
        # Adjust based on task type
        if context.task_type == "analysis":
            preferred_types = [ThoughtType.ANALYTICAL, ThoughtType.EXPLANATORY]
        elif context.task_type == "reasoning":
            preferred_types = [ThoughtType.INFERENTIAL, ThoughtType.COMPARATIVE]
        elif context.task_type == "prediction":
            preferred_types = [ThoughtType.PREDICTIVE, ThoughtType.CONTEXTUAL]
        else:
            preferred_types = types
        
        # Select type based on iteration to ensure diversity
        return preferred_types[iteration % len(preferred_types)]
    
    def _generate_single_thought(
        self,
        context: ThoughtContext,
        thought_type: ThoughtType,
        temperature: float
    ) -> Optional[ThoughtToken]:
        """Generate a single thought token"""
        try:
            # Select appropriate template for thought type
            template = self._get_thought_template(thought_type, context)
            
            # Build prompt for thought generation
            prompt = self._build_thought_prompt(template, context, thought_type)
            
            # Generate thought using model adapter
            generated_text = self.model_adapter.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=temperature,
                top_p=0.9
            )
            
            # Extract and clean thought content
            thought_content = self._extract_thought_content(generated_text)
            
            if not thought_content:
                return None
            
            # Calculate confidence score
            confidence = self._calculate_thought_confidence(
                thought_content, context, thought_type
            )
            
            # Build reasoning chain
            reasoning_chain = self._build_reasoning_chain(thought_content, context)
            
            return ThoughtToken(
                content=thought_content,
                thought_type=thought_type,
                confidence=confidence,
                reasoning_chain=reasoning_chain,
                dependencies=[]
            )
        
        except Exception as e:
            logger.warning(f"Failed to generate thought: {str(e)}")
            return None
    
    def _get_thought_template(self, thought_type: ThoughtType, context: ThoughtContext) -> str:
        """Get appropriate template for thought type"""
        templates = self.thought_templates.get(thought_type.value, [])
        
        if not templates:
            return self.thought_templates["default"][0]
        
        # Select template based on context
        if context.complexity_level > 0.7:
            return templates[0] if templates else self.thought_templates["default"][0]
        else:
            return templates[-1] if templates else self.thought_templates["default"][0]
    
    def _build_thought_prompt(
        self,
        template: str,
        context: ThoughtContext,
        thought_type: ThoughtType
    ) -> str:
        """Build prompt for thought generation"""
        # Base prompt structure
        base_prompt = f"""Task: Generate an English thought token to help understand and process the following text.

Input text: {context.text}

Thought type: {thought_type.value}
Context: {context.task_type} task in {context.domain} domain
Complexity: {context.complexity_level:.2f}

Template: {template}

Generate a clear, concise thought in English that explains your reasoning process:
Thought: """
        
        return base_prompt
    
    def _extract_thought_content(self, generated_text: str) -> str:
        """Extract clean thought content from generated text"""
        if not generated_text:
            return ""
        
        # Look for thought markers
        patterns = [
            r"<hCoT>\s*(.*?)\s*</hCoT>",
            r"Thought:\s*(.*?)(?:\n|$)",
            r"Reasoning:\s*(.*?)(?:\n|$)",
            r"Analysis:\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content:
                    return content
        
        # Fallback: use first meaningful sentence
        sentences = generated_text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.startswith(('Task:', 'Input:', 'Template:')):
                return sentence
        
        return generated_text.strip()
    
    def _calculate_thought_confidence(
        self,
        thought_content: str,
        context: ThoughtContext,
        thought_type: ThoughtType
    ) -> float:
        """Calculate confidence score for a thought"""
        confidence_factors = []
        
        # Length factor (not too short, not too long)
        length_score = min(len(thought_content) / 100, 1.0)
        if length_score < 0.1:  # Too short
            length_score = 0.1
        confidence_factors.append(length_score)
        
        # Coherence factor (simple heuristic)
        coherence_score = self._assess_thought_coherence(thought_content)
        confidence_factors.append(coherence_score)
        
        # Relevance factor
        relevance_score = self._assess_thought_relevance(thought_content, context.text)
        confidence_factors.append(relevance_score)
        
        # Type appropriateness
        type_score = self._assess_type_appropriateness(thought_content, thought_type)
        confidence_factors.append(type_score)
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.3, 0.2]  # Length, coherence, relevance, type
        return sum(f * w for f, w in zip(confidence_factors, weights))
    
    def _assess_thought_coherence(self, thought_content: str) -> float:
        """Assess coherence of thought content"""
        # Simple heuristics for coherence
        score = 0.5  # Base score
        
        # Check for complete sentences
        if thought_content.endswith('.') or thought_content.endswith('!') or thought_content.endswith('?'):
            score += 0.2
        
        # Check for reasonable word count
        word_count = len(thought_content.split())
        if 5 <= word_count <= 30:
            score += 0.2
        
        # Check for reasoning indicators
        reasoning_words = ['because', 'since', 'therefore', 'thus', 'so', 'hence', 'given']
        if any(word in thought_content.lower() for word in reasoning_words):
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_thought_relevance(self, thought_content: str, original_text: str) -> float:
        """Assess relevance of thought to original text"""
        # Simple overlap scoring
        thought_words = set(thought_content.lower().split())
        original_words = set(original_text.lower().split())
        
        if not original_words:
            return 0.5
        
        overlap = len(thought_words.intersection(original_words))
        return min(overlap / len(original_words), 1.0)
    
    def _assess_type_appropriateness(self, thought_content: str, thought_type: ThoughtType) -> float:
        """Assess how well thought matches its type"""
        content_lower = thought_content.lower()
        
        type_keywords = {
            ThoughtType.ANALYTICAL: ['analyze', 'break', 'component', 'element', 'part'],
            ThoughtType.CONTEXTUAL: ['context', 'situation', 'background', 'setting'],
            ThoughtType.INFERENTIAL: ['infer', 'conclude', 'deduce', 'imply', 'suggest'],
            ThoughtType.COMPARATIVE: ['compare', 'versus', 'than', 'similar', 'different'],
            ThoughtType.PREDICTIVE: ['predict', 'expect', 'likely', 'probably', 'future'],
            ThoughtType.EXPLANATORY: ['explain', 'because', 'reason', 'cause', 'why']
        }
        
        keywords = type_keywords.get(thought_type, [])
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        
        return min(matches / max(len(keywords), 1), 1.0)
    
    def _build_reasoning_chain(self, thought_content: str, context: ThoughtContext) -> List[str]:
        """Build reasoning chain for the thought"""
        chain = []
        
        # Add context understanding
        chain.append(f"Context: {context.task_type} task")
        
        # Add domain consideration
        if context.domain != "general":
            chain.append(f"Domain: {context.domain}")
        
        # Add complexity consideration
        if context.complexity_level > 0.7:
            chain.append("High complexity detected")
        
        # Add the thought itself
        chain.append(f"Reasoning: {thought_content}")
        
        return chain
    
    def _select_best_thoughts(
        self,
        candidates: List[ThoughtToken],
        max_thoughts: int
    ) -> List[ThoughtToken]:
        """Select best thoughts from candidates"""
        if not candidates:
            return []
        
        # Sort by confidence score
        candidates.sort(key=lambda t: t.confidence, reverse=True)
        
        # Select top candidates while ensuring diversity
        selected = []
        used_types = set()
        
        for candidate in candidates:
            if len(selected) >= max_thoughts:
                break
            
            # Prefer diversity of thought types
            if candidate.thought_type not in used_types or len(selected) < max_thoughts // 2:
                selected.append(candidate)
                used_types.add(candidate.thought_type)
        
        # Fill remaining slots with highest confidence
        while len(selected) < max_thoughts and len(selected) < len(candidates):
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    break
        
        return selected[:max_thoughts]
    
    def _post_process_thoughts(
        self,
        thoughts: List[ThoughtToken],
        context: ThoughtContext
    ) -> List[ThoughtToken]:
        """Post-process and validate selected thoughts"""
        processed = []
        
        for thought in thoughts:
            # Quality filtering
            if thought.confidence < self.quality_thresholds["min_confidence"]:
                logger.debug(f"Filtering low confidence thought: {thought.confidence}")
                continue
            
            # Content validation
            if not self._validate_thought_content(thought.content):
                logger.debug(f"Filtering invalid thought content: {thought.content[:50]}")
                continue
            
            # Add dependencies between thoughts
            thought.dependencies = self._identify_thought_dependencies(thought, processed)
            
            processed.append(thought)
        
        return processed
    
    def _validate_thought_content(self, content: str) -> bool:
        """Validate thought content quality"""
        if not content or len(content.strip()) < 5:
            return False
        
        # Check for reasonable length
        if len(content) > 500:
            return False
        
        # Check for inappropriate content (basic filtering)
        inappropriate_markers = ['error', 'failed', 'cannot', 'unable']
        if any(marker in content.lower() for marker in inappropriate_markers):
            return False
        
        return True
    
    def _identify_thought_dependencies(
        self,
        current_thought: ThoughtToken,
        previous_thoughts: List[ThoughtToken]
    ) -> List[str]:
        """Identify dependencies between thoughts"""
        dependencies = []
        
        # Simple dependency detection based on content overlap
        current_words = set(current_thought.content.lower().split())
        
        for prev_thought in previous_thoughts:
            prev_words = set(prev_thought.content.lower().split())
            overlap = len(current_words.intersection(prev_words))
            
            # If significant overlap, consider it a dependency
            if overlap > 2:
                dependencies.append(prev_thought.content)
        
        return dependencies
    
    def _load_thought_templates(self) -> Dict[str, List[str]]:
        """Load templates for different thought types"""
        return {
            "analytical": [
                "Breaking down the {concept}, we can identify {components}",
                "Analyzing the structure shows {analysis}",
                "The key elements here are {elements}"
            ],
            "contextual": [
                "In the context of {situation}, this means {meaning}",
                "Given the background of {context}, we understand {understanding}",
                "The situational factors suggest {suggestion}"
            ],
            "inferential": [
                "From this information, we can infer {inference}",
                "The evidence suggests {conclusion}",
                "Based on the patterns, it's likely {likelihood}"
            ],
            "comparative": [
                "Comparing {option1} with {option2}, we see {comparison}",
                "The differences show {differences}",
                "In contrast to {baseline}, this approach {contrast}"
            ],
            "predictive": [
                "This pattern suggests {prediction}",
                "Based on the trend, we expect {expectation}",
                "The likely outcome is {outcome}"
            ],
            "explanatory": [
                "This occurs because {reason}",
                "The explanation is {explanation}",
                "The underlying cause is {cause}"
            ],
            "default": [
                "Considering the information, {consideration}",
                "The key insight is {insight}",
                "This indicates {indication}"
            ]
        }
    
    def _load_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Load reasoning patterns for different domains"""
        return {
            "programming": [
                "algorithm analysis", "data structure evaluation", "complexity assessment"
            ],
            "mathematics": [
                "equation solving", "proof construction", "pattern recognition"
            ],
            "general": [
                "logical reasoning", "cause-effect analysis", "comparative evaluation"
            ]
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds for thought validation"""
        return {
            "min_confidence": 0.3,
            "min_relevance": 0.2,
            "min_coherence": 0.4
        }
    
    # New Option 2 methods
    
    def _format_as_tow_token(self, content: str) -> str:
        """Format content as proper <ToW> token (always English)"""
        # Ensure content is in English and properly formatted
        if not content.strip():
            return "<ToW>Contextual reasoning applied.</ToW>"
        
        # Remove any existing ToW tags
        clean_content = re.sub(r'</?ToW>', '', content).strip()

        # Enforce English-only inner content and proper tags
        clean_content = enforce_english_text(clean_content)
        return sanitize_tow_token(f"<ToW>{clean_content}</ToW>")
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain of text for classification"""
        text_lower = text.lower()
        
        # Programming keywords
        if any(word in text_lower for word in ["코드", "프로그램", "함수", "변수", "code", "function"]):
            return "programming"
        
        # Mathematics keywords
        if any(word in text_lower for word in ["수학", "방정식", "계산", "공식", "math", "equation"]):
            return "mathematics"
        
        # Science keywords
        if any(word in text_lower for word in ["실험", "연구", "이론", "experiment", "research"]):
            return "science"
        
        return "general"
    
    def generate_training_data_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Generate training data batch for Option 2 TOW system.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of training examples with TOW tokens
        """
        training_examples = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                try:
                    # Generate thoughts for this text
                    thoughts = self.generate_thoughts(text, max_thoughts=3)
                    
                    # Create training example
                    example = {
                        "input_text": text,
                        "tow_tokens": thoughts,
                        "language": self.language,
                        "domain": self._detect_domain(text),
                        "metadata": {
                            "num_thoughts": len(thoughts),
                            "text_length": len(text),
                            "word_count": len(text.split())
                        }
                    }
                    
                    training_examples.append(example)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate training example for text: {e}")
                    continue
        
        logger.info(f"Generated {len(training_examples)} training examples")
        return training_examples
    
    def export_tow_dataset(
        self, 
        texts: List[str], 
        output_path: str,
        include_classifications: bool = True
    ):
        """
        Export TOW dataset in standard format.
        
        Args:
            texts: Input texts to process
            output_path: Output file path
            include_classifications: Whether to include token classifications
        """
        import json
        
        dataset_entries = []
        
        for i, text in enumerate(texts):
            try:
                if include_classifications:
                    # Generate with classification
                    words = text.split()
                    if len(words) > 2:
                        # Simple prediction simulation
                        predicted_words = words[1:]  # Shift for prediction
                        actual_words = words[1:]
                        
                        classified_thoughts = self.generate_classified_thoughts(
                            text=" ".join(words[:-1]),
                            predicted_words=predicted_words,
                            actual_words=actual_words
                        )
                        
                        for thought_data in classified_thoughts:
                            dataset_entries.append({
                                "id": f"{i}_{len(dataset_entries)}",
                                "source_text": thought_data["text"],
                                "predicted_word": thought_data["predicted_word"],
                                "actual_word": thought_data["actual_word"],
                                "tow_token": thought_data["tow_token"],
                                "category": thought_data["category"],
                                "confidence": thought_data["confidence"],
                                "language": thought_data["language"],
                                "reasoning": thought_data["reasoning"],
                                "metadata": thought_data["metadata"]
                            })
                else:
                    # Simple thought generation
                    thoughts = self.generate_thoughts(text)
                    
                    for j, thought in enumerate(thoughts):
                        dataset_entries.append({
                            "id": f"{i}_{j}",
                            "source_text": text,
                            "tow_token": thought,
                            "language": self.language,
                            "domain": self._detect_domain(text)
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to process text {i}: {e}")
                continue
        
        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(dataset_entries)} TOW dataset entries to {output_path}")
    
    def get_option2_statistics(self) -> Dict[str, Any]:
        """Get Option 2 specific statistics"""
        classifier_stats = self.token_classifier.get_classification_statistics()
        
        return {
            "processor_type": "Option 2 - Pure Original TOW",
            "language": self.language,
            "classification_stats": classifier_stats,
            "cross_lingual_enabled": True,
            "tow_format": "<ToW>English reasoning</ToW>",
            "supported_categories": [cat.value for cat in TOWCategory]
        }