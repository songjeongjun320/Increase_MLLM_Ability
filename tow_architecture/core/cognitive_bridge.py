"""
Cognitive Bridge - Cross-Lingual Reasoning Coordinator
=====================================================

The CognitiveBridge manages the transition from English-based reasoning
to target language output, ensuring semantic consistency and cultural
appropriateness across languages.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..models.base_adapter import BaseModelAdapter
from ..utils.config import CognitiveBridgeConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BridgeStrategy(Enum):
    """Different strategies for cognitive bridging"""
    DIRECT_TRANSLATION = "direct_translation"
    SEMANTIC_MAPPING = "semantic_mapping" 
    CULTURAL_ADAPTATION = "cultural_adaptation"
    CONTEXTUAL_BRIDGING = "contextual_bridging"


@dataclass
class LanguagePair:
    """Language pair configuration"""
    source: str
    target: str
    bridge_strategy: BridgeStrategy = BridgeStrategy.CONTEXTUAL_BRIDGING
    cultural_markers: List[str] = None
    semantic_patterns: Dict[str, str] = None


@dataclass
class BridgeContext:
    """Context for cognitive bridging"""
    thought_tokens: List[str]
    source_text: str
    target_language: str
    task_type: str
    cultural_context: Optional[Dict[str, Any]] = None
    domain_knowledge: Optional[Dict[str, Any]] = None


class CognitiveBridge:
    """
    Cognitive Bridge for cross-lingual reasoning coordination.
    
    This component manages the critical transition between English-based
    reasoning (thought tokens) and target language output generation,
    ensuring semantic consistency and cultural appropriateness.
    """
    
    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        config: Optional[CognitiveBridgeConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Cognitive Bridge.
        
        Args:
            model_adapter: Model adapter for language processing
            config: Bridge configuration
            device: PyTorch device for computation
        """
        self.model_adapter = model_adapter
        self.config = config or CognitiveBridgeConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Language pair configurations
        self.language_pairs = self._initialize_language_pairs()
        
        # Cultural and semantic knowledge bases
        self.cultural_kb = self._load_cultural_knowledge()
        self.semantic_kb = self._load_semantic_knowledge()
        
        # Bridge templates and patterns
        self.bridge_templates = self._load_bridge_templates()
        
        logger.info("CognitiveBridge initialized")
    
    def bridge_languages(
        self,
        source_text: str,
        thought_tokens: List[str],
        target_language: str,
        analysis: Dict[str, Any],
        task_type: str = "generation"
    ) -> Dict[str, Any]:
        """
        Perform cognitive bridging between English thoughts and target language.
        
        Args:
            source_text: Original input text
            thought_tokens: English thought tokens
            target_language: Target language code
            analysis: Input analysis results
            task_type: Type of task being performed
            
        Returns:
            Dictionary containing bridge processing results
        """
        logger.info(f"Bridging to {target_language} with {len(thought_tokens)} thought tokens")
        
        # Create bridge context
        context = BridgeContext(
            thought_tokens=thought_tokens,
            source_text=source_text,
            target_language=target_language,
            task_type=task_type,
            cultural_context=self._get_cultural_context(target_language),
            domain_knowledge=self._extract_domain_knowledge(source_text, analysis)
        )
        
        # Determine bridging strategy
        strategy = self._select_bridge_strategy(context, analysis)
        
        # Apply bridging strategy
        bridge_result = self._apply_bridge_strategy(context, strategy)
        
        # Validate bridge quality
        quality_score = self._validate_bridge_quality(context, bridge_result)
        
        return {
            "bridge_strategy": strategy.value,
            "bridged_thoughts": bridge_result["bridged_thoughts"],
            "semantic_mapping": bridge_result["semantic_mapping"],
            "cultural_adaptations": bridge_result["cultural_adaptations"],
            "quality_score": quality_score,
            "context": context,
            "metadata": bridge_result.get("metadata", {})
        }
    
    def _initialize_language_pairs(self) -> Dict[str, LanguagePair]:
        """Initialize supported language pairs with their configurations"""
        pairs = {}
        
        # English-Korean pair
        pairs["en-ko"] = LanguagePair(
            source="en",
            target="ko",
            bridge_strategy=BridgeStrategy.CONTEXTUAL_BRIDGING,
            cultural_markers=["honorifics", "formality_levels", "age_hierarchy"],
            semantic_patterns={
                "subject_drop": "Korean allows subject dropping",
                "verb_final": "Korean has SOV word order",
                "honorific_system": "Korean has complex honorific system"
            }
        )
        
        # English-Chinese pair
        pairs["en-zh"] = LanguagePair(
            source="en",
            target="zh",
            bridge_strategy=BridgeStrategy.SEMANTIC_MAPPING,
            cultural_markers=["collectivism", "hierarchy", "face_concept"],
            semantic_patterns={
                "no_inflection": "Chinese lacks grammatical inflection",
                "measure_words": "Chinese uses measure word system",
                "tone_meaning": "Chinese uses tones for meaning"
            }
        )
        
        return pairs
    
    def _load_cultural_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural knowledge base for different languages"""
        return {
            "ko": {
                "communication_style": "high_context",
                "formality_importance": "high",
                "hierarchy_respect": "critical",
                "directness": "low",
                "cultural_concepts": ["nunchi", "jeong", "han"]
            },
            "zh": {
                "communication_style": "high_context", 
                "formality_importance": "medium",
                "hierarchy_respect": "high",
                "directness": "medium",
                "cultural_concepts": ["guanxi", "mianzi", "li"]
            },
            "en": {
                "communication_style": "low_context",
                "formality_importance": "low",
                "hierarchy_respect": "medium",
                "directness": "high",
                "cultural_concepts": ["individualism", "efficiency", "directness"]
            }
        }
    
    def _load_semantic_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Load semantic knowledge for cross-lingual mapping"""
        return {
            "grammatical_patterns": {
                "ko": {
                    "word_order": "SOV",
                    "subject_dropping": True,
                    "honorifics": True,
                    "particles": True
                },
                "zh": {
                    "word_order": "SVO",
                    "measure_words": True,
                    "tones": True,
                    "characters": True
                }
            },
            "semantic_fields": {
                "emotions": {
                    "ko": {"한": "deep sorrow", "정": "affection/attachment"},
                    "zh": {"面子": "face/dignity", "关系": "relationships"}
                },
                "social_concepts": {
                    "ko": {"눈치": "social awareness", "선후배": "senior-junior"},
                    "zh": {"孝": "filial piety", "和": "harmony"}
                }
            }
        }
    
    def _load_bridge_templates(self) -> Dict[str, List[str]]:
        """Load templates for different bridging scenarios"""
        return {
            "thought_integration": [
                "Based on the reasoning that {thought}, in {target_lang} this would be expressed as:",
                "Considering {thought}, the culturally appropriate way to convey this in {target_lang} is:",
                "The thought '{thought}' translates to the following {target_lang} concept:"
            ],
            "cultural_adaptation": [
                "Adapting for {target_culture} communication style:",
                "Considering {target_culture} cultural norms:",
                "In {target_culture} context, this should be framed as:"
            ],
            "semantic_mapping": [
                "Mapping English concept '{concept}' to {target_lang} semantic field:",
                "The semantic structure in {target_lang} requires:",
                "Cross-linguistic mapping shows:"
            ]
        }
    
    def _select_bridge_strategy(
        self, 
        context: BridgeContext, 
        analysis: Dict[str, Any]
    ) -> BridgeStrategy:
        """Select appropriate bridging strategy based on context and analysis"""
        
        # Get language pair configuration
        pair_key = f"en-{context.target_language}"
        if pair_key in self.language_pairs:
            default_strategy = self.language_pairs[pair_key].bridge_strategy
        else:
            default_strategy = BridgeStrategy.SEMANTIC_MAPPING
        
        # Adjust strategy based on analysis
        complexity_score = analysis.get("complexity_score", 0.5)
        task_requirements = analysis.get("task_requirements", {})
        
        if task_requirements.get("requires_accuracy"):
            return BridgeStrategy.DIRECT_TRANSLATION
        elif complexity_score > 0.7:
            return BridgeStrategy.CONTEXTUAL_BRIDGING
        elif task_requirements.get("requires_creativity"):
            return BridgeStrategy.CULTURAL_ADAPTATION
        else:
            return default_strategy
    
    def _apply_bridge_strategy(
        self, 
        context: BridgeContext, 
        strategy: BridgeStrategy
    ) -> Dict[str, Any]:
        """Apply the selected bridging strategy"""
        
        if strategy == BridgeStrategy.DIRECT_TRANSLATION:
            return self._direct_translation_bridge(context)
        elif strategy == BridgeStrategy.SEMANTIC_MAPPING:
            return self._semantic_mapping_bridge(context)
        elif strategy == BridgeStrategy.CULTURAL_ADAPTATION:
            return self._cultural_adaptation_bridge(context)
        elif strategy == BridgeStrategy.CONTEXTUAL_BRIDGING:
            return self._contextual_bridging(context)
        else:
            logger.warning(f"Unknown strategy {strategy}, falling back to semantic mapping")
            return self._semantic_mapping_bridge(context)
    
    def _direct_translation_bridge(self, context: BridgeContext) -> Dict[str, Any]:
        """Apply direct translation bridging strategy"""
        bridged_thoughts = []
        semantic_mapping = {}
        cultural_adaptations = []
        
        for thought in context.thought_tokens:
            # Direct semantic mapping
            bridged_thought = self._map_thought_directly(thought, context.target_language)
            bridged_thoughts.append(bridged_thought)
            semantic_mapping[thought] = bridged_thought
        
        return {
            "bridged_thoughts": bridged_thoughts,
            "semantic_mapping": semantic_mapping,
            "cultural_adaptations": cultural_adaptations,
            "metadata": {"strategy": "direct_translation"}
        }
    
    def _semantic_mapping_bridge(self, context: BridgeContext) -> Dict[str, Any]:
        """Apply semantic mapping bridging strategy"""
        bridged_thoughts = []
        semantic_mapping = {}
        cultural_adaptations = []
        
        # Get semantic patterns for target language
        target_patterns = self.semantic_kb["grammatical_patterns"].get(
            context.target_language, {}
        )
        
        for thought in context.thought_tokens:
            # Apply semantic transformations
            mapped_thought = self._apply_semantic_transformations(
                thought, target_patterns, context.target_language
            )
            bridged_thoughts.append(mapped_thought)
            semantic_mapping[thought] = mapped_thought
        
        return {
            "bridged_thoughts": bridged_thoughts,
            "semantic_mapping": semantic_mapping,
            "cultural_adaptations": cultural_adaptations,
            "metadata": {
                "strategy": "semantic_mapping",
                "patterns_applied": list(target_patterns.keys())
            }
        }
    
    def _cultural_adaptation_bridge(self, context: BridgeContext) -> Dict[str, Any]:
        """Apply cultural adaptation bridging strategy"""
        bridged_thoughts = []
        semantic_mapping = {}
        cultural_adaptations = []
        
        # Get cultural context
        cultural_info = context.cultural_context or {}
        
        for thought in context.thought_tokens:
            # Apply cultural adaptations
            adapted_thought, adaptations = self._adapt_culturally(
                thought, context.target_language, cultural_info
            )
            bridged_thoughts.append(adapted_thought)
            semantic_mapping[thought] = adapted_thought
            cultural_adaptations.extend(adaptations)
        
        return {
            "bridged_thoughts": bridged_thoughts,
            "semantic_mapping": semantic_mapping,
            "cultural_adaptations": cultural_adaptations,
            "metadata": {
                "strategy": "cultural_adaptation",
                "cultural_markers": cultural_info.get("cultural_concepts", [])
            }
        }
    
    def _contextual_bridging(self, context: BridgeContext) -> Dict[str, Any]:
        """Apply comprehensive contextual bridging strategy"""
        bridged_thoughts = []
        semantic_mapping = {}
        cultural_adaptations = []
        
        # Combine all bridging approaches
        for thought in context.thought_tokens:
            # Start with semantic mapping
            semantic_result = self._apply_semantic_transformations(
                thought, 
                self.semantic_kb["grammatical_patterns"].get(context.target_language, {}),
                context.target_language
            )
            
            # Apply cultural adaptations
            cultural_result, adaptations = self._adapt_culturally(
                semantic_result, 
                context.target_language,
                context.cultural_context or {}
            )
            
            # Final contextual refinement
            final_thought = self._refine_contextually(
                cultural_result, context
            )
            
            bridged_thoughts.append(final_thought)
            semantic_mapping[thought] = final_thought
            cultural_adaptations.extend(adaptations)
        
        return {
            "bridged_thoughts": bridged_thoughts,
            "semantic_mapping": semantic_mapping,
            "cultural_adaptations": cultural_adaptations,
            "metadata": {
                "strategy": "contextual_bridging",
                "processing_stages": ["semantic", "cultural", "contextual"]
            }
        }
    
    def _map_thought_directly(self, thought: str, target_language: str) -> str:
        """Map thought directly without major transformations"""
        # Simplified direct mapping
        return f"[{target_language.upper()}] {thought}"
    
    def _apply_semantic_transformations(
        self, 
        thought: str, 
        target_patterns: Dict[str, Any], 
        target_language: str
    ) -> str:
        """Apply semantic transformations based on target language patterns"""
        transformed = thought
        
        # Apply grammatical pattern transformations
        if target_patterns.get("word_order") == "SOV":
            transformed = f"[SOV-adapted] {transformed}"
        
        if target_patterns.get("honorifics"):
            transformed = f"[With-honorifics] {transformed}"
        
        if target_patterns.get("particles"):
            transformed = f"[With-particles] {transformed}"
        
        return transformed
    
    def _adapt_culturally(
        self, 
        thought: str, 
        target_language: str, 
        cultural_info: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Adapt thought for cultural appropriateness"""
        adapted = thought
        adaptations = []
        
        cultural_data = self.cultural_kb.get(target_language, {})
        
        # Apply communication style adaptations
        comm_style = cultural_data.get("communication_style")
        if comm_style == "high_context":
            adapted = f"[High-context] {adapted}"
            adaptations.append("high_context_communication")
        
        # Apply formality adaptations
        formality = cultural_data.get("formality_importance")
        if formality == "high":
            adapted = f"[Formal] {adapted}"
            adaptations.append("formal_register")
        
        return adapted, adaptations
    
    def _refine_contextually(self, thought: str, context: BridgeContext) -> str:
        """Apply final contextual refinement"""
        refined = thought
        
        # Task-specific refinements
        if context.task_type == "translation":
            refined = f"[Translation-optimized] {refined}"
        elif context.task_type == "generation":
            refined = f"[Generation-optimized] {refined}"
        
        return refined
    
    def _get_cultural_context(self, target_language: str) -> Dict[str, Any]:
        """Get cultural context for target language"""
        return self.cultural_kb.get(target_language, {})
    
    def _extract_domain_knowledge(
        self, 
        source_text: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract domain-specific knowledge from source text and analysis"""
        return {
            "domain": "general",  # Could be enhanced with domain detection
            "technical_terms": [],  # Could extract technical terms
            "specialized_concepts": []  # Could identify specialized concepts
        }
    
    def _validate_bridge_quality(
        self, 
        context: BridgeContext, 
        bridge_result: Dict[str, Any]
    ) -> float:
        """Validate the quality of cognitive bridging"""
        quality_factors = []
        
        # Check thought coverage
        original_count = len(context.thought_tokens)
        bridged_count = len(bridge_result.get("bridged_thoughts", []))
        coverage_score = min(bridged_count / original_count, 1.0) if original_count > 0 else 0.0
        quality_factors.append(coverage_score)
        
        # Check cultural adaptation completeness
        adaptations = bridge_result.get("cultural_adaptations", [])
        adaptation_score = min(len(adaptations) / 3, 1.0)  # Expect up to 3 adaptations
        quality_factors.append(adaptation_score)
        
        # Check semantic mapping consistency
        semantic_mapping = bridge_result.get("semantic_mapping", {})
        mapping_score = min(len(semantic_mapping) / original_count, 1.0) if original_count > 0 else 0.0
        quality_factors.append(mapping_score)
        
        # Calculate weighted quality score
        if quality_factors:
            return sum(quality_factors) / len(quality_factors)
        else:
            return 0.0