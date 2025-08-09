"""
Configuration Classes for TOW System (Option 2)
===============================================

Configuration classes for the Pure Original TOW Implementation with:
- Data augmentation pipeline settings  
- Token classification parameters
- Cross-lingual TOW generation options
- Training dataset configurations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass 
class DataAugmentationConfig:
    """Configuration for data augmentation pipeline"""
    
    # Processing settings
    min_text_length: int = 50
    min_sentence_length: int = 10
    max_entries_per_text: int = 10
    max_positions_per_sentence: int = 5
    processing_timeout: float = 30.0
    
    # Batch processing
    batch_size: int = 100
    max_workers: int = 4
    
    # Quality filtering
    min_confidence_threshold: float = 0.3
    min_thought_length: int = 10
    max_thought_length: int = 200
    
    # Output formatting
    output_format: str = "jsonl"  # jsonl, json, txt
    include_metadata: bool = True
    include_statistics: bool = True


@dataclass
class TokenClassifierConfig:
    """Configuration for token classification system"""
    
    # Classification thresholds
    trivial_frequency_threshold: float = 0.8
    exact_match_threshold: float = 1.0
    soft_consistency_threshold: float = 0.6
    unpredictable_threshold: float = 0.3
    
    # Language-specific settings
    language: str = "ko"
    enable_cross_lingual: bool = True
    
    # Quality metrics
    min_confidence: float = 0.3
    min_relevance: float = 0.2
    min_coherence: float = 0.4


@dataclass
class CrossLingualConfig:
    """Configuration for cross-lingual TOW system"""
    
    # Language settings
    source_languages: List[str] = field(default_factory=lambda: ["ko", "zh", "ja", "en"])
    thought_language: str = "en"  # Always English for thoughts
    enable_auto_detection: bool = True
    
    # Reasoning enhancement
    include_linguistic_patterns: bool = True
    include_domain_context: bool = True
    include_category_specific_reasoning: bool = True
    
    # Template settings
    use_dynamic_templates: bool = True
    max_reasoning_length: int = 200
    min_reasoning_length: int = 20


@dataclass
class ThoughtProcessorConfig:
    """Configuration for thought token processor"""
    
    # Generation settings
    max_thought_tokens: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens_per_thought: int = 50
    
    # Quality thresholds
    min_confidence: float = 0.3
    min_relevance: float = 0.2
    min_coherence: float = 0.4
    
    # Processing options
    enable_classification: bool = True
    enable_cross_lingual: bool = True
    include_reasoning_chains: bool = True


@dataclass
class MultilingualProcessorConfig:
    """Configuration for multilingual processor"""
    
    # Language settings
    default_target_language: str = "ko"
    supported_languages: List[str] = field(default_factory=lambda: ["ko", "en", "zh", "ja"])
    
    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Style settings
    default_style: str = "neutral"
    cultural_adaptation: bool = True
    
    # Quality settings
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.6
    alignment_threshold: float = 0.5


@dataclass
class CognitiveBridgeConfig:
    """Configuration for cognitive bridge system"""
    
    # Bridge settings
    bridge_strategy: str = "contextual_bridging"
    enable_semantic_mapping: bool = True
    enable_cultural_adaptation: bool = True
    
    # Quality settings
    min_bridge_quality: float = 0.6
    max_bridge_attempts: int = 3
    
    # Processing settings
    bridge_timeout: float = 10.0
    enable_caching: bool = True


@dataclass
class TOWConfig:
    """Main configuration class for TOW system"""
    
    # Component configurations
    data_augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)
    token_classifier: TokenClassifierConfig = field(default_factory=TokenClassifierConfig)
    cross_lingual: CrossLingualConfig = field(default_factory=CrossLingualConfig)
    thought_processor: ThoughtProcessorConfig = field(default_factory=ThoughtProcessorConfig)
    multilingual_processor: MultilingualProcessorConfig = field(default_factory=MultilingualProcessorConfig)
    cognitive_bridge: CognitiveBridgeConfig = field(default_factory=CognitiveBridgeConfig)
    
    # Global settings
    option_type: str = "Option 2 - Pure Original TOW"
    version: str = "1.0.0"
    language: str = "ko"
    device: str = "auto"  # auto, cpu, cuda
    
    # Paths
    model_cache_dir: Optional[str] = None
    output_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_detailed_logging: bool = False
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: float = 60.0
    memory_limit_gb: Optional[float] = None
    
    # Feature flags
    enable_token_classification: bool = True
    enable_cross_lingual_thoughts: bool = True
    enable_data_augmentation: bool = True
    enable_quality_filtering: bool = True
    enable_statistics_tracking: bool = True
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        
        # Check required settings
        if not self.language:
            raise ValueError("Language must be specified")
        
        if self.language not in self.cross_lingual.source_languages:
            raise ValueError(f"Language '{self.language}' not supported")
        
        # Validate thresholds
        if not 0 < self.token_classifier.soft_consistency_threshold < 1:
            raise ValueError("Soft consistency threshold must be between 0 and 1")
        
        if not 0 < self.thought_processor.min_confidence < 1:
            raise ValueError("Min confidence must be between 0 and 1")
        
        # Validate paths
        if self.output_dir and not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.temp_dir and not Path(self.temp_dir).exists():
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        return True
    
    def get_language_specific_config(self, language: str) -> Dict[str, Any]:
        """Get language-specific configuration"""
        
        language_configs = {
            "ko": {
                "trivial_words": ["은", "는", "이", "가", "을", "를", "에", "서", "로"],
                "function_words": ["그", "이", "저", "여기", "거기", "저기"],
                "particles": ["은", "는", "이", "가", "을", "를", "도", "만", "부터", "까지"],
                "honorific_markers": ["습니다", "입니다", "해요", "세요"],
                "cultural_markers": ["hierarchy", "age_respect", "social_distance"]
            },
            "en": {
                "trivial_words": ["the", "a", "an", "and", "or", "but", "in", "on", "at"],
                "function_words": ["this", "that", "these", "those", "here", "there"],
                "articles": ["a", "an", "the"],
                "prepositions": ["in", "on", "at", "to", "for", "with", "by"],
                "cultural_markers": ["directness", "individualism", "efficiency"]
            },
            "zh": {
                "trivial_words": ["的", "了", "在", "是", "有", "我", "你", "他"],
                "function_words": ["这", "那", "哪", "什么", "谁", "怎么"],
                "particles": ["的", "了", "过", "着"],
                "classifiers": ["个", "只", "本", "张", "件"],
                "cultural_markers": ["collectivism", "face_saving", "hierarchy"]
            },
            "ja": {
                "trivial_words": ["は", "が", "を", "に", "で", "から", "まで", "と"],
                "function_words": ["この", "その", "あの", "どの", "ここ", "そこ"],
                "particles": ["は", "が", "を", "に", "で", "から", "まで", "と", "も"],
                "honorific_markers": ["です", "ます", "でしょう", "ございます"],
                "cultural_markers": ["hierarchy", "group_harmony", "indirect_communication"]
            }
        }
        
        return language_configs.get(language, language_configs["en"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TOWConfig":
        """Create configuration from dictionary"""
        
        # Handle nested dataclasses
        if "data_augmentation" in data and isinstance(data["data_augmentation"], dict):
            data["data_augmentation"] = DataAugmentationConfig(**data["data_augmentation"])
        
        if "token_classifier" in data and isinstance(data["token_classifier"], dict):
            data["token_classifier"] = TokenClassifierConfig(**data["token_classifier"])
        
        if "cross_lingual" in data and isinstance(data["cross_lingual"], dict):
            data["cross_lingual"] = CrossLingualConfig(**data["cross_lingual"])
        
        if "thought_processor" in data and isinstance(data["thought_processor"], dict):
            data["thought_processor"] = ThoughtProcessorConfig(**data["thought_processor"])
        
        if "multilingual_processor" in data and isinstance(data["multilingual_processor"], dict):
            data["multilingual_processor"] = MultilingualProcessorConfig(**data["multilingual_processor"])
        
        if "cognitive_bridge" in data and isinstance(data["cognitive_bridge"], dict):
            data["cognitive_bridge"] = CognitiveBridgeConfig(**data["cognitive_bridge"])
        
        return cls(**data)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "TOWConfig":
        """Load configuration from JSON file"""
        import json
        from pathlib import Path
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        from pathlib import Path
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# Backward compatibility alias for legacy imports/tests
# Some modules/tests expect `ToWConfig`; provide alias to `TOWConfig`.
ToWConfig = TOWConfig


# Predefined configurations for different use cases

def get_development_config() -> TOWConfig:
    """Get configuration optimized for development/testing"""
    config = TOWConfig()
    
    # Reduce batch sizes for faster testing
    config.data_augmentation.batch_size = 10
    config.data_augmentation.max_workers = 2
    config.data_augmentation.processing_timeout = 10.0
    
    # Lower thresholds for more permissive classification
    config.token_classifier.soft_consistency_threshold = 0.4
    config.token_classifier.min_confidence = 0.2
    
    # Enable detailed logging
    config.log_level = "DEBUG"
    config.enable_detailed_logging = True
    
    return config


def get_production_config() -> TOWConfig:
    """Get configuration optimized for production"""
    config = TOWConfig()
    
    # Increase batch sizes for better throughput
    config.data_augmentation.batch_size = 200
    config.data_augmentation.max_workers = 8
    config.data_augmentation.processing_timeout = 60.0
    
    # Higher quality thresholds
    config.token_classifier.soft_consistency_threshold = 0.7
    config.token_classifier.min_confidence = 0.5
    config.thought_processor.min_confidence = 0.5
    
    # Performance optimizations
    config.max_concurrent_requests = 20
    config.request_timeout = 120.0
    
    # Production logging
    config.log_level = "INFO"
    config.enable_detailed_logging = False
    
    return config


def get_multilingual_config(languages: List[str]) -> TOWConfig:
    """Get configuration optimized for multilingual processing"""
    config = TOWConfig()
    
    # Set supported languages
    config.cross_lingual.source_languages = languages
    config.multilingual_processor.supported_languages = languages
    
    # Enable all cross-lingual features
    config.cross_lingual.enable_auto_detection = True
    config.cross_lingual.include_linguistic_patterns = True
    config.cross_lingual.include_domain_context = True
    config.cross_lingual.include_category_specific_reasoning = True
    
    # Cultural adaptation
    config.multilingual_processor.cultural_adaptation = True
    config.cognitive_bridge.enable_cultural_adaptation = True
    
    return config


def get_research_config() -> TOWConfig:
    """Get configuration optimized for research/analysis"""
    config = TOWConfig()
    
    # Enable all features for maximum analysis
    config.enable_statistics_tracking = True
    config.thought_processor.include_reasoning_chains = True
    config.cognitive_bridge.enable_semantic_mapping = True
    
    # Detailed output
    config.data_augmentation.include_metadata = True
    config.data_augmentation.include_statistics = True
    
    # Research logging
    config.log_level = "DEBUG"
    config.enable_detailed_logging = True
    
    return config