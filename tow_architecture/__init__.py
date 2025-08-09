"""
Thoughts of Words (ToW) Architecture
===================================

A comprehensive AI architecture for improving multilingual LLM accuracy through
English intermediary reasoning. This system enables models to perform cognitive
bridging between English reasoning and target language output.

Core Components:
- ToW Mechanism: English intermediary reasoning system
- Model Adapters: Support for DeepSeek, Llama, Qwen models
- Cognitive Bridge: Cross-lingual reasoning coordination
- Evaluation Pipeline: Multilingual accuracy assessment

Author: ToW Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ToW Research Team"

from .core import (
    ToWEngine,
    CognitiveBridge,
    ThoughtTokenProcessor,
    MultilingualProcessor
)

from .models import (
    ModelAdapterFactory,
    DeepSeekAdapter,
    LlamaAdapter,
    QwenAdapter
)

from .inference import (
    ToWInferenceEngine,
    BatchInferenceProcessor,
    StreamingProcessor
)

from .evaluation import (
    MultilingualEvaluator,
    ToWBenchmark,
    AccuracyMetrics
)

from .training import (
    ToWTrainer,
    DatasetBuilder,
    FineTuningManager
)

__all__ = [
    # Core components
    "ToWEngine",
    "CognitiveBridge", 
    "ThoughtTokenProcessor",
    "MultilingualProcessor",
    
    # Model adapters
    "ModelAdapterFactory",
    "DeepSeekAdapter",
    "LlamaAdapter", 
    "QwenAdapter",
    
    # Inference engine
    "ToWInferenceEngine",
    "BatchInferenceProcessor",
    "StreamingProcessor",
    
    # Evaluation
    "MultilingualEvaluator",
    "ToWBenchmark",
    "AccuracyMetrics",
    
    # Training
    "ToWTrainer",
    "DatasetBuilder",
    "FineTuningManager"
]