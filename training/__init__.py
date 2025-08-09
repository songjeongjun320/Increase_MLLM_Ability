"""
Training package for ToW multilingual LLM research.

This package contains training pipelines, data processing, and model fine-tuning
components for the Thoughts of Words methodology.
"""

from .core.trainer import ToWTrainer
from .data.dataset import ToWDataset, ParallelCorporaDataset
from .models.adapter_trainer import LoRATrainer, QLoRATrainer
from .utils.metrics import MultilingualMetrics

__version__ = "0.1.0"
__all__ = [
    "ToWTrainer",
    "ToWDataset",
    "ParallelCorporaDataset", 
    "LoRATrainer",
    "QLoRATrainer",
    "MultilingualMetrics",
]