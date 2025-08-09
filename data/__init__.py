"""
Data management package for ToW multilingual LLM research.

This package handles data collection, preprocessing, augmentation,
and management for multilingual training datasets.
"""

from .processors.parallel_corpora import ParallelCorporaProcessor
from .processors.tow_augmentation import ToWAugmentationProcessor
from .loaders.multilingual import MultilingualDataLoader
from .utils.language_detection import LanguageDetector
from .utils.quality_filtering import QualityFilter

__version__ = "0.1.0"
__all__ = [
    "ParallelCorporaProcessor",
    "ToWAugmentationProcessor",
    "MultilingualDataLoader",
    "LanguageDetector", 
    "QualityFilter",
]