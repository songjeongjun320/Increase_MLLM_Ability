"""
Data Augmentation Module for TOW System (Option 2)
=================================================

This module provides data augmentation capabilities for generating
TOW training datasets with proper token classification and 
cross-lingual English thought generation.

Key Components:
- Pipeline: Main data augmentation pipeline
- TokenClassifier: TOW token classification (trivial/exact/soft/unpredictable)  
- CrossLingualTOW: English-only thought generation for any language
- Training utilities: Dataset generation and export tools
"""

from .pipeline import (
    TOWDataAugmentationPipeline,
    TOWEntry,
    ProcessingStats
)

__all__ = [
    "TOWDataAugmentationPipeline",
    "TOWEntry", 
    "ProcessingStats"
]

__version__ = "1.0.0"
