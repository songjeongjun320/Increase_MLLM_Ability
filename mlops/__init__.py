"""
MLOps Infrastructure for Thoughts of Words (ToW) Research Project
================================================================

This package provides comprehensive MLOps capabilities including:
- Experiment tracking and model versioning
- Model deployment and serving infrastructure  
- Automated CI/CD pipelines
- Performance monitoring and alerting
- Resource management and cost optimization
"""

__version__ = "1.0.0"
__author__ = "ToW Research Team"

from .tracking import ExperimentTracker, ModelRegistry
from .deployment import ToWModelServer, DeploymentManager
from .monitoring import MetricsCollector, AlertManager
from .pipeline import TrainingPipeline, EvaluationPipeline
from .config import MLOpsConfig

__all__ = [
    "ExperimentTracker",
    "ModelRegistry", 
    "ToWModelServer",
    "DeploymentManager",
    "MetricsCollector",
    "AlertManager",
    "TrainingPipeline",
    "EvaluationPipeline",
    "MLOpsConfig"
]