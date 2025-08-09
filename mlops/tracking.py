"""
Experiment Tracking and Model Registry
====================================

Provides comprehensive experiment tracking, model versioning, and registry
management for ToW research. Integrates with MLflow and Weights & Biases.
"""

import json
import time
import logging
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

import torch
import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import MLOpsConfig, ExperimentConfig


logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics"""
    # Performance metrics
    accuracy: float = 0.0
    bleu_score: float = 0.0
    rouge_l: float = 0.0
    bert_score: float = 0.0
    perplexity: float = 0.0
    
    # ToW-specific metrics
    thought_coherence: float = 0.0
    cultural_adaptation_score: float = 0.0
    translation_quality: float = 0.0
    reasoning_accuracy: float = 0.0
    
    # Training metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    
    # System metrics
    memory_usage_mb: float = 0.0
    training_time_hours: float = 0.0
    gpu_utilization: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        result = {
            'accuracy': self.accuracy,
            'bleu_score': self.bleu_score,
            'rouge_l': self.rouge_l,
            'bert_score': self.bert_score,
            'perplexity': self.perplexity,
            'thought_coherence': self.thought_coherence,
            'cultural_adaptation_score': self.cultural_adaptation_score,
            'translation_quality': self.translation_quality,
            'reasoning_accuracy': self.reasoning_accuracy,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'learning_rate': self.learning_rate,
            'gradient_norm': self.gradient_norm,
            'memory_usage_mb': self.memory_usage_mb,
            'training_time_hours': self.training_time_hours,
            'gpu_utilization': self.gpu_utilization,
            'throughput_samples_per_sec': self.throughput_samples_per_sec
        }
        result.update(self.custom_metrics)
        return result


@dataclass
class ExperimentInfo:
    """Container for experiment information"""
    experiment_id: str
    run_id: str
    run_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "RUNNING"
    tags: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    artifacts: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    Unified experiment tracking interface for ToW research.
    Supports MLflow and Weights & Biases backends.
    """
    
    def __init__(self, config: ExperimentConfig, project_name: str = "tow-research"):
        self.config = config
        self.project_name = project_name
        self.current_run = None
        self._setup_backends()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.tracking_uri)
        if config.artifact_location:
            os.makedirs(config.artifact_location, exist_ok=True)
        
        self.client = MlflowClient()
        
        # Initialize W&B if available
        self.wandb_run = None
        if WANDB_AVAILABLE and config.wandb_api_key:
            wandb.login(key=config.wandb_api_key)
    
    def _setup_backends(self):
        """Setup tracking backends"""
        # Ensure MLflow experiment exists
        try:
            experiment = mlflow.get_experiment_by_name(self.project_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.project_name,
                    artifact_location=self.config.artifact_location
                )
                logger.info(f"Created MLflow experiment: {experiment_id}")
            else:
                logger.info(f"Using existing MLflow experiment: {experiment.experiment_id}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
    
    @contextmanager
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  nested: bool = False):
        """
        Start a new experiment run with context management.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            nested: Whether this is a nested run
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"tow_experiment_{timestamp}"
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            self.current_run = run
            
            # Start W&B run if configured
            if WANDB_AVAILABLE and self.config.wandb_api_key:
                wandb_config = {}
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=run_name,
                    config=wandb_config,
                    reinit=True
                )
            
            # Log initial tags
            if tags:
                self.log_tags(tags)
            
            # Log system information
            if self.config.log_system_metrics:
                self._log_system_info()
            
            try:
                yield ExperimentInfo(
                    experiment_id=run.info.experiment_id,
                    run_id=run.info.run_id,
                    run_name=run_name,
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                    tags=tags or {}
                )
            finally:
                # Finish W&B run
                if self.wandb_run:
                    self.wandb_run.finish()
                    self.wandb_run = None
                
                self.current_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters"""
        if not self.current_run:
            logger.warning("No active run to log parameters")
            return
        
        # Log to MLflow
        mlflow.log_params(params)
        
        # Log to W&B
        if self.wandb_run:
            self.wandb_run.config.update(params)
    
    def log_metrics(self, metrics: Union[Dict[str, float], ExperimentMetrics], step: Optional[int] = None) -> None:
        """Log experiment metrics"""
        if not self.current_run:
            logger.warning("No active run to log metrics")
            return
        
        if isinstance(metrics, ExperimentMetrics):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = metrics
        
        # Log to MLflow
        mlflow.log_metrics(metrics_dict, step=step)
        
        # Log to W&B
        if self.wandb_run:
            log_data = metrics_dict.copy()
            if step is not None:
                log_data['step'] = step
            self.wandb_run.log(log_data)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric"""
        self.log_metrics({key: value}, step=step)
    
    def log_tags(self, tags: Dict[str, str]) -> None:
        """Log experiment tags"""
        if not self.current_run:
            logger.warning("No active run to log tags")
            return
        
        # Log to MLflow
        mlflow.set_tags(tags)
        
        # Log to W&B as tags
        if self.wandb_run:
            self.wandb_run.tags = list(tags.values())
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log an artifact (file or directory)"""
        if not self.current_run:
            logger.warning("No active run to log artifact")
            return
        
        local_path = Path(local_path)
        
        # Log to MLflow
        if local_path.is_file():
            mlflow.log_artifact(str(local_path), artifact_path)
        elif local_path.is_dir():
            mlflow.log_artifacts(str(local_path), artifact_path)
        else:
            logger.warning(f"Artifact path does not exist: {local_path}")
            return
        
        # Log to W&B
        if self.wandb_run:
            if local_path.is_file():
                self.wandb_run.save(str(local_path))
            # W&B doesn't support logging directories directly
    
    def log_model(self, 
                  model: torch.nn.Module,
                  model_name: str = "model",
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a PyTorch model"""
        if not self.current_run:
            logger.warning("No active run to log model")
            return ""
        
        # Prepare model info
        model_info = {
            "model_type": type(model).__name__,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        if metadata:
            model_info.update(metadata)
        
        # Log to MLflow
        model_uri = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_name,
            metadata=model_info
        ).model_uri
        
        # Log to W&B
        if self.wandb_run:
            # Save model as artifact in W&B
            model_artifact = wandb.Artifact(model_name, type="model", metadata=model_info)
            
            # Create temporary model file
            temp_path = Path(f"/tmp/{model_name}_{int(time.time())}.pth")
            torch.save(model.state_dict(), temp_path)
            
            model_artifact.add_file(str(temp_path))
            self.wandb_run.log_artifact(model_artifact)
            
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)
        
        return model_uri
    
    def log_dataset(self, 
                    dataset_path: Union[str, Path],
                    dataset_name: str = "dataset",
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a dataset"""
        if not self.current_run:
            logger.warning("No active run to log dataset")
            return
        
        dataset_path = Path(dataset_path)
        
        # Calculate dataset hash for versioning
        dataset_hash = self._calculate_file_hash(dataset_path)
        
        # Prepare metadata
        dataset_info = {
            "name": dataset_name,
            "path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size if dataset_path.exists() else 0,
            "hash": dataset_hash
        }
        
        if metadata:
            dataset_info.update(metadata)
        
        # Log dataset info as parameters
        self.log_params({f"dataset_{dataset_name}": dataset_info})
        
        # Log dataset file as artifact
        if dataset_path.exists():
            self.log_artifact(dataset_path, f"datasets/{dataset_name}")
    
    def log_code(self, code_path: Union[str, Path] = ".") -> None:
        """Log source code"""
        if not self.current_run:
            logger.warning("No active run to log code")
            return
        
        code_path = Path(code_path)
        
        # Log to MLflow
        mlflow.log_artifacts(str(code_path), "code")
        
        # Log to W&B
        if self.wandb_run:
            code_artifact = wandb.Artifact("source_code", type="code")
            code_artifact.add_dir(str(code_path))
            self.wandb_run.log_artifact(code_artifact)
    
    def _log_system_info(self) -> None:
        """Log system information"""
        import psutil
        import platform
        
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
        }
        
        # GPU information
        if torch.cuda.is_available():
            system_info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            })
        else:
            system_info["cuda_available"] = False
        
        self.log_params({"system": system_info})
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        if not file_path.exists():
            return ""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def get_experiment_runs(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """Get runs from an experiment"""
        if experiment_name is None:
            experiment_name = self.project_name
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_name}")
            return pd.DataFrame()
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ALL
        )
        
        return runs
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs"""
        runs_data = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                }
                
                # Add parameters
                run_data.update({f"param_{k}": v for k, v in run.data.params.items()})
                
                # Add metrics
                run_data.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
                
                runs_data.append(run_data)
                
            except Exception as e:
                logger.warning(f"Failed to get run {run_id}: {e}")
        
        return pd.DataFrame(runs_data)


class ModelRegistry:
    """
    Model registry for managing ToW model versions and deployments.
    Provides model versioning, staging, and deployment capabilities.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = MlflowClient()
        self._setup_registry()
    
    def _setup_registry(self):
        """Setup model registry"""
        # Create registered model if it doesn't exist
        try:
            self.client.get_registered_model(self.config.model_registry_name)
            logger.info(f"Using existing registered model: {self.config.model_registry_name}")
        except MlflowException:
            self.client.create_registered_model(
                self.config.model_registry_name,
                description="ToW (Thoughts of Words) enhanced language models"
            )
            logger.info(f"Created registered model: {self.config.model_registry_name}")
    
    def register_model(self, 
                      model_uri: str,
                      model_name: Optional[str] = None,
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> str:
        """Register a model from an MLflow run"""
        if model_name is None:
            model_name = self.config.model_registry_name
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            description=description,
            tags=tags
        )
        
        logger.info(f"Registered model version: {model_version.version}")
        return model_version.version
    
    def transition_model_stage(self, 
                              model_name: str,
                              version: str,
                              stage: str,
                              archive_existing_versions: bool = False) -> None:
        """Transition model to different stage (Staging, Production, Archived)"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        
        logger.info(f"Transitioned model {model_name} v{version} to {stage}")
    
    def get_model_version(self, 
                         model_name: str,
                         version: Optional[str] = None,
                         stage: Optional[str] = None) -> Any:
        """Get specific model version"""
        if version:
            return self.client.get_model_version(model_name, version)
        elif stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            return versions[0] if versions else None
        else:
            # Get latest version
            versions = self.client.get_latest_versions(model_name)
            return versions[0] if versions else None
    
    def load_model(self, 
                   model_name: str,
                   version: Optional[str] = None,
                   stage: Optional[str] = None) -> torch.nn.Module:
        """Load a registered model"""
        model_version = self.get_model_version(model_name, version, stage)
        
        if model_version is None:
            raise ValueError(f"Model not found: {model_name}")
        
        model_uri = f"models:/{model_name}/{model_version.version}"
        model = mlflow.pytorch.load_model(model_uri)
        
        logger.info(f"Loaded model {model_name} v{model_version.version}")
        return model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []
        
        for registered_model in self.client.search_registered_models():
            latest_versions = self.client.get_latest_versions(registered_model.name)
            
            model_info = {
                "name": registered_model.name,
                "description": registered_model.description,
                "creation_timestamp": registered_model.creation_timestamp,
                "last_updated_timestamp": registered_model.last_updated_timestamp,
                "latest_versions": [
                    {
                        "version": version.version,
                        "stage": version.current_stage,
                        "creation_timestamp": version.creation_timestamp
                    }
                    for version in latest_versions
                ]
            }
            
            models.append(model_info)
        
        return models
    
    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a specific model version"""
        self.client.delete_model_version(model_name, version)
        logger.info(f"Deleted model version {model_name} v{version}")
    
    def update_model_version(self, 
                           model_name: str,
                           version: str,
                           description: Optional[str] = None) -> None:
        """Update model version description"""
        if description:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
            logger.info(f"Updated model version {model_name} v{version}")


def create_tracker(config: MLOpsConfig) -> ExperimentTracker:
    """Create experiment tracker from MLOps config"""
    return ExperimentTracker(config.experiments, config.project_name)


def create_registry(config: MLOpsConfig) -> ModelRegistry:
    """Create model registry from MLOps config"""
    return ModelRegistry(config.experiments)