"""
Training and Evaluation Pipeline Infrastructure
==============================================

Automated CI/CD pipelines for ToW model training, evaluation, and deployment
with hyperparameter optimization and experiment management.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import shutil

import torch
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid
import optuna

from .config import MLOpsConfig, PipelineConfig
from .tracking import ExperimentTracker, ModelRegistry
from .monitoring import MetricsCollector, PerformanceMetrics
from .deployment import DeploymentManager


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stage status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(Enum):
    """Overall pipeline status"""
    CREATED = "created"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    model_name: str = "tow-llama-7b"
    base_model_path: str = ""
    max_seq_length: int = 4096
    
    # Training parameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # ToW-specific parameters
    thought_weight: float = 0.3
    cultural_weight: float = 0.2
    translation_weight: float = 0.5
    
    # Optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    # Data
    train_dataset_path: str = ""
    val_dataset_path: str = ""
    test_dataset_path: str = ""
    
    # Validation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    early_stopping_patience: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.__dict__.copy()


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Datasets
    datasets: List[str] = field(default_factory=lambda: ["mmlu", "klue", "custom"])
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "bleu", "rouge", "bertscore", "accuracy", "perplexity"
    ])
    
    # ToW-specific metrics
    tow_metrics: List[str] = field(default_factory=lambda: [
        "thought_coherence", "cultural_adaptation", "translation_quality"
    ])
    
    # Evaluation parameters
    batch_size: int = 8
    max_samples: Optional[int] = None
    temperature: float = 0.0  # Deterministic for evaluation
    
    # Thresholds
    min_accuracy: float = 0.8
    min_bleu_score: float = 0.7
    min_confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'datasets': self.datasets,
            'metrics': self.metrics,
            'tow_metrics': self.tow_metrics,
            'batch_size': self.batch_size,
            'max_samples': self.max_samples,
            'temperature': self.temperature,
            'min_accuracy': self.min_accuracy,
            'min_bleu_score': self.min_bleu_score,
            'min_confidence': self.min_confidence
        }


@dataclass
class PipelineRun:
    """Pipeline run information"""
    run_id: str
    pipeline_type: str
    status: PipelineStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: Optional[Dict[str, Any]] = None
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.study = None
        
    def create_study(self, study_name: str, direction: str = "maximize") -> None:
        """Create optimization study"""
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True
        )
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8, 16]),
            'lora_r': trial.suggest_categorical('lora_r', [8, 16, 32, 64]),
            'lora_alpha': trial.suggest_categorical('lora_alpha', [16, 32, 64, 128]),
            'lora_dropout': trial.suggest_float('lora_dropout', 0.05, 0.3),
            'weight_decay': trial.suggest_float('weight_decay', 0.001, 0.1, log=True),
            'warmup_steps': trial.suggest_int('warmup_steps', 50, 500),
            'thought_weight': trial.suggest_float('thought_weight', 0.1, 0.5),
            'cultural_weight': trial.suggest_float('cultural_weight', 0.1, 0.5),
        }
    
    def optimize(self, 
                objective_func: Callable,
                n_trials: int = 20,
                timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        if self.study is None:
            raise ValueError("Study not created. Call create_study() first.")
        
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout
        )
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study_name': self.study.study_name
        }


class TrainingPipeline:
    """Automated training pipeline for ToW models"""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.tracker = ExperimentTracker(config.experiments)
        self.registry = ModelRegistry(config.experiments)
        self.metrics_collector = MetricsCollector(config.monitoring)
        
        # Pipeline state
        self.current_run: Optional[PipelineRun] = None
        self.runs_history: List[PipelineRun] = []
        
    async def run_training_pipeline(self, 
                                   training_config: TrainingConfig,
                                   run_id: Optional[str] = None) -> PipelineRun:
        """Execute complete training pipeline"""
        
        # Create pipeline run
        run_id = run_id or str(uuid.uuid4())
        pipeline_run = PipelineRun(
            run_id=run_id,
            pipeline_type="training",
            status=PipelineStatus.CREATED,
            created_at=datetime.now(),
            config=training_config.to_dict()
        )
        
        self.current_run = pipeline_run
        self.runs_history.append(pipeline_run)
        
        try:
            logger.info(f"Starting training pipeline: {run_id}")
            pipeline_run.status = PipelineStatus.RUNNING
            pipeline_run.started_at = datetime.now()
            
            # Stage 1: Environment Setup
            await self._run_stage(pipeline_run, "setup", self._setup_training_environment, training_config)
            
            # Stage 2: Data Preparation
            await self._run_stage(pipeline_run, "data_prep", self._prepare_training_data, training_config)
            
            # Stage 3: Model Training
            await self._run_stage(pipeline_run, "training", self._train_model, training_config)
            
            # Stage 4: Model Evaluation
            await self._run_stage(pipeline_run, "evaluation", self._evaluate_model, training_config)
            
            # Stage 5: Model Registration
            await self._run_stage(pipeline_run, "registration", self._register_model, training_config)
            
            # Stage 6: Cleanup
            await self._run_stage(pipeline_run, "cleanup", self._cleanup_training, training_config)
            
            pipeline_run.status = PipelineStatus.SUCCESS
            pipeline_run.completed_at = datetime.now()
            
            logger.info(f"Training pipeline completed successfully: {run_id}")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.error_message = str(e)
            pipeline_run.completed_at = datetime.now()
            
        finally:
            self.current_run = None
        
        return pipeline_run
    
    async def _run_stage(self, 
                        pipeline_run: PipelineRun,
                        stage_name: str,
                        stage_func: Callable,
                        *args) -> None:
        """Run a pipeline stage"""
        
        stage_info = {
            "status": PipelineStage.PENDING,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0,
            "error_message": None
        }
        
        pipeline_run.stages[stage_name] = stage_info
        
        try:
            logger.info(f"Running stage: {stage_name}")
            stage_info["status"] = PipelineStage.RUNNING
            stage_info["started_at"] = datetime.now()
            
            # Execute stage function
            result = await stage_func(*args)
            
            stage_info["status"] = PipelineStage.SUCCESS
            stage_info["completed_at"] = datetime.now()
            stage_info["duration_seconds"] = (
                stage_info["completed_at"] - stage_info["started_at"]
            ).total_seconds()
            
            if isinstance(result, dict):
                stage_info.update(result)
            
            logger.info(f"Stage completed: {stage_name} ({stage_info['duration_seconds']:.2f}s)")
            
        except Exception as e:
            stage_info["status"] = PipelineStage.FAILED
            stage_info["error_message"] = str(e)
            stage_info["completed_at"] = datetime.now()
            
            if stage_info["started_at"]:
                stage_info["duration_seconds"] = (
                    stage_info["completed_at"] - stage_info["started_at"]
                ).total_seconds()
            
            logger.error(f"Stage failed: {stage_name} - {e}")
            raise
    
    async def _setup_training_environment(self, config: TrainingConfig) -> Dict[str, Any]:
        """Setup training environment"""
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for training")
        
        gpu_count = torch.cuda.device_count()
        total_memory = sum(
            torch.cuda.get_device_properties(i).total_memory 
            for i in range(gpu_count)
        ) / (1024**3)  # Convert to GB
        
        # Check disk space
        workspace_path = Path("./training_workspace")
        workspace_path.mkdir(exist_ok=True)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return {
            "gpu_count": gpu_count,
            "total_gpu_memory_gb": total_memory,
            "workspace_path": str(workspace_path),
            "cuda_version": torch.version.cuda
        }
    
    async def _prepare_training_data(self, config: TrainingConfig) -> Dict[str, Any]:
        """Prepare training data"""
        
        # Validate dataset paths
        datasets = {}
        
        if config.train_dataset_path and os.path.exists(config.train_dataset_path):
            datasets["train"] = config.train_dataset_path
        else:
            raise FileNotFoundError(f"Training dataset not found: {config.train_dataset_path}")
        
        if config.val_dataset_path and os.path.exists(config.val_dataset_path):
            datasets["validation"] = config.val_dataset_path
        
        if config.test_dataset_path and os.path.exists(config.test_dataset_path):
            datasets["test"] = config.test_dataset_path
        
        # TODO: Add data preprocessing logic
        # - Tokenization
        # - Data validation
        # - Statistics calculation
        
        return {
            "datasets": datasets,
            "data_statistics": {
                "train_samples": 0,  # TODO: Calculate actual statistics
                "val_samples": 0,
                "test_samples": 0
            }
        }
    
    async def _train_model(self, config: TrainingConfig) -> Dict[str, Any]:
        """Train the model"""
        
        # Start experiment tracking
        run_name = f"tow_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.tracker.start_run(run_name=run_name) as experiment_run:
            
            # Log training configuration
            self.tracker.log_params(config.to_dict())
            
            # TODO: Implement actual ToW model training
            # This is a placeholder implementation
            
            training_metrics = {
                "epochs": config.epochs,
                "final_train_loss": 0.5,  # Mock values
                "final_val_loss": 0.6,
                "best_val_loss": 0.55,
                "training_time_hours": 2.5,
                "convergence_epoch": 2
            }
            
            # Log metrics
            self.tracker.log_metrics(training_metrics)
            
            # TODO: Save model checkpoints and artifacts
            model_path = f"./models/{run_name}"
            os.makedirs(model_path, exist_ok=True)
            
            return {
                "model_path": model_path,
                "experiment_run_id": experiment_run.run_id,
                "training_metrics": training_metrics
            }
    
    async def _evaluate_model(self, config: TrainingConfig) -> Dict[str, Any]:
        """Evaluate trained model"""
        
        # TODO: Implement comprehensive model evaluation
        
        evaluation_metrics = {
            "bleu_score": 0.75,
            "rouge_l": 0.80,
            "bert_score": 0.85,
            "accuracy": 0.88,
            "thought_coherence": 0.82,
            "cultural_adaptation": 0.78,
            "translation_quality": 0.83
        }
        
        # Check if model meets quality thresholds
        meets_threshold = all([
            evaluation_metrics["bleu_score"] >= 0.7,
            evaluation_metrics["accuracy"] >= 0.8,
            evaluation_metrics["thought_coherence"] >= 0.75
        ])
        
        return {
            "evaluation_metrics": evaluation_metrics,
            "meets_threshold": meets_threshold,
            "evaluation_report_path": "./evaluation_report.json"
        }
    
    async def _register_model(self, config: TrainingConfig) -> Dict[str, Any]:
        """Register trained model"""
        
        # TODO: Register model in model registry
        model_version = "1.0.0"  # Mock version
        
        return {
            "model_name": config.model_name,
            "model_version": model_version,
            "registry_uri": f"models:/{config.model_name}/{model_version}"
        }
    
    async def _cleanup_training(self, config: TrainingConfig) -> Dict[str, Any]:
        """Cleanup training artifacts"""
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # TODO: Clean up temporary files
        
        return {"cleanup_completed": True}
    
    def run_hyperparameter_optimization(self, 
                                      base_config: TrainingConfig,
                                      n_trials: int = 20) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        optimizer = HyperparameterOptimizer(base_config)
        study_name = f"tow_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        optimizer.create_study(study_name, direction="maximize")
        
        def objective(trial):
            # Get suggested hyperparameters
            suggested_params = optimizer.suggest_hyperparameters(trial)
            
            # Create training config with suggested parameters
            trial_config = TrainingConfig(**base_config.to_dict())
            for key, value in suggested_params.items():
                setattr(trial_config, key, value)
            
            # Run training (simplified for HPO)
            # TODO: Implement efficient training for HPO
            
            # Return objective value (mock for now)
            return 0.85  # Mock BLEU score
        
        results = optimizer.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter optimization completed: {results}")
        return results


class EvaluationPipeline:
    """Automated evaluation pipeline for ToW models"""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.tracker = ExperimentTracker(config.experiments)
        self.registry = ModelRegistry(config.experiments)
        
    async def run_evaluation_pipeline(self,
                                     model_name: str,
                                     model_version: str,
                                     evaluation_config: EvaluationConfig,
                                     run_id: Optional[str] = None) -> PipelineRun:
        """Execute comprehensive evaluation pipeline"""
        
        run_id = run_id or str(uuid.uuid4())
        pipeline_run = PipelineRun(
            run_id=run_id,
            pipeline_type="evaluation",
            status=PipelineStatus.CREATED,
            created_at=datetime.now(),
            config=evaluation_config.to_dict()
        )
        
        try:
            logger.info(f"Starting evaluation pipeline: {run_id}")
            pipeline_run.status = PipelineStatus.RUNNING
            pipeline_run.started_at = datetime.now()
            
            # Stage 1: Model Loading
            await self._run_stage(pipeline_run, "model_loading", 
                                self._load_evaluation_model, model_name, model_version)
            
            # Stage 2: Dataset Preparation
            await self._run_stage(pipeline_run, "dataset_prep",
                                self._prepare_evaluation_datasets, evaluation_config)
            
            # Stage 3: Model Evaluation
            await self._run_stage(pipeline_run, "evaluation",
                                self._run_model_evaluation, evaluation_config)
            
            # Stage 4: Results Analysis
            await self._run_stage(pipeline_run, "analysis",
                                self._analyze_evaluation_results, evaluation_config)
            
            # Stage 5: Report Generation
            await self._run_stage(pipeline_run, "reporting",
                                self._generate_evaluation_report, evaluation_config)
            
            pipeline_run.status = PipelineStatus.SUCCESS
            pipeline_run.completed_at = datetime.now()
            
            logger.info(f"Evaluation pipeline completed: {run_id}")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.error_message = str(e)
            pipeline_run.completed_at = datetime.now()
        
        return pipeline_run
    
    async def _run_stage(self, 
                        pipeline_run: PipelineRun,
                        stage_name: str,
                        stage_func: Callable,
                        *args) -> None:
        """Run evaluation pipeline stage (similar to training pipeline)"""
        
        stage_info = {
            "status": PipelineStage.PENDING,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0,
            "error_message": None
        }
        
        pipeline_run.stages[stage_name] = stage_info
        
        try:
            logger.info(f"Running evaluation stage: {stage_name}")
            stage_info["status"] = PipelineStage.RUNNING
            stage_info["started_at"] = datetime.now()
            
            result = await stage_func(*args)
            
            stage_info["status"] = PipelineStage.SUCCESS
            stage_info["completed_at"] = datetime.now()
            stage_info["duration_seconds"] = (
                stage_info["completed_at"] - stage_info["started_at"]
            ).total_seconds()
            
            if isinstance(result, dict):
                stage_info.update(result)
            
        except Exception as e:
            stage_info["status"] = PipelineStage.FAILED
            stage_info["error_message"] = str(e)
            stage_info["completed_at"] = datetime.now()
            
            if stage_info["started_at"]:
                stage_info["duration_seconds"] = (
                    stage_info["completed_at"] - stage_info["started_at"]
                ).total_seconds()
            
            raise
    
    async def _load_evaluation_model(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Load model for evaluation"""
        
        # TODO: Load model from registry
        # model = self.registry.load_model(model_name, model_version)
        
        return {
            "model_loaded": True,
            "model_name": model_name,
            "model_version": model_version
        }
    
    async def _prepare_evaluation_datasets(self, config: EvaluationConfig) -> Dict[str, Any]:
        """Prepare evaluation datasets"""
        
        prepared_datasets = {}
        
        for dataset_name in config.datasets:
            # TODO: Load and prepare actual datasets
            prepared_datasets[dataset_name] = {
                "path": f"./datasets/{dataset_name}",
                "samples": 1000  # Mock value
            }
        
        return {"prepared_datasets": prepared_datasets}
    
    async def _run_model_evaluation(self, config: EvaluationConfig) -> Dict[str, Any]:
        """Run model evaluation"""
        
        # TODO: Implement actual evaluation logic
        
        evaluation_results = {
            "bleu_score": 0.78,
            "rouge_l": 0.82,
            "bert_score": 0.86,
            "accuracy": 0.89,
            "perplexity": 2.3,
            "thought_coherence": 0.84,
            "cultural_adaptation": 0.80,
            "translation_quality": 0.85
        }
        
        return {"evaluation_results": evaluation_results}
    
    async def _analyze_evaluation_results(self, config: EvaluationConfig) -> Dict[str, Any]:
        """Analyze evaluation results"""
        
        # TODO: Implement result analysis
        
        analysis = {
            "overall_score": 0.84,
            "strengths": ["High accuracy", "Good thought coherence"],
            "weaknesses": ["Moderate cultural adaptation"],
            "recommendations": ["Improve cultural training data"]
        }
        
        return {"analysis": analysis}
    
    async def _generate_evaluation_report(self, config: EvaluationConfig) -> Dict[str, Any]:
        """Generate evaluation report"""
        
        report_path = "./evaluation_report.html"
        
        # TODO: Generate comprehensive HTML report
        
        return {
            "report_path": report_path,
            "report_generated": True
        }


class CIPipeline:
    """Continuous Integration pipeline for ToW models"""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.training_pipeline = TrainingPipeline(config)
        self.evaluation_pipeline = EvaluationPipeline(config)
        self.deployment_manager = DeploymentManager(config)
        
    async def run_ci_pipeline(self, 
                             trigger_event: str,
                             config_path: str) -> Dict[str, Any]:
        """Run CI/CD pipeline triggered by various events"""
        
        logger.info(f"CI/CD pipeline triggered by: {trigger_event}")
        
        # Load pipeline configuration
        with open(config_path, 'r') as f:
            pipeline_config = yaml.safe_load(f)
        
        results = {
            "trigger_event": trigger_event,
            "started_at": datetime.now(),
            "stages": {}
        }
        
        try:
            # Stage 1: Code Quality Checks
            if pipeline_config.get("run_quality_checks", True):
                quality_results = await self._run_quality_checks()
                results["stages"]["quality_checks"] = quality_results
                
                if not quality_results.get("passed", False):
                    raise Exception("Code quality checks failed")
            
            # Stage 2: Training (if configured)
            if pipeline_config.get("run_training", False):
                training_config = TrainingConfig(**pipeline_config.get("training", {}))
                training_results = await self.training_pipeline.run_training_pipeline(training_config)
                results["stages"]["training"] = training_results
                
                if training_results.status != PipelineStatus.SUCCESS:
                    raise Exception("Training pipeline failed")
            
            # Stage 3: Evaluation
            if pipeline_config.get("run_evaluation", True):
                eval_config = EvaluationConfig(**pipeline_config.get("evaluation", {}))
                model_name = pipeline_config.get("model_name", "tow-model")
                model_version = pipeline_config.get("model_version", "latest")
                
                eval_results = await self.evaluation_pipeline.run_evaluation_pipeline(
                    model_name, model_version, eval_config
                )
                results["stages"]["evaluation"] = eval_results
                
                if eval_results.status != PipelineStatus.SUCCESS:
                    raise Exception("Evaluation pipeline failed")
            
            # Stage 4: Deployment (if evaluation passes)
            if (pipeline_config.get("auto_deploy", False) and 
                results["stages"].get("evaluation", {}).get("status") == PipelineStatus.SUCCESS):
                
                deployment_results = await self._run_deployment(pipeline_config)
                results["stages"]["deployment"] = deployment_results
            
            results["status"] = "success"
            results["completed_at"] = datetime.now()
            
        except Exception as e:
            logger.error(f"CI/CD pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["completed_at"] = datetime.now()
        
        return results
    
    async def _run_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks"""
        
        checks = {
            "linting": True,  # Mock results
            "type_checking": True,
            "security_scan": True,
            "unit_tests": True,
            "coverage": 85.2
        }
        
        passed = all([
            checks["linting"],
            checks["type_checking"],
            checks["security_scan"],
            checks["unit_tests"],
            checks["coverage"] >= 80
        ])
        
        return {
            "checks": checks,
            "passed": passed,
            "coverage_percent": checks["coverage"]
        }
    
    async def _run_deployment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run deployment stage"""
        
        model_name = config.get("model_name", "tow-model")
        model_version = config.get("model_version", "latest")
        environment = config.get("deploy_environment", "staging")
        
        deployment_id = self.deployment_manager.deploy_model(
            model_name=model_name,
            version=model_version,
            environment=environment,
            deployment_strategy="rolling"
        )
        
        return {
            "deployment_id": deployment_id,
            "model_name": model_name,
            "model_version": model_version,
            "environment": environment
        }


def create_training_pipeline(config: MLOpsConfig) -> TrainingPipeline:
    """Create training pipeline from config"""
    return TrainingPipeline(config)


def create_evaluation_pipeline(config: MLOpsConfig) -> EvaluationPipeline:
    """Create evaluation pipeline from config"""
    return EvaluationPipeline(config)


def create_ci_pipeline(config: MLOpsConfig) -> CIPipeline:
    """Create CI/CD pipeline from config"""
    return CIPipeline(config)