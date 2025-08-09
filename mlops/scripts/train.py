#!/usr/bin/env python3
"""
ToW Training Script
==================

Comprehensive training script for ToW models with support for:
- Single-node and distributed training
- Hyperparameter optimization
- Experiment tracking
- Evaluation and benchmarking
- Model registry integration
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
import torch
import wandb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mlops.config import MLOpsConfig, load_config
from mlops.pipeline import TrainingPipeline, EvaluationPipeline, TrainingConfig, EvaluationConfig
from mlops.tracking import ExperimentTracker, ModelRegistry


logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates ToW model training workflows"""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.training_pipeline = TrainingPipeline(config)
        self.evaluation_pipeline = EvaluationPipeline(config)
        self.tracker = ExperimentTracker(config.experiments, config.project_name)
        self.registry = ModelRegistry(config.experiments)
    
    async def train_model(self, 
                         training_config: TrainingConfig,
                         run_evaluation: bool = True,
                         register_model: bool = True) -> Dict[str, Any]:
        """Train a ToW model with full pipeline"""
        
        logger.info(f"Starting training for model: {training_config.model_name}")
        
        training_result = {
            "model_name": training_config.model_name,
            "started_at": datetime.now(),
            "status": "started",
            "steps": {}
        }
        
        try:
            # Step 1: Run training pipeline
            logger.info("Running training pipeline...")
            pipeline_result = await self.training_pipeline.run_training_pipeline(training_config)
            
            training_result["steps"]["training"] = {
                "status": pipeline_result.status.value,
                "run_id": pipeline_result.run_id,
                "metrics": pipeline_result.metrics,
                "artifacts": pipeline_result.artifacts
            }
            
            if pipeline_result.status.value != "success":
                raise Exception(f"Training pipeline failed: {pipeline_result.error_message}")
            
            # Step 2: Run evaluation (if requested)
            if run_evaluation:
                logger.info("Running evaluation pipeline...")
                
                eval_config = EvaluationConfig(
                    datasets=["mmlu", "klue", "custom"],
                    metrics=["bleu", "rouge", "bertscore", "accuracy"],
                    batch_size=8
                )
                
                eval_result = await self.evaluation_pipeline.run_evaluation_pipeline(
                    training_config.model_name,
                    "latest",
                    eval_config
                )
                
                training_result["steps"]["evaluation"] = {
                    "status": eval_result.status.value,
                    "run_id": eval_result.run_id,
                    "metrics": eval_result.metrics
                }
                
                # Check if model meets quality thresholds
                if eval_result.status.value == "success":
                    evaluation_metrics = eval_result.metrics.get("evaluation_results", {})
                    meets_threshold = self._check_quality_thresholds(evaluation_metrics, eval_config)
                    training_result["meets_quality_threshold"] = meets_threshold
                    
                    if not meets_threshold:
                        logger.warning("Model does not meet quality thresholds")
            
            # Step 3: Register model (if requested and meets thresholds)
            should_register = (
                register_model and 
                training_result.get("meets_quality_threshold", True)  # Default to True if no evaluation
            )
            
            if should_register:
                logger.info("Registering model...")
                
                # Get model URI from training artifacts
                model_uri = training_result["steps"]["training"]["artifacts"][0] if training_result["steps"]["training"]["artifacts"] else None
                
                if model_uri:
                    model_version = self.registry.register_model(
                        model_uri=model_uri,
                        model_name=training_config.model_name,
                        description=f"ToW model trained on {datetime.now().strftime('%Y-%m-%d')}",
                        tags={
                            "training_config": training_config.to_dict(),
                            "architecture": "tow",
                            "base_model": training_config.base_model_path
                        }
                    )
                    
                    training_result["steps"]["registration"] = {
                        "status": "success",
                        "model_version": model_version,
                        "model_uri": model_uri
                    }
                else:
                    logger.warning("No model URI found, skipping registration")
            
            training_result["status"] = "completed"
            training_result["completed_at"] = datetime.now()
            
            logger.info(f"Training completed successfully for {training_config.model_name}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            training_result["status"] = "failed"
            training_result["error"] = str(e)
            training_result["completed_at"] = datetime.now()
        
        return training_result
    
    def _check_quality_thresholds(self, metrics: Dict[str, float], eval_config: EvaluationConfig) -> bool:
        """Check if model meets quality thresholds"""
        
        thresholds = {
            "accuracy": eval_config.min_accuracy,
            "bleu_score": eval_config.min_bleu_score,
            "confidence_score": eval_config.min_confidence
        }
        
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                if metrics[metric_name] < threshold:
                    logger.warning(f"{metric_name} ({metrics[metric_name]:.3f}) below threshold ({threshold})")
                    return False
        
        return True
    
    async def optimize_hyperparameters(self, 
                                     base_config: TrainingConfig,
                                     n_trials: int = 20,
                                     study_name: Optional[str] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        if study_name is None:
            study_name = f"tow_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting hyperparameter optimization: {study_name}")
        
        # Run optimization
        optimization_result = self.training_pipeline.run_hyperparameter_optimization(
            base_config, n_trials
        )
        
        logger.info(f"Hyperparameter optimization completed: {optimization_result}")
        
        return {
            "study_name": study_name,
            "n_trials": n_trials,
            "best_params": optimization_result["best_params"],
            "best_score": optimization_result["best_value"],
            "completed_at": datetime.now()
        }
    
    async def benchmark_model(self, 
                            model_name: str,
                            model_version: str = "latest") -> Dict[str, Any]:
        """Run comprehensive model benchmarking"""
        
        logger.info(f"Benchmarking model: {model_name} v{model_version}")
        
        # Comprehensive evaluation configuration
        eval_config = EvaluationConfig(
            datasets=["mmlu", "klue", "hellaswag", "arc", "truthfulqa"],
            metrics=["bleu", "rouge", "bertscore", "accuracy", "perplexity"],
            tow_metrics=["thought_coherence", "cultural_adaptation", "translation_quality"],
            batch_size=4,  # Smaller batch size for comprehensive evaluation
            max_samples=1000  # Limit samples for faster benchmarking
        )
        
        # Run evaluation pipeline
        eval_result = await self.evaluation_pipeline.run_evaluation_pipeline(
            model_name, model_version, eval_config
        )
        
        return {
            "model_name": model_name,
            "model_version": model_version,
            "benchmark_results": eval_result.metrics,
            "evaluation_run_id": eval_result.run_id,
            "benchmarked_at": datetime.now()
        }
    
    def list_experiments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent experiments"""
        
        runs_df = self.tracker.get_experiment_runs()
        
        if runs_df.empty:
            return []
        
        # Sort by start time and limit
        recent_runs = runs_df.head(limit)
        
        experiments = []
        for _, run in recent_runs.iterrows():
            experiments.append({
                "run_id": run.get("run_id", ""),
                "experiment_id": run.get("experiment_id", ""),
                "status": run.get("status", ""),
                "start_time": run.get("start_time", ""),
                "metrics": {
                    col.replace("metrics.", ""): run[col] 
                    for col in run.index if col.startswith("metrics.")
                }
            })
        
        return experiments
    
    def compare_models(self, model_versions: List[str]) -> Dict[str, Any]:
        """Compare multiple model versions"""
        
        if len(model_versions) < 2:
            raise ValueError("At least 2 model versions required for comparison")
        
        # Get run IDs for model versions
        run_ids = []
        for version in model_versions:
            # This would need to be implemented based on how versions map to runs
            # For now, using placeholder logic
            run_ids.append(f"run_id_for_{version}")
        
        # Compare runs
        comparison_df = self.tracker.compare_runs(run_ids)
        
        return {
            "model_versions": model_versions,
            "comparison_data": comparison_df.to_dict("records"),
            "compared_at": datetime.now()
        }


def create_training_config_from_args(args) -> TrainingConfig:
    """Create training configuration from command line arguments"""
    
    config = TrainingConfig()
    
    # Update config with provided arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.base_model:
        config.base_model_path = args.base_model
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.train_data:
        config.train_dataset_path = args.train_data
    if args.val_data:
        config.val_dataset_path = args.val_data
    if args.test_data:
        config.test_dataset_path = args.test_data
    
    return config


def main():
    """Main training script"""
    
    parser = argparse.ArgumentParser(
        description="ToW Model Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py train --model-name tow-llama-7b --train-data /path/to/train.jsonl
  
  # Training with evaluation
  python train.py train --model-name tow-llama-7b --train-data /path/to/train.jsonl --evaluate
  
  # Hyperparameter optimization
  python train.py optimize --model-name tow-llama-7b --trials 50
  
  # Model benchmarking
  python train.py benchmark --model-name tow-llama-7b --version v1.0.0
  
  # List experiments
  python train.py list --limit 10
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model-name", required=True, help="Model name")
    train_parser.add_argument("--base-model", help="Base model path")
    train_parser.add_argument("--train-data", required=True, help="Training data path")
    train_parser.add_argument("--val-data", help="Validation data path")
    train_parser.add_argument("--test-data", help="Test data path")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    train_parser.add_argument("--register", action="store_true", help="Register model after training")
    train_parser.add_argument("--config", help="Training configuration file")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize hyperparameters")
    optimize_parser.add_argument("--model-name", required=True, help="Model name")
    optimize_parser.add_argument("--base-model", help="Base model path")
    optimize_parser.add_argument("--train-data", required=True, help="Training data path")
    optimize_parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials")
    optimize_parser.add_argument("--study-name", help="Optimization study name")
    optimize_parser.add_argument("--config", help="Base configuration file")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model")
    benchmark_parser.add_argument("--model-name", required=True, help="Model name")
    benchmark_parser.add_argument("--version", default="latest", help="Model version")
    benchmark_parser.add_argument("--config", help="MLOps configuration file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of experiments to show")
    list_parser.add_argument("--config", help="MLOps configuration file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare model versions")
    compare_parser.add_argument("--versions", nargs="+", required=True, help="Model versions to compare")
    compare_parser.add_argument("--config", help="MLOps configuration file")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    if hasattr(args, 'config') and args.config:
        config = MLOpsConfig.from_file(args.config)
    else:
        config = load_config()
    
    logger.info(f"Using configuration environment: {config.environment.value}")
    
    # Initialize W&B if configured
    if config.experiments.wandb_api_key:
        wandb.login(key=config.experiments.wandb_api_key)
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    # Execute command
    async def run_command():
        if args.command == "train":
            # Create training config
            if hasattr(args, 'config') and args.config and Path(args.config).exists():
                with open(args.config, 'r') as f:
                    config_dict = yaml.safe_load(f)
                training_config = TrainingConfig(**config_dict)
            else:
                training_config = create_training_config_from_args(args)
            
            # Run training
            result = await orchestrator.train_model(
                training_config=training_config,
                run_evaluation=args.evaluate,
                register_model=args.register
            )
            
            print(json.dumps(result, indent=2, default=str))
            
        elif args.command == "optimize":
            # Create base training config
            training_config = create_training_config_from_args(args)
            
            # Run optimization
            result = await orchestrator.optimize_hyperparameters(
                base_config=training_config,
                n_trials=args.trials,
                study_name=args.study_name
            )
            
            print(json.dumps(result, indent=2, default=str))
            
        elif args.command == "benchmark":
            result = await orchestrator.benchmark_model(
                model_name=args.model_name,
                model_version=args.version
            )
            
            print(json.dumps(result, indent=2, default=str))
            
        elif args.command == "list":
            experiments = orchestrator.list_experiments(limit=args.limit)
            print(json.dumps(experiments, indent=2, default=str))
            
        elif args.command == "compare":
            result = orchestrator.compare_models(args.versions)
            print(json.dumps(result, indent=2, default=str))
    
    # Run async command
    try:
        asyncio.run(run_command())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()