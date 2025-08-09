"""
Model Deployment and Serving Infrastructure
==========================================

Provides comprehensive model deployment, serving, and management capabilities
for ToW-enhanced language models with A/B testing, load balancing, and auto-scaling.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .config import MLOpsConfig, ModelServerConfig
from .monitoring import MetricsCollector
from .tracking import ModelRegistry


logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    TERMINATED = "terminated"


class ModelVersion(BaseModel):
    """Model version information"""
    model_id: str
    version: str
    stage: str = "staging"
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    """Request model for inference"""
    text: str = Field(..., description="Input text to process")
    source_language: str = Field(default="en", description="Source language code")
    target_language: str = Field(default="ko", description="Target language code")
    task_type: str = Field(default="translation", description="Task type (translation, reasoning, generation)")
    
    # Generation parameters
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    
    # ToW-specific parameters
    max_thought_tokens: int = Field(default=3, ge=1, le=10)
    enable_cultural_adaptation: bool = Field(default=True)
    thought_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    
    # Request metadata
    request_id: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    
    @validator('request_id', pre=True, always=True)
    def generate_request_id(cls, v):
        return v or str(uuid.uuid4())


class InferenceResponse(BaseModel):
    """Response model for inference"""
    request_id: str
    output_text: str
    thought_tokens: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Performance metrics
    processing_time_ms: float
    queue_time_ms: float = 0.0
    model_inference_time_ms: float = 0.0
    
    # Quality metrics
    validation_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    model_loaded: bool
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    queue_size: int = 0
    active_requests: int = 0
    uptime_seconds: float = 0.0


class ModelLoadBalancer:
    """Load balancer for multiple model instances"""
    
    def __init__(self, config: ModelServerConfig):
        self.config = config
        self.instances: List[Dict[str, Any]] = []
        self.current_instance = 0
        self._lock = asyncio.Lock()
    
    def add_instance(self, instance_id: str, model, weight: float = 1.0) -> None:
        """Add a model instance to the load balancer"""
        instance = {
            "id": instance_id,
            "model": model,
            "weight": weight,
            "active_requests": 0,
            "total_requests": 0,
            "last_request_time": time.time(),
            "healthy": True
        }
        self.instances.append(instance)
        logger.info(f"Added model instance: {instance_id}")
    
    def remove_instance(self, instance_id: str) -> None:
        """Remove a model instance"""
        self.instances = [inst for inst in self.instances if inst["id"] != instance_id]
        logger.info(f"Removed model instance: {instance_id}")
    
    async def get_instance(self, strategy: str = "round_robin"):
        """Get next available instance based on load balancing strategy"""
        async with self._lock:
            if not self.instances:
                raise HTTPException(status_code=503, detail="No healthy model instances available")
            
            healthy_instances = [inst for inst in self.instances if inst["healthy"]]
            if not healthy_instances:
                raise HTTPException(status_code=503, detail="No healthy model instances available")
            
            if strategy == "round_robin":
                instance = healthy_instances[self.current_instance % len(healthy_instances)]
                self.current_instance = (self.current_instance + 1) % len(healthy_instances)
            
            elif strategy == "least_connections":
                instance = min(healthy_instances, key=lambda x: x["active_requests"])
            
            elif strategy == "weighted_round_robin":
                # Simple weighted selection (can be improved)
                total_weight = sum(inst["weight"] for inst in healthy_instances)
                target = (self.current_instance % total_weight)
                cumulative_weight = 0
                
                for inst in healthy_instances:
                    cumulative_weight += inst["weight"]
                    if target < cumulative_weight:
                        instance = inst
                        break
                else:
                    instance = healthy_instances[0]
                
                self.current_instance += 1
            
            else:
                instance = healthy_instances[0]  # Default to first healthy instance
            
            instance["active_requests"] += 1
            instance["total_requests"] += 1
            instance["last_request_time"] = time.time()
            
            return instance
    
    async def release_instance(self, instance_id: str) -> None:
        """Release an instance after request completion"""
        async with self._lock:
            for instance in self.instances:
                if instance["id"] == instance_id:
                    instance["active_requests"] = max(0, instance["active_requests"] - 1)
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "total_instances": len(self.instances),
            "healthy_instances": len([inst for inst in self.instances if inst["healthy"]]),
            "instances": [
                {
                    "id": inst["id"],
                    "weight": inst["weight"],
                    "active_requests": inst["active_requests"],
                    "total_requests": inst["total_requests"],
                    "healthy": inst["healthy"]
                }
                for inst in self.instances
            ]
        }


class ABTestManager:
    """A/B testing manager for model deployments"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.experiments: Dict[str, Dict[str, Any]] = {}
    
    def create_experiment(self, 
                         experiment_id: str,
                         control_model: str,
                         treatment_model: str,
                         traffic_split: float = 0.1,
                         duration_hours: int = 24) -> None:
        """Create a new A/B test experiment"""
        experiment = {
            "id": experiment_id,
            "control_model": control_model,
            "treatment_model": treatment_model,
            "traffic_split": traffic_split,  # Percentage of traffic to treatment
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(hours=duration_hours),
            "active": True,
            "control_requests": 0,
            "treatment_requests": 0
        }
        
        self.experiments[experiment_id] = experiment
        
        # Store in Redis for persistence
        self.redis_client.hset("ab_experiments", experiment_id, json.dumps(experiment, default=str))
        
        logger.info(f"Created A/B test experiment: {experiment_id}")
    
    def get_model_for_request(self, request_id: str, user_id: Optional[str] = None) -> Tuple[str, str]:
        """Determine which model to use for a request"""
        # Find active experiments
        active_experiments = [exp for exp in self.experiments.values() if exp["active"]]
        
        if not active_experiments:
            return "default", "control"
        
        # Use first active experiment (could be extended for multiple experiments)
        experiment = active_experiments[0]
        
        # Check if experiment has expired
        if datetime.now() > experiment["end_time"]:
            experiment["active"] = False
            return experiment["control_model"], "control"
        
        # Use user_id for consistent assignment, fallback to request_id
        hash_key = user_id if user_id else request_id
        hash_value = hash(hash_key) % 100
        
        if hash_value < experiment["traffic_split"] * 100:
            experiment["treatment_requests"] += 1
            return experiment["treatment_model"], "treatment"
        else:
            experiment["control_requests"] += 1
            return experiment["control_model"], "control"
    
    def record_result(self, 
                     experiment_id: str,
                     group: str,
                     metric_name: str,
                     metric_value: float) -> None:
        """Record experiment result"""
        key = f"ab_results:{experiment_id}:{group}:{metric_name}"
        self.redis_client.lpush(key, metric_value)
        self.redis_client.expire(key, 7 * 24 * 3600)  # Keep for 7 days
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results and statistics"""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        results = {
            "experiment_id": experiment_id,
            "control_model": experiment["control_model"],
            "treatment_model": experiment["treatment_model"],
            "control_requests": experiment["control_requests"],
            "treatment_requests": experiment["treatment_requests"],
            "metrics": {}
        }
        
        # Get metrics from Redis
        for group in ["control", "treatment"]:
            for metric in ["confidence_score", "processing_time_ms", "bleu_score"]:
                key = f"ab_results:{experiment_id}:{group}:{metric}"
                values = [float(v) for v in self.redis_client.lrange(key, 0, -1)]
                
                if values:
                    results["metrics"][f"{group}_{metric}"] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
        
        return results


class ToWModelServer:
    """ToW model serving infrastructure with load balancing and A/B testing"""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.model_config = config.model_server
        self.load_balancer = ModelLoadBalancer(self.model_config)
        self.ab_test_manager = ABTestManager()
        self.metrics_collector = MetricsCollector(config.monitoring)
        self.registry = ModelRegistry(config.experiments)
        
        # Metrics
        self.request_counter = Counter('tow_requests_total', 'Total requests', ['model_version', 'task_type'])
        self.request_duration = Histogram('tow_request_duration_seconds', 'Request duration')
        self.active_requests = Gauge('tow_active_requests', 'Active requests')
        self.model_load_time = Histogram('tow_model_load_seconds', 'Model loading time')
        
        # Cache
        self.cache = None
        if self.model_config.enable_caching:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=1)
                self.cache.ping()  # Test connection
            except:
                logger.warning("Redis not available, caching disabled")
                self.cache = None
        
        # Initialize FastAPI app
        self.app = self._create_app()
        
        # Background tasks
        self.background_tasks = set()
        
        # Server state
        self.start_time = time.time()
        self.is_ready = False
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting ToW Model Server...")
            await self._startup()
            yield
            # Shutdown
            logger.info("Shutting down ToW Model Server...")
            await self._shutdown()
        
        app = FastAPI(
            title="ToW Model Server",
            description="Thoughts of Words enhanced language model serving API",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Middleware
        if self.config.security.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.security.allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
        
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes"""
        
        @app.get("/health", response_model=HealthStatus)
        async def health_check():
            """Health check endpoint"""
            return HealthStatus(
                status="healthy" if self.is_ready else "starting",
                version="1.0.0",
                model_loaded=len(self.load_balancer.instances) > 0,
                memory_usage_mb=self.metrics_collector.get_memory_usage(),
                gpu_memory_mb=self.metrics_collector.get_gpu_memory_usage(),
                queue_size=0,  # TODO: Implement queue monitoring
                active_requests=int(self.active_requests._value.get()),
                uptime_seconds=time.time() - self.start_time
            )
        
        @app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint"""
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        
        @app.post("/predict", response_model=InferenceResponse)
        async def predict(
            request: InferenceRequest,
            background_tasks: BackgroundTasks,
            auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
        ):
            """Main prediction endpoint"""
            start_time = time.time()
            
            # Authentication (if enabled)
            if self.config.security.enable_auth and not auth:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Rate limiting (basic implementation)
            if self.config.security.enable_rate_limiting:
                # TODO: Implement proper rate limiting
                pass
            
            # Input validation
            if len(request.text) > self.config.security.max_input_length:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Input text too long (max {self.config.security.max_input_length} characters)"
                )
            
            self.active_requests.inc()
            
            try:
                # A/B testing - determine which model to use
                model_id, group = self.ab_test_manager.get_model_for_request(
                    request.request_id, request.user_id
                )
                
                # Get model instance
                instance = await self.load_balancer.get_instance()
                
                # Check cache
                cache_key = None
                if self.cache and self.model_config.enable_caching:
                    cache_key = self._generate_cache_key(request)
                    cached_response = self._get_cached_response(cache_key)
                    if cached_response:
                        logger.info(f"Cache hit for request {request.request_id}")
                        return cached_response
                
                # Process request
                queue_time = time.time() - start_time
                inference_start = time.time()
                
                # TODO: Replace with actual ToW model inference
                result = await self._process_with_tow_model(instance["model"], request)
                
                inference_time = time.time() - inference_start
                total_time = time.time() - start_time
                
                # Create response
                response = InferenceResponse(
                    request_id=request.request_id,
                    output_text=result.get("output_text", ""),
                    thought_tokens=result.get("thought_tokens", []),
                    confidence_score=result.get("confidence_score", 0.0),
                    processing_time_ms=total_time * 1000,
                    queue_time_ms=queue_time * 1000,
                    model_inference_time_ms=inference_time * 1000,
                    validation_scores=result.get("validation_scores", {}),
                    model_version=instance["id"]
                )
                
                # Cache response
                if cache_key and self.cache:
                    self._cache_response(cache_key, response)
                
                # Record metrics
                self.request_counter.labels(
                    model_version=instance["id"], 
                    task_type=request.task_type
                ).inc()
                
                self.request_duration.observe(total_time)
                
                # Record A/B test results
                if group in ["control", "treatment"]:
                    background_tasks.add_task(
                        self._record_ab_test_results,
                        group,
                        request,
                        response
                    )
                
                return response
                
            finally:
                self.active_requests.dec()
                if 'instance' in locals():
                    await self.load_balancer.release_instance(instance["id"])
        
        @app.get("/models")
        async def list_models():
            """List available models"""
            models = self.registry.list_models()
            load_balancer_stats = self.load_balancer.get_stats()
            
            return {
                "registered_models": models,
                "active_instances": load_balancer_stats
            }
        
        @app.post("/models/deploy")
        async def deploy_model(
            model_name: str,
            version: str = "latest",
            stage: str = "staging",
            weight: float = 1.0
        ):
            """Deploy a new model version"""
            try:
                # Load model from registry
                model = self.registry.load_model(model_name, version, stage)
                
                # Create instance ID
                instance_id = f"{model_name}_{version}_{int(time.time())}"
                
                # Add to load balancer
                self.load_balancer.add_instance(instance_id, model, weight)
                
                return {
                    "status": "success",
                    "instance_id": instance_id,
                    "message": f"Deployed {model_name} v{version}"
                }
                
            except Exception as e:
                logger.error(f"Failed to deploy model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.delete("/models/{instance_id}")
        async def undeploy_model(instance_id: str):
            """Remove a model instance"""
            self.load_balancer.remove_instance(instance_id)
            return {"status": "success", "message": f"Removed instance {instance_id}"}
        
        @app.post("/experiments")
        async def create_ab_experiment(
            experiment_id: str,
            control_model: str,
            treatment_model: str,
            traffic_split: float = 0.1,
            duration_hours: int = 24
        ):
            """Create A/B test experiment"""
            self.ab_test_manager.create_experiment(
                experiment_id, control_model, treatment_model, traffic_split, duration_hours
            )
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "message": f"Created A/B test with {traffic_split*100}% traffic split"
            }
        
        @app.get("/experiments/{experiment_id}")
        async def get_experiment_results(experiment_id: str):
            """Get A/B test results"""
            return self.ab_test_manager.get_experiment_results(experiment_id)
    
    async def _startup(self):
        """Server startup tasks"""
        logger.info("Loading default models...")
        
        # Load default model (if configured)
        # TODO: Load from model registry
        
        self.is_ready = True
        logger.info("ToW Model Server is ready!")
    
    async def _shutdown(self):
        """Server shutdown tasks"""
        logger.info("Performing cleanup...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Clear model instances
        self.load_balancer.instances.clear()
    
    async def _process_with_tow_model(self, model, request: InferenceRequest) -> Dict[str, Any]:
        """Process request with ToW model (placeholder implementation)"""
        # TODO: Integrate with actual ToW architecture
        
        # Simulate processing
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Mock response
        return {
            "output_text": f"[ToW Response] {request.text} -> {request.target_language}",
            "thought_tokens": ["thinking about translation", "considering context", "finalizing output"],
            "confidence_score": 0.85,
            "validation_scores": {
                "language_consistency": 0.90,
                "thought_alignment": 0.88,
                "cultural_adaptation": 0.82
            }
        }
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "text": request.text,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "task_type": request.task_type,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"tow_cache:{hash(key_str)}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[InferenceResponse]:
        """Get cached response"""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return InferenceResponse(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: InferenceResponse) -> None:
        """Cache response"""
        if not self.cache:
            return
        
        try:
            # Convert response to dict for JSON serialization
            data = response.dict()
            data['timestamp'] = data['timestamp'].isoformat()
            
            self.cache.setex(
                cache_key, 
                3600,  # 1 hour expiry
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    async def _record_ab_test_results(self, 
                                     group: str,
                                     request: InferenceRequest,
                                     response: InferenceResponse) -> None:
        """Record A/B test results in background"""
        try:
            # Find active experiment
            active_experiments = [exp for exp in self.ab_test_manager.experiments.values() if exp["active"]]
            if not active_experiments:
                return
            
            experiment_id = active_experiments[0]["id"]
            
            # Record metrics
            self.ab_test_manager.record_result(
                experiment_id, group, "confidence_score", response.confidence_score
            )
            self.ab_test_manager.record_result(
                experiment_id, group, "processing_time_ms", response.processing_time_ms
            )
            
            # Record validation scores
            for metric, score in response.validation_scores.items():
                self.ab_test_manager.record_result(
                    experiment_id, group, metric, score
                )
        
        except Exception as e:
            logger.error(f"Failed to record A/B test results: {e}")
    
    def run(self, host: str = None, port: int = None):
        """Run the server"""
        host = host or self.model_config.host
        port = port or self.model_config.port
        
        logger.info(f"Starting ToW Model Server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=1,  # Use 1 worker for now due to model loading
            access_log=True,
            log_level="info"
        )


class DeploymentManager:
    """
    Manages model deployments across different environments and stages.
    Supports blue-green deployments, canary releases, and rollbacks.
    """
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.registry = ModelRegistry(config.experiments)
        self.deployments: Dict[str, Dict[str, Any]] = {}
        
        # Redis for deployment state
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=2)
            self.redis_client.ping()
        except:
            logger.warning("Redis not available, using in-memory deployment state")
            self.redis_client = None
    
    def deploy_model(self, 
                    model_name: str,
                    version: str,
                    environment: str = "staging",
                    deployment_strategy: str = "rolling",
                    health_check_url: str = "/health") -> str:
        """Deploy a model to specified environment"""
        
        deployment_id = f"{model_name}_{version}_{environment}_{int(time.time())}"
        
        deployment = {
            "id": deployment_id,
            "model_name": model_name,
            "version": version,
            "environment": environment,
            "strategy": deployment_strategy,
            "status": DeploymentStatus.DEPLOYING.value,
            "created_at": datetime.now(),
            "health_check_url": health_check_url,
            "instances": [],
            "metrics": {}
        }
        
        self.deployments[deployment_id] = deployment
        
        # Store in Redis
        if self.redis_client:
            self.redis_client.hset("deployments", deployment_id, json.dumps(deployment, default=str))
        
        logger.info(f"Started deployment: {deployment_id}")
        
        # TODO: Implement actual deployment logic based on strategy
        # For now, mark as healthy
        deployment["status"] = DeploymentStatus.HEALTHY.value
        
        return deployment_id
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment"""
        if deployment_id not in self.deployments:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        deployment = self.deployments[deployment_id]
        
        # TODO: Implement rollback logic
        deployment["status"] = DeploymentStatus.TERMINATED.value
        
        logger.info(f"Rolled back deployment: {deployment_id}")
        return True
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """List deployments, optionally filtered by environment"""
        deployments = list(self.deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d["environment"] == environment]
        
        return deployments
    
    def promote_model(self, model_name: str, version: str, from_stage: str, to_stage: str) -> bool:
        """Promote model between stages"""
        try:
            self.registry.transition_model_stage(
                model_name=model_name,
                version=version,
                stage=to_stage,
                archive_existing_versions=(to_stage == "Production")
            )
            
            logger.info(f"Promoted {model_name} v{version} from {from_stage} to {to_stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False


def create_model_server(config: MLOpsConfig) -> ToWModelServer:
    """Create ToW model server from config"""
    return ToWModelServer(config)


def create_deployment_manager(config: MLOpsConfig) -> DeploymentManager:
    """Create deployment manager from config"""
    return DeploymentManager(config)