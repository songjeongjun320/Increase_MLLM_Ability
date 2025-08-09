"""
MLOps Configuration Management
=============================

Centralized configuration management for the MLOps infrastructure.
Supports multiple environments (dev, staging, production) and cloud providers.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class Environment(Enum):
    """Deployment environments"""
    DEV = "development"
    STAGING = "staging"
    PROD = "production"


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp" 
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class DatabaseConfig:
    """Database configuration for experiment tracking"""
    host: str = "localhost"
    port: int = 5432
    database: str = "tow_experiments"
    username: str = "tow_user"
    password: str = ""
    ssl_mode: str = "prefer"
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string"""
        return (f"postgresql://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}")


@dataclass
class ModelServerConfig:
    """Model serving configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_workers: int = 4
    timeout: int = 300
    max_batch_size: int = 8
    enable_batching: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    rate_limit_per_minute: int = 100
    
    # Model-specific settings
    max_sequence_length: int = 4096
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    
    # Hardware optimization
    device: str = "auto"  # auto, cpu, cuda:0, etc.
    quantization: Optional[str] = "4bit"  # None, 8bit, 4bit
    torch_compile: bool = False
    flash_attention: bool = True


@dataclass
class ResourceConfig:
    """Resource management configuration"""
    # Compute resources
    gpu_memory_fraction: float = 0.9
    cpu_cores: int = 8
    memory_limit_gb: int = 64
    
    # Auto-scaling
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 70
    target_memory_percent: int = 80
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    
    # Cost optimization
    use_spot_instances: bool = False
    max_hourly_cost: float = 10.0
    auto_shutdown_idle_minutes: int = 30


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    # Metrics collection
    metrics_port: int = 9090
    collection_interval: int = 15  # seconds
    retention_days: int = 30
    
    # Performance thresholds
    max_latency_ms: int = 5000
    min_throughput_rps: float = 1.0
    max_error_rate: float = 0.05
    min_confidence_score: float = 0.7
    
    # Alert channels
    slack_webhook: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    discord_webhook: Optional[str] = None
    
    # Alert conditions
    alert_on_high_latency: bool = True
    alert_on_low_throughput: bool = True
    alert_on_high_error_rate: bool = True
    alert_on_model_drift: bool = True
    alert_on_resource_usage: bool = True


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    # MLflow settings
    tracking_uri: str = "sqlite:///mlruns.db"
    artifact_location: str = "./mlartifacts"
    registry_uri: Optional[str] = None
    
    # Wandb settings (optional)
    wandb_project: str = "tow-research"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    # Experiment metadata
    auto_log_model: bool = True
    auto_log_metrics: bool = True
    auto_log_artifacts: bool = True
    log_system_metrics: bool = True
    
    # Model versioning
    model_registry_name: str = "ToW-Models"
    stage_transition_webhook: Optional[str] = None


@dataclass
class SecurityConfig:
    """Security configuration"""
    # Authentication
    enable_auth: bool = True
    jwt_secret: str = "your-secret-key-here"
    jwt_expiry_hours: int = 24
    
    # API security
    enable_rate_limiting: bool = True
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Model security
    enable_input_validation: bool = True
    max_input_length: int = 10000
    sanitize_inputs: bool = True
    
    # Network security
    enable_https: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """CI/CD Pipeline configuration"""
    # Training pipeline
    auto_trigger_training: bool = False
    training_schedule: Optional[str] = None  # cron expression
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    
    # Evaluation pipeline
    evaluation_datasets: List[str] = field(default_factory=lambda: ["mmlu", "klue"])
    evaluation_metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "bertscore"])
    benchmark_threshold: float = 0.85
    
    # Deployment pipeline
    auto_deploy_on_improvement: bool = False
    deployment_approval_required: bool = True
    rollback_on_failure: bool = True
    canary_deployment: bool = True
    canary_traffic_percent: int = 10


@dataclass
class MLOpsConfig:
    """Main MLOps configuration container"""
    environment: Environment = Environment.DEV
    cloud_provider: CloudProvider = CloudProvider.LOCAL
    project_name: str = "tow-research"
    version: str = "1.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model_server: ModelServerConfig = field(default_factory=ModelServerConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Environment-specific overrides
    _env_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'MLOpsConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls._from_dict(config_data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'MLOpsConfig':
        """Create config from dictionary"""
        config = cls()
        
        # Set main fields
        if 'environment' in data:
            config.environment = Environment(data['environment'])
        if 'cloud_provider' in data:
            config.cloud_provider = CloudProvider(data['cloud_provider'])
        if 'project_name' in data:
            config.project_name = data['project_name']
        if 'version' in data:
            config.version = data['version']
        
        # Set component configurations
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'model_server' in data:
            config.model_server = ModelServerConfig(**data['model_server'])
        if 'resources' in data:
            config.resources = ResourceConfig(**data['resources'])
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
        if 'experiments' in data:
            config.experiments = ExperimentConfig(**data['experiments'])
        if 'security' in data:
            config.security = SecurityConfig(**data['security'])
        if 'pipeline' in data:
            config.pipeline = PipelineConfig(**data['pipeline'])
        
        return config
    
    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {
            'environment': self.environment.value,
            'cloud_provider': self.cloud_provider.value,
            'project_name': self.project_name,
            'version': self.version,
            'database': self.database.__dict__,
            'model_server': self.model_server.__dict__,
            'resources': self.resources.__dict__,
            'monitoring': self.monitoring.__dict__,
            'experiments': self.experiments.__dict__,
            'security': self.security.__dict__,
            'pipeline': self.pipeline.__dict__
        }
        
        # Apply environment-specific overrides
        env_key = self.environment.value
        if env_key in self._env_overrides:
            self._apply_overrides(result, self._env_overrides[env_key])
        
        return result
    
    def _apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """Apply environment-specific overrides"""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    
    def get_env_var(self, key: str, default: Any = None) -> Any:
        """Get environment variable with fallback"""
        return os.getenv(key, default)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate database connection
        if not self.database.host:
            errors.append("Database host is required")
        
        # Validate model server
        if self.model_server.port <= 0 or self.model_server.port > 65535:
            errors.append("Model server port must be between 1 and 65535")
        
        # Validate resources
        if self.resources.gpu_memory_fraction <= 0 or self.resources.gpu_memory_fraction > 1:
            errors.append("GPU memory fraction must be between 0 and 1")
        
        # Validate monitoring
        if self.monitoring.max_latency_ms <= 0:
            errors.append("Max latency must be positive")
        
        # Validate security
        if self.security.enable_auth and not self.security.jwt_secret:
            errors.append("JWT secret is required when authentication is enabled")
        
        return errors


def get_default_config(env: Environment = Environment.DEV) -> MLOpsConfig:
    """Get default configuration for specified environment"""
    config = MLOpsConfig(environment=env)
    
    if env == Environment.DEV:
        # Development settings
        config.model_server.max_workers = 2
        config.resources.max_replicas = 2
        config.monitoring.retention_days = 7
        config.security.enable_auth = False
        
    elif env == Environment.STAGING:
        # Staging settings
        config.model_server.max_workers = 4
        config.resources.max_replicas = 5
        config.monitoring.retention_days = 14
        config.security.enable_auth = True
        config.pipeline.deployment_approval_required = True
        
    elif env == Environment.PROD:
        # Production settings
        config.model_server.max_workers = 8
        config.resources.max_replicas = 10
        config.monitoring.retention_days = 30
        config.security.enable_auth = True
        config.security.enable_https = True
        config.pipeline.auto_deploy_on_improvement = False
        config.pipeline.deployment_approval_required = True
        config.pipeline.rollback_on_failure = True
    
    return config


# Load configuration from environment
def load_config() -> MLOpsConfig:
    """Load configuration based on environment variables"""
    env_name = os.getenv('MLOPS_ENV', 'development')
    config_file = os.getenv('MLOPS_CONFIG_FILE')
    
    try:
        env = Environment(env_name)
    except ValueError:
        env = Environment.DEV
    
    if config_file and os.path.exists(config_file):
        config = MLOpsConfig.from_file(config_file)
    else:
        config = get_default_config(env)
    
    # Override with environment variables
    if os.getenv('DATABASE_URL'):
        config.database.host = os.getenv('DATABASE_URL')
    
    if os.getenv('MODEL_SERVER_PORT'):
        config.model_server.port = int(os.getenv('MODEL_SERVER_PORT'))
    
    if os.getenv('WANDB_API_KEY'):
        config.experiments.wandb_api_key = os.getenv('WANDB_API_KEY')
    
    if os.getenv('SLACK_WEBHOOK'):
        config.monitoring.slack_webhook = os.getenv('SLACK_WEBHOOK')
    
    return config