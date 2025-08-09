# ToW MLOps Infrastructure

Comprehensive MLOps infrastructure for the **Thoughts of Words (ToW)** research project, providing enterprise-grade model training, deployment, monitoring, and management capabilities.

## 🏗️ Architecture Overview

The ToW MLOps infrastructure consists of the following components:

- **Experiment Tracking**: MLflow and Weights & Biases integration for comprehensive experiment management
- **Model Deployment**: Scalable model serving with A/B testing, load balancing, and auto-scaling
- **Training Pipeline**: Automated training workflows with hyperparameter optimization
- **Monitoring & Alerting**: Real-time performance monitoring, drift detection, and alert management
- **CI/CD Pipeline**: Automated testing, building, and deployment workflows
- **Resource Management**: Cost optimization and resource scheduling for 70B models

## 📊 Key Features

### 🔬 Experiment Management
- **Unified Tracking**: MLflow and W&B integration for experiment versioning
- **Model Registry**: Centralized model versioning with stage transitions
- **Artifact Management**: Automatic logging of models, datasets, and metrics
- **Reproducible Experiments**: Environment and dependency tracking

### 🚀 Model Deployment
- **Multi-Environment Support**: Development, staging, and production deployments
- **A/B Testing**: Built-in experimentation framework for model comparison
- **Auto-Scaling**: Kubernetes-based horizontal pod autoscaling
- **Load Balancing**: Intelligent request routing with health checks

### 📈 Performance Monitoring
- **Real-time Metrics**: Latency, throughput, error rates, and quality scores
- **Model Drift Detection**: Automatic detection of performance degradation
- **Alert Management**: Multi-channel notifications (Slack, Discord, Email)
- **Dashboard Integration**: Grafana dashboards with Prometheus metrics

### 🔄 Automated Pipelines
- **Training Orchestration**: Distributed training with fault tolerance
- **Evaluation Workflows**: Comprehensive model benchmarking
- **Deployment Automation**: CI/CD with quality gates and rollback capabilities
- **Resource Optimization**: Cost-aware scheduling and spot instance usage

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for training)
- Kubernetes cluster (for production deployment)

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Increase_MLLM_Ability

# Install MLOps dependencies
pip install -r requirements-mlops.txt

# Install the ToW package
pip install -e .
```

### 2. Environment Setup

```bash
# Copy and configure environment settings
cp mlops/configs/development.yaml mlops/configs/local.yaml

# Edit configuration as needed
# Update database URLs, API keys, and resource limits

# Set environment variables
export MLOPS_ENV=development
export MLOPS_CONFIG_FILE=mlops/configs/local.yaml
```

### 3. Start Infrastructure Services

```bash
# Start all services with Docker Compose
cd mlops/docker
docker-compose up -d

# Or start specific services
docker-compose up -d postgres redis mlflow minio prometheus grafana
```

### 4. Verify Installation

```bash
# Check service health
curl http://localhost:5000/health  # MLflow
curl http://localhost:3000        # Grafana (admin/tow_grafana_password)
curl http://localhost:9090        # Prometheus

# Test model server (if deployed)
curl http://localhost:8000/health
```

## 📚 Usage Guide

### Training a Model

#### Basic Training
```bash
# Train a ToW model with default configuration
python mlops/scripts/train.py train \
  --model-name tow-llama-7b \
  --train-data /path/to/training_data.jsonl \
  --evaluate \
  --register
```

#### Advanced Training with Configuration
```bash
# Create training configuration
cat > training_config.yaml << EOF
model_name: tow-llama-7b-v2
base_model_path: /path/to/base/model
epochs: 5
batch_size: 8
learning_rate: 1e-5
use_lora: true
lora_r: 32
train_dataset_path: /path/to/train.jsonl
val_dataset_path: /path/to/val.jsonl
EOF

# Run training with configuration
python mlops/scripts/train.py train \
  --config training_config.yaml \
  --evaluate \
  --register
```

#### Hyperparameter Optimization
```bash
# Run Optuna optimization
python mlops/scripts/train.py optimize \
  --model-name tow-llama-7b \
  --train-data /path/to/train.jsonl \
  --trials 50 \
  --study-name tow-hpo-experiment
```

#### Distributed Training
```bash
# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python mlops/scripts/train.py train \
  --model-name tow-llama-7b-distributed \
  --train-data /path/to/train.jsonl \
  --config distributed_config.yaml
```

### Model Deployment

#### Development Deployment
```bash
# Deploy to local development environment
python mlops/scripts/deploy.py deploy \
  --model tow-llama-7b \
  --version v1.0.0 \
  --environment development
```

#### Production Deployment
```bash
# Deploy to production with canary strategy
python mlops/scripts/deploy.py deploy \
  --model tow-llama-7b \
  --version v1.2.0 \
  --environment production \
  --strategy canary
```

#### Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f mlops/k8s/model-server-deployment.yaml

# Check deployment status
kubectl get pods -n tow-production
kubectl logs -f deployment/tow-model-server -n tow-production
```

### Model Inference

#### REST API
```bash
# Make inference request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "source_language": "en",
    "target_language": "ko",
    "task_type": "translation",
    "temperature": 0.7
  }'
```

#### Python Client
```python
import requests
import json

# Make inference request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Explain quantum computing in simple terms",
        "source_language": "en",
        "target_language": "ja",
        "task_type": "generation",
        "max_tokens": 512,
        "temperature": 0.8
    }
)

result = response.json()
print(f"Output: {result['output_text']}")
print(f"Thought tokens: {result['thought_tokens']}")
print(f"Confidence: {result['confidence_score']:.3f}")
```

### Monitoring and Alerts

#### View Metrics
- **MLflow UI**: http://localhost:5000
- **Grafana Dashboard**: http://localhost:3000 (admin/tow_grafana_password)
- **Prometheus Metrics**: http://localhost:9090

#### Configure Alerts
```yaml
# Update monitoring configuration
monitoring:
  slack_webhook: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  email_recipients: ["team@company.com"]
  
  # Alert thresholds
  max_latency_ms: 5000
  max_error_rate: 0.05
  min_confidence_score: 0.75
```

#### Check Model Health
```bash
# Get current health status
curl http://localhost:8000/health

# View recent metrics
curl http://localhost:8000/metrics
```

## 🏭 Production Deployment

### Infrastructure Requirements

#### Hardware Specifications
- **Training**: 4x NVIDIA A100 80GB GPUs, 128GB RAM, 32 CPU cores
- **Inference**: 2x NVIDIA V100 32GB GPUs, 64GB RAM, 16 CPU cores
- **Storage**: High-performance SSD with 2TB+ for model storage

#### Software Dependencies
- **Kubernetes**: v1.25+
- **NVIDIA GPU Operator**: Latest version
- **Persistent Storage**: ReadWriteMany storage class
- **Load Balancer**: NGINX Ingress or cloud load balancer

### Production Setup

#### 1. Configure Production Environment
```bash
# Copy production configuration
cp mlops/configs/production.yaml /etc/tow-mlops/config.yaml

# Set production environment variables
export MLOPS_ENV=production
export MLOPS_CONFIG_FILE=/etc/tow-mlops/config.yaml
export DATABASE_PASSWORD=<secure-password>
export JWT_SECRET=<jwt-secret-key>
export WANDB_API_KEY=<wandb-api-key>
```

#### 2. Deploy Infrastructure
```bash
# Create namespace
kubectl create namespace tow-production

# Apply secrets
kubectl create secret generic tow-database-secret \
  --from-literal=connection-string=postgresql://user:pass@host:5432/db \
  -n tow-production

kubectl create secret generic tow-auth-secret \
  --from-literal=jwt-secret=<your-jwt-secret> \
  -n tow-production

# Deploy services
kubectl apply -f mlops/k8s/ -n tow-production
```

#### 3. Configure Monitoring
```bash
# Deploy monitoring stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Configure Grafana dashboards
kubectl apply -f mlops/k8s/monitoring/
```

#### 4. Setup CI/CD Pipeline
```bash
# Configure GitHub Actions or GitLab CI
# See .github/workflows/mlops-pipeline.yml for example

# Set up automated training schedule
kubectl apply -f mlops/k8s/training-job.yaml
```

## 🔧 Configuration

### Environment Configuration

The system supports multiple environments with different configurations:

- **Development**: Local development with minimal resources
- **Staging**: Pre-production testing environment
- **Production**: High-availability production deployment

### Key Configuration Files

| File | Purpose |
|------|---------|
| `mlops/configs/development.yaml` | Development environment settings |
| `mlops/configs/staging.yaml` | Staging environment settings |
| `mlops/configs/production.yaml` | Production environment settings |
| `mlops/docker/docker-compose.yml` | Docker services configuration |
| `mlops/k8s/` | Kubernetes deployment manifests |

### Configuration Parameters

#### Model Server Settings
```yaml
model_server:
  host: "0.0.0.0"
  port: 8000
  max_workers: 8
  timeout: 300
  max_batch_size: 16
  enable_caching: true
  quantization: "4bit"
  flash_attention: true
```

#### Resource Management
```yaml
resources:
  gpu_memory_fraction: 0.95
  min_replicas: 2
  max_replicas: 20
  target_cpu_percent: 70
  use_spot_instances: true
  max_hourly_cost: 50.0
```

#### Monitoring Configuration
```yaml
monitoring:
  max_latency_ms: 3000
  min_throughput_rps: 5.0
  max_error_rate: 0.02
  alert_on_model_drift: true
  slack_webhook: "https://hooks.slack.com/..."
```

## 🔍 Troubleshooting

### Common Issues

#### 1. GPU Out of Memory
```bash
# Reduce batch size or enable gradient checkpointing
export CUDA_VISIBLE_DEVICES=0
# or
export MODEL_QUANTIZATION=4bit
```

#### 2. MLflow Connection Issues
```bash
# Check MLflow server status
curl http://localhost:5000/health

# Restart MLflow service
docker-compose restart mlflow
```

#### 3. Model Loading Failures
```bash
# Check model registry
python -c "
from mlops.tracking import ModelRegistry
from mlops.config import load_config
registry = ModelRegistry(load_config().experiments)
print(registry.list_models())
"
```

#### 4. Kubernetes Deployment Issues
```bash
# Check pod status
kubectl get pods -n tow-production
kubectl describe pod <pod-name> -n tow-production
kubectl logs <pod-name> -n tow-production

# Check resources
kubectl top nodes
kubectl top pods -n tow-production
```

### Performance Tuning

#### Model Serving Optimization
- Enable model quantization (4-bit/8-bit)
- Use flash attention for memory efficiency
- Configure appropriate batch sizes
- Enable response caching for repeated queries

#### Training Optimization
- Use gradient checkpointing for memory efficiency
- Enable mixed precision training (fp16)
- Use distributed training for large models
- Configure optimal learning rate schedules

### Monitoring and Debugging

#### View Logs
```bash
# Application logs
docker-compose logs -f tow-model-server

# Kubernetes logs
kubectl logs -f deployment/tow-model-server -n tow-production

# Training logs
kubectl logs -f job/tow-training-job -n tow-training
```

#### Check Metrics
```bash
# Prometheus metrics
curl http://localhost:9090/api/v1/query?query=tow_requests_total

# Custom metrics
curl http://localhost:8000/metrics
```

## 🔐 Security

### Authentication and Authorization
- JWT-based API authentication
- Role-based access control (RBAC)
- API key management for external services
- Secure secret management

### Network Security
- TLS/SSL encryption for all communications
- Network policies for Kubernetes deployments
- VPC isolation and security groups
- Regular security scans and updates

### Data Protection
- Encryption at rest for model artifacts
- Input validation and sanitization
- Audit logging for compliance
- GDPR-compliant data handling

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run tests: `pytest mlops/tests/`
6. Submit a pull request

### Code Standards
- Follow PEP 8 for Python code style
- Use type hints for all functions
- Add comprehensive docstrings
- Write unit tests for new features
- Update documentation for changes

### Testing
```bash
# Run unit tests
pytest mlops/tests/

# Run integration tests
pytest mlops/tests/integration/

# Run with coverage
pytest --cov=mlops mlops/tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- MLflow team for the excellent experiment tracking framework
- Weights & Biases for advanced experiment management
- FastAPI for the high-performance web framework
- Prometheus and Grafana for monitoring infrastructure
- Kubernetes community for container orchestration

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation and FAQ
- Join our community discussions

---

**ToW MLOps Infrastructure** - Scaling Multilingual AI Research with Enterprise-Grade MLOps