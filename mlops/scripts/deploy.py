#!/usr/bin/env python3
"""
ToW MLOps Deployment Script
==========================

Comprehensive deployment script for ToW models with support for:
- Docker containerization
- Kubernetes deployment
- Model server deployment
- CI/CD pipeline integration
- Health checks and monitoring
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
import docker
import kubernetes
from kubernetes import client, config

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mlops.config import MLOpsConfig, load_config
from mlops.deployment import DeploymentManager, create_model_server
from mlops.tracking import ModelRegistry


logger = logging.getLogger(__name__)


class DeploymentOrchestrator:
    """Orchestrates ToW model deployments across different environments"""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.deployment_manager = DeploymentManager(config)
        self.registry = ModelRegistry(config.experiments)
        
        # Initialize clients
        self.docker_client = None
        self.k8s_client = None
        
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Docker client: {e}")
        
        try:
            config.load_incluster_config() if self._is_running_in_cluster() else config.load_kube_config()
            self.k8s_client = client.ApiClient()
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {e}")
    
    def _is_running_in_cluster(self) -> bool:
        """Check if running inside a Kubernetes cluster"""
        return os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount')
    
    async def deploy_model_server(self, 
                                 model_name: str,
                                 model_version: str = "latest",
                                 environment: str = "staging",
                                 deployment_strategy: str = "rolling") -> Dict[str, Any]:
        """Deploy ToW model server"""
        
        logger.info(f"Deploying {model_name} v{model_version} to {environment}")
        
        deployment_result = {
            "model_name": model_name,
            "model_version": model_version,
            "environment": environment,
            "deployment_strategy": deployment_strategy,
            "started_at": datetime.now(),
            "status": "started",
            "steps": []
        }
        
        try:
            # Step 1: Validate model exists
            logger.info("Validating model in registry...")
            model_info = self.registry.get_model_version(model_name, model_version)
            if not model_info:
                raise ValueError(f"Model {model_name} v{model_version} not found in registry")
            
            deployment_result["steps"].append({
                "step": "model_validation",
                "status": "completed",
                "message": f"Model {model_name} v{model_version} validated"
            })
            
            # Step 2: Build container image (if needed)
            if self.docker_client:
                logger.info("Building container image...")
                image_tag = await self._build_model_server_image(model_name, model_version)
                deployment_result["image_tag"] = image_tag
                deployment_result["steps"].append({
                    "step": "image_build",
                    "status": "completed",
                    "message": f"Built image: {image_tag}"
                })
            
            # Step 3: Deploy based on environment
            if environment in ["development", "staging"] and self.docker_client:
                # Docker Compose deployment
                deploy_result = await self._deploy_with_docker_compose(
                    model_name, model_version, environment
                )
            elif environment == "production" and self.k8s_client:
                # Kubernetes deployment
                deploy_result = await self._deploy_with_kubernetes(
                    model_name, model_version, deployment_strategy
                )
            else:
                # Direct deployment
                deploy_result = await self._deploy_direct(
                    model_name, model_version, environment
                )
            
            deployment_result.update(deploy_result)
            deployment_result["steps"].append({
                "step": "deployment",
                "status": "completed",
                "message": f"Deployed to {environment}"
            })
            
            # Step 4: Health checks
            logger.info("Performing health checks...")
            health_check_result = await self._perform_health_checks(deploy_result)
            deployment_result["health_check"] = health_check_result
            deployment_result["steps"].append({
                "step": "health_check",
                "status": "completed" if health_check_result["healthy"] else "failed",
                "message": health_check_result["message"]
            })
            
            # Step 5: Update deployment registry
            deployment_id = self.deployment_manager.deploy_model(
                model_name=model_name,
                version=model_version,
                environment=environment,
                deployment_strategy=deployment_strategy
            )
            
            deployment_result["deployment_id"] = deployment_id
            deployment_result["status"] = "completed"
            deployment_result["completed_at"] = datetime.now()
            
            logger.info(f"Deployment completed successfully: {deployment_id}")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployment_result["status"] = "failed"
            deployment_result["error"] = str(e)
            deployment_result["completed_at"] = datetime.now()
            
            # Attempt rollback
            if deployment_result.get("deployment_id"):
                logger.info("Attempting rollback...")
                rollback_success = self.deployment_manager.rollback_deployment(
                    deployment_result["deployment_id"]
                )
                deployment_result["rollback_attempted"] = True
                deployment_result["rollback_success"] = rollback_success
        
        return deployment_result
    
    async def _build_model_server_image(self, model_name: str, model_version: str) -> str:
        """Build Docker image for model server"""
        
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile.model-server"
        context_path = Path(__file__).parent.parent.parent
        
        image_tag = f"tow-research/model-server:{model_name}-{model_version}"
        
        logger.info(f"Building Docker image: {image_tag}")
        
        # Build image
        build_logs = self.docker_client.api.build(
            path=str(context_path),
            dockerfile=str(dockerfile_path.relative_to(context_path)),
            tag=image_tag,
            rm=True,
            decode=True
        )
        
        # Process build logs
        for log in build_logs:
            if 'stream' in log:
                logger.debug(log['stream'].strip())
            elif 'error' in log:
                raise Exception(f"Docker build failed: {log['error']}")
        
        logger.info(f"Successfully built image: {image_tag}")
        return image_tag
    
    async def _deploy_with_docker_compose(self, 
                                        model_name: str,
                                        model_version: str,
                                        environment: str) -> Dict[str, Any]:
        """Deploy using Docker Compose"""
        
        compose_file = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        
        # Update compose file with model configuration
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Update model server service
        if 'tow-model-server' in compose_config['services']:
            service = compose_config['services']['tow-model-server']
            service['environment']['MODEL_NAME'] = model_name
            service['environment']['MODEL_VERSION'] = model_version
            service['environment']['MLOPS_ENV'] = environment
        
        # Write updated compose file
        temp_compose_file = compose_file.parent / f"docker-compose-{environment}.yml"
        with open(temp_compose_file, 'w') as f:
            yaml.safe_dump(compose_config, f)
        
        # Deploy with Docker Compose
        cmd = [
            "docker-compose",
            "-f", str(temp_compose_file),
            "up", "-d",
            "tow-model-server"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Docker Compose deployment failed: {result.stderr}")
        
        # Get container information
        containers = self.docker_client.containers.list(
            filters={"label": "com.docker.compose.service=tow-model-server"}
        )
        
        return {
            "deployment_type": "docker_compose",
            "compose_file": str(temp_compose_file),
            "containers": [{"id": c.id, "name": c.name} for c in containers],
            "service_url": f"http://localhost:{self.config.model_server.port}"
        }
    
    async def _deploy_with_kubernetes(self, 
                                    model_name: str,
                                    model_version: str,
                                    strategy: str) -> Dict[str, Any]:
        """Deploy using Kubernetes"""
        
        k8s_manifests_dir = Path(__file__).parent.parent / "k8s"
        deployment_file = k8s_manifests_dir / "model-server-deployment.yaml"
        
        # Load and update Kubernetes manifests
        with open(deployment_file, 'r') as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Update deployment with model information
        for manifest in manifests:
            if manifest.get('kind') == 'Deployment':
                containers = manifest['spec']['template']['spec']['containers']
                for container in containers:
                    if container['name'] == 'tow-model-server':
                        # Update image tag
                        container['image'] = f"tow-research/model-server:{model_name}-{model_version}"
                        
                        # Add model-specific environment variables
                        if 'env' not in container:
                            container['env'] = []
                        
                        container['env'].extend([
                            {"name": "MODEL_NAME", "value": model_name},
                            {"name": "MODEL_VERSION", "value": model_version}
                        ])
        
        # Apply manifests
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        applied_resources = []
        
        for manifest in manifests:
            try:
                if manifest.get('kind') == 'Deployment':
                    # Apply deployment
                    deployment = apps_v1.create_namespaced_deployment(
                        namespace=manifest['metadata']['namespace'],
                        body=manifest
                    )
                    applied_resources.append({
                        "kind": "Deployment",
                        "name": deployment.metadata.name,
                        "namespace": deployment.metadata.namespace
                    })
                
                elif manifest.get('kind') == 'Service':
                    # Apply service
                    service = core_v1.create_namespaced_service(
                        namespace=manifest['metadata']['namespace'],
                        body=manifest
                    )
                    applied_resources.append({
                        "kind": "Service",
                        "name": service.metadata.name,
                        "namespace": service.metadata.namespace
                    })
                
                # Add support for other resource types as needed
                
            except kubernetes.client.exceptions.ApiException as e:
                if e.status == 409:  # Resource already exists
                    logger.info(f"Resource {manifest.get('kind')} already exists, updating...")
                    # Handle updates here if needed
                else:
                    raise
        
        return {
            "deployment_type": "kubernetes",
            "strategy": strategy,
            "applied_resources": applied_resources,
            "namespace": "tow-production"
        }
    
    async def _deploy_direct(self, 
                           model_name: str,
                           model_version: str,
                           environment: str) -> Dict[str, Any]:
        """Deploy directly without orchestration"""
        
        # Start model server directly
        model_server = create_model_server(self.config)
        
        # This would typically be run in a separate process
        # For demonstration, we'll return the configuration
        
        return {
            "deployment_type": "direct",
            "model_name": model_name,
            "model_version": model_version,
            "environment": environment,
            "service_url": f"http://{self.config.model_server.host}:{self.config.model_server.port}",
            "message": "Direct deployment configured (server needs to be started separately)"
        }
    
    async def _perform_health_checks(self, deployment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health checks on deployed service"""
        
        import httpx
        
        service_url = deployment_result.get("service_url")
        if not service_url:
            return {"healthy": False, "message": "No service URL provided"}
        
        health_url = f"{service_url}/health"
        
        # Wait for service to be ready
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(health_url, timeout=10)
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        return {
                            "healthy": True,
                            "message": "Service is healthy",
                            "health_data": health_data,
                            "attempts": attempt + 1
                        }
                    else:
                        logger.warning(f"Health check failed (attempt {attempt + 1}): HTTP {response.status_code}")
                        
            except Exception as e:
                logger.warning(f"Health check failed (attempt {attempt + 1}): {e}")
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(10)  # Wait 10 seconds between attempts
        
        return {
            "healthy": False,
            "message": f"Service health check failed after {max_attempts} attempts",
            "attempts": max_attempts
        }
    
    async def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment"""
        
        logger.info(f"Rolling back deployment: {deployment_id}")
        
        success = self.deployment_manager.rollback_deployment(deployment_id)
        
        return {
            "deployment_id": deployment_id,
            "rollback_success": success,
            "timestamp": datetime.now()
        }
    
    def list_deployments(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active deployments"""
        return self.deployment_manager.list_deployments(environment)


def main():
    """Main deployment script"""
    
    parser = argparse.ArgumentParser(
        description="ToW MLOps Deployment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy latest model to staging
  python deploy.py deploy --model tow-llama-7b --environment staging
  
  # Deploy specific version to production with canary strategy
  python deploy.py deploy --model tow-llama-7b --version v1.2.0 --environment production --strategy canary
  
  # Rollback deployment
  python deploy.py rollback --deployment-id abc123
  
  # List deployments
  python deploy.py list --environment production
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model")
    deploy_parser.add_argument("--model", required=True, help="Model name")
    deploy_parser.add_argument("--version", default="latest", help="Model version")
    deploy_parser.add_argument("--environment", default="staging", 
                              choices=["development", "staging", "production"],
                              help="Deployment environment")
    deploy_parser.add_argument("--strategy", default="rolling",
                              choices=["rolling", "blue_green", "canary"],
                              help="Deployment strategy")
    deploy_parser.add_argument("--config", help="MLOps configuration file")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback deployment")
    rollback_parser.add_argument("--deployment-id", required=True, help="Deployment ID to rollback")
    rollback_parser.add_argument("--config", help="MLOps configuration file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List deployments")
    list_parser.add_argument("--environment", help="Filter by environment")
    list_parser.add_argument("--config", help="MLOps configuration file")
    
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
    if args.config:
        config = MLOpsConfig.from_file(args.config)
    else:
        config = load_config()
    
    # Create orchestrator
    orchestrator = DeploymentOrchestrator(config)
    
    # Execute command
    async def run_command():
        if args.command == "deploy":
            result = await orchestrator.deploy_model_server(
                model_name=args.model,
                model_version=args.version,
                environment=args.environment,
                deployment_strategy=args.strategy
            )
            print(json.dumps(result, indent=2, default=str))
            
        elif args.command == "rollback":
            result = await orchestrator.rollback_deployment(args.deployment_id)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.command == "list":
            deployments = orchestrator.list_deployments(args.environment)
            print(json.dumps(deployments, indent=2, default=str))
    
    # Run async command
    asyncio.run(run_command())


if __name__ == "__main__":
    main()