"""
Model Monitoring and Performance Tracking
========================================

Comprehensive monitoring system for ToW models including performance metrics,
model drift detection, alerting, and automated reporting.
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

import numpy as np
import pandas as pd
import torch
import requests
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, Summary, push_to_gateway
import matplotlib.pyplot as plt
import seaborn as sns

from .config import MLOpsConfig, MonitoringConfig


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    # Response metrics
    latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    
    # Quality metrics
    confidence_score: float = 0.0
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    bert_score: float = 0.0
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Model-specific metrics
    thought_coherence: float = 0.0
    cultural_adaptation: float = 0.0
    translation_quality: float = 0.0
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'latency_ms': self.latency_ms,
            'throughput_rps': self.throughput_rps,
            'error_rate': self.error_rate,
            'confidence_score': self.confidence_score,
            'bleu_score': self.bleu_score,
            'rouge_score': self.rouge_score,
            'bert_score': self.bert_score,
            'cpu_usage': self.cpu_usage,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_usage': self.gpu_usage,
            'gpu_memory_mb': self.gpu_memory_mb,
            'thought_coherence': self.thought_coherence,
            'cultural_adaptation': self.cultural_adaptation,
            'translation_quality': self.translation_quality,
            'timestamp': self.timestamp.isoformat()
        }


class ModelDriftDetector:
    """Detects model performance drift over time"""
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        
        # Historical data windows
        self.confidence_scores = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.error_rates = deque(maxlen=window_size)
        
        # Baseline metrics (from training/validation)
        self.baseline_confidence = None
        self.baseline_response_time = None
        self.baseline_error_rate = None
        
        # Drift status
        self.drift_detected = False
        self.last_drift_check = datetime.now()
    
    def set_baseline(self, 
                    confidence: float,
                    response_time: float,
                    error_rate: float) -> None:
        """Set baseline metrics for drift detection"""
        self.baseline_confidence = confidence
        self.baseline_response_time = response_time
        self.baseline_error_rate = error_rate
        
        logger.info(f"Set baseline metrics: confidence={confidence:.3f}, "
                   f"response_time={response_time:.3f}, error_rate={error_rate:.3f}")
    
    def add_sample(self, 
                  confidence: float,
                  response_time: float,
                  error_rate: float) -> None:
        """Add a new sample for drift detection"""
        self.confidence_scores.append(confidence)
        self.response_times.append(response_time)
        self.error_rates.append(error_rate)
    
    def check_drift(self) -> Dict[str, Any]:
        """Check for model drift"""
        if len(self.confidence_scores) < 100:  # Need minimum samples
            return {"drift_detected": False, "reason": "Insufficient samples"}
        
        drift_results = {
            "drift_detected": False,
            "confidence_drift": False,
            "performance_drift": False,
            "error_drift": False,
            "metrics": {}
        }
        
        # Check confidence score drift
        if self.baseline_confidence is not None:
            current_confidence = np.mean(list(self.confidence_scores)[-100:])  # Last 100 samples
            confidence_change = abs(current_confidence - self.baseline_confidence) / self.baseline_confidence
            
            if confidence_change > self.threshold:
                drift_results["confidence_drift"] = True
                drift_results["drift_detected"] = True
            
            drift_results["metrics"]["confidence_change"] = confidence_change
            drift_results["metrics"]["current_confidence"] = current_confidence
            drift_results["metrics"]["baseline_confidence"] = self.baseline_confidence
        
        # Check response time drift
        if self.baseline_response_time is not None:
            current_response_time = np.mean(list(self.response_times)[-100:])
            response_time_change = abs(current_response_time - self.baseline_response_time) / self.baseline_response_time
            
            if response_time_change > self.threshold:
                drift_results["performance_drift"] = True
                drift_results["drift_detected"] = True
            
            drift_results["metrics"]["response_time_change"] = response_time_change
            drift_results["metrics"]["current_response_time"] = current_response_time
            drift_results["metrics"]["baseline_response_time"] = self.baseline_response_time
        
        # Check error rate drift
        if self.baseline_error_rate is not None:
            current_error_rate = np.mean(list(self.error_rates)[-100:])
            
            # For error rates, we're more concerned about increases
            if current_error_rate > self.baseline_error_rate + self.threshold:
                drift_results["error_drift"] = True
                drift_results["drift_detected"] = True
            
            drift_results["metrics"]["current_error_rate"] = current_error_rate
            drift_results["metrics"]["baseline_error_rate"] = self.baseline_error_rate
        
        self.drift_detected = drift_results["drift_detected"]
        self.last_drift_check = datetime.now()
        
        if self.drift_detected:
            logger.warning(f"Model drift detected: {drift_results}")
        
        return drift_results


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldown = {}  # Prevent spam
        
    def send_alert(self, 
                  alert_type: str,
                  message: str,
                  severity: str = "warning",
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send an alert through configured channels"""
        
        # Check cooldown
        cooldown_key = f"{alert_type}_{severity}"
        if cooldown_key in self.alert_cooldown:
            last_sent = self.alert_cooldown[cooldown_key]
            if datetime.now() - last_sent < timedelta(minutes=5):
                logger.debug(f"Alert {cooldown_key} in cooldown, skipping")
                return
        
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.alert_history.append(alert)
        self.alert_cooldown[cooldown_key] = datetime.now()
        
        logger.info(f"Sending {severity} alert: {message}")
        
        # Send to configured channels
        if self.config.slack_webhook:
            self._send_slack_alert(alert)
        
        if self.config.email_recipients:
            self._send_email_alert(alert)
        
        if self.config.discord_webhook:
            self._send_discord_alert(alert)
    
    def _send_slack_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to Slack"""
        try:
            color_map = {
                "info": "#36a64f",
                "warning": "#ff9900", 
                "error": "#ff0000",
                "critical": "#990000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert["severity"], "#cccccc"),
                    "title": f"🤖 ToW Model Alert - {alert['type'].title()}",
                    "text": alert["message"],
                    "fields": [
                        {"title": "Severity", "value": alert["severity"].title(), "short": True},
                        {"title": "Time", "value": alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ],
                    "footer": "ToW MLOps Monitoring",
                    "ts": int(alert["timestamp"].timestamp())
                }]
            }
            
            response = requests.post(self.config.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert via email"""
        try:
            # This is a basic implementation - would need SMTP configuration
            subject = f"ToW Model Alert: {alert['type'].title()} - {alert['severity'].title()}"
            
            body = f"""
            Alert Details:
            - Type: {alert['type']}
            - Severity: {alert['severity']}
            - Message: {alert['message']}
            - Time: {alert['timestamp']}
            
            Metadata:
            {json.dumps(alert['metadata'], indent=2)}
            
            ---
            ToW MLOps Monitoring System
            """
            
            # TODO: Implement actual email sending with SMTP configuration
            logger.info(f"Email alert prepared: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_discord_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to Discord"""
        try:
            color_map = {
                "info": 0x36a64f,
                "warning": 0xff9900,
                "error": 0xff0000,
                "critical": 0x990000
            }
            
            payload = {
                "embeds": [{
                    "title": f"🤖 ToW Model Alert - {alert['type'].title()}",
                    "description": alert["message"],
                    "color": color_map.get(alert["severity"], 0xcccccc),
                    "fields": [
                        {"name": "Severity", "value": alert["severity"].title(), "inline": True},
                        {"name": "Time", "value": alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
                    ],
                    "footer": {"text": "ToW MLOps Monitoring"},
                    "timestamp": alert["timestamp"].isoformat()
                }]
            }
            
            response = requests.post(self.config.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert["timestamp"] > cutoff]


class MetricsCollector:
    """Collects and aggregates metrics from ToW models"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_buffer = deque(maxlen=10000)
        self.aggregated_metrics = defaultdict(list)
        self.drift_detector = ModelDriftDetector()
        self.alert_manager = AlertManager(config)
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.setup_prometheus_metrics()
        
        # Background monitoring task
        self.monitoring_task = None
        self.is_running = False
    
    def setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics"""
        self.prom_latency = Histogram(
            'tow_request_latency_seconds',
            'Request latency in seconds',
            registry=self.registry
        )
        
        self.prom_throughput = Counter(
            'tow_requests_total',
            'Total number of requests',
            ['model_version', 'status'],
            registry=self.registry
        )
        
        self.prom_error_rate = Gauge(
            'tow_error_rate',
            'Current error rate',
            registry=self.registry
        )
        
        self.prom_confidence = Histogram(
            'tow_confidence_score',
            'Model confidence scores',
            registry=self.registry
        )
        
        self.prom_resource_usage = Gauge(
            'tow_resource_usage',
            'Resource usage metrics',
            ['resource_type']
        )
    
    def collect_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect performance metrics"""
        self.metrics_buffer.append(metrics)
        
        # Update Prometheus metrics
        self.prom_latency.observe(metrics.latency_ms / 1000.0)
        self.prom_confidence.observe(metrics.confidence_score)
        self.prom_resource_usage.labels('cpu').set(metrics.cpu_usage)
        self.prom_resource_usage.labels('memory').set(metrics.memory_usage_mb)
        self.prom_resource_usage.labels('gpu').set(metrics.gpu_usage)
        
        # Add to drift detector
        self.drift_detector.add_sample(
            metrics.confidence_score,
            metrics.latency_ms,
            metrics.error_rate
        )
        
        # Check thresholds and send alerts
        self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check metric thresholds and trigger alerts"""
        
        # High latency alert
        if (self.config.alert_on_high_latency and 
            metrics.latency_ms > self.config.max_latency_ms):
            self.alert_manager.send_alert(
                "high_latency",
                f"High latency detected: {metrics.latency_ms:.2f}ms (threshold: {self.config.max_latency_ms}ms)",
                "warning",
                {"latency_ms": metrics.latency_ms, "threshold": self.config.max_latency_ms}
            )
        
        # Low throughput alert
        if (self.config.alert_on_low_throughput and 
            metrics.throughput_rps < self.config.min_throughput_rps):
            self.alert_manager.send_alert(
                "low_throughput",
                f"Low throughput detected: {metrics.throughput_rps:.2f} RPS (threshold: {self.config.min_throughput_rps})",
                "warning",
                {"throughput_rps": metrics.throughput_rps, "threshold": self.config.min_throughput_rps}
            )
        
        # High error rate alert
        if (self.config.alert_on_high_error_rate and 
            metrics.error_rate > self.config.max_error_rate):
            self.alert_manager.send_alert(
                "high_error_rate",
                f"High error rate detected: {metrics.error_rate:.3f} (threshold: {self.config.max_error_rate})",
                "error",
                {"error_rate": metrics.error_rate, "threshold": self.config.max_error_rate}
            )
        
        # Low confidence alert
        if metrics.confidence_score < self.config.min_confidence_score:
            self.alert_manager.send_alert(
                "low_confidence",
                f"Low model confidence: {metrics.confidence_score:.3f} (threshold: {self.config.min_confidence_score})",
                "warning",
                {"confidence_score": metrics.confidence_score, "threshold": self.config.min_confidence_score}
            )
        
        # Resource usage alerts
        if self.config.alert_on_resource_usage:
            if metrics.cpu_usage > 90:
                self.alert_manager.send_alert(
                    "high_cpu_usage",
                    f"High CPU usage: {metrics.cpu_usage:.1f}%",
                    "warning",
                    {"cpu_usage": metrics.cpu_usage}
                )
            
            if metrics.gpu_usage > 95:
                self.alert_manager.send_alert(
                    "high_gpu_usage",
                    f"High GPU usage: {metrics.gpu_usage:.1f}%",
                    "warning",
                    {"gpu_usage": metrics.gpu_usage}
                )
    
    def get_aggregated_metrics(self, 
                             time_window: str = "1h",
                             metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get aggregated metrics over time window"""
        
        # Parse time window
        if time_window == "1h":
            cutoff = datetime.now() - timedelta(hours=1)
        elif time_window == "24h":
            cutoff = datetime.now() - timedelta(hours=24)
        elif time_window == "7d":
            cutoff = datetime.now() - timedelta(days=7)
        else:
            cutoff = datetime.now() - timedelta(hours=1)  # Default to 1 hour
        
        # Filter metrics by time window
        recent_metrics = [
            m for m in self.metrics_buffer 
            if m.timestamp > cutoff
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate aggregations
        aggregated = {}
        
        metrics_to_process = metric_names or [
            'latency_ms', 'throughput_rps', 'error_rate', 'confidence_score',
            'cpu_usage', 'memory_usage_mb', 'gpu_usage', 'thought_coherence'
        ]
        
        for metric_name in metrics_to_process:
            values = [getattr(m, metric_name) for m in recent_metrics if hasattr(m, metric_name)]
            
            if values:
                aggregated[metric_name] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'count': len(values)
                }
        
        aggregated['time_window'] = time_window
        aggregated['sample_count'] = len(recent_metrics)
        
        return aggregated
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                return allocated / (1024 * 1024)  # Convert to MB
            return 0.0
        except:
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def get_gpu_usage(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            if torch.cuda.is_available():
                # This is a simplified version - would need nvidia-ml-py for accurate GPU utilization
                return 0.0  # Placeholder
            return 0.0
        except:
            return 0.0
    
    def check_model_drift(self) -> Dict[str, Any]:
        """Check for model drift"""
        return self.drift_detector.check_drift()
    
    def generate_report(self, 
                       time_window: str = "24h",
                       include_charts: bool = True) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        # Get aggregated metrics
        metrics = self.get_aggregated_metrics(time_window)
        
        # Check for drift
        drift_results = self.check_model_drift()
        
        # Get recent alerts
        recent_alerts = self.alert_manager.get_recent_alerts(24)
        
        # Current system status
        current_status = {
            'cpu_usage': self.get_cpu_usage(),
            'memory_usage_mb': self.get_memory_usage(),
            'gpu_usage': self.get_gpu_usage(),
            'gpu_memory_mb': self.get_gpu_memory_usage()
        }
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_window': time_window,
            'aggregated_metrics': metrics,
            'drift_analysis': drift_results,
            'recent_alerts': recent_alerts,
            'current_system_status': current_status,
            'summary': self._generate_summary(metrics, drift_results, recent_alerts)
        }
        
        # Generate charts if requested
        if include_charts:
            charts_path = self._generate_charts(time_window)
            report['charts_path'] = charts_path
        
        return report
    
    def _generate_summary(self, 
                         metrics: Dict[str, Any],
                         drift_results: Dict[str, Any],
                         alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of monitoring status"""
        
        # Overall health score (0-100)
        health_score = 100
        issues = []
        
        # Check metrics health
        if 'error_rate' in metrics and metrics['error_rate']['mean'] > self.config.max_error_rate:
            health_score -= 20
            issues.append("High error rate detected")
        
        if 'latency_ms' in metrics and metrics['latency_ms']['mean'] > self.config.max_latency_ms:
            health_score -= 15
            issues.append("High latency detected")
        
        if 'confidence_score' in metrics and metrics['confidence_score']['mean'] < self.config.min_confidence_score:
            health_score -= 10
            issues.append("Low model confidence")
        
        # Check drift
        if drift_results.get('drift_detected', False):
            health_score -= 25
            issues.append("Model drift detected")
        
        # Check recent alerts
        critical_alerts = [a for a in alerts if a['severity'] in ['error', 'critical']]
        if critical_alerts:
            health_score -= len(critical_alerts) * 5
            issues.append(f"{len(critical_alerts)} critical alerts in last 24h")
        
        health_score = max(0, health_score)  # Ensure non-negative
        
        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            'health_score': health_score,
            'status': status,
            'issues': issues,
            'total_samples': metrics.get('sample_count', 0),
            'alert_count_24h': len(alerts)
        }
    
    def _generate_charts(self, time_window: str) -> str:
        """Generate monitoring charts"""
        try:
            # Create charts directory
            charts_dir = Path("monitoring_charts")
            charts_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get recent metrics for plotting
            if time_window == "1h":
                cutoff = datetime.now() - timedelta(hours=1)
            elif time_window == "24h":
                cutoff = datetime.now() - timedelta(hours=24)
            else:
                cutoff = datetime.now() - timedelta(hours=1)
            
            recent_metrics = [
                m for m in self.metrics_buffer 
                if m.timestamp > cutoff
            ]
            
            if not recent_metrics:
                return ""
            
            # Convert to DataFrame
            data = []
            for m in recent_metrics:
                data.append({
                    'timestamp': m.timestamp,
                    'latency_ms': m.latency_ms,
                    'confidence_score': m.confidence_score,
                    'error_rate': m.error_rate,
                    'cpu_usage': m.cpu_usage,
                    'memory_usage_mb': m.memory_usage_mb
                })
            
            df = pd.DataFrame(data)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'ToW Model Monitoring Dashboard - {time_window}', fontsize=16)
            
            # Latency over time
            axes[0, 0].plot(df['timestamp'], df['latency_ms'])
            axes[0, 0].set_title('Response Latency')
            axes[0, 0].set_ylabel('Latency (ms)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Confidence score distribution
            axes[0, 1].hist(df['confidence_score'], bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('Confidence Score Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            
            # Resource usage over time
            axes[1, 0].plot(df['timestamp'], df['cpu_usage'], label='CPU %', alpha=0.7)
            axes[1, 0].plot(df['timestamp'], df['memory_usage_mb']/10, label='Memory (GB)', alpha=0.7)
            axes[1, 0].set_title('Resource Usage')
            axes[1, 0].set_ylabel('Usage')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Error rate over time
            axes[1, 1].plot(df['timestamp'], df['error_rate'], color='red', alpha=0.7)
            axes[1, 1].set_title('Error Rate')
            axes[1, 1].set_ylabel('Error Rate')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = f"monitoring_dashboard_{timestamp}.png"
            chart_path = charts_dir / chart_filename
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated monitoring charts: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
            return ""
    
    def start_monitoring(self) -> None:
        """Start background monitoring task"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started background monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring task"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Stopped background monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                current_metrics = PerformanceMetrics(
                    cpu_usage=self.get_cpu_usage(),
                    memory_usage_mb=self.get_memory_usage(),
                    gpu_usage=self.get_gpu_usage(),
                    gpu_memory_mb=self.get_gpu_memory_usage()
                )
                
                # Only add to buffer, don't trigger alerts for system-only metrics
                self.prom_resource_usage.labels('cpu').set(current_metrics.cpu_usage)
                self.prom_resource_usage.labels('memory').set(current_metrics.memory_usage_mb)
                self.prom_resource_usage.labels('gpu').set(current_metrics.gpu_usage)
                
                # Check for drift periodically (every 5 minutes)
                now = datetime.now()
                if (now - self.drift_detector.last_drift_check).total_seconds() > 300:  # 5 minutes
                    drift_results = self.check_model_drift()
                    if drift_results.get('drift_detected', False):
                        self.alert_manager.send_alert(
                            "model_drift",
                            f"Model drift detected: {drift_results}",
                            "warning",
                            drift_results
                        )
                
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.collection_interval)


def create_metrics_collector(config: MLOpsConfig) -> MetricsCollector:
    """Create metrics collector from config"""
    return MetricsCollector(config.monitoring)


def create_alert_manager(config: MLOpsConfig) -> AlertManager:
    """Create alert manager from config"""
    return AlertManager(config.monitoring)