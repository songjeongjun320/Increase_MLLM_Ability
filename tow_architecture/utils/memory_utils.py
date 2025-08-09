"""
Memory Utilities - GPU Memory Management
=======================================

This module provides utilities for managing GPU memory,
clearing caches, and monitoring memory usage across
the ToW architecture system.
"""

import gc
import logging
from typing import Dict, List, Optional, Any

import torch

from .logger import get_logger

logger = get_logger(__name__)


def clear_gpu_memory(device_ids: Optional[List[int]] = None):
    """
    Clear GPU memory across specified devices.
    
    Args:
        device_ids: List of GPU device IDs to clear. If None, clears all available GPUs.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU memory clear")
        return
    
    # Clear Python garbage collection first
    gc.collect()
    
    # Determine devices to clear
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    # Clear memory for each device
    for device_id in device_ids:
        if device_id < torch.cuda.device_count():
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Also clear IPC memory if available
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
        else:
            logger.warning(f"Device ID {device_id} not available")
    
    logger.info(f"Cleared GPU memory for devices: {device_ids}")


def get_memory_stats(device_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Get comprehensive memory statistics for GPU devices.
    
    Args:
        device_ids: List of GPU device IDs to query. If None, queries all available GPUs.
        
    Returns:
        Dictionary containing memory statistics for each device
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # Determine devices to query
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    stats = {
        "total_devices": torch.cuda.device_count(),
        "devices": {},
        "summary": {}
    }
    
    total_allocated = 0
    total_reserved = 0
    total_memory = 0
    
    for device_id in device_ids:
        if device_id >= torch.cuda.device_count():
            continue
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(device_id)
        
        # Get memory info
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        total_device_memory = device_props.total_memory
        
        # Convert to GB
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        total_gb = total_device_memory / (1024**3)
        
        # Calculate usage percentages
        allocated_percent = (allocated / total_device_memory) * 100
        reserved_percent = (reserved / total_device_memory) * 100
        
        # Get memory summary if available (PyTorch 1.4+)
        memory_summary = {}
        if hasattr(torch.cuda, 'memory_summary'):
            try:
                memory_summary = torch.cuda.memory_stats(device_id)
            except Exception as e:
                logger.warning(f"Failed to get detailed memory stats: {e}")
        
        stats["devices"][f"gpu_{device_id}"] = {
            "name": device_props.name,
            "total_memory_gb": total_gb,
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "free_gb": total_gb - reserved_gb,
            "allocated_percent": allocated_percent,
            "reserved_percent": reserved_percent,
            "compute_capability": f"{device_props.major}.{device_props.minor}",
            "multiprocessor_count": device_props.multi_processor_count,
            "memory_summary": memory_summary
        }
        
        # Accumulate totals
        total_allocated += allocated_gb
        total_reserved += reserved_gb
        total_memory += total_gb
    
    # Calculate summary statistics
    stats["summary"] = {
        "total_memory_gb": total_memory,
        "total_allocated_gb": total_allocated,
        "total_reserved_gb": total_reserved,
        "total_free_gb": total_memory - total_reserved,
        "average_utilization_percent": (total_reserved / total_memory * 100) if total_memory > 0 else 0,
        "device_count": len(stats["devices"])
    }
    
    return stats


def monitor_memory_usage(
    threshold_percent: float = 80.0,
    auto_clear: bool = True,
    device_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Monitor memory usage and optionally clear memory when threshold is exceeded.
    
    Args:
        threshold_percent: Memory usage threshold percentage for warnings
        auto_clear: Whether to automatically clear memory when threshold exceeded
        device_ids: List of GPU device IDs to monitor
        
    Returns:
        Dictionary with monitoring results and actions taken
    """
    stats = get_memory_stats(device_ids)
    
    if "error" in stats:
        return stats
    
    results = {
        "status": "ok",
        "warnings": [],
        "actions_taken": [],
        "memory_stats": stats
    }
    
    # Check each device against threshold
    for device_name, device_stats in stats["devices"].items():
        usage_percent = device_stats["reserved_percent"]
        
        if usage_percent > threshold_percent:
            warning_msg = f"{device_name}: {usage_percent:.1f}% memory usage exceeds threshold ({threshold_percent}%)"
            results["warnings"].append(warning_msg)
            results["status"] = "warning"
            
            logger.warning(warning_msg)
            
            # Auto-clear if enabled
            if auto_clear:
                device_id = int(device_name.split('_')[1])
                clear_gpu_memory([device_id])
                results["actions_taken"].append(f"Cleared memory for {device_name}")
    
    return results


def estimate_model_memory_requirements(
    parameter_count: str,
    precision: str = "fp16",
    quantization: Optional[str] = None,
    overhead_factor: float = 1.3
) -> Dict[str, float]:
    """
    Estimate memory requirements for loading a model.
    
    Args:
        parameter_count: Parameter count (e.g., "7B", "13B", "70B")
        precision: Model precision ("fp32", "fp16", "bf16")
        quantization: Quantization method ("4bit", "8bit", None)
        overhead_factor: Factor for additional memory overhead
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Parse parameter count
    param_str = parameter_count.upper()
    if param_str.endswith('B'):
        params = float(param_str[:-1]) * 1e9
    elif param_str.endswith('M'):
        params = float(param_str[:-1]) * 1e6
    else:
        try:
            params = float(param_str)
        except ValueError:
            raise ValueError(f"Invalid parameter count format: {parameter_count}")
    
    # Calculate base memory per parameter
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    if quantization == "4bit":
        bytes_per_param = precision_bytes["int4"]
    elif quantization == "8bit":
        bytes_per_param = precision_bytes["int8"]
    else:
        bytes_per_param = precision_bytes.get(precision, 2)
    
    # Calculate memory requirements
    base_memory_gb = (params * bytes_per_param) / (1024**3)
    
    # Add overhead for activations, gradients, optimizer states, etc.
    total_memory_gb = base_memory_gb * overhead_factor
    
    # Estimate KV cache memory (rough approximation)
    kv_cache_gb = base_memory_gb * 0.1  # ~10% of model size for reasonable context
    
    return {
        "base_model_gb": base_memory_gb,
        "total_estimated_gb": total_memory_gb,
        "kv_cache_gb": kv_cache_gb,
        "recommended_gpu_memory_gb": total_memory_gb + kv_cache_gb,
        "parameters": params,
        "bytes_per_parameter": bytes_per_param,
        "precision": precision,
        "quantization": quantization
    }


def optimize_memory_allocation(
    available_devices: List[int],
    model_memory_gb: float,
    buffer_gb: float = 2.0
) -> Dict[str, Any]:
    """
    Optimize memory allocation across available devices.
    
    Args:
        available_devices: List of available GPU device IDs
        model_memory_gb: Required memory for the model in GB
        buffer_gb: Buffer memory to maintain free in GB
        
    Returns:
        Dictionary with allocation strategy
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # Get memory stats for available devices
    device_memory = {}
    total_available = 0
    
    for device_id in available_devices:
        if device_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_id)
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            total = props.total_memory / (1024**3)
            available = total - allocated - buffer_gb
            
            device_memory[device_id] = {
                "total_gb": total,
                "allocated_gb": allocated,
                "available_gb": max(0, available),
                "device_name": props.name
            }
            
            total_available += max(0, available)
    
    # Determine allocation strategy
    strategy = {
        "total_required_gb": model_memory_gb,
        "total_available_gb": total_available,
        "feasible": total_available >= model_memory_gb,
        "device_allocation": {},
        "recommendation": ""
    }
    
    if not strategy["feasible"]:
        strategy["recommendation"] = f"Insufficient memory. Need {model_memory_gb:.1f}GB, have {total_available:.1f}GB"
        return strategy
    
    # Single device allocation if possible
    for device_id, info in device_memory.items():
        if info["available_gb"] >= model_memory_gb:
            strategy["device_allocation"] = {device_id: model_memory_gb}
            strategy["recommendation"] = f"Single device allocation on GPU {device_id}"
            return strategy
    
    # Multi-device allocation
    remaining_memory = model_memory_gb
    allocated_devices = {}
    
    # Sort devices by available memory (descending)
    sorted_devices = sorted(
        device_memory.items(),
        key=lambda x: x[1]["available_gb"],
        reverse=True
    )
    
    for device_id, info in sorted_devices:
        if remaining_memory <= 0:
            break
        
        available = info["available_gb"]
        if available > 0:
            allocation = min(remaining_memory, available)
            allocated_devices[device_id] = allocation
            remaining_memory -= allocation
    
    strategy["device_allocation"] = allocated_devices
    strategy["recommendation"] = f"Multi-device allocation across {len(allocated_devices)} GPUs"
    
    return strategy


def get_memory_efficient_batch_size(
    model_memory_gb: float,
    available_memory_gb: float,
    sequence_length: int,
    bytes_per_token: int = 2,
    safety_factor: float = 0.7
) -> int:
    """
    Estimate optimal batch size for memory-efficient processing.
    
    Args:
        model_memory_gb: Model memory usage in GB
        available_memory_gb: Available GPU memory in GB
        sequence_length: Average sequence length in tokens
        bytes_per_token: Memory bytes per token for activations
        safety_factor: Safety factor for memory estimation
        
    Returns:
        Recommended batch size
    """
    # Available memory for batch processing
    batch_memory_gb = (available_memory_gb - model_memory_gb) * safety_factor
    
    if batch_memory_gb <= 0:
        return 1  # Minimum batch size
    
    # Estimate memory per sample
    bytes_per_sample = sequence_length * bytes_per_token
    memory_per_sample_gb = bytes_per_sample / (1024**3)
    
    # Calculate batch size
    if memory_per_sample_gb <= 0:
        return 32  # Default reasonable batch size
    
    batch_size = int(batch_memory_gb / memory_per_sample_gb)
    
    # Ensure reasonable bounds
    return max(1, min(batch_size, 128))


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations"""
    
    def __init__(
        self,
        name: str = "Operation",
        log_usage: bool = True,
        clear_on_exit: bool = False
    ):
        """
        Initialize memory monitor.
        
        Args:
            name: Name of the operation being monitored
            log_usage: Whether to log memory usage
            clear_on_exit: Whether to clear memory on context exit
        """
        self.name = name
        self.log_usage = log_usage
        self.clear_on_exit = clear_on_exit
        self.start_stats = None
        self.end_stats = None
    
    def __enter__(self):
        """Enter context and record initial memory state"""
        if torch.cuda.is_available():
            self.start_stats = get_memory_stats()
            if self.log_usage:
                logger.info(f"Memory usage before {self.name}:")
                self._log_memory_summary(self.start_stats)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record final memory state"""
        if torch.cuda.is_available():
            self.end_stats = get_memory_stats()
            
            if self.log_usage:
                logger.info(f"Memory usage after {self.name}:")
                self._log_memory_summary(self.end_stats)
                
                # Calculate and log memory difference
                self._log_memory_difference()
            
            if self.clear_on_exit:
                clear_gpu_memory()
                logger.info(f"Cleared GPU memory after {self.name}")
    
    def _log_memory_summary(self, stats: Dict[str, Any]):
        """Log memory usage summary"""
        if "summary" in stats:
            summary = stats["summary"]
            logger.info(
                f"  Total: {summary['total_memory_gb']:.1f}GB, "
                f"Reserved: {summary['total_reserved_gb']:.1f}GB "
                f"({summary['average_utilization_percent']:.1f}%)"
            )
    
    def _log_memory_difference(self):
        """Log memory usage difference"""
        if not self.start_stats or not self.end_stats:
            return
        
        start_reserved = self.start_stats["summary"]["total_reserved_gb"]
        end_reserved = self.end_stats["summary"]["total_reserved_gb"]
        difference = end_reserved - start_reserved
        
        if abs(difference) > 0.1:  # Only log significant differences
            direction = "increased" if difference > 0 else "decreased"
            logger.info(f"Memory usage {direction} by {abs(difference):.1f}GB during {self.name}")
    
    def get_memory_delta(self) -> Optional[Dict[str, float]]:
        """Get memory usage delta during the operation"""
        if not self.start_stats or not self.end_stats:
            return None
        
        return {
            "start_reserved_gb": self.start_stats["summary"]["total_reserved_gb"],
            "end_reserved_gb": self.end_stats["summary"]["total_reserved_gb"],
            "delta_gb": (
                self.end_stats["summary"]["total_reserved_gb"] - 
                self.start_stats["summary"]["total_reserved_gb"]
            )
        }