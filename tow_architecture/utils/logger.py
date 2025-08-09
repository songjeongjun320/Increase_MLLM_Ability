"""
Logging Utilities - Centralized Logging Configuration
====================================================

This module provides centralized logging configuration and utilities
for the ToW architecture system, ensuring consistent logging across
all components.
"""

import logging
import sys
import os
from typing import Optional
from datetime import datetime
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
        format_string: Custom format string
        include_timestamp: Whether to include timestamp in log file names
        
    Returns:
        Configured root logger
    """
    # Create log directory
    if log_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(log_file)
            log_file = f"{name}_{timestamp}{ext}"
        
        log_path = os.path.join(log_dir, log_file) if log_dir else log_file
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_path}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_transformers_logging(level: str = "WARNING"):
    """
    Configure logging for transformers library to reduce noise.
    
    Args:
        level: Logging level for transformers
    """
    # Reduce transformers logging noise
    transformers_loggers = [
        "transformers.modeling_utils",
        "transformers.configuration_utils", 
        "transformers.tokenization_utils_base",
        "transformers.generation_utils"
    ]
    
    for logger_name in transformers_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


def configure_torch_logging(level: str = "WARNING"):
    """
    Configure logging for PyTorch to reduce noise.
    
    Args:
        level: Logging level for PyTorch
    """
    torch_loggers = [
        "torch",
        "torch.nn",
        "torch.cuda"
    ]
    
    for logger_name in torch_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


class ToWLogger:
    """
    Specialized logger for ToW architecture with additional functionality.
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Initialize ToW logger.
        
        Args:
            name: Logger name
            log_file: Optional log file for this specific logger
        """
        self.logger = logging.getLogger(name)
        self.name = name
        
        if log_file:
            self._add_file_handler(log_file)
    
    def _add_file_handler(self, log_file: str):
        """Add file handler to logger"""
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_model_info(self, model_info: dict):
        """Log model information"""
        self.logger.info("=" * 50)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 50)
        
        for key, value in model_info.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 50)
    
    def log_performance_metrics(self, metrics: dict):
        """Log performance metrics"""
        self.logger.info("PERFORMANCE METRICS")
        self.logger.info("-" * 30)
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"{metric}: {value:.4f}")
            else:
                self.logger.info(f"{metric}: {value}")
    
    def log_memory_usage(self, memory_stats: dict):
        """Log memory usage statistics"""
        self.logger.info("MEMORY USAGE")
        self.logger.info("-" * 20)
        
        for device, stats in memory_stats.items():
            if isinstance(stats, dict):
                self.logger.info(f"{device}:")
                for key, value in stats.items():
                    if "gb" in key.lower():
                        self.logger.info(f"  {key}: {value:.2f}")
                    elif "percent" in key.lower():
                        self.logger.info(f"  {key}: {value:.1f}%")
                    else:
                        self.logger.info(f"  {key}: {value}")
    
    def log_processing_stage(self, stage: str, duration: float, details: dict = None):
        """Log processing stage completion"""
        self.logger.info(f"Stage '{stage}' completed in {duration:.2f}s")
        
        if details:
            for key, value in details.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_error_with_context(self, error: Exception, context: dict = None):
        """Log error with additional context"""
        self.logger.error(f"Error: {str(error)}")
        self.logger.error(f"Error type: {type(error).__name__}")
        
        if context:
            self.logger.error("Context:")
            for key, value in context.items():
                self.logger.error(f"  {key}: {value}")
    
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info level logging"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error level logging"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical level logging"""
        self.logger.critical(message, **kwargs)


def create_tow_logger(name: str, log_file: Optional[str] = None) -> ToWLogger:
    """
    Create a ToW-specific logger instance.
    
    Args:
        name: Logger name
        log_file: Optional log file
        
    Returns:
        ToWLogger instance
    """
    return ToWLogger(name, log_file)


def setup_tow_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    quiet_external_libs: bool = True
) -> logging.Logger:
    """
    Setup complete logging configuration for ToW system.
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging
        quiet_external_libs: Whether to reduce noise from external libraries
        
    Returns:
        Configured root logger
    """
    # Setup basic logging
    log_file = "tow_system.log" if enable_file_logging else None
    root_logger = setup_logging(
        level=level,
        log_file=log_file,
        log_dir=log_dir
    )
    
    # Quiet external libraries if requested
    if quiet_external_libs:
        configure_transformers_logging("WARNING")
        configure_torch_logging("WARNING")
        
        # Additional noisy libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
    root_logger.info("ToW logging system initialized")
    return root_logger