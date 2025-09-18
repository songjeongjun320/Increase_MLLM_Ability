"""
Configuration Loader Utility

Handles loading and managing configuration files for the multilingual analysis platform.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader and manager."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file
        """
        if config_path is None:
            # Default to config.yaml in the config directory
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            logger.info(f"Configuration loaded from {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            'models': {
                'sentence_transformer': {
                    'default_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                }
            },
            'languages': {
                'supported': [
                    {'code': 'en', 'name': 'English', 'color': '#1f77b4'},
                    {'code': 'ko', 'name': 'Korean', 'color': '#ff7f0e'}
                ],
                'default_pair': {'source': 'en', 'target': 'ko'}
            },
            'visualization': {
                'dimensionality_reduction': {
                    'methods': ['pca', 'tsne', 'umap'],
                    'default_method': 'umap'
                }
            },
            'performance': {
                'device': 'auto',
                'batch_size': 16
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_model_config(self, model_type: str = 'sentence_transformer') -> Dict[str, Any]:
        """Get model configuration."""
        return self.get(f'models.{model_type}', {})

    def get_language_config(self) -> Dict[str, Any]:
        """Get language configuration."""
        return self.get('languages', {})

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get('visualization', {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.get('performance', {})

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return self.get('languages.supported', [])

    def get_language_color(self, language_code: str) -> str:
        """Get color for a specific language."""
        languages = self.get_supported_languages()
        for lang in languages:
            if lang.get('code') == language_code:
                return lang.get('color', '#000000')
        return '#000000'  # Default black

    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")


# Global configuration instance
_global_config = None


def get_config() -> ConfigLoader:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    return _global_config


def reload_config():
    """Reload the global configuration."""
    global _global_config
    if _global_config is not None:
        _global_config.reload()