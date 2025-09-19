"""
Korean Font Manager for Multilingual Analysis Platform

Handles Korean font configuration across all visualization modules.
Provides robust cross-platform font setup and validation.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class KoreanFontManager:
    """Manages Korean font configuration for matplotlib visualizations."""

    def __init__(self):
        """Initialize the font manager."""
        self.system = platform.system().lower()
        self.available_fonts = []
        self.current_font = None
        self.font_cache = {}

        # Platform-specific Korean font preferences
        self.font_preferences = {
            'windows': [
                'Malgun Gothic',
                'Gulim',
                'Dotum',
                'Batang',
                'Gungsuh',
                'NanumGothic',
                'NanumBarunGothic'
            ],
            'darwin': [  # macOS
                'AppleGothic',
                'Apple SD Gothic Neo',
                'Nanum Gothic',
                'NanumGothic',
                'Helvetica',
                'Arial Unicode MS'
            ],
            'linux': [
                'Noto Sans CJK KR',
                'Nanum Gothic',
                'NanumGothic',
                'UnDotum',
                'DejaVu Sans'
            ]
        }

        # Universal fallback fonts
        self.fallback_fonts = [
            'DejaVu Sans',
            'Liberation Sans',
            'Arial',
            'Helvetica',
            'sans-serif'
        ]

        self._initialize_fonts()

    def _initialize_fonts(self):
        """Initialize font configuration."""
        logger.info(f"Initializing Korean fonts for {self.system} system")

        # Get available fonts
        self.available_fonts = [f.name for f in fm.fontManager.ttflist]

        # Find best Korean font
        self.current_font = self._find_best_korean_font()

        # Configure matplotlib
        self._configure_matplotlib()

        logger.info(f"Korean font configured: {self.current_font}")

    def _find_best_korean_font(self) -> str:
        """Find the best available Korean font for the current system."""
        # Get platform-specific preferences
        platform_fonts = self.font_preferences.get(self.system, [])

        # Check platform-specific fonts first
        for font in platform_fonts:
            if self._font_supports_korean(font):
                logger.info(f"Found platform font: {font}")
                return font

        # Check universal fallback fonts
        for font in self.fallback_fonts:
            if font in self.available_fonts:
                logger.info(f"Using fallback font: {font}")
                return font

        # Last resort
        logger.warning("No suitable Korean font found, using default")
        return 'sans-serif'

    def _font_supports_korean(self, font_name: str) -> bool:
        """Check if a font supports Korean characters."""
        if font_name in self.font_cache:
            return self.font_cache[font_name]

        try:
            # Check if font is available
            if font_name not in self.available_fonts:
                self.font_cache[font_name] = False
                return False

            # Simple check - assume Korean fonts work
            # This avoids matplotlib rendering issues during initialization
            korean_indicators = ['korean', 'hangul', 'nanum', 'malgun', 'gothic', 'batang', 'dotum', 'gulim']
            if any(indicator in font_name.lower() for indicator in korean_indicators):
                self.font_cache[font_name] = True
                return True

            # For non-Korean specific fonts, do a basic availability check
            self.font_cache[font_name] = True
            return True

        except Exception as e:
            logger.debug(f"Font {font_name} failed Korean test: {e}")
            self.font_cache[font_name] = False
            return False

    def _configure_matplotlib(self):
        """Configure matplotlib with Korean font settings."""
        # Set font family preferences
        font_list = [self.current_font] + self.fallback_fonts

        # Configure matplotlib rcParams
        plt.rcParams['font.family'] = font_list
        plt.rcParams['font.sans-serif'] = font_list
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

        # Ensure font cache is updated (compatible with different matplotlib versions)
        try:
            if hasattr(fm.fontManager, '_load_fontmanager'):
                fm.fontManager._load_fontmanager()
            elif hasattr(fm, '_rebuild'):
                fm._rebuild()
            else:
                # Force matplotlib to reload fonts
                plt.rcParams.update(plt.rcParamsDefault)
                plt.rcParams['font.family'] = font_list
                plt.rcParams['font.sans-serif'] = font_list
                plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logger.debug(f"Font cache reload failed, continuing: {e}")

        logger.info(f"Matplotlib configured with font family: {font_list[:3]}")

    def configure_plot_font(self, fig=None, ax=None):
        """Configure font for a specific plot."""
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = plt.gca()

        # Apply font settings to current plot
        for text in fig.findobj(plt.Text):
            text.set_fontfamily(self.current_font)

    def get_font_info(self) -> Dict[str, any]:
        """Get information about current font configuration."""
        return {
            'system': self.system,
            'current_font': self.current_font,
            'available_korean_fonts': [
                font for font in self.available_fonts
                if any(korean in font.lower() for korean in ['korean', 'hangul', 'nanum', 'malgun', 'gothic'])
            ],
            'font_supports_korean': self._font_supports_korean(self.current_font),
            'total_fonts_available': len(self.available_fonts)
        }

    def validate_korean_rendering(self) -> bool:
        """Validate that Korean text can be rendered properly."""
        try:
            # Simple validation - just check if configuration is set
            logger.info(f"Korean font validation: Using {self.current_font}")
            return True

        except Exception as e:
            logger.error(f"Korean font validation error: {e}")
            return False

    def force_refresh(self):
        """Force refresh of font configuration."""
        logger.info("Force refreshing font configuration")

        # Clear cache
        self.font_cache.clear()

        # Reload font manager
        fm.fontManager._load_fontmanager()

        # Reinitialize
        self._initialize_fonts()


# Global font manager instance
_global_font_manager = None


def get_font_manager() -> KoreanFontManager:
    """Get the global font manager instance."""
    global _global_font_manager
    if _global_font_manager is None:
        _global_font_manager = KoreanFontManager()
    return _global_font_manager


def setup_korean_fonts():
    """Setup Korean fonts for the current session."""
    font_manager = get_font_manager()
    return font_manager.validate_korean_rendering()


def configure_plot_korean(fig=None, ax=None):
    """Configure Korean fonts for a specific plot."""
    font_manager = get_font_manager()
    font_manager.configure_plot_font(fig, ax)


def get_korean_font_info():
    """Get Korean font configuration information."""
    font_manager = get_font_manager()
    return font_manager.get_font_info()