"""
Main Application Entry Point

Launch the Multilingual Language Model Analysis Platform.
"""

import sys
import os
from pathlib import Path
import logging

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'logs',
        'cache',
        'outputs',
        'outputs/plots',
        'outputs/exports',
        'outputs/reports'
    ]

    for directory in directories:
        dir_path = current_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'sentence_transformers',
        'numpy',
        'pandas',
        'matplotlib',
        'plotly',
        'seaborn',
        'scikit_learn',
        'umap_learn'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('_', '-'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        print(f"Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    logger.info("All required dependencies are installed")
    return True


def main():
    """Main function to launch the application."""
    print("🌍 Multilingual Language Model Analysis Platform")
    print("=" * 50)

    # Setup
    setup_directories()

    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages.")
        return

    # Import dashboard after dependency check
    try:
        from dashboard.main_dashboard import main as dashboard_main
        logger.info("Starting Streamlit dashboard...")
        print("🚀 Starting the analysis platform...")
        print("📊 Dashboard will open in your web browser")
        print("🔗 URL: http://localhost:8501")
        print("\n💡 Usage Tips:")
        print("   1. Use the sidebar to input texts and select languages")
        print("   2. Choose between manual entry, file upload, or sample data")
        print("   3. Click 'Run Analysis' to generate insights")
        print("   4. Explore different tabs for various analysis types")
        print("\n🎯 Features:")
        print("   • Sentence embedding visualization (PCA/t-SNE/UMAP)")
        print("   • Attention pattern analysis and heatmaps")
        print("   • Token prediction confidence analysis")
        print("   • Cross-language and model comparison")
        print("   • Interactive visualizations and exports")

        dashboard_main()

    except ImportError as e:
        logger.error(f"Failed to import dashboard: {e}")
        print(f"❌ Failed to start dashboard: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()