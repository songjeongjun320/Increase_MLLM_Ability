"""
Dashboard Runner Script

Easy script to launch the Multilingual Language Model Analysis Platform.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "dashboard.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_streamlit():
    """Check if streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements if they don't exist."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✅ Requirements installed successfully!")
    else:
        print("⚠️ requirements.txt not found. Please install dependencies manually.")

def run_streamlit_dashboard():
    """Run the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard" / "main_dashboard.py"

    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False

    print("🚀 Starting Multilingual Analysis Platform...")
    print("📊 Dashboard will open in your web browser")
    print("🌐 URL: http://localhost:8501")
    print("\n🎯 Platform Features:")
    print("   • Sentence Embedding Analysis (Sentence-BERT)")
    print("   • Attention Pattern Visualization")
    print("   • Token Prediction Confidence Analysis")
    print("   • English-Korean Comparison Tools")
    print("   • Base vs Training Model Comparison")
    print("   • Interactive Visualizations")
    print("\n💡 Quick Start:")
    print("   1. Use sidebar to input text or upload files")
    print("   2. Select languages (English & Korean supported)")
    print("   3. Choose base model and optionally add training model")
    print("   4. Click 'Run Analysis' to generate insights")
    print("   5. Explore different tabs for various analyses")
    print("\n" + "="*60)

    try:
        # Run streamlit with the dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]

        subprocess.run(cmd)
        return True

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return False

def main():
    """Main function."""
    logger = setup_logging()

    print("🌍 Multilingual Language Model Analysis Platform")
    print("=" * 60)

    # Check if streamlit is installed
    if not check_streamlit():
        print("📦 Streamlit not found. Installing requirements...")
        try:
            install_requirements()
        except Exception as e:
            print(f"❌ Failed to install requirements: {e}")
            print("Please install manually with: pip install -r requirements.txt")
            return False

    # Run the dashboard
    success = run_streamlit_dashboard()

    if success:
        print("✅ Dashboard session completed successfully!")
    else:
        print("❌ Dashboard failed to start properly")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)