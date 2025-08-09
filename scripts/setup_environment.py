#!/usr/bin/env python3
"""
Environment setup script for ToW research project.
Sets up development environment, downloads models, and prepares data.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_directories():
    """Create necessary project directories."""
    directories = [
        "experiments/logs",
        "experiments/checkpoints", 
        "experiments/results",
        "data/raw",
        "data/processed",
        "data/benchmarks",
        "models/pretrained",
        "models/fine_tuned",
        "outputs/evaluations",
        "outputs/visualizations",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies(mode="base"):
    """Install project dependencies based on mode."""
    if mode == "base":
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    elif mode == "dev":
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=True)
    elif mode == "full":
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev,mlops,gpu]"], check=True)
    
    print(f"✅ Installed {mode} dependencies")

def setup_git_hooks():
    """Setup pre-commit hooks for code quality."""
    try:
        subprocess.run(["pre-commit", "install"], check=True)
        print("✅ Pre-commit hooks installed")
    except FileNotFoundError:
        print("⚠️  pre-commit not found. Install with: pip install pre-commit")

def download_sample_models():
    """Download sample models for testing."""
    model_scripts = [
        "download_models/download_qwen25_7b_instruct.py",
        "download_models/download_mistral_8b_ins.py",
    ]
    
    for script in model_scripts:
        if Path(script).exists():
            print(f"🔄 Running {script}...")
            subprocess.run([sys.executable, script], cwd=".")
    
    print("✅ Sample models downloaded")

def create_config_files():
    """Create default configuration files."""
    configs = {
        ".env.example": """
# ToW Research Project Environment Variables
WANDB_API_KEY=your_wandb_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_key_here
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
""",
        "pyproject.toml": """
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=tow_architecture --cov=training --cov=evaluation --cov-report=term-missing"
""",
        ".pre-commit-config.yaml": """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203,W503]
"""
    }
    
    for filename, content in configs.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Created {filename}")

def main():
    parser = argparse.ArgumentParser(description="Setup ToW research environment")
    parser.add_argument(
        "--mode", 
        choices=["base", "dev", "full"], 
        default="dev",
        help="Installation mode (base/dev/full)"
    )
    parser.add_argument(
        "--skip-models", 
        action="store_true",
        help="Skip model downloads"
    )
    
    args = parser.parse_args()
    
    print("🚀 Setting up ToW Research Environment...")
    
    # Setup project structure
    setup_directories()
    
    # Install dependencies
    install_dependencies(args.mode)
    
    # Create configuration files
    create_config_files()
    
    # Setup development tools
    if args.mode in ["dev", "full"]:
        setup_git_hooks()
    
    # Download models
    if not args.skip_models:
        download_sample_models()
    
    print("\n✅ Environment setup complete!")
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env and fill in your API keys")
    print("2. Run: python scripts/validate_setup.py")
    print("3. Start development: python examples/basic_tow_usage.py")

if __name__ == "__main__":
    main()