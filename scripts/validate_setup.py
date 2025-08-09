#!/usr/bin/env python3
"""
Validation script to verify ToW research project setup.
Checks dependencies, model availability, and configuration.
"""

import sys
import importlib
import subprocess
from pathlib import Path
import torch

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scikit-learn", "Scikit-learn"),
        ("tqdm", "TQDM"),
        ("accelerate", "Accelerate"),
    ]
    
    optional_packages = [
        ("mlflow", "MLflow"),
        ("wandb", "Weights & Biases"),
        ("optuna", "Optuna"),
        ("fastapi", "FastAPI"),
        ("pytest", "PyTest"),
    ]
    
    print("\n📦 Checking required dependencies:")
    required_ok = True
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            required_ok = False
    
    print("\n🔧 Checking optional dependencies:")
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - Optional (not installed)")
    
    return required_ok

def check_cuda_availability():
    """Check CUDA availability for GPU training."""
    print("\n🖥️  GPU/CUDA Status:")
    if torch.cuda.is_available():
        print(f"✅ CUDA Available - Version: {torch.version.cuda}")
        print(f"✅ GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
        
        return True
    else:
        print("⚠️  CUDA not available - CPU only mode")
        return False

def check_project_structure():
    """Check if project directories exist."""
    print("\n📁 Checking project structure:")
    required_dirs = [
        "tow_architecture",
        "training", 
        "evaluation",
        "data",
        "mlops",
        "research_framework",
        "experiments",
        "scripts",
    ]
    
    structure_ok = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - Missing")
            structure_ok = False
    
    return structure_ok

def check_model_availability():
    """Check if downloaded models are available."""
    print("\n🤖 Checking model availability:")
    model_dirs = [
        "models/pretrained",
        "download_models",
    ]
    
    models_found = 0
    for model_dir in model_dirs:
        if Path(model_dir).exists():
            models = list(Path(model_dir).glob("*"))
            if models:
                print(f"✅ {model_dir}/ - {len(models)} items")
                models_found += len(models)
            else:
                print(f"⚠️  {model_dir}/ - Empty")
        else:
            print(f"⚠️  {model_dir}/ - Not found")
    
    return models_found > 0

def check_configuration():
    """Check configuration files."""
    print("\n⚙️  Checking configuration:")
    config_files = [
        (".env", "Environment variables", False),
        ("pyproject.toml", "Project configuration", True),
        ("requirements.txt", "Dependencies", True),
        ("setup.py", "Package setup", True),
    ]
    
    config_ok = True
    for filename, description, required in config_files:
        if Path(filename).exists():
            print(f"✅ {filename} - {description}")
        else:
            if required:
                print(f"❌ {filename} - {description} (Required)")
                config_ok = False
            else:
                print(f"⚠️  {filename} - {description} (Recommended)")
    
    return config_ok

def run_basic_test():
    """Run basic functionality test."""
    print("\n🧪 Running basic functionality test:")
    try:
        # Test ToW architecture import
        from tow_architecture import ToWEngine
        from tow_architecture.models import ModelAdapterFactory
        print("✅ ToW architecture import successful")
        
        # Test configuration loading
        from tow_architecture.utils.config import ToWConfig
        config = ToWConfig()
        print("✅ Configuration loading successful")
        
        return True
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def print_summary(checks):
    """Print validation summary."""
    print("\n" + "="*50)
    print("📋 VALIDATION SUMMARY")
    print("="*50)
    
    total_checks = len(checks)
    passed_checks = sum(1 for check, status in checks.items() if status)
    
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")
    
    print(f"\n📊 Overall Status: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\n🚀 Next steps:")
        print("1. Run: python examples/basic_tow_usage.py")
        print("2. Start training: python training/scripts/train.py")
        print("3. Run evaluation: python evaluation/scripts/evaluate.py")
        return True
    else:
        print(f"\n⚠️  {total_checks - passed_checks} issues found. Please fix them before proceeding.")
        print("\n🛠️  Troubleshooting:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Run setup script: python scripts/setup_environment.py")
        print("- Check GPU drivers if CUDA issues persist")
        return False

def main():
    print("🔍 ToW Research Project Validation")
    print("="*40)
    
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "CUDA/GPU": check_cuda_availability(),
        "Project Structure": check_project_structure(),
        "Model Availability": check_model_availability(),
        "Configuration": check_configuration(),
        "Basic Functionality": run_basic_test(),
    }
    
    success = print_summary(checks)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())