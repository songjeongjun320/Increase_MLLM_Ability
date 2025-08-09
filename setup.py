"""
Setup script for Thoughts of Words (ToW) Research Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tow-multilingual-llm",
    version="0.1.0",
    author="ToW Research Team",
    author_email="research@tow-project.org",
    description="Thoughts of Words: Improving Multilingual LLM Accuracy through English Cognitive Intermediary",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tow-project/multilingual-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pytest-cov>=3.0",
        ],
        "mlops": [
            "mlflow>=2.0",
            "wandb>=0.13",
            "optuna>=3.0",
            "kubernetes>=24.0",
            "prometheus-client>=0.14",
        ],
        "gpu": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "peft>=0.4.0",
            "bitsandbytes>=0.39.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "tow-train=training.scripts.train:main",
            "tow-evaluate=evaluation.scripts.evaluate:main",
            "tow-serve=mlops.scripts.deploy:serve",
        ],
    },
    package_data={
        "tow_architecture": ["configs/*.yaml", "templates/*.json"],
        "training": ["configs/*.yaml"],
        "evaluation": ["benchmarks/*.json"],
    },
    include_package_data=True,
)