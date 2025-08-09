# Experiments Directory

This directory contains experiment configurations, results, and tracking for ToW research.

## Structure

```
experiments/
├── configs/                 # Experiment configuration files
│   ├── baseline/           # Baseline model configurations
│   ├── tow_training/       # ToW training configurations
│   └── evaluation/         # Evaluation configurations
├── logs/                   # Experiment logs and outputs
│   ├── training/          # Training logs
│   ├── evaluation/        # Evaluation logs
│   └── debugging/         # Debug outputs
├── checkpoints/           # Model checkpoints and saved states
│   ├── baseline/          # Baseline model checkpoints
│   └── tow_models/        # ToW-enhanced model checkpoints
├── results/               # Experiment results and analyses
│   ├── metrics/           # Performance metrics
│   ├── visualizations/    # Charts and plots
│   └── reports/           # Detailed analysis reports
└── tracking/              # MLflow and experiment tracking
    ├── mlruns/            # MLflow runs
    └── wandb/             # Weights & Biases logs
```

## Experiment Types

### 1. Baseline Experiments
- Standard multilingual model evaluation
- Performance benchmarks on KLUE, MMLU, etc.
- Cross-lingual transfer analysis

### 2. ToW Training Experiments  
- Fine-tuning with ToW methodology
- Ablation studies on thought token types
- Hyperparameter optimization

### 3. Evaluation Experiments
- Multilingual accuracy assessment
- Cultural appropriateness evaluation
- Human evaluation studies

### 4. Comparative Analysis
- ToW vs. baseline comparisons
- Different model architecture comparisons
- Language-specific performance analysis

## Experiment Configuration

Each experiment uses YAML configuration files:

```yaml
experiment:
  name: "tow_korean_translation"
  description: "ToW methodology for English-Korean translation"
  
model:
  base_model: "deepseek-ai/deepseek-llm-70b"
  adapter_type: "lora"
  
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 3
  
evaluation:
  benchmarks: ["klue", "custom_translation"]
  metrics: ["bleu", "rouge", "bert_score"]
```

## Running Experiments

### Basic Training Experiment
```bash
python training/scripts/train.py \
  --config experiments/configs/tow_training/korean_translation.yaml \
  --output-dir experiments/checkpoints/tow_models/
```

### Evaluation Experiment
```bash
python evaluation/scripts/evaluate.py \
  --model-path experiments/checkpoints/tow_models/best_model \
  --config experiments/configs/evaluation/multilingual_eval.yaml
```

### Batch Experiment Execution
```bash
python scripts/run_experiments.py \
  --config-dir experiments/configs/batch/ \
  --output-dir experiments/results/
```

## Experiment Tracking

### MLflow Integration
- Automatic experiment logging
- Parameter and metric tracking
- Model registry and versioning
- Results visualization

### Weights & Biases Integration  
- Real-time training monitoring
- Hyperparameter sweeps
- Collaborative experiment sharing
- Advanced visualization

### Custom Tracking
- Experiment metadata storage
- Result aggregation and comparison
- Performance trend analysis
- Resource usage monitoring

## Results Organization

### Metrics Storage
```
results/metrics/
├── training_metrics.json     # Training loss, accuracy curves
├── evaluation_metrics.json   # Benchmark evaluation results
├── multilingual_scores.json  # Per-language performance
└── cultural_metrics.json     # Cultural appropriateness scores
```

### Visualization
```
results/visualizations/
├── training_curves.png       # Loss and accuracy plots
├── language_comparison.png   # Multi-language performance
├── cultural_analysis.png     # Cultural appropriateness charts
└── ablation_studies.png      # Component contribution analysis
```

## Reproducibility

All experiments include:
- **Environment specification** (requirements.txt, conda environment)
- **Random seed setting** for reproducible results
- **Data versioning** with checksums and lineage
- **Code versioning** with git commits and branches
- **Configuration storage** with all hyperparameters

## Best Practices

### Experiment Naming
Use descriptive names with versioning:
- `tow_korean_v1.0` - Initial Korean ToW experiment
- `baseline_multilingual_v2.1` - Baseline comparison v2.1
- `ablation_thought_types_v1.0` - Thought type ablation study

### Resource Management
- Monitor GPU memory usage
- Use checkpointing for long experiments  
- Clean up old experiments regularly
- Archive completed experiments

### Documentation
- Document experiment purpose and hypothesis
- Record unexpected findings and insights
- Maintain experiment logs and debugging notes
- Update documentation with results

## Analysis and Reporting

### Automated Reports
```bash
python scripts/generate_report.py \
  --experiment-dir experiments/results/tow_korean_v1.0 \
  --output experiments/reports/tow_korean_analysis.html
```

### Comparative Analysis
```bash
python scripts/compare_experiments.py \
  --baseline experiments/results/baseline_v1.0 \
  --treatment experiments/results/tow_korean_v1.0 \
  --output experiments/reports/comparison.html
```

## Integration with Research Pipeline

Experiments integrate with:
- **Training Pipeline** - Automated model training and evaluation
- **MLOps Pipeline** - Model deployment and monitoring
- **Research Framework** - Academic analysis and publication
- **Quality Assurance** - Validation and testing procedures

This structured approach ensures reproducible, well-documented research that supports the ToW methodology development and validation.