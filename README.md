# Increase MLLM Ability - TOW (Thoughts of Words) Implementation

## Overview

This repository implements **Option 2: Pure Original TOW Implementation** for enhancing Korean multilingual large language models through cross-lingual reasoning.

### Key Features

- **Cross-lingual TOW Approach**: Korean stories processed with English-only reasoning in `<ToW>...</ToW>` tokens
- **Data Augmentation Pipeline**: Converts Korean story corpus into TOW-enhanced training data  
- **Token Classification System**: Categorizes words as trivial/exact/soft/unpredictable
- **Comprehensive Evaluation**: KMMLU, KLUE (8 tasks), and EN→KR translation benchmarks
- **Before/After Comparison**: Proves TOW effectiveness through quantitative analysis
- **Zhikun et al. (2024) Methodology**: Official ToW paper implementation with 5-shot prompting

## Implementation Workflow

```
Korean Story Input → GPT-OSS Model → English ToW Generation → Enhanced Korean Output
```

### Example TOW Flow

```python
# Input Korean text
korean_input = "브루스 리는 쿵푸 영화의 전설적인 인물이다. 그는 홍콩에서 태어나 무술을"

# Generated output with English TOW
output = """브루스 리는 쿵푸 영화의 전설적인 인물이다. 그는 홍콩에서 태어나 무술을 <ToW>Given the context about Bruce Lee being a legendary figure in kung fu movies and being born in Hong Kong, the next word should logically be about learning martial arts, which is '배웠다' (learned).</ToW> 배웠다."""
```

## Project Structure

```
├── 1_models/                    # Model downloads and storage
│   ├── download_gpt_oss_20b.py     # GPT-OSS-20B for ToW generation (16GB GPU)
│   ├── download_gpt_oss_120b.py    # GPT-OSS-120B for ToW generation (80GB GPU)
│   ├── download_deepseek_r1_distill_qwen_7b.py  # Base model for training
│   └── download_qwen25_7b_instruct.py           # Base model for training
│
├── 2_datasets/                  # Evaluation and training datasets
│   ├── download_korean_datasets.py # Korean benchmarks downloader
│   ├── benchmarks/              # KMMLU, KLUE, translation data
│   └── korean_stories/          # Korean story corpus for ToW augmentation
│
├── 3_evaluation/                # Evaluation and comparison
│   ├── baseline_evaluation.py   # Zero-shot baseline evaluation
│   └── compare_baseline_vs_tow.py # Before/after comparison with statistical analysis
│
├── 4_tow_generation/            # ToW data generation
│   ├── korean_tow_generator.py  # Main ToW generation pipeline using GPT-OSS
│   ├── tow_prompt_generator.py  # Zhikun et al. (2024) 5-shot methodology
│   └── utils/
│       └── text_processing.py   # English enforcement utilities
│
├── 5_training/                  # Model training with ToW
│   └── finetune_with_tow.py     # Fine-tune base models with ToW-augmented data
│
├── 6_results/                   # Results and analysis
│   ├── baseline/                # Baseline evaluation results
│   ├── tow_training/            # ToW training results  
│   └── comparison/              # Before/after comparison analysis
│
│
├── config_option2.yaml          # Unified configuration for Option 2
├── run_full_workflow.py         # Complete workflow execution script
└── README.md                    # This file
```

## Quick Start

### 1. Full Automated Workflow

```bash
# Check GPU compatibility first
python run_full_workflow.py --gpu-check

# Run complete TOW Option 2 workflow
python run_full_workflow.py --all

# Or run individual phases
python run_full_workflow.py --setup         # Download models and data
python run_full_workflow.py --baseline      # Run baseline evaluation  
python run_full_workflow.py --generate-tow  # Generate ToW data
python run_full_workflow.py --train         # Train with ToW
python run_full_workflow.py --evaluate      # Final comparison
```

### 2. Manual Step-by-Step Execution

#### Phase 1: Model and Data Setup
```bash
# Download GPT-OSS for ToW generation (choose based on your GPU)
cd 1_models/

# For 16GB+ GPU (GTX 4090, RTX 3090, A100 40GB)
python download_gpt_oss_20b.py    

# For 80GB+ GPU (H100, A100 80GB) - best quality
python download_gpt_oss_120b.py   

# Download base models for training (separate from ToW generation models)
python download_deepseek_r1_distill_qwen_7b.py
python download_qwen25_7b_instruct.py

# Download Korean evaluation datasets
cd ../2_datasets/
python download_korean_datasets.py
```

#### Phase 2: Baseline Evaluation
```bash
cd ../3_evaluation/
python baseline_evaluation.py --all-models
# Saves results to: ../6_results/baseline/
```

#### Phase 3: ToW Data Generation  
```bash
cd ../4_tow_generation/

# Test ToW prompt generation
python tow_prompt_generator.py

# Generate ToW-augmented training data using GPT-OSS
python korean_tow_generator.py --batch --output ../2_datasets/tow_training_data.jsonl
```

#### Phase 4: Model Training with ToW
```bash
cd ../5_training/
python finetune_with_tow.py --all-models
# Trains: DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct
```

#### Phase 5: Evaluation and Comparison
```bash
cd ../3_evaluation/
python compare_baseline_vs_tow.py --generate-report
# Generates comprehensive comparison in ../6_results/comparison/
```

## Technical Implementation

### ToW Generation Process (Following Zhikun et al. 2024)

1. **5-Shot Prompting**: Uses 5 Korean→English ToW examples for context
2. **English-Only Constraint**: All reasoning in `<ToW>` tokens must be English
3. **Word Categorization**: Categorizes predicted words as:
   - **Trivial**: Common function words (은, 는, 이, 가, the, and)
   - **Exact**: Proper nouns, numbers, quoted text
   - **Soft**: Semantically consistent words requiring context
   - **Unpredictable**: Words requiring complex reasoning
4. **Quality Validation**: Checks English compliance and reasoning quality
5. **Cohen's Kappa**: Quality assessment following paper methodology

### Model Separation Strategy

- **ToW Generation Models**: GPT-OSS-20B/120B (OpenAI's open-source models)
- **Training Target Models**: DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct
- **Workflow**: GPT-OSS generates ToW data → Base models trained with ToW data → Compare performance

### Key Configuration (config_option2.yaml)

```yaml
tow_settings:
  enforce_english_only: true
  cross_lingual_target: "en"
  generation_model_preference: ["gpt-oss-120b", "gpt-oss-20b"]
  training_models: ["deepseek-r1-qwen-7b", "qwen25-7b-instruct"]
  word_categories: ["trivial", "exact", "soft", "unpredictable"]

methodology:
  paper_reference: "Zhikun et al. (2024) Thoughts of Words"
  prompt_shots: 5
  validation_metrics: ["english_compliance", "reasoning_quality", "cohen_kappa"]

evaluation:
  benchmarks: ["KMMLU", "KLUE", "translation"]
  klue_tasks: ["TC", "STS", "NLI", "NER", "RE", "DP", "MRC", "DST"]
  baseline_comparison: true
  
training:
  batch_size: 4
  learning_rate: 2e-5
  epochs: 3
  tow_data_ratio: 0.3
```

## System Requirements

### For GPT-OSS-20B (Recommended)
- **GPU**: 16GB+ VRAM (RTX 4090, RTX 3090, A100 40GB)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free disk space
- **CUDA**: Compatible GPU with CUDA support

### For GPT-OSS-120B (High-end, Best Quality)
- **GPU**: 80GB+ VRAM (H100, A100 80GB, or multi-GPU setup)
- **RAM**: 128GB+ system memory recommended
- **Storage**: 300GB+ free disk space  
- **Network**: High-speed internet for 240GB model download

### Check Compatibility
```bash
# Check system compatibility for both models
python run_full_workflow.py --gpu-check

# Individual model checks
cd 1_models/
python download_gpt_oss_20b.py --info
python download_gpt_oss_120b.py --check
```

## Results and Analysis

Results are automatically saved in `6_results/` with structured directories:

### Baseline Results (`6_results/baseline/`)
- Zero-shot performance on KMMLU, KLUE (8 tasks), translation
- Model-specific performance metrics
- Detailed evaluation logs

### ToW Training Results (`6_results/tow_training/`)  
- ToW data generation statistics
- Training loss curves and metrics
- Model checkpoint information

### Comparison Analysis (`6_results/comparison/`)
- **Statistical Analysis**: T-tests, effect sizes, confidence intervals
- **Performance Improvements**: Task-specific and overall improvements
- **Visualizations**: Performance comparison charts
- **Detailed Report**: Comprehensive analysis with recommendations

### Example Results Structure
```
6_results/
├── baseline/
│   ├── deepseek_r1_qwen_7b_results.json
│   ├── qwen25_7b_results.json
│   └── baseline_summary.json
├── tow_training/
│   ├── tow_generation_stats.json
│   ├── training_metrics.json  
│   └── model_checkpoints/
└── comparison/
    ├── performance_comparison.json
    ├── statistical_analysis.json
    ├── improvement_visualization.png
    └── final_report.html
```

## Research Foundation and Methodology

This implementation strictly follows:

- **Zhikun et al. (2024)**: "Thoughts of Words: Exploring the Realm of Next-Word Prediction"
- **5-Shot Methodology**: Official prompt structure from the paper
- **Validation Framework**: Cohen's Kappa, quality scoring, categorization system
- **Option 2 Approach**: Cross-lingual reasoning (Korean context → English reasoning → Korean output)
- **Korean Language Focus**: Comprehensive evaluation on Korean language understanding tasks

### Key Papers and References

1. **Primary**: Zhikun et al. (2024) - Thoughts of Words methodology
2. **KMMLU**: Korean Massive Multitask Language Understanding
3. **KLUE**: Korean Language Understanding Evaluation benchmark
4. **Cross-lingual Transfer**: Multilingual reasoning and transfer learning

## Troubleshooting

### Common Issues

1. **GPU Memory Error**: Use GPT-OSS-20B instead of 120B, or enable CPU offloading
2. **Download Timeout**: Resume interrupted downloads (scripts support resuming)
3. **CUDA Not Found**: Install CUDA toolkit compatible with PyTorch
4. **Korean Text Encoding**: Ensure UTF-8 encoding in all text processing

### Debug Mode
```bash
# Run with detailed logging
python run_full_workflow.py --all 2>&1 | tee workflow_debug.log

# Test individual components
cd 4_tow_generation/
python tow_prompt_generator.py  # Test prompt generation
python korean_tow_generator.py --test  # Test ToW generation
```

## Citation

```bibtex
@misc{tow-option2-korean-2024,
  title={TOW Option 2: Cross-lingual Thoughts of Words Implementation for Korean Language Models},
  author={TOW Research Team},
  year={2024},
  note={Implementation of Zhikun et al. ToW methodology with Korean language focus},
  url={https://github.com/your-repo/tow-korean}
}

@article{zhikun2024thoughts,
  title={Thoughts of Words: Exploring the Realm of Next-Word Prediction},
  author={Zhikun et al.},
  journal={arXiv preprint arXiv:2410.16235},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Last Updated**: January 2025  
**Project Status**: Complete implementation ready for execution  
**Methodology**: Zhikun et al. (2024) ToW with Korean language specialization