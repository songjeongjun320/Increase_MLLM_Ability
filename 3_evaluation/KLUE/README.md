# KLUE Benchmark Evaluation

This directory contains a comprehensive implementation of the KLUE (Korean Language Understanding Evaluation) benchmark for evaluating Korean language understanding capabilities of large language models.

## Overview

KLUE consists of 8 diverse Korean natural language understanding tasks:

1. **Topic Classification (TC)** - News article topic classification
2. **Sentence Textual Similarity (STS)** - Semantic similarity between sentence pairs
3. **Natural Language Inference (NLI)** - Logical reasoning and inference
4. **Named Entity Recognition (NER)** - Named entity identification and classification
5. **Relation Extraction (RE)** - Relation classification between entities
6. **Dependency Parsing (DP)** - Syntactic parsing of Korean sentences
7. **Machine Reading Comprehension (MRC)** - Reading comprehension and question answering
8. **Dialogue State Tracking (DST)** - Dialogue state tracking for task-oriented dialogue

## File Structure

```
EVALUATION/KLUE/
├── config.py              # Common configuration and model definitions
├── utils.py               # Utility functions and model loading
├── tc.py                  # Topic Classification evaluation
├── sts.py                 # Sentence Textual Similarity evaluation
├── nli.py                 # Natural Language Inference evaluation
├── ner.py                 # Named Entity Recognition evaluation
├── re.py                  # Relation Extraction evaluation
├── dp.py                  # Dependency Parsing evaluation
├── mrc.py                 # Machine Reading Comprehension evaluation
├── dst.py                 # Dialogue State Tracking evaluation
├── run_all_klue.py        # Run all tasks sequentially
├── run_model_klue.py      # Run all tasks for a specific model
└── README.md              # This file
```

## Model Configuration

Models are configured in `config.py`. The current setup includes:

### Base Models
- Qwen2.5-7B-Instruct
- Mistral-8B-Instruct-2410
- Llama-3.1-8B-Instruct
- DeepSeek-R1-0528-Qwen3-8B

### ToW Trained Models
- Qwen2.5-7B-Instruct-ToW
- Mistral-8B-Instruct-2410-ToW
- Llama-3.1-8B-Instruct-ToW
- DeepSeek-R1-0528-Qwen3-8B-ToW

## Data Format

The benchmark expects data files in the `../klue_all_tasks_json/` directory:

- `klue_tc_validation.json` - Topic Classification validation data
- `klue_sts_validation.json` - STS validation data
- `klue_nli_validation.json` - NLI validation data
- `klue_ner_validation.json` - NER validation data
- `klue_re_validation.json` - RE validation data
- `klue_dp_validation.json` - DP validation data
- `klue_mrc_validation.json` - MRC validation data
- `klue_dst_validation.json` - DST validation data

## Usage

### 1. Individual Task Evaluation

Run evaluation for a specific task:

```bash
# Topic Classification
python tc.py

# Sentence Textual Similarity
python sts.py

# Natural Language Inference
python nli.py

# Named Entity Recognition
python ner.py

# Relation Extraction
python re.py

# Dependency Parsing
python dp.py

# Machine Reading Comprehension
python mrc.py

# Dialogue State Tracking
python dst.py
```

### 2. Complete Benchmark (All Tasks)

Run all tasks sequentially:

```bash
python run_all_klue.py
```

Options:
- `--tasks`: Specify which tasks to run (default: all)
- `--skip-evaluation`: Only collect existing results without running evaluation

Examples:
```bash
# Run only TC, STS, and NLI tasks
python run_all_klue.py --tasks tc sts nli

# Collect results without running evaluation
python run_all_klue.py --skip-evaluation
```

### 3. Model-wise Evaluation

Evaluate a specific model on all tasks (memory-efficient):

```bash
# Evaluate a specific model
python run_model_klue.py --model "Qwen2.5-7B-Instruct"

# Evaluate all models
python run_model_klue.py

# Evaluate with limited samples for testing
python run_model_klue.py --model "Qwen2.5-7B-Instruct" --max-samples 100
```

Options:
- `--model`: Specify model name to evaluate (default: all models)
- `--tasks`: Specify which tasks to run (default: all)
- `--max-samples`: Limit number of samples per task for testing

## Evaluation Metrics

Each task uses specific evaluation metrics:

| Task | Primary Metric | Secondary Metrics |
|------|----------------|------------------|
| TC   | **Macro F1** (Official)      | Accuracy |
| STS  | Pearson Correlation | P-value |
| NLI  | Accuracy       | F1-Macro |
| NER  | F1 Score       | Precision, Recall |
| RE   | **Micro F1** (Official)      | Macro F1, Accuracy |
| DP   | LAS (Labeled Attachment Score) | UAS (Unlabeled Attachment Score) |
| MRC  | F1 Score       | Exact Match |
| DST  | Joint Goal Accuracy | Slot Accuracy |

## Output Structure

Results are saved with the following structure:

```
klue_evaluation_results/
├── task_evaluation_YYYYMMDD_HHMMSS/
│   ├── {model_name}_{task}_results.json
│   └── {task}_summary.json
├── klue_full_benchmark_YYYYMMDD_HHMMSS/
│   ├── klue_benchmark_results.csv
│   ├── klue_benchmark_results.json
│   └── model_results/
└── klue_model_wise_YYYYMMDD_HHMMSS/
    ├── model_{model_name}/
    │   ├── {model_name}_{task}_results.json
    │   └── {model_name}_klue_summary.json
    └── overall_summary.json
```

## Result Files

### Individual Task Results
- `{model_name}_{task}_results.json`: Detailed results for a model on a specific task
- `{task}_summary.json`: Summary of all models' performance on a task

### Comprehensive Results
- `klue_benchmark_results.csv`: Results table in CSV format
- `klue_benchmark_results.json`: Complete results in JSON format
- `overall_summary.json`: Overall evaluation summary

## Example Result Structure

```json
{
  "task_type": "tc",
  "model_name": "Qwen2.5-7B-Instruct",
  "num_samples": 3003,
  "metrics": {
    "accuracy": 0.8542,
    "f1_macro": 0.8398
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

## Memory Management

The evaluation system includes automatic memory management:

- Models are loaded and unloaded for each task to prevent OOM errors
- GPU memory is cleared between model evaluations
- Batch processing for large datasets

## Customization

### Adding New Models

Add new model configurations to `config.py`:

```python
ModelConfig(
    name="YourModel",
    model_id="/path/to/your/model",
    adapter_path="/path/to/adapter",  # Optional
    use_quantization=False
)
```

### Modifying Prompts

Update prompt templates in `config.py`:

```python
PROMPT_TEMPLATES = {
    'task_name': """Your custom prompt template with {placeholder}""",
    # ...
}
```

### Adjusting Evaluation Parameters

- Modify `max_new_tokens`, `temperature` in evaluation functions
- Adjust sample limits for testing
- Configure quantization settings per model

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- scikit-learn
- scipy
- pandas
- numpy
- tqdm

## Performance Tips

1. **Memory Usage**: Use `--max-samples` for testing to avoid OOM
2. **Model Loading**: Use model-wise evaluation (`run_model_klue.py`) for better memory management
3. **Quantization**: Enable quantization for large models if VRAM is limited
4. **Batch Size**: Adjust batch sizes based on available GPU memory

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `max_samples` or enable quantization
2. **Model Loading Errors**: Check model paths in `config.py`
3. **Data Loading Errors**: Verify data files exist in correct directory
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this evaluation code, please cite the original KLUE paper:

```
@inproceedings{park2021klue,
  title={KLUE: Korean Language Understanding Evaluation},
  author={Park, Sungjoon and Moon, Jihyung and Kim, Sungdong and Cho, Won Ik and Han, Jiyoon and Park, Jangwon and Song, Chisung and Kim, Junseong and Song, Yongsook and Taek, Oh and others},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021}
}
```