# ToW Dataset Quality Check

This module provides comprehensive quality assessment and adjustment tools for ToW (Thought of Words) datasets based on the methodology described in the ToW paper.

## Features

### Quality Assessment Criteria

1. **Valid ToW Format**: Proper `<ToW>...</ToW>` tag structure
2. **Meaningful Explanations**: Non-formulaic, substantive explanations
3. **Appropriate Length**: Between 50-500 characters for explanations
4. **Logical Coherence**: Logical connection between context, target word, and explanation
5. **Contextual Relevance**: Explanation references and relates to the given context
6. **Linguistic Quality**: Proper grammar, sentence structure, and vocabulary diversity

### Quality Adjustment Features

1. **Format Fixing**: Corrects malformed ToW tags and formatting issues
2. **Length Adjustment**: Expands short explanations and truncates overly long ones
3. **Content Enhancement**: Improves explanation quality through pattern-based improvements
4. **Grammar Correction**: Fixes basic grammatical issues and repetitions
5. **Contextual Enhancement**: Strengthens connections between context and explanations

## Files

- `tow_quality_checker.py`: Core quality assessment functionality
- `tow_quality_adjuster.py`: Quality adjustment and enhancement tools
- `run_quality_check.py`: Main script to run the complete pipeline
- `requirements.txt`: Required Python packages

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete quality check and adjustment pipeline
python run_quality_check.py
```

### Individual Components

```python
from tow_quality_checker import ToWQualityChecker
from tow_quality_adjuster import ToWQualityAdjuster

# Initialize components
checker = ToWQualityChecker()
adjuster = ToWQualityAdjuster(checker)

# Load your dataset
import json
with open('your_dataset.json', 'r') as f:
    data = json.load(f)

# Assess quality
report = checker.generate_quality_report(data)
print(report)

# Adjust quality
adjusted_data, stats = adjuster.adjust_dataset(
    data, 
    remove_low_quality=True, 
    min_quality_threshold=4
)

# Filter high-quality samples
high_quality = checker.filter_high_quality_samples(adjusted_data, min_criteria_met=5)
```

## Output Files

The pipeline generates several output files:

1. **Initial Quality Report** (`*_initial_quality_report.txt`): 
   - Quality metrics before adjustment
   - Identifies areas needing improvement

2. **Adjustment Report** (`*_adjustment_report.txt`):
   - Statistics on adjustments made
   - Retention rates and improvement counts

3. **Final Quality Report** (`*_final_quality_report.txt`):
   - Quality metrics after adjustment
   - Improvement validation

4. **Quality Adjusted Dataset** (`*_quality_adjusted.json`):
   - Dataset with quality improvements applied
   - Includes quality scores for each sample

5. **High Quality Dataset** (`*_high_quality.json`):
   - Filtered dataset with only highest quality samples
   - Meets 5/6 quality criteria

## Configuration Options

### ToWQualityChecker Parameters

- `min_tow_length`: Minimum acceptable explanation length (default: 50)
- `max_tow_length`: Maximum acceptable explanation length (default: 500)

### Quality Adjustment Parameters

- `remove_low_quality`: Whether to remove low-quality samples (default: True)
- `min_quality_threshold`: Minimum quality score to retain samples (default: 4)
- `min_criteria_met`: Minimum criteria for high-quality filtering (default: 5)

## Example Output

```
ToW Dataset Quality Assessment Report
=====================================

Dataset Overview:
- Total samples: 1000

Quality Metrics:
- Valid ToW format: 95.2% (952/1000)
- Meaningful explanations: 78.5% (785/1000)
- Appropriate length: 89.1% (891/1000)
- Logical coherence: 82.3% (823/1000)
- Contextual relevance: 87.6% (876/1000)
- Linguistic quality: 79.8% (798/1000)

Overall Assessment:
- Average quality score: 85.4%

Recommendations:
- Improve explanation meaningfulness
- Enhance logical coherence in explanations
- Improve linguistic quality
```

## Quality Criteria Details

### 1. Valid ToW Format
- Checks for proper `<ToW>...</ToW>` tags
- Ensures content exists within tags
- Fixes malformed or missing tags

### 2. Meaningful Explanations
- Detects formulaic or repetitive patterns
- Requires quality keywords indicating thoughtful analysis
- Avoids overly simple explanations

### 3. Appropriate Length
- Ensures explanations are substantial but not excessive
- Expands short explanations with contextual reasoning
- Truncates long explanations while preserving key information

### 4. Logical Coherence
- Verifies explanation mentions both context and target word
- Checks for logical connectives and reasoning structure
- Ensures explanation follows from the given context

### 5. Contextual Relevance
- Measures overlap between explanation and context words
- Ensures explanation relates to the specific scenario
- Validates contextual appropriateness

### 6. Linguistic Quality
- Checks sentence structure and grammar
- Monitors vocabulary diversity and repetition
- Ensures proper linguistic flow

## Best Practices

1. **Start with Assessment**: Always run initial quality assessment to understand dataset state
2. **Gradual Filtering**: Use progressive filtering thresholds (4/6 → 5/6 criteria)
3. **Review Reports**: Examine quality reports to understand improvement areas
4. **Validate Results**: Check final quality metrics to confirm improvements
5. **Preserve Originals**: Keep original data for comparison and fallback

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Path Issues**: Verify data directory path in `run_quality_check.py`
3. **Memory Issues**: Process large datasets in batches if needed
4. **Encoding Issues**: Ensure UTF-8 encoding for text files

### Performance Tips

1. **Batch Processing**: Process files individually to manage memory
2. **Quality Thresholds**: Adjust thresholds based on your quality requirements
3. **Logging**: Enable detailed logging for debugging and monitoring
4. **Validation**: Test on small samples before processing entire datasets