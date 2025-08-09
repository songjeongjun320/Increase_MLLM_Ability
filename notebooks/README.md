# Research Notebooks

This directory contains Jupyter notebooks for ToW research analysis, experimentation, and visualization.

## Structure

### Exploratory Analysis
- `01_data_exploration.ipynb` - Initial data analysis and corpus statistics
- `02_baseline_evaluation.ipynb` - Baseline model performance assessment
- `03_tow_mechanism_analysis.ipynb` - ToW intermediate reasoning analysis

### Experiments
- `04_training_experiments.ipynb` - Training process monitoring and analysis
- `05_multilingual_evaluation.ipynb` - Cross-lingual performance evaluation
- `06_cultural_adaptation.ipynb` - Cultural appropriateness assessment

### Visualization
- `07_results_visualization.ipynb` - Research results visualization and plots
- `08_statistical_analysis.ipynb` - Statistical significance testing
- `09_comparison_analysis.ipynb` - Model comparison and benchmarking

### Research Documentation
- `10_methodology_documentation.ipynb` - Detailed methodology documentation
- `11_case_studies.ipynb` - Specific case study examples
- `12_error_analysis.ipynb` - Error pattern analysis and insights

## Usage

1. Install Jupyter dependencies:
   ```bash
   pip install jupyter jupyterlab matplotlib seaborn plotly
   ```

2. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

3. Navigate to the notebooks directory and open desired notebook.

## Data Requirements

Some notebooks require access to:
- Preprocessed training datasets
- Model evaluation results
- Experiment tracking logs
- Benchmark evaluation outputs

Ensure data is available in the `data/` and `experiments/` directories before running notebooks.

## Reproducibility

All notebooks include:
- Environment setup cells
- Random seed configuration
- Clear documentation of dependencies
- Data path specifications

For reproducible results, run notebooks in order and ensure all dependencies are installed.