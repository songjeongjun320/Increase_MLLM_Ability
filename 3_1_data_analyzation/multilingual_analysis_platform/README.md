# 🌍 Multilingual Language Model Analysis Platform

A comprehensive platform for analyzing and visualizing multilingual language models, focusing on sentence embeddings, attention patterns, and prediction confidence. This platform is designed for academic research and experimental monitoring, enabling intuitive exploration of model performance and intrinsic uncertainty.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Technical Details](#-technical-details)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Features

### Core Analysis Modules

#### 1. 📊 Sentence Embedding Analysis
- **Multilingual Support**: Sentence-BERT with multilingual models
- **Dimensionality Reduction**: PCA, t-SNE, and UMAP visualization
- **Cross-Language Comparison**: Semantic similarity analysis between languages
- **Clustering Analysis**: Language-based clustering and purity metrics
- **Interactive Visualizations**: 2D/3D plots with language color coding

#### 2. 🔍 Attention Pattern Analysis
- **Multi-Head Attention**: Visualization of attention weights across heads and layers
- **Pattern Detection**: Entropy analysis, attention distance, and token importance
- **Cross-Model Comparison**: Compare attention patterns between base and trained models
- **Interactive Heatmaps**: Detailed attention matrices with hover information
- **Pattern Statistics**: Comprehensive analysis of attention flow and distribution

#### 3. 📈 Prediction Confidence Analysis
- **Uncertainty Quantification**: Entropy, variance, and perplexity measurements
- **Token-Level Analysis**: Position-wise confidence and uncertainty regions
- **Confidence Intervals**: Statistical confidence measures for predictions
- **Temporal Analysis**: Confidence changes across sequence positions
- **Comparative Metrics**: Cross-model and cross-language confidence comparison

#### 4. 🔄 Model Comparison Suite
- **Base vs Training Model**: Comprehensive comparison framework
- **English-Korean Specialized**: Optimized for English-Korean language pairs
- **Multi-Metric Analysis**: Attention, embedding, and confidence comparisons
- **Performance Tracking**: Accuracy, similarity, and improvement metrics
- **Visual Comparisons**: Side-by-side visualizations and difference heatmaps

### Dashboard Features

- **🖥️ Unified Interface**: Streamlit-based interactive dashboard
- **📁 Flexible Input**: Manual entry, file upload, or sample data
- **⚙️ Configurable Models**: Support for base and custom trained models
- **📊 Real-time Visualization**: Interactive plots with Plotly
- **💾 Export Capabilities**: Results export in multiple formats
- **🔧 Memory Management**: Intelligent caching and resource optimization

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- 2GB+ disk space for models and cache

### Method 1: Quick Setup

```bash
# Clone or navigate to the platform directory
cd 3_1_data_analyzation/multilingual_analysis_platform

# Install dependencies
pip install -r requirements.txt

# Run the platform
python run_dashboard.py
```

### Method 2: Manual Setup

```bash
# Install core dependencies
pip install streamlit torch transformers sentence-transformers
pip install numpy pandas matplotlib plotly seaborn
pip install scikit-learn umap-learn bertviz

# Install optional dependencies for enhanced features
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x

# Launch the dashboard
streamlit run dashboard/main_dashboard.py
```

### Method 3: Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Launch with development settings
python app.py
```

## ⚡ Quick Start

### 1. Launch the Platform

```bash
python run_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 2. Basic Usage

1. **Input Text**: Use the sidebar to enter text manually, upload files, or use sample data
2. **Select Languages**: Choose from supported languages (English, Korean, Japanese, Chinese)
3. **Configure Models**: Select base model and optionally add your trained model path
4. **Run Analysis**: Click "Run Analysis" to generate comprehensive insights
5. **Explore Results**: Navigate through different tabs for various analysis types

### 3. Sample Analysis

```python
# Example: Programmatic usage
from multilingual_analysis_platform import SentenceEmbeddingAnalyzer

analyzer = SentenceEmbeddingAnalyzer()

# Analyze English-Korean sentence pairs
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "빠른 갈색 여우가 게으른 개를 뛰어넘습니다."
]
languages = ['en', 'ko']

# Generate embeddings and analyze
results = analyzer.generate_embeddings(texts, languages)
comparison = analyzer.compare_sentence_pairs(
    sentences_en=[texts[0]],
    sentences_target=[texts[1]],
    target_language='ko'
)

print(f"Cross-language similarity: {comparison['mean_pair_similarity']:.3f}")
```

## 📖 Usage Guide

### Dashboard Navigation

#### Sidebar Controls
- **Model Selection**: Choose base model and add training model path
- **Language Configuration**: Select analysis languages
- **Text Input**: Manual entry, file upload, or sample data
- **Analysis Controls**: Run analysis and manage cache

#### Main Tabs

##### 📊 Sentence Embeddings
- **Visualization Options**: 2D/3D plots with different reduction methods
- **Similarity Analysis**: Heatmaps showing cross-sentence similarities
- **Language Clustering**: Analysis of language separation in embedding space
- **Interactive Features**: Hover information and zoom capabilities

##### 🔍 Attention Analysis
- **Layer/Head Selection**: Choose specific layers and attention heads
- **Visualization Types**: Single head, multi-head, or pattern analysis
- **Pattern Analysis**: Entropy, distance, and token importance metrics
- **Token Analysis**: Detailed token-level attention information

##### 📈 Confidence Analysis
- **Analysis Types**: Entropy timeline, confidence heatmaps, uncertainty regions
- **Metrics**: Entropy, variance, top-k probability, perplexity
- **Distribution Analysis**: Statistical analysis of confidence patterns
- **Position-wise Analysis**: Token-level confidence visualization

##### 🔄 Model Comparison
- **Comparison Types**: Attention patterns, confidence measures, side-by-side
- **Base vs Training**: Comprehensive model performance comparison
- **Cross-language Analysis**: Language-specific improvement metrics
- **Visual Comparisons**: Side-by-side attention heatmaps

##### ⚙️ Settings
- **Model Configuration**: Update model settings and parameters
- **Performance Settings**: Batch size, device selection
- **Memory Management**: Cache control and resource monitoring
- **Export/Import**: Results export and configuration management

### File Input Formats

#### CSV Format
```csv
text,language
"Hello world",en
"안녕하세요",ko
"Bonjour le monde",fr
```

#### JSON Format
```json
[
    {"text": "Hello world", "language": "en"},
    {"text": "안녕하세요", "language": "ko"},
    {"text": "Bonjour le monde", "language": "fr"}
]
```

#### Text File
Plain text files are treated as English by default.

### Advanced Usage

#### Custom Model Integration

```python
# Example: Using custom trained models
from multilingual_analysis_platform.models.model_manager import get_model_manager

model_manager = get_model_manager()

# Load custom model
model, tokenizer = model_manager.load_trained_model("/path/to/your/model")

# Use for analysis
attention_data = model_manager.get_attention_weights(
    "/path/to/your/model",
    "Your text here",
    model_type='trained'
)
```

#### Batch Processing

```python
# Example: Batch analysis of multiple texts
from multilingual_analysis_platform.core.sentence_embedding import SentenceEmbeddingAnalyzer

analyzer = SentenceEmbeddingAnalyzer()

# Process multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
languages = ["en", "en", "en"]

results = analyzer.generate_embeddings(texts, languages)

# Analyze similarities
similarity_analysis = analyzer.analyze_semantic_similarity(
    texts, languages, similarity_threshold=0.8
)
```

#### Model Comparison Pipeline

```python
# Example: Comprehensive model comparison
from multilingual_analysis_platform.utils.comparison_utils import ModelComparisonSuite

comparison_suite = ModelComparisonSuite()

# Define test sentences (English-Korean pairs)
test_sentences = [
    ("Hello world", "안녕하세요"),
    ("How are you?", "어떻게 지내세요?"),
    ("Thank you", "감사합니다")
]

# Run comprehensive comparison
results = comparison_suite.compare_base_vs_training_models(
    base_model_path="bert-base-multilingual-cased",
    training_model_path="/path/to/your/trained/model",
    test_sentences=test_sentences
)

# Export results
comparison_suite.export_comparison_results(
    results,
    "comparison_results.json"
)
```

## 🔧 Technical Details

### Architecture

```
multilingual_analysis_platform/
├── core/                          # Core analysis modules
│   ├── sentence_embedding.py      # Sentence-BERT analysis
│   ├── attention_analysis.py      # Attention pattern analysis
│   └── confidence_analysis.py     # Prediction confidence analysis
├── models/                        # Model management
│   └── model_manager.py          # Model loading and caching
├── visualization/                 # Visualization modules
│   ├── embedding_plots.py        # Embedding visualizations
│   ├── attention_plots.py        # Attention visualizations
│   └── confidence_plots.py       # Confidence visualizations
├── dashboard/                     # Streamlit dashboard
│   └── main_dashboard.py         # Main dashboard interface
├── utils/                         # Utility modules
│   ├── config_loader.py          # Configuration management
│   └── comparison_utils.py       # Model comparison utilities
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── tests/                        # Test suite
└── app.py                        # Main application entry point
```

### Supported Models

#### Base Models
- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `distilbert-base-multilingual-cased`

#### Sentence Transformers
- `paraphrase-multilingual-MiniLM-L12-v2`
- `distiluse-base-multilingual-cased`
- `LaBSE`

#### Custom Models
- Support for any Hugging Face compatible model
- Local model loading for trained/fine-tuned models
- Automatic tokenizer detection and loading

### Performance Optimization

#### Memory Management
- **Model Caching**: Intelligent caching of loaded models and tokenizers
- **Embedding Caching**: Cache embedding results for repeated analysis
- **GPU Memory**: Automatic GPU memory management and cleanup
- **Batch Processing**: Configurable batch sizes for memory efficiency

#### Computation Optimization
- **Parallel Processing**: Multi-threaded analysis where applicable
- **Dimensionality Reduction**: Optimized implementations of PCA/t-SNE/UMAP
- **Attention Analysis**: Efficient attention weight extraction and processing
- **Progressive Loading**: Lazy loading of models and data

### Configuration

#### Main Configuration (`config/config.yaml`)

```yaml
# Model settings
models:
  sentence_transformer:
    default_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  base_models:
    default: "bert-base-multilingual-cased"

# Language settings
languages:
  supported:
    - code: "en"
      name: "English"
      color: "#1f77b4"
    - code: "ko"
      name: "Korean"
      color: "#ff7f0e"

# Performance settings
performance:
  device: "auto"  # auto, cpu, cuda
  batch_size: 16
  num_workers: 4

# Visualization settings
visualization:
  dimensionality_reduction:
    default_method: "umap"
    parameters:
      umap:
        n_neighbors: 15
        min_dist: 0.1
```

## 📚 API Reference

### Core Classes

#### SentenceEmbeddingAnalyzer

```python
class SentenceEmbeddingAnalyzer:
    def generate_embeddings(self, texts: List[str], languages: List[str]) -> Dict[str, Any]
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'umap') -> np.ndarray
    def compare_sentence_pairs(self, sentences_en: List[str], sentences_target: List[str]) -> Dict[str, Any]
    def analyze_language_clusters(self, embeddings: np.ndarray, languages: List[str]) -> Dict[str, Any]
```

#### AttentionAnalyzer

```python
class AttentionAnalyzer:
    def extract_attention_weights(self, model_name: str, text: str) -> Dict[str, Any]
    def analyze_attention_patterns(self, attention_result: Dict[str, Any]) -> Dict[str, Any]
    def compare_attention_patterns(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]
```

#### ConfidenceAnalyzer

```python
class ConfidenceAnalyzer:
    def analyze_prediction_confidence(self, model_name: str, text: str) -> Dict[str, Any]
    def compare_model_confidence(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]
    def analyze_multilingual_confidence(self, model_name: str, text_pairs: List[Tuple[str, str]]) -> Dict[str, Any]
```

### Visualization Classes

#### EmbeddingVisualizer

```python
class EmbeddingVisualizer:
    def plot_embeddings_2d(self, embeddings: np.ndarray, languages: List[str]) -> go.Figure
    def plot_embeddings_3d(self, embeddings: np.ndarray, languages: List[str]) -> go.Figure
    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray, languages: List[str]) -> go.Figure
```

### Utility Classes

#### ModelComparisonSuite

```python
class ModelComparisonSuite:
    def compare_base_vs_training_models(self, base_model: str, training_model: str, test_sentences: List[Tuple[str, str]]) -> Dict[str, Any]
    def export_comparison_results(self, results: Dict[str, Any], output_path: str) -> str
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=multilingual_analysis_platform tests/

# Run specific test modules
pytest tests/test_sentence_embedding.py
pytest tests/test_attention_analysis.py
pytest tests/test_confidence_analysis.py
```

### Test Structure

```
tests/
├── test_sentence_embedding.py    # Embedding analysis tests
├── test_attention_analysis.py    # Attention analysis tests
├── test_confidence_analysis.py   # Confidence analysis tests
├── test_model_manager.py         # Model management tests
├── test_visualizations.py        # Visualization tests
├── test_dashboard.py             # Dashboard tests
└── fixtures/                     # Test data and fixtures
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository** and clone your fork
2. **Create a virtual environment**: `python -m venv venv`
3. **Activate the environment**: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. **Install in development mode**: `pip install -e .`
5. **Install development dependencies**: `pip install -r requirements-dev.txt`
6. **Run tests**: `pytest tests/`

### Code Style

- **Black**: Code formatting (`black .`)
- **isort**: Import sorting (`isort .`)
- **flake8**: Linting (`flake8`)
- **mypy**: Type checking (`mypy src/`)

### Submitting Changes

1. **Create a feature branch**: `git checkout -b feature-name`
2. **Make your changes** and add tests
3. **Run the test suite**: `pytest tests/`
4. **Check code style**: `black . && isort . && flake8`
5. **Commit your changes**: `git commit -m "Description of changes"`
6. **Push to your fork**: `git push origin feature-name`
7. **Create a pull request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers**: For the transformer model implementations
- **Sentence Transformers**: For multilingual sentence embedding models
- **Streamlit**: For the interactive dashboard framework
- **Plotly**: For interactive visualizations
- **scikit-learn**: For machine learning utilities
- **UMAP**: For dimensionality reduction algorithms

## 📞 Support

For questions, issues, or contributions:

1. **Check the documentation** in this README
2. **Search existing issues** in the repository
3. **Create a new issue** with detailed information
4. **Join discussions** in the project's discussion forum

## 🗺️ Roadmap

### Upcoming Features

- **Additional Languages**: Support for more language pairs
- **Advanced Models**: Integration with latest multilingual models
- **Batch Processing**: Enhanced batch analysis capabilities
- **API Endpoints**: RESTful API for programmatic access
- **Export Formats**: Additional export formats and templates
- **Performance Metrics**: More sophisticated evaluation metrics
- **Real-time Analysis**: Live model monitoring capabilities

### Research Integration

- **Academic Papers**: Integration with latest multilingual research
- **Benchmark Datasets**: Support for standard evaluation datasets
- **Reproducibility**: Enhanced experiment tracking and reproducibility features
- **Collaboration**: Multi-user collaboration features

---

**Built with ❤️ for multilingual AI research and development**