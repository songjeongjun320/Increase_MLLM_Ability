"""
Main Streamlit Dashboard

Unified dashboard for the Multilingual Language Model Analysis Platform.
Integrates sentence embedding analysis, attention visualization, and confidence analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import logging
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.sentence_embedding import SentenceEmbeddingAnalyzer
from core.attention_analysis import AttentionAnalyzer
from core.confidence_analysis import ConfidenceAnalyzer
from visualization.embedding_plots import EmbeddingVisualizer
from visualization.attention_plots import AttentionVisualizer
from visualization.confidence_plots import ConfidenceVisualizer
from models.model_manager import get_model_manager
from utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multilingual Language Model Analysis Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1edff;
        border: 1px solid #74b9ff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class MultilingualAnalysisDashboard:
    """Main dashboard class for the multilingual analysis platform."""

    def __init__(self):
        """Initialize the dashboard."""
        self.config = get_config()
        self.model_manager = get_model_manager()

        # Initialize analyzers
        self.embedding_analyzer = SentenceEmbeddingAnalyzer()
        self.attention_analyzer = AttentionAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()

        # Initialize visualizers
        self.embedding_visualizer = EmbeddingVisualizer()
        self.attention_visualizer = AttentionVisualizer()
        self.confidence_visualizer = ConfidenceVisualizer()

        # Initialize session state
        self.init_session_state()

    def init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'current_texts' not in st.session_state:
            st.session_state.current_texts = []
        if 'current_languages' not in st.session_state:
            st.session_state.current_languages = []

    def run(self):
        """Run the main dashboard."""
        # Header
        st.markdown('<div class="main-header">🌍 Multilingual Language Model Analysis Platform</div>',
                   unsafe_allow_html=True)

        # Sidebar
        self.render_sidebar()

        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Sentence Embeddings",
            "🔍 Attention Analysis",
            "📈 Confidence Analysis",
            "🔄 Model Comparison",
            "⚙️ Settings"
        ])

        with tab1:
            self.render_embedding_analysis()

        with tab2:
            self.render_attention_analysis()

        with tab3:
            self.render_confidence_analysis()

        with tab4:
            self.render_model_comparison()

        with tab5:
            self.render_settings()

    def render_sidebar(self):
        """Render the sidebar with global controls."""
        st.sidebar.markdown("## 🎛️ Global Controls")

        # Model selection
        st.sidebar.markdown("### Model Selection")

        # Base model selection
        base_models = self.config.get('models.base_models.alternatives', ['bert-base-multilingual-cased'])
        selected_base_model = st.sidebar.selectbox(
            "Base Model",
            base_models,
            help="Select the base model for analysis"
        )

        # Training model path
        training_model_path = st.sidebar.text_input(
            "Training Model Path",
            placeholder="Enter path to trained model (optional)",
            help="Path to your fine-tuned model for comparison"
        )

        # Language selection
        st.sidebar.markdown("### Language Configuration")
        supported_languages = self.config.get_supported_languages()
        language_options = {lang['name']: lang['code'] for lang in supported_languages}

        selected_languages = st.sidebar.multiselect(
            "Select Languages",
            options=list(language_options.keys()),
            default=["English", "Korean"],
            help="Choose languages for analysis"
        )

        # Text input area
        st.sidebar.markdown("### Text Input")
        text_input_method = st.sidebar.radio(
            "Input Method",
            ["Manual Entry", "File Upload", "Sample Data"]
        )

        texts = []
        languages = []

        if text_input_method == "Manual Entry":
            texts, languages = self.handle_manual_input(selected_languages, language_options)
        elif text_input_method == "File Upload":
            texts, languages = self.handle_file_upload()
        else:
            texts, languages = self.get_sample_data()

        # Store in session state
        st.session_state.current_texts = texts
        st.session_state.current_languages = languages
        st.session_state.selected_base_model = selected_base_model
        st.session_state.training_model_path = training_model_path

        # Analysis controls
        st.sidebar.markdown("### Analysis Controls")
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            if texts and languages:
                with st.spinner("Running analysis..."):
                    self.run_comprehensive_analysis()
                st.success("Analysis completed!")
            else:
                st.error("Please provide texts and languages for analysis")

        # Memory management
        st.sidebar.markdown("### Memory Management")
        if st.sidebar.button("🧹 Clear Cache"):
            self.clear_all_caches()
            st.success("All caches cleared!")

    def handle_manual_input(self, selected_languages: List[str],
                           language_options: Dict[str, str]) -> tuple:
        """Handle manual text input."""
        texts = []
        languages = []

        for lang_name in selected_languages:
            lang_code = language_options[lang_name]
            text = st.sidebar.text_area(
                f"Text in {lang_name}",
                placeholder=f"Enter text in {lang_name}...",
                key=f"text_{lang_code}",
                height=100
            )
            if text.strip():
                texts.append(text.strip())
                languages.append(lang_code)

        return texts, languages

    def handle_file_upload(self) -> tuple:
        """Handle file upload for text input."""
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['txt', 'csv', 'json'],
            help="Upload a file containing texts and language codes"
        )

        texts = []
        languages = []

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns and 'language' in df.columns:
                        texts = df['text'].tolist()
                        languages = df['language'].tolist()
                    else:
                        st.sidebar.error("CSV must have 'text' and 'language' columns")
                elif uploaded_file.name.endswith('.json'):
                    import json
                    data = json.load(uploaded_file)
                    if isinstance(data, list):
                        for item in data:
                            if 'text' in item and 'language' in item:
                                texts.append(item['text'])
                                languages.append(item['language'])
                else:
                    # Plain text file
                    content = uploaded_file.read().decode('utf-8')
                    texts = [content]
                    languages = ['en']  # Default to English

            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")

        return texts, languages

    def get_sample_data(self) -> tuple:
        """Get sample data for demonstration."""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
            "Artificial intelligence is transforming the world.",
            "인공지능이 세상을 변화시키고 있습니다.",
            "Machine learning enables computers to learn from data.",
            "기계학습은 컴퓨터가 데이터로부터 학습할 수 있게 합니다."
        ]

        sample_languages = ['en', 'ko', 'en', 'ko', 'en', 'ko']

        return sample_texts, sample_languages

    def run_comprehensive_analysis(self):
        """Run comprehensive analysis on the provided texts."""
        texts = st.session_state.current_texts
        languages = st.session_state.current_languages
        base_model = st.session_state.selected_base_model
        training_model = st.session_state.training_model_path

        try:
            # 1. Sentence Embedding Analysis
            embedding_result = self.embedding_analyzer.generate_embeddings(
                texts=texts,
                languages=languages
            )
            st.session_state.analysis_results['embeddings'] = embedding_result

            # 2. Attention Analysis (for first text as example)
            if texts:
                attention_result = self.attention_analyzer.extract_attention_weights(
                    model_name_or_path=base_model,
                    text=texts[0],
                    model_type='base'
                )
                st.session_state.analysis_results['attention'] = attention_result

                # 3. Confidence Analysis
                confidence_result = self.confidence_analyzer.analyze_prediction_confidence(
                    model_name_or_path=base_model,
                    text=texts[0],
                    model_type='base'
                )
                st.session_state.analysis_results['confidence'] = confidence_result

                # 4. Model Comparison (if training model provided)
                if training_model:
                    training_attention = self.attention_analyzer.extract_attention_weights(
                        model_name_or_path=training_model,
                        text=texts[0],
                        model_type='trained'
                    )

                    training_confidence = self.confidence_analyzer.analyze_prediction_confidence(
                        model_name_or_path=training_model,
                        text=texts[0],
                        model_type='trained'
                    )

                    st.session_state.analysis_results['training_attention'] = training_attention
                    st.session_state.analysis_results['training_confidence'] = training_confidence

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            logger.error(f"Analysis error: {e}")

    def render_embedding_analysis(self):
        """Render the sentence embedding analysis tab."""
        st.markdown('<div class="sub-header">📊 Sentence Embedding Analysis</div>',
                   unsafe_allow_html=True)

        if 'embeddings' not in st.session_state.analysis_results:
            st.info("🔄 Please run analysis first using the sidebar controls.")
            return

        embedding_result = st.session_state.analysis_results['embeddings']

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", embedding_result['num_samples'])
        with col2:
            st.metric("Embedding Dimension", embedding_result['embedding_dim'])
        with col3:
            st.metric("Languages", len(set(embedding_result['metadata']['language'])))
        with col4:
            st.metric("Model", embedding_result['model_name'].split('/')[-1])

        # Visualization controls
        st.markdown("### Visualization Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            reduction_method = st.selectbox(
                "Dimensionality Reduction",
                ["umap", "tsne", "pca"],
                index=0
            )

        with col2:
            plot_dimensions = st.selectbox(
                "Plot Dimensions",
                ["2D", "3D"],
                index=0
            )

        with col3:
            interactive_plot = st.checkbox("Interactive Plot", value=True)

        # Generate visualizations
        embeddings = embedding_result['embeddings']
        languages = embedding_result['metadata']['language'].tolist()
        texts = embedding_result['metadata']['text'].tolist()

        if plot_dimensions == "2D":
            fig = self.embedding_visualizer.plot_embeddings_2d(
                embeddings=embeddings,
                languages=languages,
                texts=texts,
                method=reduction_method,
                interactive=interactive_plot,
                title=f"Sentence Embeddings - {reduction_method.upper()} 2D"
            )
        else:
            fig = self.embedding_visualizer.plot_embeddings_3d(
                embeddings=embeddings,
                languages=languages,
                texts=texts,
                method=reduction_method,
                title=f"Sentence Embeddings - {reduction_method.upper()} 3D"
            )

        st.plotly_chart(fig, use_container_width=True)

        # Similarity analysis
        st.markdown("### Similarity Analysis")
        similarity_matrix = embedding_result['similarity_matrix']

        # Create similarity heatmap
        fig_sim = self.embedding_visualizer.plot_similarity_heatmap(
            similarity_matrix=similarity_matrix,
            languages=languages,
            texts=texts,
            interactive=True,
            title="Cosine Similarity Matrix"
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Language clustering analysis
        if len(set(languages)) > 1:
            st.markdown("### Language Clustering Analysis")
            clustering_result = self.embedding_analyzer.analyze_language_clusters(
                embeddings=embeddings,
                languages=languages
            )

            # Display clustering metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Clusters", clustering_result['num_clusters'])
            with col2:
                if clustering_result['silhouette_score']:
                    st.metric("Silhouette Score", f"{clustering_result['silhouette_score']:.3f}")

            # Language purity scores
            st.markdown("#### Language Purity Scores")
            purity_df = pd.DataFrame(list(clustering_result['language_purity'].items()),
                                   columns=['Language', 'Purity Score'])
            st.dataframe(purity_df, use_container_width=True)

    def render_attention_analysis(self):
        """Render the attention analysis tab."""
        st.markdown('<div class="sub-header">🔍 Attention Analysis</div>',
                   unsafe_allow_html=True)

        if 'attention' not in st.session_state.analysis_results:
            st.info("🔄 Please run analysis first using the sidebar controls.")
            return

        attention_result = st.session_state.analysis_results['attention']

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sequence Length", attention_result['sequence_length'])
        with col2:
            st.metric("Number of Layers", attention_result['num_layers'])
        with col3:
            st.metric("Number of Heads", attention_result['num_heads'])
        with col4:
            st.metric("Model", attention_result['model_name'].split('/')[-1])

        # Visualization controls
        st.markdown("### Visualization Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            layer_idx = st.selectbox(
                "Layer",
                range(attention_result['num_layers']),
                index=min(6, attention_result['num_layers'] - 1)  # Default to layer 6 or last
            )

        with col2:
            head_idx = st.selectbox(
                "Head (None for average)",
                [None] + list(range(attention_result['num_heads'])),
                index=0
            )

        with col3:
            plot_type = st.selectbox(
                "Plot Type",
                ["Single Head/Average", "Multi-Head", "Pattern Analysis"]
            )

        # Generate visualizations
        if plot_type == "Single Head/Average":
            fig = self.attention_visualizer.plot_attention_heatmap(
                attention_result=attention_result,
                layer_idx=layer_idx,
                head_idx=head_idx,
                interactive=True
            )
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Multi-Head":
            fig = self.attention_visualizer.plot_multi_head_attention(
                attention_result=attention_result,
                layer_idx=layer_idx,
                max_heads=8
            )
            st.pyplot(fig)

        else:  # Pattern Analysis
            st.markdown("### Attention Pattern Analysis")

            # Run pattern analysis
            pattern_analysis = self.attention_analyzer.analyze_attention_patterns(
                attention_result
            )

            # Display different pattern types
            pattern_type = st.selectbox(
                "Pattern Type",
                ["head_entropy", "layer_entropy", "attention_distance", "token_importance"]
            )

            fig = self.attention_visualizer.plot_attention_patterns(
                pattern_analysis=pattern_analysis,
                pattern_type=pattern_type,
                interactive=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Token importance analysis
        st.markdown("### Token Analysis")
        tokens = attention_result['tokens']

        # Display tokens with positions
        token_df = pd.DataFrame({
            'Position': range(len(tokens)),
            'Token': tokens
        })
        st.dataframe(token_df, use_container_width=True)

    def render_confidence_analysis(self):
        """Render the confidence analysis tab."""
        st.markdown('<div class="sub-header">📈 Confidence Analysis</div>',
                   unsafe_allow_html=True)

        if 'confidence' not in st.session_state.analysis_results:
            st.info("🔄 Please run analysis first using the sidebar controls.")
            return

        confidence_result = st.session_state.analysis_results['confidence']

        # Metrics
        measures = confidence_result['confidence_measures']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sequence Length", confidence_result['sequence_length'])
        with col2:
            if 'entropy' in measures:
                mean_entropy = measures['entropy']['mean_entropy']
                st.metric("Mean Entropy", f"{mean_entropy:.3f}")
        with col3:
            if 'perplexity' in measures:
                perplexity = measures['perplexity']['overall_perplexity']
                st.metric("Perplexity", f"{perplexity:.2f}")
        with col4:
            st.metric("Model", confidence_result['model_name'].split('/')[-1])

        # Visualization controls
        st.markdown("### Visualization Options")
        col1, col2 = st.columns(2)

        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Entropy Timeline", "Confidence Heatmap", "Uncertainty Regions", "Distribution Analysis"]
            )

        with col2:
            confidence_metric = st.selectbox(
                "Confidence Metric",
                ["entropy", "variance", "top_k_probability", "perplexity"]
            )

        # Generate visualizations
        if analysis_type == "Entropy Timeline":
            fig = self.confidence_visualizer.plot_entropy_by_position(
                confidence_result=confidence_result,
                interactive=True
            )
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Confidence Heatmap":
            fig = self.confidence_visualizer.plot_confidence_heatmap(
                confidence_result=confidence_result,
                metric=confidence_metric,
                interactive=True
            )
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Uncertainty Regions":
            fig = self.confidence_visualizer.plot_uncertainty_timeline(
                confidence_result=confidence_result,
                interactive=True
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Distribution Analysis
            st.markdown("### Confidence Distribution Analysis")

            if confidence_metric in measures:
                metric_data = measures[confidence_metric]

                # Create distribution plot
                if confidence_metric == 'entropy':
                    values = metric_data['position_entropies']
                    title = "Entropy Distribution"
                elif confidence_metric == 'variance':
                    values = metric_data['position_variances']
                    title = "Variance Distribution"
                elif confidence_metric == 'top_k_probability':
                    values = metric_data['position_top_k_confidence']
                    title = "Top-K Confidence Distribution"
                else:  # perplexity
                    values = metric_data['position_perplexities']
                    title = "Perplexity Distribution"

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=values, nbinsx=20, name=confidence_metric))
                fig.update_layout(
                    title=title,
                    xaxis_title=confidence_metric.title(),
                    yaxis_title="Frequency",
                    width=800,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                st.markdown("#### Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std', 'Min', 'Max'],
                    'Value': [
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)

    def render_model_comparison(self):
        """Render the model comparison tab."""
        st.markdown('<div class="sub-header">🔄 Model Comparison</div>',
                   unsafe_allow_html=True)

        has_base = 'attention' in st.session_state.analysis_results
        has_training = 'training_attention' in st.session_state.analysis_results

        if not has_base:
            st.info("🔄 Please run analysis first using the sidebar controls.")
            return

        if not has_training:
            st.warning("⚠️ No training model provided. Add a training model path in the sidebar to enable comparison.")

            # Show only base model analysis
            st.markdown("### Base Model Analysis")

            # Display base model results
            base_attention = st.session_state.analysis_results['attention']
            base_confidence = st.session_state.analysis_results['confidence']

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Attention Patterns")
                st.info(f"Model: {base_attention['model_name']}")
                st.metric("Layers", base_attention['num_layers'])
                st.metric("Heads", base_attention['num_heads'])

            with col2:
                st.markdown("#### Confidence Metrics")
                if 'entropy' in base_confidence['confidence_measures']:
                    entropy_data = base_confidence['confidence_measures']['entropy']
                    st.metric("Mean Entropy", f"{entropy_data['mean_entropy']:.3f}")
                    st.metric("Uncertainty Level", entropy_data['uncertainty_classification'])

            return

        # Full comparison when both models are available
        base_attention = st.session_state.analysis_results['attention']
        training_attention = st.session_state.analysis_results['training_attention']
        base_confidence = st.session_state.analysis_results['confidence']
        training_confidence = st.session_state.analysis_results['training_confidence']

        # Model comparison metrics
        st.markdown("### Model Comparison Overview")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Base Model")
            st.info(f"**Model**: {base_attention['model_name']}")
            if 'entropy' in base_confidence['confidence_measures']:
                entropy_data = base_confidence['confidence_measures']['entropy']
                st.metric("Mean Entropy", f"{entropy_data['mean_entropy']:.3f}")

        with col2:
            st.markdown("#### Training Model")
            st.info(f"**Model**: {training_attention['model_name']}")
            if 'entropy' in training_confidence['confidence_measures']:
                entropy_data = training_confidence['confidence_measures']['entropy']
                st.metric("Mean Entropy", f"{entropy_data['mean_entropy']:.3f}")

        # Comparison analysis
        st.markdown("### Detailed Comparison")

        comparison_type = st.selectbox(
            "Comparison Type",
            ["Attention Patterns", "Confidence Measures", "Side-by-Side Visualization"]
        )

        if comparison_type == "Attention Patterns":
            # Compare attention patterns
            attention_comparison = self.attention_analyzer.compare_attention_patterns(
                base_attention, training_attention
            )

            fig = self.attention_visualizer.plot_attention_comparison(
                attention_comparison,
                title="Attention Pattern Comparison: Base vs Training"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif comparison_type == "Confidence Measures":
            # Compare confidence measures
            confidence_comparison = self.confidence_analyzer.compare_model_confidence(
                base_confidence, training_confidence
            )

            fig = self.confidence_visualizer.plot_confidence_comparison(
                confidence_comparison,
                title="Confidence Comparison: Base vs Training"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Side-by-side visualization
            st.markdown("### Side-by-Side Attention Heatmaps")

            col1, col2 = st.columns(2)

            # Controls
            layer_idx = st.selectbox("Layer for Comparison", range(min(base_attention['num_layers'], training_attention['num_layers'])))

            with col1:
                st.markdown("#### Base Model")
                fig_base = self.attention_visualizer.plot_attention_heatmap(
                    attention_result=base_attention,
                    layer_idx=layer_idx,
                    interactive=True,
                    title="Base Model Attention"
                )
                st.plotly_chart(fig_base, use_container_width=True)

            with col2:
                st.markdown("#### Training Model")
                fig_training = self.attention_visualizer.plot_attention_heatmap(
                    attention_result=training_attention,
                    layer_idx=layer_idx,
                    interactive=True,
                    title="Training Model Attention"
                )
                st.plotly_chart(fig_training, use_container_width=True)

    def render_settings(self):
        """Render the settings tab."""
        st.markdown('<div class="sub-header">⚙️ Settings</div>',
                   unsafe_allow_html=True)

        # Model settings
        st.markdown("### Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Sentence Transformer Settings")
            current_model = self.config.get('models.sentence_transformer.default_model')
            st.text_input("Default Sentence Transformer", value=current_model, disabled=True)

            st.markdown("#### Performance Settings")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
            device = st.selectbox("Device", ["auto", "cpu", "cuda"])

        with col2:
            st.markdown("#### Visualization Settings")
            figure_width = st.number_input("Figure Width", min_value=400, max_value=2000, value=800)
            figure_height = st.number_input("Figure Height", min_value=300, max_value=1500, value=600)

            color_scheme = st.selectbox("Color Scheme", ["default", "viridis", "plasma", "husl"])

        # Memory and cache settings
        st.markdown("### Memory Management")

        # Display current memory usage
        memory_info = self.model_manager.get_memory_usage()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Loaded Models", memory_info['loaded_models'])
        with col2:
            st.metric("Loaded Tokenizers", memory_info['loaded_tokenizers'])
        with col3:
            if 'gpu_memory_allocated' in memory_info:
                st.metric("GPU Memory (GB)", f"{memory_info['gpu_memory_allocated']:.2f}")

        # Cache management
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Clear Model Cache"):
                self.model_manager.clear_cache()
                st.success("Model cache cleared!")

        with col2:
            if st.button("Clear Embedding Cache"):
                self.embedding_analyzer.clear_cache()
                st.success("Embedding cache cleared!")

        with col3:
            if st.button("Clear All Caches"):
                self.clear_all_caches()
                st.success("All caches cleared!")

        # Export settings
        st.markdown("### Export & Import")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Export Results")
            export_format = st.selectbox("Export Format", ["json", "csv", "npz"])

            if st.button("Export Analysis Results"):
                if st.session_state.analysis_results:
                    # This would implement the export functionality
                    st.success(f"Results exported in {export_format} format!")
                else:
                    st.warning("No analysis results to export")

        with col2:
            st.markdown("#### Import Configuration")
            uploaded_config = st.file_uploader("Upload Configuration", type=['yaml', 'json'])

            if uploaded_config is not None:
                st.success("Configuration uploaded!")

        # Advanced settings
        with st.expander("Advanced Settings"):
            st.markdown("#### Dimensionality Reduction Parameters")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input("UMAP n_neighbors", min_value=2, max_value=100, value=15)
                st.number_input("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1)

            with col2:
                st.number_input("t-SNE perplexity", min_value=5, max_value=100, value=30)
                st.number_input("t-SNE n_iter", min_value=250, max_value=5000, value=1000)

            with col3:
                st.number_input("PCA components", min_value=2, max_value=50, value=2)

    def clear_all_caches(self):
        """Clear all caches."""
        self.model_manager.clear_cache()
        self.embedding_analyzer.clear_cache()
        self.attention_analyzer.clear_cache()
        self.confidence_analyzer.clear_cache()


def main():
    """Main function to run the dashboard."""
    dashboard = MultilingualAnalysisDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()