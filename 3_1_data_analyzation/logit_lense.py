"""
Logit Lens Analysis for Transformer Language Models

This module implements logit lens analysis to visualize how transformer models
progressively refine their next-token predictions across different layers.

The logit lens technique applies the model's final language modeling head
(unembedding matrix) to intermediate layer activations, revealing how token
prediction distributions evolve throughout the network's processing hierarchy.

Key Features:
- Layer-wise next-token prediction analysis
- Heatmap visualizations with entropy analysis
- Support for multiple model architectures
- Configurable visualization options
- Memory-efficient processing for large models

Author: Based on research from nostalgebraist (2020) and recent advances
Date: 2025-09-28
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings


class LogitLens:
    """
    Logit Lens analyzer for transformer language models.

    This class provides methods to analyze how transformer models build up
    their predictions layer by layer, offering insights into the model's
    internal computational process.
    """

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the LogitLens analyzer.

        Args:
            model_name: Name of the model to analyze (HuggingFace model ID)
            device: Device to run the model on ("auto", "cpu", "cuda", etc.)
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if device == "auto" else None
        )

        if device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_layers = len(self.model.transformer.h) if hasattr(self.model, 'transformer') else len(self.model.model.layers)
        print(f"Model loaded with {self.num_layers} layers")

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def analyze_prompt(self,
                      prompt: str,
                      layer_range: Optional[Tuple[int, int]] = None,
                      top_k: int = 10) -> Dict:
        """
        Analyze how the model's predictions evolve across layers for a given prompt.

        Args:
            prompt: Input text to analyze
            layer_range: Tuple of (start_layer, end_layer) to analyze. If None, analyzes all layers
            top_k: Number of top predictions to track per layer

        Returns:
            Dictionary containing analysis results
        """
        print(f"Analyzing prompt: '{prompt}'")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get hidden states from all layers
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # (num_layers + 1, batch_size, seq_len, hidden_size)

        # Determine layer range
        if layer_range is None:
            start_layer, end_layer = 0, self.num_layers
        else:
            start_layer, end_layer = layer_range

        # Extract predictions for each layer
        layer_predictions = []
        layer_probabilities = []
        layer_entropies = []

        for layer_idx in range(start_layer, min(end_layer + 1, len(hidden_states))):
            hidden_state = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_size)

            # Apply language modeling head
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(hidden_state)
            elif hasattr(self.model, 'embed_out'):
                logits = self.model.embed_out(hidden_state)
            else:
                # Try to find the output layer
                for name, module in self.model.named_modules():
                    if 'lm_head' in name or 'embed_out' in name or 'output' in name:
                        logits = module(hidden_state)
                        break
                else:
                    raise ValueError("Could not find language modeling head")

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

            # Calculate entropy (uncertainty measure)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (batch_size, seq_len)

            # Get top-k predictions for the last token
            last_token_logits = logits[0, -1, :]  # (vocab_size,)
            last_token_probs = probs[0, -1, :]    # (vocab_size,)

            top_k_values, top_k_indices = torch.topk(last_token_probs, top_k)
            top_k_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices]

            layer_predictions.append({
                'layer': layer_idx,
                'top_k_tokens': top_k_tokens,
                'top_k_probs': top_k_values.cpu().numpy(),
                'top_k_indices': top_k_indices.cpu().numpy()
            })

            layer_probabilities.append(probs.cpu())
            layer_entropies.append(entropy.cpu())

        # Prepare tokens for visualization
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return {
            'prompt': prompt,
            'input_tokens': input_tokens,
            'layer_predictions': layer_predictions,
            'layer_probabilities': layer_probabilities,
            'layer_entropies': layer_entropies,
            'layer_range': (start_layer, min(end_layer, len(hidden_states) - 1)),
            'input_ids': inputs['input_ids'].cpu()
        }

    def visualize_predictions(self,
                            analysis_results: Dict,
                            candidate_tokens: Optional[List[str]] = None,
                            num_candidates: int = 5,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> None:
        """
        Create heatmap visualization of prediction evolution across layers.

        Args:
            analysis_results: Results from analyze_prompt()
            candidate_tokens: Specific tokens to track across layers (if None, uses top predictions)
            num_candidates: Number of top candidate tokens to show (when candidate_tokens is None)
            figsize: Figure size for the plot
            save_path: Path to save the figure (optional)
        """
        layer_predictions = analysis_results['layer_predictions']
        input_tokens = analysis_results['input_tokens']
        layer_range = analysis_results['layer_range']

        if candidate_tokens is None:
            # Use top predictions from the final layer as candidates
            final_layer = layer_predictions[-1]
            candidate_tokens = final_layer['top_k_tokens'][:num_candidates]

        # Get candidate token IDs
        candidate_ids = []
        for token in candidate_tokens:
            try:
                # Handle subword tokens properly
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    candidate_ids.append(token_ids[0])
                else:
                    # Fallback: try to find token in vocabulary
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    candidate_ids.append(token_id)
            except:
                warnings.warn(f"Could not find token ID for '{token}', skipping")
                continue

        # Create heatmap data: layers × candidate_tokens
        heatmap_data = []
        layer_labels = []

        for pred in layer_predictions:
            layer_idx = pred['layer']
            layer_labels.append(f"Layer {layer_idx}")

            # Get probabilities for candidate tokens
            row_data = []
            for token_id in candidate_ids:
                if token_id in pred['top_k_indices']:
                    idx = np.where(pred['top_k_indices'] == token_id)[0]
                    if len(idx) > 0:
                        prob = pred['top_k_probs'][idx[0]]
                    else:
                        prob = 0.0
                else:
                    prob = 0.0
                row_data.append(prob)

            heatmap_data.append(row_data)

        # Create the heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})

        # Main heatmap
        heatmap_array = np.array(heatmap_data)
        im = ax1.imshow(heatmap_array, aspect='auto', cmap='viridis', origin='lower')

        # Set labels and ticks
        ax1.set_yticks(range(len(layer_labels)))
        ax1.set_yticklabels(layer_labels)
        ax1.set_xticks(range(len(candidate_tokens)))
        ax1.set_xticklabels(candidate_tokens, rotation=45, ha='right')
        ax1.set_xlabel('Candidate Tokens')
        ax1.set_ylabel('Model Layers')
        ax1.set_title(f'Logit Lens: Next-Token Prediction Evolution\nPrompt: "{analysis_results["prompt"]}"')

        # Add colorbar
        plt.colorbar(im, ax=ax1, label='Probability')

        # Add probability values as text
        for i in range(len(layer_labels)):
            for j in range(len(candidate_tokens)):
                text = ax1.text(j, i, f'{heatmap_array[i, j]:.3f}',
                              ha="center", va="center", color="white" if heatmap_array[i, j] < 0.5 else "black",
                              fontsize=8)

        # Entropy subplot
        entropies = analysis_results['layer_entropies']
        if entropies:
            # Calculate mean entropy across tokens for each layer
            mean_entropies = []
            for entropy_tensor in entropies:
                mean_entropy = entropy_tensor.mean().item()
                mean_entropies.append(mean_entropy)

            ax2.plot(mean_entropies, range(len(mean_entropies)), 'bo-')
            ax2.set_ylabel('Model Layers')
            ax2.set_xlabel('Mean Entropy')
            ax2.set_title('Prediction Uncertainty')
            ax2.grid(True, alpha=0.3)
            ax2.set_yticks(range(len(layer_labels)))
            ax2.set_yticklabels(layer_labels)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def compare_predictions(self,
                          prompt: str,
                          layer_indices: List[int],
                          top_k: int = 5) -> None:
        """
        Compare predictions across specific layers in a tabular format.

        Args:
            prompt: Input text to analyze
            layer_indices: List of layer indices to compare
            top_k: Number of top predictions to show per layer
        """
        analysis_results = self.analyze_prompt(prompt, top_k=top_k)
        layer_predictions = analysis_results['layer_predictions']

        print(f"\nPrediction Comparison for: '{prompt}'")
        print("=" * 80)

        # Filter predictions for requested layers
        filtered_predictions = [
            pred for pred in layer_predictions
            if pred['layer'] in layer_indices
        ]

        # Create comparison table
        for pred in filtered_predictions:
            print(f"\nLayer {pred['layer']}:")
            print("-" * 40)
            for i, (token, prob) in enumerate(zip(pred['top_k_tokens'], pred['top_k_probs'])):
                print(f"{i+1:2d}. {token:15s} ({prob:.4f})")

    def track_token_evolution(self,
                            prompt: str,
                            target_token: str,
                            show_plot: bool = True) -> Dict:
        """
        Track how the probability of a specific token evolves across layers.

        Args:
            prompt: Input text to analyze
            target_token: Token to track across layers
            show_plot: Whether to display a plot of the evolution

        Returns:
            Dictionary with evolution data
        """
        analysis_results = self.analyze_prompt(prompt, top_k=50)  # Use larger top_k for tracking
        layer_predictions = analysis_results['layer_predictions']

        # Get target token ID
        try:
            target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
            if target_ids:
                target_id = target_ids[0]
            else:
                target_id = self.tokenizer.convert_tokens_to_ids(target_token)
        except:
            raise ValueError(f"Could not find token ID for '{target_token}'")

        # Track probability evolution
        layers = []
        probabilities = []

        for pred in layer_predictions:
            layers.append(pred['layer'])

            # Find probability for target token
            if target_id in pred['top_k_indices']:
                idx = np.where(pred['top_k_indices'] == target_id)[0]
                if len(idx) > 0:
                    prob = pred['top_k_probs'][idx[0]]
                else:
                    prob = 0.0
            else:
                prob = 0.0

            probabilities.append(prob)

        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(layers, probabilities, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Layer Index')
            plt.ylabel('Probability')
            plt.title(f'Evolution of Token "{target_token}" Probability\nPrompt: "{prompt}"')
            plt.grid(True, alpha=0.3)
            plt.xticks(layers)

            # Highlight max probability
            max_idx = np.argmax(probabilities)
            plt.axvline(x=layers[max_idx], color='red', linestyle='--', alpha=0.7,
                       label=f'Max at Layer {layers[max_idx]}')
            plt.legend()

            plt.tight_layout()
            plt.show()

        return {
            'target_token': target_token,
            'layers': layers,
            'probabilities': probabilities,
            'max_probability': max(probabilities),
            'max_layer': layers[np.argmax(probabilities)]
        }

    def create_detailed_heatmap(self,
                              analysis_results: Dict,
                              num_candidates: int = 10,
                              figsize: Tuple[int, int] = (16, 12),
                              save_path: Optional[str] = None) -> None:
        """
        Create a detailed heatmap showing top predictions for each layer.

        Args:
            analysis_results: Results from analyze_prompt()
            num_candidates: Number of top candidates to show
            figsize: Figure size for the plot
            save_path: Path to save the figure (optional)
        """
        layer_predictions = analysis_results['layer_predictions']

        # Collect all unique tokens from all layers
        all_tokens = set()
        for pred in layer_predictions:
            all_tokens.update(pred['top_k_tokens'][:num_candidates])

        all_tokens = list(all_tokens)

        # Create probability matrix: layers × tokens
        prob_matrix = np.zeros((len(layer_predictions), len(all_tokens)))
        layer_labels = []

        for i, pred in enumerate(layer_predictions):
            layer_labels.append(f"Layer {pred['layer']}")

            for j, token in enumerate(all_tokens):
                if token in pred['top_k_tokens']:
                    idx = pred['top_k_tokens'].index(token)
                    prob_matrix[i, j] = pred['top_k_probs'][idx]

        # Create the detailed heatmap
        plt.figure(figsize=figsize)
        im = plt.imshow(prob_matrix, aspect='auto', cmap='viridis', origin='lower')

        # Set labels
        plt.yticks(range(len(layer_labels)), layer_labels)
        plt.xticks(range(len(all_tokens)), all_tokens, rotation=45, ha='right')
        plt.xlabel('Predicted Tokens')
        plt.ylabel('Model Layers')
        plt.title(f'Detailed Logit Lens Analysis\nPrompt: "{analysis_results["prompt"]}"')

        # Add colorbar
        plt.colorbar(im, label='Probability')

        # Add probability values as text for high probabilities
        for i in range(len(layer_labels)):
            for j in range(len(all_tokens)):
                if prob_matrix[i, j] > 0.1:  # Only show significant probabilities
                    plt.text(j, i, f'{prob_matrix[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if prob_matrix[i, j] < 0.5 else "black",
                           fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed heatmap saved to: {save_path}")

        plt.show()

    def create_line_plot(self,
                        analysis_results: Dict,
                        num_candidates: int = 5,
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None) -> None:
        """
        Create line plots showing probability evolution for top candidates.

        Args:
            analysis_results: Results from analyze_prompt()
            num_candidates: Number of top candidates to track
            figsize: Figure size for the plot
            save_path: Path to save the figure (optional)
        """
        layer_predictions = analysis_results['layer_predictions']

        # Get top candidates from final layer
        final_layer = layer_predictions[-1]
        top_tokens = final_layer['top_k_tokens'][:num_candidates]

        plt.figure(figsize=figsize)

        # Plot each token's evolution
        for token in top_tokens:
            layers = []
            probabilities = []

            for pred in layer_predictions:
                layers.append(pred['layer'])

                if token in pred['top_k_tokens']:
                    idx = pred['top_k_tokens'].index(token)
                    prob = pred['top_k_probs'][idx]
                else:
                    prob = 0.0

                probabilities.append(prob)

            plt.plot(layers, probabilities, 'o-', linewidth=2, markersize=6, label=f'"{token}"')

        plt.xlabel('Layer Index')
        plt.ylabel('Probability')
        plt.title(f'Token Probability Evolution\nPrompt: "{analysis_results["prompt"]}"')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(layers)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Line plot saved to: {save_path}")

        plt.show()


def example_usage():
    """
    Example usage of the LogitLens analyzer.
    """
    print("LogitLens Example Usage")
    print("=" * 50)

    # Initialize analyzer (using a smaller model for demo)
    model_name = "gpt2"  # Change to your preferred model
    lens = LogitLens(model_name)

    # Example 1: Basic analysis
    prompt = "The capital of France is"
    results = lens.analyze_prompt(prompt)
    lens.visualize_predictions(results, candidate_tokens=["Paris", "London", "the", "a"])

    # Example 2: Compare specific layers
    lens.compare_predictions(prompt, layer_indices=[0, 6, 11], top_k=5)

    # Example 3: Track specific token
    evolution = lens.track_token_evolution(prompt, "Paris")
    print(f"\nToken 'Paris' reaches max probability of {evolution['max_probability']:.4f} at layer {evolution['max_layer']}")


if __name__ == "__main__":
    # Run example if script is executed directly
    example_usage()