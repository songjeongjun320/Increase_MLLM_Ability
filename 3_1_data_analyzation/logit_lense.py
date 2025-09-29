"""
Logit Lens Analysis for Transformer Language Models - Simplified Version

This module implements logit lens analysis to visualize how transformer models
progressively refine their next-token predictions across different layers.

Author: Based on research from nostalgebraist (2020) and recent advances
Date: 2025-09-28
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager
import seaborn as sns
from typing import List, Optional, Dict, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import platform
import subprocess

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def setup_fonts():
    """
    Simple font setup - if Korean fonts fail, just continue with defaults.
    """
    # Disable font warnings
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    # Basic matplotlib settings
    plt.rcParams['axes.unicode_minus'] = False
    
    # Try to use any available font without specific Korean font requirements
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    
    print("Font setup complete - warnings suppressed")
    return True


class LogitLens:
    """
    Logit Lens analyzer for transformer language models.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the LogitLens analyzer.
        
        Args:
            model_path: Path to the model
            device: Device to run the model on
        """
        # Simple font setup
        setup_fonts()
        
        self.model_path = model_path
        self.device = self._get_device(device)
        
        # Determine if it's a local path or HuggingFace model ID
        is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)
        
        # Load model and tokenizer
        if is_local_path:
            print(f"Loading model from local path: {model_path}")
        else:
            print(f"Loading model from HuggingFace Hub: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if device == "auto" else None,
            local_files_only=is_local_path
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
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 GPU detected: {gpu_name}")
                print(f"⚡ CUDA version: {torch.version.cuda}")
                print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                if "A100" in gpu_name:
                    print("🔥 A100 detected! Enabling optimizations...")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                
                return torch.device("cuda")
            else:
                print("⚠️ CUDA not available, using CPU")
                return torch.device("cpu")
        return torch.device(device)

    def _decode_token_safe(self, token_id: int) -> str:
        """
        Safely decode a token ID to a readable string.
        Returns the actual token text without any translation.
        """
        try:
            # Try direct decode
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            if decoded and decoded.strip():
                return decoded.strip()
            
            # Fallback to token string
            token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            
            # Handle special token prefixes
            if token_str.startswith('▁'):
                return token_str[1:]  # Remove SentencePiece space marker
            elif token_str.startswith('##'):
                return token_str[2:]  # Remove BERT subword marker
            elif token_str.startswith('<') and token_str.endswith('>'):
                return token_str  # Keep special tokens as-is
            else:
                return token_str
                
        except Exception:
            # If all else fails, return token ID
            return f"[{token_id}]"

    def analyze_prompt_all_positions(self,
                                   prompt: str,
                                   layer_range: Optional[Tuple[int, int]] = None,
                                   top_k: int = 5) -> Dict:
        """
        Analyze predictions for ALL token positions across layers.
        """
        print(f"Analyzing all token positions for: '{prompt}'")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get hidden states from all layers
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        # Determine layer range
        if layer_range is None:
            start_layer, end_layer = 0, self.num_layers
        else:
            start_layer, end_layer = layer_range
        
        seq_len = inputs['input_ids'].shape[1]
        
        # Extract predictions for each position and each layer
        position_predictions = []
        
        for pos_idx in range(seq_len):
            layer_predictions_for_position = []
            
            for layer_idx in range(start_layer, min(end_layer + 1, len(hidden_states))):
                hidden_state = hidden_states[layer_idx]
                position_hidden = hidden_state[0, pos_idx, :]
                
                # Apply language modeling head
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(position_hidden.unsqueeze(0))
                elif hasattr(self.model, 'embed_out'):
                    logits = self.model.embed_out(position_hidden.unsqueeze(0))
                else:
                    for name, module in self.model.named_modules():
                        if 'lm_head' in name or 'embed_out' in name or 'output' in name:
                            logits = module(position_hidden.unsqueeze(0))
                            break
                    else:
                        raise ValueError("Could not find language modeling head")
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)[0]
                
                # Get top-k predictions
                top_k_values, top_k_indices = torch.topk(probs, top_k)
                top_k_tokens = []
                for idx in top_k_indices:
                    token_id = idx.item()
                    display_token = self._decode_token_safe(token_id)
                    top_k_tokens.append(display_token)
                
                layer_predictions_for_position.append({
                    'layer': layer_idx,
                    'position': pos_idx,
                    'top_tokens': top_k_tokens,
                    'top_probs': top_k_values.detach().cpu().numpy(),
                    'top_indices': top_k_indices.detach().cpu().numpy()
                })
            
            position_predictions.append(layer_predictions_for_position)
        
        # Prepare tokens for visualization
        input_tokens = []
        for token_id in inputs['input_ids'][0]:
            token_id_val = token_id.item()
            display_token = self._decode_token_safe(token_id_val)
            input_tokens.append(display_token)
        
        return {
            'prompt': prompt,
            'input_tokens': input_tokens,
            'position_predictions': position_predictions,
            'layer_range': (start_layer, min(end_layer, len(hidden_states) - 1)),
            'input_ids': inputs['input_ids'].detach().cpu()
        }

    def create_token_position_heatmap(self,
                                    analysis_results: Dict,
                                    figsize: Tuple[int, int] = (20, 12),
                                    save_path: Optional[str] = None,
                                    show_text: bool = True) -> None:
        """
        Create heatmap visualization matching the desired format.
        
        Args:
            analysis_results: Results from analyze_prompt_all_positions()
            figsize: Figure size for the plot
            save_path: Path to save the figure
            show_text: Whether to show token text in cells
        """
        if 'position_predictions' not in analysis_results:
            raise ValueError("This visualization requires results from analyze_prompt_all_positions()")
        
        position_predictions = analysis_results['position_predictions']
        input_tokens = analysis_results['input_tokens']
        layer_range = analysis_results['layer_range']
        
        num_layers = len(position_predictions[0])
        num_positions = len(position_predictions)
        
        # Suppress warnings for this plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Prepare data
            predicted_tokens_matrix = []
            probability_matrix = []
            
            # Collect predictions for each layer
            for layer_idx in range(num_layers):
                layer_tokens = []
                layer_probs = []
                
                for pos_idx in range(num_positions):
                    if layer_idx < len(position_predictions[pos_idx]):
                        pred = position_predictions[pos_idx][layer_idx]
                        if pred['top_tokens']:
                            top_token = pred['top_tokens'][0]
                            top_prob = pred['top_probs'][0]
                        else:
                            top_token = ""
                            top_prob = 0.0
                    else:
                        top_token = ""
                        top_prob = 0.0
                    
                    layer_tokens.append(top_token)
                    layer_probs.append(top_prob)
                
                predicted_tokens_matrix.append(layer_tokens)
                probability_matrix.append(layer_probs)
            
            # Create the heatmap
            prob_array = np.array(probability_matrix)
            
            # Use a blue-based colormap for better visibility
            im = ax.imshow(prob_array, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            
            # Set title
            ax.set_title(f'Logit Lens Analysis\nPrompt: "{analysis_results["prompt"]}"', 
                        fontsize=14, pad=40)
            
            # Set y-axis (layers)
            ax.set_ylabel('Model Layer\n← closer to input        closer to output →', fontsize=12)
            layer_labels = [f"{i+layer_range[0]}" for i in range(num_layers)]
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels(layer_labels, fontsize=10)
            
            # Set x-axis for output tokens (bottom)
            ax.set_xticks(range(num_positions))
            
            # Get output tokens from the last layer
            output_labels = []
            for pos_idx in range(num_positions):
                last_layer_pred = position_predictions[pos_idx][-1]
                if last_layer_pred['top_tokens']:
                    output_labels.append(last_layer_pred['top_tokens'][0])
                else:
                    output_labels.append("")
            
            ax.set_xticklabels(output_labels, rotation=45, ha='right', fontsize=10)
            ax.set_xlabel('Output Tokens (Predictions)', fontsize=12, labelpad=10)
            
            # Add input tokens on top
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(range(num_positions))
            ax2.set_xticklabels(input_tokens, rotation=45, ha='left', fontsize=10)
            ax2.set_xlabel('Input Tokens', fontsize=12, labelpad=10)
            
            # Optionally add text annotations in cells
            if show_text:
                for i in range(num_layers):
                    for j in range(num_positions):
                        token = predicted_tokens_matrix[i][j]
                        prob = probability_matrix[i][j]
                        
                        if token and prob > 0.05:  # Only show significant predictions
                            # Choose text color based on background
                            text_color = "white" if prob > 0.5 else "black"
                            
                            # Truncate long tokens
                            display_token = token[:10] + "..." if len(token) > 10 else token
                            
                            # Add token text (smaller font for readability)
                            ax.text(j, i, display_token,
                                   ha="center", va="center",
                                   color=text_color, fontsize=7, fontweight='normal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability', fontsize=10)
            
            # Grid for better readability
            ax.set_xticks(np.arange(num_positions) - 0.5, minor=True)
            ax.set_yticks(np.arange(num_layers) - 0.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
            ax.tick_params(which="minor", size=0)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Token position heatmap saved to: {save_path}")
            
            plt.show()


def main():
    """
    Main function to run logit lens analysis.
    """
    # Configuration
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt"
    
    # You can change the prompt here to anything you want
    # 여기서 프롬프트를 원하는 대로 바꿀 수 있습니다
    prompt = "La bateau naviguait en doceur sur"
    # prompt = "The capital of France is"  # English example
    # prompt = "인공지능의 미래는"  # Another Korean example
    
    print(f"Using model: {model_path}")
    
    # Initialize LogitLens
    lens = LogitLens(model_path)
    
    # Analyze prompt
    print(f"\nAnalyzing prompt: '{prompt}'")
    results = lens.analyze_prompt_all_positions(prompt)
    
    # Create visualization
    print("Creating visualization...")
    lens.create_token_position_heatmap(
        results, 
        save_path="logit_lens_result.png",
        show_text=True  # Set to False if you don't want text in cells
    )
    
    print("\n✅ Complete! Check logit_lens_result.png")


if __name__ == "__main__":
    main()