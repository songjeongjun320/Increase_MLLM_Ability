"""
Logit Lens Analysis for Transformer Language Models - Simplified Version

This module implements logit lens analysis to visualize how transformer models
progressively refine their next-token predictions across different layers.

Author: Based on research from nostalgebraist (2020) and recent advances
Date: 2025-09-28

module load cuda-12.6.1-gcc-12.1.0
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
    Setup fonts for Korean text support in matplotlib.
    If no Korean fonts found, downloads and uses NanumGothic.
    """
    # Disable font warnings
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    # Basic matplotlib settings
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Try to find Korean fonts
    system = platform.system()

    korean_fonts = []

    if system == 'Windows':
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum', 'Batang']
    elif system == 'Darwin':  # macOS
        korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic']
    else:  # Linux
        korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'UnDotum', 'Noto Sans CJK KR']

    # Get list of available fonts - more comprehensive check
    available_fonts = set()
    for font in fm.fontManager.ttflist:
        available_fonts.add(font.name)

    print(f"\n🔍 Searching for Korean fonts on {system}...")

    # Find first available Korean font
    selected_font = None
    for font in korean_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"✓ Found Korean font: {font}")
            break

    if selected_font:
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['font.sans-serif'] = [selected_font]
    else:
        print("⚠ No Korean font found in system")

        # Check if font file exists in ~/.fonts
        font_path = os.path.expanduser("~/.fonts/NanumGothic.ttf")

        if os.path.exists(font_path):
            print(f"✓ Found font file at: {font_path}")
            print("🔄 Registering font with matplotlib...")

            try:
                # Delete matplotlib font cache to force rebuild
                import shutil
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print("✓ Cleared matplotlib cache")

                # Reload font manager
                fm._load_fontmanager(try_read_cache=False)

                # Use the font file directly
                from matplotlib.font_manager import FontProperties, fontManager

                # Add font to matplotlib's font manager
                fontManager.addfont(font_path)
                font_prop = FontProperties(fname=font_path)
                font_name = font_prop.get_name()

                # Set font for all text elements
                plt.rcParams['font.family'] = font_name
                plt.rcParams['font.sans-serif'] = [font_name]

                # Also store the font path for direct use in plotting
                plt.rcParams['font.path'] = font_path

                print(f"✓ Using font: {font_name}")
                print(f"   Font path: {font_path}")

            except Exception as e:
                print(f"❌ Failed to register font: {e}")
                print("⚠ Korean text may not display correctly")
        else:
            print(f"❌ Font file not found at: {font_path}")
            print("💡 Please download the font:")
            print(f"   mkdir -p ~/.fonts")
            print(f"   cd ~/.fonts")
            print(f"   wget https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf -O NanumGothic.ttf")
            print(f"   Then run the script again.")

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
        
        # Detect number of layers based on model architecture
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            self.num_layers = len(self.model.transformer.h)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama style
            self.num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks'):
            # Gemma3 style
            self.num_layers = len(self.model.model.blocks)
        else:
            # Try to find layers in the model
            print("⚠️ Unknown model architecture, attempting to detect layers...")

            # Debug: print model structure
            print(f"Model type: {type(self.model).__name__}")
            if hasattr(self.model, 'model'):
                print(f"Model.model type: {type(self.model.model).__name__}")

                # Check all attributes for layer-like structures
                for attr_name in dir(self.model.model):
                    if not attr_name.startswith('_'):
                        try:
                            attr = getattr(self.model.model, attr_name)
                            if isinstance(attr, torch.nn.ModuleList):
                                print(f"  Found ModuleList: {attr_name} with {len(attr)} items")
                            elif hasattr(attr, '__len__') and not isinstance(attr, str):
                                try:
                                    print(f"  Found list-like: {attr_name} with length {len(attr)}")
                                except:
                                    pass
                        except:
                            pass

            # Try config first
            if hasattr(self.model, 'config'):
                # Print ALL config attributes to debug (only for Gemma)
                if 'gemma' in type(self.model).__name__.lower():
                    print(f"\nSearching ALL integer config values...")
                    for key in sorted(dir(self.model.config)):
                        if not key.startswith('_'):
                            try:
                                value = getattr(self.model.config, key)
                                if isinstance(value, int):
                                    print(f"  {key}: {value}")
                            except:
                                pass

                if hasattr(self.model.config, 'num_hidden_layers'):
                    self.num_layers = self.model.config.num_hidden_layers
                elif hasattr(self.model.config, 'n_layer'):
                    self.num_layers = self.model.config.n_layer
                elif hasattr(self.model.config, 'num_layers'):
                    self.num_layers = self.model.config.num_layers
                elif hasattr(self.model.config, 'n_layers'):
                    self.num_layers = self.model.config.n_layers
                elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'num_hidden_layers'):
                    # Gemma3 has text_config.num_hidden_layers
                    self.num_layers = self.model.config.text_config.num_hidden_layers
                    print(f"✓ Found layers in text_config: {self.num_layers}")
                else:
                    # Last resort: try to count layers manually from printed info
                    print("\nSearching for layers in model structure...")
                    found = False
                    for attr_name in dir(self.model.model):
                        if attr_name.startswith('_'):
                            continue
                        try:
                            attr = getattr(self.model.model, attr_name)
                            if isinstance(attr, torch.nn.ModuleList):
                                print(f"✓ Using ModuleList: {attr_name} with {len(attr)} layers")
                                self.num_layers = len(attr)
                                found = True
                                break
                        except:
                            pass

                    if not found:
                        raise ValueError("Could not determine number of layers from model config or structure")
            else:
                raise ValueError("Could not determine model architecture")

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

    def analyze_prompt_generation(self,
                                 prompt: str,
                                 max_new_tokens: int = 10,
                                 layer_range: Optional[Tuple[int, int]] = None,
                                 top_k: int = 5) -> Dict:
        """
        Analyze autoregressive generation: prompt → generated tokens.
        Shows how each layer predicts the NEXT tokens during generation.
        """
        print(f"Analyzing generation for prompt: '{prompt}'")
        print(f"Generating {max_new_tokens} tokens...")

        # Determine layer range
        if layer_range is None:
            start_layer, end_layer = 0, self.num_layers
        else:
            start_layer, end_layer = layer_range

        # Tokenize initial prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(self.device)

        # Store input tokens for display
        input_tokens = []
        for token_id in input_ids[0]:
            display_token = self._decode_token_safe(token_id.item())
            input_tokens.append(display_token)

        # Storage for generation analysis
        generated_tokens = []
        layer_predictions_per_step = []  # [step][layer] = predictions

        # Autoregressive generation
        current_ids = input_ids

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            # Get predictions from each layer for the LAST position (next token)
            last_pos = current_ids.shape[1] - 1
            layer_preds_this_step = []

            for layer_idx in range(start_layer, min(end_layer + 1, len(hidden_states))):
                hidden_state = hidden_states[layer_idx]
                last_hidden = hidden_state[0, last_pos, :]

                # Apply language modeling head
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(last_hidden.unsqueeze(0))
                elif hasattr(self.model, 'embed_out'):
                    logits = self.model.embed_out(last_hidden.unsqueeze(0))
                else:
                    for name, module in self.model.named_modules():
                        if 'lm_head' in name or 'embed_out' in name or 'output' in name:
                            logits = module(last_hidden.unsqueeze(0))
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

                layer_preds_this_step.append({
                    'layer': layer_idx,
                    'step': step,
                    'top_tokens': top_k_tokens,
                    'top_probs': top_k_values.detach().cpu().numpy(),
                    'top_indices': top_k_indices.detach().cpu().numpy()
                })

            layer_predictions_per_step.append(layer_preds_this_step)

            # Get the actual next token from the final layer (greedy decoding)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)

            # Decode and store
            next_token_text = self._decode_token_safe(next_token_id.item())
            generated_tokens.append(next_token_text)

            # Append to sequence for next iteration
            current_ids = torch.cat([current_ids, next_token_id], dim=1)

            # Stop if EOS token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                print(f"EOS token generated at step {step}")
                break

        print(f"Generated tokens: {' '.join(generated_tokens)}")

        return {
            'prompt': prompt,
            'input_tokens': input_tokens,
            'generated_tokens': generated_tokens,
            'layer_predictions_per_step': layer_predictions_per_step,
            'layer_range': (start_layer, min(end_layer, len(hidden_states) - 1)),
            'num_layers': end_layer - start_layer + 1
        }

    def create_generation_heatmap(self,
                                analysis_results: Dict,
                                figsize: Tuple[int, int] = (16, 10),
                                save_path: Optional[str] = None,
                                show_text: bool = True) -> None:
        """
        Create heatmap showing how each layer predicts generated tokens.

        X-axis: Generated tokens (autoregressive sequence)
        Y-axis: Model layers (0 to num_layers)
        Color: Probability of top prediction at each step

        Args:
            analysis_results: Results from analyze_prompt_generation()
            figsize: Figure size for the plot
            save_path: Path to save the figure
            show_text: Whether to show token text in cells
        """
        if 'layer_predictions_per_step' not in analysis_results:
            raise ValueError("This visualization requires results from analyze_prompt_generation()")

        layer_predictions_per_step = analysis_results['layer_predictions_per_step']
        generated_tokens = analysis_results['generated_tokens']
        input_tokens = analysis_results['input_tokens']
        layer_range = analysis_results['layer_range']
        num_layers = analysis_results['num_layers']

        num_steps = len(generated_tokens)

        if num_steps == 0:
            print("No tokens were generated!")
            return

        # Suppress warnings for this plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Prepare data: [layer][step] = (token, prob)
            predicted_tokens_matrix = []
            probability_matrix = []

            for layer_offset in range(num_layers):
                layer_tokens_across_steps = []
                layer_probs_across_steps = []

                for step_idx in range(num_steps):
                    if step_idx < len(layer_predictions_per_step):
                        layer_preds = layer_predictions_per_step[step_idx]
                        if layer_offset < len(layer_preds):
                            pred = layer_preds[layer_offset]
                            top_token = pred['top_tokens'][0] if pred['top_tokens'] else ""
                            top_prob = pred['top_probs'][0] if len(pred['top_probs']) > 0 else 0.0
                        else:
                            top_token = ""
                            top_prob = 0.0
                    else:
                        top_token = ""
                        top_prob = 0.0

                    layer_tokens_across_steps.append(top_token)
                    layer_probs_across_steps.append(top_prob)

                predicted_tokens_matrix.append(layer_tokens_across_steps)
                probability_matrix.append(layer_probs_across_steps)

            # Create the heatmap
            prob_array = np.array(probability_matrix)

            # Use Blues colormap
            im = ax.imshow(prob_array, aspect='auto', cmap='Blues', vmin=0, vmax=1)

            # Set title with prompt
            prompt_display = analysis_results['prompt']
            if len(prompt_display) > 60:
                prompt_display = prompt_display[:60] + "..."
            ax.set_title(f'Logit Lens Analysis - Autoregressive Generation\nPrompt: "{prompt_display}"',
                        fontsize=13, pad=20)

            # Y-axis: Layers (layer 0 at top = input, higher layers at bottom = output)
            ax.set_ylabel('Model Layer (↑ closer to input        closer to output ↓)', fontsize=11)

            # Show only multiples of 5, plus the last layer
            tick_positions = []
            tick_labels = []
            for i in range(num_layers):
                layer_num = i + layer_range[0]
                if layer_num % 5 == 0 or i == num_layers - 1:  # Every 5th layer or last layer
                    tick_positions.append(i)
                    tick_labels.append(str(layer_num))

            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=9)
            # Don't invert - default order is layer 0 at top

            # X-axis: Generated tokens
            ax.set_xticks(range(num_steps))
            ax.set_xticklabels(generated_tokens, rotation=45, ha='right', fontsize=9)
            ax.set_xlabel('Generated Tokens (Next Token Predictions)', fontsize=11)

            # Add text annotations in cells
            if show_text:
                for layer_idx in range(num_layers):
                    for step_idx in range(num_steps):
                        token = predicted_tokens_matrix[layer_idx][step_idx]
                        prob = probability_matrix[layer_idx][step_idx]

                        # Show predictions even with very low probability
                        # Early layers have VERY low confidence (< 0.001)
                        if token and prob > 0.0001:  # Very low threshold to show early layer predictions
                            # Choose text color based on background
                            text_color = "white" if prob > 0.5 else "black"

                            # Truncate long tokens
                            display_token = token[:8] + "..." if len(token) > 8 else token

                            # Add token text with transparency based on probability
                            # Scale alpha to make very low probs still visible
                            alpha = max(0.4, min(1.0, prob * 100))  # Scale up low probabilities
                            ax.text(step_idx, layer_idx, display_token,
                                   ha="center", va="center",
                                   color=text_color, fontsize=9, fontweight='bold', alpha=alpha)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Top Prediction Probability', fontsize=10)

            # Grid for better readability
            ax.set_xticks(np.arange(num_steps) - 0.5, minor=True)
            ax.set_yticks(np.arange(num_layers) - 0.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.3)
            ax.tick_params(which="minor", size=0)

            # Add input prompt info
            input_text = " ".join(input_tokens)
            if len(input_text) > 80:
                input_text = input_text[:80] + "..."
            fig.text(0.5, 0.02, f"Input: {input_text}",
                    ha='center', fontsize=9, style='italic', color='gray')

            plt.tight_layout(rect=[0, 0.03, 1, 1])

            if save_path:
                # Save with font embedding to preserve Korean characters
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                           pil_kwargs={'quality': 95})
                print(f"Generation heatmap saved to: {save_path}")

            plt.show()


def main():
    """
    Main function to run logit lens analysis.
    """
    # Configuration
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.2-3B-Instruct"
    # model_path = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it"

    # You can change the prompt here to anything you want
    # 여기서 프롬프트를 원하는 대로 바꿀 수 있습니다
    # prompt = "La bateau naviguait en doceur sur"
    # prompt = "The capital of France is"  # English example
    prompt = "배는 바다위를 순조롭게 항해"  # Another Korean example

    # Number of tokens to generate
    max_new_tokens = 10

    print(f"Using model: {model_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Generating {max_new_tokens} tokens...\n")

    # Initialize LogitLens
    lens = LogitLens(model_path)

    # Analyze generation (autoregressive)
    results = lens.analyze_prompt_generation(
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )

    # Create visualization
    print("\nCreating visualization...")
    lens.create_generation_heatmap(
        results,
        save_path="logit_lens_generation.png",
        show_text=True  # Set to False if you don't want text in cells
    )

    print("\n✅ Complete! Check logit_lens_generation.png")
    print(f"Full generated sequence: {prompt} {' '.join(results['generated_tokens'])}")


if __name__ == "__main__":
    main()