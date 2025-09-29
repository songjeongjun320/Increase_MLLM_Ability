"""
Logit Lens Analysis for Transformer Language Models - Fixed Version

This module implements logit lens analysis to visualize how transformer models
progressively refine their next-token predictions across different layers.

Key improvements:
- Better Korean font handling for Linux systems
- Improved visualization matching the desired format
- Fixed font warnings and rendering issues

Author: Based on research from nostalgebraist (2020) and recent advances
Date: 2025-09-28
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
from typing import List, Optional, Dict, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import platform
import subprocess


def setup_korean_font():
    """
    Enhanced Korean font setup for matplotlib with better Linux support.
    """
    system = platform.system()
    
    # First, try to install fonts if on Linux
    if system == "Linux":
        try:
            # Try to install Korean fonts using system package manager
            import subprocess
            
            # Check if we have sudo privileges (might not in cluster environment)
            try:
                # Try to install fonts-noto-cjk if not present
                subprocess.run(['fc-list', ':lang=ko'], 
                             capture_output=True, text=True, check=False)
            except:
                pass
                
            # Get list of available fonts
            result = subprocess.run(['fc-list', ':lang=ko', 'family'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                korean_system_fonts = result.stdout.strip().split('\n')
                korean_system_fonts = [f.split(',')[0].strip() for f in korean_system_fonts if f]
                print(f"Found Korean fonts in system: {korean_system_fonts[:5]}")
            else:
                korean_system_fonts = []
        except Exception as e:
            print(f"Could not check system fonts: {e}")
            korean_system_fonts = []
    else:
        korean_system_fonts = []
    
    # Comprehensive list of Korean fonts to try
    korean_fonts = []
    
    if system == "Linux":
        # Prioritize system fonts found via fc-list
        korean_fonts.extend(korean_system_fonts)
        
        # Add common Korean fonts on Linux
        korean_fonts.extend([
            'Noto Sans CJK KR',
            'Noto Sans KR', 
            'NanumGothic',
            'NanumBarunGothic',
            'NanumMyeongjo',
            'UnBatang',
            'UnDotum',
            'Baekmuk Gulim',
            'Baekmuk Dotum',
            'Malgun Gothic',
            'D2Coding',
            'Source Han Sans KR'
        ])
    elif system == "Darwin":  # macOS
        korean_fonts = [
            'AppleGothic',
            'Apple SD Gothic Neo',
            'Noto Sans CJK KR'
        ]
    elif system == "Windows":
        korean_fonts = [
            'Malgun Gothic',
            'Gulim',
            'Dotum',
            'Batang'
        ]
    
    # Get matplotlib's font list
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # Try to find and set a Korean font
    for font_name in korean_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ Korean font set to: {font_name}")
            return font_name
    
    # Fallback: Use matplotlib's built-in font with Unicode support
    print("⚠️ No Korean fonts found. Using matplotlib fallback with Unicode support.")
    
    # Set up fallback fonts that might have Korean support
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Try to use fontconfig if available (Linux)
    if system == "Linux":
        try:
            # Force matplotlib to use system fonts
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK KR', 'NanumGothic', 'DejaVu Sans']
            
            # Clear font cache
            cache_dir = fm.get_cachedir()
            if os.path.exists(cache_dir):
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    print("Cleared matplotlib font cache")
                except:
                    pass
            
            # Rebuild font list
            fm._rebuild()
            
        except Exception as e:
            print(f"Could not configure fontconfig: {e}")
    
    return "Fallback"


def use_ascii_fallback():
    """
    Fallback function to convert Korean text to romanized form or placeholders.
    """
    def romanize_korean(text):
        """Simple Korean to English mapping for common words."""
        korean_map = {
            '광합성': 'photosyn',
            '과정': 'process',
            '의': 'of',
            '최종': 'final',
            '결과': 'result',
            '는': 'is',
            '당': 'sugar',
            '과': 'and',
            '물': 'water',
            '에서': 'from',
            '산': 'oxygen',
            '소': 'oxygen',
            '분': 'min',
            '해': 'do',
            '요': 'need',
            '주': 'main',
            '탄': 'carbon'
        }
        
        # Try to map Korean text
        for kor, eng in korean_map.items():
            text = text.replace(kor, eng)
        
        # If still has Korean characters, use token index
        if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text):
            return f"[{text[:3]}...]"  # Show partial text
            
        return text
    
    return romanize_korean


class LogitLens:
    """
    Logit Lens analyzer for transformer language models.
    """

    def __init__(self, model_path: str, device: str = "auto", use_ascii_fallback: bool = False):
        """
        Initialize the LogitLens analyzer.
        
        Args:
            model_path: Path to the model
            device: Device to run the model on
            use_ascii_fallback: If True, use ASCII characters instead of Korean
        """
        # Setup Korean font
        self.font_status = setup_korean_font()
        self.use_ascii = use_ascii_fallback or (self.font_status == "Fallback")
        
        if self.use_ascii:
            self.text_converter = use_ascii_fallback()
        else:
            self.text_converter = lambda x: x
        
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
        """Safely decode a token ID to a readable string."""
        try:
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            if decoded and decoded.strip():
                return self.text_converter(decoded.strip())
            
            token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            
            if token_str.startswith('▁'):
                return self.text_converter(token_str[1:])
            elif token_str.startswith('##'):
                return self.text_converter(token_str[2:])
            elif token_str.startswith('<') and token_str.endswith('>'):
                return token_str
            else:
                return self.text_converter(token_str)
                
        except Exception:
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
                                    save_path: Optional[str] = None) -> None:
        """
        Create heatmap with improved layout matching the desired format.
        """
        if 'position_predictions' not in analysis_results:
            raise ValueError("This visualization requires results from analyze_prompt_all_positions()")
        
        position_predictions = analysis_results['position_predictions']
        input_tokens = analysis_results['input_tokens']
        layer_range = analysis_results['layer_range']
        
        num_layers = len(position_predictions[0])
        num_positions = len(position_predictions)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        predicted_tokens_matrix = []
        probability_matrix = []
        
        # Reverse layer order for visualization (early layers at top)
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
        
        # Create colormap matching your example
        prob_array = np.array(probability_matrix)
        
        # Use a diverging colormap similar to your example
        cmap = plt.cm.RdBu_r  # Red-Blue reversed (blue for high prob, red for low)
        im = ax.imshow(prob_array, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        
        # Configure axes
        ax.set_ylabel('Model Layer\n← closer to input', fontsize=12)
        ax.set_title(f'Logit Lens Analysis\nPrompt: "{self.text_converter(analysis_results["prompt"])}"', 
                    fontsize=14, pad=40)
        
        # Set y-axis (layers)
        layer_labels = [f"{i+layer_range[0]}" for i in range(num_layers)]
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels(layer_labels, fontsize=10)
        
        # Set x-axis for positions
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels([], rotation=0)  # Hide bottom labels initially
        
        # Create output tokens labels (bottom)
        output_labels = []
        for pos_idx in range(num_positions):
            # Get the prediction from the last layer for this position
            last_layer_pred = position_predictions[pos_idx][-1]
            if last_layer_pred['top_tokens']:
                output_labels.append(last_layer_pred['top_tokens'][0])
            else:
                output_labels.append("")
        
        ax.set_xticklabels(output_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Output Tokens', fontsize=12, labelpad=10)
        
        # Add input tokens on top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(num_positions))
        ax2.set_xticklabels(input_tokens, rotation=45, ha='left', fontsize=10)
        ax2.set_xlabel('Input Tokens', fontsize=12, labelpad=10)
        
        # Add text annotations
        for i in range(num_layers):
            for j in range(num_positions):
                token = predicted_tokens_matrix[i][j]
                prob = probability_matrix[i][j]
                
                if token and prob > 0.01:
                    # Determine text color
                    text_color = "white" if prob > 0.5 else "black"
                    
                    # Add token text
                    ax.text(j, i, token,
                           ha="center", va="center",
                           color=text_color, fontsize=8, fontweight='normal')
                    
                    # Add probability below token
                    if prob > 0.1:  # Only show prob for significant predictions
                        ax.text(j, i + 0.3, f'{prob:.2f}',
                               ha="center", va="center",
                               color=text_color, fontsize=6, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability', fontsize=10)
        
        # Add arrows to indicate flow
        ax.annotate('', xy=(0, num_layers), xytext=(0, -0.5),
                   arrowprops=dict(arrowstyle='→', lw=2, color='gray'),
                   annotation_clip=False)
        
        ax.text(-0.5, num_layers/2, '← closer to input', rotation=90, 
               va='center', ha='center', fontsize=10, color='gray')
        ax.text(-0.5, num_layers/2 - 5, '← closer to output →', rotation=90, 
               va='center', ha='center', fontsize=10, color='gray')
        
        # Improve layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Token position heatmap saved to: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Model path
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt"
    
    print(f"Using model: {model_path}")
    
    # Initialize LogitLens with ASCII fallback if Korean fonts fail
    lens = LogitLens(model_path, use_ascii_fallback=False)
    
    # If Korean fonts are not working, try with ASCII fallback
    if lens.font_status == "Fallback":
        print("\n⚠️ Korean fonts not available. Using ASCII fallback mode...")
        lens = LogitLens(model_path, use_ascii_fallback=True)
    
    # Analyze prompt
    prompt = "광합성 과정의 최종 결과는 당과"
    
    print("Analyzing prompt...")
    results = lens.analyze_prompt_all_positions(prompt)
    
    # Create visualization
    print("Creating visualization...")
    lens.create_token_position_heatmap(results, save_path="logit_lens_result.png")
    
    print("완료! logit_lens_result.png 파일을 확인하세요.")