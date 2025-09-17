#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English-Korean Translation Interactive Chat
Interactive translation system using trained models
"""

import os
import torch
import warnings
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import sys
from dataclasses import dataclass, field

# Suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    use_quantization: bool = True
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

# Available models - user can select which one to use
MODEL_CONFIGS = [
    ModelConfig(
        name="llama-3.2-3b-pt",
        model_path="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/llama-3.2-3b-pt",
        use_quantization=False
    ),
]

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"

class TranslationChatBot:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self.conversation_history = []

    def load_model(self):
        """Load model and tokenizer based on configuration"""
        print(f"Loading model: {self.config.name}")
        print(f"Model path: {self.config.model_path}")

        try:
            # Load tokenizer
            print(f"Loading tokenizer from: {os.path.abspath(self.config.model_path)}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                cache_dir=CACHE_DIR,
                padding_side='left'
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            print(f"Loading model {self.config.model_path}...")
            quantization_config_bnb = None
            if self.config.use_quantization:
                quantization_config_bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.config.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=self.config.torch_dtype,
                quantization_config=quantization_config_bnb,
                device_map=DEVICE,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )

            # Resize token embeddings if needed
            if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
                print(f"Resizing model token embeddings from {self.model.get_input_embeddings().weight.shape[0]} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))

            self.model.eval()
            print("Model and tokenizer loaded successfully.")

            # Disable compilation for Gemma models
            if "gemma" in self.config.name.lower():
                torch._dynamo.config.disable = True
                print("Disabled torch compilation for Gemma model")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def create_translation_prompt(self, text: str):
        """Create a translation prompt"""
        prompt = f"You are an expert korean translator. Translate the following English text to Korean: \"{text}\"\n\nKorean translation:"
        return prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 512):
        """Generate response from the model"""
        try:
            print(f"DEBUG: Input prompt: {prompt}")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Increased from 512
            ).to(DEVICE)

            print(f"DEBUG: Input token length: {inputs['input_ids'].shape[1]}")

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 100),  # Limit tokens to prevent runaway
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.3,  # Lower temperature
                    top_p=0.95,
                    repetition_penalty=1.2,  # Higher penalty to stop repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                )

            print(f"DEBUG: Output token length: {outputs.shape[1]}")

            # Decode full output first to see what model generated
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("================================================")
            print(f"\n\nDEBUG: Full output: {full_output}\n\n")

            # Decode only the generated part
            input_length = inputs['input_ids'].shape[1]
            output_only_tokens = outputs[:, input_length:]
            generated_text = self.tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()

            print("================================================")
            print(f"\n\nDEBUG: Generated text only:\n\n '{generated_text}'\n\n")
            print("================================================")

            return generated_text

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def translate(self, text: str):
        """Translate text to Korean"""
        prompt = self.create_translation_prompt(text)
        translation = self.generate_response(prompt)

        # Store in conversation history
        self.conversation_history.append({
            "source_text": text,
            "translation": translation
        })

        return translation

    def chat_mode(self):
        """Interactive chat mode for translation"""
        print("\n" + "="*60)
        print("🌐 English-Korean Translation Chat Bot")
        print("="*60)
        print("Commands:")
        print("  'quit' or 'exit' - Exit the chat")
        print("  'history' - Show conversation history")
        print("  'clear' - Clear conversation history")
        print("  'help' - Show this help message")
        print("="*60)

        while True:
            try:
                user_input = input("\n💬 Enter text to translate: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'quit' or 'exit' - Exit the chat")
                    print("  'history' - Show conversation history")
                    print("  'clear' - Clear conversation history")
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("✅ Conversation history cleared.")
                    continue

                # Perform translation
                print("🤖 Translating...")
                translation = self.translate(user_input)
                print(f"✨ Translation: {translation}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return

        print("\n📚 Conversation History:")
        print("-" * 50)
        for i, item in enumerate(self.conversation_history[-10:], 1):  # Show last 10
            print(f"{i}. Source: {item['source_text']}")
            print(f"   Translation: {item['translation']}")
            print()

    def cleanup(self):
        """Clean up model and free memory"""
        del self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def select_model():
    """Let user select which model to use"""
    print("\n🤖 Available Models:")
    print("-" * 50)
    for i, config in enumerate(MODEL_CONFIGS, 1):
        print(f"{i}. {config.name}")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(MODEL_CONFIGS)}): ").strip()
            choice = int(choice)
            if 1 <= choice <= len(MODEL_CONFIGS):
                return MODEL_CONFIGS[choice - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(MODEL_CONFIGS)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def main():
    """Main function"""
    print("🌟 Welcome to English-Korean Translation Chat Bot!")

    # Select model
    selected_config = select_model()

    # Initialize chatbot
    chatbot = TranslationChatBot(selected_config)

    try:
        # Load model
        chatbot.load_model()

        # Start chat mode
        chatbot.chat_mode()

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if chatbot.model is not None:
            chatbot.cleanup()

if __name__ == "__main__":
    main()