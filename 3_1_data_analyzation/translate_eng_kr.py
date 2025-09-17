#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English-Korean Translation Interactive Chat
Interactive translation system using trained models with improved performance

module load cuda-12.6.1-gcc-12.1.0
"""

import os
import torch
import warnings
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import sys
from dataclasses import dataclass, field
import re

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
        self.few_shot_examples = self._get_few_shot_examples()

    def _get_few_shot_examples(self):
        """Few-shot examples for better translation quality"""
        return [
            {
                "english": "Hello, how are you today?",
                "korean": "안녕하세요, 오늘 어떻게 지내세요?"
            },
            {
                "english": "I would like to make a reservation for tomorrow.",
                "korean": "내일 예약하고 싶습니다."
            },
            {
                "english": "The weather is beautiful today.",
                "korean": "오늘 날씨가 정말 좋네요."
            },
            {
                "english": "Can you help me with this problem?",
                "korean": "이 문제를 도와주실 수 있나요?"
            },
            {
                "english": "Thank you for your assistance.",
                "korean": "도움 주셔서 감사합니다."
            }
        ]

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

    def create_translation_prompt(self, text: str, use_few_shot: bool = True):
        """Create an improved translation prompt with few-shot examples"""
        
        # System instruction
        system_prompt = """You are a professional English to Korean translator. Your translations should be:
1. Accurate and natural in Korean
2. Appropriate for the context
3. Using proper Korean grammar and honorifics when necessary

Task: Translate the given English text to Korean."""

        if use_few_shot and self.few_shot_examples:
            # Build few-shot prompt
            examples_text = "\n\nHere are some translation examples:\n"
            for i, example in enumerate(self.few_shot_examples[:3], 1):  # Use top 3 examples
                examples_text += f"\nExample {i}:\nEnglish: {example['english']}\nKorean: {example['korean']}\n"
            
            prompt = f"{system_prompt}{examples_text}\n\nNow translate this:\nEnglish: {text}\nKorean:"
        else:
            # Simple prompt without few-shot
            prompt = f"{system_prompt}\n\nEnglish: {text}\nKorean:"
        
        return prompt

    def create_chat_style_prompt(self, text: str):
        """Alternative chat-style prompt format for models trained with chat templates"""
        messages = [
            {
                "role": "system",
                "content": "You are a professional English to Korean translator. Provide accurate, natural Korean translations."
            }
        ]
        
        # Add few-shot examples as conversation history
        for example in self.few_shot_examples[:2]:
            messages.append({
                "role": "user", 
                "content": f"Translate to Korean: {example['english']}"
            })
            messages.append({
                "role": "assistant",
                "content": example['korean']
            })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": f"Translate to Korean: {text}"
        })
        
        # Convert to text format (adjust based on your model's chat template)
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n\n"
        
        prompt += "Assistant:"
        return prompt

    def post_process_translation(self, text: str) -> str:
        """Post-process the generated translation to clean it up"""
        # Remove any repeated sentences
        sentences = text.split('.')
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        cleaned_text = '. '.join(unique_sentences)
        if cleaned_text and not cleaned_text.endswith('.'):
            cleaned_text += '.'
        
        # Remove any English text that might have leaked through
        # (assuming Korean translation shouldn't contain long sequences of Latin characters)
        cleaned_text = re.sub(r'\b[A-Za-z]{20,}\b', '', cleaned_text)
        
        # Remove excessive whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text.strip()

    def generate_response(self, prompt: str, max_new_tokens: int = 256):
        """Generate response from the model with improved parameters"""
        try:
            print(f"DEBUG: Prompt length: {len(prompt)} characters")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            print(f"DEBUG: Input token length: {inputs['input_ids'].shape[1]}")

            # Improved generation parameters for translation
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # Reduced for typical translation lengths
                    min_new_tokens=5,  # Ensure minimum output
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,  # Balanced temperature
                    top_p=0.9,  # Slightly more focused
                    top_k=50,  # Add top_k for better quality
                    repetition_penalty=1.15,  # Moderate penalty
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,  # Neutral length preference
                    early_stopping=True,  # Stop at first EOS token
                    num_beams=4,  # Use beam search for better quality
                )

            # Decode only the generated part
            input_length = inputs['input_ids'].shape[1]
            output_only_tokens = outputs[:, input_length:]
            generated_text = self.tokenizer.decode(output_only_tokens[0], skip_special_tokens=True).strip()

            # Post-process the translation
            generated_text = self.post_process_translation(generated_text)

            print(f"DEBUG: Final translation: '{generated_text}'")
            print("="*60)

            return generated_text

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def translate(self, text: str, prompt_style: str = "few_shot"):
        """Translate text to Korean with different prompt styles"""
        # Choose prompt style
        if prompt_style == "chat":
            prompt = self.create_chat_style_prompt(text)
        else:
            prompt = self.create_translation_prompt(text, use_few_shot=True)
        
        # Determine appropriate max tokens based on input length
        max_new_tokens = min(256, len(text.split()) * 3)  # Rough estimate
        
        translation = self.generate_response(prompt, max_new_tokens=max_new_tokens)

        # Store in conversation history
        self.conversation_history.append({
            "source_text": text,
            "translation": translation,
            "prompt_style": prompt_style
        })

        return translation

    def batch_translate(self, texts: list) -> list:
        """Translate multiple texts in batch for efficiency"""
        translations = []
        for text in texts:
            translation = self.translate(text)
            translations.append(translation)
        return translations

    def evaluate_translation(self, source: str, translation: str) -> dict:
        """Simple evaluation metrics for translation quality"""
        metrics = {
            "source_length": len(source.split()),
            "translation_length": len(translation.split()),
            "length_ratio": len(translation.split()) / max(len(source.split()), 1),
            "contains_korean": any('\uac00' <= char <= '\ud7af' for char in translation),
            "contains_english": bool(re.search(r'[a-zA-Z]{3,}', translation))
        }
        return metrics

    def chat_mode(self):
        """Interactive chat mode for translation"""
        print("\n" + "="*60)
        print("🌐 Enhanced English-Korean Translation Chat Bot")
        print("="*60)
        print("Commands:")
        print("  'quit' or 'exit' - Exit the chat")
        print("  'history' - Show conversation history")
        print("  'clear' - Clear conversation history")
        print("  'style [few_shot/chat/simple]' - Change prompt style")
        print("  'batch' - Enter batch translation mode")
        print("  'help' - Show this help message")
        print("="*60)

        current_style = "few_shot"

        while True:
            try:
                user_input = input(f"\n💬 Enter text to translate (style: {current_style}): ").strip()

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
                    print("  'style [few_shot/chat/simple]' - Change prompt style")
                    print("  'batch' - Enter batch translation mode")
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("✅ Conversation history cleared.")
                    continue
                elif user_input.lower().startswith('style'):
                    parts = user_input.split()
                    if len(parts) > 1 and parts[1] in ['few_shot', 'chat', 'simple']:
                        current_style = parts[1]
                        print(f"✅ Switched to {current_style} prompt style")
                    else:
                        print("❌ Invalid style. Choose: few_shot, chat, or simple")
                    continue
                elif user_input.lower() == 'batch':
                    self.batch_mode()
                    continue

                # Perform translation
                print("🤖 Translating...")
                translation = self.translate(user_input, prompt_style=current_style)
                print(f"✨ Translation: {translation}")
                
                # Show evaluation metrics
                metrics = self.evaluate_translation(user_input, translation)
                print(f"📊 Metrics: Length ratio: {metrics['length_ratio']:.2f}, Contains Korean: {metrics['contains_korean']}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def batch_mode(self):
        """Batch translation mode"""
        print("\n📦 Batch Translation Mode")
        print("Enter texts line by line. Enter 'done' when finished:")
        
        texts = []
        while True:
            line = input(f"Text {len(texts) + 1}: ").strip()
            if line.lower() == 'done':
                break
            if line:
                texts.append(line)
        
        if texts:
            print(f"\n🤖 Translating {len(texts)} texts...")
            translations = self.batch_translate(texts)
            
            print("\n📝 Results:")
            for i, (source, trans) in enumerate(zip(texts, translations), 1):
                print(f"{i}. Source: {source}")
                print(f"   Translation: {trans}\n")

    def show_history(self):
        """Show conversation history with metrics"""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return

        print("\n📚 Conversation History:")
        print("-" * 50)
        for i, item in enumerate(self.conversation_history[-10:], 1):  # Show last 10
            print(f"{i}. Source: {item['source_text']}")
            print(f"   Translation: {item['translation']}")
            print(f"   Style: {item.get('prompt_style', 'unknown')}")
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
    print("🌟 Welcome to Enhanced English-Korean Translation Chat Bot!")

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