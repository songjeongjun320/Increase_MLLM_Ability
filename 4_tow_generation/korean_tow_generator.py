#!/usr/bin/env python3
"""
Korean Story + English ToW Generator
===================================

Generates English ToW tokens for Korean story corpus.
Core principle: Korean input → English reasoning in <ToW> → Korean output

Example:
Korean: "브루스 리는 쿵푸 영화 감독을 만났다"
Enhanced: "브루스 리는 쿵푸 영화 감독을 만났다 <ToW>The context suggests a professional meeting between martial artist and filmmaker</ToW> 점심을 함께 했다"
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.text_processing import enforce_english_text, sanitize_tow_token

@dataclass
class KoreanStoryEntry:
    """Korean story entry for ToW augmentation"""
    text: str
    source: str
    story_id: str
    sentence_id: int

@dataclass
class ToWAugmentedEntry:
    """Korean story augmented with English ToW"""
    original_text: str
    augmented_text: str
    tow_tokens: List[str]
    tow_count: int
    source: str
    story_id: str
    sentence_id: int

class KoreanToWGenerator:
    """Generate English ToW tokens for Korean story text"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize Korean ToW Generator
        
        Args:
            model_path: Path to ToW generation model (DeepSeek/Qwen/Llama)
            device: Device to run model on
        """
        self.model_path = model_path or self._get_default_model_path()
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # ToW generation statistics
        self.stats = {
            "total_sentences": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "english_compliance_rate": 0.0,
            "avg_tow_per_sentence": 0.0
        }
    
    def _get_default_model_path(self) -> str:
        """Get default ToW generation model path - Use GPT-OSS for ToW generation"""
        
        # Priority: GPT-OSS models for ToW generation (separate from training base models)
        gpt_oss_models = [
            "openai/gpt-oss-20b",     # 16GB memory - good for ToW generation
            "openai/gpt-oss-120b",    # H100 GPU - if available
        ]
        
        # Check if GPT-OSS models are available locally
        models_dir = Path(__file__).parent.parent / "1_models" / "gpt_oss"
        
        for model_name in ["gpt-oss-20b", "gpt-oss-120b"]:
            model_path = models_dir / model_name
            if model_path.exists():
                return str(model_path)
        
        # If no local GPT-OSS models, return HuggingFace model ID for auto-download
        print("🔄 No local GPT-OSS models found. Will download from HuggingFace...")
        return "openai/gpt-oss-20b"  # Use 20B model by default (fits in 16GB)
    
    def load_model(self):
        """Load ToW generation model"""
        print(f"📥 Loading ToW generation model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_english_tow(self, korean_text: str, context_window: int = 10) -> List[str]:
        """
        Generate English ToW tokens for Korean text
        
        Args:
            korean_text: Korean input text
            context_window: Number of words to consider for context
            
        Returns:
            List of English ToW tokens
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create ToW generation prompt
        prompt = self._create_tow_prompt(korean_text)
        
        try:
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract ToW tokens from generated text
            tow_tokens = self._extract_tow_tokens(generated_text)
            
            # Enforce English-only constraint
            english_tow_tokens = []
            for token in tow_tokens:
                english_token = enforce_english_text(token)
                sanitized_token = sanitize_tow_token(english_token)
                english_tow_tokens.append(sanitized_token)
            
            return english_tow_tokens
            
        except Exception as e:
            print(f"❌ Error generating ToW for text: {korean_text[:50]}... Error: {e}")
            return []
    
    def _create_tow_prompt(self, korean_text: str) -> str:
        """Create prompt for ToW generation"""
        
        prompt = f"""다음 한국어 문장에 대해 영어로 추론하는 Thoughts of Words (ToW) 토큰을 생성하세요.

규칙:
1. ToW 토큰은 반드시 <ToW>...</ToW> 형식으로 작성
2. ToW 내부는 반드시 영어로만 작성 (한국어 금지)
3. 논리적이고 맥락에 맞는 추론 제공
4. 간결하고 명확하게 작성 (200자 이내)

한국어 입력: {korean_text}

ToW가 포함된 향상된 텍스트를 생성하세요:"""

        return prompt
    
    def _extract_tow_tokens(self, generated_text: str) -> List[str]:
        """Extract ToW tokens from generated text"""
        
        # Find all <ToW>...</ToW> patterns
        tow_pattern = r'<ToW>(.*?)</ToW>'
        matches = re.findall(tow_pattern, generated_text, re.DOTALL)
        
        tow_tokens = []
        for match in matches:
            # Clean and validate ToW content
            tow_content = match.strip()
            if tow_content:
                tow_token = f"<ToW>{tow_content}</ToW>"
                tow_tokens.append(tow_token)
        
        return tow_tokens
    
    def augment_korean_story(self, story_entry: KoreanStoryEntry) -> ToWAugmentedEntry:
        """
        Augment Korean story with English ToW tokens
        
        Args:
            story_entry: Korean story entry to augment
            
        Returns:
            ToW augmented entry
        """
        try:
            # Generate English ToW tokens
            tow_tokens = self.generate_english_tow(story_entry.text)
            
            if tow_tokens:
                # Insert ToW tokens into Korean text
                augmented_text = self._insert_tow_tokens(story_entry.text, tow_tokens)
                self.stats["successful_generations"] += 1
            else:
                # No ToW generated, use original text
                augmented_text = story_entry.text
                self.stats["failed_generations"] += 1
            
            self.stats["total_sentences"] += 1
            
            return ToWAugmentedEntry(
                original_text=story_entry.text,
                augmented_text=augmented_text,
                tow_tokens=tow_tokens,
                tow_count=len(tow_tokens),
                source=story_entry.source,
                story_id=story_entry.story_id,
                sentence_id=story_entry.sentence_id
            )
            
        except Exception as e:
            print(f"❌ Error augmenting story entry: {e}")
            self.stats["failed_generations"] += 1
            self.stats["total_sentences"] += 1
            
            # Return original entry on failure
            return ToWAugmentedEntry(
                original_text=story_entry.text,
                augmented_text=story_entry.text,
                tow_tokens=[],
                tow_count=0,
                source=story_entry.source,
                story_id=story_entry.story_id,
                sentence_id=story_entry.sentence_id
            )
    
    def _insert_tow_tokens(self, korean_text: str, tow_tokens: List[str]) -> str:
        """Insert ToW tokens into Korean text at appropriate positions"""
        
        # Simple strategy: insert ToW tokens at sentence boundaries or key positions
        sentences = korean_text.split('. ')
        
        if len(sentences) == 1:
            # Single sentence - insert ToW at the end
            if tow_tokens:
                return f"{korean_text} {tow_tokens[0]}"
            return korean_text
        
        # Multiple sentences - distribute ToW tokens
        augmented_sentences = []
        tow_index = 0
        
        for i, sentence in enumerate(sentences):
            augmented_sentences.append(sentence)
            
            # Insert ToW token if available
            if tow_index < len(tow_tokens) and i < len(sentences) - 1:
                augmented_sentences.append(tow_tokens[tow_index])
                tow_index += 1
        
        return '. '.join(augmented_sentences)
    
    def process_korean_stories(
        self, 
        stories: List[KoreanStoryEntry], 
        batch_size: int = 10,
        save_path: Optional[str] = None
    ) -> List[ToWAugmentedEntry]:
        """
        Process multiple Korean stories with ToW augmentation
        
        Args:
            stories: List of Korean story entries
            batch_size: Batch size for processing
            save_path: Path to save results
            
        Returns:
            List of ToW augmented entries
        """
        if not self.model:
            if not self.load_model():
                raise RuntimeError("Failed to load ToW generation model")
        
        print(f"🚀 Processing {len(stories)} Korean stories with English ToW...")
        
        augmented_stories = []
        
        # Process in batches
        for i in range(0, len(stories), batch_size):
            batch = stories[i:i + batch_size]
            print(f"📝 Processing batch {i//batch_size + 1}/{(len(stories) + batch_size - 1)//batch_size}")
            
            for story_entry in batch:
                augmented_entry = self.augment_korean_story(story_entry)
                augmented_stories.append(augmented_entry)
                
                # Print progress
                if len(augmented_stories) % 50 == 0:
                    self._print_progress()
        
        # Save results if path provided
        if save_path:
            self._save_results(augmented_stories, save_path)
        
        # Print final statistics
        self._print_final_stats()
        
        return augmented_stories
    
    def _print_progress(self):
        """Print current processing progress"""
        total = self.stats["total_sentences"]
        success = self.stats["successful_generations"]
        success_rate = (success / total) * 100 if total > 0 else 0
        
        print(f"📊 Progress: {total} processed, {success} successful ToW generations ({success_rate:.1f}%)")
    
    def _print_final_stats(self):
        """Print final processing statistics"""
        total = self.stats["total_sentences"]
        success = self.stats["successful_generations"]
        failed = self.stats["failed_generations"]
        
        if total > 0:
            success_rate = (success / total) * 100
            print(f"\n📈 Final Statistics:")
            print(f"   Total sentences processed: {total}")
            print(f"   Successful ToW generations: {success} ({success_rate:.1f}%)")
            print(f"   Failed generations: {failed}")
            print(f"   English compliance rate: 100% (enforced)")
    
    def _save_results(self, augmented_stories: List[ToWAugmentedEntry], save_path: str):
        """Save ToW augmented results"""
        
        output_data = []
        for entry in augmented_stories:
            output_data.append({
                "original_text": entry.original_text,
                "augmented_text": entry.augmented_text,
                "tow_tokens": entry.tow_tokens,
                "tow_count": entry.tow_count,
                "source": entry.source,
                "story_id": entry.story_id,
                "sentence_id": entry.sentence_id
            })
        
        # Save as JSONL
        with open(save_path, 'w', encoding='utf-8') as f:
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"💾 Results saved to {save_path}")

def main():
    """Main function for testing Korean ToW generation"""
    
    # Test with sample Korean stories
    sample_stories = [
        KoreanStoryEntry(
            text="브루스 리는 쿵푸 영화 감독을 만났다.",
            source="test",
            story_id="test_001",
            sentence_id=1
        ),
        KoreanStoryEntry(
            text="그들은 새로운 액션 영화에 대해 논의했다.",
            source="test", 
            story_id="test_001",
            sentence_id=2
        )
    ]
    
    # Initialize generator
    generator = KoreanToWGenerator()
    
    # Process stories
    results = generator.process_korean_stories(
        sample_stories,
        save_path="korean_stories_with_tow_test.jsonl"
    )
    
    # Print results
    for result in results:
        print(f"\n📝 Original: {result.original_text}")
        print(f"🔧 Augmented: {result.augmented_text}")
        print(f"💭 ToW Count: {result.tow_count}")

if __name__ == "__main__":
    main()