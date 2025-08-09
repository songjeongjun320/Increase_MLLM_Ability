#!/usr/bin/env python3
"""
Convert koconovel.json to ToW Dataset
===================================

Converts Korean novel corpus to ToW-enhanced training data.
Process: Korean stories → English reasoning in <ToW> tokens → Enhanced Korean output

Usage:
    python convert_koconovel_to_tow.py --input 2_datasets/koconovel/koconovel.json --output 2_datasets/koconovel_tow.jsonl
"""

import sys
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from 4_tow_generation.korean_tow_generator import KoreanToWGenerator, KoreanStoryEntry

def load_koconovel_dataset(file_path: str) -> List[KoreanStoryEntry]:
    """Load koconovel.json and convert to KoreanStoryEntry format"""
    
    print(f"📖 Loading koconovel dataset from {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stories = []
        for item in data:
            doc_id = item.get('doc_id', 'unknown')
            text = item.get('text', '').strip()
            
            if text:  # Skip empty texts
                # Parse doc_id to extract story_id and sentence_id
                if '_' in doc_id:
                    story_id, sentence_id = doc_id.split('_', 1)
                else:
                    story_id = doc_id
                    sentence_id = '0'
                
                story_entry = KoreanStoryEntry(
                    text=text,
                    source='koconovel',
                    story_id=story_id,
                    sentence_id=int(sentence_id) if sentence_id.isdigit() else 0
                )
                stories.append(story_entry)
        
        print(f"✅ Loaded {len(stories)} story entries from koconovel dataset")
        return stories
        
    except Exception as e:
        print(f"❌ Error loading koconovel dataset: {e}")
        return []

def main():
    """Main conversion function"""
    
    parser = argparse.ArgumentParser(description="Convert koconovel.json to ToW dataset")
    parser.add_argument(
        '--input', 
        default='2_datasets/koconovel/koconovel.json',
        help='Path to input koconovel.json file'
    )
    parser.add_argument(
        '--output', 
        default='2_datasets/koconovel_tow.jsonl',
        help='Path to output ToW dataset file'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=10,
        help='Batch size for processing (default: 10)'
    )
    parser.add_argument(
        '--max-entries', 
        type=int, 
        default=None,
        help='Maximum number of entries to process (for testing)'
    )
    parser.add_argument(
        '--model-path', 
        default=None,
        help='Path to ToW generation model (auto-detect if not provided)'
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Load koconovel dataset
    stories = load_koconovel_dataset(args.input)
    if not stories:
        print("❌ No stories loaded. Exiting.")
        return
    
    # Limit entries if specified
    if args.max_entries:
        stories = stories[:args.max_entries]
        print(f"🔢 Limited to {len(stories)} entries for processing")
    
    # Initialize ToW generator
    print(f"🚀 Initializing Korean ToW Generator...")
    generator = KoreanToWGenerator(model_path=args.model_path)
    
    # Process stories with ToW augmentation
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        augmented_stories = generator.process_korean_stories(
            stories=stories,
            batch_size=args.batch_size,
            save_path=str(output_path)
        )
        
        print(f"\n🎉 Conversion completed!")
        print(f"📊 Statistics:")
        print(f"   Input entries: {len(stories)}")
        print(f"   Output entries: {len(augmented_stories)}")
        print(f"   Success rate: {(generator.stats['successful_generations'] / generator.stats['total_sentences'] * 100):.1f}%")
        print(f"   Output file: {args.output}")
        
        # Show sample results
        print(f"\n📝 Sample results:")
        for i, result in enumerate(augmented_stories[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Original: {result.original_text[:100]}...")
            print(f"Enhanced: {result.augmented_text[:150]}...")
            print(f"ToW Count: {result.tow_count}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()