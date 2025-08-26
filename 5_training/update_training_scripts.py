#!/usr/bin/env python3
"""
Update all training scripts to use the new caching system
"""

import os
import shutil
from pathlib import Path
import re

def update_training_script(script_path: Path, model_name: str):
    """Update a single training script to use caching"""
    print(f"Updating {script_path.name} for {model_name}...")
    
    # Read the original script
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original file
    backup_path = script_path.with_suffix('.py.backup')
    shutil.copy2(script_path, backup_path)
    print(f"  Created backup: {backup_path.name}")
    
    # Remove the old SmartToWDataProcessor class definition
    # Find the class definition and remove it
    class_pattern = r'class SmartToWDataProcessor:.*?(?=class|\Z)'
    content = re.sub(class_pattern, '', content, flags=re.DOTALL)
    
    # Add import for the new caching utilities after the existing imports
    import_pattern = r'(from transformers import TrainerCallback\n)'
    replacement = r'\1\n# Import the new caching utilities\nfrom dataset_cache_utils import SmartToWDataProcessor\n'
    content = re.sub(import_pattern, replacement, content)
    
    # Update the data processor initialization
    old_init_pattern = r'data_processor = SmartToWDataProcessor\(tokenizer, self\.training_config\)'
    new_init_pattern = r'data_processor = SmartToWDataProcessor(tokenizer, self.training_config, self.model_config.model_id)'
    content = re.sub(old_init_pattern, new_init_pattern, content)
    
    # Update the dataset creation call
    old_dataset_pattern = r'tow_data = data_processor\.load_tow_data\(self\.training_config\.tow_data_paths\)\s+train_dataset = data_processor\.create_training_dataset\(tow_data\)'
    new_dataset_pattern = r'train_dataset = data_processor.create_training_dataset(self.training_config.tow_data_paths)'
    content = re.sub(old_dataset_pattern, new_dataset_pattern, content, flags=re.DOTALL)
    
    # Write the updated content
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  Successfully updated {script_path.name}")

def main():
    """Update all training scripts"""
    script_dir = Path(".")
    
    # Define the training scripts to update
    scripts_to_update = [
        ("ToW_Training_llama.py", "Llama"),
        ("ToW_Training_mistral.py", "Mistral"), 
        ("ToW_Training_qwen.py", "Qwen"),
        # Skip deepseek as it's already updated
    ]
    
    print("Updating training scripts to use caching system...")
    print("=" * 50)
    
    for script_name, model_name in scripts_to_update:
        script_path = script_dir / script_name
        
        if script_path.exists():
            try:
                update_training_script(script_path, model_name)
            except Exception as e:
                print(f"  ERROR updating {script_name}: {e}")
                # Restore from backup if update failed
                backup_path = script_path.with_suffix('.py.backup')
                if backup_path.exists():
                    shutil.copy2(backup_path, script_path)
                    print(f"  Restored {script_name} from backup")
        else:
            print(f"  WARNING: {script_name} not found, skipping")
    
    print("=" * 50)
    print("Update complete!")
    print("\nBenefits of the new caching system:")
    print("- Each model gets its own cache based on model ID + data paths + max length")
    print("- No more repeated tokenization - huge time savings!")
    print("- Automatic cache invalidation when data or config changes")
    print("- Cache management utilities included")
    print("\nCache location: ./cached_datasets/")

if __name__ == "__main__":
    main()