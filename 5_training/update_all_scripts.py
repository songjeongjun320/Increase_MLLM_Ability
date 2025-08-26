#!/usr/bin/env python3
"""
Simple script to update all training scripts with caching
"""

import shutil
from pathlib import Path

# Define model configs for each script
model_configs = {
    "ToW_Training_llama.py": {
        "model_name": "Llama-3.1-8B-Instruct-ToW",
        "model_id": "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama-3.1-8B-Instruct",
        "deepspeed_config": "./deepspeed_config_llama.json"
    },
    "ToW_Training_mistral.py": {
        "model_name": "Mistral-7B-Instruct-v0.3-ToW", 
        "model_id": "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-7B-Instruct-v0.3",
        "deepspeed_config": "./deepspeed_config_mistral.json"
    },
    "ToW_Training_qwen.py": {
        "model_name": "Qwen2.5-7B-Instruct-ToW",
        "model_id": "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct", 
        "deepspeed_config": "./deepspeed_config_qwen.json"
    }
}

def update_script(script_name: str, config: dict):
    """Update a script with the new caching system"""
    print(f"Updating {script_name}...")
    
    # Create backup
    backup_name = script_name.replace('.py', '_backup.py')
    shutil.copy(script_name, backup_name)
    print(f"  Created backup: {backup_name}")
    
    # Read the deepseek script as template (it's already updated)
    with open("ToW_Training_deepseek.py", 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Replace the model config section
    updated_content = template_content.replace(
        'ModelConfig(\n        name="DeepSeek-R1-0528-Qwen3-8B-ToW",\n        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",\n        use_quantization=True,\n    )',
        f'ModelConfig(\n        name="{config["model_name"]}",\n        model_id="{config["model_id"]}",\n        use_quantization=True,\n    )'
    )
    
    # Replace deepspeed config
    updated_content = updated_content.replace(
        'deepspeed="./deepspeed_config_deepseek.json"',
        f'deepspeed="{config["deepspeed_config"]}"'
    )
    
    # Write the updated script
    with open(script_name, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"  Successfully updated {script_name}")

def main():
    print("Updating all training scripts with caching system...")
    print("=" * 60)
    
    for script_name, config in model_configs.items():
        if Path(script_name).exists():
            update_script(script_name, config)
        else:
            print(f"WARNING: {script_name} not found, skipping")
        print()
    
    print("=" * 60)
    print("Update complete!")
    print("\nAll scripts now use the intelligent caching system:")
    print("✅ Model-specific caches (no conflicts)")
    print("✅ Automatic cache detection and loading")
    print("✅ Huge time savings on subsequent runs")
    print("✅ Cache invalidation when data changes")

if __name__ == "__main__":
    main()