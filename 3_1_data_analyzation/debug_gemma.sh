#!/bin/bash

cd /scratch/jsong132/Increase_MLLM_Ability/3_1_data_analyzation

python3 << 'EOF'
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/scratch/jsong132/Increase_MLLM_Ability/Base_Models/google_gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

print("\n=== Model Structure ===")
print(f"Model type: {type(model).__name__}")

if hasattr(model, 'model'):
    print(f"\nmodel.model attributes:")
    for attr in dir(model.model):
        if not attr.startswith('_'):
            try:
                val = getattr(model.model, attr)
                if isinstance(val, torch.nn.ModuleList):
                    print(f"  {attr}: ModuleList with {len(val)} items")
            except:
                pass

print("\n=== Config ===")
if hasattr(model.config, 'num_hidden_layers'):
    print(f"num_hidden_layers: {model.config.num_hidden_layers}")
if hasattr(model.config, 'text_config'):
    print(f"text_config: {model.config.text_config}")
    if hasattr(model.config.text_config, 'num_hidden_layers'):
        print(f"  text_config.num_hidden_layers: {model.config.text_config.num_hidden_layers}")
EOF