#!/usr/bin/env python3
"""
Convert PARQUET files in 2_datasets/MMLU_ProX to JSONL format.
This script reads the PARQUET files and converts them to JSONL format for easier processing.
"""

import pandas as pd
import json
import os
from pathlib import Path

def convert_parquet_to_jsonl(parquet_file_path, jsonl_file_path):
    """
    Convert a PARQUET file to JSONL format.
    
    Args:
        parquet_file_path (str): Path to the input PARQUET file
        jsonl_file_path (str): Path to the output JSONL file
    """
    try:
        # Read the PARQUET file
        print(f"Reading PARQUET file: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)
        
        # Display basic information about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Convert to JSONL format
        print(f"Converting to JSONL format: {jsonl_file_path}")
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Convert each row to a dictionary and write as JSON line
                json_line = json.dumps(row.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"Successfully converted {len(df)} records to JSONL format")
        
    except Exception as e:
        print(f"Error converting {parquet_file_path}: {str(e)}")
        return False
    
    return True

def main():
    """Main function to convert all PARQUET files in the MMLU_ProX directory."""
    
    # Define the input and output directories
    input_dir = Path("2_datasets/MMLU_ProX")
    output_dir = Path("2_datasets/MMLU_ProX")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PARQUET files in the directory
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("No PARQUET files found in the directory")
        return
    
    print(f"Found {len(parquet_files)} PARQUET file(s) to convert:")
    for file in parquet_files:
        print(f"  - {file.name}")
    
    print("\n" + "="*50)
    
    # Convert each PARQUET file to JSONL
    success_count = 0
    for parquet_file in parquet_files:
        # Create output filename by replacing .parquet with .jsonl
        jsonl_file = output_dir / (parquet_file.stem + ".jsonl")
        
        print(f"\nProcessing: {parquet_file.name}")
        print("-" * 30)
        
        if convert_parquet_to_jsonl(str(parquet_file), str(jsonl_file)):
            success_count += 1
            print(f"✓ Successfully converted to {jsonl_file.name}")
        else:
            print(f"✗ Failed to convert {parquet_file.name}")
    
    print("\n" + "="*50)
    print(f"Conversion completed: {success_count}/{len(parquet_files)} files converted successfully")
    
    # List the created JSONL files
    jsonl_files = list(output_dir.glob("*.jsonl"))
    if jsonl_files:
        print(f"\nCreated JSONL files:")
        for file in jsonl_files:
            file_size = file.stat().st_size
            print(f"  - {file.name} ({file_size:,} bytes)")

if __name__ == "__main__":
    main()

