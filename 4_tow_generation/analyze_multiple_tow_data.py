#!/usr/bin/env python3
"""
Multiple TOW Data Analysis Script

This script analyzes the 4 JSON files in the multiple_tow_data directory:
- Counts total number of data elements
- Counts [ERROR] occurrences in tow elements
- Displays summary statistics

JSON structure:
- Each file contains a list of objects
- Each object has: id, original_sentence, context, gold_label[], tows[], completed_count, total_count
- tows array contains ToW (Theory of Wisdom) text elements
"""

import json
import os
import re
from pathlib import Path


def analyze_json_file(file_path):
    """Analyze a single JSON file and return statistics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {
            'file_name': os.path.basename(file_path),
            'total_elements': len(data),
            'total_gold_labels': 0,
            'total_tows': 0,
            'error_count': 0,
            'completed_count_sum': 0,
            'total_count_sum': 0
        }
        
        for item in data:
            # Count gold_label elements
            if 'gold_label' in item and isinstance(item['gold_label'], list):
                stats['total_gold_labels'] += len(item['gold_label'])
            
            # Count tow elements and [ERROR] occurrences
            if 'tows' in item and isinstance(item['tows'], list):
                stats['total_tows'] += len(item['tows'])
                
                # Count [ERROR] in tow elements
                for tow in item['tows']:
                    if isinstance(tow, str):
                        error_matches = re.findall(r'\[ERROR\]', tow)
                        stats['error_count'] += len(error_matches)
            
            # Sum completed_count and total_count
            if 'completed_count' in item:
                stats['completed_count_sum'] += item['completed_count']
            if 'total_count' in item:
                stats['total_count_sum'] += item['total_count']
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None


def main():
    """Main function to analyze all JSON files."""
    # Define the directory path
    data_dir = Path("multiple_tow_data")
    
    if not data_dir.exists():
        print(f"Directory {data_dir} not found!")
        return
    
    # Find all JSON files
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print("=== Multiple TOW Data Analysis ===")
    print(f"Found {len(json_files)} JSON files\n")
    
    # Initialize total statistics
    total_stats = {
        'total_elements': 0,
        'total_gold_labels': 0,
        'total_tows': 0,
        'error_count': 0,
        'completed_count_sum': 0,
        'total_count_sum': 0
    }
    
    # Analyze each file
    file_stats = []
    for json_file in sorted(json_files):
        stats = analyze_json_file(json_file)
        if stats:
            file_stats.append(stats)
            
            # Print file-specific stats
            print(f"[FILE] {stats['file_name']}")
            print(f"   Data elements: {stats['total_elements']:,}")
            print(f"   Gold labels: {stats['total_gold_labels']:,}")
            print(f"   TOW elements: {stats['total_tows']:,}")
            print(f"   [ERROR] count: {stats['error_count']:,}")
            print(f"   Completed count sum: {stats['completed_count_sum']:,}")
            print(f"   Total count sum: {stats['total_count_sum']:,}")
            print()
            
            # Add to totals
            for key in total_stats:
                total_stats[key] += stats[key]
    
    # Print summary statistics
    print("=" * 50)
    print("[SUMMARY] STATISTICS")
    print("=" * 50)
    print(f"Total JSON files processed: {len(file_stats)}")
    print(f"Total data elements: {total_stats['total_elements']:,}")
    print(f"Total gold_label elements: {total_stats['total_gold_labels']:,}")
    print(f"Total TOW elements: {total_stats['total_tows']:,}")
    print(f"Total [ERROR] occurrences: {total_stats['error_count']:,}")
    print(f"Sum of completed_count: {total_stats['completed_count_sum']:,}")
    print(f"Sum of total_count: {total_stats['total_count_sum']:,}")
    
    # Calculate some additional statistics
    if total_stats['total_elements'] > 0:
        avg_gold_labels = total_stats['total_gold_labels'] / total_stats['total_elements']
        avg_tows = total_stats['total_tows'] / total_stats['total_elements']
        print(f"\n[AVERAGES] PER DATA ELEMENT:")
        print(f"Average gold_labels per element: {avg_gold_labels:.2f}")
        print(f"Average TOW elements per element: {avg_tows:.2f}")
    
    if total_stats['total_tows'] > 0:
        error_rate = (total_stats['error_count'] / total_stats['total_tows']) * 100
        print(f"[ERROR] rate in TOW elements: {error_rate:.2f}%")
    
    if total_stats['total_count_sum'] > 0:
        completion_rate = (total_stats['completed_count_sum'] / total_stats['total_count_sum']) * 100
        print(f"Overall completion rate: {completion_rate:.2f}%")


if __name__ == "__main__":
    main()