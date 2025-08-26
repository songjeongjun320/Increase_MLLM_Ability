#!/usr/bin/env python3
"""
Cache management utility for ToW training datasets
"""

import argparse
from pathlib import Path
from dataset_cache_utils import DatasetCacheManager
import json

def list_caches(manager: DatasetCacheManager):
    """List all available caches"""
    caches = manager.list_caches()
    
    if not caches:
        print("No caches found.")
        return
    
    print(f"Found {len(caches)} cache(s):")
    print("=" * 80)
    
    for i, cache_info in enumerate(caches, 1):
        print(f"{i}. {cache_info['model_name']}")
        print(f"   Cache key: {cache_info['cache_key']}")
        print(f"   Examples: {cache_info['num_examples']:,}")
        print(f"   Max length: {cache_info['max_length']}")
        print(f"   Location: {cache_info['cache_dir']}")
        print()

def clear_all_caches(manager: DatasetCacheManager):
    """Clear all caches"""
    response = input("Are you sure you want to clear ALL caches? (y/N): ")
    if response.lower() in ['y', 'yes']:
        manager.clear_cache()
        print("All caches cleared successfully!")
    else:
        print("Operation cancelled.")

def clear_specific_cache(manager: DatasetCacheManager):
    """Clear a specific cache"""
    caches = manager.list_caches()
    
    if not caches:
        print("No caches found.")
        return
    
    print("Available caches:")
    for i, cache_info in enumerate(caches, 1):
        print(f"{i}. {cache_info['model_name']} (key: {cache_info['cache_key']})")
    
    try:
        choice = int(input("Enter cache number to clear: ")) - 1
        if 0 <= choice < len(caches):
            cache_info = caches[choice]
            cache_dir = Path(cache_info['cache_dir'])
            
            response = input(f"Clear cache for {cache_info['model_name']}? (y/N): ")
            if response.lower() in ['y', 'yes']:
                import shutil
                shutil.rmtree(cache_dir)
                print(f"Cache cleared for {cache_info['model_name']}!")
            else:
                print("Operation cancelled.")
        else:
            print("Invalid choice.")
    except (ValueError, KeyError) as e:
        print(f"Error: {e}")

def show_cache_stats(manager: DatasetCacheManager):
    """Show detailed cache statistics"""
    caches = manager.list_caches()
    
    if not caches:
        print("No caches found.")
        return
    
    total_examples = sum(cache['num_examples'] for cache in caches)
    total_size = 0
    
    print("Cache Statistics:")
    print("=" * 50)
    
    for cache_info in caches:
        cache_dir = Path(cache_info['cache_dir'])
        
        # Calculate cache size
        size_mb = 0
        if cache_dir.exists():
            for file in cache_dir.rglob('*'):
                if file.is_file():
                    size_mb += file.stat().st_size
            size_mb = size_mb / (1024 * 1024)  # Convert to MB
        
        total_size += size_mb
        
        print(f"Model: {cache_info['model_name']}")
        print(f"  Examples: {cache_info['num_examples']:,}")
        print(f"  Max length: {cache_info['max_length']}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Cache key: {cache_info['cache_key']}")
        print()
    
    print(f"Total: {len(caches)} caches, {total_examples:,} examples, {total_size:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Manage ToW training dataset caches")
    parser.add_argument('command', choices=['list', 'clear', 'clear-all', 'stats'], 
                       help='Command to execute')
    parser.add_argument('--cache-dir', default='cached_datasets',
                       help='Base cache directory (default: cached_datasets)')
    
    args = parser.parse_args()
    
    manager = DatasetCacheManager(args.cache_dir)
    
    if args.command == 'list':
        list_caches(manager)
    elif args.command == 'clear':
        clear_specific_cache(manager)
    elif args.command == 'clear-all':
        clear_all_caches(manager)
    elif args.command == 'stats':
        show_cache_stats(manager)

if __name__ == "__main__":
    main()