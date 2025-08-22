#!/usr/bin/env python3
"""
KLUE Results Analysis and Visualization
Analyze and visualize KLUE benchmark results
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import argparse
import glob
from pathlib import Path

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_results_from_directory(results_dir: str) -> dict:
    """Load results from a results directory"""
    results = {}
    
    # Look for JSON result files
    json_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract task and model info from filename or data
            filename = os.path.basename(json_file)
            
            if 'klue_benchmark_results.json' in filename:
                # Main results file
                if 'results' in data:
                    results.update(data['results'])
            elif '_results.json' in filename:
                # Individual task result
                task_type = data.get('task_type')
                model_name = data.get('model_name')
                
                if task_type and model_name:
                    if task_type not in results:
                        results[task_type] = {}
                    results[task_type][model_name] = data.get('metrics', {})
            elif '_summary.json' in filename:
                # Task summary
                if 'results_summary' in data:
                    task_name = data.get('task', '').lower()
                    # Extract task key from task name
                    task_mapping = {
                        'topic classification': 'tc',
                        'sentence textual similarity': 'sts',
                        'natural language inference': 'nli',
                        'named entity recognition': 'ner',
                        'relation extraction': 're',
                        'dependency parsing': 'dp',
                        'machine reading comprehension': 'mrc',
                        'dialogue state tracking': 'dst'
                    }
                    
                    for key, task_code in task_mapping.items():
                        if key in task_name.lower():
                            if task_code not in results:
                                results[task_code] = {}
                            results[task_code].update(data['results_summary'])
                            break
                            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return results

def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create a comparison table of all models and tasks"""
    # Task configurations for metrics
    task_metrics = {
        'tc': 'accuracy',
        'sts': 'pearson',
        'nli': 'accuracy',
        'ner': 'f1',
        're': 'macro_f1',
        'dp': 'las',
        'mrc': 'f1',
        'dst': 'joint_goal_accuracy'
    }
    
    # Get all models
    all_models = set()
    for task_results in results.values():
        all_models.update(task_results.keys())
    all_models = sorted(list(all_models))
    
    # Create table
    table_data = []
    
    for model in all_models:
        row = {'Model': model}
        
        for task, primary_metric in task_metrics.items():
            task_results = results.get(task, {})
            model_results = task_results.get(model, {})
            
            if primary_metric in model_results:
                value = model_results[primary_metric]
            elif model_results:
                # Use first available metric
                value = list(model_results.values())[0]
            else:
                value = 0.0
            
            row[task.upper()] = value
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def plot_model_comparison(df: pd.DataFrame, output_dir: str):
    """Create model comparison plots"""
    # Prepare data for plotting
    models = df['Model'].values
    tasks = [col for col in df.columns if col != 'Model']
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Extract numeric data
    data_for_plot = df[tasks].values
    
    # Create heatmap
    sns.heatmap(data_for_plot, 
                xticklabels=tasks,
                yticklabels=models,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5)
    
    plt.title('KLUE Benchmark Results - Model Comparison Heatmap', fontsize=16)
    plt.xlabel('Tasks', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'klue_model_comparison_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to: {heatmap_path}")
    
    # Create bar plots for each task
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, task in enumerate(tasks):
        if i >= len(axes):
            break
            
        task_scores = df[task].values
        
        # Sort by score
        sorted_indices = np.argsort(task_scores)[::-1]
        sorted_models = [models[j] for j in sorted_indices]
        sorted_scores = [task_scores[j] for j in sorted_indices]
        
        # Create bar plot
        bars = axes[i].bar(range(len(sorted_models)), sorted_scores)
        axes[i].set_title(f'{task}', fontsize=12)
        axes[i].set_xlabel('Models', fontsize=10)
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].set_xticks(range(len(sorted_models)))
        axes[i].set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=8)
        
        # Add value labels on bars
        for bar, score in zip(bars, sorted_scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('KLUE Benchmark Results - Task-wise Comparison', fontsize=16)
    plt.tight_layout()
    
    barplot_path = os.path.join(output_dir, 'klue_task_comparison_bars.png')
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bar plots saved to: {barplot_path}")

def plot_base_vs_tow_comparison(df: pd.DataFrame, output_dir: str):
    """Compare base models vs ToW trained models"""
    # Identify base and ToW models
    base_models = []
    tow_models = []
    
    for model in df['Model']:
        if '-ToW' in model:
            tow_models.append(model)
        else:
            base_models.append(model)
    
    if not base_models or not tow_models:
        print("Not enough base/ToW models for comparison")
        return
    
    # Create comparison plot
    tasks = [col for col in df.columns if col != 'Model']
    
    # Calculate average scores
    base_avg = df[df['Model'].isin(base_models)][tasks].mean()
    tow_avg = df[df['Model'].isin(tow_models)][tasks].mean()
    
    # Create comparison plot
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    
    bars1 = plt.bar(x - width/2, base_avg.values, width, label='Base Models', alpha=0.8)
    bars2 = plt.bar(x + width/2, tow_avg.values, width, label='ToW Models', alpha=0.8)
    
    plt.xlabel('Tasks')
    plt.ylabel('Average Score')
    plt.title('Base Models vs ToW Models - Average Performance Comparison')
    plt.xticks(x, tasks)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, 'base_vs_tow_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Base vs ToW comparison saved to: {comparison_path}")
    
    # Print improvement statistics
    print("\nBase vs ToW Model Improvements:")
    print("=" * 40)
    
    for task in tasks:
        base_score = base_avg[task]
        tow_score = tow_avg[task]
        improvement = tow_score - base_score
        improvement_pct = (improvement / base_score) * 100 if base_score > 0 else 0
        
        print(f"{task:4}: Base={base_score:.4f}, ToW={tow_score:.4f}, "
              f"Δ={improvement:+.4f} ({improvement_pct:+.2f}%)")

def generate_summary_report(results: dict, df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive summary report"""
    report = []
    report.append("# KLUE Benchmark Evaluation Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall statistics
    num_models = len(df)
    num_tasks = len([col for col in df.columns if col != 'Model'])
    
    report.append("## Overview")
    report.append(f"- **Number of models evaluated**: {num_models}")
    report.append(f"- **Number of tasks**: {num_tasks}")
    report.append(f"- **Total evaluations**: {num_models * num_tasks}")
    report.append("")
    
    # Best performing models per task
    report.append("## Best Performing Models by Task")
    tasks = [col for col in df.columns if col != 'Model']
    
    for task in tasks:
        best_idx = df[task].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_score = df.loc[best_idx, task]
        report.append(f"- **{task}**: {best_model} ({best_score:.4f})")
    
    report.append("")
    
    # Model rankings
    report.append("## Overall Model Rankings")
    # Calculate average score across all tasks
    task_cols = [col for col in df.columns if col != 'Model']
    df_copy = df.copy()
    df_copy['Average'] = df_copy[task_cols].mean(axis=1)
    df_sorted = df_copy.sort_values('Average', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        report.append(f"{i+1}. **{row['Model']}** (Average: {row['Average']:.4f})")
    
    report.append("")
    
    # Task difficulty analysis
    report.append("## Task Difficulty Analysis")
    report.append("(Based on average scores across all models)")
    
    task_averages = df[tasks].mean().sort_values(ascending=False)
    for i, (task, avg_score) in enumerate(task_averages.items()):
        difficulty = "Easy" if avg_score > 0.8 else "Medium" if avg_score > 0.5 else "Hard"
        report.append(f"{i+1}. **{task}**: {avg_score:.4f} ({difficulty})")
    
    # Save report
    report_path = os.path.join(output_dir, 'klue_evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze KLUE benchmark results')
    parser.add_argument('results_dir', help='Directory containing KLUE results')
    parser.add_argument('--output', '-o', help='Output directory for analysis', default='analysis_output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading results from: {args.results_dir}")
    print(f"Analysis output directory: {args.output}")
    
    # Load results
    results = load_results_from_directory(args.results_dir)
    
    if not results:
        print("No results found in the specified directory")
        return
    
    print(f"Found results for {len(results)} tasks")
    for task, models in results.items():
        print(f"  {task.upper()}: {len(models)} models")
    
    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(results)
    
    # Save table
    csv_path = os.path.join(args.output, 'klue_results_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_model_comparison(df, args.output)
    
    # Base vs ToW comparison if applicable
    tow_models = [model for model in df['Model'] if '-ToW' in model]
    if tow_models:
        print("Creating Base vs ToW comparison...")
        plot_base_vs_tow_comparison(df, args.output)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results, df, args.output)
    
    print(f"\nAnalysis completed! All outputs saved to: {args.output}")

if __name__ == "__main__":
    main()