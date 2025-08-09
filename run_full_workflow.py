#!/usr/bin/env python3
"""
TOW (Thoughts of Words) Option 2 Complete Workflow Execution Script
================================================================

Executes the complete TOW Option 2 workflow:
1. Download models and datasets
2. Run baseline evaluations  
3. Generate ToW-augmented training data
4. Train models with ToW data
5. Compare before/after performance

Usage:
    python run_full_workflow.py --all              # Run complete workflow
    python run_full_workflow.py --setup            # Setup only (download models/data)
    python run_full_workflow.py --baseline         # Run baseline evaluation
    python run_full_workflow.py --generate-tow     # Generate ToW data
    python run_full_workflow.py --train            # Train models with ToW
    python run_full_workflow.py --evaluate         # Final evaluation and comparison
    python run_full_workflow.py --gpu-check        # Check GPU compatibility
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path
from datetime import datetime

class TOWWorkflowRunner:
    """Execute TOW Option 2 complete workflow"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / "config_option2.yaml"
        self.log_file = self.project_root / f"workflow_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Load configuration
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"[ERROR] Configuration file not found: {self.config_file}")
            sys.exit(1)
    
    def log_and_print(self, message: str):
        """Log message to both console and log file"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} {message}\n")
    
    def run_command(self, cmd: str, cwd: Path = None) -> bool:
        """Execute shell command and return success status"""
        if cwd is None:
            cwd = self.project_root
            
        self.log_and_print(f"[CMD] {cmd}")
        self.log_and_print(f"[CWD] {cwd}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=cwd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8'
            )
            
            if result.stdout:
                self.log_and_print(f"[STDOUT] {result.stdout.strip()}")
            if result.stderr:
                self.log_and_print(f"[STDERR] {result.stderr.strip()}")
                
            if result.returncode == 0:
                self.log_and_print(f"[SUCCESS] Command completed successfully")
                return True
            else:
                self.log_and_print(f"[ERROR] Command failed with code {result.returncode}")
                return False
                
        except Exception as e:
            self.log_and_print(f"[ERROR] Exception running command: {e}")
            return False
    
    def check_gpu_compatibility(self):
        """Check GPU compatibility for different models"""
        self.log_and_print("="*60)
        self.log_and_print("[PHASE 0] GPU Compatibility Check")
        self.log_and_print("="*60)
        
        # Check GPT-OSS-20B compatibility
        success = self.run_command(
            "python download_gpt_oss_20b.py --info", 
            self.project_root / "1_models"
        )
        
        # Check GPT-OSS-120B compatibility  
        success = self.run_command(
            "python download_gpt_oss_120b.py --check",
            self.project_root / "1_models"
        )
        
        self.log_and_print("[INFO] GPU compatibility check completed")
        return True
    
    def setup_phase(self):
        """Phase 1: Download models and datasets"""
        self.log_and_print("="*60)
        self.log_and_print("[PHASE 1] Setup - Download Models and Datasets")
        self.log_and_print("="*60)
        
        # Create directories
        dirs_to_create = [
            "1_models/gpt_oss",
            "1_models/base_models", 
            "2_datasets/benchmarks",
            "2_datasets/korean_stories",
            "6_results/baseline",
            "6_results/tow_training",
            "6_results/comparison"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.log_and_print(f"[CREATED] Directory: {dir_path}")
        
        # Download GPT-OSS model (choose based on GPU)
        self.log_and_print("[DOWNLOAD] GPT-OSS Model for ToW Generation...")
        success = self.run_command(
            "python download_gpt_oss_20b.py", 
            self.project_root / "1_models"
        )
        
        if not success:
            self.log_and_print("[WARNING] GPT-OSS-20B download failed, trying alternative...")
            # Could fallback to different model or continue
        
        # Download base models for training
        base_models = [
            "download_deepseek_r1_distill_qwen_7b.py",
            "download_qwen25_7b_instruct.py"
        ]
        
        for model_script in base_models:
            self.log_and_print(f"[DOWNLOAD] Base model: {model_script}")
            success = self.run_command(
                f"python {model_script}",
                self.project_root / "1_models"
            )
            if not success:
                self.log_and_print(f"[WARNING] Failed to download {model_script}")
        
        # Download datasets
        self.log_and_print("[DOWNLOAD] Korean Evaluation Datasets...")
        success = self.run_command(
            "python download_korean_datasets.py",
            self.project_root / "2_datasets"
        )
        
        self.log_and_print("[PHASE 1] Setup completed")
        return True
    
    def baseline_evaluation_phase(self):
        """Phase 2: Run baseline evaluations"""
        self.log_and_print("="*60)
        self.log_and_print("[PHASE 2] Baseline Evaluation")
        self.log_and_print("="*60)
        
        # Run baseline evaluation on all models
        success = self.run_command(
            "python baseline_evaluation.py --all-models",
            self.project_root / "3_evaluation"
        )
        
        if success:
            self.log_and_print("[SUCCESS] Baseline evaluation completed")
        else:
            self.log_and_print("[ERROR] Baseline evaluation failed")
            return False
        
        return True
    
    def tow_generation_phase(self):
        """Phase 3: Generate ToW-augmented training data"""
        self.log_and_print("="*60)
        self.log_and_print("[PHASE 3] ToW Data Generation")
        self.log_and_print("="*60)
        
        # Generate ToW prompts and data
        success = self.run_command(
            "python korean_tow_generator.py --batch --output ../2_datasets/tow_training_data.jsonl",
            self.project_root / "4_tow_generation"
        )
        
        if success:
            self.log_and_print("[SUCCESS] ToW data generation completed")
        else:
            self.log_and_print("[ERROR] ToW data generation failed")
            return False
        
        return True
    
    def training_phase(self):
        """Phase 4: Train models with ToW data"""
        self.log_and_print("="*60)
        self.log_and_print("[PHASE 4] Model Training with ToW")
        self.log_and_print("="*60)
        
        # Train models with ToW data
        success = self.run_command(
            "python finetune_with_tow.py --all-models",
            self.project_root / "5_training"
        )
        
        if success:
            self.log_and_print("[SUCCESS] ToW training completed")
        else:
            self.log_and_print("[ERROR] ToW training failed")
            return False
        
        return True
    
    def evaluation_phase(self):
        """Phase 5: Final evaluation and comparison"""
        self.log_and_print("="*60)
        self.log_and_print("[PHASE 5] Final Evaluation and Comparison")
        self.log_and_print("="*60)
        
        # Compare baseline vs ToW-trained models
        success = self.run_command(
            "python compare_baseline_vs_tow.py --generate-report",
            self.project_root / "3_evaluation"
        )
        
        if success:
            self.log_and_print("[SUCCESS] Final evaluation completed")
            self.log_and_print(f"[RESULTS] Check results in: {self.project_root / '6_results'}")
        else:
            self.log_and_print("[ERROR] Final evaluation failed")
            return False
        
        return True
    
    def run_full_workflow(self):
        """Execute complete TOW Option 2 workflow"""
        self.log_and_print("="*80)
        self.log_and_print("TOW (Thoughts of Words) Option 2 - Complete Workflow Execution")
        self.log_and_print("="*80)
        self.log_and_print(f"[START] Workflow started at {datetime.now()}")
        self.log_and_print(f"[CONFIG] Using configuration: {self.config_file}")
        self.log_and_print(f"[LOG] Logging to: {self.log_file}")
        
        phases = [
            ("Setup", self.setup_phase),
            ("Baseline Evaluation", self.baseline_evaluation_phase),
            ("ToW Generation", self.tow_generation_phase),
            ("Model Training", self.training_phase),
            ("Final Evaluation", self.evaluation_phase)
        ]
        
        for phase_name, phase_func in phases:
            self.log_and_print(f"\n[STARTING] {phase_name}")
            success = phase_func()
            
            if success:
                self.log_and_print(f"[COMPLETED] {phase_name}")
            else:
                self.log_and_print(f"[FAILED] {phase_name}")
                self.log_and_print("[ERROR] Workflow terminated due to failure")
                return False
        
        self.log_and_print("\n" + "="*80)
        self.log_and_print("[SUCCESS] Complete TOW Option 2 workflow finished successfully!")
        self.log_and_print(f"[END] Workflow completed at {datetime.now()}")
        self.log_and_print(f"[LOG] Full log saved to: {self.log_file}")
        self.log_and_print("="*80)
        
        return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='TOW Option 2 Workflow Runner')
    parser.add_argument('--all', action='store_true', help='Run complete workflow')
    parser.add_argument('--setup', action='store_true', help='Setup only (download models/data)')
    parser.add_argument('--baseline', action='store_true', help='Run baseline evaluation')
    parser.add_argument('--generate-tow', action='store_true', help='Generate ToW data')
    parser.add_argument('--train', action='store_true', help='Train models with ToW')
    parser.add_argument('--evaluate', action='store_true', help='Final evaluation and comparison')
    parser.add_argument('--gpu-check', action='store_true', help='Check GPU compatibility')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    runner = TOWWorkflowRunner()
    
    try:
        if args.gpu_check:
            runner.check_gpu_compatibility()
        elif args.setup:
            runner.setup_phase()
        elif args.baseline:
            runner.baseline_evaluation_phase()
        elif args.generate_tow:
            runner.tow_generation_phase()
        elif args.train:
            runner.training_phase()
        elif args.evaluate:
            runner.evaluation_phase()
        elif args.all:
            runner.run_full_workflow()
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()