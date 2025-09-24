#!/usr/bin/env python3
"""
Complete Pipeline Execution Script for Weighted-AST SQL-to-Text Generation

This script runs the entire Weighted-AST pipeline from start to finish:
1. Convert SQL queries to AST features
2. Learn feature importance weights
3. Find similar examples using weighted similarity
4. Generate SQL-to-Text translations
5. Evaluate results

Usage:
    python run_experiments.py --datasets spider sparc cosql --models mistral code-llama gpt-j
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import time

class ExperimentRunner:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "experiment_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Available models and datasets
        self.available_models = ["mistral", "code-llama", "gpt-j"]
        self.available_datasets = ["spider", "sparc", "cosql"]
        
        # Pipeline steps
        self.pipeline_steps = [
            "convert_ast",
            "learn_weights", 
            "find_similar",
            "generate_translations",
            "evaluate"
        ]

    def log_step(self, step: str, message: str, status: str = "INFO"):
        """Log a pipeline step with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{step}] [{status}] {message}"
        print(log_entry)
        
        # Also write to log file
        log_file = self.results_dir / "pipeline.log"
        with open(log_file, 'a') as f:
            f.write(log_entry + "\n")

    def run_command(self, command: List[str], step_name: str) -> bool:
        """Run a command and handle errors."""
        self.log_step(step_name, f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                cwd=self.base_dir
            )
            
            if result.returncode == 0:
                self.log_step(step_name, "✅ Completed successfully", "SUCCESS")
                return True
            else:
                self.log_step(step_name, f"❌ Failed with return code {result.returncode}", "ERROR")
                self.log_step(step_name, f"STDERR: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log_step(step_name, f"❌ Exception: {str(e)}", "ERROR")
            return False

    def convert_to_ast(self, dataset: str) -> bool:
        """Convert SQL queries to AST features."""
        self.log_step("convert_ast", f"Converting {dataset} to AST features")
        
        # Create AST datasets directory
        ast_dir = self.base_dir / "ast-datasets"
        ast_dir.mkdir(exist_ok=True)
        
        # Run feature importance script in convert mode
        command = [
            "python", "util/feature_importance.py",
            "--dataset", dataset,
            "--mode", "convert"
        ]
        
        return self.run_command(command, f"convert_ast_{dataset}")

    def learn_weights(self, dataset: str) -> bool:
        """Learn feature importance weights."""
        self.log_step("learn_weights", f"Learning weights for {dataset}")
        
        command = [
            "python", "util/feature_importance.py", 
            "--dataset", dataset,
            "--mode", "train"
        ]
        
        return self.run_command(command, f"learn_weights_{dataset}")

    def find_similar_queries(self) -> bool:
        """Find similar queries using weighted similarity."""
        self.log_step("find_similar", "Finding similar queries")
        
        command = ["python", "util/weighted_similarity.py"]
        
        return self.run_command(command, "find_similar")

    def generate_translations(self, datasets: List[str], models: List[str]) -> bool:
        """Generate SQL-to-Text translations."""
        self.log_step("generate_translations", f"Generating translations for {datasets} with {models}")
        
        all_success = True
        
        for model in models:
            self.log_step("generate_translations", f"Processing model: {model}")
            
            command = [
                "python", "util/prompt.py",
                "--model_name", model,
                "--datasets"] + datasets
            
            if not self.run_command(command, f"generate_translations_{model}"):
                all_success = False
        
        return all_success

    def evaluate_results(self) -> bool:
        """Evaluate the generated translations."""
        self.log_step("evaluate", "Evaluating results")
        
        command = ["python", "evaluation.py"]
        
        return self.run_command(command, "evaluate")

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        self.log_step("prerequisites", "Checking prerequisites")
        
        # Check if datasets exist
        required_dirs = [
            "text2sql-datasets/spider",
            "text2sql-datasets/sparc", 
            "text2sql-datasets/cosql"
        ]
        
        for dir_path in required_dirs:
            if not (self.base_dir / dir_path).exists():
                self.log_step("prerequisites", f"❌ Missing directory: {dir_path}", "ERROR")
                return False
        
        # Check if required files exist
        required_files = [
            "util/feature_importance.py",
            "util/weighted_similarity.py", 
            "util/prompt.py",
            "evaluation.py"
        ]
        
        for file_path in required_files:
            if not (self.base_dir / file_path).exists():
                self.log_step("prerequisites", f"❌ Missing file: {file_path}", "ERROR")
                return False
        
        self.log_step("prerequisites", "✅ All prerequisites met", "SUCCESS")
        return True

    def run_pipeline(self, datasets: List[str], models: List[str], 
                    skip_steps: List[str] = None) -> bool:
        """Run the complete pipeline."""
        self.log_step("pipeline", f"Starting pipeline for datasets: {datasets}, models: {models}")
        
        if skip_steps is None:
            skip_steps = []
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        pipeline_success = True
        
        # Step 1: Convert to AST
        if "convert_ast" not in skip_steps:
            for dataset in datasets:
                if not self.convert_to_ast(dataset):
                    pipeline_success = False
                    break
        
        # Step 2: Learn weights
        if pipeline_success and "learn_weights" not in skip_steps:
            for dataset in datasets:
                if not self.learn_weights(dataset):
                    pipeline_success = False
                    break
        
        # Step 3: Find similar queries
        if pipeline_success and "find_similar" not in skip_steps:
            if not self.find_similar_queries():
                pipeline_success = False
        
        # Step 4: Generate translations
        if pipeline_success and "generate_translations" not in skip_steps:
            if not self.generate_translations(datasets, models):
                pipeline_success = False
        
        # Step 5: Evaluate
        if pipeline_success and "evaluate" not in skip_steps:
            if not self.evaluate_results():
                pipeline_success = False
        
        if pipeline_success:
            self.log_step("pipeline", "🎉 Pipeline completed successfully!", "SUCCESS")
        else:
            self.log_step("pipeline", "❌ Pipeline failed", "ERROR")
        
        return pipeline_success

    def generate_summary(self, datasets: List[str], models: List[str]) -> Dict:
        """Generate a summary of the experiment."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": datasets,
            "models": models,
            "output_directories": {
                "learned_weights": "learned-weights/",
                "similar_queries": "similar_queries_results_v2/",
                "translations": "ast_result2/",
                "evaluation": "EX_EM_results/"
            }
        }
        
        # Check which outputs exist
        for output_type, output_dir in summary["output_directories"].items():
            output_path = self.base_dir / output_dir
            if output_path.exists():
                files = list(output_path.glob("*"))
                summary["output_directories"][output_type] = {
                    "path": str(output_path),
                    "files_count": len(files),
                    "files": [f.name for f in files[:10]]  # First 10 files
                }
        
        # Save summary
        summary_file = self.results_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log_step("summary", f"Experiment summary saved to {summary_file}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Run complete Weighted-AST pipeline")
    parser.add_argument("--datasets", nargs="+", default=["spider", "sparc", "cosql"],
                       help="Datasets to process")
    parser.add_argument("--models", nargs="+", default=["mistral"],
                       help="Models to use for translation")
    parser.add_argument("--skip-steps", nargs="+", default=[],
                       help="Pipeline steps to skip")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check prerequisites, don't run pipeline")
    
    args = parser.parse_args()
    
    # Validate arguments
    runner = ExperimentRunner()
    
    invalid_datasets = [d for d in args.datasets if d not in runner.available_datasets]
    if invalid_datasets:
        print(f"❌ Invalid datasets: {invalid_datasets}")
        print(f"Available datasets: {runner.available_datasets}")
        sys.exit(1)
    
    invalid_models = [m for m in args.models if m not in runner.available_models]
    if invalid_models:
        print(f"❌ Invalid models: {invalid_models}")
        print(f"Available models: {runner.available_models}")
        sys.exit(1)
    
    # Check prerequisites
    if not runner.check_prerequisites():
        print("\n❌ Prerequisites not met. Please run setup_datasets.py first.")
        sys.exit(1)
    
    if args.check_only:
        print("✅ Prerequisites check passed!")
        sys.exit(0)
    
    # Run pipeline
    print(f"\n🚀 Starting Weighted-AST pipeline...")
    print(f"📊 Datasets: {args.datasets}")
    print(f"🤖 Models: {args.models}")
    print(f"⏭️  Skipping: {args.skip_steps}")
    print("="*60)
    
    success = runner.run_pipeline(
        datasets=args.datasets,
        models=args.models,
        skip_steps=args.skip_steps
    )
    
    # Generate summary
    summary = runner.generate_summary(args.datasets, args.models)
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved in: {runner.results_dir}")
        print(f"📋 Summary: {runner.results_dir}/experiment_summary.json")
    else:
        print("\n❌ Pipeline failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
