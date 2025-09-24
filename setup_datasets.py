#!/usr/bin/env python3
"""
Automated Dataset Setup Script for Weighted-AST SQL-to-Text Generation

This script downloads and organizes the required datasets for the Weighted-AST system.
It sets up the proper directory structure and downloads the original Spider, SParC, 
and CoSQL datasets along with their SQL-to-Text variants.
"""

import os
import sys
import json
import zipfile
import requests
from pathlib import Path
from typing import Dict, List
import argparse

class DatasetSetup:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "text2sql-datasets"
        self.ast_dir = self.base_dir / "ast-datasets"
        
        # Dataset URLs and configurations
        self.dataset_configs = {
            'spider': {
                'url': 'https://yale-lily.github.io/spider/dataset/spider.zip',
                'structure': {
                    'train': 'train_spider.json',
                    'dev': 'dev.json'
                }
            },
            'sparc': {
                'url': 'https://yale-lily.github.io/sparc/dataset/sparc.zip', 
                'structure': {
                    'train': 'train.json',
                    'dev': 'dev.json'
                }
            },
            'cosql': {
                'url': 'https://yale-lily.github.io/cosql/dataset/cosql.zip',
                'structure': {
                    'train': 'sql_state_tracking/cosql_train.json',
                    'dev': 'sql_state_tracking/cosql_dev.json'
                }
            }
        }
        
        # S2T variant source (ast-icl repository)
        self.ast_icl_repo = 'https://github.com/aliwister/ast-icl.git'

    def create_directory_structure(self):
        """Create the required directory structure."""
        print("Creating directory structure...")
        
        # Create main directories
        self.datasets_dir.mkdir(exist_ok=True)
        self.ast_dir.mkdir(exist_ok=True)
        
        # Create dataset-specific directories
        for dataset in ['spider', 'sparc', 'cosql']:
            dataset_dir = self.datasets_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            
            if dataset == 'cosql':
                # CoSQL has a nested structure
                (dataset_dir / 'sql_state_tracking').mkdir(exist_ok=True)
        
        print(f"✅ Directory structure created in {self.datasets_dir}")

    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL with progress indication."""
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Downloaded {filename}")
            return True
            
        except Exception as e:
            print(f"\n❌ Error downloading {filename}: {e}")
            return False

    def extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """Extract a zip file to the specified directory."""
        try:
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✅ Extracted {zip_path}")
            return True
        except Exception as e:
            print(f"❌ Error extracting {zip_path}: {e}")
            return False

    def setup_dataset(self, dataset_name: str, download_s2t: bool = False) -> bool:
        """Setup a specific dataset."""
        print(f"\n{'='*50}")
        print(f"Setting up {dataset_name.upper()} dataset")
        print(f"{'='*50}")
        
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.datasets_dir / dataset_name
        
        # Download original dataset
        zip_filename = f"{dataset_name}.zip"
        if not self.download_file(config['url'], zip_filename):
            return False
        
        # Extract to temporary directory first
        temp_dir = Path(f"temp_{dataset_name}")
        temp_dir.mkdir(exist_ok=True)
        
        if not self.extract_zip(zip_filename, str(temp_dir)):
            return False
        
        # Move files to correct locations
        try:
            # Find the extracted files (they might be in a subdirectory)
            extracted_files = list(temp_dir.rglob("*.json"))
            
            for file_path in extracted_files:
                filename = file_path.name
                
                # Determine target location based on filename
                if 'train' in filename.lower():
                    if dataset_name == 'cosql':
                        target = dataset_dir / 'sql_state_tracking' / config['structure']['train']
                    else:
                        target = dataset_dir / config['structure']['train']
                elif 'dev' in filename.lower():
                    if dataset_name == 'cosql':
                        target = dataset_dir / 'sql_state_tracking' / config['structure']['dev']
                    else:
                        target = dataset_dir / config['structure']['dev']
                else:
                    continue
                
                # Copy file to target location
                target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(file_path, target)
                print(f"✅ Moved {filename} to {target}")
            
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)
            os.remove(zip_filename)
            
        except Exception as e:
            print(f"❌ Error organizing {dataset_name} files: {e}")
            return False
        
        # Download S2T variant if requested
        if download_s2t:
            print(f"Note: S2T variants should be obtained from the ast-icl repository:")
            print(f"git clone {self.ast_icl_repo}")
            print(f"Then copy s2t-datasets/* to text2sql-datasets/")
        
        return True

    def verify_setup(self) -> bool:
        """Verify that all required files are in place."""
        print("\nVerifying dataset setup...")
        
        required_files = {
            'spider': ['train_spider.json', 'dev.json'],
            'sparc': ['train.json', 'dev.json'],
            'cosql': ['sql_state_tracking/cosql_train.json', 'sql_state_tracking/cosql_dev.json']
        }
        
        all_good = True
        for dataset, files in required_files.items():
            dataset_dir = self.datasets_dir / dataset
            for file in files:
                file_path = dataset_dir / file
                if file_path.exists():
                    print(f"✅ {dataset}/{file}")
                else:
                    print(f"❌ Missing: {dataset}/{file}")
                    all_good = False
        
        if all_good:
            print("\n🎉 All datasets are properly set up!")
        else:
            print("\n⚠️  Some files are missing. Please check the setup.")
        
        return all_good

    def create_sample_config(self):
        """Create a sample configuration file."""
        config = {
            "datasets": {
                "spider": {
                    "train_path": "text2sql-datasets/spider/train_spider.json",
                    "dev_path": "text2sql-datasets/spider/dev.json"
                },
                "sparc": {
                    "train_path": "text2sql-datasets/sparc/train.json", 
                    "dev_path": "text2sql-datasets/sparc/dev.json"
                },
                "cosql": {
                    "train_path": "text2sql-datasets/cosql/sql_state_tracking/cosql_train.json",
                    "dev_path": "text2sql-datasets/cosql/sql_state_tracking/cosql_dev.json"
                }
            },
            "output_dirs": {
                "learned_weights": "learned-weights/",
                "similar_queries": "similar_queries_results_v2/",
                "translations": "ast_result2/",
                "ast_datasets": "ast-datasets/"
            }
        }
        
        config_file = self.base_dir / "dataset_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created configuration file: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="Setup datasets for Weighted-AST SQL-to-Text Generation")
    parser.add_argument("--datasets", nargs="+", default=["spider", "sparc", "cosql"],
                       help="Datasets to setup")
    parser.add_argument("--download-s2t", action="store_true",
                       help="Also download SQL-to-Text variants")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip downloading, only verify existing setup")
    
    args = parser.parse_args()
    
    setup = DatasetSetup()
    
    if not args.skip_download:
        # Create directory structure
        setup.create_directory_structure()
        
        # Setup each dataset
        success_count = 0
        for dataset in args.datasets:
            if dataset in setup.dataset_configs:
                if setup.setup_dataset(dataset, args.download_s2t):
                    success_count += 1
            else:
                print(f"❌ Unknown dataset: {dataset}")
        
        print(f"\n📊 Setup Summary: {success_count}/{len(args.datasets)} datasets completed")
    
    # Verify setup
    if setup.verify_setup():
        setup.create_sample_config()
        print("\n🚀 Ready to run experiments!")
        print("\nNext steps:")
        print("1. python util/feature_importance.py --dataset spider --mode convert")
        print("2. python util/feature_importance.py --dataset spider --mode train")
        print("3. python util/weighted_similarity.py")
        print("4. python util/prompt.py --model_name mistral")
        print("5. python evaluation.py")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
