# Weighted-AST SQL-to-Text Generation

This repository implements the Weighted-AST retrieval with prompting system for SQL-to-Text generation, as described in our research paper. The system uses Abstract Syntax Tree (AST) features with learned weights to retrieve semantically relevant examples for few-shot prompting, enabling accurate natural language descriptions of SQL queries.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/iamsriom/SQL-to-Text-Generation-with-Weighted-AST-Few-Shot-Prompting.git
cd SQL-to-Text-Generation-with-Weighted-AST-Few-Shot-Prompting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

#### Download Original Datasets

**Spider Dataset:**
```bash
# Download Spider dataset
wget https://yale-lily.github.io/spider/dataset/spider.zip
unzip spider.zip
```

**SParC Dataset:**
```bash
# Download SParC dataset  
wget https://yale-lily.github.io/sparc/dataset/sparc.zip
unzip sparc.zip
```

**CoSQL Dataset:**
```bash
# Download CoSQL dataset
wget https://yale-lily.github.io/cosql/dataset/cosql.zip
unzip cosql.zip
```

#### Download SQL-to-Text (S2T) Variants

The S2T variants are the original dev datasets for Spider, SParC, and CoSQL with additional natural language translations. You can obtain them from the [ast-icl repository](https://github.com/aliwister/ast-icl):

```bash
# Clone the ast-icl repository to get S2T variants
git clone https://github.com/aliwister/ast-icl.git
cd ast-icl

# The S2T datasets are located in:
# - s2t-datasets/spider-s2t/
# - s2t-datasets/sparc-s2t/
# - s2t-datasets/cosql-s2t/
```

**Note**: The original dev datasets for Spider, SParC, and CoSQL are the S2T variants - they contain the same SQL queries with additional natural language translations for evaluation.

#### Dataset Directory Structure

Create the following directory structure:

```
WeightedAST/
├── text2sql-datasets/
│   ├── spider/
│   │   ├── train_spider.json
│   │   └── dev.json
│   ├── sparc/
│   │   ├── train.json
│   │   └── dev.json
│   └── cosql/
│       └── sql_state_tracking/
│           ├── cosql_train.json
│           └── cosql_dev.json
├── ast-datasets/  # Empty folder for AST-converted datasets
└── ...
```

### 3. Dataset Organization

Organize your datasets in the following directory structure:

```bash
# Create directory structure
mkdir -p text2sql-datasets/spider
mkdir -p text2sql-datasets/sparc  
mkdir -p text2sql-datasets/cosql/sql_state_tracking
mkdir -p ast-datasets

# Copy your datasets to the appropriate directories
# Place Spider train/dev files in text2sql-datasets/spider/
# Place SParC train/dev files in text2sql-datasets/sparc/
# Place CoSQL train/dev files in text2sql-datasets/cosql/sql_state_tracking/
```

**For S2T variants**, copy them from the ast-icl repository:

```bash
# Clone the ast-icl repository to get S2T variants
git clone https://github.com/aliwister/ast-icl.git
cd ast-icl

# Copy S2T datasets to your project
cp -r s2t-datasets/* ../SQL-to-Text-Generation-with-Weighted-AST-Few-Shot-Prompting/text2sql-datasets/
```

## 📊 Complete Workflow

### Step 1: Convert SQL Queries to AST Features

```bash
# Convert training datasets to AST format
python util/feature_importance.py --dataset spider --mode convert
python util/feature_importance.py --dataset sparc --mode convert  
python util/feature_importance.py --dataset cosql --mode convert
```

### Step 2: Learn Feature Weights

```bash
# Learn feature importance weights for each dataset
python util/feature_importance.py --dataset spider --mode train
python util/feature_importance.py --dataset sparc --mode train
python util/feature_importance.py --dataset cosql --mode train
```

**Output:** Weights are stored in `learned-weights/` directory:
- `spider_feature_importance_v2.json`
- `sparc_feature_importance_v2.json` 
- `cosql_feature_importance_v2.json`

### Step 3: Find Similar Examples

```bash
# Find top-k similar queries for each dev query
python util/weighted_similarity.py
```

**Output:** Similar queries stored in `similar_queries_results_v2/` directory.

### Step 4: Generate Translations

```bash
# Generate SQL-to-Text translations using learned weights
python util/prompt.py --model_name mistral --datasets spider sparc cosql
```

**Output:** Translations stored in `ast_result/` directory.

### Step 5: Complete the Loop with REDSQL_VLDB

To complete the evaluation loop by translating the generated natural language back to SQL, we use the REDSQL_VLDB codebase:

```bash
# Clone the REDSQL_VLDB repository
git clone https://github.com/your-repo/REDSQL_VLDB-main.git
cd REDSQL_VLDB-main

# Set up REDSQL environment
pip install -r requirements.txt

# Use the generated translations from ast_result/ directory
# Feed them into REDSQL to get SQL queries back
python redsql_main.py --input_dir ../SQL-to-Text-Generation-with-Weighted-AST-Few-Shot-Prompting/ast_result/ --output_dir redsql_output/

# Evaluate the round-trip results
python evaluate_roundtrip.py --original_sql_dir ../text2sql-datasets/ --generated_sql_dir redsql_output/
```

This completes the full pipeline: SQL → Natural Language → SQL, allowing us to measure both Exact Match (EM) and Execution Accuracy (EX) for the complete round-trip translation.

## 🔧 Complete Pipeline Execution

Run the entire pipeline with a single command:

```bash
python run_experiments.py --datasets spider sparc cosql --models mistral code-llama gpt-j
```

## 📁 Directory Structure

```
WeightedAST/
├── util/
│   ├── feature_importance.py    # AST feature extraction and weight learning
│   ├── weighted_similarity.py   # Similarity computation and retrieval
│   └── prompt.py                 # LLM translation with few-shot prompting
├── text2sql-datasets/           # Original datasets (Spider, SParC, CoSQL)
├── ast-datasets/                # AST-converted datasets (empty initially)
├── learned-weights/             # Learned feature weights
├── similar_queries_results_v2/  # Retrieved similar examples
├── ast_result/                 # Generated translations
├── requirements.txt             # Python dependencies
├── setup_datasets.py           # Automated dataset setup
├── run_experiments.py          # Complete pipeline execution
└── evaluation.py               # Evaluation metrics
```


## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- SQLite3
- CUDA (optional, for GPU acceleration)

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{chakrabarti2024weighted,
  title={SQL-to-Text Generation with Weighted-AST Few-Shot Prompting},
  author={Chakrabarti, Sriom and Ma, Chuangtao and Khan, Arijit and Link, Sebastian},
  journal={Proceedings of the VLDB Endowment},
  year={2024}
}
```
