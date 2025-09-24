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

### Step 5: Complete the Loop with REDSQL

To complete the evaluation loop by translating the generated natural language back to SQL, we use the [REDSQL_VLDB repository](https://github.com/httdty/REDSQL_VLDB):

#### 5.1 Setup REDSQL Environment

```bash
# Clone the REDSQL repository
git clone https://github.com/httdty/REDSQL_VLDB.git
cd REDSQL_VLDB

# Install system requirements
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk

# Create conda environment
conda create -n red python=3.9
conda activate red

# Install dependencies
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge nmslib
pip install -r requirements.txt

# Create output directories
mkdir output logs
```

#### 5.2 Prepare Your Generated Translations

Convert your generated translations from `ast_result/` to REDSQL format:

```bash
# Convert your Weighted-AST translations to REDSQL input format
python convert_translations_for_redsql.py \
    --input_dir ../SQL-to-Text-Generation-with-Weighted-AST-Few-Shot-Prompting/ast_result/ \
    --output_file ./datasets/spider/dev.json \
    --dataset spider
```

#### 5.3 Build Content Index (REDSQL's Key Innovation)

```bash
# Build searchable index of database content for value matching
python -m pre_processing.build_contents_index \
    --output_dir=./index/spider/db_contents_index/ \
    --db_dir=./datasets/spider/database/
```

#### 5.4 Run REDSQL for SQL Generation

```bash
# Run REDSQL to convert natural language back to SQL
python -m main.run \
    --model_name=gpt-4o-2024-08-06 \
    --batch_size=2 \
    --exp_name=weighted_ast_redsql \
    --bug_fix \
    --consistency_num=30 \
    --stage=dev \
    --db_content_index_path=./index/spider/db_contents_index/ \
    --annotation=./datasets/spider/dev_annotation.json \
    --output_dir=./output \
    --dev_file=./datasets/spider/dev.json \
    --table_file=./datasets/spider/dev_tables.json \
    --db_dir=./datasets/spider/database
```

#### 5.5 Evaluate Results (EM and EX)

```bash
# Evaluate Exact Match (EM) and Execution Accuracy (EX)
python -m eval.evaluate \
    --gold_file=./datasets/spider/dev_gold.sql \
    --pred_file=./output/predicted_sql.txt \
    --db_dir=./datasets/spider/database \
    --output_file=./output/evaluation_results.json
```

**REDSQL Key Features:**
- **Content Retrieval**: Uses Lucene search to find relevant database values
- **Value Matching**: Matches natural language to actual database content  
- **Schema Analysis**: Comprehensive table structures with PKs/FKs
- **LLM Integration**: Real API calls for SQL generation
- **Bug Fixing**: Automatic SQL error correction
- **Consistency Checking**: Multiple validation passes

This completes the full pipeline: SQL → Natural Language → SQL, allowing us to measure both Exact Match (EM) and Execution Accuracy (EX) for the complete round-trip translation.

## 🔧 Complete Pipeline Execution

Run the entire pipeline with a single command:

```bash
python run_experiments.py --datasets spider sparc cosql --models mistral code-llama gpt-j
```

## 📁 Directory Structure

```
SQL-to-Text-Generation-with-Weighted-AST-Few-Shot-Prompting/
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
├── run_experiments.py          # Complete pipeline execution
└── evaluation.py               # Evaluation metrics
```


## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- SQLite3
- CUDA (optional, for GPU acceleration)

