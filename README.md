# Weighted-AST SQL-to-Text Generation

This repository implements the Weighted-AST retrieval with prompting system for SQL-to-Text generation, as described in our research paper. The system uses Abstract Syntax Tree (AST) features with learned weights to retrieve semantically relevant examples for few-shot prompting, enabling accurate natural language descriptions of SQL queries.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd WeightedAST

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

### 3. Automated Setup

Use our setup script to automatically download and organize datasets:

```bash
python setup_datasets.py
```

**For S2T variants**, you'll need to manually copy them from the ast-icl repository:

```bash
# After cloning ast-icl repository
cp -r ast-icl/s2t-datasets/* text2sql-datasets/
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

**Output:** Translations stored in `ast_result2/` directory.

### Step 5: Evaluate Results

```bash
# Evaluate using exact match and execution accuracy
python evaluation.py
```

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
├── ast_result2/                # Generated translations
├── requirements.txt             # Python dependencies
├── setup_datasets.py           # Automated dataset setup
├── run_experiments.py          # Complete pipeline execution
└── evaluation.py               # Evaluation metrics
```

## 🧠 Methodology

### Feature Extraction
Our approach extracts two categories of features:

1. **SQL Surface Features**: Keywords, aggregation functions, table names
2. **AST-based Structural Features**: 
   - Node-type features (TYPE:Statement, TYPE:Identifier)
   - Keyword features within syntactic context
   - Function features (FUNCTION:AVG, FUNCTION:COUNT)
   - Identifier features (IDENTIFIER:singer, IDENTIFIER:age)
   - Depth features capturing hierarchical structure

### Feature Weighting
Weights combine two complementary signals:

```
w(f) = α·IDF(f) + (1-α)·Attn(f)
```

- **IDF**: Global informativeness (down-weights common elements like SELECT, FROM)
- **Attention**: Query-specific contextual relevance (learned via self-supervised model)
- **α = 0.5**: Balances global and local importance

### Similarity Computation
Weighted similarity between queries:

```
S(Qt,Qi) = Σ w(f)·min(cQt(f),cQi(f)) / Σ w(f)
```

### Few-Shot Prompting
- Retrieves top-k=5 most similar training examples
- Uses chain-of-thought prompting with structured guidelines
- Generates natural language descriptions with semantic fidelity

## 📈 Evaluation

### Metrics
- **Exact Match (EM)**: Syntactic equivalence between generated and reference SQL
- **Execution Accuracy (EX)**: Whether queries yield identical results
- **Human Evaluation**: Expert assessment of semantic correctness

### Results
Our method outperforms Graph-AST ICL baseline:
- **Spider-S2T**: Up to 91.97% accuracy (vs 73.72% baseline)
- **SParC-S2T**: Up to 65.7% accuracy (vs 56.1% baseline)  
- **CoSQL-S2T**: Up to 74.5% accuracy (vs 21.4% baseline)

## 🔑 Key Files

- **`util/feature_importance.py`**: Core feature extraction and weight learning
- **`util/weighted_similarity.py`**: Similarity computation and retrieval
- **`util/prompt.py`**: LLM translation with few-shot prompting
- **`evaluation.py`**: Comprehensive evaluation metrics
- **`run_experiments.py`**: Complete pipeline execution

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- SQLite3
- CUDA (optional, for GPU acceleration)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

## 🆘 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use CPU mode by setting `CUDA_VISIBLE_DEVICES=""`
2. **Dataset Not Found**: Ensure datasets are in correct directory structure
3. **Permission Denied**: Check file permissions and virtual environment activation

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review the [GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md) for detailed setup instructions
- Contact the authors for research-related questions

## 🔄 Workflow Summary

1. **Setup**: Download datasets and install dependencies
2. **Convert**: Transform SQL queries to AST features  
3. **Learn**: Train feature importance weights
4. **Retrieve**: Find similar examples using weighted similarity
5. **Translate**: Generate natural language descriptions
6. **Evaluate**: Assess quality using multiple metrics

This complete pipeline enables reproducible research and easy experimentation with the Weighted-AST methodology for SQL-to-Text generation.
