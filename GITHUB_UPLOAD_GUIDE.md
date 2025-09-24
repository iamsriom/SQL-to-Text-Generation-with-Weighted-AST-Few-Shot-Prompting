# GitHub Upload Guide

This guide provides step-by-step instructions for uploading the Weighted-AST codebase to GitHub.

## 📋 Prerequisites

- Git installed on your system
- GitHub account
- GitHub repository created (empty or with README)

## 🚀 Step-by-Step Upload Process

### Step 1: Initialize Git Repository

```bash
# Navigate to your project directory
cd /Users/xv76wk/Desktop/WeightedAST

# Initialize git repository
git init

# Add all files to staging
git add .

# Make initial commit
git commit -m "Initial commit: Weighted-AST SQL-to-Text Generation

- Complete implementation of Weighted-AST methodology
- AST feature extraction and weight learning
- Weighted similarity computation and retrieval
- Few-shot prompting for SQL-to-Text translation
- Comprehensive evaluation framework
- Automated dataset setup and pipeline execution scripts"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in repository details:
   - **Repository name**: `WeightedAST-SQL-to-Text`
   - **Description**: `Weighted-AST retrieval with prompting for SQL-to-Text generation using learned feature importance`
   - **Visibility**: Public (recommended for research)
   - **Initialize**: Do NOT check "Add a README file" (we already have one)

### Step 3: Connect Local Repository to GitHub

```bash
# Add remote origin (replace with your actual repository URL)
git remote add origin https://github.com/YOUR_USERNAME/WeightedAST-SQL-to-Text.git

# Verify remote connection
git remote -v
```

### Step 4: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

### Step 5: Verify Upload

1. Go to your GitHub repository
2. Verify all files are present:
   - ✅ `README.md` - Comprehensive documentation
   - ✅ `requirements.txt` - Python dependencies
   - ✅ `setup_datasets.py` - Automated dataset setup
   - ✅ `run_experiments.py` - Complete pipeline execution
   - ✅ `util/` directory - Core implementation
   - ✅ `evaluation.py` - Evaluation framework
   - ✅ `.gitignore` - Excludes datasets and results
   - ✅ `LICENSE` - MIT license

3. Check that datasets are excluded:
   - ❌ `ast-datasets/` should be empty folder
   - ❌ `text2sql-datasets/` should not be present
   - ❌ `learned-weights/` should not be present
   - ❌ `results/` should not be present

## 🔧 Repository Structure After Upload

```
WeightedAST-SQL-to-Text/
├── .gitignore                 # Excludes datasets and results
├── LICENSE                    # MIT License
├── README.md                  # Comprehensive documentation
├── requirements.txt           # Python dependencies
├── setup_datasets.py         # Automated dataset setup
├── run_experiments.py        # Complete pipeline execution
├── evaluation.py             # Evaluation framework
├── util/                     # Core implementation
│   ├── feature_importance.py # AST feature extraction and weight learning
│   ├── weighted_similarity.py # Similarity computation and retrieval
│   └── prompt.py             # LLM translation with few-shot prompting
└── ast-datasets/            # Empty folder for AST-converted datasets
```

## 🎯 Post-Upload Tasks

### 1. Update Repository Settings

1. Go to repository **Settings**
2. Scroll to **Features** section
3. Enable **Issues** and **Wiki** if desired
4. Set up **Branch protection rules** for main branch

### 2. Create Release

1. Go to **Releases** section
2. Click **Create a new release**
3. Tag version: `v1.0.0`
4. Release title: `Weighted-AST SQL-to-Text Generation v1.0.0`
5. Description:
   ```
   Initial release of Weighted-AST SQL-to-Text Generation system.
   
   Features:
   - AST-based feature extraction with learned weights
   - Weighted similarity computation for few-shot retrieval
   - Chain-of-thought prompting for SQL-to-Text translation
   - Comprehensive evaluation framework
   - Automated dataset setup and pipeline execution
   
   This release includes the complete implementation described in our research paper.
   ```

### 3. Add Topics/Tags

Add relevant topics to your repository:
- `sql-to-text`
- `ast`
- `few-shot-learning`
- `nlp`
- `machine-learning`
- `sql`
- `natural-language-processing`

## 🔍 Verification Checklist

Before considering the upload complete, verify:

- [ ] All source code files are present
- [ ] README.md is comprehensive and accurate
- [ ] .gitignore properly excludes datasets and results
- [ ] No sensitive information (API keys, passwords) in code
- [ ] All file paths are relative, not absolute
- [ ] Documentation includes setup instructions
- [ ] License file is included
- [ ] Repository is public and accessible

## 🚨 Common Issues and Solutions

### Issue: "Repository not found"
**Solution**: Check that the remote URL is correct and you have push permissions.

### Issue: "Authentication failed"
**Solution**: Use GitHub CLI or set up SSH keys for authentication.

### Issue: "Large files detected"
**Solution**: Ensure .gitignore excludes large dataset files.

### Issue: "Merge conflicts"
**Solution**: If repository was initialized with README, you may need to merge:
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

## 📞 Support

If you encounter issues during upload:

1. Check the [GitHub Documentation](https://docs.github.com/)
2. Verify your Git configuration: `git config --list`
3. Ensure you have proper permissions for the repository
4. Check the repository size (should be < 100MB without datasets)

## 🎉 Success!

Once uploaded successfully, your repository will be available at:
`https://github.com/YOUR_USERNAME/WeightedAST-SQL-to-Text`

Users can now:
1. Clone the repository
2. Follow the README instructions
3. Set up datasets using `setup_datasets.py`
4. Run experiments using `run_experiments.py`
5. Contribute to the project

## 🔄 Future Updates

To update the repository with new changes:

```bash
# Make changes to files
git add .
git commit -m "Update: Brief description of changes"
git push origin main
```

This guide ensures a smooth upload process and a professional repository presentation for your Weighted-AST research project.
