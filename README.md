<div align="center">

# Explainable Fake News Detection with Large Language Models via Mental-Model Approximation

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📋 Overview

This repository implements **"Explainable Fake News Detection with Large Language Models via Mental-Model Approximation"** with a focus on transparency and reproducibility. The framework features:

- **Deterministic data preparation** with reproducible splits
- **Interpretable features** with supporting evidence
- **Linear transition** from LLM embeddings to interpretable space
- **Transparent classifier** with explainable decision boundaries

### 🔍 Full Run with Sentence Transformers

For complete accuracy using pre-trained language models:

1. **Prerequisites**:
   - Ensure `sentence-transformers/all-mpnet-base-v2` is cached locally in your Hugging Face cache
   - Install the spaCy model:
     ```bash
     python -m spacy download en_core_web_sm
     ```

2. **Run the full pipeline**:
   ```bash
   # Process the dataset with full feature extraction
   python -m mmx_news.cli prepare-data --config configs/default.yaml

   # Train the model with deterministic seed
   python -m mmx_news.cli train --config configs/default.yaml --seed 13

   # Evaluate on test set
   python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test
   ```

## 📊 Datasets

### Enhanced Dataset Pipeline 🚀

This project now includes a comprehensive dataset pipeline that can automatically download and process fake news datasets from multiple sources:

**Supported Sources:**
- 📦 **Kaggle** - ISOT, LIAR, COVID-19 Fake News, and more
- 📚 **Mendeley Data** - Academic fake news datasets
- 🤗 **Hugging Face** - FEVER, Multi-FC datasets
- 🌐 **Direct URLs** - Any CSV/JSON dataset
- 💻 **GitHub** - FakeNewsNet, BuzzFeed datasets

### Quick Start with Dataset Pipeline

```bash
# List all available datasets
python -m mmx_news.cli dataset list-datasets

# Download single dataset with API key
python -m mmx_news.cli dataset download isot --kaggle-key "username:api_key"

# Download multiple datasets and combine
python -m mmx_news.cli dataset download-multiple \
    --datasets "isot,liar,covid19_fake" \
    --kaggle-key "username:api_key" \
    --combine --balance

# Integrate into main pipeline and train
python -m mmx_news.cli dataset integrate --prepare
python -m mmx_news.cli train --config configs/default.yaml --seed 13
```

### Available Datasets

| Dataset  | Source | Samples | Description |
|----------|--------|---------|-------------|
| **ISOT** | Kaggle | 44,898 | Primary dataset with news articles |
| **LIAR** | Kaggle | 12,800 | Short statements from PolitiFact |
| **COVID-19** | Kaggle | ~10,000 | COVID-related misinformation |
| **FakeNewsNet** | GitHub | ~23,000 | Multi-domain with social context |
| **FEVER** | HuggingFace | 185,445 | Fact extraction and verification |
| **Custom** | Any | - | Support for custom datasets |

> 📝 **Full Documentation:** See [Dataset Pipeline Guide](docs/DATASET_PIPELINE.md) for detailed instructions

## 🛠️ Environment Setup

### Using Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate mmx-news

# On Windows PowerShell
# conda activate mmx-news
```

### Using Pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -U pip wheel
pip install -e . -r requirements.txt
```

## 🔄 Reproducibility

We prioritize consistent, reproducible results through multiple mechanisms:

- **Deterministic execution**:
  - Fixed seeds for Python, NumPy, and scikit-learn
  - Environment variables for consistent threading and randomization

- **Artifact persistence**:
  - Dataset splits saved to `data/processed/splits/`
  - Model provenance tracked in `runs/<exp>/run.json`
  - Configuration hashes for stable run identifiers

- **Version control**:
  - Environment specifications captured
  - Full pipeline parameters recorded

## 💻 CLI Reference

### Dataset Pipeline Commands

```bash
# List available datasets
python -m mmx_news.cli dataset list-datasets

# Download datasets (with API keys)
python -m mmx_news.cli dataset download isot --kaggle-key "username:key"
python -m mmx_news.cli dataset download-multiple --datasets all --kaggle-key "username:key"

# Process custom dataset
python -m mmx_news.cli dataset process-custom \
    --url "https://example.com/dataset.csv" \
    --label-col "label" --text-cols "content"

# Integrate datasets
python -m mmx_news.cli dataset integrate --prepare
```

### Core Pipeline Commands

```bash
# Data preparation
python -m mmx_news.cli prepare-data --config configs/default.yaml

# Model training
python -m mmx_news.cli train --config configs/default.yaml --seed 13

# Model evaluation
python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test
```

### Experimental Commands

```bash
# Run ablation studies
python -m mmx_news.cli ablation --config configs/default.yaml --grid configs/grids/ablations.yaml

# Compare against baselines
python -m mmx_news.cli baselines --config configs/default.yaml
```

> 📚 For detailed experiment orchestration, see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md)

## 🚀 Running the Optimized Pipeline

This project has been enhanced with several optimizations to make the experimental pipeline more efficient for local execution.

### Overview of Optimizations

*   **Efficient Deduplication:** The near-duplicate detection algorithm has been optimized to run much faster on large datasets.
*   **Caching:** Intermediate results like processed data, embeddings, and features are now cached to disk. This significantly speeds up repeated runs of the pipeline, as expensive computations are not repeated. The cache is stored in the `cache/` directory.
*   **Configurable Models:** You can now easily switch between a high-performance "production" model and a smaller, faster "development" model. This is useful for quick tests and debugging.
*   **Parallelization:** The feature extraction process has been parallelized to take advantage of multi-core machines, leading to significant speedups.

### Configuration

The optimizations can be configured in the `configs/default.yaml` file:

*   **Deduplication:** You can enable or disable deduplication by setting `data.dedupe.enabled` to `true` or `false`. It is recommended to keep it enabled to get the best results.
*   **Embedding Models:** You can switch between the production and development models by setting `embeddings.mode` to `prod` or `dev`.
*   **Parallelization:** You can control the number of parallel jobs for feature extraction by setting `evaluation.n_jobs`. A value of `-1` will use all available CPU cores.

### Running the Pipeline

Here's how to run the optimized pipeline:

**1. Quick Test (Smoke Mode)**

To run a quick end-to-end test of the pipeline on a small sample of the data, use the `--mode smoke` flag with the `train` command:

```bash
python -m mmx_news.cli train --config configs/default.yaml --mode smoke
```

This will run the entire pipeline, including data preparation, training, and evaluation, on the `smoke.csv` dataset. This is a great way to verify that your environment is set up correctly.

**2. Full Run**

To run the full pipeline on the ISOT dataset, use the `train` command without the `--mode` flag:

```bash
# Make sure you have downloaded the ISOT dataset and placed it in data/isot/
python -m mmx_news.cli train --config configs/default.yaml --seed 13
```

This will run the full pipeline with the configuration specified in `configs/default.yaml`. The first run will be slow as it needs to compute and cache all the intermediate results. Subsequent runs will be much faster.

You can also run the other experiments (baselines and ablations) as described in the "Experimental Commands" section. These will also benefit from the caching and other optimizations.

## 📁 Project Structure

```
├── configs/            # Configuration files for experiments
│   └── grids/          # Parameter grids for ablation studies
├── data/               # Data directory
│   ├── isot/           # ISOT dataset files
│   ├── liar/           # LIAR dataset files (optional)
│   └── processed/      # Processed data and cached splits
├── docs/               # Documentation
├── mmx_news/           # Source code
├── runs/               # Experiment artifacts and results
└── tests/              # Test suite
```

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{mental-model-approximation,
  title={Explainable Fake News Detection with Large Language Models via Mental-Model Approximation},
  author={Author, A. and Author, B.},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.
