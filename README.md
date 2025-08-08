# mmx-news — Mental-Model Approximation Reproduction

This repository reproduces the specification **"Explainable Fake News Detection with Large Language Models via Mental-Model Approximation"**:
deterministic data preparation, interpretable features with evidence, linear transition from LLM embeddings to interpretable space, and a transparent classifier.

## Quickstart (CPU, smoke test without internet)
```bash
# 1) Create environment (pip or conda)
python -m venv .venv && . .venv/bin/activate && python -m pip install -U pip wheel
pip install -e . -r requirements.txt

# 2) Run smoke test on bundled tiny dataset using hash-based embeddings (no internet required)
python -m mmx_news.cli prepare-data --config configs/default.yaml --mode smoke
python -m mmx_news.cli train --config configs/default.yaml --seed 13 --mode smoke
python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test --mode smoke
```

## Full run with sentence-transformers
Ensure the model **sentence-transformers/all-mpnet-base-v2** is cached locally (Hugging Face cache).
If spaCy is available, install `en_core_web_sm` once:
```bash
python -m spacy download en_core_web_sm
```
Then:
```bash
python -m mmx_news.cli prepare-data --config configs/default.yaml
python -m mmx_news.cli train --config configs/default.yaml --seed 13
python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test
```

## Datasets
- **ISOT** (primary). Place CSV files under `data/isot/` as described in `docs/DATASETS.md`.
- **LIAR** (optional, external generalization). See `docs/DATASETS.md`.

## Reproducibility
- Deterministic seeds for Python/NumPy/sklearn.
- Saved split indices under `data/processed/splits/`.
- Version/provenance recorded in `runs/<exp>/run.json`.
- Config hash yields a stable run ID.

## Commands
```bash
python -m mmx_news.cli prepare-data --config configs/default.yaml
python -m mmx_news.cli train --config configs/default.yaml --seed 13
python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test

# Ablations (examples)
python -m mmx_news.cli ablation --config configs/default.yaml --grid configs/grids/ablations.yaml
python -m mmx_news.cli baselines --config configs/default.yaml
```

See `docs/EXPERIMENTS.md` and `docs/RESULTS_MAP.md` for full study orchestration.
