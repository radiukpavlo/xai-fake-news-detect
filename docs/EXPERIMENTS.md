# Experiments

## Core Reproduction
```bash
python -m mmx_news.cli prepare-data --config configs/default.yaml
python -m mmx_news.cli train --config configs/default.yaml --seed 13
python -m mmx_news.cli evaluate --checkpoint runs/exp1/best.joblib --split test
```

## Baselines
```bash
python -m mmx_news.cli baselines --config configs/default.yaml
```

## Ablations
```bash
python -m mmx_news.cli ablation --config configs/default.yaml --grid configs/grids/ablations.yaml
```

## External Generalization (optional; requires LIAR)
```bash
python -m mmx_news.cli xgen --config configs/default.yaml
```
