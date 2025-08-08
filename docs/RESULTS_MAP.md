# Results Map

| Artifact | Command | Output |
|---|---|---|
| Transition fidelity (B vs ÂT) | part of training | `runs/<exp>/transition_fidelity.json` |
| Test metrics (mean±std across 7 splits) | `mmx_news.cli train` across seeds | `runs/<exp>/metrics.csv` and `metrics.jsonl` |
| Confusion matrices | `mmx_news.cli evaluate` | `runs/<exp>/confusion_<seed>.json` |
| PCA/MDS/tSNE | `mmx_news.cli evaluate` | `runs/<exp>/plots/*.png` |
| Ablations table | `mmx_news.cli ablation` | `runs/<exp>/ablations.csv` |
