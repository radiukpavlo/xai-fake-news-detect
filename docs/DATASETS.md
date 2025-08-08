# Datasets

## ISOT (Primary)
- Source: Kaggle "Fake and Real News" (ISOT).
- Expected files under `data/isot/`:
  - `Fake.csv` (columns: title, text, possibly subject/source)
  - `True.csv` (columns: title, text, possibly subject/source)

### Placement
1. Create directory `data/isot/`.
2. Put the two CSVs there with exact names `Fake.csv` and `True.csv`.
3. No modifications; preprocessing is performed by the pipeline.

## LIAR (Secondary, optional)
- Source: ACL 2017 LIAR.
- Place TSV files under `data/liar/` as `train.tsv`, `val.tsv`, `test.tsv` with official splits.
- Binarization (assumption; configurable):
  - fake: {`pants-fire`, `false`}
  - real: {`true`, `mostly-true`}
  - drop: `half-true` (or map via a config override).

## Smoke Dataset (Bundled)
- `data/samples/smoke.csv` is included for quick CI.
- Use with `--mode smoke` on CLI to bypass external datasets.
