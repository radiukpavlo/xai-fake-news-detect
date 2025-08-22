# Datasets

## Automated Download

Both the **ISOT** and **LIAR** datasets are now automatically downloaded from Kaggle if they are not found locally. When you run the `prepare-data` command, the script will check for the existence of the datasets and download them if necessary.

**Note:** This requires the Kaggle API to be installed and configured on your system.
1. Install the Kaggle API: `pip install kaggle`
2. Obtain an API token from your Kaggle account page.
3. Place the `kaggle.json` file in the `~/.kaggle/` directory.

## ISOT (Primary)
- Source: Kaggle "Fake and Real News" (ISOT).
- Downloaded to: `data/isot/`
- Expected files: `Fake.csv`, `True.csv`

## LIAR (Secondary, optional)
- Source: ACL 2017 LIAR.
- Downloaded to: `data/liar/`
- Expected files: `train.tsv`, `valid.tsv`, `test.tsv`

## Smoke Dataset (Bundled)
- `data/samples/smoke.csv` is included for quick CI.
- Use with `--mode smoke` on CLI to bypass external datasets.
