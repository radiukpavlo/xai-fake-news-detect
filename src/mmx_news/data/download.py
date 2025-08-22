from __future__ import annotations

import os
import zipfile
from pathlib import Path

def _download_dataset(dataset: str, root: str | Path, expected_files: list[str]) -> None:
    """
    Downloads and unzips a dataset from Kaggle if it doesn't already exist.

    Args:
        dataset: The Kaggle dataset identifier (e.g., "user/dataset-name").
        root: The directory to download and extract the dataset to.
        expected_files: A list of filenames to check for to determine if the
                        dataset already exists.
    """
    root = Path(root)
    if all((root / f).exists() for f in expected_files):
        print(f"{dataset.split('/')[1]} dataset already exists. Skipping download.")
        return

    print(f"{dataset.split('/')[1]} dataset not found. Attempting to download from Kaggle...")
    root.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading dataset '{dataset}' to '{root}'...")
        exit_code = os.system(f"kaggle datasets download -d {dataset} -p '{root}' --unzip")
        if exit_code != 0:
            raise OSError(f"Kaggle CLI command failed with exit code {exit_code}.")

        # Fallback to manual unzip if --unzip didn't work as expected
        if not all((root / f).exists() for f in expected_files):
            zip_path = root / f"{dataset.split('/')[1]}.zip"
            if zip_path.exists():
                print(f"Unzipping '{zip_path}'...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_path)
            else:
                raise FileNotFoundError("Downloaded zip file not found for unzipping.")

        if not all((root / f).exists() for f in expected_files):
            raise FileNotFoundError(
                f"Successfully downloaded but expected files not found in the archive at {root}."
            )

        print("Dataset downloaded and verified successfully.")

    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("\nPlease ensure the Kaggle API is configured correctly.")
        print("1. Install kaggle: `pip install kaggle`")
        print("2. Get an API token from your Kaggle account page (kaggle.com -> Account -> Create New API Token)")
        print("3. Place the downloaded 'kaggle.json' in the '~/.kaggle/' directory.")
        raise

def download_isot_dataset(root: str | Path) -> None:
    """Check if ISOT dataset files exist, otherwise download from Kaggle."""
    _download_dataset(
        dataset="csmalarkodi/isot-fake-news-dataset",
        root=root,
        expected_files=["Fake.csv", "True.csv"],
    )

def download_liar_dataset(root: str | Path) -> None:
    """Check if LIAR dataset files exist, otherwise download from Kaggle."""
    _download_dataset(
        dataset="doanquanvietnamca/liar-dataset",
        root=root,
        expected_files=["train.tsv", "test.tsv", "valid.tsv"],
    )

def check_or_raise_dataset(isot_root: str | Path, liar_root: str | Path) -> None:
    """Verify dataset files exist; if not, download them."""
    download_isot_dataset(isot_root)
    download_liar_dataset(liar_root)
