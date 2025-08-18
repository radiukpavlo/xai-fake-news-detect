from __future__ import annotations

import os
import zipfile
from pathlib import Path


def download_isot_dataset(root: str | Path) -> None:
    """Check if ISOT dataset files exist, otherwise download from Kaggle."""
    root = Path(root)
    fake_csv = root / "Fake.csv"
    true_csv = root / "True.csv"

    if fake_csv.exists() and true_csv.exists():
        print("ISOT dataset already exists. Skipping download.")
        return

    print("ISOT dataset not found. Attempting to download from Kaggle...")

    dataset = "clmentbisaillon/fake-and-real-news-dataset"
    zip_path = root / f"{dataset.split('/')[1]}.zip"

    root.mkdir(parents=True, exist_ok=True)

    # Note: Requires kaggle API to be configured on the system.
    # (i.e., `~/.kaggle/kaggle.json` should exist)
    try:
        print(f"Downloading dataset '{dataset}' to '{zip_path}'...")
        # Using os.system to run the shell command
        exit_code = os.system(f"kaggle datasets download -d {dataset} -p '{root}' --unzip")
        if exit_code != 0:
            raise OSError(f"Kaggle CLI command failed with exit code {exit_code}.")

        # The kaggle tool has an --unzip flag, but if it fails or for older versions,
        # we can handle unzipping manually. Let's assume --unzip works.
        # If the files are not there after unzip, something is wrong.
        if not fake_csv.exists() or not true_csv.exists():
             # Fallback to manual unzip if --unzip didn't work as expected
            for item in os.listdir(root):
                if item.endswith(".zip"):
                    zip_path = root / item
                    break

            if zip_path.exists():
                print(f"Unzipping '{zip_path}'...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_path)
            else:
                raise FileNotFoundError("Downloaded zip file not found for unzipping.")

        if not fake_csv.exists() or not true_csv.exists():
            raise FileNotFoundError(
                f"Successfully downloaded but 'Fake.csv' or 'True.csv' not found in the archive at {root}."
            )

        print("Dataset downloaded and verified successfully.")

    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("\nPlease ensure the Kaggle API is configured correctly.")
        print("1. Install kaggle: `pip install kaggle`")
        print("2. Get an API token from your Kaggle account page (kaggle.com -> Account -> Create New API Token)")
        print("3. Place the downloaded 'kaggle.json' in the '~/.kaggle/' directory.")
        raise

def check_or_raise_dataset(root: str | Path) -> None:
    """Verify dataset files exist; if not, download them."""
    download_isot_dataset(root)
