from __future__ import annotations

from pathlib import Path


def check_or_raise_dataset(root: str | Path) -> None:
    """Verify dataset files exist; raise with instructions if missing."""
    root = Path(root)
    fake = root / "Fake.csv"
    true = root / "True.csv"
    if not fake.exists() or not true.exists():
        raise FileNotFoundError(
            f"ISOT dataset not found under {root}. Expected Fake.csv and True.csv. "
            "See docs/DATASETS.md for placement instructions."
        )
