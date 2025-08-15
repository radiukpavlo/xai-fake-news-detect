from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ..utils.constants import LABEL_MAP
from .preprocess import simple_sentence_split


@dataclass
class Article:
    id: str
    title: str
    text: str
    label: int
    source: str | None = None


def _read_isot(root: Path) -> List[Article]:
    fake = pd.read_csv(root / "Fake.csv")
    true = pd.read_csv(root / "True.csv")
    rows: List[Article] = []
    for i, r in fake.iterrows():
        title = str(r.get("title", ""))
        text = str(r.get("text", ""))
        rows.append(Article(id=f"fake_{i}", title=title, text=text, label=LABEL_MAP["fake"]))
    for i, r in true.iterrows():
        title = str(r.get("title", ""))
        text = str(r.get("text", ""))
        rows.append(Article(id=f"real_{i}", title=title, text=text, label=LABEL_MAP["real"]))
    return rows


def _read_smoke(path: Path) -> List[Article]:
    rows: List[Article] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Article(
                    id=str(r["id"]),
                    title=str(r["title"]),
                    text=str(r["text"]),
                    label=LABEL_MAP[str(r["label"]).strip().lower()],
                )
            )
    return rows


def simhash(text: str, bits: int = 64) -> int:
    """Compute SimHash of text using whitespace tokens."""
    tokens = text.lower().split()
    v = [0] * bits
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def group_near_duplicates(arts: List[Article], threshold_bits: int = 3) -> List[List[int]]:
    """Group near-duplicates via SimHash Hamming distance ≤ threshold."""
    sigs = [simhash(a.text) for a in arts]
    groups: List[List[int]] = []
    used = [False] * len(arts)
    for i in range(len(arts)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(arts)):
            if not used[j] and hamming(sigs[i], sigs[j]) <= threshold_bits:
                group.append(j)
                used[j] = True
        groups.append(group)
    return groups


def stratified_splits(
    arts: List[Article],
    n_splits: int,
    split_ratio: Tuple[float, float, float],
    seed_list: List[int],
    dedupe_enabled: bool = True,
    dedupe_threshold_bits: int = 3,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Create reproducible stratified splits (train/val/test) per seed.

    Returns a dict: seed -> { 'train_idx': np.ndarray, 'val_idx': ..., 'test_idx': ... }
    """
    assert len(seed_list) >= n_splits
    y = np.array([a.label for a in arts], dtype=int)
    indices = np.arange(len(arts))

    # Handle deduplication by grouping and ensuring each group stays within a split
    groups = [[i] for i in indices]
    if dedupe_enabled:
        groups = group_near_duplicates(arts, threshold_bits=dedupe_threshold_bits)

    # Build group labels by majority for stratification
    group_labels = np.array([int(round(np.mean([y[i] for i in g]))) for g in groups], dtype=int)
    group_ids = np.arange(len(groups))

    results: Dict[int, Dict[str, np.ndarray]] = {}
    for split_id in range(n_splits):
        seed = seed_list[split_id]

        # First split groups into train+val and test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio[2], random_state=seed)
        trainval_groups_idx, test_groups_idx = next(sss1.split(group_ids, group_labels))

        # Split train+val into train and val
        train_ratio = split_ratio[0] / (split_ratio[0] + split_ratio[1])
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed + 1)
        train_groups_idx, val_groups_idx = next(sss2.split(trainval_groups_idx, group_labels[trainval_groups_idx]))

        # Map groups back to article indices
        def flatten(idx_list: np.ndarray) -> np.ndarray:
            out: List[int] = []
            for gi in idx_list:
                out.extend(groups[int(gi)])
            return np.array(out, dtype=int)

        train_idx = flatten(trainval_groups_idx[train_groups_idx])
        val_idx = flatten(trainval_groups_idx[val_groups_idx])
        test_idx = flatten(test_groups_idx)

        results[seed] = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
    return results


def prepare_splits(cfg: Dict, mode: str = "full") -> None:
    """Prepare splits and dataset statistics, saving to data/processed."""
    if mode == "smoke":
        arts = _read_smoke(Path(cfg["data"]["smoke"]["path"]))
    else:
        arts = _read_isot(Path(cfg["data"]["root"]))
    out_dir = Path("data/processed")
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    splits = stratified_splits(
        arts,
        n_splits=int(cfg["data"]["splits"]),
        split_ratio=tuple(cfg["data"]["split_ratio"]),
        seed_list=list(cfg["data"]["seed_list"]),
        dedupe_enabled=bool(cfg["data"]["dedupe"]["enabled"]),
        dedupe_threshold_bits=int(cfg["data"]["dedupe"]["threshold_bits"]),
    )
    # Save per-seed splits as .npz
    for seed, idx in splits.items():
        np.savez(out_dir / "splits" / f"{seed}.npz", **idx)

    # Stats per split
    import pandas as pd

    rows = []
    for seed, idx in splits.items():
        for split_name in ["train_idx", "val_idx", "test_idx"]:
            ids = idx[split_name]
            subset = [arts[i] for i in ids]
            n = len(subset)
            if n == 0:
                med_tokens = 0
                med_sents = 0
                pct_fake = 0.0
            else:
                tokens = [len(a.text.split()) for a in subset]
                sents = [len(simple_sentence_split(a.text)) for a in subset]
                med_tokens = int(np.median(tokens))
                med_sents = int(np.median(sents))
                pct_fake = 100.0 * sum(a.label == LABEL_MAP["fake"] for a in subset) / n
            rows.append(
                {
                    "seed": seed,
                    "split": {"train_idx": "Train", "val_idx": "Val", "test_idx": "Test"}[split_name],
                    "n_articles": n,
                    "pct_fake": round(pct_fake, 2),
                    "median_tokens": med_tokens,
                    "median_sentences": med_sents,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "stats.csv", index=False)
