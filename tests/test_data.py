from __future__ import annotations

from pathlib import Path

from mmx_news.data.loaders import _read_smoke, stratified_splits


def test_smoke_load_and_split() -> None:
    arts = _read_smoke(Path("data/samples/smoke.csv"))
    assert len(arts) >= 4
    splits = stratified_splits(arts, n_splits=2, split_ratio=(0.64, 0.16, 0.20), seed_list=[13, 17])
    assert len(splits) == 2
    for seed, idx in splits.items():
        assert set(idx.keys()) == {"train_idx", "val_idx", "test_idx"}
        assert len(idx["train_idx"]) > 0
