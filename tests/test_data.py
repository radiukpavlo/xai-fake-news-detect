from __future__ import annotations

import tempfile
from pathlib import Path

from mmx_news.data.loaders import (
    Article,
    _read_isot,
    _read_smoke,
    group_near_duplicates,
    hamming,
    simhash,
    stratified_splits,
)
from mmx_news.utils.constants import LABEL_MAP


def test_smoke_load_and_split() -> None:
    arts = _read_smoke(Path("data/samples/smoke.csv"))
    assert len(arts) >= 4
    splits = stratified_splits(arts, n_splits=2, split_ratio=(0.64, 0.16, 0.20), seed_list=[13, 17])
    assert len(splits) == 2
    for seed, idx in splits.items():
        assert set(idx.keys()) == {"train_idx", "val_idx", "test_idx"}
        assert len(idx["train_idx"]) > 0


def test_read_isot():
    """Tests reading the ISOT dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "Fake.csv").write_text("title,text\nfake title,fake text")
        (tmpdir / "True.csv").write_text("title,text\ntrue title,true text")
        articles = _read_isot(tmpdir)
        assert len(articles) == 2
        assert articles[0].label == LABEL_MAP["fake"]
        assert articles[1].label == LABEL_MAP["real"]


def test_simhash_and_hamming():
    """Tests the simhash and hamming distance functions."""
    hash1 = simhash("this is a test")
    hash2 = simhash("this is a test")
    hash3 = simhash("this is another test")
    assert hamming(hash1, hash2) == 0
    assert hamming(hash1, hash3) > 0


def test_group_near_duplicates():
    """Tests the grouping of near-duplicate articles."""
    articles = [
        Article(id="1", title="", text="this is a test", label=0),
        Article(id="2", title="", text="this is a test", label=0),
        Article(id="3", title="", text="this is another test", label=0),
    ]
    groups = group_near_duplicates(articles, threshold_bits=1)
    assert len(groups) == 2
    assert sorted(groups[0]) == [0, 1]
    assert groups[1] == [2]


def test_stratified_splits():
    """Tests the stratified splits function."""
    articles = [Article(id=str(i), title="", text=f"text {i}", label=i % 2) for i in range(20)]
    splits = stratified_splits(
        articles,
        n_splits=1,
        split_ratio=(0.6, 0.2, 0.2),
        seed_list=[42],
        dedupe_enabled=False,
    )
    assert 42 in splits
    split = splits[42]
    assert "train_idx" in split
    assert "val_idx" in split
    assert "test_idx" in split
    assert abs(len(split["train_idx"]) - 12) <= 1
    assert abs(len(split["val_idx"]) - 4) <= 1
    assert abs(len(split["test_idx"]) - 4) <= 1
