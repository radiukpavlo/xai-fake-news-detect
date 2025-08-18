from __future__ import annotations

import tempfile
from pathlib import Path

from mmx_news.training.train import run_training
from mmx_news.training.utils import load_config


def test_smoke_training_runs() -> None:
    run_training("configs/default.yaml", seed=13, mode="smoke")


def test_training_produces_artifacts():
    """Tests that the training script produces the expected artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        run_training("configs/default.yaml", seed=13, mode="smoke", out_dir=out_dir)

        # Check for the existence of the expected files
        assert (out_dir / "A.npy").exists()
        assert (out_dir / "B.npy").exists()
        assert (out_dir / "B_raw.npy").exists()
        assert (out_dir / "Bhat.npy").exists()
        assert (out_dir / "split_indices.npz").exists()
        assert (out_dir / "minmax_lo.npy").exists()
        assert (out_dir / "minmax_width.npy").exists()
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "T.npy").exists()
        assert (out_dir / "best.joblib").exists()
        assert (out_dir / "val_report.txt").exists()
        assert (out_dir / "test_report.txt").exists()
        assert (out_dir / "transition_fidelity.json").exists()
        assert (out_dir / "metrics.csv").exists()
