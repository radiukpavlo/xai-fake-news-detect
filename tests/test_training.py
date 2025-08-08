from __future__ import annotations

from mmx_news.training.train import run_training
from mmx_news.training.utils import load_config


def test_smoke_training_runs() -> None:
    cfg = load_config("configs/default.yaml")
    run_training(cfg, seed=13, mode="smoke")
