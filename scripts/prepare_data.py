from __future__ import annotations

import argparse

from mmx_news.data.loaders import prepare_splits
from mmx_news.training.utils import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices={"smoke", "full"}, default="full")
    args = p.parse_args()
    cfg = load_config(args.config)
    prepare_splits(cfg, mode=args.mode)
    print("Prepared splits and dataset statistics under data/processed/.")


if __name__ == "__main__":
    main()
