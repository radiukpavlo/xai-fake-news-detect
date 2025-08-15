from __future__ import annotations

import argparse

from .evaluation.validate import evaluate_checkpoint
from .experiments.ablations import run_ablations
from .experiments.baselines import run_baselines
from .training.train import run_training
from .data.loaders import prepare_splits
from .training.utils import load_config


from .data.loaders import prepare_splits


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmx_news", description="LLM mental-model proxy")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare-data", help="Prepare dataset and splits")
    p_prep.add_argument("--config", required=True, help="Path to the configuration file")
    p_prep.add_argument("--mode", choices={"smoke", "full"}, default="full", help="Run mode (smoke or full)")

    p_train = sub.add_parser("train", help="Run training for one seed")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--mode", choices={"smoke", "full"}, default="full")

    p_eval = sub.add_parser("evaluate", help="Evaluate checkpoint on a split")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--split", choices={"val", "test"}, default="test")
    p_eval.add_argument("--mode", choices={"smoke", "full"}, default="full")

    p_abl = sub.add_parser("ablation", help="Run ablations grid")
    p_abl.add_argument("--config", required=True)
    p_abl.add_argument("--grid", required=True)

    p_base = sub.add_parser("baselines", help="Run baseline classifiers")
    p_base.add_argument("--config", required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "prepare-data":
        cfg = load_config(args.config)
        prepare_splits(cfg, mode=args.mode)
        print("Data preparation completed.")
    elif args.cmd == "train":
        run_training(args.config, seed=args.seed, mode=args.mode)
    elif args.cmd == "evaluate":
        evaluate_checkpoint(args.checkpoint, split=args.split)
    elif args.cmd == "ablation":
        run_ablations(args.config, args.grid)
    elif args.cmd == "baselines":
        run_baselines(args.config)


if __name__ == "__main__":
    main()
