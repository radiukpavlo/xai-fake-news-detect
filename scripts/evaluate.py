from __future__ import annotations

import argparse

from mmx_news.evaluation.validate import evaluate_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices={"val", "test"}, default="test")
    args = p.parse_args()
    evaluate_checkpoint(args.checkpoint, args.split)


if __name__ == "__main__":
    main()
