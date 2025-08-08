from __future__ import annotations

import argparse

from mmx_news.experiments.ablations import run_ablations
from mmx_news.experiments.baselines import run_baselines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--grid", default="configs/grids/ablations.yaml")
    p.add_argument("--which", choices={"baselines", "ablations"}, default="ablations")
    args = p.parse_args()
    if args.which == "baselines":
        run_baselines(args.config)
    else:
        run_ablations(args.config, args.grid)


if __name__ == "__main__":
    main()
