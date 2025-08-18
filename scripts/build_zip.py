from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def build_zip(out_path: str | Path = "mmx_news_repro.zip") -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = Path(out_path)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_dir():
                continue
            # Exclude large or irrelevant paths
            rel = p.relative_to(root)
            if any(str(rel).startswith(ex) for ex in ["data/raw", "data/interim", ".git", ".venv", "runs"]):
                continue
            zf.write(p, arcname=str(rel))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="mmx_news_repro.zip")
    args = p.parse_args()
    build_zip(args.out)
