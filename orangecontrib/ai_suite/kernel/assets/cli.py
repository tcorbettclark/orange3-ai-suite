"""``orange-ai-assets`` / ``python -m ...assets``: the headless asset CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalogue import ASSETS, BRITLLM_4BIT
from .hf import AssetError, ensure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "list"))
    parser.add_argument("asset", nargs="?", default=BRITLLM_4BIT.name)
    parser.add_argument(
        "--cache",
        help=(
            "cache root (default: the Hugging Face hub cache for models, "
            "the datasets cache for prepared datasets)"
        ),
    )
    args = parser.parse_args(argv)

    if args.command == "list":
        for asset in ASSETS.values():
            size = f"{asset.size_bytes / 1e9:.2f} GB" if asset.size_bytes else "?"
            print(f"{asset.name}\t{asset.repo_type}\t{size}\t{asset.repo_id}")
        return 0
    asset = ASSETS.get(args.asset)
    if asset is None:
        parser.error(f"unknown asset {args.asset!r}; choose from {', '.join(ASSETS)}")

    cache = Path(args.cache) if args.cache else None
    last = [-1]

    def report(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct >= last[0] + 5 or done == total:
            last[0] = pct
            shown = f"{done / 1e9:5.2f} / {total / 1e9:.2f} GB" if total else f"{done / 1e6:.1f} MB"
            print(f"\r{pct:3d}%  {shown}", end="", file=sys.stderr)
            if total and done == total:
                print(file=sys.stderr)

    try:
        path = ensure(asset, cache, report)
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0
