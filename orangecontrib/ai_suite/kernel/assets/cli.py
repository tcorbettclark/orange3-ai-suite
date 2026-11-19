"""``orange-ai-assets`` / ``python -m ...assets``: the headless asset CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalogue import ASSETS
from .hf import AssetError, cached_path, ensure, prefer_classic_lfs


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--cache",
        metavar="DIR",
        help=(
            "cache root (default: the Hugging Face hub cache for models, "
            "the datasets cache for prepared datasets)"
        ),
    )
    parser = argparse.ArgumentParser(
        prog="orange-ai-assets",
        description="Download and cache the suite's Hugging Face models and datasets.",
        epilog="""examples:
  orange-ai-assets list
  orange-ai-assets fetch mnist-digits
  orange-ai-assets fetch britllm-4bit --cache ./hf-cache""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    fetch = commands.add_parser(
        "fetch",
        parents=[common],
        help="download or reuse one asset and print its local path",
    )
    fetch.add_argument(
        "asset",
        metavar="ASSET",
        choices=tuple(ASSETS),
        help=f"asset to fetch: {', '.join(ASSETS)}",
    )
    commands.add_parser(
        "list",
        parents=[common],
        help="list assets with size and cache status",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Before anything can import the Hub: this process chooses its own transfer
    # path and prints it, rather than inheriting whatever a helper decided.
    print(f"transfer: {prefer_classic_lfs()}", file=sys.stderr)
    args = _parser().parse_args(argv)
    cache = Path(args.cache) if args.cache else None

    if args.command == "list":
        print(f"{'name':32} {'type':8} {'size':>8}  {'cache':7}  repo")
        for asset in ASSETS.values():
            size = f"{asset.size_bytes / 1e9:.2f} GB" if asset.size_bytes else "?"
            status = "cached" if cached_path(asset, cache) is not None else "missing"
            print(f"{asset.name:32} {asset.repo_type:8} {size:>8}  {status:7}  {asset.repo_id}")
        return 0
    asset = ASSETS[args.asset]
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
