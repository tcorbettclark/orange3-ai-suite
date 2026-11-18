"""Fetch the pinned 4-bit BritLLM model into the Hugging Face cache.

This is the headless form of the BritLLM Models widget. It downloads the exact
artifact the LoRA widgets expect, verifies its weights and prints the snapshot
directory; nothing is converted here. Run it from anywhere:

    uv run --extra lora python examples/fetch_britllm.py

The cache is the standard Hugging Face hub cache (``HF_HUB_CACHE``, else
``$HF_HOME/hub``, else ``~/.cache/huggingface/hub``). Set ``HF_HOME`` before
launching to share a cache, or pass ``--cache``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orangecontrib.ai_suite.kernel.assets import AssetError, get_britllm, hub_cache


def report(done: int, total: int) -> None:
    pct = int(done * 100 / total) if total else 0
    print(f"\r{pct:3d}%  {done / 1e9:5.2f} / {total / 1e9:.2f} GB", end="", file=sys.stderr)
    if done == total:
        print(file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", help=f"cache root (default: {hub_cache()})")
    args = parser.parse_args()
    try:
        path = get_britllm(
            Path(args.cache) if args.cache else None,
            report,
        )
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
