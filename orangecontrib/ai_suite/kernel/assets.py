"""Downloaded assets and their cache: a pinned source, one local directory.

An :class:`Asset` records the exact bytes the suite expects (repository,
revision, weights SHA-256, total size). :func:`ensure` returns a local directory,
downloading through the Hugging Face cache first when needed, and verifies the
weights before returning them.

The cache is Hugging Face's own: ``HF_HUB_CACHE``, else ``$HF_HOME/hub``, else
``~/.cache/huggingface/hub``. Sharing that cache with any other HF consumer is
intended; resume, offline mode and the ``models--org--name`` layout come from
``huggingface_hub``.

Transfers use the classic Hugging Face LFS path by default. It leaves a
resumable ``.incomplete`` file under ``blobs``, which the progress poller reads,
and it has been more reliable than Xet for the 1.4 GB artifact. Set
``HF_HUB_DISABLE_XET=0`` before importing to opt into Xet.

Progress is read from the cache's ``blobs`` directory rather than from a library
progress bar: the downloader writes network bytes there, and the asset's pinned
size is the denominator. ``on_progress`` is called from a poller thread while the
download runs, so a caller with a GUI must marshal it to its own thread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Progress = Callable[[int, int], None]

_WEIGHTS = "model.safetensors"
_PATTERNS = ("*.json", "*.safetensors", "*.model")
_VERIFIED = ".orange3-verified.json"
_POLL_SECONDS = 0.25

# Classic LFS by default: reliable progress and a resumable .incomplete file.
# ``setdefault`` lets HF_HUB_DISABLE_XET=0 opt in; if HF imported first, its
# constant is already fixed, so patch it (file_download reads it at call time).
_DISABLE_XET = os.environ.setdefault("HF_HUB_DISABLE_XET", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
if _DISABLE_XET and "huggingface_hub.constants" in sys.modules:
    import huggingface_hub.constants as _hf_constants

    _hf_constants.HF_HUB_DISABLE_XET = True


class AssetError(RuntimeError):
    """An asset is missing, incomplete, corrupt or cannot be downloaded."""


@dataclass(frozen=True)
class Asset:
    """One downloadable artifact, pinned to its exact bytes."""

    name: str
    repo_id: str
    revision: str | None  # a commit SHA once published; None = the default branch
    sha256: str
    size_bytes: int
    weights: str = _WEIGHTS
    patterns: tuple[str, ...] = _PATTERNS


# The 4-bit MLX BritLLM the LoRA tutorials start from. The SHA-256 and size are
# of the converted directory in this repository; pin the revision to the upload's
# commit SHA once the model is on the Hub (examples/build_britllm_4bit.py prints
# the exact constant).
BRITLLM_4BIT = Asset(
    name="britllm-3b-v0.1-4bit-mlx",
    repo_id="tcorbettclark/britllm-3b-v0.1-4bit-mlx",
    revision=None,
    sha256="7b67b9b99848cfc7be1d6ce5f60ef0240501e27e453dcda9ec5e02416fe608e8",
    size_bytes=1_504_298_599,
)

ASSETS: dict[str, Asset] = {BRITLLM_4BIT.name: BRITLLM_4BIT}


def hub_cache() -> Path:
    """The Hugging Face hub cache root, following HF's environment convention."""
    if root := os.environ.get("HF_HUB_CACHE"):
        return Path(root)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_folder(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def _snapshot_download(asset: Asset, cache_dir: Path, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=asset.repo_id,
            revision=asset.revision,
            cache_dir=str(cache_dir),
            allow_patterns=list(asset.patterns),
            local_files_only=local_files_only,
        )
    )


def cached_path(asset: Asset, cache_dir: str | Path | None = None) -> Path | None:
    """The local snapshot directory if the asset is already on disk, else None.

    Never downloads. With a pinned revision this is an offline check; with the
    default branch it may resolve ``refs/main`` from the local cache.
    """
    root = Path(cache_dir) if cache_dir is not None else hub_cache()
    try:
        path = _snapshot_download(asset, root, local_files_only=True)
    except Exception:
        return None
    return path if (path / asset.weights).is_file() else None


def _verified(asset: Asset, path: Path) -> bool:
    marker = path / _VERIFIED
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("sha256") == asset.sha256


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(asset: Asset, path: str | Path) -> Path:
    """Check the weights against the pinned SHA-256 and record the verification."""
    path = Path(path)
    weights = path / asset.weights
    if not weights.is_file():
        raise AssetError(f"{asset.name} has no {asset.weights} in {path}")
    digest = _sha256(weights)
    if digest != asset.sha256:
        raise AssetError(
            f"{asset.name}: {asset.weights} has SHA-256 {digest}, "
            f"expected {asset.sha256} - delete {path} and download again"
        )
    (path / _VERIFIED).write_text(
        json.dumps({"sha256": digest, "size": weights.stat().st_size}, indent=1)
    )
    return path


def _blob_bytes(root: Path, asset: Asset) -> int:
    blobs = root / _repo_folder(asset.repo_id) / "blobs"
    if not blobs.is_dir():
        return 0
    return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file())


class _Poller(threading.Thread):
    """Report cache growth while a classic LFS download runs.

    The classic path writes both completed blobs and ``.incomplete`` files into
    ``blobs``; summing that directory is the byte counter. Xet would stage
    chunks elsewhere, which is why Xet is off by default.
    """

    def __init__(self, root: Path, asset: Asset, on_progress: Progress | None):
        super().__init__(daemon=True)
        self.root, self.asset, self.on_progress = root, asset, on_progress
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.wait(_POLL_SECONDS):
            if self.on_progress is None:
                continue
            done = min(_blob_bytes(self.root, self.asset), self.asset.size_bytes)
            self.on_progress(done, self.asset.size_bytes)

    def finish(self, ok: bool) -> None:
        self.stop.set()
        self.join(timeout=2)
        if ok and self.on_progress is not None:
            self.on_progress(self.asset.size_bytes, self.asset.size_bytes)


def _check_space(root: Path, size_bytes: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < size_bytes * 1.1:
        raise AssetError(
            f"not enough free space in {root}: need about "
            f"{size_bytes / 1e9:.1f} GB, have {free / 1e9:.1f} GB"
        )


def ensure(
    asset: Asset,
    cache_dir: str | Path | None = None,
    on_progress: Progress | None = None,
) -> Path:
    """Return the verified local directory for ``asset``, downloading if needed."""
    root = Path(cache_dir) if cache_dir is not None else hub_cache()
    path = cached_path(asset, root)
    if path is not None:
        if not _verified(asset, path):
            verify(asset, path)
        if on_progress is not None:
            on_progress(asset.size_bytes, asset.size_bytes)
        return path

    _check_space(root, asset.size_bytes)
    poller = _Poller(root, asset, on_progress)
    poller.start()
    try:
        path = _snapshot_download(asset, root, local_files_only=False)
    except ImportError as exc:  # huggingface_hub absent
        poller.finish(False)
        raise AssetError(
            "downloading needs huggingface_hub: install orange3-ai-suite[lora]"
        ) from exc
    except Exception as exc:
        poller.finish(False)
        raise AssetError(
            f"could not download {asset.name} into {root}: {exc}"
        ) from exc
    poller.finish(True)
    return verify(asset, path)


def main(argv: list[str] | None = None) -> int:
    """``python -m ...assets`` / ``orange-ai-assets``: fetch headlessly."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("fetch", "path", "verify", "list"))
    parser.add_argument("asset", nargs="?", default=BRITLLM_4BIT.name)
    parser.add_argument("--cache", help="cache root (default: the Hugging Face hub cache)")
    args = parser.parse_args(argv)

    if args.command == "list":
        for asset in ASSETS.values():
            print(f"{asset.name}\t{asset.size_bytes / 1e9:.1f} GB\t{asset.repo_id}")
        return 0
    asset = ASSETS.get(args.asset)
    if asset is None:
        parser.error(f"unknown asset {args.asset!r}; choose from {', '.join(ASSETS)}")

    cache = Path(args.cache) if args.cache else None
    if args.command == "path":
        path = cached_path(asset, cache)
        if path is None:
            print(f"{asset.name} is not cached", file=sys.stderr)
            return 1
        print(path)
        return 0
    if args.command == "verify":
        path = cached_path(asset, cache)
        if path is None:
            print(f"{asset.name} is not cached", file=sys.stderr)
            return 1
        verify(asset, path)
        print(f"{asset.name} verified at {path}")
        return 0

    last = [-1]

    def report(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct >= last[0] + 5 or done == total:
            last[0] = pct
            print(f"\r{pct:3d}%  {done / 1e9:5.2f} / {total / 1e9:.2f} GB", end="", file=sys.stderr)
            if done == total:
                print(file=sys.stderr)

    try:
        path = ensure(asset, cache, report)
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
