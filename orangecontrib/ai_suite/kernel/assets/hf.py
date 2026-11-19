"""Hugging Face assets and their caches.

Every asset is a Hub repository: models are fetched as snapshots, datasets are
loaded with ``datasets`` and reduced to the folder layout the widgets expect. This
module names the assets, verifies them and returns the usable directory.

The Hub cache is ``HF_HUB_CACHE``, else ``$HF_HOME/hub``, else
``~/.cache/huggingface/hub``; prepared datasets live under the ``datasets`` cache's
``prepared`` subdirectory. Transfers take the classic LFS path, which
:func:`prefer_classic_lfs` sets once per process.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Progress = Callable[[int, int], None]
Prepare = Callable[[Any, Path], None]
RepoType = Literal["model", "dataset"]

_WEIGHTS = "model.safetensors"
_VERIFIED = ".orange3-verified.json"
_PREPARED = ".orange3-prepared.json"


#: The Hub's own switch. Read once, when ``huggingface_hub.constants`` is
#: imported, so setting it later only works through the constant itself.
_XET_ENV = "HF_HUB_DISABLE_XET"
_OPT_OUT = {"0", "false", "no", "off"}


def prefer_classic_lfs() -> str:
    """Set the Hub's switch to classic LFS transfer and return a one-line note.

    Call it at a process entry point: ``HF_HUB_DISABLE_XET`` is read into
    ``huggingface_hub.constants`` when that module is imported, so a process whose Hub
    is already imported needs the constant set too, which this does.

    An explicit ``HF_HUB_DISABLE_XET=0`` (or ``false``, ``no``, ``off``) opts back into
    Xet and is left alone. Idempotent: calling it twice is harmless.
    """
    if os.environ.get(_XET_ENV, "").strip().lower() in _OPT_OUT:
        return f"Xet transfer (as asked by {_XET_ENV}=0)"
    os.environ[_XET_ENV] = "1"
    if "huggingface_hub.constants" not in sys.modules:
        return "classic LFS transfer"
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_DISABLE_XET = True
    return "classic LFS transfer (the Hub was already imported)"


class AssetError(RuntimeError):
    """An asset is missing, incomplete, corrupt or cannot be downloaded."""


@dataclass(frozen=True)
class Asset:
    """One Hugging Face repository the suite downloads.

    A model asset is a snapshot, verified against the SHA-256 of its ``weights`` file
    when one is given. A dataset asset with ``prepare`` is loaded through ``datasets``
    and reduced into a folder tree, which is the usable asset.
    """

    name: str
    repo_id: str
    repo_type: RepoType = "model"
    revision: str | None = None  # a commit SHA once published; None = default branch
    patterns: tuple[str, ...] | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    weights: str | None = None
    required: tuple[str, ...] = ()
    prepare: Prepare | None = None


def hub_cache() -> Path:
    """The Hugging Face hub cache root, following HF's environment convention."""
    if root := os.environ.get("HF_HUB_CACHE"):
        return Path(root)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def datasets_cache() -> Path:
    """The Hugging Face datasets cache root, following HF's convention."""
    if root := os.environ.get("HF_DATASETS_CACHE"):
        return Path(root)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "datasets"
    return Path.home() / ".cache" / "huggingface" / "datasets"


def _root(asset: Asset, cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return datasets_cache() if asset.prepare is not None else hub_cache()


def _repo_folder(repo_id: str, repo_type: RepoType) -> str:
    prefix = "datasets--" if repo_type == "dataset" else "models--"
    return f"{prefix}{repo_id.replace('/', '--')}"


def _snapshot_download(
    asset: Asset,
    cache_dir: Path,
    local_files_only: bool,
    tqdm_class: type | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": asset.repo_id,
        "repo_type": asset.repo_type,
        "revision": asset.revision,
        "cache_dir": str(cache_dir),
        "allow_patterns": list(asset.patterns) if asset.patterns else None,
        "local_files_only": local_files_only,
    }
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class
    return Path(snapshot_download(**kwargs))


def _prepared_path(asset: Asset, root: Path) -> Path:
    revision = asset.revision or "main"
    return root / "prepared" / asset.name / revision


def _prepared(asset: Asset, path: Path) -> bool:
    marker = path / _PREPARED
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("repo_id") == asset.repo_id and record.get("revision") == asset.revision


def _write_prepared(asset: Asset, path: Path) -> None:
    record = {
        "repo_id": asset.repo_id,
        "revision": asset.revision,
        "required": list(asset.required),
    }
    (path / _PREPARED).write_text(json.dumps(record, indent=1))


def _required_ok(asset: Asset, path: Path) -> bool:
    return all((path / name).exists() for name in asset.required)


def cached_path(asset: Asset, cache_dir: str | Path | None = None) -> Path | None:
    """The usable local path if the asset is already on disk, else None.

    Never downloads. A prepared dataset is checked by its prepared marker and
    required paths; a model/dataset snapshot is checked by its files. With a
    pinned revision this is an offline check; with the default branch it may
    resolve ``refs/main`` from the local cache.
    """
    root = _root(asset, cache_dir)
    if asset.prepare is not None:
        path = _prepared_path(asset, root)
        return path if _prepared(asset, path) and _required_ok(asset, path) else None
    try:
        path = _snapshot_download(asset, root, local_files_only=True)
    except Exception:
        return None
    if asset.weights and not (path / asset.weights).is_file():
        return None
    return path if _required_ok(asset, path) else None


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
    """Check the usable tree, record the check in it, and return the path.

    A prepared dataset has no single file to hash; its marker and required folders are
    the check. A model with a pinned SHA-256 is hashed and the digest recorded in the
    snapshot.
    """
    path = Path(path)
    if not _required_ok(asset, path):
        missing = [name for name in asset.required if not (path / name).exists()]
        raise AssetError(f"{asset.name} is missing {', '.join(missing)} in {path}")
    if asset.prepare is not None:
        if not _prepared(asset, path):
            raise AssetError(f"{asset.name} is not prepared in {path}; run ensure first")
        return path
    if asset.weights is not None:
        weights = path / asset.weights
        if not weights.is_file():
            raise AssetError(f"{asset.name} has no {asset.weights} in {path}")
        if asset.sha256 is not None:
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


def _check_space(root: Path, size_bytes: int | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if size_bytes is None:
        return
    free = shutil.disk_usage(root).free
    if free < size_bytes * 1.1:
        raise AssetError(
            f"not enough free space in {root}: need about "
            f"{size_bytes / 1e9:.1f} GB, have {free / 1e9:.1f} GB"
        )


def _progress_class(on_progress: Progress | None) -> type | None:
    """A tqdm subclass that forwards the Hub's download bars, or None without a callback."""
    if on_progress is None:
        return None
    from tqdm.auto import tqdm

    class _Progress(tqdm):
        def update(self, n: int = 1) -> bool | None:
            result = super().update(n)
            if self.total:
                on_progress(min(int(self.n), int(self.total)), int(self.total))
            return result

    return _Progress


def _ensure_prepared(asset: Asset, root: Path, on_progress: Progress | None) -> Path:
    from datasets import load_dataset

    dest = _prepared_path(asset, root)
    _check_space(root, asset.size_bytes)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = load_dataset(
            asset.repo_id,
            revision=asset.revision,
            cache_dir=str(root),
        )
    except Exception as exc:
        raise AssetError(f"could not load {asset.name} from {asset.repo_id}: {exc}") from exc
    try:
        assert asset.prepare is not None
        asset.prepare(data, dest)
    except Exception as exc:
        raise AssetError(f"could not prepare {asset.name}: {exc}") from exc
    _write_prepared(asset, dest)
    total = asset.size_bytes or 0
    if on_progress is not None:
        on_progress(total, total)
    return verify(asset, dest)


def ensure(
    asset: Asset,
    cache_dir: str | Path | None = None,
    on_progress: Progress | None = None,
) -> Path:
    """Return the usable local path for ``asset``, downloading/preparing if needed."""
    root = _root(asset, cache_dir)
    path = cached_path(asset, root)
    if path is not None:
        if asset.prepare is None and not _verified(asset, path):
            verify(asset, path)
        if on_progress is not None:
            total = asset.size_bytes or 0
            on_progress(total, total)
        return path

    if asset.prepare is not None:
        return _ensure_prepared(asset, root, on_progress)

    _check_space(root, asset.size_bytes)
    try:
        path = _snapshot_download(asset, root, False, _progress_class(on_progress))
    except ImportError as exc:  # huggingface_hub absent
        raise AssetError(
            "downloading needs huggingface_hub: install orange3-ai-suite[lora]"
        ) from exc
    except Exception as exc:
        raise AssetError(f"could not download {asset.name} into {root}: {exc}") from exc
    if on_progress is not None:
        total = asset.size_bytes or 0
        on_progress(total, total)
    return verify(asset, path)
