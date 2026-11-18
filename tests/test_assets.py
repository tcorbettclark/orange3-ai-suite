"""Asset download/cache tests. No network, no Hugging Face install required."""

import hashlib
import json
import shutil
import time

import pytest

from orangecontrib.ai_suite.kernel import assets
from orangecontrib.ai_suite.kernel.assets import Asset, AssetError, cached_path, ensure, verify

WEIGHTS = b"weights"


def _asset(**kwargs):
    values = dict(
        name="tiny-model",
        repo_id="org/tiny-model",
        revision="rev1",
        sha256=hashlib.sha256(WEIGHTS).hexdigest(),
        size_bytes=len(WEIGHTS),
    )
    values.update(kwargs)
    return Asset(**values)


def _cache(root, asset):
    return root / f"models--{asset.repo_id.replace('/', '--')}"


def _snapshot(root, asset):
    return _cache(root, asset) / "snapshots" / asset.revision


@pytest.fixture
def fake_download(monkeypatch):
    """Stand in for huggingface_hub.snapshot_download, with a real blob write."""
    calls = []
    state = {"bytes": WEIGHTS, "delay": 0.0}

    def fake(asset, cache_dir, local_files_only):
        calls.append((asset, cache_dir, local_files_only))
        snapshot = _snapshot(cache_dir, asset)
        if local_files_only:
            if not (snapshot / asset.weights).is_file():
                raise FileNotFoundError("not cached")
            return snapshot
        blobs = _cache(cache_dir, asset) / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        with (blobs / "blob").open("wb") as f:
            f.write(state["bytes"][:3])
            f.flush()
            if state["delay"]:
                time.sleep(state["delay"])
            f.write(state["bytes"][3:])
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / asset.weights).write_bytes(state["bytes"])
        return snapshot

    monkeypatch.setattr(assets, "_snapshot_download", fake)
    return calls, state


def test_ensure_downloads_verifies_and_writes_a_marker(tmp_path, fake_download):
    asset = _asset()
    path = ensure(asset, cache_dir=tmp_path)
    assert path == _snapshot(tmp_path, asset)
    assert (path / asset.weights).read_bytes() == WEIGHTS
    assert json.loads((path / assets._VERIFIED).read_text())["sha256"] == asset.sha256
    calls, _ = fake_download
    assert [c[2] for c in calls] == [True, False]  # cached check, then download


def test_ensure_reuses_a_cached_snapshot_without_downloading(tmp_path, fake_download):
    asset = _asset()
    snapshot = _snapshot(tmp_path, asset)
    snapshot.mkdir(parents=True)
    (snapshot / asset.weights).write_bytes(WEIGHTS)
    verify(asset, snapshot)
    path = ensure(asset, cache_dir=tmp_path)
    assert path == snapshot
    calls, _ = fake_download
    assert [c[2] for c in calls] == [True]


def test_progress_is_reported_while_downloading(tmp_path, fake_download):
    asset = _asset()
    _, state = fake_download
    state["delay"] = 0.6
    seen = []
    ensure(asset, cache_dir=tmp_path, on_progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (asset.size_bytes, asset.size_bytes)
    assert any(0 < done <= asset.size_bytes for done, _ in seen)


def test_a_checksum_mismatch_is_an_error(tmp_path, fake_download):
    asset = _asset()
    _, state = fake_download
    state["bytes"] = b"wrong!"
    with pytest.raises(AssetError, match="SHA-256"):
        ensure(asset, cache_dir=tmp_path)


def test_a_missing_hub_install_is_reported(tmp_path, monkeypatch):
    def missing(asset, cache_dir, local_files_only):
        raise ImportError("No module named 'huggingface_hub'")

    monkeypatch.setattr(assets, "_snapshot_download", missing)
    with pytest.raises(AssetError, match="huggingface_hub"):
        ensure(_asset(), cache_dir=tmp_path)


def test_insufficient_disk_is_reported_before_downloading(tmp_path, monkeypatch, fake_download):
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: type("Usage", (), {"free": 1})()
    )
    with pytest.raises(AssetError, match="free space"):
        ensure(_asset(), cache_dir=tmp_path)


def test_cached_path_is_offline_and_verified_by_verify(tmp_path, fake_download):
    asset = _asset()
    assert cached_path(asset, tmp_path) is None
    snapshot = _snapshot(tmp_path, asset)
    snapshot.mkdir(parents=True)
    (snapshot / asset.weights).write_bytes(WEIGHTS)
    assert cached_path(asset, tmp_path) == snapshot
    verify(asset, snapshot)
    assert verify(asset, snapshot) == snapshot
