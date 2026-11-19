"""Asset download/cache/prepare tests. No network."""

import hashlib
import json
import os
import shutil
import time

import pytest

from orangecontrib.ai_suite.kernel import assets
from orangecontrib.ai_suite.kernel.assets.clouds import _prepare_clouds
from orangecontrib.ai_suite.kernel.assets.hf import (
    Asset,
    AssetError,
    _repo_folder,
    cached_path,
    ensure,
    prefer_classic_lfs,
    verify,
)
from orangecontrib.ai_suite.kernel.assets.mnist import _prepare_mnist

WEIGHTS = b"weights"


def _asset(**kwargs):
    values = dict(
        name="tiny-model",
        repo_id="org/tiny-model",
        repo_type="model",
        revision="rev1",
        patterns=("*.safetensors",),
        weights="model.safetensors",
        sha256=hashlib.sha256(WEIGHTS).hexdigest(),
        size_bytes=len(WEIGHTS),
    )
    values.update(kwargs)
    return Asset(**values)


def _cache(root, asset):
    return root / _repo_folder(asset.repo_id, asset.repo_type)


def _snapshot(root, asset):
    return _cache(root, asset) / "snapshots" / asset.revision


@pytest.fixture
def fake_download(monkeypatch):
    """Stand in for huggingface_hub.snapshot_download, with a real blob write."""
    calls = []
    state = {"bytes": WEIGHTS, "delay": 0.0}

    def fake(asset, cache_dir, local_files_only, tqdm_class=None):
        calls.append((asset, cache_dir, local_files_only))
        snapshot = _snapshot(cache_dir, asset)
        if local_files_only:
            if not (snapshot / asset.weights).is_file():
                raise FileNotFoundError("not cached")
            return snapshot
        blobs = _cache(cache_dir, asset) / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        bar = tqdm_class(total=len(state["bytes"]), unit="B") if tqdm_class else None
        with (blobs / "blob").open("wb") as f:
            f.write(state["bytes"][:3])
            f.flush()
            if bar:
                bar.update(3)
            if state["delay"]:
                time.sleep(state["delay"])
            f.write(state["bytes"][3:])
            if bar:
                bar.update(len(state["bytes"]) - 3)
                bar.close()
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / asset.weights).write_bytes(state["bytes"])
        return snapshot

    monkeypatch.setattr(assets.hf, "_snapshot_download", fake)
    return calls, state


def test_ensure_downloads_verifies_and_writes_a_marker(tmp_path, fake_download):
    asset = _asset()
    path = ensure(asset, cache_dir=tmp_path)
    assert path == _snapshot(tmp_path, asset)
    assert (path / asset.weights).read_bytes() == WEIGHTS
    assert json.loads((path / assets.hf._VERIFIED).read_text())["sha256"] == asset.sha256
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
    def missing(asset, cache_dir, local_files_only, tqdm_class=None):
        raise ImportError("No module named 'huggingface_hub'")

    monkeypatch.setattr(assets.hf, "_snapshot_download", missing)
    with pytest.raises(AssetError, match="huggingface_hub"):
        ensure(_asset(), cache_dir=tmp_path)


def test_insufficient_disk_is_reported_before_downloading(tmp_path, monkeypatch, fake_download):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: type("Usage", (), {"free": 1})())
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


def test_repo_folder_uses_the_dataset_prefix():
    assert _repo_folder("org/data", "dataset") == "datasets--org--data"
    assert _repo_folder("org/model", "model") == "models--org--model"


def test_prepared_dataset_uses_datasets_and_a_marker(tmp_path, monkeypatch):
    calls = []

    def prepare(data, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "train").mkdir()
        (dest / "train" / "0").write_text("image")
        (dest / "README.md").write_text("readme")

    def fake_load(repo_id, revision, cache_dir):
        calls.append((repo_id, revision, cache_dir))
        return {"train": []}

    monkeypatch.setattr("datasets.load_dataset", fake_load)
    asset = Asset(
        name="tiny-data",
        repo_id="org/tiny-data",
        repo_type="dataset",
        revision="rev1",
        required=("train", "README.md"),
        prepare=prepare,
    )
    seen = []
    path = ensure(
        asset, cache_dir=tmp_path, on_progress=lambda done, total: seen.append((done, total))
    )
    assert path == tmp_path / "prepared" / asset.name / asset.revision
    assert (path / "train" / "0").is_file()
    assert (path / assets.hf._PREPARED).is_file()
    assert cached_path(asset, tmp_path) == path
    assert ensure(asset, cache_dir=tmp_path) == path
    assert len(calls) == 1
    assert seen[-1] == (0, 0)


def _dataset_dict(names, images, labels):
    from datasets import ClassLabel, Dataset, DatasetDict, Image

    dataset = Dataset.from_dict({"image": images, "label": labels})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", ClassLabel(names=names))
    return DatasetDict({"train": dataset, "test": dataset})


def test_prepare_mnist_writes_the_500_100_layout(tmp_path):
    from PIL import Image

    names = [str(i) for i in range(10)]
    images = [Image.new("L", (28, 28), color=i) for i in range(10) for _ in range(2)]
    labels = [i for i in range(10) for _ in range(2)]
    _prepare_mnist(_dataset_dict(names, images, labels), tmp_path)
    assert (tmp_path / "train" / "0").is_dir()
    assert (tmp_path / "val" / "9").is_dir()
    assert len(list((tmp_path / "train").rglob("*.png"))) == 20
    assert len(list((tmp_path / "val").rglob("*.png"))) == 20
    image = Image.open(next((tmp_path / "train").rglob("*.png")))
    assert image.size == (28, 28)
    assert (tmp_path / "README.md").is_file()


def test_prepare_ccsn_keeps_the_five_genera_and_credits(tmp_path):
    from PIL import Image

    names = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
    images = [Image.new("RGB", (256, 256), color=(i, i, i)) for i in range(len(names))]
    labels = list(range(len(names)))
    _prepare_clouds(_dataset_dict(names, images, labels), tmp_path)
    for genus in ("ac", "sc", "cu", "cb", "ci"):
        assert (tmp_path / genus).is_dir()
    assert len(list(tmp_path.glob("*/*.jpg"))) == 5
    assert (tmp_path / "CREDITS.md").is_file()


def test_cli_lists_all_assets(capsys):
    from orangecontrib.ai_suite.kernel.assets.cli import main

    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "britllm-4bit" in output
    assert "mnist-digits" in output
    assert "ccsn-clouds" in output


def test_asset_getters_delegate_to_ensure(monkeypatch, tmp_path):
    from orangecontrib.ai_suite.kernel.assets import britllm, clouds, mnist

    calls = []
    for module, getter, name in (
        (britllm, britllm.get_britllm, britllm.BRITLLM_4BIT.name),
        (mnist, mnist.get_mnist, mnist.MNIST_DIGITS.name),
        (clouds, clouds.get_clouds, clouds.CCSN_CLOUDS.name),
    ):
        monkeypatch.setattr(
            module,
            "ensure",
            lambda asset, cache, progress, name=name: (
                calls.append((name, asset.name, cache, progress)) or tmp_path
            ),
        )
        assert getter(tmp_path, None) == tmp_path
    assert [call[0] for call in calls] == [
        britllm.BRITLLM_4BIT.name,
        mnist.MNIST_DIGITS.name,
        clouds.CCSN_CLOUDS.name,
    ]


def test_cli_list_reports_cache_status(tmp_path, capsys):
    from orangecontrib.ai_suite.kernel.assets.cli import main

    assert main(["list", "--cache", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "cache" in output
    assert output.count("missing") == 3


# ---- the transfer policy: owned by an entry point, never by an import ------


def test_importing_the_assets_module_decides_nothing(monkeypatch):
    # The lesson of the previous round: a helper module must not change process
    # state, so the choice is made by the entry point that owns it.
    import subprocess
    import sys

    probe = (
        "import os, sys;"
        "import orangecontrib.ai_suite.kernel.assets.hf;"
        "print(os.environ.get('HF_HUB_DISABLE_XET'), 'huggingface_hub.constants' in sys.modules)"
    )
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_XET"}
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["None", "False"]


def test_a_fresh_process_gets_classic_lfs_from_the_environment():
    # In a subprocess, because the test session may already have imported the Hub.
    import subprocess
    import sys

    probe = (
        "import os;"
        "from orangecontrib.ai_suite.kernel.assets.hf import prefer_classic_lfs;"
        "print(prefer_classic_lfs());"
        "print(os.environ['HF_HUB_DISABLE_XET'])"
    )
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_XET"}
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    # nothing imported the Hub yet, so the environment variable is enough
    assert done.stdout.splitlines() == ["classic LFS transfer", "1"]


@pytest.mark.parametrize("value", ["0", "false", "No", " OFF "])
def test_an_explicit_opt_in_is_left_alone(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", value)
    note = prefer_classic_lfs()
    assert os.environ["HF_HUB_DISABLE_XET"] == value  # not overwritten
    assert note.startswith("Xet transfer")


def test_an_already_imported_hub_gets_the_constant_set(monkeypatch):
    # Inside Orange the Hub is usually imported before a widget runs, and the
    # env var is then read by nobody: the constant is what is_xet_available()
    # consults on every download.
    import sys
    import types

    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_DISABLE_XET = False
    hub = types.ModuleType("huggingface_hub")
    hub.constants = constants
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    note = prefer_classic_lfs()
    assert constants.HF_HUB_DISABLE_XET is True
    assert note.endswith("(the Hub was already imported)")


def test_calling_it_twice_is_harmless(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    assert prefer_classic_lfs() == prefer_classic_lfs()


def test_the_cli_states_the_choice_it_made(monkeypatch, capsys):
    from orangecontrib.ai_suite.kernel.assets.cli import main

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    assert main(["list"]) == 0
    captured = capsys.readouterr()
    assert captured.err.startswith("transfer: ")
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
