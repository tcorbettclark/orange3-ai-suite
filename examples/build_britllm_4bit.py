"""Builds (and optionally publishes) the pinned 4-bit MLX BritLLM artifact; a maintainer script.

``uv run --extra lora python examples/build_britllm_4bit.py`` writes a 4-bit affine
(group 64) MLX copy of the fp16 source, with the minimal chat template the masked SFT
path needs, plus a manifest, a model card and the ``Asset`` block for
``kernel/assets/britllm.py``. ``--upload`` publishes that copy to
``tcorbettclark/britllm-3b-v0.1-4bit-mlx`` (``--repo-id`` overrides) and prints the
commit SHA to pin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOURCE_REPO = "britllm/britllm-3b-v0.1"
DEFAULT_REPO = "tcorbettclark/britllm-3b-v0.1-4bit-mlx"
MINIMAL_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: "
    "{{ message['content'] }}\\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


def add_minimal_chat_template(model_dir: Path) -> bool:
    """Add a minimal chat template to ``tokenizer_config.json``, returning whether it added one.

    True means the file gained the template and False that it already had one; weights
    are untouched. A raw base model's tokenizer ships none, and mlx-lm's prompt-masking
    SFT path fails without one.
    """
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("chat_template"):
        return False
    cfg["chat_template"] = MINIMAL_CHAT_TEMPLATE
    cfg_path.write_text(json.dumps(cfg, indent=1))
    return True


def parameter_count(source: Path) -> float:
    """Billions of parameters, counted from the source fp16 shard headers.

    ``source`` is the fp16 snapshot, not the converted directory: quantised leaves report
    packed storage rather than parameter counts.
    """
    index = json.loads((source / "model.safetensors.index.json").read_text())
    param_bytes = 0
    for shard in sorted(set(index["weight_map"].values())):
        with open(source / shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            if name != "__metadata__":
                param_bytes += abs(meta["data_offsets"][1] - meta["data_offsets"][0])
    return round(param_bytes / 2 / 1e9, 2)  # F16 = 2 bytes/param


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_model_card(target: Path, manifest: dict) -> None:
    q = manifest["quantization"]
    source = manifest["source_repo"]
    weights_gb = manifest["weights_bytes"] / 1e9
    (target / "README.md").write_text(
        f"""---
license: odc-by
base_model: {manifest["source_repo"]}
library_name: mlx
tags: [mlx, 4-bit, lora, text-generation]
---

# BritLLM 3B v0.1 - MLX 4-bit (unofficial conversion)

A quantized derivative of [{source}](https://huggingface.co/{source})
for Apple Silicon MLX, converted from source revision
`{manifest["source_revision"]}`.
BritLLM is licensed **ODC-BY**; this derivative keeps that licence and attributes
the original authors. See the source model card for the original terms.

## Changes

- affine quantization to **{q["bits"]}-bit**, group size
  **{q["group_size"]}**, mode **{q["mode"]}**
  (`mlx_lm convert --quantize` defaults);
- a minimal chat template was added to `tokenizer_config.json` for mlx-lm's
  prompt-masking SFT path. No other weight changes were made.

## Files

- `model.safetensors` - {weights_gb:.2f} GB, SHA-256
  `{manifest["weights_sha256"]}`
- `n_params_fp16_estimate_billions`: {manifest["n_params_fp16_estimate_billions"]}
- `manifest.json` - the build record (versions, source revision, hashes)

## Use

The Orange AI Suite LoRA widgets take this directory as their base model; the
BritLLM Models widget downloads it from this repository into the Hugging Face
cache. The same directory also works directly with `mlx_lm`:

```bash
python -m mlx_lm lora --model <this-snapshot> --train --data <dir> ...
```

## Provenance

Built with `mlx-lm {manifest["mlx_lm_version"]}` and `mlx {manifest["mlx_version"]}`
on {manifest["built_at"]}. The source snapshot was
`{manifest["source_repo"]}@{manifest["source_revision"]}`.
"""
    )


def build(source_repo: str, target: Path, rebuild: bool) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    source_revision = api.model_info(source_repo).sha
    if target.exists() and rebuild:
        shutil.rmtree(target)
    previous = {}
    candidates = (target / "manifest.json",)
    for manifest_path in candidates:
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())
            break
    n_params = 0.0
    if (target / "model.safetensors").exists():
        templated = add_minimal_chat_template(target)
        state = "added" if templated else "present"
        print(f"reusing converted model at {target} (chat template {state})")
    else:
        t0 = time.perf_counter()
        print("downloading the fp16 source (cached by Hugging Face) ...")
        source = Path(snapshot_download(source_repo))
        print(f"source snapshot ready ({time.perf_counter() - t0:.0f}s): {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        print("converting to 4-bit MLX (this can take a few minutes) ...", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                str(source),
                "--mlx-path",
                str(target),
                "--quantize",
            ]
        )
        if result.returncode != 0:
            raise SystemExit(f"mlx_lm convert failed ({result.returncode})")
        print(f"converted in {time.perf_counter() - t0:.0f}s")
        templated = add_minimal_chat_template(target)
        n_params = parameter_count(source)
    n_params = previous.get("n_params_fp16_estimate_billions") or n_params

    print("verifying the converted model loads ...", flush=True)
    from mlx_lm.utils import load

    load(str(target))  # the converted directory must load as an MLX model

    print("hashing model.safetensors ...", flush=True)
    config = json.loads((target / "config.json").read_text())
    weights = target / "model.safetensors"
    manifest = {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "quantization": config.get("quantization"),
        "n_params_fp16_estimate_billions": n_params,
        "chat_template_added": templated,
        "weights_sha256": sha256(weights),
        "weights_bytes": weights.stat().st_size,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "mlx_version": importlib.metadata.version("mlx"),
        "built_at": time.strftime("%Y-%m-%d"),
    }
    write_model_card(target, manifest)
    manifest["size_bytes"] = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def asset_block(repo_id: str, revision: str | None, manifest: dict) -> str:
    revision_text = "None" if revision is None else repr(revision)
    note = "  # pin the upload's commit SHA" if revision is None else ""
    return f"""BRITLLM_4BIT = Asset(
    name="britllm-4bit",
    repo_id="{repo_id}",
    repo_type="model",
    revision={revision_text},{note}
    patterns=("*.json", "*.safetensors", "*.model"),
    sha256="{manifest["weights_sha256"]}",
    size_bytes={manifest["size_bytes"]},
    weights="model.safetensors",
)"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=SOURCE_REPO)
    parser.add_argument(
        "--target",
        type=Path,
        help="where to build (default: a new temporary directory)",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--rebuild", action="store_true", help="delete the target and convert again"
    )
    parser.add_argument(
        "--upload", action="store_true", help="create/upload the HF repo and print the pin"
    )
    parser.add_argument(
        "--xet",
        action="store_true",
        help="use the chunked Xet uploader (default: classic LFS)",
    )
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_XET"] = "0" if args.xet else "1"

    # mkdtemp creates its directory; mlx_lm convert refuses an existing --mlx-path,
    # so the build target is a child that does not exist yet.
    target = args.target or Path(tempfile.mkdtemp(prefix="britllm-4bit-")) / "model"
    print(f"building in {target}")
    manifest = build(args.source, target, args.rebuild)
    print(f"\nmanifest: {target / 'manifest.json'}")
    print(f"weights SHA-256: {manifest['weights_sha256']}")
    print(f"size: {manifest['size_bytes'] / 1e9:.2f} GB")
    print(
        f"\nAsset for kernel/assets/britllm.py (unpinned):\n{asset_block(args.repo_id, None, manifest)}"
    )

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        print(f"\nuploading {target} -> {args.repo_id} ...", flush=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(target),
            commit_message="BritLLM 3B v0.1, MLX 4-bit (affine, group 64)",
        )
        revision = api.model_info(args.repo_id).sha
        print(f"uploaded at revision {revision}")
        print(f"\nPinned Asset:\n{asset_block(args.repo_id, revision, manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
