"""Fetch britllm/britllm-3b-v0.1 and convert it to MLX 4-bit.

The script downloads the Hugging Face snapshot (cache-aware, so a warm
``~/.cache/huggingface`` costs no re-download) and converts it with
``mlx_lm convert --quantize``: 4-bit affine, group size 64, the MLX defaults.

The result goes to untracked ``data/models/britllm-4bit/``. The base model is
a raw base model ("should be further finetuned"), which is exactly what makes
the LoRA lessons visible. A sibling manifest records the versions, size,
parameter count and quantisation config.

    uv run python demo/fetch_britllm.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "britllm/britllm-3b-v0.1"
TARGET = Path(__file__).resolve().parent.parent / "data" / "models" / "britllm-4bit"


MINIMAL_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: "
    "{{ message['content'] }}\\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


def add_minimal_chat_template(model_dir: Path) -> bool:
    """Add a minimal chat template when the tokenizer has none.

    A raw base model's tokenizer ships no chat template, but mlx-lm's
    prompt-masking SFT path requires one. The converted copy gets a minimal
    generic role template ('user: .../assistant: ...'); weights are untouched.
    Whether the template was added is recorded in the manifest.
    """
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("chat_template"):
        return False  # already templated
    cfg["chat_template"] = MINIMAL_CHAT_TEMPLATE
    cfg_path.write_text(json.dumps(cfg, indent=1))
    return True


def main() -> None:
    if (TARGET / "model.safetensors").exists():
        # idempotent: ensure the SFT template ships even on pre-existing
        # conversions (the check below is the self-containment guarantee)
        templated = add_minimal_chat_template(TARGET)
        print(f"already converted: {TARGET} (chat template {'added' if templated else 'present'})")
        return
    t0 = time.perf_counter()
    src = snapshot_download(REPO_ID)
    print(f"snapshot ready ({time.perf_counter() - t0:.0f}s): {src}")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            str(src),
            "--mlx-path",
            str(TARGET),
            "--quantize",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"mlx_lm convert failed ({result.returncode})")
    converted_seconds = time.perf_counter() - t0
    templated = add_minimal_chat_template(TARGET)

    import struct

    from mlx_lm.utils import load

    # honest parameter count: sum the *source* fp16 shard headers, since MLX
    # quantised leaf sizes report packed storage, not parameter counts
    src_index = json.loads((Path(src) / "model.safetensors.index.json").read_text())
    param_bytes = 0
    for shard in sorted(set(src_index["weight_map"].values())):
        with open(Path(src) / shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            if name != "__metadata__":
                param_bytes += abs(meta["data_offsets"][1] - meta["data_offsets"][0])
    n_params = round(param_bytes / 2 / 1e9, 2)  # F16 = 2 bytes/param

    load(str(TARGET))
    config = json.loads((TARGET / "config.json").read_text())
    manifest = {
        "hf_repo": REPO_ID,
        "mlx_path": str(TARGET),
        "quantization": config.get("quantization"),
        "n_params_fp16_estimate_billions": n_params,
        "model_type": config.get("model_type"),
        "converted_seconds": round(converted_seconds, 1),
        "chat_template_added": templated,
        "size_mb": round(sum(p.stat().st_size for p in TARGET.rglob("*") if p.is_file()) / 1e6, 1),
    }
    (TARGET.parent / "britllm-4bit-MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    for key, value in manifest.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
