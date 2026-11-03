"""Fetch britllm/britllm-3b-v0.1 and convert it to MLX 4-bit — self-contained.

Action item §8.1 (nn-suite-architecture.md): "Verify/produce an MLX
conversion of BritLLM". This script produces the repo's own conversion from
the Hugging Face snapshot (cache-aware: a warm ~/.cache/huggingface costs no
re-download), using `mlx_lm convert --quantize` (4-bit affine, group 64 —
the standard MLX defaults).

Writes to UNTRACKED `data/models/britllm-4bit/` (weights stay © their
licences; the model card states it is a raw base model — "should be further
finetuned", which is exactly the Stage-3 lesson). Also writes
`data/models/britllm-4bit-MANIFEST.json`: versions, sizes, parameter count,
quantisation config — the machine-annotation the §8.2 timing record needs.

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


def main() -> None:
    if (TARGET / "model.safetensors").exists():
        print(f"already converted: {TARGET} — delete to refetch")
        return
    t0 = time.perf_counter()
    src = snapshot_download(REPO_ID)
    print(f"snapshot ready ({time.perf_counter() - t0:.0f}s): {src}")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "convert",
         "--hf-path", str(src),
         "--mlx-path", str(TARGET),
         "--quantize"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"mlx_lm convert failed ({result.returncode})")
    converted_seconds = time.perf_counter() - t0

    import mlx.core as mx
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

    model, _ = load(str(TARGET))
    config = json.loads((TARGET / "config.json").read_text())
    manifest = {
        "hf_repo": REPO_ID,
        "mlx_path": str(TARGET),
        "quantization": config.get("quantization"),
        "n_params_fp16_estimate_billions": n_params,
        "model_type": config.get("model_type"),
        "converted_seconds": round(converted_seconds, 1),
        "size_mb": round(sum(p.stat().st_size for p in TARGET.rglob("*") if p.is_file())
                         / 1e6, 1),
    }
    (TARGET.parent / "britllm-4bit-MANIFEST.json").write_text(
        json.dumps(manifest, indent=1))
    for key, value in manifest.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()