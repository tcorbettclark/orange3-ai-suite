"""What a run leaves on disk, and which of its files counts as the adapter.

Run it from the repository root with ``uv run python examples/lora_mlx_artifact.py``.
Nothing here needs MLX, a model or a GPU: the two directories are written by hand in
the shape mlx-lm writes them, then read back with the two questions a caller asks -
is there an adapter here, and how far did the run get?
"""

import json
import tempfile
from pathlib import Path

from orangecontrib.ai_suite.kernel.lora import mlx_layout

with tempfile.TemporaryDirectory(prefix="lora-files-") as scratch:
    scratch = Path(scratch)

    # 1. The data directory. Two shapes: raw text, and prompt/completion pairs. A missing
    #    valid.jsonl means no validation at all - mlx-lm fails on an empty one, so the
    #    writer omits it rather than writing zero lines.
    text_dir = mlx_layout.write_dataset(
        scratch / "text", "text", ["Stratocumulus are the commonest cloud.", "Fog is stratus."], []
    )
    pair_dir = mlx_layout.write_dataset(
        scratch / "pairs", "pairs", [("Q: what is fog?\nA:", "Stratus at the surface.")], []
    )
    print("text rows  ", (text_dir / "train.jsonl").read_text().strip())
    print("pair rows  ", (pair_dir / "train.jsonl").read_text().strip())
    print("valid file ", sorted(p.name for p in text_dir.iterdir()))

    # 2. A run stopped part-way. mlx-lm writes the config once, at the start, and a
    #    numbered weights file every `save_every` iterations - and no final adapter,
    #    because a stop is not the end it was writing towards.
    adapter_dir = scratch / "run-1" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / mlx_layout.ADAPTER_CONFIG).write_text(json.dumps({"fine_tune_type": "lora"}))
    for iteration in (30, 60):
        (adapter_dir / f"{iteration:07d}_{mlx_layout.ADAPTER_WEIGHTS}").write_bytes(b"a checkpoint")
    print(f"\nstopped    {sorted(p.name for p in adapter_dir.iterdir())}")

    weights = mlx_layout.adapter_weights(adapter_dir)
    print(f"           adapter_weights -> {weights.name}")
    print(f"           checkpoint_iteration -> {mlx_layout.checkpoint_iteration(weights)}")

    # 3. A run that ended cleanly writes the unnumbered name, which then answers first.
    (adapter_dir / mlx_layout.ADAPTER_WEIGHTS).write_bytes(b"the final adapter")
    weights = mlx_layout.adapter_weights(adapter_dir)
    print(f"\nfinished   adapter_weights -> {weights.name}")
    print(f"           checkpoint_iteration -> {mlx_layout.checkpoint_iteration(weights)}")

    # 4. A directory with neither is not an adapter directory; the caller gets None and
    #    decides whether that is a failure or "no adapter at all".
    empty = scratch / "empty"
    empty.mkdir()
    print(f"\nempty      adapter_weights -> {mlx_layout.adapter_weights(empty)}")
