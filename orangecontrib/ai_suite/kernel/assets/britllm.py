"""The pinned 4-bit MLX BritLLM model used by the LoRA widgets."""

from __future__ import annotations

from pathlib import Path

from .hf import Asset, Progress, ensure

BRITLLM_4BIT = Asset(
    name="britllm-4bit",
    repo_id="tcorbettclark/britllm-3b-v0.1-4bit-mlx",
    repo_type="model",
    revision="02241a91ab39f8296bb98e6c25afa5a98c676a4f",
    patterns=("*.json", "*.safetensors", "*.model"),
    sha256="7b67b9b99848cfc7be1d6ce5f60ef0240501e27e453dcda9ec5e02416fe608e8",
    size_bytes=1_504_298_117,
    weights="model.safetensors",
)


def get_britllm(
    cache_dir: str | Path | None = None,
    on_progress: Progress | None = None,
) -> Path:
    """Download, verify and return the pinned 4-bit MLX model snapshot."""
    return ensure(BRITLLM_4BIT, cache_dir, on_progress)
