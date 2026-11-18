"""Hugging Face assets: the model, the prepared datasets and their caches."""

from .catalogue import ASSETS, BRITLLM_4BIT, CCSN_CLOUDS, MNIST_DIGITS, MODEL_ASSETS
from .cli import main
from .hf import (
    Asset,
    AssetError,
    Progress,
    cached_path,
    datasets_cache,
    ensure,
    hub_cache,
    verify,
)

__all__ = [
    "ASSETS",
    "BRITLLM_4BIT",
    "CCSN_CLOUDS",
    "MNIST_DIGITS",
    "MODEL_ASSETS",
    "Asset",
    "AssetError",
    "Progress",
    "cached_path",
    "datasets_cache",
    "ensure",
    "hub_cache",
    "main",
    "verify",
]
