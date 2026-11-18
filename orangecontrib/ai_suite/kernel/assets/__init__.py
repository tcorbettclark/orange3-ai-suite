"""Hugging Face assets: the model, the prepared datasets and their caches."""

from .britllm import BRITLLM_4BIT, get_britllm
from .catalogue import ASSETS, MODEL_ASSETS
from .cli import main
from .clouds import CCSN_CLOUDS, get_clouds
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
from .mnist import MNIST_DIGITS, get_mnist

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
    "get_britllm",
    "get_clouds",
    "get_mnist",
    "hub_cache",
    "main",
    "verify",
]
