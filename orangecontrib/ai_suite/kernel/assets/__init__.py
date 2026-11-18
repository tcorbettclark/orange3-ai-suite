"""Hugging Face assets: the model and the prepared datasets."""

from .britllm import get_britllm
from .clouds import get_clouds
from .mnist import get_mnist

__all__ = ["get_britllm", "get_clouds", "get_mnist"]
