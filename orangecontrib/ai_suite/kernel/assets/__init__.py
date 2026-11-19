"""The suite's Hugging Face assets, each behind a ``get_*`` download function.

BritLLM as a pinned 4-bit model snapshot; CCSN clouds and MNIST digits as prepared
datasets.
"""

from .britllm import get_britllm
from .clouds import get_clouds
from .mnist import get_mnist

__all__ = ["get_britllm", "get_clouds", "get_mnist"]
