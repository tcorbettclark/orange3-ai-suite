"""The suite's downloadable Hugging Face artifacts, keyed by name.

``ASSETS`` holds every asset; ``MODEL_ASSETS`` holds the model alone, because the
BritLLM Models widget must not list datasets. The image feature catalogue is a
separate registry, under ``kernel/images/features``.
"""

from __future__ import annotations

from .britllm import BRITLLM_4BIT
from .clouds import CCSN_CLOUDS
from .mnist import MNIST_DIGITS

ASSETS = {asset.name: asset for asset in (BRITLLM_4BIT, MNIST_DIGITS, CCSN_CLOUDS)}

# The BritLLM Models widget is a model downloader; it must not list datasets.
MODEL_ASSETS = {BRITLLM_4BIT.name: BRITLLM_4BIT}
