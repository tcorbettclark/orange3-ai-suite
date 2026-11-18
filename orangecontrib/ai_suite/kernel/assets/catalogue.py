"""The suite's Hugging Face assets: one model and two prepared datasets."""

from __future__ import annotations

from .hf import Asset
from .prepare import prepare_ccsn, prepare_mnist

# The 4-bit MLX BritLLM the LoRA tutorials start from. The SHA-256 and size are
# of the converted directory in this repository; pin the revision to the upload's
# commit SHA once the model is on the Hub (examples/build_britllm_4bit.py prints
# the exact constant).
BRITLLM_4BIT = Asset(
    name="britllm-3b-v0.1-4bit-mlx",
    repo_id="tcorbettclark/britllm-3b-v0.1-4bit-mlx",
    repo_type="model",
    revision=None,
    patterns=("*.json", "*.safetensors", "*.model"),
    sha256="7b67b9b99848cfc7be1d6ce5f60ef0240501e27e453dcda9ec5e02416fe608e8",
    size_bytes=1_504_298_599,
    weights="model.safetensors",
)

# Canonical MNIST, MIT, 60k/10k parquet. prepare_mnist writes the 500/100
# images per digit at the source 28x28 resolution.
MNIST_DIGITS = Asset(
    name="mnist-digits",
    repo_id="ylecun/mnist",
    repo_type="dataset",
    revision="77f3279092a1c1579b2250db8eafed0ad422088c",
    size_bytes=18_157_506,
    required=("train", "val", "README.md"),
    prepare=prepare_mnist,
)

# CCSN (Zhang et al. 2018), CC0 1.0 at Harvard Dataverse, MIT mirror on the Hub.
# prepare_ccsn keeps the five genera the cloud tutorial uses.
CCSN_CLOUDS = Asset(
    name="ccsn-clouds",
    repo_id="aduuuuuu/CCSN",
    repo_type="dataset",
    revision="cc195cf92fbb4b668b7db0611db8c1015c7f5cd4",
    size_bytes=103_773_179,
    required=("ac", "sc", "cu", "cb", "ci", "CREDITS.md"),
    prepare=prepare_ccsn,
)

ASSETS: dict[str, Asset] = {
    asset.name: asset for asset in (BRITLLM_4BIT, MNIST_DIGITS, CCSN_CLOUDS)
}

# The BritLLM Models widget is a model downloader; it must not list datasets.
MODEL_ASSETS: dict[str, Asset] = {BRITLLM_4BIT.name: BRITLLM_4BIT}
