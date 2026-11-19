"""Loading, generating and scoring through mlx-lm, in this process.

:func:`load_model`, :func:`stream_text` and :func:`evaluate_text` import mlx inside the
function, so importing this module costs nothing on a machine without it, and
:func:`available` answers before any import is tried. ``stream_text`` returns the
completion without the prompt, in mlx-lm's own chunks.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

_IMPORT_ERROR = (
    "MLX not available - orange3-ai-suite[lora] requires macOS on Apple Silicon "
    "(pip install orange3-ai-suite[lora])"
)

#: ids per perplexity window; fixed so runs stay comparable.
EVAL_WINDOW = 512


def available() -> bool:
    """Whether this machine can run MLX at all, without importing it."""
    import importlib.util

    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def require() -> None:
    """Raise the user-facing error if MLX is missing; otherwise do nothing."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise ImportError(_IMPORT_ERROR) from exc


def load_model(base_model: str, adapter_path: str | None = None) -> tuple[Any, Any]:
    """The model and its tokenizer, with an adapter fitted when one is given."""
    require()
    from mlx_lm import load

    return load(base_model, adapter_path=adapter_path)


def stream_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    temperature: float,
    max_new_tokens: int,
    seed: int | None = None,
) -> Iterator[str]:
    """The completion in the chunks mlx-lm produces, without the prompt.

    ``seed`` seeds MLX's global generator, the only one mlx-lm's sampler reads, so it
    applies process-wide.
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0 (0 = greedy)")
    from mlx.core import random as mx_random
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    if seed is not None:
        mx_random.seed(int(seed))
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=int(max_new_tokens),
        sampler=make_sampler(temp=float(temperature)),
    ):
        yield response.text


def segments(ids: list[int], max_tokens: int, window: int) -> list[list[int]]:
    """Windows of at most ``window`` inputs, plus the target that follows each.

    Consecutive windows share one id: the last target of one window is the final
    input of the next, so every id after the first is scored exactly once. Only
    the first ``max_tokens`` ids are used.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    limit = min(len(ids), max_tokens)
    step = window - 1
    return [ids[start : min(start + window, limit)] for start in range(0, limit - 1, step)]


def evaluate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_tokens: int,
    window: int = EVAL_WINDOW,
) -> dict:
    """Loss and perplexity over ``text``, in fixed ``window``-token slices.

    Returns ``{'loss', 'perplexity'}``; a text of fewer than two ids scores NaN, and
    ``max_tokens`` below 2 raises.
    """
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2")
    import mlx.core as mx
    import mlx.nn as nn

    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return {"loss": float("nan"), "perplexity": float("nan")}
    nll_sum, n_tokens = 0.0, 0
    for segment in segments(ids, max_tokens, window):
        logits = model(mx.array([segment[:-1]]))
        log_probs = nn.log_softmax(logits.reshape(-1, logits.shape[-1]), axis=-1)
        picked = log_probs[mx.arange(len(segment) - 1), mx.array(segment[1:])]
        nll_sum += float(-picked.sum())
        n_tokens += len(segment) - 1
    loss = nll_sum / max(n_tokens, 1)
    return {"loss": float(loss), "perplexity": float(math.exp(loss))}
