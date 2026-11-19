"""The streaming convention on the MLX side: one sampling path, chunked.

``generate`` is ``stream`` joined up: the prompt echo, the sampling settings and the
seed are set in ``stream``, and the chunk boundaries are mlx-lm's, so a caller may only
concatenate them. The mlx-lm sampler is replaced and no weights are loaded.
"""

import pytest

pytest.importorskip("mlx_lm", reason="the lora extra (mlx-lm) is not installed")

from edu_ai_suite.lora import LoRAModel, mlx_runtime


@pytest.fixture
def model(monkeypatch):
    """A LoRAModel whose mlx variants are placeholders."""
    loaded = LoRAModel("base-model")
    monkeypatch.setattr(loaded, "_variant", lambda with_adapter: (object(), object()))
    return loaded


def _chunks(*pieces):
    return lambda *args, **kwargs: iter(pieces)


def test_the_prompt_is_the_first_chunk_so_it_can_be_shown_at_once(model, monkeypatch):
    monkeypatch.setattr(mlx_runtime, "stream_text", _chunks(" the", " end"))
    assert list(model.stream("Once", temperature=0.0)) == ["Once the", " end"]


def test_generate_is_the_stream_joined(model, monkeypatch):
    monkeypatch.setattr(mlx_runtime, "stream_text", _chunks(" the", " end"))
    assert model.generate("Once", temperature=0.0) == "Once the end"


def test_an_empty_completion_leaves_only_the_prompt(model, monkeypatch):
    monkeypatch.setattr(mlx_runtime, "stream_text", _chunks())
    assert list(model.stream("Once")) == ["Once"]
    assert model.generate("Once") == "Once"


def test_the_chunks_come_straight_from_the_sampler(model, monkeypatch):
    # Whatever mlx-lm chunks like is what a caller gets; the echo is decided
    # once, against the first chunk, and then passed through.
    # "p" ends in a letter and "a" starts with one, so the echo inserts the one
    # space mlx-lm does not promise; the rest is passed through untouched.
    monkeypatch.setattr(mlx_runtime, "stream_text", _chunks("a", "b", "c"))
    assert list(model.stream("p")) == ["p a", "b", "c"]


def test_the_stream_carries_the_sampling_settings(model, monkeypatch):
    seen = {}

    def fake(_model, _tokenizer, prompt, *, temperature, max_new_tokens, seed=None):
        seen.update(
            prompt=prompt, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed
        )
        return iter(["x"])

    monkeypatch.setattr(mlx_runtime, "stream_text", fake)
    list(model.stream("p", 0.5, 7, 3))
    assert seen == {"prompt": "p", "temperature": 0.5, "max_new_tokens": 7, "seed": 3}


def test_a_negative_temperature_is_refused_before_anything_is_loaded(model):
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        list(model.stream("p", temperature=-0.5))
