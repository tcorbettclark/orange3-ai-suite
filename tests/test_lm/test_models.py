"""Keras builder + LanguageModel tests.

These tests need the [lm] extra (Keras/TensorFlow) and run in plain Python:
no Orange, no Qt. They must also stay collectible under the Orange
interpreter, where they are skipped when Keras is absent.
"""

import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before any TF import

import pytest

pytest.importorskip("keras", reason="[lm] extra (Keras) not installed")

import numpy as np

from orangecontrib.ai_suite.kernel.lm import (
    LanguageModel,
    tokenizer_from_text,
    train_language_model,
)
from orangecontrib.ai_suite.kernel.lm.builders import ARCHITECTURES, LSTM, Bigram, MLPWindow
from orangecontrib.ai_suite.kernel.lm.models import _sample

CORPUS = "the cat sat on the mat. the cat saw the dog. a dog and a cat ran."


@pytest.fixture(scope="module")
def tokenizer():
    return tokenizer_from_text(CORPUS, "char")


@pytest.fixture(scope="module")
def mlp_model(tokenizer):
    return train_language_model(
        tokenizer,
        CORPUS,
        MLPWindow(embedding_dim=8, hidden_units=16),
        context_size=8,
        epochs=30,
        batch_size=32,
        seed=0,
    )


# ---- builders ------------------------------------------------------------


def test_forward_shape_through_take(tokenizer):
    for cls in ARCHITECTURES.values():
        architecture = cls()
        graph = architecture.build(vocab_size=tokenizer.vocab_size, context_size=8, seed=0)
        # Feeding what `take` yields is the contract every caller relies on.
        x = architecture.take(np.zeros((2, 8), dtype=np.int32))
        assert x.shape[1] == (1 if cls is Bigram else 8)
        assert tuple(graph(x).shape) == (2, tokenizer.vocab_size)


def test_bigram_parameter_count_is_the_count_table(tokenizer):
    model = Bigram().build(vocab_size=tokenizer.vocab_size, context_size=1, seed=0)
    assert model.count_params() == tokenizer.vocab_size**2  # Flatten adds none


def test_architectures_declare_only_their_own_parameters():
    # The point of one estimator per model: nothing to ignore.
    assert Bigram().get_params() == {}
    assert MLPWindow().get_params() == {"embedding_dim": 32, "hidden_units": 128}
    assert LSTM().get_params() == {"embedding_dim": 32, "units": 256, "layers": 1}
    assert list(ARCHITECTURES) == ["bigram", "mlp_window", "lstm"]


def test_architectures_are_cloneable_estimators():
    from sklearn.base import clone

    config = MLPWindow(embedding_dim=8, hidden_units=16)
    assert clone(config).get_params() == config.get_params()
    assert config.set_params(hidden_units=64).hidden_units == 64


def test_builder_error_paths(tokenizer):
    with pytest.raises(ValueError, match="vocab_size"):
        MLPWindow().build(vocab_size=2, context_size=2)
    with pytest.raises(ValueError, match="context_size"):
        MLPWindow().build(vocab_size=10, context_size=0)
    with pytest.raises(ValueError, match="layers"):
        LSTM(layers=3).build(vocab_size=10, context_size=2)
    for cls in (MLPWindow, LSTM):
        cls().build(vocab_size=10, context_size=2)  # constructs fine


# ---- training ------------------------------------------------------------


def test_training_reduces_loss_and_reports_stats(tokenizer):
    seen = []

    def hook(stats):
        seen.append(stats)

    model = train_language_model(
        tokenizer,
        CORPUS,
        MLPWindow(embedding_dim=8, hidden_units=16),
        context_size=8,
        epochs=12,
        batch_size=32,
        seed=0,
        on_epoch=hook,
    )
    assert [s["epoch"] for s in seen] == list(range(1, 13))
    assert seen[-1]["loss"] < seen[0]["loss"]  # the loss must fall
    for s in seen:
        assert s["perplexity"] == pytest.approx(math.exp(s["loss"]), rel=1e-9)
        assert s["val_loss"] > 0 and s["val_perplexity"] == pytest.approx(
            math.exp(s["val_loss"]), rel=1e-9
        )
    hp = model.hyperparameters
    assert hp["n_train_windows"] + hp["n_val_windows"] == hp["n_windows_total"]
    assert hp["n_val_windows"] > 0 and hp["n_train_windows"] > 0
    assert 0 < hp["validation_fraction"] < 1


def test_stop_requested_stops_training_cleanly(tokenizer):
    seen = []

    def hook(stats):
        seen.append(stats["epoch"])

    def stop_requested():
        return len(seen) >= 2  # the widget's Stop button analogue

    model = train_language_model(
        tokenizer,
        CORPUS,
        MLPWindow(embedding_dim=8, hidden_units=16),
        context_size=8,
        epochs=50,
        batch_size=32,
        seed=0,
        on_epoch=hook,
        stop_requested=stop_requested,
    )
    assert model is not None
    assert max(seen) <= 3 and len(seen) < 50  # training ended early


def test_train_validation_split_is_mandatory():
    with pytest.raises(ValueError, match="validation_fraction"):
        train_language_model(
            tokenizer_from_text(CORPUS, "char"), CORPUS, MLPWindow(), validation_fraction=0.0
        )
    with pytest.raises(ValueError, match="at least 2"):
        # a single sentence cannot yield a sentence-level split
        train_language_model(
            tokenizer_from_text("one sentence only.", "char"),
            "one sentence only.",
            MLPWindow(),
        )
    with pytest.raises(ValueError, match="epochs"):
        train_language_model(tokenizer_from_text(CORPUS, "char"), CORPUS, MLPWindow(), epochs=0)


# ---- sampling ------------------------------------------------------------


def test_greedy_generation_is_deterministic(mlp_model):
    a = mlp_model.generate("", temperature=0, max_new_tokens=30)
    b = mlp_model.generate("", temperature=0, max_new_tokens=30)
    assert a == b and len(a) > 0


def test_seed_determinism(mlp_model):
    a = mlp_model.generate("", temperature=1.0, max_new_tokens=30, seed=7)
    b = mlp_model.generate("", temperature=1.0, max_new_tokens=30, seed=7)
    assert a == b


def test_different_seeds_diverge_at_high_temperature(mlp_model):
    # A 30-epoch model on a tiny corpus is near-argmax, so use the low-epoch
    # distribution trick: temperature 3 keeps sampling stochastic.
    outs = {
        mlp_model.generate("", temperature=3.0, max_new_tokens=25, seed=s) for s in (1, 2, 3, 4, 5)
    }
    assert len(outs) > 1


def test_long_prompt_uses_last_window(mlp_model):
    # In-vocabulary prompt far longer than context_size 8: only its tail can
    # influence the continuation.
    long_prompt = "the cat sat " * 6
    out = mlp_model.generate(long_prompt, temperature=0, max_new_tokens=10)
    assert out.startswith(long_prompt)  # prompt echoed, generation appended


def test_word_level_generate_smoke():
    tok = tokenizer_from_text(CORPUS, "word")
    model = train_language_model(
        tok, CORPUS, Bigram(), context_size=1, epochs=8, batch_size=16, seed=0
    )
    out = model.generate("the", temperature=0, max_new_tokens=6)
    assert out.startswith("the")


def test_sample_temperature_zero_is_argmax():
    rng = np.random.default_rng(0)
    logits = np.array([1.0, 3.0, 2.0])
    assert _sample(logits, 0.0, rng) == 1
    assert _sample(logits, -1.0, rng) == 1  # any non-positive temperature = greedy


# ---- perplexity ------------------------------------------------------------


def test_perplexity_prefers_indomain_text(mlp_model):
    in_domain = mlp_model.perplexity("the cat sat on the mat. the cat saw the dog.")
    out_domain = mlp_model.perplexity("zx qv jk wp lk bf hg.")
    assert in_domain["perplexity"] > 1.0
    assert in_domain["perplexity"] < out_domain["perplexity"]


# ---- persistence ----------------------------------------------------------


def test_save_load_round_trip(mlp_model, tmp_path):
    path = mlp_model.save(tmp_path / "demo.nnlm")
    assert (path / "model.keras").exists() and (path / "meta.json").exists()
    loaded = LanguageModel.load(path)
    assert loaded.tokenizer.tokens == mlp_model.tokenizer.tokens
    assert loaded.hyperparameters == mlp_model.hyperparameters
    assert loaded.parameter_count() == mlp_model.parameter_count()
    # deterministic comparison at greedy temperature: identical generations
    prompt = "the cat "
    assert mlp_model.generate(prompt, temperature=0, max_new_tokens=20) == loaded.generate(
        prompt, temperature=0, max_new_tokens=20
    )


def test_language_model_is_serialisable_in_principle(mlp_model):
    # tokenizer json has no exotic types
    import json

    json.dumps(mlp_model.tokenizer.to_dict())


def test_stop_requested_works_without_epoch_hook(tokenizer):
    """The stop flag must not depend on an on_epoch callback being present."""
    model = train_language_model(
        tokenizer,
        CORPUS,
        MLPWindow(embedding_dim=8, hidden_units=16),
        context_size=8,
        epochs=5,
        batch_size=32,
        stop_requested=lambda: True,
    )
    assert math.isfinite(model.hyperparameters["final_loss"])
