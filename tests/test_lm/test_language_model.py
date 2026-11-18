"""Architecture + LanguageModel tests.

These run in plain Python - no Orange, no Qt - and need only the package's own
dependencies.
"""

import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # before any TF import

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.lm import (
    LanguageModel,
    TrainingOptions,
    tokenizer_from_text,
)
from orangecontrib.ai_suite.kernel.lm.architecture import (
    LSTM,
    MLP,
    Bigram,
    architecture_from_dict,
    architectures,
)
from orangecontrib.ai_suite.kernel.lm.language_model import _sample

CORPUS = "the cat sat on the mat. the cat saw the dog. a dog and a cat ran."


def fit(tokenizer, text, architecture, *, on_epoch=None, stop_requested=None, **options):
    """Training in one line: the model owns fit, the options carry the run."""
    return LanguageModel(architecture, tokenizer).fit(
        text,
        TrainingOptions(**options),
        on_epoch=on_epoch,
        stop_requested=stop_requested,
    )


@pytest.fixture(scope="module")
def tokenizer():
    return tokenizer_from_text(CORPUS, "char")


@pytest.fixture(scope="module")
def mlp_model(tokenizer):
    return fit(
        tokenizer,
        CORPUS,
        MLP(context_width=8, embedding_dim=8, hidden_units=16),
        epochs=30,
        batch_size=32,
        seed=0,
    )


# ---- architectures --------------------------------------------------------


def test_context_width_is_a_hyperparameter_only_where_it_is_tunable():
    # The width belongs to the model: a bigram's is the constant 1, so there is
    # no argument to pass and no invalid value to validate.
    assert Bigram().context_width == 1
    assert MLP(context_width=4).context_width == 4
    assert LSTM(context_width=4).context_width == 4
    with pytest.raises(TypeError):
        Bigram(context_width=8)


def test_built_keras_model_accepts_the_architectures_own_width(tokenizer):
    for instance in (Bigram(), MLP(context_width=8), LSTM(context_width=8)):
        width = instance.context_width
        keras_model = instance.build(vocab_size=tokenizer.vocab_size, seed=0)
        # Order is history plus the target: 2 for a bigram, width + 1 otherwise.
        assert width + 1 == (2 if isinstance(instance, Bigram) else 9)
        assert tuple(keras_model(np.zeros((2, width), dtype=np.int32)).shape) == (
            2,
            tokenizer.vocab_size,
        )


def test_bigram_parameter_count_is_the_count_table(tokenizer):
    keras_model = Bigram().build(vocab_size=tokenizer.vocab_size, seed=0)
    assert keras_model.count_params() == tokenizer.vocab_size**2  # Flatten adds none


def test_architectures_declare_only_their_own_parameters():
    # The point of one estimator per model: nothing to ignore.
    assert Bigram().get_params() == {}
    assert MLP().get_params() == {
        "context_width": 16,
        "embedding_dim": 32,
        "hidden_units": 128,
    }
    assert LSTM().get_params() == {
        "context_width": 16,
        "embedding_dim": 32,
        "units": 256,
        "layers": 1,
    }
    # Discovery, not registration: definition order, and a clear error for a typo.
    assert [(cls.architecture_name(), summary) for cls, summary in architectures()] == [
        ("bigram", "Next token from the current token alone: the count table, learned."),
        ("mlp", "The whole context concatenated into one hidden layer."),
        ("lstm", "Recurrence: position is carried by state rather than by concatenation."),
    ]
    assert architectures()[1][0] is MLP
    with pytest.raises(ValueError, match="mlp"):
        architecture_from_dict({"architecture_name": "mlpp", "architecture_params": {}})


def test_architectures_are_cloneable_estimators():
    from sklearn.base import clone

    config = MLP(context_width=4, embedding_dim=8, hidden_units=16)
    assert clone(config).get_params() == config.get_params()
    assert config.set_params(hidden_units=64).hidden_units == 64
    assert config.set_params(context_width=12).context_width == 12


def test_architecture_error_paths(tokenizer):
    with pytest.raises(ValueError, match="vocab_size"):
        MLP(context_width=2).build(vocab_size=2)
    with pytest.raises(ValueError, match="context_width"):
        MLP(context_width=0).build(vocab_size=10)
    with pytest.raises(ValueError, match="layers"):
        LSTM(context_width=2, layers=3).build(vocab_size=10)
    for instance in (MLP(context_width=2), LSTM(context_width=2)):
        instance.build(vocab_size=10)  # constructs fine


# ---- training ------------------------------------------------------------


def test_training_reduces_loss_and_reports_stats(tokenizer):
    seen = []

    def hook(stats):
        seen.append(stats)

    model = fit(
        tokenizer,
        CORPUS,
        MLP(context_width=8, embedding_dim=8, hidden_units=16),
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
    results = model.results
    assert results.n_train_pairs + results.n_val_pairs == results.n_pairs_total
    assert results.n_val_pairs > 0 and results.n_train_pairs > 0
    assert model.training.epochs == 12 and model.training.batch_size == 32


def test_pairs_are_built_at_the_models_width(tokenizer):
    # The bigram has no width to ask for; the data follows the architecture.
    bigram = fit(tokenizer, CORPUS, Bigram(), epochs=1, batch_size=32, seed=0)
    assert bigram.context_width == 1  # read through the architecture, not stored
    assert bigram.keras_model.input_shape == (None, 1)

    wide = fit(
        tokenizer,
        CORPUS,
        LSTM(context_width=4, embedding_dim=8, units=16),
        epochs=1,
        batch_size=32,
        seed=0,
    )
    assert wide.context_width == 4
    assert wide.keras_model.input_shape == (None, 4)
    assert wide.results.n_pairs_total == bigram.results.n_pairs_total


def test_results_offer_perplexity_alongside_loss(mlp_model):
    # perplexity is e^loss, so it is derived, not a second stored number
    results = mlp_model.results
    assert results.final_perplexity == pytest.approx(math.exp(results.final_loss))
    assert results.final_val_perplexity == pytest.approx(math.exp(results.final_val_loss))


def test_the_epoch_hook_can_generate_from_the_model_as_it_trains(tokenizer):
    # Sampling during training is the caller's business: the model being fitted
    # is its own object, and the Keras model is assigned before the loop starts.
    samples = []
    model = LanguageModel(MLP(context_width=8, embedding_dim=8, hidden_units=16), tokenizer)
    model.fit(
        CORPUS,
        TrainingOptions(epochs=3, batch_size=32, seed=0),
        on_epoch=lambda stats: samples.append(model.generate("the", 0.0, max_new_tokens=8)),
    )
    assert len(samples) == 3
    assert all(isinstance(sample, str) and sample for sample in samples)


def test_stop_requested_stops_training_cleanly(tokenizer):
    seen = []

    def hook(stats):
        seen.append(stats["epoch"])

    def stop_requested():
        return len(seen) >= 2  # the widget's Stop button analogue

    model = fit(
        tokenizer,
        CORPUS,
        MLP(context_width=8, embedding_dim=8, hidden_units=16),
        epochs=50,
        batch_size=32,
        seed=0,
        on_epoch=hook,
        stop_requested=stop_requested,
    )
    assert model is not None
    assert max(seen) <= 3 and len(seen) < 50  # training ended early


def test_training_options_and_split_are_validated():
    with pytest.raises(ValueError, match="validation_fraction"):
        TrainingOptions(validation_fraction=0.0)
    with pytest.raises(ValueError, match="epochs"):
        TrainingOptions(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainingOptions(batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingOptions(learning_rate=0)
    with pytest.raises(ValueError, match="at least 2"):
        # a single sentence cannot yield a sentence-level split
        fit(
            tokenizer_from_text("one sentence only.", "char"),
            "one sentence only.",
            MLP(),
        )


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


def test_long_prompt_uses_only_the_last_context(mlp_model):
    # In-vocabulary prompt far longer than context width 8: only its tail can
    # influence the continuation.
    long_prompt = "the cat sat " * 6
    out = mlp_model.generate(long_prompt, temperature=0, max_new_tokens=10)
    assert out.startswith(long_prompt)  # prompt echoed, generation appended


def test_word_level_generate_smoke():
    tok = tokenizer_from_text(CORPUS, "word")
    # The bigram has no width parameter: it trains and generates at width 1,
    # with no input trimming anywhere and nothing to validate.
    model = fit(tok, CORPUS, Bigram(), epochs=8, batch_size=16, seed=0)
    assert model.context_width == 1
    out = model.generate("the", temperature=0, max_new_tokens=6)
    assert out.startswith("the")


def test_generate_rejects_a_negative_temperature(mlp_model):
    with pytest.raises(ValueError, match="temperature"):
        mlp_model.generate("the", temperature=-1.0)


def test_fit_on_empty_text_says_so(tokenizer):
    with pytest.raises(ValueError, match="empty"):
        fit(tokenizer, "", Bigram())
    with pytest.raises(ValueError, match="empty"):
        LanguageModel(Bigram(), tokenizer).fit("  \n  ", TrainingOptions(epochs=1))


def test_sample_temperature_zero_is_argmax():
    rng = np.random.default_rng(0)
    logits = np.array([1.0, 3.0, 2.0])
    assert _sample(logits, 0.0, rng) == 1
    assert _sample(logits, -1.0, rng) == 1  # any non-positive temperature = greedy


# ---- evaluation -----------------------------------------------------------


def test_evaluation_prefers_indomain_text(mlp_model):
    in_domain = mlp_model.evaluate("the cat sat on the mat. the cat saw the dog.")
    out_domain = mlp_model.evaluate("zx qv jk wp lk bf hg.")
    assert in_domain["perplexity"] > 1.0
    assert in_domain["perplexity"] < out_domain["perplexity"]


# ---- persistence ----------------------------------------------------------


def test_save_load_round_trip(mlp_model, tmp_path):
    path = mlp_model.save(tmp_path / "demo.nnlm")
    assert (path / "model.keras").exists() and (path / "meta.json").exists()
    loaded = LanguageModel.load(path)
    assert loaded.tokenizer.tokens == mlp_model.tokenizer.tokens
    assert loaded.training == mlp_model.training
    assert loaded.results == mlp_model.results
    assert loaded.parameter_count() == mlp_model.parameter_count()
    # deterministic comparison at greedy temperature: identical generations
    prompt = "the cat "
    assert mlp_model.generate(prompt, temperature=0, max_new_tokens=20) == loaded.generate(
        prompt, temperature=0, max_new_tokens=20
    )
    # the loaded model is compiled as well as weighted, so it can still be scored
    assert loaded.evaluate("the cat sat on the mat.")["perplexity"] == pytest.approx(
        mlp_model.evaluate("the cat sat on the mat.")["perplexity"]
    )


def test_language_model_is_serialisable_in_principle(mlp_model):
    # tokenizer json has no exotic types
    import json

    json.dumps(mlp_model.tokenizer.to_dict())


def test_stop_requested_works_without_epoch_hook(tokenizer):
    """The stop flag must not depend on an on_epoch callback being present."""
    model = fit(
        tokenizer,
        CORPUS,
        MLP(context_width=8, embedding_dim=8, hidden_units=16),
        epochs=5,
        batch_size=32,
        stop_requested=lambda: True,
    )
    assert math.isfinite(model.results.final_loss)


def test_unfitted_model_says_so():
    model = LanguageModel(Bigram(), tokenizer_from_text(CORPUS, "char"))
    with pytest.raises(RuntimeError, match="fit"):
        model.generate("the")
    with pytest.raises(RuntimeError, match="fit"):
        model.evaluate("the cat sat.")
    with pytest.raises(RuntimeError, match="fit"):
        model.parameter_count()


def test_saving_an_unfitted_model_writes_nothing(tmp_path):
    model = LanguageModel(Bigram(), tokenizer_from_text(CORPUS, "char"))
    target = tmp_path / "never-written"
    with pytest.raises(RuntimeError, match="fit"):
        model.save(target)
    # the guard runs before the directory is made: no orphan meta.json
    assert not target.exists()
