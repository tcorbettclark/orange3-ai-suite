"""Adapters: the conversion between Orange tables and edu_ai_suite's arguments and results."""

import numpy as np
from Orange.data import Domain, StringVariable, Table

from orangecontrib.ai_suite.adapters import (
    checkpoint_table,
    generations_strings,
    generations_table,
    table_texts,
)


def test_table_texts_joins_string_metas():
    domain = Domain([], metas=[StringVariable("a"), StringVariable("b")])
    metas = np.asarray([["one one", "second doc"]], dtype=object)
    table = Table.from_numpy(domain, np.empty((1, 0)), None, metas)
    assert table_texts(table) == ["one one\nsecond doc"]


def test_table_texts_prefers_text_column():
    domain = Domain(
        [],
        metas=[StringVariable("source"), StringVariable("text")],
    )
    metas = np.asarray([["book.txt", "the actual document"]], dtype=object)
    table = Table.from_numpy(domain, np.empty((1, 0)), None, metas)
    assert table_texts(table) == ["the actual document"]


def test_table_texts_rejects_textless_tables():
    domain = Domain([], metas=[])
    table = Table.from_numpy(domain, np.empty((2, 0)))
    import pytest

    with pytest.raises(ValueError, match="no text found"):
        table_texts(table)


def test_generations_table_round_trip():
    table = generations_table(["alpha beta", "gamma"])
    assert len(table) == 2
    assert table.domain.metas[0].name == "generation"
    assert generations_strings(table)[1] == "gamma"


def test_checkpoint_table_columns_and_sample():
    rows = [
        {
            "epoch": 1,
            "loss": 3.0,
            "val_loss": 3.1,
            "perplexity": 20.1,
            "val_perplexity": 22.2,
            "sample": "gibberish",
        },
        {
            "epoch": 2,
            "loss": 2.0,
            "val_loss": 2.1,
            "perplexity": 7.4,
            "val_perplexity": 8.2,
            "sample": "less gibberish",
        },
    ]
    table = checkpoint_table(rows)
    # Structural tie: the columns are edu_ai_suite's own epoch-stats key order,
    # taken from the leaf module that names them - not a copy spelled here, and
    # not an import that would drag Keras into the widget layer.
    from edu_ai_suite.lm.stats import EPOCH_STATS_KEYS

    assert [v.name for v in table.domain.attributes] == list(EPOCH_STATS_KEYS)
    assert table.domain.metas[0].name == "sample"
    assert table.metas[1, 0] == "less gibberish"
    assert table[0, "loss"] == 3.0


def test_checkpoint_table_tolerates_missing_sample():
    rows = [{"epoch": 1, "loss": 3.0, "val_loss": 3.1, "perplexity": 20.1, "val_perplexity": 22.2}]
    table = checkpoint_table(rows)
    assert not np.isnan(table[0, "perplexity"])
    assert str(table.metas[0, 0]) == ""


def test_importing_the_adapters_does_not_import_the_ml_stack():
    # adapters is imported by the image widgets too, and the epoch-stats key order
    # it needs lives beside a module that imports Keras (about two seconds and a
    # few hundred megabytes). The constant is therefore read lazily, in
    # checkpoint_table, and this pins that.
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    probe = "import sys, orangecontrib.ai_suite.adapters; print('keras' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=root)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False", done.stdout
