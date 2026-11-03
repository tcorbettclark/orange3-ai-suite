"""Adapters: the sanctioned boundary between Orange objects and nn-kernel."""

import numpy as np
from Orange.data import Domain, StringVariable, Table

from nn_suite.adapters import (
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
        {"epoch": 1, "loss": 3.0, "val_loss": 3.1, "perplexity": 20.1,
         "val_perplexity": 22.2, "sample": "gibberish"},
        {"epoch": 2, "loss": 2.0, "val_loss": 2.1, "perplexity": 7.4,
         "val_perplexity": 8.2, "sample": "less gibberish"},
    ]
    table = checkpoint_table(rows)
    assert [v.name for v in table.domain.attributes] == [
        "epoch", "loss", "val_loss", "perplexity", "val_perplexity",
    ]
    assert table.domain.metas[0].name == "sample"
    assert table.metas[1, 0] == "less gibberish"
    assert table[0, "loss"] == 3.0


def test_checkpoint_table_tolerates_missing_sample():
    rows = [{"epoch": 1, "loss": 3.0, "val_loss": 3.1, "perplexity": 20.1,
             "val_perplexity": 22.2}]
    table = checkpoint_table(rows)
    assert not np.isnan(table[0, "perplexity"])
    assert str(table.metas[0, 0]) == ""