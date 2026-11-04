"""LoRA SFT widget tests: prompt/completion data contract and helpers."""

import numpy as np
import pytest
from Orange.data import Domain, StringVariable, Table

from nn_suite.adapters import table_pairs
from nn_suite.widgets.ow_lora_sft import OWLoRASFT


def _pair_table(pairs):
    domain = Domain(
        [],
        metas=[StringVariable("prompt"), StringVariable("completion")],
    )
    rows = np.asarray(pairs, dtype=object).reshape(-1, 2)
    return Table.from_numpy(domain, np.empty((len(pairs), 0)), None, rows)


def test_table_pairs_helper():
    domain = Domain(
        [],
        metas=[
            StringVariable("completion"),
            StringVariable("prompt"),
            StringVariable("note"),
        ],
    )
    rows = [["answer a", "question a", "x"], ["answer b", "question b", "y"]]
    table = Table.from_numpy(domain, np.empty((2, 0)), None, np.asarray(rows, dtype=object))
    assert table_pairs(table) == [
        ("question a", "answer a"),
        ("question b", "answer b"),
    ]


def test_table_pairs_requires_two_columns():
    domain = Domain([], metas=[StringVariable("only")])
    table = Table.from_numpy(
        domain, np.empty((1, 0)), None, np.asarray([["x"]], dtype=object).reshape(1, 1)
    )
    with pytest.raises(ValueError, match="two text columns"):
        table_pairs(table)


def test_sft_builds_prompt_completion_pairs(qapp):
    w = OWLoRASFT()
    table = _pair_table([(f"question {i}", f"answer {i}") for i in range(5)])
    w.set_data(table)
    data = w._make_lora_data()
    assert data.kind == "pairs"
    assert data.train[0] == ("question 0", "answer 0")
    assert len(data.train) + len(data.valid) == 5
