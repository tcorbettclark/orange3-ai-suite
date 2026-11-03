"""Orange <-> kernel adapters.

The architecture doc's discipline 1 lives HERE and only here: all
Orange-type conversion (Table, Corpus, meta columns) is adapter code in the
widget layer. The kernel never imports Orange; this module never imports Qt.

Inputs are read duck-typed where cheap: an Orange3-Text ``Corpus`` *is* a
``Table`` subclass (so a plain ``Table`` input accepts it automatically);
``.documents`` is used when present, string meta columns otherwise.
"""

from __future__ import annotations

import numpy as np
from Orange.data import (
    ContinuousVariable,
    Domain,
    StringVariable,
    Table,
)

CHECKPOINT_COLUMNS = ("epoch", "loss", "val_loss", "perplexity", "val_perplexity")


def table_texts(table: Table) -> list[str]:
    """Extract one document per row: string meta columns joined by newline.

    Prefers ``table.documents`` when the input is an Orange3-Text Corpus.
    """
    docs = getattr(table, "documents", None)
    if docs is not None and len(docs) == len(table):
        return [str(d) for d in docs]
    string_metas = [v for v in table.domain.metas if v.is_string]
    if not string_metas:
        raise ValueError(
            "no text found: connect data with string (text) columns, "
            "or an Import Documents / Corpus output"
        )
    return [
        "\n".join(str(row[var]) for var in string_metas).strip()
        for row in table
    ]


def generations_table(generations: list[str]) -> Table:
    """Generations as a one-column Table (readable in Corpus Viewer /
    Orange3-Text widgets when that add-on is installed)."""
    domain = Domain([], metas=[StringVariable("generation")])
    metas = np.asarray(generations, dtype=object).reshape(-1, 1)
    return Table.from_numpy(domain, np.empty((len(generations), 0)), None, metas)


def checkpoint_table(stats: list[dict]) -> Table:
    """Epoch -> loss/perplexity + sample text (the gibberish -> pastiche table).

    Keys expected: epoch, loss, val_loss, perplexity, val_perplexity, and
    optionally sample.
    """
    domain = Domain(
        [ContinuousVariable(name) for name in CHECKPOINT_COLUMNS],
        metas=[StringVariable("sample")],
    )
    missing = float("nan")
    x = np.asarray(
        [[row.get(name, missing) for name in CHECKPOINT_COLUMNS] for row in stats],
        dtype=np.float64,
    ).reshape(len(stats), len(CHECKPOINT_COLUMNS))
    metas = np.asarray(
        [[str(row.get("sample", ""))] for row in stats], dtype=object
    ).reshape(len(stats), 1)
    return Table.from_numpy(domain, x, None, metas)


def generations_strings(table: Table) -> list[str]:
    """Inverse of generations_table, for tests and downstream inspection."""
    return [str(row) for row in table.metas[:, 0]]