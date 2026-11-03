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
    DiscreteVariable,
    Domain,
    StringVariable,
    Table,
)

CHECKPOINT_COLUMNS = ("epoch", "loss", "val_loss", "perplexity", "val_perplexity")


def image_paths(table: Table):
    """Image column and resolved paths from an Import-Images-style table.

    Returns ``(column, paths)`` where paths are resolved against the
    column's ``origin`` attribute — exactly the contract Import Images
    (Orange3-ImageAnalytics) produces. Uses that add-on's public helpers via
    a lazy import; the error message covers its absence.
    """
    try:
        from orangecontrib.imageanalytics.utils.image_utils import extract_paths
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ValueError(
            "Orange3-ImageAnalytics is required for image inputs "
            "(Install via Options > Add-ons)."
        ) from exc
    # Import Images marks its path column explicitly; require that marker —
    # ImageAnalytics' own filter_image_attributes is too loose (it accepts
    # ANY string/categorical meta, merely sorting marked ones first).
    candidates = [
        v for v in table.domain.metas
        if isinstance(v, (StringVariable, DiscreteVariable))
        and v.attributes.get("type") == "image"
    ]
    candidates.sort(key=lambda v: v.name)
    if not candidates:
        raise ValueError(
            "no image column found: connect the Images output of Import Images "
            "(a text column marked as an image path)."
        )
    column = candidates[0]
    return column, extract_paths(table, column)


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


def table_pairs(table: Table) -> list[tuple[str, str]]:
    """Prompt–response pairs from a Table (the SFT data format, spec §4.3).

    Looks for columns literally named 'prompt'/'completion' (case-
    insensitive); otherwise takes the first two string metas in display
    order. A missing second column is an error.
    """
    string_metas = [v for v in table.domain.metas if v.is_string]
    by_name = {v.name.lower(): v for v in string_metas}
    prompt_var = by_name.get("prompt") or by_name.get("question")
    completion_var = by_name.get("completion") or by_name.get("response") \
        or by_name.get("answer")
    if prompt_var is None and len(string_metas) < 2 or (
        prompt_var is not None and completion_var is None
    ):
        raise ValueError(
            "prompt–response pairs need two text columns (named prompt/ "
            "completion, or the first two text columns)"
        )
    prompt_var = prompt_var or string_metas[0]
    completion_var = completion_var or string_metas[1]
    return [
        (str(row[prompt_var]), str(row[completion_var]))
        for row in table
        if str(row[prompt_var]).strip() and str(row[completion_var]).strip()
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