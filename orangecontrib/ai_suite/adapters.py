"""Orange Tables built from kernel results, and kernel arguments read out of Tables.

The one place Orange types are converted: Table, Corpus and meta-column inputs
become kernel arguments, and a list of generations or of epoch stats becomes a
Table.

The kernel never imports Orange, and this module never imports Qt.
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


def image_paths(table: Table):
    """The image column of a table, and its paths resolved against the column's ``origin``.

    The column must be the one Import Images marks as an image path, and the paths
    come from that add-on's own ``extract_paths``. A table without such a column,
    or without the add-on installed, raises ``ValueError``.
    """
    try:
        from orangecontrib.imageanalytics.utils.image_utils import extract_paths
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ValueError(
            "Orange3-ImageAnalytics is required for image inputs "
            "(install it via Options > Add-ons)."
        ) from exc
    # Import Images marks its path column explicitly; require that marker.
    # Image Analytics' own filter_image_attributes is too loose (it accepts
    # any string/categorical meta, merely sorting marked ones first).
    candidates = [
        v
        for v in table.domain.metas
        if isinstance(v, (StringVariable, DiscreteVariable)) and v.attributes.get("type") == "image"
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
    """One document per row: the string meta columns joined by newline.

    ``table.documents`` wins when the input is an Orange3-Text Corpus, and a
    string column named ``text`` is used on its own. A table with no string
    column raises ``ValueError``.
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
    by_name = {v.name.lower(): v for v in string_metas}
    text_metas = [by_name["text"]] if "text" in by_name else string_metas
    return ["\n".join(str(row[var]) for var in text_metas).strip() for row in table]


def corpus_text(table: Table) -> str:
    """All documents of a table as the one text the kernel takes.

    Rows are separated by a newline, a hard break for the kernel's sentence
    splitting.
    """
    return "\n".join(table_texts(table))


def table_pairs(table: Table) -> list[tuple[str, str]]:
    """Prompt-response pairs from a table, the supervised fine-tuning format.

    Columns named ``prompt``/``completion`` (or their common synonyms, matched
    case-insensitively) are used when present, otherwise the first two string
    metas in display order. Pairs with an empty side are dropped, and a missing
    second text column raises ``ValueError``.
    """
    string_metas = [v for v in table.domain.metas if v.is_string]
    by_name = {v.name.lower(): v for v in string_metas}
    prompt_var = by_name.get("prompt") or by_name.get("question")
    completion_var = by_name.get("completion") or by_name.get("response") or by_name.get("answer")
    missing_prompt = prompt_var is None and len(string_metas) < 2
    missing_completion = prompt_var is not None and completion_var is None
    if missing_prompt or missing_completion:
        raise ValueError(
            "prompt-response pairs need two text columns (named prompt/"
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
    """Generations as a one-column Table of ``generation`` strings.

    The column is a string meta, which Corpus Viewer and the Orange3-Text widgets
    read when that add-on is installed.
    """
    domain = Domain([], metas=[StringVariable("generation")])
    metas = np.asarray(generations, dtype=object).reshape(-1, 1)
    return Table.from_numpy(domain, np.empty((len(generations), 0)), None, metas)


def checkpoint_table(stats: list[dict]) -> Table:
    """One row per epoch of loss, perplexity and sample text.

    The columns follow the kernel's epoch-stats key order, taken from the module
    that produces them; ``sample`` is the widget's addition, not the kernel's, and
    an epoch without one gets an empty string. The import stays inside this
    function: the epoch-stats module imports Keras, and the image widgets import
    this module.
    """
    from edu_ai_suite.lm.stats import EPOCH_STATS_KEYS as columns

    domain = Domain(
        [ContinuousVariable(name) for name in columns],
        metas=[StringVariable("sample")],
    )
    missing = float("nan")
    x = np.asarray(
        [[row.get(name, missing) for name in columns] for row in stats],
        dtype=np.float64,
    ).reshape(len(stats), len(columns))
    metas = np.asarray([[str(row.get("sample", ""))] for row in stats], dtype=object).reshape(
        len(stats), 1
    )
    return Table.from_numpy(domain, x, None, metas)


def generations_strings(table: Table) -> list[str]:
    """The generations of a :func:`generations_table` Table, in row order."""
    return [str(row) for row in table.metas[:, 0]]
