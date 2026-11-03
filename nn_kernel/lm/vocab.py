"""Vocab Inspector logic — read-only statistics over a Tokenizer.

Spec commitments (tiny-lm-suite-design.md §4.4): the inspector has no
pipeline role and no outputs that feed back into training/generation; it
makes the Trainer's tokenizer dropdown a *testable hypothesis* ("char-level:
smaller vocab, longer sequences"). Everything here is plain data for a
widget to render.
"""

from __future__ import annotations

from .tokenizer import UNK_ID, BOS_ID, Tokenizer


def vocab_stats(tok: Tokenizer) -> dict:
    """Summary: size, longest tokens, length spread, hapax fraction."""
    real = [(t, c) for i, (t, c) in enumerate(zip(tok.tokens, tok.counts)) if i >= BOS_ID + 2]
    by_length = sorted(real, key=lambda tc: (-len(tc[0]), -tc[1]))
    lengths = [len(t) for t, _ in real]
    n = len(real)
    return {
        "level": tok.level,
        "vocab_size": tok.vocab_size,
        "special_tokens": (tok.token(BOS_ID), tok.token(UNK_ID)),
        "longest_tokens": by_length[:10],
        "mean_token_length": (sum(lengths) / n) if n else 0.0,
        "single_occurrence_fraction": (
            sum(1 for _, c in real if c == 1) / n if n else 0.0
        ),
    }


def frequency_table(tok: Tokenizer, n: int = 20) -> list[tuple[int, str, int]]:
    """Top-n tokens as (rank, token, count), 1-based ranks."""
    ranked = sorted(
        (
            (c, t)
            for i, (t, c) in enumerate(zip(tok.tokens, tok.counts))
            if i >= BOS_ID + 2
        ),
        key=lambda tc: (-tc[0], tc[1]),
    )[:n]
    return [(rank, token, count) for rank, (count, token) in enumerate(ranked, start=1)]


def zipf_summary(tok: Tokenizer, top: int = 100) -> dict:
    """Zipf's-law side-lesson: rank x count roughly constant for a natural corpus.

    ``zipf_product_mean`` is the mean of count*rank over the top ``top``
    tokens; lower/more stable values mean closer adherence to Zipf's law.
    """
    table = frequency_table(tok, top)
    if not table:
        return {"n": 0}
    products = [count * rank for rank, _, count in table]
    total = sum(tok.counts)
    return {
        "n": len(table),
        "count_top1": table[0][2],
        "top1_share": table[0][2] / total if total else 0.0,
        "top10_share": sum(c for _, _, c in table[:10]) / total if total else 0.0,
        "zipf_product_mean": sum(products) / len(products),
        "zipf_product_spread": (max(products) - min(products)) / (sum(products) / len(products)),
    }


def side_by_side(tok_char: Tokenizer, tok_word: Tokenizer, text: str) -> dict:
    """One text, both levels: (token, id) lists for the inspector's comparison view."""
    if tok_char.level != "char" or tok_word.level != "word":
        raise ValueError("side_by_side expects a char-level and a word-level tokenizer")
    return {
        "char": [(tok_char.token(i), i) for i in tok_char.encode(text)],
        "word": [(tok_word.token(i), i) for i in tok_word.encode(text)],
    }