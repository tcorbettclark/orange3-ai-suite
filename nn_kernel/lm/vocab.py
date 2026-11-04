"""Read-only statistics over a :class:`~nn_kernel.lm.tokenizer.Tokenizer`.

The LM Vocab Viewer has no pipeline role: these functions return plain data for
a widget to render and never feed back into training or generation. They make
the LM Pre-trainer's tokenizer choice testable ("char level: smaller vocab, longer
sequences").
"""

from __future__ import annotations

from .tokenizer import BOS_ID, SPECIAL_TOKENS, UNK_ID, Tokenizer


def _counted(tok: Tokenizer) -> list[tuple[str, int]]:
    """(token, corpus count) pairs, excluding the special tokens."""
    n = len(SPECIAL_TOKENS)
    return list(zip(tok.tokens[n:], tok.counts[n:], strict=True))


def vocab_stats(tok: Tokenizer) -> dict:
    """Summary: size, longest tokens, length spread, hapax fraction."""
    real = _counted(tok)
    by_length = sorted(real, key=lambda tc: (-len(tc[0]), -tc[1]))
    lengths = [len(t) for t, _ in real]
    n = len(real)
    return {
        "level": tok.level,
        "vocab_size": tok.vocab_size,
        "special_tokens": (tok.token(BOS_ID), tok.token(UNK_ID)),
        "longest_tokens": by_length[:10],
        "mean_token_length": (sum(lengths) / n) if n else 0.0,
        "single_occurrence_fraction": (sum(1 for _, c in real if c == 1) / n if n else 0.0),
    }


def frequency_table(tok: Tokenizer, n: int = 20) -> list[tuple[int, str, int]]:
    """Top-n tokens as (rank, token, count), 1-based ranks."""
    ranked = sorted(_counted(tok), key=lambda tc: (-tc[1], tc[0]))[:n]
    return [(rank, token, count) for rank, (token, count) in enumerate(ranked, start=1)]


def zipf_summary(tok: Tokenizer, top: int = 100) -> dict:
    """Zipf's-law side lesson: rank x count is roughly constant for natural text.

    ``zipf_product_mean`` is the mean of count*rank over the top ``top``
    tokens; ``zipf_product_spread`` is its relative range.
    """
    table = frequency_table(tok, top)
    if not table:
        return {"n": 0}
    products = [count * rank for rank, _, count in table]
    total = sum(tok.counts)
    mean_product = sum(products) / len(products)
    return {
        "n": len(table),
        "count_top1": table[0][2],
        "top1_share": table[0][2] / total if total else 0.0,
        "top10_share": sum(c for _, _, c in table[:10]) / total if total else 0.0,
        "zipf_product_mean": mean_product,
        "zipf_product_spread": (max(products) - min(products)) / mean_product,
    }


def side_by_side(tok_char: Tokenizer, tok_word: Tokenizer, text: str) -> dict:
    """One text at both levels: (token, id) lists for the comparison view."""
    if tok_char.level != "char" or tok_word.level != "word":
        raise ValueError("side_by_side expects a char-level and a word-level tokenizer")
    return {
        "char": [(tok_char.token(i), i) for i in tok_char.encode(text)],
        "word": [(tok_word.token(i), i) for i in tok_word.encode(text)],
    }
