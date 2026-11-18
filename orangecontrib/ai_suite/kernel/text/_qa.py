"""Q/A blocks: fixed ``Q:``/``A:`` text becomes prompt/completion pairs."""

from __future__ import annotations

from dataclasses import dataclass

from ._clean import reflow


@dataclass(frozen=True)
class QAPair:
    prompt: str
    completion: str


@dataclass(frozen=True)
class QAParseResult:
    pairs: tuple[QAPair, ...]
    skipped: int
    preamble_lines: int


def _is_q(line: str) -> bool:
    return line.lstrip().startswith("Q:")


def _is_a(line: str) -> bool:
    return line.lstrip().startswith("A:")


def parse_qa(text: str) -> QAParseResult:
    """Parse fixed Q:/A: blocks into prompt/completion pairs.

    A new pair starts at any line beginning with ``Q:``. The prompt includes
    the ``A:`` marker line; the completion runs until the next ``Q:`` line.
    """
    lines = text.splitlines()
    pairs: list[QAPair] = []
    skipped = 0
    preamble = 0
    i = 0
    seen_q = False
    while i < len(lines):
        if not _is_q(lines[i]):
            if not seen_q and lines[i].strip():
                preamble += 1
            i += 1
            continue
        seen_q = True
        q_start = i
        j = i + 1
        while j < len(lines) and not _is_a(lines[j]) and not _is_q(lines[j]):
            j += 1
        if j >= len(lines) or not _is_a(lines[j]):
            skipped += 1
            i = j
            continue
        a_line = j
        a_text = lines[a_line]
        a_col = a_text.find("A:")
        k = j + 1
        while k < len(lines) and not _is_q(lines[k]):
            k += 1
        prompt = "\n".join([*lines[q_start:a_line], a_text[: a_col + 2].strip()]).strip()
        completion = "\n".join([a_text[a_col + 2 :].strip(), *lines[a_line + 1 : k]]).strip()
        if prompt and completion:
            pairs.append(QAPair(prompt, completion))
        else:
            skipped += 1
        i = k
    return QAParseResult(tuple(pairs), skipped, preamble)


def reflow_qa_pair(pair: QAPair) -> QAPair:
    """Reflow a Q/A pair without merging the prompt and its ``A:`` marker."""
    prompt_lines = pair.prompt.splitlines()
    a_index = next(
        (i for i, line in enumerate(prompt_lines) if _is_a(line)),
        len(prompt_lines) - 1,
    )
    q_text = " ".join(line.strip() for line in prompt_lines[:a_index] if line.strip())
    a_text = prompt_lines[a_index].strip() if a_index < len(prompt_lines) else "A:"
    prompt = f"{q_text}\n{a_text}".strip()
    completion = reflow(pair.completion)
    return QAPair(prompt, completion)
