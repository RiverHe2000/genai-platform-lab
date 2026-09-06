"""Grounded-QA prompt, citation parsing and abstention detection."""

from __future__ import annotations

import re
from collections.abc import Sequence

ABSTAIN_MARKER = "I don't know"

SYSTEM_PROMPT = (
    "You are an assistant for a bank's risk and policy team. Answer the question using ONLY "
    "the numbered context passages. Be concise and factual. After each claim, cite the passage "
    "number in square brackets, e.g. [1] or [2]. If the passages do not contain the "
    f"information needed, reply exactly: {ABSTAIN_MARKER}"
)

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "does not contain",
    "do not contain",
    "not contain the information",
    "no information",
    "cannot be determined from",
    "not provided in the context",
)


def build_qa_prompt(question: str, contexts: Sequence[tuple[str, str]]) -> str:
    """``contexts`` are (title, text) pairs in rank order; numbering starts at 1."""
    if not contexts:
        body = "(no passages retrieved)"
    else:
        body = "\n\n".join(
            f"[{i}] ({title})\n{text.strip()}" for i, (title, text) in enumerate(contexts, start=1)
        )
    return f"Context passages:\n\n{body}\n\nQuestion: {question.strip()}\nAnswer:"


def extract_citations(answer: str) -> list[int]:
    """Unique citation numbers in order of first appearance."""
    seen: list[int] = []
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def is_abstention(answer: str) -> bool:
    text = answer.strip().lower().replace("\u2019", "'")
    return any(p in text for p in _ABSTAIN_PATTERNS)
