"""Lexical preprocessing shared by BM25, the hashing embedder and the chunkers.

Everything here returns *spans* (character offsets) where possible so callers can slice
the original text rather than work with a normalised copy.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_WORD_RE = re.compile(r"\S+")
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
# End of sentence: terminal punctuation (optionally followed by a closing quote/bracket)
# that is followed by whitespace and then a capital letter, digit or opening quote/bracket.
_SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s+[A-Z0-9\"'(\[])")
_ABBREVIATION_TEXT = "e.g i.e no nos sec cl para etc vs approx mr ms mrs dr fig s ss cf ref"
_ABBREVIATIONS = frozenset(_ABBREVIATION_TEXT.split())

_STOPWORD_TEXT = """
    a an and are as at be by for from has have had in is it its of on or that the to was were
    will with this these those which who whom whose what when where why how not no nor but if
    then than so such into onto over under out up down about above below between through during
    before after again further once here there all any both each few more most other some own
    same only very can just should would could may might must shall do does did done being been
    """
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())


def stem(token: str) -> str:
    """Conservative suffix stripping (plurals only). A full Porter stemmer buys little for
    policy prose and would make the tokens harder to read in retrieval traces."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True, apply_stem: bool = True) -> list[str]:
    """Lower-case word tokens, keeping internal dots/apostrophes ("cet1", "1.5", "don't")."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if apply_stem:
        tokens = [stem(t) for t in tokens]
    return tokens


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of whitespace-delimited words."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Spans separated by blank lines; leading/trailing whitespace excluded."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _PARAGRAPH_BREAK_RE.finditer(text):
        trimmed = _trim(text, start, m.start())
        if trimmed:
            spans.append(trimmed)
        start = m.end()
    trimmed = _trim(text, start, len(text))
    if trimmed:
        spans.append(trimmed)
    return spans


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence spans with a small abbreviation guard ("e.g. The" does not split)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        preceding = text[start : m.start()].split()
        last = preceding[-1].lower().rstrip(".") if preceding else ""
        if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
            continue
        trimmed = _trim(text, start, m.end())
        if trimmed:
            spans.append(trimmed)
        start = m.end()
    trimmed = _trim(text, start, len(text))
    if trimmed:
        spans.append(trimmed)
    return spans
