"""Personally identifiable information detection with checksums where the format has one,
so that a random 9-digit number is not reported as a tax file number and an order number is
not reported as a card."""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_AU_PHONE_RE = re.compile(
    r"(?<![\w-])(?:\+61[ -]?|0)(?:4\d{2}[ -]?\d{3}[ -]?\d{3}|[2378][ -]?\d{4}[ -]?\d{4})(?![\w-])"
)
_TFN_RE = re.compile(r"(?<![\w-])\d{3}[ -]?\d{3}[ -]?\d{2,3}(?![\w-])")
_CARD_RE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_BSB_ACCOUNT_RE = re.compile(r"(?<![\w-])\d{3}-\d{3}[ -]?\d{6,10}(?![\w-])")

TFN_WEIGHTS_9 = (1, 4, 3, 7, 5, 8, 6, 9, 10)
TFN_WEIGHTS_8 = (10, 7, 8, 4, 6, 3, 5, 1)


@dataclass(frozen=True, slots=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    text: str


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def tfn_checksum_valid(number: str) -> bool:
    digits = _digits(number)
    weights: tuple[int, ...]
    if len(digits) == 9:
        weights = TFN_WEIGHTS_9
    elif len(digits) == 8:
        weights = TFN_WEIGHTS_8
    else:
        return False
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


def luhn_valid(number: str) -> bool:
    digits = _digits(number)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect(text: str) -> list[PIIMatch]:
    """All matches, longest-first priority, overlaps removed."""
    raw: list[PIIMatch] = []
    for m in _CARD_RE.finditer(text):
        if luhn_valid(m.group()):
            raw.append(PIIMatch("CARD", m.start(), m.end(), m.group()))
    for m in _BSB_ACCOUNT_RE.finditer(text):
        raw.append(PIIMatch("BSB_ACCOUNT", m.start(), m.end(), m.group()))
    for m in _TFN_RE.finditer(text):
        if tfn_checksum_valid(m.group()):
            raw.append(PIIMatch("TFN", m.start(), m.end(), m.group()))
    for m in _AU_PHONE_RE.finditer(text):
        raw.append(PIIMatch("PHONE", m.start(), m.end(), m.group()))
    for m in _EMAIL_RE.finditer(text):
        raw.append(PIIMatch("EMAIL", m.start(), m.end(), m.group()))
    raw.sort(key=lambda p: (p.start, -(p.end - p.start)))
    kept: list[PIIMatch] = []
    for p in raw:
        if kept and p.start < kept[-1].end:
            continue
        kept.append(p)
    return kept


def redact(text: str) -> tuple[str, list[PIIMatch]]:
    matches = detect(text)
    if not matches:
        return text, []
    out: list[str] = []
    cursor = 0
    for m in matches:
        out.append(text[cursor : m.start])
        out.append(f"[{m.kind}]")
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out), matches
