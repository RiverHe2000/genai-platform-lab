"""Scope control: the assistant serves a credit-policy desk. Requests outside that remit are
declined politely; requests for personal financial, legal, tax or medical advice are
*restricted* (a compliance line, not a capability line)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Label = Literal["in_scope", "out_of_scope", "restricted"]

IN_SCOPE_TERMS: frozenset[str] = frozenset(
    [
        "loan",
        "loans",
        "mortgage",
        "mortgages",
        "credit",
        "policy",
        "policies",
        "lvr",
        "dti",
        "dscr",
        "portfolio",
        "arrears",
        "dpd",
        "stage",
        "stages",
        "ecl",
        "ifrs",
        "exposure",
        "exposures",
        "balance",
        "balances",
        "borrower",
        "borrowers",
        "customer",
        "customers",
        "segment",
        "segments",
        "sme",
        "retail",
        "corporate",
        "provision",
        "provisions",
        "risk",
        "risks",
        "limit",
        "limits",
        "covenant",
        "covenants",
        "hardship",
        "review",
        "reviews",
        "serviceability",
        "valuation",
        "valuations",
        "rate",
        "rates",
        "interest",
        "default",
        "defaults",
        "impairment",
        "apra",
        "cps",
        "model",
        "models",
        "watchlist",
        "flag",
        "delegated",
        "committee",
        "lmi",
        "investor",
        "investors",
        "owner",
        "occupier",
        "overdraft",
        "facility",
        "facilities",
        "calculate",
        "calculation",
        "sum",
        "average",
        "count",
        "total",
        "how",
        "many",
        "what",
        "which",
        "show",
        "list",
        "query",
        "loanbook",
        "book",
        "product",
        "products",
        "state",
        "nsw",
        "vic",
        "qld",
        "wa",
        "sa",
        "originated",
        "past",
        "due",
        "days",
    ]
)
GREETINGS = frozenset({"hi", "hello", "hey", "thanks", "thank", "good", "morning", "afternoon"})
RESTRICTED_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "investment_advice",
        r"\b(should i|shall i|is it (a good|the right) time to|is now a good time to)\b"
        r".{0,30}\b(buy|sell|invest|refinance)\b",
    ),
    (
        "investment_advice",
        r"\b(which|what) (stocks?|shares|crypto\w*|etfs?|funds?)\b.{0,30}\b(buy|invest|pick)\b",
    ),
    ("tax_advice", r"\b(minimi[sz]e|avoid|reduce) (my |our )?tax\b|\btax advice\b"),
    ("legal_advice", r"\blegal advice\b|\bcan (i|we) sue\b|\bis it legal\b"),
    ("medical", r"\b(diagnos\w+|medical advice|prescri\w+)\b"),
)
_RESTRICTED = [(name, re.compile(p, re.IGNORECASE)) for name, p in RESTRICTED_PATTERNS]
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class TopicVerdict:
    label: Label
    reason: str


def classify(text: str) -> TopicVerdict:
    for name, regex in _RESTRICTED:
        if regex.search(text):
            return TopicVerdict("restricted", name)
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return TopicVerdict("out_of_scope", "empty")
    hits = [t for t in tokens if t in IN_SCOPE_TERMS]
    if hits:
        return TopicVerdict("in_scope", f"terms: {', '.join(sorted(set(hits))[:5])}")
    if len(tokens) <= 4 and any(t in GREETINGS for t in tokens):
        return TopicVerdict("in_scope", "greeting")
    return TopicVerdict("out_of_scope", "no credit-policy terms found")
