"""Keyword search over short credit-policy clauses. Deliberately simple (token overlap with
tf weighting) — the point of this project is the agent and its rails, not retrieval; the
companion ``rag-pipeline-eval`` project covers retrieval properly."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from agentguard.tools.base import ToolContext, ToolResult, ToolSpec

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
        "what",
        "which",
        "how",
    ]
)


@dataclass(frozen=True, slots=True)
class PolicyClause:
    id: str
    title: str
    text: str


DEFAULT_CLAUSES: tuple[PolicyClause, ...] = (
    PolicyClause(
        "CP-1.2",
        "Residential LVR limits",
        "The maximum loan-to-value ratio without lenders mortgage insurance is 80%. With LMI "
        "the maximum LVR is 95% for owner-occupiers and 90% for investors.",
    ),
    PolicyClause(
        "CP-1.4",
        "Debt-to-income",
        "Applications with a debt-to-income ratio above 6 require Credit Committee approval; a "
        "DTI above 8 is not permitted under any delegation.",
    ),
    PolicyClause(
        "CP-2.1",
        "SME delegated limits",
        "The maximum aggregate exposure to one SME group under delegated authority is AUD 5 "
        "million; larger exposures go to the Credit Committee.",
    ),
    PolicyClause(
        "CP-2.3",
        "SME covenants",
        "New SME facilities require a minimum debt service coverage ratio of 1.25 times and "
        "interest cover of at least 2.0 times.",
    ),
    PolicyClause(
        "IM-3.1",
        "IFRS 9 staging triggers",
        "An exposure moves to Stage 2 when it is 30 days past due, when lifetime PD has at "
        "least doubled with an absolute increase of 50 basis points, on watchlist placement or "
        "forbearance. Default (Stage 3) is 90 days past due.",
    ),
    PolicyClause(
        "IM-3.4",
        "Watchlist review",
        "Stage 2 and Stage 3 exposures above AUD 1 million are reviewed monthly by the "
        "Watchlist Committee; a relationship manager may flag any loan for review with a reason.",
    ),
    PolicyClause(
        "CH-4.1",
        "Financial hardship",
        "Customers who notify financial hardship are referred to the Financial Assistance team "
        "within 2 business days and hardship complaints are resolved within 21 calendar days.",
    ),
    PolicyClause(
        "DP-5.2",
        "Tax file numbers",
        "Tax file numbers are stored only in encrypted fields and must never appear in "
        "free-text notes, emails, chat transcripts or logs.",
    ),
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class PolicySearch:
    def __init__(self, clauses: Iterable[PolicyClause] = DEFAULT_CLAUSES) -> None:
        self._clauses = list(clauses)

    def add(self, clause: PolicyClause) -> None:
        self._clauses.append(clause)

    def search(self, query: str, k: int = 3) -> list[tuple[PolicyClause, float]]:
        q = tokenize(query)
        if not q:
            return []
        scored: list[tuple[PolicyClause, float]] = []
        for clause in self._clauses:
            doc = tokenize(clause.title + " " + clause.text)
            if not doc:
                continue
            counts: dict[str, int] = {}
            for t in doc:
                counts[t] = counts.get(t, 0) + 1
            score = sum(1.0 + 0.1 * counts.get(t, 0) for t in set(q) if t in counts)
            if score > 0:
                scored.append((clause, score))
        scored.sort(key=lambda cs: (-cs[1], cs[0].id))
        return scored[:k]


class PolicySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Keywords describing the policy topic.")
    k: int = Field(3, ge=1, le=5, description="Number of clauses to return.")


def make_policy_tool(search: PolicySearch) -> ToolSpec:
    def handler(args: BaseModel, ctx: ToolContext) -> ToolResult:
        del ctx
        query = str(getattr(args, "query", ""))
        k = int(getattr(args, "k", 3))
        hits = search.search(query, k=k)
        if not hits:
            return ToolResult(ok=True, output="no matching policy clauses", data={"ids": []})
        body = "\n\n".join(f"[{c.id}] {c.title}: {c.text}" for c, _ in hits)
        return ToolResult(ok=True, output=body, data={"ids": [c.id for c, _ in hits]})

    return ToolSpec(
        name="search_policy",
        description="Find credit-policy clauses by keyword. Quote clause ids in your answer.",
        args_model=PolicySearchArgs,
        handler=handler,
    )


def clauses_from_dicts(rows: Sequence[dict[str, str]]) -> list[PolicyClause]:
    return [PolicyClause(id=r["id"], title=r["title"], text=r["text"]) for r in rows]
