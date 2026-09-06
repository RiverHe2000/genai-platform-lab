"""Evaluation samples: question, reference answer (``None`` = unanswerable, the pipeline
should abstain), gold document ids for retrieval metrics, and free-form tags for slicing."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ragpipe.documents import read_jsonl


class EvalSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth: str | None = None
    gold_doc_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.ground_truth is not None


def load_evalset(path: Path | str) -> list[EvalSample]:
    samples = [EvalSample.model_validate(row) for row in read_jsonl(path)]
    seen: set[str] = set()
    for s in samples:
        if s.id in seen:
            msg = f"duplicate sample id {s.id!r} in {path}"
            raise ValueError(msg)
        seen.add(s.id)
    if not samples:
        msg = f"evalset {path} is empty"
        raise ValueError(msg)
    return samples


def check_against_corpus(samples: Iterable[EvalSample], doc_ids: Iterable[str]) -> list[str]:
    """Return human-readable problems (gold docs that do not exist in the corpus)."""
    known = set(doc_ids)
    problems: list[str] = []
    for s in samples:
        for d in s.gold_doc_ids:
            if d not in known:
                problems.append(f"sample {s.id}: gold doc {d!r} not in corpus")
    return problems
