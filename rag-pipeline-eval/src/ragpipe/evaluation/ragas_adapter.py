"""Cross-check our metric implementations against the official ``ragas`` package.

``ragas`` is imported lazily: the core harness has no dependency on it (the package pins a
large LangChain/OpenAI dependency tree that breaks on import with some LangChain releases —
see ``pyproject.toml``). The adapter wraps our ``LLM``/``Embedder`` protocols in the minimal
LangChain interfaces ragas accepts, so the *same* local judge scores both implementations
and any disagreement is about the metric definition, not the model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragpipe.embeddings import Embedder
from ragpipe.evaluation.runner import SampleRecord
from ragpipe.llm import LLM

OFFICIAL_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def to_ragas_rows(records: Sequence[SampleRecord]) -> list[dict[str, Any]]:
    """Answerable samples with at least one context, in the column names ragas expects."""
    rows: list[dict[str, Any]] = []
    for r in records:
        if r.ground_truth is None or not r.contexts:
            continue
        rows.append(
            {
                "id": r.id,
                "user_input": r.question,
                "response": r.answer,
                "retrieved_contexts": list(r.contexts),
                "reference": r.ground_truth,
            }
        )
    return rows


def _build_langchain_llm(llm: LLM, max_tokens: int, temperature: float) -> Any:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class RagpipeChatModel(BaseChatModel):  # type: ignore[misc]
        inner: Any = None

        @property
        def _llm_type(self) -> str:
            return "ragpipe"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            del stop, run_manager  # LangChain passes them by keyword; unused here
            system_parts = [str(m.content) for m in messages if m.type == "system"]
            user_parts = [str(m.content) for m in messages if m.type != "system"]
            n = int(kwargs.get("n", 1) or 1)
            generations = []
            for _ in range(n):
                response = self.inner.complete(
                    "\n\n".join(user_parts),
                    system="\n\n".join(system_parts) or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                generations.append(ChatGeneration(message=AIMessage(content=response.text)))
            return ChatResult(generations=generations)

    return RagpipeChatModel(inner=llm)


def _build_langchain_embeddings(embedder: Embedder) -> Any:
    from langchain_core.embeddings import Embeddings

    class RagpipeEmbeddings(Embeddings):  # type: ignore[misc]
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(x) for x in row] for row in embedder.embed_documents(texts)]

        def embed_query(self, text: str) -> list[float]:
            return [float(x) for x in embedder.embed_queries([text])[0]]

    return RagpipeEmbeddings()


def run_official_ragas(
    records: Sequence[SampleRecord],
    llm: LLM,
    embedder: Embedder,
    *,
    metrics: Sequence[str] = OFFICIAL_METRICS,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, dict[str, float | None]]:
    """Return ``{metric: {sample_id: score}}`` from the official implementation."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    registry: dict[str, Any] = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }
    unknown = [m for m in metrics if m not in registry]
    if unknown:
        msg = f"unknown ragas metrics: {unknown}"
        raise ValueError(msg)
    rows = to_ragas_rows(records)
    if not rows:
        return {m: {} for m in metrics}
    dataset = EvaluationDataset.from_list([{k: v for k, v in r.items() if k != "id"} for r in rows])
    result = evaluate(
        dataset,
        metrics=[registry[m] for m in metrics],
        llm=_build_langchain_llm(llm, max_tokens, temperature),
        embeddings=_build_langchain_embeddings(embedder),
        show_progress=False,
        raise_exceptions=False,
    )
    scores: list[dict[str, Any]] = list(result.scores)
    out: dict[str, dict[str, float | None]] = {m: {} for m in metrics}
    for row, score in zip(rows, scores, strict=True):
        for m in metrics:
            column = registry[m].name
            value = score.get(column)
            out[m][row["id"]] = (
                None
                if value is None or (isinstance(value, float) and math.isnan(value))
                else float(value)
            )
    return out


@dataclass(frozen=True, slots=True)
class Agreement:
    metric: str
    n: int
    pearson: float
    spearman: float
    mean_abs_diff: float
    mean_ours: float
    mean_theirs: float


def _rank(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # average ties
    for v in np.unique(values):
        mask = values == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def _pearson(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def agreement(
    metric: str, ours: Mapping[str, float | None], theirs: Mapping[str, float | None]
) -> Agreement:
    pairs = [
        (o, t)
        for sid, o in ours.items()
        if o is not None
        and not math.isnan(o)
        and (t := theirs.get(sid)) is not None
        and not math.isnan(t)
    ]
    if not pairs:
        return Agreement(metric, 0, math.nan, math.nan, math.nan, math.nan, math.nan)
    a = np.asarray([p[0] for p in pairs], dtype=np.float64)
    b = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return Agreement(
        metric=metric,
        n=len(pairs),
        pearson=_pearson(a, b),
        spearman=_pearson(_rank(a), _rank(b)),
        mean_abs_diff=float(np.abs(a - b).mean()),
        mean_ours=float(a.mean()),
        mean_theirs=float(b.mean()),
    )


def render_agreement(rows: Sequence[Agreement]) -> str:
    lines = [
        "| Metric | n | Ours (mean) | ragas (mean) | Mean abs diff | Pearson r | Spearman rho |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.metric} | {r.n} | {r.mean_ours:.3f} | {r.mean_theirs:.3f} | "
            f"{r.mean_abs_diff:.3f} | {r.pearson:.2f} | {r.spearman:.2f} |"
        )
    return "\n".join(lines) + "\n"
