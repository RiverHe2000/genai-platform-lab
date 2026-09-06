from __future__ import annotations

import math

import pytest

from ragpipe.evaluation.ragas_adapter import agreement, render_agreement, to_ragas_rows
from ragpipe.evaluation.runner import SampleRecord


def _record(sid: str, gt: str | None, contexts: list[str]) -> SampleRecord:
    return SampleRecord(
        id=sid,
        question=f"q{sid}",
        ground_truth=gt,
        tags=[],
        answerable=gt is not None,
        answer="a",
        abstained=False,
        model="m",
        gold_doc_ids=[],
        retrieved_doc_ids=[],
        retrieved_chunk_ids=[],
        cited_chunk_ids=[],
        contexts=contexts,
        metrics={},
    )


def test_to_ragas_rows_filters_unanswerable_and_empty_contexts() -> None:
    rows = to_ragas_rows(
        [_record("1", "g", ["c"]), _record("2", None, ["c"]), _record("3", "g", [])]
    )
    assert [r["id"] for r in rows] == ["1"]
    assert rows[0] == {
        "id": "1",
        "user_input": "q1",
        "response": "a",
        "retrieved_contexts": ["c"],
        "reference": "g",
    }


def test_agreement_statistics() -> None:
    ours = {"a": 1.0, "b": 0.5, "c": 0.0, "d": None, "e": 0.7}
    theirs = {"a": 0.9, "b": 0.4, "c": -0.1, "d": 0.3, "z": 1.0}
    a = agreement("faithfulness", ours, theirs)
    assert a.n == 3
    assert a.pearson == pytest.approx(1.0)
    assert a.spearman == pytest.approx(1.0)
    assert a.mean_abs_diff == pytest.approx(0.1)
    assert a.mean_ours == pytest.approx(0.5)
    assert "| faithfulness | 3 |" in render_agreement([a])


def test_agreement_edge_cases() -> None:
    empty = agreement("m", {"a": None}, {"a": 1.0})
    assert empty.n == 0 and math.isnan(empty.pearson)
    constant = agreement("m", {"a": 1.0, "b": 1.0}, {"a": 0.2, "b": 0.9})
    assert math.isnan(constant.pearson)
    ties = agreement("m", {"a": 0.5, "b": 0.5, "c": 1.0}, {"a": 0.1, "b": 0.1, "c": 0.9})
    assert ties.spearman == pytest.approx(1.0)


def test_official_ragas_adapter_builds_langchain_wrappers() -> None:
    pytest.importorskip("ragas")
    from ragpipe.embeddings import HashEmbedder
    from ragpipe.evaluation.ragas_adapter import _build_langchain_embeddings, _build_langchain_llm
    from ragpipe.llm import FakeLLM

    chat = _build_langchain_llm(FakeLLM(default="hello"), max_tokens=8, temperature=0.0)
    result = chat.invoke("ping")
    assert result.content == "hello"
    emb = _build_langchain_embeddings(HashEmbedder(dim=32))
    assert len(emb.embed_query("x")) == 32
    assert len(emb.embed_documents(["x", "y"])) == 2
