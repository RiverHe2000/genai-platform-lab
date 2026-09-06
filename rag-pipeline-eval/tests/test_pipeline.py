from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ragpipe.chunking import RecursiveChunker
from ragpipe.config import Settings
from ragpipe.documents import Document
from ragpipe.embeddings import HashEmbedder
from ragpipe.factory import build_pipeline, build_retriever
from ragpipe.llm import FakeLLM
from ragpipe.pipeline import IndexBundle, IndexMismatchError, RAGPipeline, build_index
from ragpipe.retrieval import BM25Retriever, Hit
from ragpipe.vectorstore import SqliteVectorStore


def test_answer_cites_contexts_and_records_timings(bundle: IndexBundle) -> None:
    llm = FakeLLM(rules=[(r"lenders mortgage insurance", "The maximum LVR is 80% [1], see [9].")])
    pipeline = RAGPipeline(BM25Retriever(bundle.bm25), bundle.chunk_map, llm, top_k=3)
    result = pipeline.answer("What is the maximum LVR without lenders mortgage insurance?")
    assert result.abstained is False
    # BM25 only returns chunks with a positive score: one chunk mentions LMI
    assert 1 <= len(result.contexts) <= 3
    assert result.cited_chunk_ids == [result.contexts[0].chunk.chunk_id]  # [9] is out of range
    assert result.retrieved_doc_ids[0] == "mortgages"
    assert set(result.timings_s) == {"retrieve", "generate", "total"}
    assert result.usage["prompt_tokens"] > 0
    assert "[1] (Residential Mortgages)" in result.prompt
    assert llm.calls[0].system is not None
    assert result.model == "fake"


def test_answer_abstains_by_default(bundle: IndexBundle) -> None:
    pipeline = RAGPipeline(BM25Retriever(bundle.bm25), bundle.chunk_map, FakeLLM(), top_k=2)
    result = pipeline.answer("Who is the CRO?")
    assert result.abstained is True
    assert result.cited_chunk_ids == []


def test_retrieve_rejects_unknown_chunk(bundle: IndexBundle) -> None:
    class Bogus:
        name = "bogus"

        def retrieve(self, _query: str, _k: int) -> list[Hit]:
            return [Hit(chunk_id="nope", score=1.0, rank=1, sources={})]

    pipeline = RAGPipeline(Bogus(), bundle.chunk_map, FakeLLM())
    with pytest.raises(KeyError):
        pipeline.retrieve("q")


def test_build_index_manifest_and_save_load_roundtrip(
    docs: list[Document], embedder: HashEmbedder, tmp_path: Path
) -> None:
    bundle = build_index(docs, RecursiveChunker(max_words=40), embedder, embed_batch_size=3)
    assert bundle.manifest["n_docs"] == 4
    assert bundle.manifest["n_chunks"] == len(bundle.chunks)
    assert bundle.manifest["embedder"] == "hash-256"
    assert len(bundle.store) == len(bundle.chunks)

    out = bundle.save(tmp_path / "index")
    assert {p.name for p in out.iterdir()} == {
        "chunks.jsonl",
        "bm25.json",
        "vectors.sqlite",
        "manifest.json",
    }
    loaded = IndexBundle.load(out, HashEmbedder(dim=256))
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in bundle.chunks]
    assert loaded.store.ids() == bundle.store.ids()
    q = embedder.embed_queries(["VaR limit"])[0]
    assert [h.id for h in loaded.store.search(q, 3)] == [h.id for h in bundle.store.search(q, 3)]
    assert loaded.bm25.search(["lvr"], 1) == bundle.bm25.search(["lvr"], 1)

    # saving again to the same directory (with the sqlite store already there) is idempotent
    loaded.save(out)
    assert len(IndexBundle.load(out, HashEmbedder(dim=256)).store) == len(bundle.chunks)


def test_load_detects_embedder_mismatch_and_missing(tmp_path: Path, bundle: IndexBundle) -> None:
    out = bundle.save(tmp_path / "idx")
    with pytest.raises(IndexMismatchError):
        IndexBundle.load(out, HashEmbedder(dim=128))
    with pytest.raises(FileNotFoundError):
        IndexBundle.load(tmp_path / "missing", HashEmbedder(dim=256))


def test_build_index_with_external_store_and_empty_corpus(
    docs: list[Document], tmp_path: Path
) -> None:
    store = SqliteVectorStore(tmp_path / "v.sqlite")
    bundle = build_index(docs, RecursiveChunker(), HashEmbedder(dim=64), store=store)
    assert len(store) == len(bundle.chunks)
    with pytest.raises(ValueError, match="no chunks"):
        build_index([Document("e", "E", "   ")], RecursiveChunker(), HashEmbedder(dim=64))


@pytest.mark.parametrize(
    ("retriever", "fusion", "rerank", "expected_prefix"),
    [
        ("bm25", "rrf", False, "bm25"),
        ("dense", "rrf", False, "dense[hash-256]"),
        ("hybrid", "rrf", False, "hybrid[rrf]"),
        ("hybrid", "convex", False, "hybrid[convex]"),
    ],
)
def test_factory_builds_each_retriever(
    bundle: IndexBundle, retriever: str, fusion: str, rerank: bool, expected_prefix: str
) -> None:
    settings = Settings(retriever=retriever, fusion=fusion, rerank=rerank, top_k=2, candidate_k=4)
    r = build_retriever(bundle, settings)
    assert r.name.startswith(expected_prefix)
    hits = r.retrieve("liquidity coverage ratio trigger", 2)
    assert len(hits) == 2
    assert bundle.chunk_map[hits[0].chunk_id].doc_id == "liquidity"


def test_factory_pipeline_with_reranker(bundle: IndexBundle) -> None:
    class Reverse:
        name = "reverse"

        def score(self, _query: str, passages: Sequence[str]) -> list[float]:
            return [float(i) for i in range(len(passages))]

    settings = Settings(retriever="hybrid", top_k=2, candidate_k=3)
    pipeline = build_pipeline(bundle, settings, FakeLLM(), reranker=Reverse())
    assert pipeline.retriever_name.endswith("+rerank")
    assert len(pipeline.retrieve("privacy breach")) == 2
