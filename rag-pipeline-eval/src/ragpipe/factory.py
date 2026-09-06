"""Builds concrete components from ``Settings`` (the only place that knows every backend)."""

from __future__ import annotations

from pathlib import Path

from ragpipe.chunking import Chunker, FixedWindowChunker, RecursiveChunker
from ragpipe.config import LLMSettings, Settings
from ragpipe.embeddings import CachedEmbedder, Embedder, HashEmbedder, SentenceTransformerEmbedder
from ragpipe.llm import LLM, FakeLLM, HFLocalLLM, OpenAICompatibleLLM
from ragpipe.pipeline import IndexBundle, RAGPipeline
from ragpipe.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RerankingRetriever,
    Retriever,
)


def build_chunker(settings: Settings) -> Chunker:
    if settings.chunker == "fixed":
        return FixedWindowChunker(
            window_words=settings.chunk_words, overlap_words=settings.chunk_overlap_words
        )
    return RecursiveChunker(max_words=settings.chunk_words)


def build_embedder(settings: Settings) -> Embedder:
    inner: Embedder
    if settings.embedder == "hash":
        inner = HashEmbedder(dim=settings.hash_dim, seed=settings.seed)
    else:
        inner = SentenceTransformerEmbedder(settings.embedding_model)
    if settings.embedding_cache is not None:
        Path(settings.embedding_cache).parent.mkdir(parents=True, exist_ok=True)
        return CachedEmbedder(inner, settings.embedding_cache)
    return inner


def build_llm(cfg: LLMSettings, *, seed: int | None = None) -> LLM:
    if cfg.kind == "fake":
        return FakeLLM()
    if cfg.kind == "openai":
        return OpenAICompatibleLLM(
            cfg.base_url,
            cfg.model,
            api_key_env=cfg.api_key_env,
            api_style=cfg.api_style,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
    return HFLocalLLM(cfg.model, device=cfg.device, seed=seed)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.rerank:
        return None
    return CrossEncoderReranker(settings.reranker_model)


def build_retriever(
    bundle: IndexBundle, settings: Settings, *, reranker: Reranker | None = None
) -> Retriever:
    bm25 = BM25Retriever(bundle.bm25)
    dense = DenseRetriever(bundle.embedder, bundle.store)
    base: Retriever
    if settings.retriever == "bm25":
        base = bm25
    elif settings.retriever == "dense":
        base = dense
    else:
        base = HybridRetriever(
            [bm25, dense],
            fusion=settings.fusion,
            rrf_k=settings.rrf_k,
            weights=[1.0 - settings.dense_weight, settings.dense_weight],
            candidate_k=settings.candidate_k,
        )
    if reranker is None:
        return base
    chunk_map = bundle.chunk_map
    return RerankingRetriever(
        base, reranker, lambda cid: chunk_map[cid].text, candidate_k=settings.candidate_k
    )


def build_pipeline(
    bundle: IndexBundle, settings: Settings, llm: LLM, *, reranker: Reranker | None = None
) -> RAGPipeline:
    retriever = build_retriever(bundle, settings, reranker=reranker)
    return RAGPipeline(
        retriever,
        bundle.chunk_map,
        llm,
        top_k=settings.top_k,
        max_tokens=settings.generator.max_tokens,
        temperature=settings.generator.temperature,
    )
