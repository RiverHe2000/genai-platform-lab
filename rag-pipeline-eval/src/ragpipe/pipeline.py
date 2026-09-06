"""Index bundle (persisted artefacts) and the RAG pipeline itself."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragpipe import __version__
from ragpipe.bm25 import BM25Index
from ragpipe.chunking import Chunker, chunk_documents
from ragpipe.documents import Chunk, Document, corpus_fingerprint, read_jsonl, write_jsonl
from ragpipe.embeddings import Embedder
from ragpipe.llm import LLM
from ragpipe.prompts import SYSTEM_PROMPT, build_qa_prompt, extract_citations, is_abstention
from ragpipe.retrieval import Hit, Retriever
from ragpipe.textproc import tokenize
from ragpipe.vectorstore import NumpyVectorStore, SqliteVectorStore, VectorStore

CHUNKS_FILE = "chunks.jsonl"
BM25_FILE = "bm25.json"
VECTORS_FILE = "vectors.sqlite"
MANIFEST_FILE = "manifest.json"


class IndexMismatchError(RuntimeError):
    """The persisted index was built with different components than the ones supplied."""


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    bm25: BM25Index
    store: VectorStore
    embedder: Embedder
    manifest: dict[str, Any]
    _chunk_map: dict[str, Chunk] = field(default_factory=dict, repr=False)

    @property
    def chunk_map(self) -> dict[str, Chunk]:
        if not self._chunk_map:
            self._chunk_map = {c.chunk_id: c for c in self.chunks}
        return self._chunk_map

    def save(self, directory: Path | str) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / CHUNKS_FILE, (c.to_dict() for c in self.chunks))
        (out / BM25_FILE).write_text(json.dumps(self.bm25.to_dict()), encoding="utf-8")
        vec_path = out / VECTORS_FILE
        if self.manifest.get("vector_path") != str(vec_path.resolve()):
            if vec_path.exists():
                vec_path.unlink()
            durable = SqliteVectorStore(vec_path)
            ids = self.store.ids()
            if ids:
                import numpy as np

                durable.add(ids, np.stack([self.store.vector(i) for i in ids]))
            durable.close()
        (out / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: Path | str, embedder: Embedder) -> IndexBundle:
        src = Path(directory)
        manifest_path = src / MANIFEST_FILE
        if not manifest_path.exists():
            msg = f"no index at {src}"
            raise FileNotFoundError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedder") != embedder.name:
            msg = (
                f"index was built with embedder {manifest.get('embedder')!r}, "
                f"but {embedder.name!r} was supplied"
            )
            raise IndexMismatchError(msg)
        chunks = [Chunk.from_dict(row) for row in read_jsonl(src / CHUNKS_FILE)]
        bm25 = BM25Index.from_dict(json.loads((src / BM25_FILE).read_text(encoding="utf-8")))
        store = SqliteVectorStore(src / VECTORS_FILE)
        manifest["vector_path"] = str((src / VECTORS_FILE).resolve())
        return cls(chunks=chunks, bm25=bm25, store=store, embedder=embedder, manifest=manifest)


def build_index(
    documents: Sequence[Document],
    chunker: Chunker,
    embedder: Embedder,
    *,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    store: VectorStore | None = None,
    embed_batch_size: int = 64,
) -> IndexBundle:
    chunks = chunk_documents(list(documents), chunker)
    if not chunks:
        msg = "corpus produced no chunks"
        raise ValueError(msg)
    bm25 = BM25Index.build(
        [c.chunk_id for c in chunks], [tokenize(c.text) for c in chunks], k1=bm25_k1, b=bm25_b
    )
    vstore: VectorStore = store if store is not None else NumpyVectorStore()
    for i in range(0, len(chunks), embed_batch_size):
        batch = chunks[i : i + embed_batch_size]
        vstore.add([c.chunk_id for c in batch], embedder.embed_documents([c.text for c in batch]))
    manifest: dict[str, Any] = {
        "version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder": embedder.name,
        "dim": embedder.dim,
        "n_docs": len(documents),
        "n_chunks": len(chunks),
        "corpus_fingerprint": corpus_fingerprint(documents),
        "chunker": repr(chunker),
        "bm25": {"k1": bm25_k1, "b": bm25_b},
    }
    return IndexBundle(chunks=chunks, bm25=bm25, store=vstore, embedder=embedder, manifest=manifest)


# ----- pipeline -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    chunk: Chunk
    hit: Hit


@dataclass(slots=True)
class RAGResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    cited_chunk_ids: list[str]
    abstained: bool
    model: str
    prompt: str
    timings_s: dict[str, float]
    usage: dict[str, int]

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [c.chunk.chunk_id for c in self.contexts]

    @property
    def retrieved_doc_ids(self) -> list[str]:
        seen: list[str] = []
        for c in self.contexts:
            if c.chunk.doc_id not in seen:
                seen.append(c.chunk.doc_id)
        return seen

    @property
    def context_texts(self) -> list[str]:
        return [c.chunk.text for c in self.contexts]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        chunks: Mapping[str, Chunk],
        llm: LLM,
        *,
        top_k: int = 5,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._retriever = retriever
        self._chunks = chunks
        self._llm = llm
        self._top_k = top_k
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt

    @property
    def retriever_name(self) -> str:
        return self._retriever.name

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedContext]:
        hits = self._retriever.retrieve(question, k or self._top_k)
        out: list[RetrievedContext] = []
        for h in hits:
            chunk = self._chunks.get(h.chunk_id)
            if chunk is None:
                msg = f"retriever returned unknown chunk id {h.chunk_id!r}"
                raise KeyError(msg)
            out.append(RetrievedContext(chunk=chunk, hit=h))
        return out

    def answer(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        contexts = self.retrieve(question)
        t1 = time.perf_counter()
        prompt = build_qa_prompt(question, [(c.chunk.title, c.chunk.text) for c in contexts])
        response = self._llm.complete(
            prompt,
            system=self._system_prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        t2 = time.perf_counter()
        cited = [
            contexts[n - 1].chunk.chunk_id
            for n in extract_citations(response.text)
            if 1 <= n <= len(contexts)
        ]
        usage = {
            "prompt_tokens": response.prompt_tokens or 0,
            "completion_tokens": response.completion_tokens or 0,
        }
        return RAGResult(
            question=question,
            answer=response.text.strip(),
            contexts=contexts,
            cited_chunk_ids=cited,
            abstained=is_abstention(response.text),
            model=response.model,
            prompt=prompt,
            timings_s={"retrieve": t1 - t0, "generate": t2 - t1, "total": t2 - t0},
            usage=usage,
        )
