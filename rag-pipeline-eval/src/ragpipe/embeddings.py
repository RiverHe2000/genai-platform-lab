"""Embedding backends behind one protocol.

* ``HashEmbedder`` — deterministic feature hashing of word uni/bigrams. No model download, so
  every test and the CI smoke run are fully offline. It is a *lexical* embedding: good enough to
  exercise the dense path and the metrics, not a substitute for a trained encoder.
* ``SentenceTransformerEmbedder`` — any sentence-transformers model (default BAAI/bge-small),
  with the query instruction prefix that asymmetric retrievers such as bge expect.
* ``CachedEmbedder`` — SQLite cache keyed by (model, mode, sha256(text)); re-ingesting an
  unchanged corpus or re-running an evaluation does not re-embed anything.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ragpipe.textproc import tokenize

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


def l2_normalize(matrix: NDArray[Any]) -> NDArray[np.float32]:
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2:
        msg = f"expected a 2-D array, got shape {m.shape}"
        raise ValueError(msg)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(m / norms, dtype=np.float32)


class HashEmbedder:
    """Signed feature hashing (Weinberger et al., 2009) over uni- and bigrams, sublinear tf,
    L2-normalised. blake2b rather than ``hash()`` so vectors are stable across processes."""

    def __init__(self, dim: int = 1024, *, seed: int = 0) -> None:
        if dim < 8:
            msg = "dim must be >= 8"
            raise ValueError(msg)
        self._dim = dim
        self._key = seed.to_bytes(8, "little", signed=True)

    @property
    def name(self) -> str:
        return f"hash-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        return tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]

    def _embed_one(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for feat in self._features(text):
            h = hashlib.blake2b(feat.encode(), digest_size=8, key=self._key).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            vec[idx] += 1.0 if h[4] & 1 else -1.0
        return np.sign(vec) * np.log1p(np.abs(vec))

    def _embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.stack([self._embed_one(t) for t in texts]))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embed(texts)


class SentenceEncoderLike(Protocol):
    """The slice of ``sentence_transformers.SentenceTransformer`` we rely on."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = 32,
        device: str | None = None,
        model: SentenceEncoderLike | None = None,
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._device = device
        self._model: SentenceEncoderLike | None = model
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"st[{self._model_name}]"

    def _load(self) -> SentenceEncoderLike:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            model = self._load()
            dim = model.get_sentence_embedding_dimension()
            if dim is None:
                dim = int(self._encode(["probe"]).shape[1])
            self._dim = int(dim)
        return self._dim

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        model = self._load()
        out = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self._query_prefix + t for t in texts])


class CachedEmbedder:
    """Transparent SQLite cache in front of any ``Embedder``."""

    def __init__(self, inner: Embedder, path: Path | str = ":memory:") -> None:
        self._inner = inner
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
        )
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dim(self) -> int:
        return self._inner.dim

    def _key(self, mode: str, text: str) -> str:
        return f"{self._inner.name}|{mode}|{hashlib.sha256(text.encode()).hexdigest()}"

    def _embed(self, mode: str, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [self._key(mode, t) for t in texts]
        rows = self._conn.execute(
            f"SELECT key, vec FROM embedding_cache WHERE key IN ({','.join('?' * len(keys))})",
            keys,
        ).fetchall()
        found = {str(k): np.frombuffer(v, dtype=np.float32) for k, v in rows}
        missing = sorted({k for k in keys if k not in found})
        self.hits += len(keys) - sum(1 for k in keys if k in missing)
        self.misses += len(missing)
        if missing:
            first_text = dict(zip(keys, texts, strict=True))
            batch = [first_text[k] for k in missing]
            vectors = (
                self._inner.embed_queries(batch)
                if mode == "query"
                else self._inner.embed_documents(batch)
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (key, vec) VALUES (?, ?)",
                [
                    (k, v.astype(np.float32).tobytes())
                    for k, v in zip(missing, vectors, strict=True)
                ],
            )
            self._conn.commit()
            for k, v in zip(missing, vectors, strict=True):
                found[k] = np.asarray(v, dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embed("doc", texts)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embed("query", texts)

    def close(self) -> None:
        self._conn.close()
