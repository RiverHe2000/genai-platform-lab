"""Vector stores with identical semantics behind one protocol.

Exact cosine search over an in-memory matrix is the right tool at this scale (thousands of
chunks): it is deterministic, needs no tuning, and is trivially verifiable. The SQLite store
adds durability with the same query path, so the two are property-tested for equivalence.
When a corpus outgrows RAM the protocol is the seam for an ANN backend (FAISS, pgvector,
Qdrant) — see the README for the trade-offs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VectorHit:
    id: str
    score: float


class VectorStore(Protocol):
    def add(self, ids: Sequence[str], vectors: NDArray[np.float32]) -> None: ...

    def search(self, query: NDArray[np.float32], k: int) -> list[VectorHit]: ...

    def ids(self) -> list[str]: ...

    def vector(self, id: str) -> NDArray[np.float32]: ...

    def __len__(self) -> int: ...


def top_k_cosine(
    matrix: NDArray[np.float32], ids: Sequence[str], query: NDArray[np.float32], k: int
) -> list[VectorHit]:
    """Top-k by dot product (cosine for unit vectors). Stable sort → ties resolve by
    insertion order, so results never depend on argpartition internals."""
    if k <= 0 or len(ids) == 0:
        return []
    q = np.asarray(query, dtype=np.float32)
    if q.ndim != 1 or q.shape[0] != matrix.shape[1]:
        msg = f"query must have shape ({matrix.shape[1]},), got {q.shape}"
        raise ValueError(msg)
    scores = matrix @ q
    order = np.argsort(-scores, kind="stable")[: min(k, len(ids))]
    return [VectorHit(id=ids[i], score=float(scores[i])) for i in order]


def _validate(ids: Sequence[str], vectors: NDArray[np.float32], dim: int | None) -> int:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(ids):
        msg = f"vectors must have shape ({len(ids)}, d), got {arr.shape}"
        raise ValueError(msg)
    if dim is not None and arr.shape[1] != dim:
        msg = f"dimension mismatch: store has {dim}, got {arr.shape[1]}"
        raise ValueError(msg)
    if len(set(ids)) != len(ids):
        msg = "ids within one add() call must be unique"
        raise ValueError(msg)
    return int(arr.shape[1])


class NumpyVectorStore:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: NDArray[np.float32] | None = None

    def add(self, ids: Sequence[str], vectors: NDArray[np.float32]) -> None:
        dim = self._matrix.shape[1] if self._matrix is not None else None
        _validate(ids, vectors, dim)
        dupes = [i for i in ids if i in self._index]
        if dupes:
            msg = f"ids already present: {dupes[:3]}"
            raise ValueError(msg)
        arr = np.asarray(vectors, dtype=np.float32)
        self._matrix = arr.copy() if self._matrix is None else np.vstack([self._matrix, arr])
        for i in ids:
            self._index[i] = len(self._ids)
            self._ids.append(i)

    def search(self, query: NDArray[np.float32], k: int) -> list[VectorHit]:
        if self._matrix is None:
            return []
        return top_k_cosine(self._matrix, self._ids, query, k)

    def ids(self) -> list[str]:
        return list(self._ids)

    def vector(self, id: str) -> NDArray[np.float32]:
        if self._matrix is None or id not in self._index:
            raise KeyError(id)
        return np.asarray(self._matrix[self._index[id]], dtype=np.float32).copy()

    def __len__(self) -> int:
        return len(self._ids)


class SqliteVectorStore:
    """Durable store: vectors as float32 blobs, loaded into a matrix on first search."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "rowid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, "
            "dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        self._conn.commit()
        self._cache: tuple[list[str], NDArray[np.float32]] | None = None

    def _dim(self) -> int | None:
        row = self._conn.execute("SELECT dim FROM vectors LIMIT 1").fetchone()
        return int(row[0]) if row else None

    def add(self, ids: Sequence[str], vectors: NDArray[np.float32]) -> None:
        dim = _validate(ids, vectors, self._dim())
        arr = np.asarray(vectors, dtype=np.float32)
        try:
            self._conn.executemany(
                "INSERT INTO vectors (id, dim, vec) VALUES (?, ?, ?)",
                [(i, dim, arr[n].tobytes()) for n, i in enumerate(ids)],
            )
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            msg = "ids already present"
            raise ValueError(msg) from exc
        self._conn.commit()
        self._cache = None

    def _load(self) -> tuple[list[str], NDArray[np.float32]]:
        if self._cache is None:
            rows = self._conn.execute("SELECT id, dim, vec FROM vectors ORDER BY rowid").fetchall()
            if not rows:
                self._cache = ([], np.zeros((0, 0), dtype=np.float32))
            else:
                ids = [str(r[0]) for r in rows]
                matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                self._cache = (ids, matrix)
        return self._cache

    def search(self, query: NDArray[np.float32], k: int) -> list[VectorHit]:
        ids, matrix = self._load()
        if not ids:
            return []
        return top_k_cosine(matrix, ids, query, k)

    def ids(self) -> list[str]:
        return list(self._load()[0])

    def vector(self, id: str) -> NDArray[np.float32]:
        row = self._conn.execute("SELECT vec FROM vectors WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise KeyError(id)
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
