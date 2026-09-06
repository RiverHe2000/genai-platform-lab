from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ragpipe.embeddings import l2_normalize
from ragpipe.vectorstore import NumpyVectorStore, SqliteVectorStore, VectorStore, top_k_cosine


def _unit(n: int, d: int, seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=(n, d)))


@pytest.fixture(params=["numpy", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> VectorStore:
    if request.param == "numpy":
        return NumpyVectorStore()
    return SqliteVectorStore(tmp_path / "v.sqlite")


def test_top_k_cosine_orders_and_breaks_ties_by_insertion() -> None:
    matrix = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    hits = top_k_cosine(matrix, ["a", "b", "c"], np.array([1, 0], dtype=np.float32), k=3)
    assert [(h.id, round(h.score, 6)) for h in hits] == [("a", 1.0), ("c", 1.0), ("b", 0.0)]
    assert top_k_cosine(matrix, ["a", "b", "c"], np.array([1, 0], dtype=np.float32), k=0) == []
    with pytest.raises(ValueError):
        top_k_cosine(matrix, ["a", "b", "c"], np.zeros(3, dtype=np.float32), k=1)


def test_store_add_search_len_ids_vector(store: VectorStore) -> None:
    vecs = _unit(5, 8, seed=1)
    ids = [f"c{i}" for i in range(5)]
    store.add(ids, vecs)
    assert len(store) == 5
    assert store.ids() == ids
    np.testing.assert_allclose(store.vector("c3"), vecs[3], atol=1e-6)
    hits = store.search(vecs[2], k=3)
    assert hits[0].id == "c2"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(store.search(vecs[0], k=50)) == 5
    with pytest.raises(KeyError):
        store.vector("missing")


def test_store_rejects_duplicates_and_bad_shapes(store: VectorStore) -> None:
    store.add(["a"], _unit(1, 4, seed=0))
    with pytest.raises(ValueError):
        store.add(["a"], _unit(1, 4, seed=0))
    with pytest.raises(ValueError):
        store.add(["b"], _unit(1, 5, seed=0))
    with pytest.raises(ValueError):
        store.add(["b", "c"], _unit(1, 4, seed=0))
    with pytest.raises(ValueError):
        store.add(["b", "b"], _unit(2, 4, seed=0))
    assert len(store) == 1


def test_empty_store_search(store: VectorStore) -> None:
    assert store.search(np.zeros(4, dtype=np.float32), k=3) == []
    assert store.ids() == []


def test_numpy_and_sqlite_are_equivalent(tmp_path: Path) -> None:
    vecs = _unit(200, 32, seed=7)
    ids = [f"id{i}" for i in range(200)]
    a = NumpyVectorStore()
    b = SqliteVectorStore(tmp_path / "eq.sqlite")
    for s in (a, b):
        s.add(ids[:120], vecs[:120])
        s.add(ids[120:], vecs[120:])
    for q in _unit(25, 32, seed=9):
        ha = a.search(q, k=10)
        hb = b.search(q, k=10)
        assert [h.id for h in ha] == [h.id for h in hb]
        np.testing.assert_allclose([h.score for h in ha], [h.score for h in hb], atol=1e-6)


def test_sqlite_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "p.sqlite"
    vecs = _unit(3, 6, seed=3)
    s1 = SqliteVectorStore(path)
    s1.add(["x", "y", "z"], vecs)
    s1.close()
    s2 = SqliteVectorStore(path)
    assert s2.ids() == ["x", "y", "z"]
    assert s2.search(vecs[1], k=1)[0].id == "y"
    s2.close()
