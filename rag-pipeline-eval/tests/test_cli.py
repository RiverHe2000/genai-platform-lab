from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragpipe.cli import main

EVALSET = [
    {
        "id": "e1",
        "question": "What is the maximum LVR without lenders mortgage insurance?",
        "ground_truth": "80%",
        "gold_doc_ids": ["mortgages"],
        "tags": ["credit"],
    },
    {
        "id": "e2",
        "question": "What is the foreign exchange desk VaR limit?",
        "ground_truth": "AUD 2 million",
        "gold_doc_ids": ["var"],
        "tags": ["market"],
    },
    {
        "id": "e3",
        "question": "Who is the CRO?",
        "ground_truth": None,
        "gold_doc_ids": [],
        "tags": ["unanswerable"],
    },
]


@pytest.fixture
def evalset(tmp_path: Path) -> Path:
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in EVALSET) + "\n", encoding="utf-8")
    return path


def test_ingest_query_eval_gate_compare_flow(
    corpus_dir: Path, evalset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = tmp_path / "index"
    assert (
        main(
            [
                "--plain-logs",
                "ingest",
                "--corpus",
                str(corpus_dir),
                "--index",
                str(index),
                "--embedder",
                "hash",
            ]
        )
        == 0
    )
    assert (index / "manifest.json").exists()
    assert "indexed 4 documents" in capsys.readouterr().out

    assert (
        main(
            [
                "--plain-logs",
                "query",
                "--index",
                str(index),
                "--show-contexts",
                "--top-k",
                "2",
                "What is the LCR trigger?",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "I don't know" in out and "[1]" in out and "retriever=hybrid" in out

    run = tmp_path / "run"
    code = main(
        [
            "--plain-logs",
            "eval",
            "--index",
            str(index),
            "--evalset",
            str(evalset),
            "--out",
            str(run),
            "--metrics",
            "retrieval,lexical",
            "--generator",
            "fake",
            "--bootstrap",
            "20",
        ]
    )
    assert code == 0
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["n_samples"] == 3
    assert report["summaries"]["retrieval/hit_rate@5"]["mean"] == 1.0
    assert "faithfulness" not in report["summaries"]
    capsys.readouterr()

    gates = tmp_path / "gates.yaml"
    gates.write_text("gates:\n  - metric: retrieval/hit_rate@5\n    min: 0.9\n", encoding="utf-8")
    assert main(["gate", "--report", str(run / "report.json"), "--gates", str(gates)]) == 0
    assert "GATES PASSED" in capsys.readouterr().out
    gates.write_text("gates:\n  - metric: correct_abstention\n    max: 0.5\n", encoding="utf-8")
    assert main(["gate", "--report", str(run / "report.json"), "--gates", str(gates)]) == 1
    assert "GATES FAILED" in capsys.readouterr().out

    assert (
        main(
            [
                "compare",
                "--candidate",
                str(run / "report.json"),
                "--baseline",
                str(run / "report.json"),
                "--metrics",
                "retrieval/mrr",
                "--bootstrap",
                "10",
            ]
        )
        == 0
    )
    assert "non-inferior" in capsys.readouterr().out


def test_eval_with_ragas_metrics_and_same_judge(
    corpus_dir: Path, evalset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = tmp_path / "index"
    main(["--plain-logs", "ingest", "--corpus", str(corpus_dir), "--index", str(index)])
    run = tmp_path / "run"
    code = main(
        [
            "--plain-logs",
            "eval",
            "--index",
            str(index),
            "--evalset",
            str(evalset),
            "--out",
            str(run),
            "--generator",
            "fake",
            "--judge",
            "same",
            "--bootstrap",
            "0",
            "--limit",
            "2",
        ]
    )
    assert code == 0
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["n_samples"] == 2
    assert report["judge"] == "fake"
    # the FakeLLM abstains, so faithfulness is skipped and the judge is only asked for context metrics
    assert report["summaries"]["abstained"]["mean"] == 1.0
    assert report["judge_calls"] > 0
    capsys.readouterr()


def test_eval_rejects_unknown_gold_docs(corpus_dir: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    main(["--plain-logs", "ingest", "--corpus", str(corpus_dir), "--index", str(index)])
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "x", "question": "q", "ground_truth": "a", "gold_doc_ids": ["ghost"]})
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--plain-logs",
                "eval",
                "--index",
                str(index),
                "--evalset",
                str(bad),
                "--out",
                str(tmp_path / "r"),
            ]
        )
        == 2
    )


def test_ingest_empty_corpus_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        main(["--plain-logs", "ingest", "--corpus", str(empty), "--index", str(tmp_path / "i")])
        == 2
    )


def test_retrieval_bench_command(
    corpus_dir: Path, evalset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bench" / "bench.md"
    code = main(
        [
            "--plain-logs",
            "retrieval-bench",
            "--corpus",
            str(corpus_dir),
            "--evalset",
            str(evalset),
            "--out",
            str(out),
            "--bootstrap",
            "10",
            "--top-k",
            "3",
            "--candidate-k",
            "3",
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "| bm25 |" in text and "hybrid[rrf]" in text and "hybrid[convex]" in text
    assert out.with_suffix(".json").exists()
    assert "hit_rate@3" in capsys.readouterr().out


def test_evalset_validation_errors(tmp_path: Path) -> None:
    from ragpipe.evaluation.dataset import load_evalset

    dup = tmp_path / "dup.jsonl"
    dup.write_text(
        '{"id": "a", "question": "q"}\n{"id": "a", "question": "q2"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_evalset(dup)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_evalset(empty)
    extra = tmp_path / "extra.jsonl"
    extra.write_text('{"id": "a", "question": "q", "bogus": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_evalset(extra)


def test_ragas_crosscheck_command_uses_adapter(
    corpus_dir: Path,
    evalset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index = tmp_path / "index"
    main(["--plain-logs", "ingest", "--corpus", str(corpus_dir), "--index", str(index)])
    run = tmp_path / "run"
    main(
        [
            "--plain-logs",
            "eval",
            "--index",
            str(index),
            "--evalset",
            str(evalset),
            "--out",
            str(run),
            "--generator",
            "fake",
            "--judge",
            "same",
            "--bootstrap",
            "0",
        ]
    )
    capsys.readouterr()

    import ragpipe.evaluation.ragas_adapter as adapter

    def fake_official(
        records: object, llm: object, embedder: object, **kwargs: object
    ) -> dict[str, dict[str, float | None]]:
        return {m: {"e1": 0.5, "e2": None} for m in adapter.OFFICIAL_METRICS}

    monkeypatch.setattr(adapter, "run_official_ragas", fake_official)
    out = tmp_path / "agree" / "agreement.md"
    code = main(
        [
            "--plain-logs",
            "ragas-crosscheck",
            "--report",
            str(run / "report.json"),
            "--out",
            str(out),
            "--judge",
            "fake",
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "| faithfulness |" in text and "| context_recall |" in text
    assert (
        json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["faithfulness"]["e1"]
        == 0.5
    )
