"""``ragpipe {ingest,query,eval,gate,compare,retrieval-bench}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragpipe.config import Settings
from ragpipe.documents import load_corpus
from ragpipe.evaluation.dataset import check_against_corpus, load_evalset
from ragpipe.evaluation.gates import (
    GateSpec,
    all_passed,
    compare_reports,
    evaluate_gates,
    render_comparison,
    render_gates,
)
from ragpipe.evaluation.judge import Judge
from ragpipe.evaluation.retrieval_bench import bench_retrievers, render_bench_markdown
from ragpipe.evaluation.runner import (
    EvalConfig,
    SampleRecord,
    load_report,
    run_evaluation,
    save_report,
)
from ragpipe.factory import (
    build_chunker,
    build_embedder,
    build_llm,
    build_pipeline,
    build_reranker,
    build_retriever,
)
from ragpipe.llm import LLM
from ragpipe.logging_utils import configure_logging, log_event
from ragpipe.pipeline import IndexBundle, build_index
from ragpipe.retrieval import Retriever
from ragpipe.vectorstore import SqliteVectorStore

log = logging.getLogger("ragpipe")


# ----- settings from args -------------------------------------------------------------------

_TOP_LEVEL = (
    "chunker",
    "chunk_words",
    "embedder",
    "embedding_model",
    "retriever",
    "fusion",
    "top_k",
    "candidate_k",
    "seed",
)


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    for key in _TOP_LEVEL:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "rerank", False):
        overrides["rerank"] = True
    for role in ("generator", "judge"):
        nested: dict[str, Any] = {}
        for field_name, arg_name in (
            ("kind", role),
            ("model", f"{role}_model"),
            ("base_url", f"{role}_base_url"),
            ("max_tokens", f"{role}_max_tokens"),
            ("temperature", f"{role}_temperature"),
            ("device", f"{role}_device"),
        ):
            value = getattr(args, arg_name, None)
            if value is not None and value != "same":
                nested[field_name] = value
        if nested:
            overrides[role] = nested
    return Settings(**overrides)


def _add_model_args(parser: argparse.ArgumentParser, role: str, *, allow_same: bool) -> None:
    choices = ["fake", "openai", "hf"] + (["same"] if allow_same else [])
    parser.add_argument(f"--{role}", choices=choices, default=None)
    parser.add_argument(f"--{role}-model", dest=f"{role}_model", default=None)
    parser.add_argument(f"--{role}-base-url", dest=f"{role}_base_url", default=None)
    parser.add_argument(f"--{role}-max-tokens", dest=f"{role}_max_tokens", type=int, default=None)
    parser.add_argument(
        f"--{role}-temperature", dest=f"{role}_temperature", type=float, default=None
    )
    parser.add_argument(f"--{role}-device", dest=f"{role}_device", default=None)


def _add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default=None)
    parser.add_argument("--fusion", choices=["rrf", "convex"], default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--candidate-k", dest="candidate_k", type=int, default=None)
    parser.add_argument("--rerank", action="store_true")


def _add_embedder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedder", choices=["hash", "sentence-transformers"], default=None)
    parser.add_argument("--embedding-model", dest="embedding_model", default=None)


# ----- commands -----------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    docs = load_corpus(args.corpus)
    if not docs:
        log.error("no documents found in %s", args.corpus)
        return 2
    index_dir = Path(args.index)
    index_dir.mkdir(parents=True, exist_ok=True)
    vec_path = index_dir / "vectors.sqlite"
    if vec_path.exists():
        vec_path.unlink()
    embedder = build_embedder(settings)
    store = SqliteVectorStore(vec_path)
    bundle = build_index(
        docs,
        build_chunker(settings),
        embedder,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        store=store,
    )
    bundle.manifest["vector_path"] = str(vec_path.resolve())
    bundle.save(index_dir)
    store.close()
    log_event(
        log,
        "ingested",
        docs=len(docs),
        chunks=len(bundle.chunks),
        embedder=embedder.name,
        index=str(index_dir),
    )
    sys.stdout.write(
        f"indexed {len(docs)} documents → {len(bundle.chunks)} chunks "
        f"({embedder.name}) at {index_dir}\n"
    )
    return 0


def _load_bundle(args: argparse.Namespace, settings: Settings) -> IndexBundle:
    return IndexBundle.load(args.index, build_embedder(settings))


def cmd_query(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    bundle = _load_bundle(args, settings)
    llm = build_llm(settings.generator, seed=settings.seed)
    pipeline = build_pipeline(bundle, settings, llm, reranker=build_reranker(settings))
    result = pipeline.answer(args.question)
    if args.show_contexts:
        for i, c in enumerate(result.contexts, start=1):
            sources = ", ".join(f"{k}={v:.3f}" for k, v in c.hit.sources.items())
            sys.stdout.write(f"[{i}] {c.chunk.chunk_id} ({sources})\n{c.chunk.text}\n\n")
    sys.stdout.write(result.answer + "\n")
    sys.stdout.write(
        f"-- retriever={pipeline.retriever_name} model={result.model} "
        f"cited={result.cited_chunk_ids} abstained={result.abstained} "
        f"retrieve={result.timings_s['retrieve'] * 1000:.0f}ms "
        f"generate={result.timings_s['generate'] * 1000:.0f}ms\n"
    )
    return 0


def _progress(i: int, n: int, record: SampleRecord) -> None:
    log_event(
        log,
        "sample_done",
        i=i,
        n=n,
        id=record.id,
        abstained=record.abstained,
        metrics={k: v for k, v in record.metrics.items() if v is not None},
    )


def cmd_eval(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    bundle = _load_bundle(args, settings)
    samples = load_evalset(args.evalset)
    problems = check_against_corpus(samples, {c.doc_id for c in bundle.chunks})
    if problems:
        for p in problems:
            log.error(p)
        return 2
    if args.limit:
        samples = samples[: args.limit]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    cfg = EvalConfig(
        k=settings.top_k,
        metrics=metrics,
        n_bootstrap=args.bootstrap,
        seed=settings.seed,
    )
    generator = build_llm(settings.generator, seed=settings.seed)
    judge_llm: LLM | None = None
    if "ragas" in metrics:
        judge_llm = generator if args.judge in (None, "same") else build_llm(settings.judge)
    judge = Judge(judge_llm, max_tokens=settings.judge.max_tokens) if judge_llm else None
    pipeline = build_pipeline(bundle, settings, generator, reranker=build_reranker(settings))
    report = run_evaluation(
        pipeline,
        samples,
        cfg,
        judge=judge,
        embedder=bundle.embedder,
        generator_name=generator.name,
        progress=_progress,
    )
    json_path, md_path = save_report(report, args.out)
    sys.stdout.write(md_path.read_text(encoding="utf-8"))
    sys.stdout.write(f"\nwrote {json_path} and {md_path}\n")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    report = load_report(args.report)
    spec = GateSpec.load(args.gates)
    outcomes = evaluate_gates(report, spec)
    sys.stdout.write(render_gates(outcomes))
    ok = all_passed(outcomes)
    sys.stdout.write("GATES PASSED\n" if ok else "GATES FAILED\n")
    return 0 if ok else 1


def cmd_compare(args: argparse.Namespace) -> int:
    candidate = load_report(args.candidate)
    baseline = load_report(args.baseline)
    metrics = (
        [m.strip() for m in args.metrics.split(",")] if args.metrics else list(candidate.summaries)
    )
    rows = [
        compare_reports(
            candidate,
            baseline,
            m,
            n_boot=args.bootstrap,
            seed=args.seed,
            non_inferiority_margin=args.margin,
        )
        for m in metrics
    ]
    sys.stdout.write(render_comparison(rows))
    worse = [r.metric for r in rows if r.verdict == "worse"]
    if worse:
        sys.stdout.write(f"REGRESSION on: {', '.join(worse)}\n")
        return 1
    return 0


def cmd_retrieval_bench(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    docs = load_corpus(args.corpus)
    samples = load_evalset(args.evalset)
    embedder = build_embedder(settings)
    bundle = build_index(
        docs, build_chunker(settings), embedder, bm25_k1=settings.bm25_k1, bm25_b=settings.bm25_b
    )
    reranker = build_reranker(settings)
    configs: dict[str, Retriever] = {}
    for kind in ("bm25", "dense", "hybrid"):
        for fusion in ("rrf", "convex") if kind == "hybrid" else ("rrf",):
            s = settings.model_copy(update={"retriever": kind, "fusion": fusion, "rerank": False})
            configs[build_retriever(bundle, s).name] = build_retriever(bundle, s)
    if reranker is not None:
        s = settings.model_copy(update={"retriever": "hybrid", "fusion": "rrf"})
        r = build_retriever(bundle, s, reranker=reranker)
        configs[r.name] = r
    rows = bench_retrievers(
        bundle, configs, samples, k=settings.top_k, n_boot=args.bootstrap, seed=settings.seed
    )
    md = render_bench_markdown(rows, k=settings.top_k)
    sys.stdout.write(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps([r.to_dict() for r in rows], indent=2), encoding="utf-8"
        )
    return 0


def cmd_ragas_crosscheck(args: argparse.Namespace) -> int:
    """Score an existing report's records with the official ragas package and report the
    agreement with our implementation (same judge model, same embeddings)."""
    from ragpipe.evaluation.ragas_adapter import (
        OFFICIAL_METRICS,
        agreement,
        render_agreement,
        run_official_ragas,
    )

    settings = settings_from_args(args)
    report = load_report(args.report)
    embedder = build_embedder(settings)
    judge_llm = build_llm(settings.judge, seed=settings.seed)
    theirs = run_official_ragas(
        report.records, judge_llm, embedder, max_tokens=settings.judge.max_tokens
    )
    rows = [agreement(m, report.metric_values(m), theirs[m]) for m in OFFICIAL_METRICS]
    md = render_agreement(rows)
    sys.stdout.write(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(theirs, indent=2), encoding="utf-8")
    return 0


# ----- parser -------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragpipe", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--plain-logs", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk, index and embed a corpus directory")
    p.add_argument("--corpus", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--chunker", choices=["fixed", "recursive"], default=None)
    p.add_argument("--chunk-words", dest="chunk_words", type=int, default=None)
    _add_embedder_args(p)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="answer one question against an index")
    p.add_argument("question")
    p.add_argument("--index", required=True)
    p.add_argument("--show-contexts", action="store_true")
    _add_embedder_args(p)
    _add_retrieval_args(p)
    _add_model_args(p, "generator", allow_same=False)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("eval", help="run an evaluation set and write report.json/report.md")
    p.add_argument("--index", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metrics", default="retrieval,lexical,ragas")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=1000)
    _add_embedder_args(p)
    _add_retrieval_args(p)
    _add_model_args(p, "generator", allow_same=False)
    _add_model_args(p, "judge", allow_same=True)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("gate", help="apply quality gates to a report (exit 1 on failure)")
    p.add_argument("--report", required=True)
    p.add_argument("--gates", required=True)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("compare", help="paired comparison of two reports")
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--metrics", default="")
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("retrieval-bench", help="compare retriever configs (no LLM)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--chunker", choices=["fixed", "recursive"], default=None)
    p.add_argument("--chunk-words", dest="chunk_words", type=int, default=None)
    _add_embedder_args(p)
    _add_retrieval_args(p)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_retrieval_bench)

    p = sub.add_parser("ragas-crosscheck", help="re-score a report with the official ragas package")
    p.add_argument("--report", required=True)
    p.add_argument("--out", default="")
    _add_embedder_args(p)
    _add_model_args(p, "judge", allow_same=False)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_ragas_crosscheck)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_lines=not args.plain_logs)
    func: Any = args.func
    return int(func(args))
