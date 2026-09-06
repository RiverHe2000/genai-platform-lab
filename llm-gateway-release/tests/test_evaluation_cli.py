from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from llmgate.api import create_app
from llmgate.cli import main
from llmgate.config import GatewaySettings
from llmgate.evaluation.compare import compare_runs, mcnemar_exact, render_comparison
from llmgate.evaluation.promote import PromotionPolicy, decide, render_report, save_decision
from llmgate.evaluation.runner import CaseResult, EvalRunReport, load_report, run_suite, save_report
from llmgate.evaluation.suite import EvalCase, load_suite, normalize, score_case
from llmgate.gateway import Gateway
from tests.conftest import ROOT, make_backends, no_sleep, settings_dict

FINANCE = ROOT / "evalsets" / "finance_qa.jsonl"


def _case(kind: str, **kw: object) -> EvalCase:
    return EvalCase.model_validate({"id": "c", "prompt": "p", "kind": kind, **kw})


def test_suite_loading_and_validation(tmp_path: Path) -> None:
    cases = load_suite(FINANCE)
    assert len(cases) >= 30 and len({c.id for c in cases}) == len(cases)
    kinds = {c.kind for c in cases}
    assert kinds == {"contains", "numeric", "regex", "json_schema", "refusal"}
    with pytest.raises(ValueError, match="exactly one"):
        EvalCase.model_validate({"id": "a", "kind": "exact", "expected": "x"})
    with pytest.raises(ValueError, match="need a schema"):
        EvalCase.model_validate({"id": "a", "prompt": "p", "kind": "json_schema"})
    with pytest.raises(ValueError, match="expected value"):
        EvalCase.model_validate({"id": "a", "prompt": "p", "kind": "exact"})
    dup = tmp_path / "d.jsonl"
    dup.write_text(
        '{"id": "a", "prompt": "p", "kind": "refusal"}\n{"id": "a", "prompt": "p", "kind": "refusal"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(dup)
    empty = tmp_path / "e.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        load_suite(empty)
    bad = tmp_path / "b.jsonl"
    bad.write_text("{oops}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_suite(bad)
    msgs = EvalCase.model_validate(
        {
            "id": "m",
            "messages": [{"role": "user", "content": "q"}],
            "system": "s",
            "kind": "refusal",
        }
    )
    r = msgs.to_request("gw")
    assert [m.role for m in r.messages] == ["system", "user"] and r.temperature == 0.0


def test_scorers() -> None:
    assert normalize("The Loan-to-Value ratio!") == "loantovalue ratio"
    assert score_case(_case("exact", expected="Canberra"), "canberra.").score == 1.0
    assert score_case(_case("exact", expected="Canberra"), "Sydney").score == 0.0
    assert (
        score_case(
            _case("contains", expected=["loan-to-value", "loan to value"]),
            "LVR means loan to value ratio",
        ).score
        == 1.0
    )
    assert score_case(_case("contains", expected="x"), "nothing").score == 0.0
    assert score_case(_case("regex", expected="^\\W*yes\\b"), "Yes, it is").score == 1.0
    assert score_case(_case("regex", expected="^\\W*yes\\b"), "No").score == 0.0
    assert (
        score_case(_case("numeric", expected=30, tolerance=0.5), "The answer is 30.2").score == 1.0
    )
    assert score_case(_case("numeric", expected=30), "31").score == 0.0
    assert score_case(_case("numeric", expected=30), "none").reason == "no number in output"
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    js = _case("json_schema", schema=schema, expected={"a": 1})
    assert score_case(js, '{"a": 1}').score == 1.0
    assert score_case(js, '{"a": 2}').score == 0.5
    assert score_case(js, '{"a": "x"}').score == 0.0
    assert score_case(js, "no json").score == 0.0
    assert score_case(_case("refusal"), "I'm sorry, I can't help with that").score == 1.0
    assert score_case(_case("refusal"), "Sure, here it is").score == 0.0


def _app(keyword_answers: dict[str, str]) -> FastAPI:
    settings = GatewaySettings.model_validate(settings_dict())
    backends = make_backends(
        primary={"keyword_answers": keyword_answers}, backup={"keyword_answers": {}}
    )
    return create_app(Gateway(settings, backends, sleep=no_sleep), settings)


GOOD_ANSWERS = {
    "lvr stand": "loan-to-value ratio",
    "apra": "Australian Prudential Regulation Authority",
    "12% of 250": "30",
    "secured loan": "yes",
    "chief executive": "I'm sorry, I can't share personal information.",
    "capital city of australia": "Canberra",
    "central bank": "The Reserve Bank of Australia.",
}


async def test_run_suite_through_gateway_and_reports(tmp_path: Path) -> None:
    cases = load_suite(FINANCE)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(GOOD_ANSWERS)), base_url="http://gw"
    ) as client:
        candidate = await run_suite(
            client, cases, model="primary", target="asgi", concurrency=3, progress=lambda _r: None
        )
        baseline = await run_suite(client, cases, model="backup", target="asgi", concurrency=3)
        missing = await run_suite(client, cases[:3], model="ghost", target="asgi")
    assert candidate.n == len(cases) and candidate.n_error == 0
    assert candidate.mean_score > baseline.mean_score
    assert candidate.scores()["acr_lvr"] == 1.0 and baseline.scores()["acr_lvr"] == 0.0
    assert (
        candidate.by_tag["acronym"]["n"] > 0
        and candidate.p95_latency_ms >= candidate.p50_latency_ms
    )
    assert missing.n_error == 3 and missing.error_rate == 1.0 and missing.results[0].status == 404
    json_path, md_path = save_report(candidate, tmp_path / "cand")
    loaded = load_report(json_path)
    assert loaded.model_dump() == candidate.model_dump()
    assert "| acr_lvr | contains | 1.0 |" in md_path.read_text(encoding="utf-8")

    comparison = compare_runs(candidate, baseline, n_boot=300)
    assert comparison.verdict == "better" and comparison.wins > 0 and comparison.losses == 0
    assert comparison.mcnemar_p < 0.05 and "| better |" in render_comparison(comparison)
    same = compare_runs(candidate, candidate, n_boot=300)
    assert same.verdict == "non-inferior" and same.delta == 0.0 and same.mcnemar_p == 1.0
    worse = compare_runs(baseline, candidate, n_boot=300, non_inferiority_margin=0.01)
    assert worse.verdict == "worse"
    assert compare_runs(candidate, missing, n_boot=10).n_pairs == 3

    policy = PromotionPolicy(
        min_mean_score=0.05,
        non_inferiority_margin=0.05,
        min_cases=10,
        max_error_rate=0.0,
        max_p95_latency_ms=60_000,
        n_boot=300,
    )
    decision = decide(candidate, baseline, policy)
    assert decision.promote and all(c.passed for c in decision.checks)
    hold = decide(baseline, candidate, policy)
    assert not hold.promote and [c.name for c in hold.checks if not c.passed] == [
        "absolute quality floor",
        "quality vs baseline",
    ]
    strict = decide(
        candidate, candidate, PromotionPolicy(require_improvement=True, min_cases=1, n_boot=100)
    )
    assert not strict.promote
    report = render_report(decision, candidate, baseline, policy)
    assert "PROMOTE" in report and "## 5. Policy checks" in report and "| acronym |" in report
    md, js = save_decision(decision, report, tmp_path / "dec")
    assert json.loads(js.read_text(encoding="utf-8"))["promote"] is True and md.exists()


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(8, 0) == pytest.approx(2 / 256)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(6, 1) == pytest.approx(2 * (1 + 7) / 128)


def test_compare_edge_cases() -> None:
    def rep(scores: dict[str, float]) -> EvalRunReport:
        return EvalRunReport(
            created_at="t",
            version="0",
            model="m",
            target="t",
            n=len(scores),
            n_error=0,
            mean_score=sum(scores.values()) / max(len(scores), 1),
            error_rate=0.0,
            p50_latency_ms=1.0,
            p95_latency_ms=2.0,
            total_tokens=0,
            results=[
                CaseResult(
                    id=k,
                    tags=[],
                    kind="exact",
                    score=v,
                    reason="",
                    output="",
                    latency_ms=1.0,
                    status=200,
                )
                for k, v in scores.items()
            ],
        )

    assert compare_runs(rep({"a": 1.0}), rep({"b": 1.0})).verdict == "no pairs"
    single = compare_runs(rep({"a": 1.0}), rep({"a": 0.0}))
    assert single.n_pairs == 1 and single.delta == 1.0 and single.ci_low == single.ci_high == 1.0
    noisy = compare_runs(
        rep({f"c{i}": float(i % 2) for i in range(20)}),
        rep({f"c{i}": float((i + 1) % 2) for i in range(20)}),
        n_boot=200,
    )
    assert noisy.verdict == "inconclusive"


def test_cli_check_config_eval_promote_loadtest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cfg = str(ROOT / "deploy" / "gateway.fake.yaml")
    assert main(["--plain-logs", "check-config", "--config", fake_cfg]) == 0
    assert "fake-flaky" in capsys.readouterr().out

    base_out = tmp_path / "base"
    cand_out = tmp_path / "cand"
    assert (
        main(
            [
                "--plain-logs",
                "eval",
                "--config",
                fake_cfg,
                "--model",
                "fake-baseline",
                "--suite",
                str(FINANCE),
                "--out",
                str(base_out),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--plain-logs",
                "eval",
                "--config",
                fake_cfg,
                "--model",
                "fake-candidate",
                "--suite",
                str(FINANCE),
                "--out",
                str(cand_out),
            ]
        )
        == 0
    )
    capsys.readouterr()
    ci_policy = str(ROOT / "deploy" / "promotion_policy.ci.yaml")
    assert (
        main(
            [
                "--plain-logs",
                "promote",
                "--candidate",
                str(cand_out / "report.json"),
                "--baseline",
                str(base_out / "report.json"),
                "--policy",
                ci_policy,
                "--out",
                str(tmp_path / "prom"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "PROMOTE" in out and (tmp_path / "prom" / "promotion_report.md").exists()
    strict = str(ROOT / "deploy" / "promotion_policy.yaml")
    assert (
        main(
            [
                "--plain-logs",
                "promote",
                "--candidate",
                str(cand_out / "report.json"),
                "--baseline",
                str(base_out / "report.json"),
                "--policy",
                strict,
            ]
        )
        == 1
    )
    assert "HOLD" in capsys.readouterr().out
    assert main(
        [
            "--plain-logs",
            "promote",
            "--candidate",
            str(cand_out / "report.json"),
            "--baseline",
            str(base_out / "report.json"),
        ]
    ) in (0, 1)
    capsys.readouterr()

    assert (
        main(
            [
                "--plain-logs",
                "loadtest",
                "--config",
                fake_cfg,
                "--model",
                "fake-baseline",
                "--concurrency",
                "1",
                "2",
                "--requests",
                "4",
                "--out",
                str(tmp_path / "lt" / "load.md"),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "| in-process:gateway.fake.yaml | fake-baseline | 2 | 4 |" in text
    assert (tmp_path / "lt" / "load.json").exists()
    assert (
        main(
            [
                "--plain-logs",
                "loadtest",
                "--config",
                fake_cfg,
                "--model",
                "fake-baseline",
                "--concurrency",
                "1",
                "--requests",
                "2",
                "--stream",
            ]
        )
        == 0
    )
    capsys.readouterr()
