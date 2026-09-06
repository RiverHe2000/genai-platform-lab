from __future__ import annotations

import json

import pytest

from ragpipe.embeddings import HashEmbedder
from ragpipe.evaluation.judge import Judge
from ragpipe.evaluation.ragas_metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    exact_match,
    faithfulness,
    normalize_answer,
    reference_coverage,
    token_f1,
)
from ragpipe.llm import FakeLLM

CONTEXTS = ["The maximum LVR without LMI is 80%.", "With LMI the maximum LVR is 95%."]


def _judge(rules: list[tuple[str, str]], default: str = "{}") -> Judge:
    return Judge(FakeLLM(rules=rules, default=default), max_retries=0)


def test_faithfulness_ratio_of_supported_statements() -> None:
    judge = _judge(
        [
            (
                r"Break the ANSWER",
                json.dumps({"statements": ["Max LVR without LMI is 80%.", "Max LVR is 99%."]}),
            ),
            (
                r"For each STATEMENT",
                json.dumps(
                    {
                        "verdicts": [
                            {"id": 1, "verdict": 1, "reason": "in [1]"},
                            {"id": 2, "verdict": 0, "reason": "contradicted"},
                        ]
                    }
                ),
            ),
        ]
    )
    r = faithfulness(judge, "q", "The max LVR is 80% without LMI and 99% otherwise.", CONTEXTS)
    assert r.value == pytest.approx(0.5)
    assert r.detail["n_supported"] == 1
    assert len(r.detail["statements"]) == 2
    assert r.detail["n_unjudged"] == 0


def test_faithfulness_aligns_verdicts_by_id_and_drops_placeholders() -> None:
    judge = _judge(
        [
            (
                r"Break the ANSWER",
                json.dumps(
                    {
                        "statements": [
                            "...",
                            "The LVR cap is 80%.",
                            "  ",
                            "Investors get 90% with LMI.",
                        ]
                    }
                ),
            ),
            (
                r"STATEMENTS TO CHECK",
                json.dumps(
                    {
                        "verdicts": [
                            {"id": 2, "verdict": 1},
                            {"id": 7, "verdict": 0},
                            {"id": 2, "verdict": 0},
                        ]
                    }
                ),
            ),
        ]
    )
    r = faithfulness(judge, "q", "answer", CONTEXTS)
    assert r.detail["statements"] == ["The LVR cap is 80%.", "Investors get 90% with LMI."]
    assert r.value == 1.0  # only S2 was judged (first verdict per id wins); S1 unjudged
    assert r.detail["n_unjudged"] == 1
    no_match = _judge(
        [
            (r"Break the ANSWER", json.dumps({"statements": ["one statement here"]})),
            (r"STATEMENTS TO CHECK", json.dumps({"verdicts": [{"id": 9, "verdict": 1}]})),
        ]
    )
    assert faithfulness(no_match, "q", "answer", CONTEXTS).value is None


def test_faithfulness_missing_cases() -> None:
    assert faithfulness(_judge([]), "q", "answer", []).value is None
    assert faithfulness(_judge([]), "q", "   ", CONTEXTS).value is None
    no_statements = _judge([(r"Break the ANSWER", '{"statements": []}')])
    assert (
        faithfulness(no_statements, "q", "answer", CONTEXTS).detail["reason"]
        == "no statements extracted"
    )
    bad_judge = _judge([], default="not json")
    assert faithfulness(bad_judge, "q", "answer", CONTEXTS).value is None
    verdict_fail = _judge(
        [(r"Break the ANSWER", '{"statements": ["a statement"]}')], default="nope"
    )
    assert verdict_fail.ask is not None
    assert (
        faithfulness(verdict_fail, "q", "answer", CONTEXTS).detail["reason"]
        == "judge failed to produce verdicts"
    )


def test_answer_relevancy_uses_embeddings_and_noncommittal_flag() -> None:
    emb = HashEmbedder(dim=256)
    question = "What is the maximum LVR without lenders mortgage insurance?"
    same = _judge(
        [
            (
                r"Write 2 different questions",
                json.dumps({"questions": [question, question], "noncommittal": 0}),
            )
        ]
    )
    r = answer_relevancy(same, emb, question, "80%", n_questions=2)
    assert r.value == pytest.approx(1.0, abs=1e-5)
    assert len(r.detail["similarities"]) == 2

    different = _judge(
        [
            (
                r"questions",
                json.dumps({"questions": ["What colour is the sky?"], "noncommittal": 0}),
            )
        ]
    )
    assert (answer_relevancy(different, emb, question, "blue").value or 0.0) < 0.3

    nc = _judge([(r"questions", json.dumps({"questions": ["x"], "noncommittal": 1}))])
    assert answer_relevancy(nc, emb, question, "I don't know").value == 0.0
    # a small judge over-flagging a factual answer is overruled by the lexical check
    nc_real = _judge([(r"questions", json.dumps({"questions": [question], "noncommittal": 1}))])
    overruled = answer_relevancy(
        nc_real, emb, question, "The maximum LVR without LMI is 80 percent."
    )
    assert (overruled.value or 0.0) > 0.0
    assert overruled.detail["judge_noncommittal"] is True
    assert overruled.detail["lexical_noncommittal"] is False
    short = answer_relevancy(nc, emb, question, "Not sure.")
    assert short.value == 0.0 and short.detail["noncommittal"] is True
    assert answer_relevancy(_judge([], default="???"), emb, question, "a").value is None
    assert answer_relevancy(_judge([]), emb, question, "   ").value == 0.0


def test_context_precision_formula() -> None:
    # verdicts per context: [1, 0, 1] -> (1/1*1 + 0 + 2/3*1) / 2
    replies = iter(['{"useful": 1}', '{"useful": 0}', '{"useful": 1}'])
    judge = Judge(FakeLLM(default=lambda _p: next(replies)), max_retries=0)
    r = context_precision(judge, "q", "ref", ["c1", "c2", "c3"])
    assert r.value == pytest.approx((1.0 + 2 / 3) / 2)
    assert r.detail["verdicts"] == [1, 0, 1]
    assert context_precision(_judge([], default='{"useful": 0}'), "q", "ref", ["c"]).value == 0.0
    assert context_precision(_judge([]), "q", "ref", []).value is None
    assert context_precision(_judge([], default="x"), "q", "ref", ["c"]).value is None


def test_context_recall_ratio() -> None:
    judge = _judge(
        [
            (
                r"Split ONLY the REFERENCE",
                json.dumps(
                    {
                        "items": [
                            {"sentence": "a", "attributed": 1},
                            {"sentence": "b", "attributed": 1},
                            {"sentence": "c", "attributed": 0},
                        ]
                    }
                ),
            )
        ]
    )
    r = context_recall(judge, "q", "a. b. c.", CONTEXTS)
    assert r.value == pytest.approx(2 / 3)


def test_verdict_ids_accept_statement_labels() -> None:
    from ragpipe.evaluation.ragas_metrics import Verdict

    assert Verdict.model_validate({"id": "S2", "verdict": 1}).id == 2
    assert Verdict.model_validate({"id": " s7 ", "verdict": 0}).id == 7
    assert Verdict.model_validate({"id": 3, "verdict": 0}).id == 3
    with pytest.raises(ValueError):
        Verdict.model_validate({"id": "statement two", "verdict": 1})
    assert context_recall(_judge([]), "q", "ref", []).value is None
    assert context_recall(_judge([], default='{"items": []}'), "q", "ref", CONTEXTS).value is None


def test_lexical_proxies() -> None:
    assert normalize_answer("The LVR is 80%!") == ["lvr", "is", "80"]
    assert token_f1("the maximum LVR is 80%", "maximum LVR 80%") == pytest.approx(
        2 * (3 / 4) * 1.0 / (3 / 4 + 1.0)
    )
    assert token_f1("", "") == 1.0
    assert token_f1("abc", "") == 0.0
    assert token_f1("xyz", "abc") == 0.0
    assert exact_match("The LVR is 80%.", "lvr is 80") == 1.0
    assert exact_match("81", "80") == 0.0
    assert reference_coverage("the answer is 80% with LMI", "80% LMI") == 1.0
    assert reference_coverage("nothing", "80% LMI") == 0.0
    assert reference_coverage("anything", "") == 1.0
