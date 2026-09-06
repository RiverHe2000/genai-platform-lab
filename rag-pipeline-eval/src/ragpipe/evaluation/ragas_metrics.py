"""The four RAGAS metrics (Es et al., 2023) implemented from their definitions, plus cheap
deterministic answer-correctness proxies.

* **faithfulness** = supported statements / statements extracted from the answer
* **answer relevancy** = mean cosine(question, questions generated from the answer); 0 if the
  answer is non-committal
* **context precision** = Σ_k precision@k · v_k / Σ v_k, where v_k = 1 if context k is useful
  for arriving at the reference answer
* **context recall** = reference sentences attributable to the contexts / reference sentences

Prompt design notes (learned from running a 1.5 B judge): never put placeholder strings such
as ``"..."`` in the JSON template (small models copy them); number the statements and align
verdicts by id instead of by text; put the passages *before* the question so the judge reads
them; and corroborate the judge's "non-committal" flag with a lexical check because small
judges over-flag it.

Every metric returns ``MetricResult(value=None, ...)`` with a reason when the judge fails,
so missing values are visible in the report instead of being absorbed into an average.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ragpipe.embeddings import Embedder
from ragpipe.evaluation.judge import Judge
from ragpipe.prompts import is_abstention


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float | None
    detail: dict[str, Any] = field(default_factory=dict)


# ----- judge output schemas -----------------------------------------------------------------


class Statements(BaseModel):
    statements: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    id: int
    verdict: Literal[0, 1]
    reason: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: object) -> object:
        """Accept "S3", "s3" or "3" — small judges echo the statement label."""
        if isinstance(value, str):
            digits = value.strip().lstrip("Ss").strip()
            if digits.isdigit():
                return int(digits)
        return value


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(default_factory=list)
    noncommittal: int = Field(0, ge=0, le=1)


class ContextVerdict(BaseModel):
    useful: Literal[0, 1]
    reason: str = ""


class Attribution(BaseModel):
    sentence: str = ""
    attributed: Literal[0, 1]
    reason: str = ""


class Attributions(BaseModel):
    items: list[Attribution] = Field(default_factory=list)


# ----- prompts ------------------------------------------------------------------------------


def _numbered(contexts: Sequence[str]) -> str:
    return "\n\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(contexts, start=1))


def statements_prompt(question: str, answer: str) -> str:
    return (
        "Break the ANSWER into short, self-contained factual statements. Each statement must "
        "be understandable on its own (resolve pronouns). Do not add information that is not "
        "in the answer.\n\n"
        f"QUESTION: {question}\nANSWER: {answer}\n\n"
        'Return a JSON object with one key "statements" whose value is a list of strings, '
        "one string per statement."
    )


def verdicts_prompt(contexts: Sequence[str], statements: Sequence[str]) -> str:
    listed = "\n".join(f"S{i}: {s}" for i, s in enumerate(statements, start=1))
    return (
        "You will check statements against passages.\n\n"
        f"PASSAGES:\n{_numbered(contexts)}\n\n"
        f"STATEMENTS TO CHECK:\n{listed}\n\n"
        "For each STATEMENT (S1, S2, ...) decide whether it can be directly inferred from the "
        "PASSAGES. Use verdict 1 only if the passages support it; use 0 if it is contradicted "
        'or not present. Return a JSON object with one key "verdicts" whose value is a list '
        'with exactly one entry per statement, each entry an object {"id": <statement number>, '
        '"verdict": <0 or 1>, "reason": "<short reason>"}.'
    )


def questions_prompt(answer: str, n: int) -> str:
    return (
        f"Write {n} different questions that the ANSWER below would be a direct answer to. "
        "Then decide whether the answer is non-committal: non-committal means the answer "
        "refuses, says it does not know, or gives no information (for example \"I don't "
        'know"). An answer that states a fact is committal.\n\n'
        f"ANSWER: {answer}\n\n"
        'Return a JSON object with two keys: "questions" (a list of strings) and '
        '"noncommittal" (1 if the answer is non-committal, otherwise 0).'
    )


def context_precision_prompt(question: str, reference: str, context: str) -> str:
    return (
        "You will judge whether one CONTEXT passage contains the information needed to give "
        "the REFERENCE answer to the QUESTION.\n\n"
        f"CONTEXT PASSAGE:\n{context}\n\n"
        f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\n\n"
        "Does the passage state the facts given in the reference answer (fully or in part)? "
        'Return a JSON object {"useful": 1, "reason": "<short reason>"} if yes, or '
        '{"useful": 0, "reason": "<short reason>"} if the passage does not contain those facts.'
    )


def context_recall_prompt(question: str, reference: str, contexts: Sequence[str]) -> str:
    return (
        f"PASSAGES:\n{_numbered(contexts)}\n\n"
        f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\n\n"
        "Split ONLY the REFERENCE ANSWER into its sentences (do not use sentences from the "
        "passages). For each reference sentence decide whether it is supported by the "
        "PASSAGES: attributed = 1 if yes, 0 if no. Return a JSON object with one key "
        '"items" whose value is a list of objects {"sentence": "<reference sentence>", '
        '"attributed": <0 or 1>, "reason": "<short reason>"}.'
    )


# ----- metrics ------------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"^[\s.\-_*]*$")


def clean_statements(statements: Sequence[str]) -> list[str]:
    """Drop empty strings, template placeholders such as ``...`` and duplicates."""
    out: list[str] = []
    for s in statements:
        text = s.strip()
        if not text or _PLACEHOLDER_RE.match(text) or len(text.split()) < 2:
            continue
        if text not in out:
            out.append(text)
    return out


def faithfulness(judge: Judge, question: str, answer: str, contexts: Sequence[str]) -> MetricResult:
    name = "faithfulness"
    if not contexts or not answer.strip():
        return MetricResult(name, None, {"reason": "no contexts or empty answer"})
    st = judge.ask(f"{name}/statements", statements_prompt(question, answer), Statements)
    if st is None:
        return MetricResult(name, None, {"reason": "judge failed to extract statements"})
    statements = clean_statements(st.statements)
    if not statements:
        return MetricResult(name, None, {"reason": "no statements extracted"})
    vd = judge.ask(f"{name}/verdicts", verdicts_prompt(contexts, statements), Verdicts)
    if vd is None or not vd.verdicts:
        return MetricResult(name, None, {"reason": "judge failed to produce verdicts"})
    by_id: dict[int, int] = {}
    for v in vd.verdicts:
        if 1 <= v.id <= len(statements) and v.id not in by_id:
            by_id[v.id] = v.verdict
    if not by_id:
        return MetricResult(name, None, {"reason": "verdict ids did not match statements"})
    supported = sum(by_id.values())
    return MetricResult(
        name,
        supported / len(by_id),
        {
            "statements": statements,
            "verdicts": [v.model_dump() for v in vd.verdicts],
            "n_supported": supported,
            "n_judged": len(by_id),
            "n_unjudged": len(statements) - len(by_id),
        },
    )


def answer_relevancy(
    judge: Judge, embedder: Embedder, question: str, answer: str, *, n_questions: int = 3
) -> MetricResult:
    name = "answer_relevancy"
    if not answer.strip():
        return MetricResult(name, 0.0, {"reason": "empty answer"})
    gq = judge.ask(name, questions_prompt(answer, n_questions), GeneratedQuestions)
    if gq is None:
        return MetricResult(name, None, {"reason": "judge failed to generate questions"})
    questions = [q.strip() for q in gq.questions if q.strip()]
    judge_flag = bool(gq.noncommittal)
    lexical_flag = is_abstention(answer)
    # Small judges over-flag "non-committal"; require the lexical check to agree unless the
    # answer is too short to carry a fact.
    noncommittal = lexical_flag or (judge_flag and len(answer.split()) < 8)
    detail: dict[str, Any] = {
        "questions": questions,
        "judge_noncommittal": judge_flag,
        "lexical_noncommittal": lexical_flag,
    }
    if noncommittal or not questions:
        detail["noncommittal"] = True
        return MetricResult(name, 0.0, detail)
    q_vec = embedder.embed_queries([question])[0]
    g_vecs = embedder.embed_queries(questions)
    sims = np.clip(g_vecs @ q_vec, 0.0, 1.0)
    detail["similarities"] = [float(s) for s in sims]
    return MetricResult(name, float(sims.mean()), detail)


def context_precision(
    judge: Judge, question: str, reference: str, contexts: Sequence[str]
) -> MetricResult:
    name = "context_precision"
    if not contexts:
        return MetricResult(name, None, {"reason": "no contexts"})
    verdicts: list[int] = []
    for i, ctx in enumerate(contexts, start=1):
        cv = judge.ask(
            f"{name}/{i}", context_precision_prompt(question, reference, ctx), ContextVerdict
        )
        if cv is None:
            return MetricResult(name, None, {"reason": f"judge failed on context {i}"})
        verdicts.append(cv.useful)
    total_relevant = sum(verdicts)
    if total_relevant == 0:
        return MetricResult(name, 0.0, {"verdicts": verdicts})
    score = 0.0
    hits = 0
    for k, v in enumerate(verdicts, start=1):
        hits += v
        score += (hits / k) * v
    return MetricResult(name, score / total_relevant, {"verdicts": verdicts})


def context_recall(
    judge: Judge, question: str, reference: str, contexts: Sequence[str]
) -> MetricResult:
    name = "context_recall"
    if not contexts:
        return MetricResult(name, None, {"reason": "no contexts"})
    at = judge.ask(name, context_recall_prompt(question, reference, contexts), Attributions)
    if at is None or not at.items:
        return MetricResult(name, None, {"reason": "judge failed to classify sentences"})
    attributed = sum(i.attributed for i in at.items)
    return MetricResult(
        name, attributed / len(at.items), {"items": [i.model_dump() for i in at.items]}
    )


# ----- deterministic proxies ----------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> list[str]:
    """SQuAD-style normalisation: lower-case, strip punctuation and articles."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return _ARTICLES_RE.sub(" ", lowered).split()


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def reference_coverage(prediction: str, reference: str) -> float:
    """Fraction of the reference's (normalised) tokens present in the prediction — a recall
    proxy that is robust to verbose but correct answers."""
    pred = Counter(normalize_answer(prediction))
    ref = Counter(normalize_answer(reference))
    if not ref:
        return 1.0
    return sum((pred & ref).values()) / sum(ref.values())
