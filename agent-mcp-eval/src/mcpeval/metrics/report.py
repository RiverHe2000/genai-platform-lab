"""Aggregating grades into a report, and comparing two runs well enough to gate a release.

Three commitments shape this module.

**Every headline number carries an interval.** Eighty tasks is a small sample; the gap
between 0.72 and 0.68 is three items, and three items is inside the noise of which seed
the world was built with. A point estimate printed alone invites a decision the data
cannot support, so :func:`aggregate` attaches a bootstrap interval to every metric, at
every level of the breakdown.

**The Markdown is byte-stable.** Given the same grades and the same seed, the same bytes
come out --- no timestamps, no dict iteration order, no locale-dependent formatting. That
is what makes a report reviewable: it can be committed, and a diff between two runs shows
only what actually changed. A generation time can be injected, which keeps the choice with
the caller who knows whether the output is going into git or onto a wiki.

**Comparisons are paired.** Both runs answer the same tasks, so the difference is measured
per task and the between-task variance --- which dwarfs the effect being measured ---
cancels. See :mod:`mcpeval.metrics.stats` for why that matters and how much it buys.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from mcpeval.metrics.stats import (
    Interval,
    PairedResult,
    ReleaseDecision,
    bootstrap_ci,
    decide_release,
    mcnemar_exact,
    paired_bootstrap_diff,
)
from mcpeval.schemas import FailureClass, Grade, TaskFamily

__all__ = [
    "METRIC_KEYS",
    "Aggregate",
    "Comparison",
    "FailureSummary",
    "Summary",
    "aggregate",
    "compare",
    "render_comparison_markdown",
    "render_json",
    "render_markdown",
]


@dataclass(frozen=True, slots=True)
class _MetricSpec:
    """One reported quantity: how to get it out of a grade and how to print it.

    Holding the projection and the format together is what makes the report stable and
    the module short --- there is exactly one list of metrics, and the tables, the JSON
    and the comparison all walk it in the same order.
    """

    key: str
    label: str
    get: Callable[[Grade], float]
    digits: int


_METRICS: Final[tuple[_MetricSpec, ...]] = (
    _MetricSpec("success", "Success", lambda g: float(g.success), 4),
    _MetricSpec("answer_score", "Answer score", lambda g: g.answer_score, 4),
    _MetricSpec("call_precision", "Call precision", lambda g: g.call_precision, 4),
    _MetricSpec("call_recall", "Call recall", lambda g: g.call_recall, 4),
    _MetricSpec("call_f1", "Call F1", lambda g: g.call_f1, 4),
    _MetricSpec("step_efficiency", "Step efficiency", lambda g: g.step_efficiency, 4),
    _MetricSpec("redundant_calls", "Redundant calls", lambda g: float(g.redundant_calls), 2),
    _MetricSpec(
        "forbidden_violations", "Forbidden violations", lambda g: float(g.forbidden_violations), 2
    ),
    _MetricSpec("steps", "Steps", lambda g: float(g.steps), 2),
    _MetricSpec("tokens", "Tokens", lambda g: float(g.tokens), 1),
    _MetricSpec("wall_ms", "Wall ms", lambda g: g.wall_ms, 1),
)

#: The metric names, in report order. Public so a caller can index a summary safely.
METRIC_KEYS: Final[tuple[str, ...]] = tuple(m.key for m in _METRICS)

_SPEC_BY_KEY: Final[dict[str, _MetricSpec]] = {m.key: m for m in _METRICS}

#: The four columns of the per-family table. The full set is available in the JSON; a
#: table wide enough to need horizontal scrolling is a table nobody reads.
_FAMILY_METRICS: Final[tuple[str, ...]] = (
    "success",
    "answer_score",
    "call_f1",
    "step_efficiency",
)

_MIXED: Final = "mixed"


class Summary(BaseModel):
    """Every metric for one slice of a run, each with its interval."""

    model_config = ConfigDict(frozen=True)

    label: str
    n: int = Field(ge=1)
    metrics: dict[str, Interval]


class FailureSummary(BaseModel):
    """How often one failure class occurred, with an interval on the rate.

    The rate is the share of *trajectories* carrying the class, not a share of failures:
    classes overlap, so shares of failures would not sum to anything meaningful and would
    invite exactly the misreading that a percentage sign encourages.
    """

    model_config = ConfigDict(frozen=True)

    failure: FailureClass
    count: int = Field(ge=0)
    rate: Interval


class Aggregate(BaseModel):
    """A whole run, summarised: overall, per family, and per failure class.

    The bootstrap settings travel with the numbers because an interval quoted without its
    resample count and seed cannot be reproduced, and this object is the thing that gets
    serialised, committed and quoted six months later.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    architecture: str
    model: str
    n: int = Field(ge=1)
    overall: Summary
    by_family: tuple[Summary, ...]
    failures: tuple[FailureSummary, ...]
    alpha: float = Field(gt=0.0, lt=1.0)
    n_boot: int = Field(ge=1)
    seed: int


def _uniform(values: Sequence[str]) -> str:
    """The single value they all share, or ``"mixed"``.

    A report whose rows came from two different models must say so in the header rather
    than silently claim the first one.
    """
    unique = set(values)
    if len(unique) == 1:
        return next(iter(unique))
    return _MIXED


def _summarise(
    grades: Sequence[Grade], label: str, *, n_boot: int, alpha: float, seed: int
) -> Summary:
    return Summary(
        label=label,
        n=len(grades),
        metrics={
            spec.key: bootstrap_ci(
                [spec.get(g) for g in grades], n_boot=n_boot, alpha=alpha, seed=seed
            )
            for spec in _METRICS
        },
    )


def aggregate(
    grades: Sequence[Grade],
    *,
    label: str = "run",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Aggregate:
    """Summarise a run of graded trajectories.

    Families and failure classes are emitted in the declaration order of their enums, and
    only those actually present appear. Fixed order rather than "sorted by count" is what
    makes two reports diffable: a class that gains one occurrence should move one number,
    not reshuffle the table.

    Every interval uses the same seed. Sharing it means all metrics are bootstrapped over
    the same resampled index sets, so the intervals in one row are mutually consistent ---
    and, more practically, so the report is reproducible from a single number.

    Args:
        grades: The run's grades; must be non-empty.
        label: A name for the run, used in the report heading.
        n_boot: Bootstrap resamples per interval.
        alpha: One minus the nominal coverage.
        seed: Seed for every bootstrap in this aggregate.

    Returns:
        The aggregate.

    Raises:
        ValueError: If ``grades`` is empty. There is no defensible summary of no data, and
            returning zeros would put a row in a report that reads as a result.
    """
    if not grades:
        msg = "cannot aggregate an empty run"
        raise ValueError(msg)

    by_family = tuple(
        _summarise(rows, family.value, n_boot=n_boot, alpha=alpha, seed=seed)
        for family in TaskFamily
        if (rows := [g for g in grades if g.family is family])
    )

    failures: list[FailureSummary] = []
    for failure in FailureClass:
        flags = [1.0 if failure in g.failures else 0.0 for g in grades]
        count = int(sum(flags))
        if count == 0:
            continue
        failures.append(
            FailureSummary(
                failure=failure,
                count=count,
                rate=bootstrap_ci(flags, n_boot=n_boot, alpha=alpha, seed=seed),
            )
        )

    return Aggregate(
        label=label,
        architecture=_uniform([g.architecture for g in grades]),
        model=_uniform([g.model for g in grades]),
        n=len(grades),
        overall=_summarise(grades, "overall", n_boot=n_boot, alpha=alpha, seed=seed),
        by_family=by_family,
        failures=tuple(failures),
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
    )


def _fmt(value: float, digits: int) -> str:
    """Fixed-width decimal. Never scientific notation, which breaks column alignment."""
    return f"{value:.{digits}f}"


def _cell(interval: Interval, digits: int) -> str:
    """A point estimate and its bounds in one table cell, for the narrower tables."""
    point = _fmt(interval.point, digits)
    return f"{point} [{_fmt(interval.low, digits)}, {_fmt(interval.high, digits)}]"


def render_markdown(agg: Aggregate, *, generated_at: str | None = None) -> str:
    """Render an aggregate as Markdown.

    The output is a pure function of its arguments: no clock, no environment, no
    unordered iteration. Rendering the same aggregate twice produces identical bytes, so
    the report can live in the repository and its diff can be reviewed like code. A
    timestamp is only ever present if the caller passes one.

    Args:
        agg: The aggregate to render.
        generated_at: Optional timestamp line. Supplying it makes the output vary, which
            is the caller's decision to make.

    Returns:
        The Markdown document, ending in a newline.
    """
    lines = [
        f"# Trajectory report: {agg.label}",
        "",
        f"- Tasks: {agg.n}",
        f"- Architecture: `{agg.architecture}`",
        f"- Model: `{agg.model}`",
        f"- Intervals: {1.0 - agg.alpha:.0%} percentile bootstrap, "
        f"{agg.n_boot} resamples, seed {agg.seed}",
    ]
    if generated_at is not None:
        lines.append(f"- Generated: {generated_at}")
    lines += [
        "",
        "## Overall",
        "",
        "| Metric | Point | Low | High |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for spec in _METRICS:
        interval = agg.overall.metrics[spec.key]
        lines.append(
            f"| {spec.label} | {_fmt(interval.point, spec.digits)} "
            f"| {_fmt(interval.low, spec.digits)} | {_fmt(interval.high, spec.digits)} |"
        )

    header = " | ".join(_SPEC_BY_KEY[k].label for k in _FAMILY_METRICS)
    lines += [
        "",
        "## By family",
        "",
        f"| Family | n | {header} |",
        "| :--- | ---: |" + " ---: |" * len(_FAMILY_METRICS),
    ]
    for summary in agg.by_family:
        cells = " | ".join(
            _cell(summary.metrics[k], _SPEC_BY_KEY[k].digits) for k in _FAMILY_METRICS
        )
        lines.append(f"| {summary.label} | {summary.n} | {cells} |")

    lines += [
        "",
        "## Failures",
        "",
        "| Class | Count | Rate | Low | High |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    if not agg.failures:
        lines.append("| (none recorded) | 0 | 0.0000 | 0.0000 | 0.0000 |")
    for row in agg.failures:
        lines.append(
            f"| {row.failure.value} | {row.count} | {_fmt(row.rate.point, 4)} "
            f"| {_fmt(row.rate.low, 4)} | {_fmt(row.rate.high, 4)} |"
        )
    return "\n".join(lines) + "\n"


def render_json(agg: Aggregate) -> str:
    """Render an aggregate as indented JSON, ending in a newline.

    Field order follows the model declaration and dict order follows insertion, so this is
    as stable as the Markdown and is the form the dashboards and the release gate read.
    """
    return agg.model_dump_json(indent=2) + "\n"


class Comparison(BaseModel):
    """A paired comparison of two runs over the tasks they both attempted.

    ``only_a`` and ``only_b`` are the discordant counts McNemar's test is computed from,
    and they are kept because they are the most legible number in the whole object: "won
    six, lost one" settles arguments that a p-value starts.
    """

    model_config = ConfigDict(frozen=True)

    label_a: str
    label_b: str
    n: int = Field(ge=1)
    paired: dict[str, PairedResult]
    only_a: int = Field(ge=0)
    only_b: int = Field(ge=0)
    mcnemar_p: float = Field(ge=0.0, le=1.0)
    decision: ReleaseDecision

    @property
    def promote(self) -> bool:
        """Whether the gate says the candidate may ship."""
        return self.decision.promote


def compare(
    a: Sequence[Grade],
    b: Sequence[Grade],
    margin: float = 0.05,
    *,
    label_a: str = "candidate",
    label_b: str = "baseline",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Comparison:
    """Compare a candidate run against a baseline, task by task, and gate on the result.

    Only tasks present in both runs are compared, and they are aligned by ``task_id``
    rather than by position: a run that skipped a task must not silently shift every
    subsequent pair by one, which is the failure mode of zipping two lists of results.

    The promotion decision is a non-inferiority test on success rate, not a significance
    test. "No significant difference" is what an underpowered run says about everything,
    and it is not evidence that the candidate is safe to ship; a lower confidence bound
    above ``-margin`` is a positive claim, and it fails by itself when the run is too small
    to support one. See :func:`mcpeval.metrics.stats.decide_release`.

    Args:
        a: The candidate's grades.
        b: The baseline's grades.
        margin: The largest success-rate regression still acceptable, strictly positive.
        label_a: Name for the candidate in the report.
        label_b: Name for the baseline.
        n_boot: Bootstrap resamples per interval.
        alpha: One minus the nominal coverage.
        seed: Seed for every bootstrap in this comparison.

    Returns:
        The comparison, including the promote / hold / reject decision.

    Raises:
        ValueError: If either run grades the same task twice --- which would make the
            pairing ambiguous --- or if the two runs share no tasks at all.
    """
    left = _index(a, label_a)
    right = _index(b, label_b)
    shared = sorted(set(left) & set(right))
    if not shared:
        msg = f"{label_a!r} and {label_b!r} have no tasks in common, so nothing can be paired"
        raise ValueError(msg)

    rows_a = [left[t] for t in shared]
    rows_b = [right[t] for t in shared]

    paired = {
        spec.key: paired_bootstrap_diff(
            [spec.get(g) for g in rows_a],
            [spec.get(g) for g in rows_b],
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
        for spec in _METRICS
    }
    only_a = sum(1 for x, y in zip(rows_a, rows_b, strict=True) if x.success and not y.success)
    only_b = sum(1 for x, y in zip(rows_a, rows_b, strict=True) if y.success and not x.success)

    return Comparison(
        label_a=label_a,
        label_b=label_b,
        n=len(shared),
        paired=paired,
        only_a=only_a,
        only_b=only_b,
        mcnemar_p=mcnemar_exact(only_a, only_b),
        decision=decide_release(paired["success"].ci, margin=margin),
    )


def _index(grades: Sequence[Grade], label: str) -> dict[str, Grade]:
    seen: dict[str, Grade] = {}
    for g in grades:
        if g.task_id in seen:
            msg = f"run {label!r} grades task {g.task_id!r} more than once"
            raise ValueError(msg)
        seen[g.task_id] = g
    return seen


def render_comparison_markdown(cmp: Comparison, *, generated_at: str | None = None) -> str:
    """Render a comparison as Markdown, byte-stable on the same inputs.

    Args:
        cmp: The comparison.
        generated_at: Optional timestamp line, omitted by default.

    Returns:
        The Markdown document, ending in a newline.
    """
    decision = cmp.decision
    lines = [
        f"# {cmp.label_a} vs {cmp.label_b}",
        "",
        f"- Paired tasks: {cmp.n}",
        f"- Decision: **{decision.decision.value}**",
        f"- Reason: {decision.reason}",
        f"- Discordant: {cmp.only_a} to {cmp.label_a}, {cmp.only_b} to {cmp.label_b} "
        f"(McNemar p = {cmp.mcnemar_p:.4f})",
    ]
    if generated_at is not None:
        lines.append(f"- Generated: {generated_at}")
    lines += [
        "",
        f"| Metric | {cmp.label_a} | {cmp.label_b} | Difference | Low | High |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in _METRICS:
        result = cmp.paired[spec.key]
        lines.append(
            f"| {spec.label} | {_fmt(result.mean_a, spec.digits)} "
            f"| {_fmt(result.mean_b, spec.digits)} | {_fmt(result.diff, spec.digits)} "
            f"| {_fmt(result.ci.low, spec.digits)} | {_fmt(result.ci.high, spec.digits)} |"
        )
    return "\n".join(lines) + "\n"
