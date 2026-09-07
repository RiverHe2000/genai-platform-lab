"""The inferential statistics the whole project reports with.

A benchmark that prints "architecture A scored 0.72, architecture B scored 0.68" has said
almost nothing. On seventy-two tasks that gap is three items, and three items is inside the noise
of which seed was used. Every headline number here therefore arrives with an interval, and
every architecture comparison is paired --- both systems answer the *same* tasks, so the
difference is measured per task and the between-task variance, which dwarfs the effect,
cancels out.

Everything is implemented from the definitions on the standard library. There is no SciPy
dependency: the four procedures below are a hundred lines between them, the formulas are
in the docstrings, and a reviewer can check them by hand, which matters more for a number
that gates a release than saving the hundred lines would.

All results are frozen pydantic models, so a figure and the interval it was drawn from
cannot drift apart in a caller that thought it was editing a copy.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from enum import StrEnum
from fractions import Fraction
from statistics import NormalDist, fmean

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Decision",
    "Interval",
    "PairedResult",
    "ReleaseDecision",
    "Statistic",
    "bootstrap_ci",
    "decide_release",
    "mcnemar_exact",
    "mean",
    "non_inferiority",
    "paired_bootstrap_diff",
    "percentile",
    "wilson_interval",
]

#: A summary of a sample. Anything with this shape can be bootstrapped, which is the whole
#: appeal of the bootstrap: no sampling distribution has to be derived for it first.
Statistic = Callable[[Sequence[float]], float]


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, as a named function so it can be a default argument."""
    return fmean(values)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-quantile by linear interpolation between order statistics.

    This is the "type 7" definition (the default in R and NumPy): with ``n`` sorted values
    the quantile sits at position ``h = (n - 1) * q``, and non-integer ``h`` interpolates
    linearly between its neighbours. Interpolating rather than picking the nearest order
    statistic keeps the percentile bootstrap smooth in ``n_boot``, so a CI does not jump
    when the resample count changes by one.

    Args:
        values: A non-empty sample; sorted internally, the caller's order is untouched.
        q: A quantile in [0, 1].

    Returns:
        The interpolated quantile.

    Raises:
        ValueError: If ``values`` is empty or ``q`` lies outside [0, 1].
    """
    if not values:
        msg = "percentile of an empty sample is undefined"
        raise ValueError(msg)
    if not 0.0 <= q <= 1.0:
        msg = f"q must lie in [0, 1], got {q}"
        raise ValueError(msg)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def _ordered_bounds(stats: Sequence[float], alpha: float) -> tuple[float, float]:
    """The two percentile bounds, in order.

    A degenerate sample -- every observation identical, which a paired comparison reaches
    whenever two arms differ by the same amount on every task -- makes every bootstrap
    replicate equal, so the two quantiles are the same number computed two different ways.
    Linear interpolation can then put them an ulp apart in the wrong order, and `Interval`
    refuses an inverted interval. Swapping is exact rather than a fudge: when the replicates
    are all equal the true interval has zero width. Found by hypothesis on the sibling
    implementation in `sft-dpo-alignment`.
    """
    low = percentile(stats, alpha / 2.0)
    high = percentile(stats, 1.0 - alpha / 2.0)
    return (high, low) if high < low else (low, high)


class Interval(BaseModel):
    """A point estimate with a confidence interval around it.

    ``alpha`` and ``n`` travel with the bounds because an interval quoted without its
    coverage level or its sample size is not interpretable, and in practice they get
    separated the moment the numbers reach a slide.
    """

    model_config = ConfigDict(frozen=True)

    point: float
    low: float
    high: float
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    n: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Interval:
        if self.low > self.high:
            msg = f"interval bounds are inverted: low={self.low} > high={self.high}"
            raise ValueError(msg)
        return self

    @property
    def width(self) -> float:
        """How much the estimate is not pinned down by; the honest headline number."""
        return self.high - self.low

    @property
    def confidence(self) -> float:
        """Nominal coverage, e.g. 0.95 for ``alpha=0.05``."""
        return 1.0 - self.alpha

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero."""
        return self.low > 0.0 or self.high < 0.0

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies within the closed interval."""
        return self.low <= value <= self.high


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Statistic = mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap confidence interval for any statistic.

    The sample is treated as a stand-in for the population: draw ``n_boot`` resamples of
    the same size with replacement, recompute the statistic on each, and read the
    ``alpha/2`` and ``1 - alpha/2`` quantiles off the resulting distribution. No normality
    is assumed, which matters here because the quantities being summarised --- success
    rates, step counts, token totals --- are bounded, discrete or heavy-tailed, and a
    t-interval on any of them would be quietly wrong.

    Determinism is a hard requirement, not a convenience: the CI is part of a release gate,
    so the same inputs and seed must always produce the same bounds. A private
    :class:`random.Random` is used rather than the module-level generator so a caller
    seeding ``random`` elsewhere cannot move a published interval.

    Args:
        values: The observed sample; must be non-empty.
        statistic: The summary to bootstrap. Defaults to the mean.
        n_boot: Number of resamples. 2000 is the usual floor for a stable 95% percentile
            interval --- with fewer, each tail is estimated from a handful of draws.
        alpha: One minus the nominal coverage.
        seed: Seed for the resampling generator.

    Returns:
        The interval, whose ``point`` is the statistic on the original sample --- not the
        mean of the resamples, which carries the bootstrap's own bias.

    Raises:
        ValueError: If the sample is empty, ``n_boot`` is below 1, or ``alpha`` is not
            strictly between 0 and 1.
    """
    if not values:
        msg = "cannot bootstrap an empty sample"
        raise ValueError(msg)
    if n_boot < 1:
        msg = f"n_boot must be at least 1, got {n_boot}"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie strictly between 0 and 1, got {alpha}"
        raise ValueError(msg)

    rng = random.Random(seed)
    sample = list(values)
    n = len(sample)
    stats = [statistic(rng.choices(sample, k=n)) for _ in range(n_boot)]
    low, high = _ordered_bounds(stats, alpha)
    return Interval(point=statistic(sample), low=low, high=high, alpha=alpha, n=n)


class PairedResult(BaseModel):
    """A paired comparison of two systems over the same items.

    ``wins`` / ``losses`` / ``ties`` are reported alongside the mean difference because
    they answer a different question. A mean difference of +0.02 built from twenty wins and
    eighteen losses is a very different finding from the same +0.02 built from two wins and
    no losses, and only the win/loss split shows which one happened.
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=1)
    mean_a: float
    mean_b: float
    diff: float
    ci: Interval
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)
    tie_tolerance: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _outcomes_account_for_every_pair(self) -> PairedResult:
        """Every pair is exactly one of a win, a loss or a tie.

        Enforced in the model rather than left to the constructor because this is the
        invariant a reader of the report assumes without checking.
        """
        total = self.wins + self.losses + self.ties
        if total != self.n:
            msg = f"wins + losses + ties = {total}, expected n = {self.n}"
            raise ValueError(msg)
        return self

    @property
    def significant(self) -> bool:
        """Whether the CI for the difference excludes zero."""
        return self.ci.excludes_zero

    @property
    def win_rate(self) -> float:
        """Share of items on which A beat B, ties counted against A."""
        return self.wins / self.n


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    tie_tolerance: float = 0.0,
) -> PairedResult:
    """Mean difference ``a - b`` over paired observations, with a bootstrap CI.

    Resampling is over *item indices*, not over the two samples independently. Both systems
    saw the same tasks, so task difficulty is a shared nuisance term; keeping each pair
    intact cancels it, and the interval then describes the difference rather than the
    spread of the benchmark. Resampling the two arrays separately would throw that away and
    typically widens the interval by an order of magnitude.

    For a mean this is arithmetically the same as bootstrapping the per-item differences,
    but it is written over indices so the same routine holds for a statistic that is not
    linear in the pairs.

    Args:
        a: The candidate's per-item scores.
        b: The baseline's per-item scores, item-aligned with ``a``.
        n_boot: Number of resamples.
        alpha: One minus the nominal coverage.
        seed: Seed for the resampling generator.
        tie_tolerance: Differences with magnitude at or below this count as ties. Zero is
            right for 0/1 correctness; on latencies or token counts a small tolerance stops
            a floating-point wobble of 1e-12 being reported as a win.

    Returns:
        The paired result.

    Raises:
        ValueError: If the samples differ in length, are empty, or the numeric arguments
            are out of range.
    """
    if len(a) != len(b):
        msg = f"paired samples must be the same length, got {len(a)} and {len(b)}"
        raise ValueError(msg)
    if not a:
        msg = "cannot compare empty samples"
        raise ValueError(msg)
    if n_boot < 1:
        msg = f"n_boot must be at least 1, got {n_boot}"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie strictly between 0 and 1, got {alpha}"
        raise ValueError(msg)
    if tie_tolerance < 0.0:
        msg = f"tie_tolerance must not be negative, got {tie_tolerance}"
        raise ValueError(msg)

    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b, strict=True)]
    wins = sum(1 for d in diffs if d > tie_tolerance)
    losses = sum(1 for d in diffs if d < -tie_tolerance)

    rng = random.Random(seed)
    indices = range(n)
    stats = [fmean([diffs[i] for i in rng.choices(indices, k=n)]) for _ in range(n_boot)]
    low, high = _ordered_bounds(stats, alpha)
    ci = Interval(point=fmean(diffs), low=low, high=high, alpha=alpha, n=n)
    return PairedResult(
        n=n,
        mean_a=fmean(a),
        mean_b=fmean(b),
        diff=fmean(diffs),
        ci=ci,
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
        tie_tolerance=tie_tolerance,
    )


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for two paired binary classifiers.

    ``b`` is the count of items the first system got right and the second got wrong, ``c``
    the reverse. Items both got right and items both got wrong carry no information about
    which is better and do not appear.

    Under the null hypothesis the two systems are equally likely to be the one that
    succeeds on a discordant item, so ``b ~ Binomial(b + c, 1/2)``. The two-sided p-value
    is therefore the total probability of a split at least as lopsided as the one observed,
    which by the symmetry of the binomial at p = 1/2 is::

        p = min(1, 2 * sum_{k=0}^{min(b, c)} C(n, k) / 2**n),   n = b + c

    The exact form is used rather than the chi-square approximation because the benchmark
    is seventy-two tasks: discordant counts land in the single digits, where the continuity of
    chi-square is a poor fit and the approximation is anti-conservative exactly when the
    result is about to be believed. The sum is accumulated as an integer and divided once
    as a :class:`fractions.Fraction`, so there is no cancellation and no overflow of
    ``2**n`` for large ``n``.

    With no discordant pairs at all (``b = c = 0``) the sum is ``C(0, 0) = 1``, the doubled
    value is 2, and the cap returns 1.0: no evidence of any difference, which is the right
    answer rather than an error.

    Args:
        b: Discordant items favouring the first system.
        c: Discordant items favouring the second system.

    Returns:
        The two-sided p-value in [0, 1].

    Raises:
        ValueError: If either count is negative.
    """
    if b < 0 or c < 0:
        msg = f"discordant counts must not be negative, got b={b}, c={c}"
        raise ValueError(msg)
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, float(2 * Fraction(tail, 2**n)))


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion.

    Inverting the score test rather than using the Wald interval ``p +- z * sqrt(p(1-p)/n)``
    matters at the edges, which is where benchmark proportions live: a policy that refused
    every unauthorised attempt scores 20/20, and Wald reports the interval [1, 1] --- a
    claim of certainty from twenty observations. Wilson reports roughly [0.84, 1], never
    leaves [0, 1], and never collapses to a point.

    With ``phat = successes / n`` and ``z`` the standard normal quantile at ``1 - alpha/2``::

        centre = (phat + z^2 / (2n)) / (1 + z^2 / n)
        half   = z / (1 + z^2 / n) * sqrt(phat(1 - phat)/n + z^2 / (4n^2))

    The interval always contains ``phat``, since ``phat`` solves the score equation with a
    statistic of zero.

    Args:
        successes: Number of successes observed.
        n: Number of trials; must be positive.
        alpha: One minus the nominal coverage.

    Returns:
        The interval, with ``point`` the raw proportion.

    Raises:
        ValueError: If ``n`` is not positive, ``successes`` lies outside [0, n], or
            ``alpha`` is not strictly between 0 and 1.
    """
    if n <= 0:
        msg = f"n must be positive, got {n}"
        raise ValueError(msg)
    if not 0 <= successes <= n:
        msg = f"successes must lie in [0, {n}], got {successes}"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie strictly between 0 and 1, got {alpha}"
        raise ValueError(msg)

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z / denom * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    # Clamped to [0, 1] and to the observed proportion. The second clamp is not cosmetic:
    # at successes = 0 the algebra gives centre == half exactly, but the two are computed by
    # different routes and the subtraction lands an ulp above zero, which would report a
    # lower bound above an outcome that was actually observed. Enforcing containment here
    # keeps the interval's defining property true in floating point as well as on paper.
    return Interval(
        point=phat,
        low=min(phat, max(0.0, centre - half)),
        high=max(phat, min(1.0, centre + half)),
        alpha=alpha,
        n=n,
    )


def non_inferiority(diff_ci: Interval, margin: float) -> bool:
    """Whether a candidate is non-inferior to its baseline by the given margin.

    ``diff_ci`` is an interval for ``candidate - baseline`` on a metric where higher is
    better. The candidate passes when the whole interval sits above ``-margin``: even the
    pessimistic end of what the data support is a loss small enough to accept.

    This is the shape a release gate needs and a significance test cannot give. "No
    significant difference" is not evidence of equivalence --- it is what an underpowered
    run says about everything --- whereas a lower bound above ``-margin`` is a positive
    claim, and it fails automatically when the run is too small to support one.

    Args:
        diff_ci: Interval for the difference, candidate minus baseline.
        margin: The largest regression that is still acceptable, as a positive number.

    Returns:
        True when the candidate is non-inferior.

    Raises:
        ValueError: If ``margin`` is not strictly positive. A zero margin silently turns
            this into a superiority test, which is a different decision wearing this one's
            name.
    """
    if margin <= 0.0:
        msg = f"margin must be strictly positive, got {margin}"
        raise ValueError(msg)
    return diff_ci.low > -margin


class Decision(StrEnum):
    """The three outcomes of a release gate.

    HOLD exists so that "we cannot tell yet" has somewhere to go. With only promote and
    reject, an inconclusive run is forced into one of them, and it is always forced into
    whichever one the person reading it wanted.
    """

    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


class ReleaseDecision(BaseModel):
    """A gate outcome with the reasoning that produced it.

    The reasoning is stored, not re-derived at print time, so the sentence in the report
    and the verdict in the pipeline can never disagree.
    """

    model_config = ConfigDict(frozen=True)

    decision: Decision
    reason: str
    margin: float
    ci: Interval

    @property
    def promote(self) -> bool:
        return self.decision is Decision.PROMOTE


def decide_release(diff_ci: Interval, *, margin: float) -> ReleaseDecision:
    """Turn a difference interval and a margin into a promote / hold / reject.

    The three cases partition the possibilities, and exactly one always applies:

    * **PROMOTE** --- the lower bound is above ``-margin``: non-inferior.
    * **REJECT** --- the *upper* bound is at or below ``-margin``: the whole interval is a
      regression larger than the margin allows, so more data will not rescue it.
    * **HOLD** --- the interval straddles ``-margin``: the run cannot separate an
      acceptable loss from an unacceptable one, so the answer is more data, not a coin toss.

    Args:
        diff_ci: Interval for candidate minus baseline, higher being better.
        margin: The largest acceptable regression, strictly positive.

    Returns:
        The decision and its reasoning.

    Raises:
        ValueError: If ``margin`` is not strictly positive.
    """
    if non_inferiority(diff_ci, margin):
        return ReleaseDecision(
            decision=Decision.PROMOTE,
            reason=(
                f"non-inferior: the {diff_ci.confidence:.0%} interval for the difference "
                f"is [{diff_ci.low:+.4f}, {diff_ci.high:+.4f}], whose lower bound clears "
                f"the margin of {-margin:+.4f}"
            ),
            margin=margin,
            ci=diff_ci,
        )
    if diff_ci.high <= -margin:
        return ReleaseDecision(
            decision=Decision.REJECT,
            reason=(
                f"regression: the whole {diff_ci.confidence:.0%} interval "
                f"[{diff_ci.low:+.4f}, {diff_ci.high:+.4f}] lies at or below the margin of "
                f"{-margin:+.4f}"
            ),
            margin=margin,
            ci=diff_ci,
        )
    return ReleaseDecision(
        decision=Decision.HOLD,
        reason=(
            f"inconclusive: the {diff_ci.confidence:.0%} interval "
            f"[{diff_ci.low:+.4f}, {diff_ci.high:+.4f}] straddles the margin of "
            f"{-margin:+.4f}, so {diff_ci.n} observations cannot separate an acceptable "
            f"loss from an unacceptable one"
        ),
        margin=margin,
        ci=diff_ci,
    )
