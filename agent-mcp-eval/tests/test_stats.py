"""Tests for the statistics module.

Where a closed form exists the expected value is computed by hand in the test and written
as a literal, so a test failure points at the formula rather than at another implementation
of the same mistake. The bootstrap has no closed form, so it is pinned by its invariants
instead: determinism under a seed, bounds inside the sample's own range, and a CI that
brackets the statistic it was built around.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mcpeval.metrics.stats import (
    Decision,
    Interval,
    PairedResult,
    ReleaseDecision,
    bootstrap_ci,
    decide_release,
    mcnemar_exact,
    mean,
    non_inferiority,
    paired_bootstrap_diff,
    percentile,
    wilson_interval,
)

SCORES = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]

_FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64)
_SAMPLES = st.lists(_FINITE, min_size=1, max_size=30)
_COUNTS = st.integers(min_value=0, max_value=120)


def _median(values: Sequence[float]) -> float:
    """A ``Statistic``-shaped median, for the custom-statistic tests."""
    return float(median(values))


# --------------------------------------------------------------------------------------
# percentile
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (0.0, 1.0),
        (0.25, 2.0),
        (0.5, 3.0),
        (0.75, 4.0),
        (1.0, 5.0),
        (0.125, 1.5),
        (0.875, 4.5),
    ],
)
def test_percentile_matches_the_type_7_definition(q: float, expected: float) -> None:
    # h = (n - 1) * q on [1, 2, 3, 4, 5], so q = 0.125 sits half way between 1 and 2.
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], q) == pytest.approx(expected)


def test_percentile_sorts_its_input() -> None:
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0


def test_percentile_of_one_value_is_that_value() -> None:
    assert percentile([7.5], 0.99) == 7.5


def test_percentile_leaves_the_caller_s_list_alone() -> None:
    values = [3.0, 1.0, 2.0]
    percentile(values, 0.5)
    assert values == [3.0, 1.0, 2.0]


def test_percentile_of_an_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        percentile([], 0.5)


@pytest.mark.parametrize("q", [-0.01, 1.01, 2.0])
def test_percentile_rejects_a_quantile_outside_the_unit_interval(q: float) -> None:
    with pytest.raises(ValueError, match=r"q must lie in \[0, 1\]"):
        percentile([1.0, 2.0], q)


@given(values=_SAMPLES, q=st.floats(min_value=0.0, max_value=1.0))
def test_percentile_never_leaves_the_sample_range(values: list[float], q: float) -> None:
    assert min(values) <= percentile(values, q) <= max(values)


# --------------------------------------------------------------------------------------
# Interval
# --------------------------------------------------------------------------------------


def test_interval_reports_width_and_confidence() -> None:
    interval = Interval(point=0.5, low=0.4, high=0.7, alpha=0.05)
    assert interval.width == pytest.approx(0.3)
    assert interval.confidence == pytest.approx(0.95)


def test_interval_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError, match="inverted"):
        Interval(point=0.0, low=1.0, high=0.0)


@pytest.mark.parametrize(
    ("low", "high", "excludes"),
    [
        (0.1, 0.3, True),
        (-0.3, -0.1, True),
        (-0.1, 0.2, False),
        (0.0, 0.2, False),
        (0.0, 0.0, False),
    ],
)
def test_interval_knows_whether_it_straddles_zero(low: float, high: float, excludes: bool) -> None:
    assert Interval(point=low, low=low, high=high).excludes_zero is excludes


@pytest.mark.parametrize(
    ("value", "inside"), [(0.4, True), (0.7, True), (0.55, True), (0.39, False), (0.71, False)]
)
def test_interval_containment_is_closed(value: float, inside: bool) -> None:
    assert Interval(point=0.5, low=0.4, high=0.7).contains(value) is inside


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_interval_rejects_an_impossible_alpha(alpha: float) -> None:
    with pytest.raises(ValidationError):
        Interval(point=0.0, low=0.0, high=0.0, alpha=alpha)


def test_interval_is_frozen() -> None:
    interval = Interval(point=0.5, low=0.4, high=0.7)
    field = "low"
    with pytest.raises(ValidationError):
        setattr(interval, field, 0.0)


# --------------------------------------------------------------------------------------
# bootstrap_ci
# --------------------------------------------------------------------------------------


def test_bootstrap_point_is_the_statistic_on_the_original_sample() -> None:
    # Not the mean of the resamples, which carries the bootstrap's own bias.
    interval = bootstrap_ci(SCORES, n_boot=200, seed=1)
    assert interval.point == pytest.approx(0.7)


def test_bootstrap_is_deterministic_under_its_seed() -> None:
    first = bootstrap_ci(SCORES, n_boot=500, seed=4)
    second = bootstrap_ci(SCORES, n_boot=500, seed=4)
    assert first == second


def test_bootstrap_uses_a_private_generator() -> None:
    # Seeding the module-level generator between calls must not move a published interval.
    import random

    random.seed(1)
    first = bootstrap_ci(SCORES, n_boot=300, seed=0)
    random.seed(99)
    second = bootstrap_ci(SCORES, n_boot=300, seed=0)
    assert first == second


def test_a_different_seed_moves_the_bounds_but_not_the_estimate() -> None:
    # On a continuous sample, not on SCORES: with 0/1 data the resample means are so
    # coarsely quantised that two seeds routinely land on identical percentiles, which
    # would make this assertion a coin toss rather than a test.
    values = [0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4, 8.1, 10.0]
    first = bootstrap_ci(values, n_boot=300, seed=0)
    second = bootstrap_ci(values, n_boot=300, seed=1)
    assert first.point == second.point
    assert (first.low, first.high) != (second.low, second.high)


def test_a_constant_sample_gives_a_degenerate_interval() -> None:
    interval = bootstrap_ci([2.0] * 12, n_boot=100, seed=0)
    assert (interval.low, interval.point, interval.high) == (2.0, 2.0, 2.0)
    assert interval.width == 0.0


def test_a_single_observation_cannot_be_resampled_into_uncertainty() -> None:
    interval = bootstrap_ci([3.0], n_boot=50, seed=0)
    assert (interval.low, interval.high) == (3.0, 3.0)
    assert interval.n == 1


def test_bootstrap_accepts_any_statistic() -> None:
    interval = bootstrap_ci(SCORES, statistic=_median, n_boot=200, seed=0)
    assert interval.point == 1.0


def test_the_default_statistic_is_the_mean() -> None:
    assert mean(SCORES) == pytest.approx(0.7)
    assert bootstrap_ci(SCORES, n_boot=50, seed=0) == bootstrap_ci(
        SCORES, statistic=mean, n_boot=50, seed=0
    )


def test_a_smaller_alpha_gives_a_wider_interval() -> None:
    narrow = bootstrap_ci(SCORES, n_boot=1000, alpha=0.20, seed=2)
    wide = bootstrap_ci(SCORES, n_boot=1000, alpha=0.01, seed=2)
    assert wide.width >= narrow.width


def test_bootstrap_records_its_own_settings() -> None:
    interval = bootstrap_ci(SCORES, n_boot=100, alpha=0.10, seed=0)
    assert interval.alpha == 0.10
    assert interval.n == len(SCORES)


def test_bootstrap_of_an_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


@pytest.mark.parametrize("n_boot", [0, -1])
def test_bootstrap_needs_at_least_one_resample(n_boot: int) -> None:
    with pytest.raises(ValueError, match="n_boot must be at least 1"):
        bootstrap_ci(SCORES, n_boot=n_boot)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5, 2.0])
def test_bootstrap_rejects_an_impossible_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must lie"):
        bootstrap_ci(SCORES, alpha=alpha)


@given(values=_SAMPLES, seed=st.integers(min_value=0, max_value=64))
def test_a_bootstrap_ci_brackets_the_sample_statistic(values: list[float], seed: int) -> None:
    # Every resample mean lies between the sample's own min and max, and the bootstrap
    # distribution of the mean is centred on the sample mean, so neither tail can cross it.
    # The slack is for the arithmetic, not the statistics: a mean is a sum divided by n, and
    # that division can round a resample of n identical values an ulp outside them.
    interval = bootstrap_ci(values, n_boot=200, seed=seed)
    tol = 1e-9 * max(1.0, abs(min(values)), abs(max(values)))
    assert interval.contains(interval.point)
    assert min(values) - tol <= interval.low
    assert interval.high <= max(values) + tol


@given(values=_SAMPLES)
def test_bootstrap_is_reproducible_for_any_sample(values: list[float]) -> None:
    assert bootstrap_ci(values, n_boot=50, seed=7) == bootstrap_ci(values, n_boot=50, seed=7)


# --------------------------------------------------------------------------------------
# paired_bootstrap_diff
# --------------------------------------------------------------------------------------

A = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
B = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_paired_diff_is_the_difference_of_the_means() -> None:
    result = paired_bootstrap_diff(A, B, n_boot=200, seed=0)
    assert result.diff == pytest.approx(result.mean_a - result.mean_b)
    assert result.diff == pytest.approx(0.25)


def test_paired_counts_wins_losses_and_ties() -> None:
    result = paired_bootstrap_diff(A, B, n_boot=100, seed=0)
    assert (result.wins, result.losses, result.ties) == (2, 0, 6)
    assert result.win_rate == pytest.approx(0.25)


def test_a_tolerance_stops_floating_point_noise_becoming_a_win() -> None:
    a = [1.0, 1.0 + 1e-12, 1.0]
    b = [1.0, 1.0, 1.0]
    strict = paired_bootstrap_diff(a, b, n_boot=20, seed=0)
    forgiving = paired_bootstrap_diff(a, b, n_boot=20, seed=0, tie_tolerance=1e-9)
    assert strict.wins == 1
    assert forgiving.wins == 0
    assert forgiving.ties == 3


def test_a_uniform_gain_is_significant() -> None:
    result = paired_bootstrap_diff([1.0] * 20, [0.0] * 20, n_boot=500, seed=0)
    assert result.significant
    assert result.ci.low == 1.0


def test_a_noisy_wash_is_not_significant() -> None:
    a = [1.0, 0.0] * 10
    b = [0.0, 1.0] * 10
    result = paired_bootstrap_diff(a, b, n_boot=500, seed=0)
    assert not result.significant
    assert result.diff == pytest.approx(0.0)


def test_paired_samples_must_be_the_same_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff([1.0, 2.0], [1.0])


def test_paired_samples_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="empty samples"):
        paired_bootstrap_diff([], [])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_boot": 0}, "n_boot must be at least 1"),
        ({"alpha": 0.0}, "alpha must lie"),
        ({"alpha": 1.0}, "alpha must lie"),
        ({"tie_tolerance": -0.1}, "tie_tolerance must not be negative"),
    ],
)
def test_paired_rejects_out_of_range_settings(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        paired_bootstrap_diff(A, B, **kwargs)  # type: ignore[arg-type]


def test_paired_result_rejects_counts_that_do_not_account_for_every_pair() -> None:
    interval = Interval(point=0.0, low=-1.0, high=1.0)
    with pytest.raises(ValidationError, match="expected n"):
        PairedResult(n=10, mean_a=0.0, mean_b=0.0, diff=0.0, ci=interval, wins=1, losses=1, ties=1)


@given(
    a=st.lists(_FINITE, min_size=1, max_size=20),
    data=st.data(),
)
def test_every_pair_is_a_win_a_loss_or_a_tie(a: list[float], data: st.DataObject) -> None:
    b = data.draw(st.lists(_FINITE, min_size=len(a), max_size=len(a)))
    result = paired_bootstrap_diff(a, b, n_boot=30, seed=0)
    assert result.wins + result.losses + result.ties == result.n == len(a)


@given(a=st.lists(_FINITE, min_size=1, max_size=15), data=st.data())
def test_swapping_the_arms_mirrors_the_comparison(a: list[float], data: st.DataObject) -> None:
    # With the same seed the resampled index sets are identical, so every bootstrap
    # statistic simply negates and the interval reflects about zero.
    b = data.draw(st.lists(_FINITE, min_size=len(a), max_size=len(a)))
    forward = paired_bootstrap_diff(a, b, n_boot=40, seed=5)
    reverse = paired_bootstrap_diff(b, a, n_boot=40, seed=5)
    # Relative, not absolute. The symmetry is exact in real arithmetic and holds to floating
    # point in ours; an absolute 1e-9 asserts something stronger than float64 can deliver on
    # values of the magnitude hypothesis generates here, and it fails on inputs around 1e6 --
    # a property of the tolerance rather than of the statistic.
    assert reverse.diff == pytest.approx(-forward.diff, rel=1e-9, abs=1e-9)
    assert reverse.ci.low == pytest.approx(-forward.ci.high, rel=1e-9, abs=1e-9)
    assert reverse.ci.high == pytest.approx(-forward.ci.low, rel=1e-9, abs=1e-9)
    assert (reverse.wins, reverse.losses) == (forward.losses, forward.wins)


# --------------------------------------------------------------------------------------
# mcnemar_exact
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        # p = min(1, 2 * sum_{k=0}^{min(b,c)} C(n, k) / 2**n), n = b + c.
        (0, 0, 1.0),  # no discordant pairs: 2 * C(0,0)/1 = 2, capped at 1
        (1, 0, 1.0),  # 2 * 1/2
        (2, 0, 0.5),  # 2 * 1/4
        (3, 0, 0.25),  # 2 * 1/8
        (5, 0, 0.0625),  # 2 * 1/32
        (6, 0, 0.03125),  # 2 * 1/64
        (4, 1, 0.375),  # 2 * (1 + 5)/32
        (10, 2, 0.03857421875),  # 2 * (1 + 12 + 66)/4096
        (12, 4, 0.076812744140625),  # 2 * (1 + 16 + 120 + 560 + 1820)/65536
        (1, 1, 1.0),  # 2 * (1 + 2)/4 = 1.5, capped
        (3, 2, 1.0),  # 2 * (1 + 5 + 10)/32 = 1.0
    ],
)
def test_mcnemar_matches_hand_computed_binomial_tails(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(expected)


def test_mcnemar_with_no_discordant_pairs_is_one() -> None:
    # The degenerate case: nothing distinguishes the two systems, so there is no evidence.
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_is_not_the_chi_square_approximation() -> None:
    # The uncorrected chi-square statistic for (5, 0) is 5, giving p = 0.0253; the exact
    # test gives 0.0625. Only one of those is below a 0.05 threshold.
    assert mcnemar_exact(5, 0) == 0.0625


@pytest.mark.parametrize(("b", "c"), [(-1, 0), (0, -1), (-2, -3)])
def test_mcnemar_rejects_negative_counts(b: int, c: int) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        mcnemar_exact(b, c)


def test_mcnemar_survives_counts_that_would_overflow_a_float() -> None:
    # 2**900 is far past the float range; the exact rational arithmetic must not care.
    p = mcnemar_exact(500, 400)
    assert 0.0 < p < 0.01


@given(b=_COUNTS, c=_COUNTS)
def test_mcnemar_is_symmetric_in_its_two_counts(b: int, c: int) -> None:
    # The null hypothesis has no preferred direction, so neither can the p-value.
    assert mcnemar_exact(b, c) == mcnemar_exact(c, b)


@given(b=_COUNTS, c=_COUNTS)
def test_mcnemar_returns_a_probability(b: int, c: int) -> None:
    assert 0.0 <= mcnemar_exact(b, c) <= 1.0


@given(n=st.integers(min_value=2, max_value=60), data=st.data())
def test_a_more_lopsided_split_is_never_less_significant(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(min_value=0, max_value=n // 2 - 1))
    assert mcnemar_exact(k, n - k) <= mcnemar_exact(k + 1, n - k - 1)


# --------------------------------------------------------------------------------------
# wilson_interval
# --------------------------------------------------------------------------------------


def test_wilson_matches_the_textbook_interval() -> None:
    interval = wilson_interval(5, 10)
    assert interval.point == 0.5
    assert interval.low == pytest.approx(0.2366, abs=1e-4)
    assert interval.high == pytest.approx(0.7634, abs=1e-4)


def test_wilson_does_not_claim_certainty_from_a_perfect_score() -> None:
    # Wald would report [1.0, 1.0] here, which is the reason this function exists.
    interval = wilson_interval(20, 20)
    assert interval.point == 1.0
    assert interval.high == 1.0
    assert interval.low == pytest.approx(0.8389, abs=1e-4)
    assert interval.width > 0.0


def test_wilson_is_bounded_below_at_zero_successes() -> None:
    interval = wilson_interval(0, 20)
    assert interval.low == 0.0
    assert interval.high == pytest.approx(0.1611, abs=1e-4)


def test_wilson_narrows_as_evidence_accumulates() -> None:
    assert wilson_interval(50, 100).width < wilson_interval(5, 10).width


def test_wilson_widens_at_a_higher_confidence_level() -> None:
    assert wilson_interval(5, 10, alpha=0.01).width > wilson_interval(5, 10, alpha=0.10).width


@pytest.mark.parametrize("n", [0, -3])
def test_wilson_needs_at_least_one_trial(n: int) -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        wilson_interval(0, n)


@pytest.mark.parametrize(("successes", "n"), [(-1, 10), (11, 10)])
def test_wilson_rejects_impossible_success_counts(successes: int, n: int) -> None:
    with pytest.raises(ValueError, match="successes must lie"):
        wilson_interval(successes, n)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 3.0])
def test_wilson_rejects_an_impossible_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must lie"):
        wilson_interval(5, 10, alpha=alpha)


@given(n=st.integers(min_value=1, max_value=500), data=st.data())
def test_a_wilson_interval_always_contains_the_observed_proportion(
    n: int, data: st.DataObject
) -> None:
    # phat solves the score equation with a statistic of zero, so it can never fall outside.
    successes = data.draw(st.integers(min_value=0, max_value=n))
    interval = wilson_interval(successes, n)
    assert interval.contains(interval.point)
    assert 0.0 <= interval.low <= interval.high <= 1.0


@given(n=st.integers(min_value=1, max_value=500), data=st.data())
def test_wilson_is_symmetric_under_relabelling_success_as_failure(
    n: int, data: st.DataObject
) -> None:
    successes = data.draw(st.integers(min_value=0, max_value=n))
    forward = wilson_interval(successes, n)
    flipped = wilson_interval(n - successes, n)
    assert forward.low == pytest.approx(1.0 - flipped.high, abs=1e-12)
    assert forward.high == pytest.approx(1.0 - flipped.low, abs=1e-12)


# --------------------------------------------------------------------------------------
# non_inferiority and the release gate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "margin", "expected"),
    [
        (-0.01, 0.05, 0.05, True),
        (-0.04999, 0.02, 0.05, True),
        (0.01, 0.09, 0.05, True),
        (-0.05, 0.02, 0.05, False),  # exactly on the margin is not above it
        (-0.06, 0.20, 0.05, False),
        (-0.90, -0.10, 0.05, False),
    ],
)
def test_non_inferiority_looks_only_at_the_lower_bound(
    low: float, high: float, margin: float, expected: bool
) -> None:
    interval = Interval(point=(low + high) / 2, low=low, high=high)
    assert non_inferiority(interval, margin) is expected


def test_a_wide_interval_fails_non_inferiority_however_good_the_point_estimate() -> None:
    # An underpowered run cannot make a positive claim, which is the point of the test.
    interval = Interval(point=0.30, low=-0.40, high=0.90, n=4)
    assert not non_inferiority(interval, 0.05)


@pytest.mark.parametrize("margin", [0.0, -0.01])
def test_a_non_positive_margin_is_refused(margin: float) -> None:
    with pytest.raises(ValueError, match="margin must be strictly positive"):
        non_inferiority(Interval(point=0.0, low=-0.1, high=0.1), margin)


def test_decide_release_promotes_a_non_inferior_candidate() -> None:
    result = decide_release(Interval(point=0.01, low=-0.01, high=0.03), margin=0.05)
    assert result.decision is Decision.PROMOTE
    assert result.promote
    assert "non-inferior" in result.reason


def test_decide_release_rejects_a_clear_regression() -> None:
    result = decide_release(Interval(point=-0.20, low=-0.30, high=-0.10), margin=0.05)
    assert result.decision is Decision.REJECT
    assert not result.promote
    assert "regression" in result.reason


def test_decide_release_holds_when_the_interval_straddles_the_margin() -> None:
    result = decide_release(Interval(point=-0.02, low=-0.12, high=0.08, n=40), margin=0.05)
    assert result.decision is Decision.HOLD
    assert "inconclusive" in result.reason
    assert "40" in result.reason


def test_the_reject_boundary_is_inclusive() -> None:
    # An interval sitting exactly on the margin at both ends is a regression, not a hold.
    result = decide_release(Interval(point=-0.07, low=-0.09, high=-0.05), margin=0.05)
    assert result.decision is Decision.REJECT


def test_a_release_decision_carries_the_evidence_it_was_made_from() -> None:
    interval = Interval(point=0.01, low=-0.01, high=0.03, alpha=0.10, n=80)
    result = decide_release(interval, margin=0.05)
    assert result.ci == interval
    assert result.margin == 0.05
    assert "90%" in result.reason


def test_a_release_decision_is_frozen() -> None:
    result = decide_release(Interval(point=0.0, low=-0.01, high=0.01), margin=0.05)
    field = "decision"
    with pytest.raises(ValidationError):
        setattr(result, field, Decision.REJECT)


def test_decision_values_are_stable_strings() -> None:
    # They are written into result files, so renaming one is a breaking change.
    assert [d.value for d in Decision] == ["promote", "hold", "reject"]


@given(
    low=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    width=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    margin=st.floats(min_value=0.001, max_value=0.5, allow_nan=False),
)
def test_exactly_one_release_decision_applies(low: float, width: float, margin: float) -> None:
    interval = Interval(point=low + width / 2, low=low, high=low + width)
    result = decide_release(interval, margin=margin)
    assert isinstance(result, ReleaseDecision)
    assert (result.decision is Decision.PROMOTE) == non_inferiority(interval, margin)
    if result.decision is Decision.REJECT:
        assert interval.high <= -margin
    if result.decision is Decision.HOLD:
        assert interval.low <= -margin < interval.high
    assert result.reason


@given(
    a=st.lists(st.sampled_from([0.0, 1.0]), min_size=4, max_size=20),
    data=st.data(),
)
def test_the_gate_agrees_with_the_paired_comparison_it_is_fed(
    a: list[float], data: st.DataObject
) -> None:
    b = data.draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=len(a), max_size=len(a)))
    paired = paired_bootstrap_diff(a, b, n_boot=60, seed=3)
    result = decide_release(paired.ci, margin=0.10)
    assert result.promote == (paired.ci.low > -0.10)
    assert math.isfinite(result.ci.width)


def test_a_degenerate_sample_gives_a_zero_width_interval_not_an_error() -> None:
    """Every observation identical is not a pathological input; it is a paired tie.

    Two arms that differ by the same amount on every task make every bootstrap replicate
    equal, so the two percentiles are the same number computed two different ways. Linear
    interpolation can put them an ulp apart in the wrong order, and `Interval` refuses an
    inverted interval, so the comparison raised instead of reporting a zero-width interval.
    Found by hypothesis on the sibling implementation in `sft-dpo-alignment`.
    """
    identical = [0.24750492437346597] * 32
    interval = bootstrap_ci(identical, n_boot=64, seed=0)
    assert interval.low <= interval.point <= interval.high
    assert interval.high - interval.low == pytest.approx(0.0, abs=1e-12)


def test_a_paired_comparison_of_two_arms_that_differ_identically_does_not_raise() -> None:
    """The route the degenerate case actually arrives by."""
    a = [0.1120596935642216] * 24
    b = [0.3595646179376875] * 24
    result = paired_bootstrap_diff(a, b, n_boot=64, seed=0)
    assert result.ci.low <= result.ci.point <= result.ci.high
    assert result.ci.high - result.ci.low == pytest.approx(0.0, abs=1e-12)
    assert result.wins == 0
    assert result.losses == len(a)
