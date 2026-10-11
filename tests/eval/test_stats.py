"""stats.py: the bootstrap interval's sanity and the permutation test's calibration.

The interval must be degenerate on a constant sample, contain the mean, and
separate two clearly different samples. The test must return p near 1 for
identical samples and p near 0 for clearly different ones. Both must be
deterministic given a seed.
"""

from __future__ import annotations

import statistics

import pytest

from entente.eval.stats import bootstrap_ci, permutation_test


def test_bootstrap_ci_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([])


def test_bootstrap_ci_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=1.5)


def test_bootstrap_ci_is_degenerate_on_a_constant_sample() -> None:
    lo, hi = bootstrap_ci([0.5] * 20, resamples=500)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_ci_contains_the_sample_mean() -> None:
    samples = [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.2]
    lo, hi = bootstrap_ci(samples, resamples=2000)
    assert lo <= statistics.fmean(samples) <= hi


def test_bootstrap_ci_is_deterministic_given_a_seed() -> None:
    samples = [0.1, 0.9, 0.3, 0.7, 0.5]
    first = bootstrap_ci(samples, resamples=500, seed=3)
    second = bootstrap_ci(samples, resamples=500, seed=3)
    assert first == second


def test_bootstrap_ci_separates_two_clearly_different_samples() -> None:
    strong = [0.85, 0.9, 0.88, 0.92, 0.87, 0.91, 0.89] * 5
    weak = [0.1, 0.15, 0.12, 0.08, 0.11, 0.14, 0.09] * 5
    strong_lo, _ = bootstrap_ci(strong, resamples=2000)
    _, weak_hi = bootstrap_ci(weak, resamples=2000)
    assert strong_lo > weak_hi  # CI-separated, the tournament's acceptance bar


def test_permutation_test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        permutation_test([], [1.0])


def test_permutation_test_rejects_non_positive_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        permutation_test([1.0], [2.0], iterations=0)


def test_permutation_test_p_value_is_high_for_identical_samples() -> None:
    a = [0.4, 0.5, 0.6, 0.5, 0.4, 0.6]
    p = permutation_test(a, list(a), iterations=2000)
    assert p > 0.5


def test_permutation_test_p_value_is_low_for_clearly_different_samples() -> None:
    strong = [0.85, 0.9, 0.88, 0.92, 0.87, 0.91, 0.89] * 5
    weak = [0.1, 0.15, 0.12, 0.08, 0.11, 0.14, 0.09] * 5
    p = permutation_test(strong, weak, iterations=2000)
    assert p < 0.01


def test_permutation_test_is_deterministic_given_a_seed() -> None:
    a, b = [0.1, 0.5, 0.9], [0.2, 0.4, 0.8]
    first = permutation_test(a, b, iterations=500, seed=1)
    second = permutation_test(a, b, iterations=500, seed=1)
    assert first == second
