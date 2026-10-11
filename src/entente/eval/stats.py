"""Bootstrap confidence intervals and a permutation test, on the stdlib alone.

These are the two significance tools the evaluation protocol calls for when
comparing two agents' per-game scores, such as the Sum-of-Squares shares from
:mod:`entente.eval.scoring`. Both are built from ``random`` and ``statistics``,
with no numpy or scipy dependency, which keeps the core install stdlib-only.
"""

from __future__ import annotations

import random
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute a percentile bootstrap confidence interval for the mean of ``samples``.

    ``len(samples)`` values are resampled with replacement ``resamples`` times,
    each resample's mean is taken, and the ``confidence`` central interval of
    that distribution of means is reported. Two agents' scores are
    "CI-separated" when their intervals do not overlap.
    """
    if not samples:
        raise ValueError("samples must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    n = len(samples)
    means = sorted(
        statistics.fmean(samples[rng.randrange(n)] for _ in range(n)) for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = max(int(alpha * resamples), 0)
    hi = min(int((1.0 - alpha) * resamples), resamples - 1)
    return means[lo], means[hi]


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> float:
    """Compute the two-sided permutation-test p-value for ``mean(a) != mean(b)``.

    Both samples are pooled, the pooled values are repeatedly reshuffled into
    two groups of the original sizes, and the reported figure is the fraction of
    reshuffles whose difference in means is at least as large as the one
    actually observed. This is the standard nonparametric test to reach for when
    no distributional assumption is warranted, which is the case here: per-game
    scores are nowhere near normal, and a solo rate is mostly 0s and 1s.
    """
    if not a or not b:
        raise ValueError("a and b must be non-empty")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    rng = random.Random(seed)
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    pooled = list(a) + list(b)
    split = len(a)
    at_least_as_extreme = 0
    for _ in range(iterations):
        rng.shuffle(pooled)
        diff = abs(statistics.fmean(pooled[:split]) - statistics.fmean(pooled[split:]))
        if diff >= observed - 1e-12:
            at_least_as_extreme += 1
    return at_least_as_extreme / iterations
