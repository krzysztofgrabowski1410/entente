"""Hand-computed exactness for the one reweighting hook outside code may use.

Every expectation is worked out by hand, applying docs/pikl.tex's
validate-before-use discipline to the single seam ``negotiation/`` reaches
through.
"""

from __future__ import annotations

import pytest

from entente.search.pikl import reweight_anchor


class TestReweightAnchor:
    """The tilt itself: exact weights, exact renormalization, exact limits."""

    def test_tilts_and_renormalizes(self) -> None:
        # tilted = [0.5*2, 0.3*1, 0.2*1] = [1.0, 0.3, 0.2], sum = 1.5
        result = reweight_anchor([0.5, 0.3, 0.2], [2.0, 1.0, 1.0])
        assert result == pytest.approx([1.0 / 1.5, 0.3 / 1.5, 0.2 / 1.5], abs=1e-12)

    def test_all_ones_is_identity_up_to_renormalization(self) -> None:
        anchor = [0.5, 0.3, 0.2]
        assert reweight_anchor(anchor, [1.0, 1.0, 1.0]) == pytest.approx(anchor, abs=1e-12)

    def test_zero_bias_everywhere_falls_back_to_the_original_anchor(self) -> None:
        anchor = [0.5, 0.3, 0.2]
        assert reweight_anchor(anchor, [0.0, 0.0, 0.0]) == pytest.approx(anchor, abs=1e-12)

    def test_boost_dominates_as_it_grows(self) -> None:
        anchor = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        result = reweight_anchor(anchor, [1000.0, 1.0, 1.0])
        assert result[0] > 0.99


class TestReweightAnchorValidation:
    """What the hook refuses, so a bad bias never reaches a learner."""

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            reweight_anchor([0.5, 0.5], [1.0, 1.0, 1.0])

    def test_rejects_negative_bias(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            reweight_anchor([0.5, 0.5], [1.0, -1.0])
