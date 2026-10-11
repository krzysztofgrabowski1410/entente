"""Pin message parsing against the fixed proposal grammar, and the reweighting."""

from __future__ import annotations

import pytest

from entente.engine.orders import parse_order
from entente.negotiation.generate import StubProvider
from entente.negotiation.intents import Intent
from entente.negotiation.parse import MessageParseError, parse_message, reweight_from_intent
from entente.search.candidates import CandidateSet


def _intent(rationale: str = "") -> Intent:
    return Intent(
        proposer="GERMANY",
        ally="RUSSIA",
        phase="S1901M",
        proposer_action=(parse_order("A BER - SIL"), parse_order("A MUN H")),
        ally_action=(parse_order("A WAR - GAL"),),
        rationale=rationale,
    )


class TestParseMessage:
    """Reading a stub-generated message back to the intent that wrote it."""

    def test_round_trips_stub_generated_text(self) -> None:
        intent = _intent("stay out of each other's way")
        text = StubProvider().generate(intent)
        assert parse_message(text) == intent

    def test_round_trips_without_a_rationale(self) -> None:
        intent = _intent()
        text = StubProvider().generate(intent)
        parsed = parse_message(text)
        assert parsed == intent
        assert parsed.rationale == ""

    def test_tolerates_surrounding_whitespace(self) -> None:
        intent = _intent("trimmed")
        text = f"  {StubProvider().generate(intent)}  \n"
        assert parse_message(text) == intent

    def test_rejects_text_outside_the_grammar(self) -> None:
        with pytest.raises(MessageParseError):
            parse_message("Let's ally against Turkey, no take backs.")

    def test_rejects_an_unbalanced_bracket(self) -> None:
        with pytest.raises(MessageParseError):
            parse_message(
                "PROPOSAL from GERMANY to RUSSIA for S1901M: "
                "GERMANY orders [A BER - SIL; RUSSIA orders [A WAR - GAL]."
            )


def _candidate_set(power: str, order_sets: list[list[str]], anchor: list[float]) -> CandidateSet:
    return CandidateSet(
        power=power,
        order_sets=tuple(tuple(parse_order(text) for text in sheet) for sheet in order_sets),
        log_probs=(0.0,) * len(anchor),
        anchor=tuple(anchor),
    )


class TestReweightFromIntent:
    """How a parsed intent tilts a power's candidate anchor."""

    def test_boosts_the_matching_candidate_and_renormalizes(self) -> None:
        candidates = _candidate_set(
            "GERMANY",
            [["A BER - SIL", "A MUN H"], ["A BER H", "A MUN H"]],
            [0.5, 0.5],
        )
        intent = _intent()
        result = reweight_from_intent(candidates, intent, boost=4.0)
        # tilted = [0.5*4, 0.5*1] = [2.0, 0.5], sum = 2.5
        assert result == pytest.approx((2.0 / 2.5, 0.5 / 2.5), abs=1e-12)

    def test_matching_is_order_independent_within_a_sheet(self) -> None:
        # Same two orders, written in the opposite sequence: still one sheet.
        candidates = _candidate_set(
            "GERMANY",
            [["A MUN H", "A BER - SIL"], ["A BER H", "A MUN H"]],
            [0.5, 0.5],
        )
        result = reweight_from_intent(candidates, _intent(), boost=4.0)
        assert result[0] > result[1]

    def test_returns_the_anchor_unchanged_for_an_uninvolved_power(self) -> None:
        candidates = _candidate_set("FRANCE", [["A PAR H"], ["A PAR - BUR"]], [0.5, 0.5])
        assert reweight_from_intent(candidates, _intent()) == candidates.anchor

    def test_returns_the_anchor_unchanged_when_nothing_matches(self) -> None:
        candidates = _candidate_set(
            "GERMANY",
            [["A BER H", "A MUN H"], ["A BER - KIE", "A MUN H"]],
            [0.5, 0.5],
        )
        assert reweight_from_intent(candidates, _intent()) == pytest.approx(
            candidates.anchor, abs=1e-12
        )
