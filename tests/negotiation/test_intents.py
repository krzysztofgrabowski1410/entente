"""Intent schema: validation, `action_for`, and dict/JSON round trips."""

from __future__ import annotations

import pytest

from entente.engine.orders import parse_order
from entente.negotiation.intents import Intent


def _intent(**overrides: object) -> Intent:
    defaults: dict[str, object] = {
        "proposer": "GERMANY",
        "ally": "RUSSIA",
        "phase": "S1901M",
        "proposer_action": (parse_order("A BER - SIL"), parse_order("A MUN H")),
        "ally_action": (parse_order("A WAR - GAL"),),
        "rationale": "take Galicia instead",
    }
    defaults.update(overrides)
    return Intent(**defaults)  # type: ignore[arg-type]


class TestIntentValidation:
    """The fields an intent refuses to be built without."""

    def test_proposer_and_ally_must_differ(self) -> None:
        with pytest.raises(ValueError, match="different powers"):
            _intent(ally="GERMANY")

    def test_phase_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="phase"):
            _intent(phase="")

    def test_proposer_action_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="proposer_action"):
            _intent(proposer_action=())

    def test_ally_action_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="ally_action"):
            _intent(ally_action=())


class TestActionFor:
    """Which action an intent names for which power."""

    def test_returns_proposer_action_for_proposer(self) -> None:
        intent = _intent()
        assert intent.action_for("GERMANY") == intent.proposer_action

    def test_returns_ally_action_for_ally(self) -> None:
        intent = _intent()
        assert intent.action_for("RUSSIA") == intent.ally_action

    def test_returns_none_for_an_uninvolved_power(self) -> None:
        assert _intent().action_for("FRANCE") is None


class TestSerialization:
    """An intent's round trip through dict and JSON."""

    def test_to_dict_formats_orders_as_canonical_text(self) -> None:
        data = _intent().to_dict()
        assert data["proposer_action"] == ["A BER - SIL", "A MUN H"]
        assert data["ally_action"] == ["A WAR - GAL"]
        assert data["rationale"] == "take Galicia instead"

    def test_dict_round_trip(self) -> None:
        intent = _intent()
        assert Intent.from_dict(intent.to_dict()) == intent

    def test_json_round_trip(self) -> None:
        intent = _intent()
        assert Intent.from_json(intent.to_json()) == intent

    def test_from_dict_defaults_rationale_to_empty(self) -> None:
        intent = _intent()
        data = intent.to_dict()
        del data["rationale"]
        assert Intent.from_dict(data).rationale == ""
