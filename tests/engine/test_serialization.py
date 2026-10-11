"""Pin the save format's contract and its failure modes.

Version 1 is the classic document and version 2 adds the variant fragment;
both are checked here, along with what a malformed document does.

The hypothesis round-trip over whole random games lives in
tests/properties/test_game_properties.py; these are the deterministic
contract cases (mid-retreat saves, tampering, version gates, pending grants).
"""

from __future__ import annotations

import pytest

from entente.engine.game import Game
from entente.engine.orders import parse_order, parse_retreat_order
from entente.engine.serialization import (
    SerializationError,
    game_from_json,
    game_from_text,
    game_to_json,
    game_to_text,
)


def scripted_game() -> Game:
    """Play a fixed opening that leaves the game mid-retreat, with history behind it."""
    game = Game.new("standard", "standard")
    game.submit("GERMANY", [parse_order("A MUN - BUR")])
    game.adjudicate()
    game.submit("FRANCE", [parse_order("A PAR - BUR"), parse_order("A MAR S A PAR - BUR")])
    game.adjudicate()
    return game  # at F1901R with a pending German retreat decision


class TestRoundTrip:
    """A game survives a save and a load unchanged."""

    def test_mid_retreat_save_restores_the_movement_context(self) -> None:
        game = scripted_game()
        game.submit("GERMANY", [parse_retreat_order("A BUR - MUN")])
        text = game_to_text(game)
        restored = game_from_text(text)
        assert restored.phase == "F1901R"
        assert restored.pending_sheets == game.pending_sheets
        assert [str(d.unit) for d in restored.pending_dislodgements] == ["A BUR"]
        # The restored game continues exactly like the original.
        game.adjudicate()
        restored.adjudicate()
        assert restored.state == game.state
        assert restored.ownership == game.ownership
        assert restored.phase == game.phase

    def test_text_form_is_a_fixpoint(self) -> None:
        game = scripted_game()
        text = game_to_text(game)
        assert game_to_text(game_from_text(text)) == text

    def test_finished_game_round_trips(self) -> None:
        game = scripted_game()
        game.adjudicate()  # no retreat submitted: the default disband applies
        restored = game_from_text(game_to_text(game))
        assert restored.state == game.state
        assert [record.phase for record in restored.history] == [
            record.phase for record in game.history
        ]


class TestPendingGrants:
    """An unflushed grant is written, restored, and left out when empty."""

    """A knowledge grant issued in the phase that has not adjudicated yet.

    ``Game.grant`` applies the fact immediately but files the record in
    ``_pending_grants``, which only reaches the phase record at
    :meth:`Game.adjudicate`. The live effect is already visible in the
    hooks' snapshot fragment, so a save taken in that window must carry the
    grant too -- otherwise the replay cannot reproduce its own ``"l2"``.
    """

    @staticmethod
    def granted_game() -> Game:
        """Build a seeded Beleriand game carrying one unflushed grant."""
        game = Game.new("beleriand", "beleriand", seed=7)
        game.grant("TELERI", "MEN", "fleet-building")
        return game  # at S25M, grant live but not yet flushed to history

    def test_grant_before_adjudication_survives_round_trip(self) -> None:
        game = self.granted_game()
        restored = game_from_text(game_to_text(game))
        assert "MEN" in restored.hooks.fleet_knowledge
        assert restored.pending_grants == game.pending_grants
        # and the restored game continues identically
        game.adjudicate()
        restored.adjudicate()
        assert restored.state == game.state
        assert [record.grants for record in restored.history] == [
            record.grants for record in game.history
        ]

    def test_text_form_is_a_fixpoint_with_a_pending_grant(self) -> None:
        text = game_to_text(self.granted_game())
        assert game_to_text(game_from_text(text)) == text

    def test_pending_grants_replay_in_order(self) -> None:
        # The second grant is legal only once the first has landed: Men can
        # pass on the fleet knowledge only after receiving it.
        game = Game.new("beleriand", "beleriand", seed=7)
        game.grant("TELERI", "MEN", "fleet-building")
        game.grant("MEN", "NAUGRIM", "fleet-building")
        restored = game_from_text(game_to_text(game))
        assert {"TELERI", "MEN", "NAUGRIM"} <= set(restored.hooks.fleet_knowledge)

    def test_adjudication_moves_the_grant_into_the_phase_record(self) -> None:
        # Guards against double application: once flushed, the grant lives in
        # phases[] and the pending block must be gone.
        game = self.granted_game()
        game.adjudicate()
        data = game_to_json(game)
        assert "pending_grants" not in data
        assert data["phases"][0]["grants"] == [["TELERI", "MEN", "fleet-building"]]
        assert game_from_json(data).hooks.fleet_knowledge == game.hooks.fleet_knowledge

    def test_absent_when_nothing_is_pending(self) -> None:
        # The field is written only when it is non-empty, so a game with
        # nothing pending serializes without it at all.
        assert "pending_grants" not in game_to_json(Game.new("standard", "standard"))
        assert "pending_grants" not in game_to_json(Game.new("beleriand", "beleriand", seed=7))
        assert "pending_grants" not in game_to_json(scripted_game())

    def test_rejects_malformed_pending_grant(self) -> None:
        data = game_to_json(self.granted_game())
        data["pending_grants"] = [["TELERI", "MEN"]]
        with pytest.raises(SerializationError, match="malformed grant entry"):
            game_from_json(data)

    def test_rejects_a_pending_grant_the_rules_refuse(self) -> None:
        data = game_to_json(self.granted_game())
        data["pending_grants"] = [["TELERI", "MORGOTH", "fleet-building"]]
        with pytest.raises(SerializationError, match="pending grant rejected"):
            game_from_json(data)


class TestFailureModes:
    """A malformed document fails the load loudly."""

    def test_rejects_wrong_format_and_version(self) -> None:
        with pytest.raises(SerializationError, match="not an entente-game"):
            game_from_json({"format": "something-else", "version": 1})
        data = game_to_json(Game.new("standard", "standard"))
        data["version"] = 999
        with pytest.raises(SerializationError, match="unsupported version"):
            game_from_json(data)

    def test_rejects_bad_json_text(self) -> None:
        with pytest.raises(SerializationError, match="not JSON"):
            game_from_text("{nope")
        with pytest.raises(SerializationError, match="top-level"):
            game_from_text("[1, 2]")

    def test_rejects_unknown_ruleset(self) -> None:
        data = game_to_json(Game.new("standard", "standard"))
        data["ruleset"] = "no-such-rules"
        with pytest.raises(SerializationError, match="cannot start game"):
            game_from_json(data)

    def test_rejects_tampered_snapshot(self) -> None:
        game = scripted_game()
        data = game_to_json(game)
        data["snapshot"]["ownership"]["PAR"] = "GERMANY"
        with pytest.raises(SerializationError, match="snapshot mismatch"):
            game_from_json(data)

    def test_rejects_tampered_phase_sequence(self) -> None:
        game = scripted_game()
        data = game_to_json(game)
        data["phases"][0]["phase"] = "F1901M"
        with pytest.raises(SerializationError, match="replay expected phase"):
            game_from_json(data)

    def test_rejects_alien_auto_records(self) -> None:
        game = scripted_game()
        data = game_to_json(game)
        data["phases"].insert(1, {"phase": "S1901R", "auto": True, "sheets": {}})
        with pytest.raises(SerializationError, match="does not match recorded"):
            game_from_json(data)

    def test_rejects_pending_sheet_for_wrong_phase_kind(self) -> None:
        game = scripted_game()
        data = game_to_json(game)
        data["pending"] = {"GERMANY": ["Build A MUN"]}
        with pytest.raises(SerializationError):
            game_from_json(data)

    def test_refuses_saves_of_hand_constructed_games(self) -> None:
        # Only Game.new + adjudicate histories replay; a game built directly
        # around a mid-game position produces a save the verification must
        # refuse (documented format scope).
        from entente.engine.map import Location, UnitKind, load_map
        from entente.engine.phase import Phase, PhaseKind, Season
        from entente.engine.rules import STANDARD
        from entente.engine.state import State, Unit

        state = State.of("standard", "F1901M", [Unit("FRANCE", UnitKind.ARMY, Location("BUR"))])
        game = Game(
            STANDARD,
            load_map("standard"),
            state,
            {"PAR": "FRANCE"},
            Phase(1901, Season.FALL, PhaseKind.MOVEMENT),
        )
        with pytest.raises(SerializationError):
            game_from_text(game_to_text(game))
