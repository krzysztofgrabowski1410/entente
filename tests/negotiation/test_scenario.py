"""Scenario loading and the scripted piKL wiring behind the negotiation demo.

These exercise the real engine's adjudication and the real piKL search, with
no network and no torch, exactly as the committed scenarios run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entente.engine.map.types import Location, UnitKind
from entente.engine.orders import parse_order
from entente.engine.state import Unit
from entente.negotiation.scenario import (
    Scenario,
    _order_succeeded,
    load_scenario,
    make_utility_fn,
    new_game,
    run_predictions,
)
from entente.search.candidates import CandidateSet

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "src/entente/negotiation/scenarios"


class TestLoadScenario:
    """Loading the committed scenario files."""

    @pytest.mark.parametrize("name", ["s1", "s2", "s3"])
    def test_loads_every_committed_scenario(self, name: str) -> None:
        scenario = load_scenario(SCENARIOS_DIR / f"{name}.json")
        assert scenario.proposer in scenario.candidates
        assert scenario.ally in scenario.candidates
        for power in (scenario.proposer, scenario.ally):
            candidates = scenario.candidates[power]
            assert len(candidates.order_sets) == len(candidates.anchor)
            assert sum(candidates.anchor) == pytest.approx(1.0, abs=1e-9)
        proposer_candidates = scenario.candidates[scenario.proposer]
        assert 0 <= scenario.true_move_index < len(proposer_candidates.order_sets)

    def test_rejects_mismatched_order_sets_and_anchor_lengths(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(
            """
            {"phase": "S1901M", "self_power": "X", "proposer": "A", "ally": "B",
             "message": "m", "true_move_index": 0,
             "candidates": {"A": {"order_sets": [["A PAR H"]], "anchor": [0.5, 0.5]},
                            "B": {"order_sets": [["A MAR H"]], "anchor": [1.0]}}}
            """,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="same length"):
            load_scenario(bad)


class TestOrderSucceeded:
    """The success test the toy utility scores an order by."""

    def test_move_succeeds_when_it_reaches_its_destination(self) -> None:
        order = parse_order("A PAR - BUR")
        arrived = [Unit("FRANCE", UnitKind.ARMY, Location("BUR"))]
        assert _order_succeeded(order, "FRANCE", arrived) is True

    def test_move_fails_when_the_unit_never_arrives(self) -> None:
        order = parse_order("A PAR - BUR")
        stayed_home = [Unit("FRANCE", UnitKind.ARMY, Location("PAR"))]
        assert _order_succeeded(order, "FRANCE", stayed_home) is False

    def test_hold_succeeds_only_if_not_displaced(self) -> None:
        order = parse_order("A PAR H")
        held = [Unit("FRANCE", UnitKind.ARMY, Location("PAR"))]
        displaced = [Unit("FRANCE", UnitKind.ARMY, Location("BUR"))]
        assert _order_succeeded(order, "FRANCE", held)
        assert not _order_succeeded(order, "FRANCE", displaced)


class TestMakeUtilityFn:
    """The utility oracle built over real adjudication."""

    def test_uncontested_moves_all_succeed(self) -> None:
        # Each power's other units are absent from its sheet here and so
        # default to holding in place (branch_game's contract) -- BUR and ION
        # are clear of both those defaults, so neither move is contested.
        game = new_game()
        candidates = [
            CandidateSet(
                power="FRANCE",
                order_sets=((parse_order("A PAR - BUR"),),),
                log_probs=(0.0,),
                anchor=(1.0,),
            ),
            CandidateSet(
                power="ITALY",
                order_sets=((parse_order("F NAP - ION"),),),
                log_probs=(0.0,),
                anchor=(1.0,),
            ),
        ]
        utility_fn = make_utility_fn(game, ("FRANCE", "ITALY"), candidates)
        (utilities,) = utility_fn([(0, 0)])
        assert utilities == (1.0, 1.0)

    def test_two_powers_bouncing_into_the_same_province_both_fail(self) -> None:
        game = new_game()
        candidates = [
            CandidateSet(
                power="GERMANY",
                order_sets=((parse_order("A BER - SIL"),),),
                log_probs=(0.0,),
                anchor=(1.0,),
            ),
            CandidateSet(
                power="RUSSIA",
                order_sets=((parse_order("A WAR - SIL"),),),
                log_probs=(0.0,),
                anchor=(1.0,),
            ),
        ]
        utility_fn = make_utility_fn(game, ("GERMANY", "RUSSIA"), candidates)
        (utilities,) = utility_fn([(0, 0)])
        assert utilities == (0.0, 0.0)


class TestRunPredictions:
    """The two piKL runs a scenario compares, against the committed numbers."""

    @pytest.mark.parametrize(
        ("name", "expect_improves"),
        [("s1", True), ("s2", True), ("s3", False)],
    )
    def test_matches_the_committed_probe_numbers(self, name: str, expect_improves: bool) -> None:
        scenario = load_scenario(SCENARIOS_DIR / f"{name}.json")
        result = run_predictions(scenario)
        baseline = result.baseline.mean_strategies[0][scenario.true_move_index]
        conditioned = result.conditioned.mean_strategies[0][scenario.true_move_index]
        if expect_improves:
            assert conditioned > baseline
        else:
            assert conditioned < baseline

    def test_s1_is_deterministic_and_reproducible(self) -> None:
        scenario = load_scenario(SCENARIOS_DIR / "s1.json")
        first = run_predictions(scenario)
        second = run_predictions(scenario)
        assert first.baseline.mean_strategies == second.baseline.mean_strategies
        assert first.conditioned.mean_strategies == second.conditioned.mean_strategies

    def test_conditioning_actually_changes_the_proposer_prediction(self) -> None:
        scenario = load_scenario(SCENARIOS_DIR / "s1.json")
        result = run_predictions(scenario)
        assert result.baseline.mean_strategies[0] != result.conditioned.mean_strategies[0]


class TestScenarioValidation:
    """What an incomplete scenario is rejected for."""

    def test_rejects_a_placeholder_dataclass_without_required_fields(self) -> None:
        with pytest.raises(TypeError):
            Scenario()  # type: ignore[call-arg]
