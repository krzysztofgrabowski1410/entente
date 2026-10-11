"""Load a scripted scenario and wire it to piKL for the demo and the probe.

The toy utility's limits are set out in negotiation/README.md, Limitations.

Scenarios (``scenarios/*.json``) are scripted, small-scale stand-ins for a
full BC-driven piKL subgame (docs/pikl.tex): candidate order-sets and anchors
are authored directly rather than sampled from a trained policy, and the
utility oracle scores real 1-ply adjudication (via
:func:`entente.search.rollout.branch_game`) by order-success rather than a
value network -- both honestly documented simplifications that keep the demo
and the probe fully offline (no ``ml``/``llm`` group, no GPU, no checkpoint).
See ``negotiation/README.md`` Sec. Limitations. The subgame is exactly the two
named powers (``proposer``, ``ally``) -- piKL does not need every power in the
game, only the ones whose interaction is being modeled (
:func:`entente.search.rollout.branch_game` holds every absent power's units).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from entente.engine.game import Game
from entente.engine.orders import Move, Waive, parse_order
from entente.negotiation.parse import parse_message, reweight_from_intent
from entente.search.candidates import CandidateSet
from entente.search.pikl import PiklPlayer, PiklResult, run_pikl
from entente.search.rollout import branch_game

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.orders import AnyOrder, GameOrder
    from entente.engine.state import Unit
    from entente.search.pikl import ProfileUtilityFn

DEFAULT_ITERATIONS = 64
DEFAULT_LAM = 1.0


@dataclass(frozen=True, slots=True)
class Scenario:
    """Hold one scripted negotiation scenario.

    A scenario is a two-power piKL subgame over ``proposer`` and ``ally``, the
    scripted incoming ``message``, and the ground-truth ``true_move_index`` the
    probe checks predictions against, which indexes
    ``candidates[proposer].order_sets``.
    """

    name: str
    phase: str
    self_power: str
    proposer: str
    ally: str
    message: str
    true_move_index: int
    candidates: Mapping[str, CandidateSet]
    lam: float
    iterations: int
    note: str = ""


@dataclass(frozen=True, slots=True)
class PredictionRun:
    """Pair the unconditioned and intent-conditioned piKL results for one subgame.

    Both index their players as ``(proposer, ally)``, so
    ``mean_strategies[0]`` is always the proposer's and the two are directly
    comparable position by position.
    """

    baseline: PiklResult
    conditioned: PiklResult


def _load_candidate_set(power: str, data: Mapping[str, Any]) -> CandidateSet:
    # scenario files carry standard movement orders; anything else fails the
    # downstream legality validation, so the narrowing cast is safe
    order_sets = cast(
        "tuple[tuple[AnyOrder, ...], ...]",
        tuple(tuple(parse_order(text) for text in sheet) for sheet in data["order_sets"]),
    )
    anchor = tuple(float(p) for p in data["anchor"])
    if len(order_sets) != len(anchor):
        raise ValueError(f"{power}: order_sets and anchor must have the same length")
    return CandidateSet(
        power=power, order_sets=order_sets, log_probs=(0.0,) * len(anchor), anchor=anchor
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load one scenario JSON file (schema: module docstring + scenarios/*.json)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = {
        power: _load_candidate_set(power, entry) for power, entry in data["candidates"].items()
    }
    pikl_config = data.get("pikl", {})
    return Scenario(
        name=path.stem,
        phase=data["phase"],
        self_power=data["self_power"],
        proposer=data["proposer"],
        ally=data["ally"],
        message=data["message"],
        true_move_index=int(data["true_move_index"]),
        candidates=candidates,
        lam=float(pikl_config.get("lam", DEFAULT_LAM)),
        iterations=int(pikl_config.get("iterations", DEFAULT_ITERATIONS)),
        note=str(data.get("note", "")),
    )


def new_game() -> Game:
    """Build the shared board every scenario reasons over: a fresh standard start.

    Every scenario is scripted at S1901M. Only ``proposer`` and ``ally`` submit
    orders (:func:`make_utility_fn`); every other power takes the engine's
    phase default through :func:`~entente.search.rollout.branch_game`, which is
    a hold.
    """
    return Game.new("standard", "standard")


def _order_succeeded(order: GameOrder, power: str, units_after: Sequence[Unit]) -> bool:
    """Say whether the ordered unit ended this phase where the order intended it.

    A move must have reached its destination and a hold must not have been
    displaced; this is the toy utility's atomic signal. Movement-phase orders
    only, matching every scenario's Hold-and-Move candidates. A bare ``Waive``
    has no location and could only come from a mistake in a scenario file, not
    from a real input, so it fails loudly here rather than being scored as
    something, as :func:`entente.search.candidates._validate_legal` also does.
    """
    if isinstance(order, Waive):
        raise ValueError("scenarios are movement-phase only; a candidate order-set has a Waive")
    target = order.destination if isinstance(order, Move) else order.location
    return any(
        unit.power == power and unit.kind == order.unit_kind and unit.location == target
        for unit in units_after
    )


def make_utility_fn(
    game: Game, powers: Sequence[str], candidate_sets: Sequence[CandidateSet]
) -> ProfileUtilityFn:
    """Build a torch-free, engine-grounded one-ply utility oracle.

    A power's utility is the fraction of its own submitted orders that reached
    their written effect, scored against real adjudication: a scripted
    stand-in for the value head, so the demo needs no checkpoint. Every
    scenario's candidates are Hold and Move only, so a bounce is the only
    interaction available -- with no Support order, nothing can be dislodged --
    which is what keeps ``branch_game`` landing cleanly on the next movement
    phase every time.
    """
    if len(powers) != len(candidate_sets):
        raise ValueError("powers and candidate_sets must be parallel")

    def utility_fn(profiles: Sequence[tuple[int, ...]]) -> list[tuple[float, ...]]:
        results = []
        for profile in profiles:
            sheets = {
                power: candidate_sets[i].order_sets[profile[i]] for i, power in enumerate(powers)
            }
            branch = branch_game(game, sheets)
            row = []
            for power in powers:
                orders = sheets[power]
                successes = sum(
                    1 for order in orders if _order_succeeded(order, power, branch.state.units)
                )
                row.append(successes / len(orders))
            results.append(tuple(row))
        return results

    return utility_fn


def run_predictions(scenario: Scenario) -> PredictionRun:
    """Run piKL twice over the ``{proposer, ally}`` subgame, with and without the message.

    The first run uses the proposer's raw anchor and the second the anchor
    reweighted by parsing ``scenario.message``, so the difference between them
    is attributable to the message alone. ``mode="expected"`` is exact and
    affordable at this scenario's tiny scale, single-digit candidate counts per
    power, so both runs are fully deterministic and need no ``rng``.
    """
    powers = (scenario.proposer, scenario.ally)
    candidate_sets = [scenario.candidates[power] for power in powers]
    game = new_game()
    utility_fn = make_utility_fn(game, powers, candidate_sets)

    def run(anchors: Sequence[tuple[float, ...]]) -> PiklResult:
        players = [PiklPlayer(anchor=anchor, lam=scenario.lam) for anchor in anchors]
        return run_pikl(players, utility_fn, iterations=scenario.iterations, mode="expected")

    baseline = run([cs.anchor for cs in candidate_sets])

    intent = parse_message(scenario.message)
    reweighted_proposer_anchor = reweight_from_intent(candidate_sets[0], intent)
    conditioned = run([reweighted_proposer_anchor, candidate_sets[1].anchor])

    return PredictionRun(baseline=baseline, conditioned=conditioned)


__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_LAM",
    "PredictionRun",
    "Scenario",
    "load_scenario",
    "make_utility_fn",
    "new_game",
    "run_predictions",
]
