"""Command-line interface to the engine, the agents and the evaluation harness.

``entente adjudicate <game.json>`` loads a serialized game (the
:mod:`entente.engine.serialization` format, v1 or v2), adjudicates its current
phase using the pending sheets stored in the file, prints the phase log, and
writes the advanced game back (in place, or to ``--out``).

``entente play --agents random --seed 7 --max-year 1914`` plays a full game
between agents, printing a phase-by-phase log; ``--out`` saves the finished
game as serialization JSON (v2 when the ruleset ships a variant layer). The
run is fully deterministic in the seed. The year cap is
calendar-aware: on countdown rulesets (Beleriand counts 25 down to 0) an
explicit ``--max-year N`` stops once the year drops below ``N``, and the
default is no cap at all -- a countdown game ends when its own calendar runs
out, so an outside cap would only cut it short.

``entente beleriand-demo`` plays a full seeded Beleriand game to the Valar's
intervention, choreographed to parade the variant's engine-level mechanics.

``entente tournament configs/tournament/main.yaml`` runs a 1-vs-6 or
round-robin tournament from a YAML config (:mod:`entente.eval.config`),
prints a per-agent Sum-of-Squares/survival/solo summary with bootstrap CIs,
and (if the config sets ``output``) writes every game as JSONL.

``entente negotiate-demo <scenario.json>`` runs the language-model negotiation
prototype (:mod:`entente.negotiation`) on a scripted scenario. It is registered
through ``importlib`` rather than a static import, so this module never imports
:mod:`entente.negotiation` at all: the prototype is deletable by construction,
and deleting it silently drops the subcommand instead of leaving a dangling
import behind.
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from entente.agents import Agent
from entente.agents.registry import REGISTRY
from entente.engine.game import Game, GameError, PhaseRecord
from entente.engine.serialization import SerializationError, game_from_text, game_to_text
from entente.eval.config import ConfigError, load_tournament_config
from entente.eval.scoring import per_agent_solo, per_agent_sos, per_agent_survival
from entente.eval.stats import bootstrap_ci
from entente.eval.tournament import SeatedOutcome, one_vs_six, round_robin, write_seated_jsonl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.orders import GameOrder
    from entente.engine.state import Unit
    from entente.variants.beleriand.ledger import LedgerUnit


def _print_record(game: Game, record: PhaseRecord) -> None:
    auto = " (auto)" if record.auto else ""
    print(f"== {record.phase}{auto}")
    for grant in record.grants:
        print(f"  grant: {grant.giver} passes {grant.fact} to {grant.receiver}")
    for entry in record.resolutions:
        print(f"  {entry.power}: {entry.text} .. {entry.resolution.value}")
    for dislodgement in record.dislodgements:
        unit = dislodgement.unit
        if dislodgement.dislodger_origin is None:
            cause = "devastation"
        else:
            how = "by convoy" if dislodgement.by_convoy else "overland"
            cause = f"attacked from {dislodgement.dislodger_origin}, {how}"
        print(f"  dislodged: {unit.power} {unit} ({cause})")
    for unit in record.disbanded:
        print(f"  disbanded: {unit.power} {unit}")
    for unit in record.built:
        print(f"  built: {unit.power} {unit}")
    for unit in record.civil_disorder:
        print(f"  civil disorder removal: {unit.power} {unit}")
    for line in record.events:
        print(f"  event: {line}")
    if record.phase.endswith("A"):
        counts = ", ".join(
            f"{power} {game.supply_count(power)}"
            for power in game.powers
            if game.supply_count(power)
        )
        print(f"  centers: {counts}")


def _print_new_records(game: Game, seen: int) -> int:
    for record in game.history[seen:]:
        _print_record(game, record)
    return len(game.history)


def _cmd_adjudicate(args: argparse.Namespace) -> int:
    path = Path(args.game)
    try:
        game = game_from_text(path.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"cannot read {path}: {error}", file=sys.stderr)
        return 1
    except SerializationError as error:
        print(f"cannot load {path}: {error}", file=sys.stderr)
        return 1
    if game.is_over:
        print(f"game is over ({game.phase}); nothing to adjudicate", file=sys.stderr)
        return 1
    seen = len(game.history)
    try:
        game.adjudicate()
    except GameError as error:
        print(f"adjudication failed: {error}", file=sys.stderr)
        return 1
    _print_new_records(game, seen)
    _print_status(game)
    out = Path(args.out) if args.out else path
    out.write_text(game_to_text(game), encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


def _print_status(game: Game) -> None:
    if game.winner is not None:
        print(f"RESULT: {game.winner} wins with {game.supply_count(game.winner)} centers")
    elif game.is_over:
        print("RESULT: game over (board frozen)")
    else:
        print(f"next phase: {game.phase}")


def _cmd_play(args: argparse.Namespace) -> int:
    try:
        factory = REGISTRY[args.agents]
    except KeyError:
        print(f"unknown agent {args.agents!r} (available: {sorted(REGISTRY)})", file=sys.stderr)
        return 1
    agent: Agent = factory(args.seed)
    try:
        game = Game.new(args.ruleset, args.map)
    except (GameError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 1

    # The cap must respect the ruleset's calendar direction (standard counts
    # up, Beleriand counts 25 down to 0): "stop after year N" is year > N
    # ascending but year < N descending. Unset means 1914 on ascending
    # calendars and no cap on countdown ones, whose games end by their own
    # rules when the calendar runs out.
    ascending = game.hooks.initial_phase(None).year_step > 0
    max_year: int | None = args.max_year
    if max_year is None and ascending:
        max_year = 1914

    def year_within_cap() -> bool:
        if max_year is None:
            return True
        return game.year <= max_year if ascending else game.year >= max_year

    seen = 0
    while not game.is_over and year_within_cap():
        for power in game.powers:
            sheet = agent.orders(game, power)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()
        seen = _print_new_records(game, seen)

    for line in game.hooks.drain_events():  # boundary events after the last record
        print(f"  event: {line}")
    if game.winner is not None:
        print(f"RESULT: {game.winner} wins with {game.supply_count(game.winner)} centers")
    else:
        counts = ", ".join(f"{power} {game.supply_count(power)}" for power in game.powers)
        by = f"by {max_year}" if max_year is not None else "declared"
        print(f"RESULT: no winner {by}; centers: {counts}")
    if args.out:
        Path(args.out).write_text(game_to_text(game), encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    return 0


def _cmd_beleriand_demo(args: argparse.Namespace) -> int:
    """Play a full seeded Beleriand game, from the start to the Valar's intervention.

    The play is random, choreographed just enough to parade the variant's
    engine-level mechanics: dwarven mask-making and a mask handover, knowledge
    grants, Glaurung's devastations after his Spring-20 arrival, a Morgoth army
    sent to the coast for Ulmo's wrath, migration and the countdown.
    Deterministic in ``--seed``.
    """
    from entente.variants.beleriand.layer2 import BeleriandHooks

    game = Game.new("beleriand", "beleriand", seed=args.seed)
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    agent: Agent = REGISTRY["random"](args.seed)
    print(f"Beleriand -- the countdown of the Valar (seed {args.seed})")
    dice = ", ".join(f"{p} {d}" for p, d in sorted(hooks.ulmo_delays.items()))
    print(f"Ulmo's dice (province: phases of grace): {dice}")

    choreographer = _BeleriandChoreographer(game)
    seen = 0
    steps = 0
    while not game.is_over and steps < 400:
        choreographer.grants()
        for power in game.powers:
            sheet = list(agent.orders(game, power))
            sheet = choreographer.amend(power, sheet)
            if sheet:
                game.submit(power, sheet)
        game.adjudicate()
        seen = _print_new_records(game, seen)
        steps += 1

    for line in hooks.drain_events():  # boundary events after the last record
        print(f"  event: {line}")
    result = hooks.game_result()
    if result is not None:
        print(f"RESULT: {result.reason}")
        print(f"winners: {', '.join(result.winners) if result.winners else 'nobody'}")
        for power in sorted(result.scores):
            print(f"  {power}: net {result.scores[power]:+d} centers")
    if args.out:
        Path(args.out).write_text(game_to_text(game), encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    return 0


class _BeleriandChoreographer:
    """Deterministic order overrides that make the demo exercise every mechanic.

    Amending a random agent's sheets, rather than scripting the whole game, is
    what keeps the demo a real play-through instead of a replay.
    """

    def __init__(self, game: Game) -> None:
        from entente.variants.beleriand.layer2 import BeleriandHooks

        assert isinstance(game.hooks, BeleriandHooks)
        self.game = game
        self.hooks = game.hooks
        self.mask_passed = False
        self._pair_phase: str | None = None
        self._pair: tuple[LedgerUnit, Unit] | None = None

    def grants(self) -> None:
        """Hand the scheduled knowledge facts over, so the demo shows both kinds."""
        from entente.engine.phase import PhaseKind

        game = self.game
        if game.phase_kind is not PhaseKind.MOVEMENT:
            return
        try:
            if game.phase == "S24M":
                game.grant("TELERI", "MEN", "fleet-building")
            if game.phase == "S23M":
                game.grant("DORIATH", "MEN", "girdle-permission")
        except GameError:  # pragma: no cover - fixed schedule is always legal
            pass

    def amend(self, power: str, sheet: list[GameOrder]) -> list[GameOrder]:
        """Rewrite one power's movement sheet, where that power stages a mechanic."""
        from entente.engine.phase import PhaseKind

        if self.game.phase_kind is not PhaseKind.MOVEMENT:
            return sheet
        if power == "NAUGRIM":
            return self._naugrim(sheet)
        if power == "MEN":
            return self._men(sheet)
        if power == "MORGOTH":
            return self._morgoth(sheet)
        return sheet

    def _drop_orders_at(self, sheet: list[GameOrder], province: str) -> list[GameOrder]:
        """Remove every order the province's own unit was given, to replace them."""
        kept: list[GameOrder] = []
        for order in sheet:
            location = getattr(order, "location", None)
            if location is None or location.province != province:
                kept.append(order)
        return kept

    def _naugrim(self, sheet: list[GameOrder]) -> list[GameOrder]:
        """Make masks in the dwarf cities, then offer one to a neighbouring man."""
        from entente.engine.map.types import Location
        from entente.engine.orders import GiveMask, MakeMasks, UnitRef
        from entente.variants.beleriand.layer2 import DWARF_CITIES

        state = self.game.state
        for unit in state.units:
            if unit.power == "NAUGRIM" and unit.location.province in DWARF_CITIES:
                sheet = self._drop_orders_at(sheet, unit.location.province)
                sheet.append(MakeMasks(unit.kind, unit.location))
        handover = self._handover()
        if handover is not None:
            giver, receiver = handover
            sheet = self._drop_orders_at(sheet, giver.location.province)
            sheet.append(
                GiveMask(
                    giver.kind,
                    giver.location,
                    UnitRef(receiver.power, receiver.kind, Location(receiver.location.province)),
                )
            )
            self.mask_passed = True  # one handover suffices for the showcase
        return sheet

    def _handover(self) -> tuple[LedgerUnit, Unit] | None:
        """Find the first masked dwarf standing adjacent to a mannish unit.

        It is cached per phase and used once per game: both powers' sheets must
        name the *same* giver and receiver pair, and the powers amend their
        sheets in alphabetical order.
        """
        if self.mask_passed:
            return None
        if self._pair_phase == self.game.phase:
            return self._pair
        self._pair_phase = self.game.phase
        self._pair = None
        state = self.game.state
        for entry in self.hooks.ledger.board():
            if entry.power != "NAUGRIM" or entry.masks < 1:
                continue
            neighbors = self.hooks.graph.army_adjacency.get(entry.location.province, frozenset())
            for unit in state.units:
                if unit.power == "MEN" and (
                    unit.location.province in neighbors
                    or unit.location.province == entry.location.province
                ):
                    self._pair = (entry, unit)
                    return self._pair
        return None

    def _men(self, sheet: list[GameOrder]) -> list[GameOrder]:
        """Accept the dwarves' offered mask, holding the receiving unit in place."""
        from entente.engine.map.types import Location
        from entente.engine.orders import AcceptMask, Hold, UnitRef

        handover = self._handover()
        if handover is None:
            return sheet
        giver, receiver = handover
        sheet = self._drop_orders_at(sheet, receiver.location.province)
        sheet.append(Hold(receiver.kind, receiver.location))
        sheet.append(
            AcceptMask(
                receiver.kind,
                receiver.location,
                UnitRef(giver.power, giver.kind, Location(giver.location.province)),
            )
        )
        return sheet

    def _morgoth(self, sheet: list[GameOrder]) -> list[GameOrder]:
        """Send one army toward Ulmo's coast and set Glaurung devastating."""
        from entente.engine.map.types import Location
        from entente.engine.orders import Devastate, Move
        from entente.variants.beleriand.layer2 import ULMO_PROVINCES
        from entente.variants.beleriand.ledger import GLAURUNG, SPECIAL_DESIGNATOR

        graph = self.hooks.graph
        # a special is named only with its designator: every order the demo
        # writes for Glaurung carries his (G), or the game refuses it
        dragon_mark = SPECIAL_DESIGNATOR[GLAURUNG]
        state = self.game.state
        # the coast-runner: one army walks a free path toward Ulmo's provinces
        runner = next(
            (u for u in state.units if u.power == "MORGOTH" and u.strength == 1),
            None,
        )
        if runner is not None and runner.location.province not in ULMO_PROVINCES:
            step = self._step_toward(runner.location.province, set(ULMO_PROVINCES))
            if step is not None:
                sheet = self._drop_orders_at(sheet, runner.location.province)
                sheet.append(Move(runner.kind, runner.location, Location(step)))
        glaurung = self.hooks.ledger.special(GLAURUNG)
        if glaurung is not None and not glaurung.limbo:
            province = glaurung.location.province
            if province == "ANG":
                sheet = self._drop_orders_at(sheet, province)
                sheet.append(
                    Move(
                        glaurung.kind,
                        glaurung.location,
                        Location("ANF"),
                        strength=3,
                        special=dragon_mark,
                    )
                )
            elif province not in self.hooks.devastated:
                sheet = self._drop_orders_at(sheet, province)
                sheet.append(Devastate(glaurung.kind, glaurung.location, 3, dragon_mark))
            else:
                fresh = {
                    key
                    for key, p in graph.provinces.items()
                    if p.is_supply_center and key not in self.hooks.devastated and key != "ANG"
                }
                step = self._step_toward(province, fresh, dragon=True)
                if step is not None:
                    sheet = self._drop_orders_at(sheet, province)
                    sheet.append(
                        Move(
                            glaurung.kind,
                            glaurung.location,
                            Location(step),
                            strength=3,
                            special=dragon_mark,
                        )
                    )
        return sheet

    def _step_toward(self, origin: str, goals: set[str], *, dragon: bool = False) -> str | None:
        """Find the next step toward the nearest goal, over army adjacency.

        The search runs breadth-first through provinces that are free right now.
        The dragon walks devastated ground; every other runner avoids it.
        """
        graph = self.hooks.graph
        occupied = {u.location.province for u in self.game.state.units}
        frontier = [origin]
        parents: dict[str, str] = {origin: origin}
        while frontier:
            nxt: list[str] = []
            for province in frontier:
                for neighbor in sorted(graph.army_adjacency.get(province, frozenset())):
                    if neighbor in parents:
                        continue
                    blocked = neighbor in occupied or (
                        not dragon and neighbor in self.hooks.devastated
                    )
                    restricted = graph.provinces[neighbor].restricted_to is not None
                    girdled = neighbor in ("FOR", "FON")
                    if blocked or restricted or girdled:
                        continue
                    parents[neighbor] = province
                    if neighbor in goals:
                        step = neighbor
                        while parents[step] != origin:
                            step = parents[step]
                        return step
                    nxt.append(neighbor)
            frontier = nxt
        return None


def _print_tournament_table(results: Sequence[SeatedOutcome]) -> None:
    sos = per_agent_sos(results)
    survival = per_agent_survival(results)
    solo = per_agent_solo(results)
    header = (
        f"{'agent':10s} {'games':>6s} {'mean SoS':>10s} {'95% CI':>18s} {'surv.':>6s} {'solo':>6s}"
    )
    print(header)
    for name in sorted(sos):
        scores = sos[name]
        lo, hi = bootstrap_ci(scores)
        mean = statistics.fmean(scores)
        surv_rate = statistics.fmean(1.0 if alive else 0.0 for alive in survival[name])
        solo_rate = statistics.fmean(1.0 if won else 0.0 for won in solo[name])
        ci = f"[{lo:.4f}, {hi:.4f}]"
        print(
            f"{name:10s} {len(scores):6d} {mean:10.4f} {ci:>18s} {surv_rate:6.3f} {solo_rate:6.3f}"
        )


def _cmd_tournament(args: argparse.Namespace) -> int:
    try:
        config = load_tournament_config(args.config, overrides=args.set or ())
    except ConfigError as error:
        print(f"cannot load {args.config}: {error}", file=sys.stderr)
        return 1

    try:
        if config.format == "one_vs_six":
            agent_a, agent_b = config.agents
            results: list[SeatedOutcome] = one_vs_six(
                agent_a,
                agent_b,
                config.seeds,
                ruleset=config.ruleset,
                map_name=config.map_name,
                max_year=config.max_year,
                max_workers=config.max_workers,
            )
        else:
            results = round_robin(
                config.agents,
                config.seeds,
                ruleset=config.ruleset,
                map_name=config.map_name,
                max_year=config.max_year,
                max_workers=config.max_workers,
            )
    except KeyError as error:
        print(f"unknown agent {error}: available {sorted(REGISTRY)}", file=sys.stderr)
        return 1
    except ValueError as error:  # e.g. the harness's standard-map-only guard
        print(str(error), file=sys.stderr)
        return 1

    print(f"ran {len(results)} games ({config.format}, agents={list(config.agents)})")
    if config.output:
        count = write_seated_jsonl(results, config.output)
        print(f"wrote {count} games to {config.output}")
    _print_tournament_table(results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser, every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog="entente",
        description="A ruleset-generalized Diplomacy engine.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    adjudicate = commands.add_parser(
        "adjudicate", help="adjudicate the current phase of a saved game"
    )
    adjudicate.add_argument("game", help="path to an entente-game JSON file")
    adjudicate.add_argument("--out", help="write the advanced game here (default: in place)")
    adjudicate.set_defaults(func=_cmd_adjudicate)

    play = commands.add_parser("play", help="play a full game between agents")
    play.add_argument("--agents", default="random", help="agent for all powers (default: random)")
    play.add_argument("--seed", type=int, default=0, help="agent RNG seed (default: 0)")
    play.add_argument(
        "--max-year",
        type=int,
        help="stop after this game year, respecting the ruleset's calendar direction "
        "(default: 1914 on ascending calendars; countdown rulesets run to their own end)",
    )
    play.add_argument("--map", default="standard", help="map name (default: standard)")
    play.add_argument("--ruleset", default="standard", help="ruleset name (default: standard)")
    play.add_argument("--out", help="save the finished game as JSON")
    play.set_defaults(func=_cmd_play)

    tournament = commands.add_parser("tournament", help="run a tournament from a YAML config")
    tournament.add_argument("config", help="path to a tournament YAML config")
    tournament.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a config value (dotted key), e.g. --set max_year=1905; repeatable",
    )
    tournament.set_defaults(func=_cmd_tournament)

    demo = commands.add_parser(
        "beleriand-demo",
        help="a full seeded Beleriand game, start to the Valar's intervention",
    )
    demo.add_argument("--seed", type=int, default=0, help="game + agent seed (default: 0)")
    demo.add_argument("--out", help="save the finished game as JSON (format v2)")
    demo.set_defaults(func=_cmd_beleriand_demo)

    _register_optional_commands(commands)
    return parser


def _register_optional_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the subcommands owned by cuttable, isolated packages (see the module docstring).

    ``entente.negotiation`` is reached by name through ``importlib`` -- never
    a static import -- so this file has nothing dangling to import once that
    package is deleted; it just silently stops offering the subcommand.
    """
    try:
        negotiation_cli = importlib.import_module("entente.negotiation.cli")
    except ModuleNotFoundError:
        return
    negotiation_cli.register(commands)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the chosen subcommand, returning its exit status."""
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
