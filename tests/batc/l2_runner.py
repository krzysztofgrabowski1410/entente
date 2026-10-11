"""Executor for the BATC-L2 game-scenario YAML cases.

Layer-2 mechanics are *game-level* -- timers, boundaries, winters, knowledge,
victory -- so unlike the movement-level BATC these cases drive the public
``Game`` facade across whole phases. A case optionally constructs a mid-game
position (with Layer-2 state overrides applied to a hand-built
``BeleriandHooks``), scripts a sequence of adjudications with order sheets
and knowledge grants, and asserts over the last executed record plus the
final game/hooks state.

Schema::

    section: "BATC-L2 M. Dwarven masks"
    cases:
      - id: BATC-L2.M.1
        title: ...
        seed: 0                        # optional, default 0
        start:                         # optional; omitted = Game.new (year 25)
          phase: S25M                  # movement phases only
          units: {NAUGRIM: ["A BEL"], ...}
          ownership: {BRI: MEN, EGL: null, ...}      # overrides on the home map
          l2:                          # all optional
            fleet_knowledge: [TELERI, MEN]
            girdle: [DORIATH]
            devastated: [HIT]
            devastation_years: {HIT: 3}
            converted_homes: [HIT]
            eriador_centers: 2
            autumn_count: 3
            delayed_supply: {EGL: 9}
            owner_since: {EGL: 2}
            double_eligible: [NAR]     # list = full weight each; {SOS: 2} names a point count
            masks: {"NAUGRIM A BEL": 2}              # int or per-twin list
            bound_to: {"FINROD 2A NAR": NAR}  # two different unit keys may target one province
            glaurung: ANG              # tag a (non-Morgoth-unit) 3A there
            morgoth_unit: ANG          # re-tag the Morgoth unit (canonical-first default)
            ulmo_delays: {LAM: 1}
            ulmo_armed: {LAM: 3}       # absolute movement ordinals (2*(25-year)+fall)
            gondolin_shield: false
            morgoth_defeated: true
            sail_west_ever: true
            glaurung_arrived: true
        script:
          - phase: S25M                # guard: the engine must be here
            grants: [[TELERI, MEN, fleet-building]]
            grants_refused: [[TELERI, MORGOTH, fleet-building]]
            swear: DORIATH             # Morgoth's Oath of the Second
            swear_refused: [MORGOTH]   # each name must be refused
            break: true                # break the standing Oath
            break_refused: true
            orders: {NAUGRIM: ["A BEL MSK"], ...}
            adjudicate: false          # submit only; the next step resolves
        after:                         # post-game acts, once the script ends
          second: DORIATH
          second_refused: [MEN]
        expect:
          resolutions: {"A BEL MSK": succeeds}       # on the last executed record
          events: ["NAUGRIM A BEL makes 2 masks"]    # subset of last record events + buffered
          units: {NAUGRIM: ["A BEL"]}                # exact per listed power
          ownership: {BRI: MEN}                      # subset
          masks: {"NAUGRIM A BEL": 2}                # exact over all mask holders
          fleet_knowledge/girdle/devastated/converted_homes: [...]
          eriador_centers: 2
          double_bindings: {NAR: FINROD}    # or {SOS: [FINROD, FINROD]} -- 2+, a sorted list
          specials: {glaurung: {province: LOT, limbo: false}}  # exact, over live specials
          phase: F25M
          over: true
          winners: [DORIATH]
          reason_contains: "Valar"
          morgoth_defeated: true
          gondolin_shield: false
          oath: DORIATH              # the standing Oath, its Year and its price
          oath_year: 25
          oaths_broken: 1
          oath_stabbed: false
          victory_bar: 41
          second: DORIATH
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from entente.engine.game import Game, GameError
from entente.engine.map import load_map
from entente.engine.orders import (
    format_order,
    parse_adjustment_order,
    parse_order,
    parse_retreat_order,
    parse_unit_spec,
)
from entente.engine.phase import Phase, PhaseKind
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import BeleriandHooks
from entente.variants.beleriand.ledger import GLAURUNG, MORGOTH_UNIT, LedgerUnit


@dataclass(frozen=True)
class L2Case:
    """One loaded BATC-L2 case: its start, its script and its expectations."""

    id: str
    title: str
    section: str
    seed: int
    start: dict[str, Any] | None
    script: list[dict[str, Any]]
    expect: dict[str, Any]
    after: dict[str, Any] = field(default_factory=dict)
    """Post-game acts run once the script has ended: second-place designation
    and its refusals. They have no phase to ride, so they cannot live in a
    script step -- steps drive a live game (each ends in an adjudication
    unless ``adjudicate: false``), and a finished game refuses that.
    Defaulted, so cases built in code (rather than loaded from
    YAML) need not mention it."""

    @property
    def pytest_id(self) -> str:
        """The case id, which is what pytest prints."""
        return self.id


# Every key _check_expectations reads from ``expect``. The loader validates
# against it because a case with a typo'd or omitted ``expect`` would run its
# script and pass while asserting nothing.
_VALID_EXPECT_KEYS = frozenset(
    {
        "resolutions",
        "events",
        "units",
        "ownership",
        "masks",
        "fleet_knowledge",
        "girdle",
        "devastated",
        "converted_homes",
        "eriador_centers",
        "double_bindings",
        "specials",
        "phase",
        "over",
        "winners",
        "reason_contains",
        "morgoth_defeated",
        "gondolin_shield",
        "oath",
        "oath_year",
        "oaths_broken",
        "oath_stabbed",
        "victory_bar",
        "second",
    }
)

_VALID_AFTER_KEYS = frozenset({"second", "second_refused"})

_VALID_STEP_KEYS = frozenset(
    {
        "phase",
        "grants",
        "grants_refused",
        "swear",
        "swear_refused",
        "break",
        "break_refused",
        "orders",
        "adjudicate",
    }
)


def load_all(directory: Path) -> list[L2Case]:
    """Load and validate every case file in ``directory``, in file-name order."""
    cases: list[L2Case] = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        section = document["section"]
        for raw in document["cases"]:
            case = L2Case(
                id=raw["id"],
                title=raw["title"],
                section=section,
                seed=int(raw.get("seed", 0)),
                start=raw.get("start"),
                script=raw.get("script", []),
                expect=raw.get("expect", {}),
                after=raw.get("after", {}),
            )
            _validate_l2(case, path)
            cases.append(case)
    return cases


def _validate_l2(case: L2Case, path: Path) -> None:
    context = f"{path.name}:{case.id}"
    if not case.expect:
        raise ValueError(f"{context}: case asserts nothing (empty or missing expect)")
    for label, keys, known in (
        ("expect", set(case.expect), _VALID_EXPECT_KEYS),
        ("after", set(case.after), _VALID_AFTER_KEYS),
        ("script step", {k for step in case.script for k in step}, _VALID_STEP_KEYS),
    ):
        unknown = sorted(keys - known)
        if unknown:
            raise ValueError(
                f"{context}: unknown {label} keys {unknown} (known: {sorted(known)}) -- "
                "a typo would silently drop the assertion"
            )


def _find_entries(hooks: BeleriandHooks, key: str) -> list[LedgerUnit]:
    """Ledger entries matching ``"POWER [X]K PROV"`` in uid order."""
    power, _, spec = key.partition(" ")
    kind, location, strength = parse_unit_spec(spec)
    return [
        entry
        for entry in hooks.ledger.alive()
        if entry.power == power
        and entry.kind is kind
        and entry.strength == strength
        and entry.location == location
    ]


def build_game(case: L2Case) -> Game:
    """Return the game a case starts from: the standard opening, or its hand-built position."""
    if case.start is None:
        return Game.new("beleriand", "beleriand", seed=case.seed)
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, case.seed)
    parsed = Phase.parse(case.start["phase"])
    if parsed.kind is not PhaseKind.MOVEMENT:
        raise ValueError(f"{case.id}: constructed starts must be movement phases")
    phase = Phase(parsed.year, parsed.season, parsed.kind, year_step=-1)
    units = [
        Unit(power, *parse_unit_spec(spec))
        for power, specs in sorted(case.start.get("units", {}).items())
        for spec in specs
    ]
    state = State.of("beleriand", phase.tag, units)
    # Default ownership: home centers of the powers actually present -- absent
    # powers owning centers would open winter build decisions in every
    # multi-year scenario. Overrides add or remove entries explicitly.
    present = {unit.power for unit in units}
    ownership: dict[str, str] = {
        center: power
        for power, centers in sorted(graph.home_centers.items())
        if power in present
        for center in centers
    }
    for province, owner in case.start.get("ownership", {}).items():
        if owner is None:
            ownership.pop(province, None)
        else:
            ownership[province] = owner
    hooks.register_start(state, ownership)
    _apply_l2_overrides(hooks, case.start.get("l2", {}))
    return Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)


def _apply_l2_overrides(hooks: BeleriandHooks, overrides: dict[str, Any]) -> None:
    simple_sets = {
        "fleet_knowledge": "fleet_knowledge",
        "girdle": "girdle",
        "devastated": "devastated",
        "converted_homes": "converted_homes",
    }
    for key, attribute in simple_sets.items():
        if key in overrides:
            setattr(hooks, attribute, set(overrides[key]))
    if "double_eligible" in overrides:
        # a plain list names provinces eligible for their full weight (one
        # point for every weight-1 province); a dict gives explicit
        # per-province point counts, needed to exercise a two-Centre
        # province's second point
        value = overrides["double_eligible"]
        if isinstance(value, dict):
            hooks.double_eligible = dict(value)
        else:
            hooks.double_eligible = {
                province: hooks.graph.province(province).supply_center_count for province in value
            }
    simple_dicts = {
        "devastation_years": "devastation_years",
        "delayed_supply": "delayed_supply",
        "owner_since": "owner_since",
        "ulmo_armed": "ulmo_armed",
    }
    for key, attribute in simple_dicts.items():
        if key in overrides:
            getattr(hooks, attribute).update(overrides[key])
    if "ulmo_delays" in overrides:
        hooks.ulmo_delays.update(overrides["ulmo_delays"])
    for key in (
        "eriador_centers",
        "autumn_count",
        "gondolin_shield",
        "morgoth_defeated",
        "sail_west_ever",
        "glaurung_arrived",
    ):
        if key in overrides:
            setattr(hooks, key, overrides[key])
    if "morgoth_unit" in overrides:
        for entry in hooks.ledger.alive():
            if entry.special == MORGOTH_UNIT:
                entry.special = None
        province = overrides["morgoth_unit"]
        if province is not None:
            for entry in hooks.ledger.alive():
                if (
                    entry.power == "MORGOTH"
                    and entry.strength == 3
                    and entry.location.province == province
                    and entry.special is None
                ):
                    entry.special = MORGOTH_UNIT
                    break
    if "glaurung" in overrides:
        province = overrides["glaurung"]
        for entry in hooks.ledger.alive():
            if (
                entry.power == "MORGOTH"
                and entry.strength == 3
                and entry.location.province == province
                and entry.special is None
            ):
                entry.special = GLAURUNG
                hooks.glaurung_arrived = True
                break
        else:
            raise ValueError(f"no untagged 3A MORGOTH unit in {province} for Glaurung")
    for key, value in overrides.get("masks", {}).items():
        entries = _find_entries(hooks, key)
        counts = value if isinstance(value, list) else [value]
        for entry, count in zip(entries, counts, strict=False):
            entry.masks = count
    for key, province in overrides.get("bound_to", {}).items():
        entries = _find_entries(hooks, key)
        if entries:
            entries[0].bound_to = province


def _parse_sheet(kind: PhaseKind, texts: list[str]) -> list:
    if kind is PhaseKind.MOVEMENT:
        return [parse_order(text) for text in texts]
    if kind is PhaseKind.RETREAT:
        return [parse_retreat_order(text) for text in texts]
    return [parse_adjustment_order(text) for text in texts]


def _refuses(case: L2Case, what: str, act: Any) -> None:
    try:
        act()
    except GameError:
        return
    raise AssertionError(f"{case.id}: {what} was allowed")


def run_case(case: L2Case) -> None:
    """Drive the case's script through the game and check its expectations."""
    game = build_game(case)
    record = None
    for step in case.script:
        expected_phase = step.get("phase")
        if expected_phase is not None:
            assert game.phase == expected_phase, (
                f"{case.id}: script expected {expected_phase}, game is at {game.phase}"
            )
        for giver, receiver, fact in step.get("grants", []):
            game.grant(giver, receiver, fact)
        for triple in step.get("grants_refused", []):
            _refuses(
                case,
                f"grant {triple[0]}->{triple[1]} {triple[2]}",
                lambda t=triple: game.grant(*t),
            )
        # An Oath is sworn or broken BEFORE the season's orders are placed,
        # which is also the order this step applies them in; a case that
        # needs the other order splits across two steps with `adjudicate: false`.
        for power in step.get("swear_refused", []):
            _refuses(case, f"swearing to {power}", lambda p=power: game.swear_oath(p))
        if step.get("break_refused"):
            _refuses(case, "breaking the Oath", game.break_oath)
        if step.get("break"):
            game.break_oath()
        if "swear" in step:
            game.swear_oath(step["swear"])
        for power, texts in sorted(step.get("orders", {}).items()):
            game.submit(power, _parse_sheet(game.phase_kind, texts))
        if step.get("adjudicate", True):
            record = game.adjudicate()
    for power in case.after.get("second_refused", []):
        _refuses(case, f"designating {power} second", lambda p=power: game.designate_second(p))
    if "second" in case.after:
        game.designate_second(case.after["second"])
    _check_expectations(case, game, record)


def _check_expectations(case: L2Case, game: Game, record: Any) -> None:
    expect = case.expect
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    if "resolutions" in expect:
        assert record is not None, f"{case.id}: no phase executed"
        rows = {}
        for row in record.resolutions:
            rows.setdefault(row.text, row.resolution.value)
        for text, verdict in expect["resolutions"].items():
            canonical = _canonical(game, record.phase, text)
            assert canonical in rows, f"{case.id}: no resolution for {canonical!r} in {rows}"
            assert rows[canonical] == verdict, (
                f"{case.id}: {canonical!r} resolved {rows[canonical]}, expected {verdict}"
            )
    if "events" in expect:
        seen = list(record.events) if record is not None else []
        seen.extend(hooks._events)  # boundary events still buffered
        for line in expect["events"]:
            assert line in seen, f"{case.id}: event {line!r} not in {seen}"
    if "units" in expect:
        for power, specs in expect["units"].items():
            placed = sorted(str(u) for u in game.state.units if u.power == power)
            assert placed == sorted(specs), f"{case.id}: {power} units {placed} != {sorted(specs)}"
    if "ownership" in expect:
        for province, owner in expect["ownership"].items():
            actual = game.ownership.get(province)
            assert actual == owner, f"{case.id}: {province} owned by {actual}, expected {owner}"
    if "masks" in expect:
        actual_masks: dict[str, int] = {}
        for entry in hooks.ledger.alive():
            if entry.masks:
                key = f"{entry.power} {entry.spec()}"
                actual_masks[key] = actual_masks.get(key, 0) + entry.masks
        assert actual_masks == dict(expect["masks"]), (
            f"{case.id}: masks {actual_masks} != {expect['masks']}"
        )
    for key, attribute in (
        ("fleet_knowledge", "fleet_knowledge"),
        ("girdle", "girdle"),
        ("devastated", "devastated"),
        ("converted_homes", "converted_homes"),
    ):
        if key in expect:
            assert sorted(getattr(hooks, attribute)) == sorted(expect[key]), (
                f"{case.id}: {key} {sorted(getattr(hooks, attribute))} != {sorted(expect[key])}"
            )
    if "eriador_centers" in expect:
        assert hooks.eriador_centers == expect["eriador_centers"], case.id
    if "double_bindings" in expect:
        # Read through hooks.snapshot() (the production writer) rather than
        # re-deriving from the ledger, so a two-Centre province's second live
        # binding is asserted in its real scalar-or-list shape instead of a
        # hand-rolled dict comprehension silently collapsing it.
        snapshot = hooks.snapshot()
        bindings = snapshot["double_bindings"] if snapshot is not None else {}
        expected = {
            province: sorted(value) if isinstance(value, list) else value
            for province, value in expect["double_bindings"].items()
        }
        assert bindings == expected, f"{case.id}: bindings {bindings} != {expected}"
    if "specials" in expect:
        snapshot = hooks.snapshot()
        assert snapshot is not None
        actual_specials = snapshot["specials"]
        assert actual_specials == expect["specials"], (
            f"{case.id}: specials {actual_specials} != {expect['specials']}"
        )
    if "phase" in expect:
        assert game.phase == expect["phase"], (
            f"{case.id}: at {game.phase}, expected {expect['phase']}"
        )
    if "over" in expect:
        assert game.is_over == expect["over"], f"{case.id}: over={game.is_over}"
    if "winners" in expect:
        result = hooks.game_result()
        winners = list(result.winners) if result is not None else []
        assert winners == list(expect["winners"]), f"{case.id}: winners {winners}"
    if "reason_contains" in expect:
        result = hooks.game_result()
        assert result is not None and expect["reason_contains"] in result.reason, (
            f"{case.id}: reason {result.reason if result else None!r}"
        )
    if "morgoth_defeated" in expect:
        assert hooks.morgoth_defeated == expect["morgoth_defeated"], case.id
    if "gondolin_shield" in expect:
        assert hooks.gondolin_shield == expect["gondolin_shield"], case.id
    if "second" in expect:
        assert game.second == expect["second"], f"{case.id}: second {game.second}"
    for key, actual in (
        ("oath", game.oath),
        ("oath_year", game.oath_year),
        ("oaths_broken", game.oaths_broken),
        ("oath_stabbed", game.oath_stabbed),
        ("victory_bar", hooks.victory_bar),
    ):
        if key in expect:
            assert actual == expect[key], f"{case.id}: {key} {actual!r} != {expect[key]!r}"


def _canonical(game: Game, phase_tag: str, text: str) -> str:
    del game
    kind = Phase.parse(phase_tag).kind
    return format_order(_parse_sheet(kind, [text])[0])
