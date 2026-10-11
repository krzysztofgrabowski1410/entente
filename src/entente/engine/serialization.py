"""Versioned JSON serialization of full game histories.

The serialized artifact is the *decision record*: map, ruleset, start year,
and every executed phase's submitted sheets in canonical notation. Loading
**replays** those sheets through the engine -- the engine itself is the
authority on outcomes -- and then verifies the file's snapshot (units,
ownership, phase, game-over flag) against the replayed game. A snapshot
mismatch means the engine's rules drifted since the file was written (or the
file was edited); either way loading fails loudly rather than resurrecting a
position the current engine would never produce.

Phases the game loop executed automatically (``auto: true`` records -- forced
disbands, civil-disorder removals) are stored for completeness but not
re-submitted: the replaying loop re-executes them itself, and the replay
asserts the histories align one-to-one.

Scope: only games created by ``Game.new`` and driven through
``adjudicate()`` serialize faithfully -- a ``Game`` constructed directly
around a mid-game position has no history connecting it to the map's start,
so its save cannot replay and loading refuses it (that refusal is the
verification working, not a bug).

Format v1 (``version`` bumps on any incompatible change)::

    {
      "format": "entente-game",
      "version": 1,
      "map": "standard",
      "ruleset": "standard",
      "start_year": 1901,
      "phases": [{"phase": "S1901M", "auto": false,
                  "sheets": {"FRANCE": ["A PAR - BUR", ...], ...}}, ...],
      "pending": {"GERMANY": ["A MUN H", ...], ...},
      "snapshot": {"phase": "F1901M", "over": false,
                   "units": {"FRANCE": ["A BUR", ...], ...},
                   "ownership": {"PAR": "FRANCE", ...}}
    }

Format v2 extends v1 for games with a variant layer:
a top-level ``"seed"`` (the variant RNG stream), per-phase ``"grants"``
(knowledge transfers, replayed before that phase's sheets), and a
``"snapshot"."l2"`` fragment (the hooks' own state summary) verified after
replay like every other snapshot field. A Morgoth win that designated a
second place carries an optional top-level ``"second"`` -- a post-game act
(the ``grant`` channel's out-of-phase cousin) replayed after every phase,
before snapshot verification; it is absent unless a second was named, so
standard games and every undesignated variant game stay byte-identical.

A grant is applied the moment it is made but only reaches its phase record
at the next adjudication, so a save taken in between shows the effect in
``"snapshot"."l2"`` with nothing in ``"phases"`` to replay it. Those
unflushed transfers ride in a top-level ``"pending_grants"``, re-applied
before ``"pending"``'s sheets just as ``_replay_phase`` orders them. Like
``"second"`` it is written only when non-empty, so it cannot change a save
that had none. The loader accepts both versions.

Morgoth's Oath of the Second rides four more optional top-level keys --
``"oath"``, ``"oath_year"``, ``"oaths_broken"``, ``"oath_stabbed"`` -- and is
re-applied through ``Game.restore_oath_state``, NOT by re-performing the acts
that produced it. Swearing and breaking are phase-gated (refused once Morgoth
has placed his orders) and priced (each break raises his victory threshold for
the rest of the game), and a replay has neither a phase to fire the gate in
nor any business paying the price a second time: like ``"second"``, the
recorded state is a replay *input*. It validates loudly on the way in, so a
hand-edited count or a classic document carrying an Oath fails to load rather
than producing a game the engine would never have reached.

It is applied at BOTH ends of the phase replay, and each end is load-bearing:

* **Before** the phases, because the Oath is not a post-game fact the way
  ``"second"`` is. It sets the victory threshold every Autumn boundary tests
  against, and an Oath standing at a Morgoth win places second *at that
  moment*. Restoring only afterwards would replay the whole game at the wrong
  bar and reproduce a win with no second in it. (Applying the final count from
  the first phase cannot itself change the history: the count only ever rises,
  and a Morgoth who had reached a *lower* bar in an earlier Autumn would have
  won there and left no later phases to replay.)
* **After** them, because the free-break flag is derived during phase
  resolution from the sworn power's orders. With the Oath restored from the
  first phase, the replay computes that flag over a wider window than the real
  game did -- every phase, rather than those since the Oath was sworn -- so the
  recorded value, not the recomputed one, is the authority.

Like every other optional key here they are written only when the state is not
the default, so a game that never touched the channel carries none of them. A
win that an Oath placed second writes no top-level
``"second"``: the replay reproduces it from the Oath, and re-designating it
would be refused by the very rule that bound it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from entente.engine.game import Game, GameError
from entente.engine.orders import (
    OrderSyntaxError,
    parse_adjustment_order,
    parse_order,
    parse_retreat_order,
)
from entente.engine.phase import Phase, PhaseKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.orders import GameOrder

FORMAT = "entente-game"
VERSION = 1
VERSION_L2 = 2


class SerializationError(ValueError):
    """The file is not a loadable game.

    Wrong format/version, malformed content, or a snapshot that the replayed
    history does not reproduce.
    """


def _units_by_power(game: Game) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for unit in game.state.units:
        grouped.setdefault(unit.power, []).append(str(unit))
    return {power: sorted(units) for power, units in sorted(grouped.items())}


def game_to_json(game: Game) -> dict[str, Any]:
    """Return the JSON document (a plain dict) for a game at any point.

    Version 1, or version 2 when the game has a variant layer.
    """
    fragment = game.hooks.snapshot()
    phases: list[dict[str, Any]] = []
    for record in game.history:
        entry: dict[str, Any] = {
            "phase": record.phase,
            "auto": record.auto,
            "sheets": {power: list(sheet) for power, sheet in sorted(record.sheets.items())},
        }
        if record.grants:
            entry["grants"] = [[g.giver, g.receiver, g.fact] for g in record.grants]
        phases.append(entry)
    data: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION if fragment is None else VERSION_L2,
        "map": game.state.map_name,
        "ruleset": game.ruleset.name,
        "start_year": game.start_year,
        "phases": phases,
        "pending": {power: list(sheet) for power, sheet in game.pending_sheets.items()},
        "snapshot": {
            "phase": game.phase,
            "over": game.is_over,
            "units": _units_by_power(game),
            "ownership": dict(sorted(game.ownership.items())),
        },
    }
    # Grants issued in the phase that has not adjudicated yet are live in the
    # snapshot but absent from ``phases``; written only when there are any, so
    # a save with none carries no key for them.
    if game.pending_grants:
        data["pending_grants"] = [[g.giver, g.receiver, g.fact] for g in game.pending_grants]
    if fragment is not None:
        data["seed"] = game.seed
        # An Oath-bound second needs no key of its own: the restored Oath binds
        # it again during the replay, and designating it a second time is
        # exactly what the Oath refuses.
        if game.second is not None and game.second != game.oath:
            data["second"] = game.second
        if game.oath is not None:
            data["oath"] = game.oath
            data["oath_year"] = game.oath_year
        if game.oaths_broken:
            data["oaths_broken"] = game.oaths_broken
        if game.oath_stabbed:
            data["oath_stabbed"] = True
        data["snapshot"]["l2"] = fragment
    return data


def game_to_text(game: Game) -> str:
    """Return the stable text form of :func:`game_to_json` (sorted keys, 2-space indent)."""
    return json.dumps(game_to_json(game), indent=2, sort_keys=True) + "\n"


def _parse_sheet(phase_tag: str, texts: Sequence[str]) -> list[GameOrder]:
    kind = Phase.parse(phase_tag).kind
    if kind is PhaseKind.MOVEMENT:
        return [parse_order(text) for text in texts]
    if kind is PhaseKind.RETREAT:
        return [parse_retreat_order(text) for text in texts]
    return [parse_adjustment_order(text) for text in texts]


def _grant_triple(grant: Any) -> tuple[str, str, str]:
    try:
        giver, receiver, fact = grant
    except (TypeError, ValueError) as error:
        raise SerializationError(f"malformed grant entry {grant!r}") from error
    return giver, receiver, fact


def _replay_phase(game: Game, entry: Mapping[str, Any]) -> None:
    """Replay one recorded phase: its grants, then its sheets, then the adjudication."""
    tag = entry["phase"]
    if game.phase != tag:
        raise SerializationError(f"replay expected phase {tag}, engine is at {game.phase}")
    for grant in entry.get("grants", ()):
        giver, receiver, fact = _grant_triple(grant)
        game.grant(giver, receiver, fact)
    for power, texts in sorted(entry.get("sheets", {}).items()):
        game.submit(power, _parse_sheet(tag, texts))
    game.adjudicate()


def _restore_oath(game: Game, data: Mapping[str, Any], when: str) -> None:
    """Re-apply the recorded Oath state.

    See the module docstring for why this happens on both sides of the phase
    replay.
    """
    if not any(key in data for key in ("oath", "oath_year", "oaths_broken", "oath_stabbed")):
        return
    try:
        game.restore_oath_state(
            data.get("oath"),
            data.get("oath_year"),
            data.get("oaths_broken", 0),
            stabbed=bool(data.get("oath_stabbed", False)),
        )
    except GameError as error:
        raise SerializationError(f"oath state rejected ({when} replay): {error}") from error


def game_from_json(data: Mapping[str, Any]) -> Game:
    """Replay a v1/v2 document into a live game, verifying its snapshot."""
    if data.get("format") != FORMAT:
        raise SerializationError(f"not an {FORMAT} document")
    if data.get("version") not in (VERSION, VERSION_L2):
        raise SerializationError(f"unsupported version {data.get('version')!r}")
    try:
        game = Game.new(
            data["ruleset"],
            data["map"],
            start_year=data.get("start_year"),
            seed=data.get("seed"),
        )
    except (GameError, FileNotFoundError, KeyError) as error:
        raise SerializationError(f"cannot start game: {error}") from error

    _restore_oath(game, data, "before")

    try:
        for entry in data.get("phases", ()):
            if entry.get("auto"):
                continue  # re-executed by the loop itself; verified below
            _replay_phase(game, entry)
    except (GameError, OrderSyntaxError) as error:
        raise SerializationError(f"replay failed: {error}") from error

    recorded = [(e["phase"], bool(e.get("auto"))) for e in data.get("phases", ())]
    replayed = [(record.phase, record.auto) for record in game.history]
    if recorded != replayed:
        raise SerializationError(
            f"replayed history {replayed} does not match recorded phases {recorded}"
        )

    # The Oath's free-break flag is derived from the sworn power's orders, so
    # the replay just recomputed it over a wider window than the real game had
    # (module docstring): the recorded value wins. Everything else it re-asserts
    # is what the replay already carried in.
    _restore_oath(game, data, "after")

    # A second-place designation is a post-game act with no phase to carry it:
    # re-apply it once the replay has ended the game,
    # so the l2 snapshot's ``"second"`` verifies against a live designation.
    # An Oath-bound second never appears here -- the Oath placed it during the
    # replay, and this channel is exactly what it refuses.
    second = data.get("second")
    if second is not None:
        try:
            game.designate_second(second)
        except GameError as error:
            raise SerializationError(f"second-place designation rejected: {error}") from error

    # Grants issued in the current, unadjudicated phase: re-applied before the
    # pending sheets exactly as ``_replay_phase`` orders them, so a drafted
    # order that relies on a freshly granted fact still submits.
    for grant in data.get("pending_grants", ()):
        giver, receiver, fact = _grant_triple(grant)
        try:
            game.grant(giver, receiver, fact)
        except GameError as error:
            raise SerializationError(f"pending grant rejected: {error}") from error

    for power, texts in sorted(data.get("pending", {}).items()):
        try:
            game.submit(power, _parse_sheet(game.phase, texts))
        except (GameError, OrderSyntaxError) as error:
            raise SerializationError(f"pending sheet rejected: {error}") from error

    snapshot = data.get("snapshot", {})
    checks: list[tuple[str, Any, Any]] = [
        ("phase", snapshot.get("phase"), game.phase),
        ("over", snapshot.get("over"), game.is_over),
        ("units", snapshot.get("units"), _units_by_power(game)),
        ("ownership", snapshot.get("ownership"), dict(sorted(game.ownership.items()))),
    ]
    if data.get("version") == VERSION_L2:
        checks.append(("l2", snapshot.get("l2"), _normalized(game.hooks.snapshot())))
    for name, expected, actual in checks:
        if expected != actual:
            raise SerializationError(
                f"snapshot mismatch on {name!r}: file has {expected!r}, replay produced {actual!r}"
            )
    return game


def _normalized(fragment: dict[str, Any] | None) -> Any:
    """Return the hooks fragment as JSON round-tripping leaves it (keys stringified, tuples listed).

    That makes the replay comparison representation-blind.
    """
    if fragment is None:
        return None
    return json.loads(json.dumps(fragment))


def game_from_text(text: str) -> Game:
    """Parse the text form back into a live game; the inverse of :func:`game_to_text`."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise SerializationError(f"not JSON: {error}") from error
    if not isinstance(data, dict):
        raise SerializationError("top-level JSON value must be an object")
    return game_from_json(data)
