"""Read the Paquette et al. (2019) no-press game dump behind a source interface.

The interface is what lets a synthetic source stand in for the real dump, so the
rest of the pipeline never depends on having the download.

The dataset stores one game per JSONL line as a saved-game object. Only the
*facts of play* this pipeline needs are read -- the schema below is a documented
subset (fields the reader ignores are simply dropped), so the reader is defined
by *this* file, not by any external engine:

    {"id": "...", "map": "standard",
     "phases": [
       {"name": "S1901M",
        "state":  {"units":   {"AUSTRIA": ["A VIE", "F TRI"], ...},
                   "centers": {"AUSTRIA": ["VIE", "BUD", "TRI"], ...}},
        "orders": {"AUSTRIA": ["A VIE - GAL", ...], ...}},
       ... ]}

Order/unit notation is standard Diplomacy notation, which
:func:`entente.engine.orders.parse_order` already accepts (case-insensitive,
optional trailing ``H``); the pipeline re-canonicalizes every string through the
engine, so nothing here depends on the source's exact spelling. In the real
dump any of the per-power list values above may be ``null`` (civil-disordered
or eliminated powers, most terminal ``COMPLETED`` snapshots) -- read as empty,
never as malformed. A unit spec may carry a ``*`` prefix (``"*A GRE"``): a
dislodged, retreat-pending unit. This reader passes specs through verbatim;
board reconstruction drops the starred transients
(:func:`entente.data.dataset.units_from_specs`).

:class:`GameSource` is the interface preprocessing consumes; :class:`PaquetteJSONLSource`
reads a ``.jsonl`` file or a ``.zip`` of them (the dataset ships as a zip), and
:func:`iter_game_records` parses one line. The reader validates lightly and
skips malformed lines with a recorded reason rather than aborting a 150k-game
pass (:attr:`PaquetteJSONLSource.skipped`).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    """Hold one phase of a game as the dump records it.

    That is the pre-adjudication board and the orders each power submitted.
    """

    name: str
    units: dict[str, list[str]]
    centers: dict[str, list[str]]
    orders: dict[str, list[str]]

    @property
    def is_movement(self) -> bool:
        """Say whether this is a movement phase, the only kind the pipeline keeps."""
        return self.name.endswith("M")


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A whole game: its id, map, and phases in play order."""

    game_id: str
    map_name: str
    phases: tuple[PhaseRecord, ...]

    @property
    def final_centers(self) -> dict[str, list[str]]:
        """Give supply-center ownership at the last recorded phase.

        This is what the value target is computed from. Empty if the game has no
        phases at all.
        """
        return dict(self.phases[-1].centers) if self.phases else {}


@runtime_checkable
class GameSource(Protocol):
    r"""Yield :class:`GameRecord`\ s from somewhere.

    Preprocessing depends on this seam rather than on the reader, so a test can
    substitute a synthetic source for the real dump.
    """

    def __iter__(self) -> Iterator[GameRecord]: ...


def _power_lists(mapping: Any) -> dict[str, list[str]]:
    """Read a power-to-list-of-strings mapping as the dump really writes it.

    The whole mapping may be ``null``, and an individual power's value may be
    ``null`` too, which is how the dump records a civil-disordered or eliminated
    power and is therefore pervasive in exactly the completed games this pipeline
    keeps. Both forms mean "empty": treating a value-level ``null`` as malformed
    discards around 80% of the dump, and does so with a bias toward the games
    worth training on.
    """
    return {p: (list(v) if v is not None else []) for p, v in (mapping or {}).items()}


def _phase_from_json(raw: dict[str, Any]) -> PhaseRecord:
    state = raw.get("state") or {}
    return PhaseRecord(
        name=str(raw["name"]),
        units=_power_lists(state.get("units")),
        centers=_power_lists(state.get("centers")),
        orders=_power_lists(raw.get("orders")),
    )


def game_from_json(raw: dict[str, Any]) -> GameRecord:
    """Parse one saved-game object into a :class:`GameRecord`.

    Raises :class:`KeyError` or :class:`TypeError` on a structurally invalid
    object; the source catches those and skips the record.
    """
    phases = tuple(_phase_from_json(p) for p in raw["phases"])
    return GameRecord(
        game_id=str(raw.get("id", "")),
        map_name=str(raw.get("map", "standard")),
        phases=phases,
    )


def iter_game_records(lines: Iterator[str]) -> Iterator[GameRecord]:
    """Parse a stream of JSONL lines into games, skipping blank lines.

    A malformed line raises. Callers that must survive one use
    :class:`PaquetteJSONLSource`, which records and skips them instead.
    """
    for line in lines:
        line = line.strip()
        if line:
            yield game_from_json(json.loads(line))


class PaquetteJSONLSource:
    """Read games from a ``.jsonl`` file or a ``.zip`` archive of ``.jsonl`` members.

    Malformed lines are counted in :attr:`skipped` rather than raised, so one
    corrupt record does not abort a pass over the whole dataset.

    ``skip`` and ``limit`` select the window ``[skip, skip + limit)`` of the
    *successfully parsed* stream, in the file's deterministic order. Both bounds
    count on the same basis (malformed lines advance neither), so two reads
    whose windows touch partition the stream exactly, which is what makes a
    holdout provably disjoint from its training set: train is ``limit=N`` and
    holdout is ``skip=N, limit=M`` (docs/setup.md, "Behavioral cloning").

    ``member`` restricts a zip to the one ``.jsonl`` member with that basename
    (the real dataset zip carries five: variant maps, no-press, and three press
    splits, concatenated in sorted order when ``member`` is None -- for training
    windows you almost always want exactly ``standard_no_press.jsonl``). A
    named member that is absent raises ``FileNotFoundError`` rather than
    yielding a silently empty stream.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        limit: int | None = None,
        skip: int = 0,
        member: str | None = None,
    ) -> None:
        if skip < 0:
            raise ValueError(f"skip must be non-negative, got {skip}")
        self.path = Path(path)
        self.limit = limit
        self.skip = skip
        self.member = member
        self.skipped: list[str] = []

    def __iter__(self) -> Iterator[GameRecord]:
        self.skipped = []
        parsed = yielded = 0
        for line in self._lines():
            if self.limit is not None and yielded >= self.limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                record = game_from_json(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                self.skipped.append(f"{type(error).__name__}: {error}")
                continue
            parsed += 1
            if parsed <= self.skip:
                continue
            yielded += 1
            yield record

    def _lines(self) -> Iterator[str]:
        if self.path.suffix == ".zip":
            with zipfile.ZipFile(self.path) as archive:
                names = [n for n in sorted(archive.namelist()) if n.endswith(".jsonl")]
                if self.member is not None:
                    names = [n for n in names if n.rsplit("/", 1)[-1] == self.member]
                    if not names:
                        raise FileNotFoundError(
                            f"zip member {self.member!r} not found in {self.path}"
                        )
                for name in names:
                    with archive.open(name) as handle:
                        yield from (raw.decode("utf-8") for raw in handle)
        else:
            with self.path.open(encoding="utf-8") as handle:
                yield from handle
