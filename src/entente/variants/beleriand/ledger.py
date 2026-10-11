"""The Layer-2 unit ledger: per-unit facts that outlive the fungible board value.

The engine's :class:`~entente.engine.state.Unit` is a value: equal units are
fungible and the adjudicator may bind orders to either occurrence
(docs/adjudication.tex Sec. 1). Beleriand Layer 2 needs per-unit facts that
outlive phases -- mask counts (rulebook "Dwarven Masks"), double-unit bindings
(*"Double Units are associated with a certain Supply Centre"*), and the two
special identities (the Morgoth unit and Glaurung, both ``3A`` values of
MORGOTH). This module keeps those facts in creation-ordered identities and
re-binds them to board occurrences after every phase.

Where fungible twins leave a binding underdetermined, canonical rules apply:
occurrences bind in record-index order to identities in ascending-uid order,
except that outcomes *needing* a credential (a mask for devastated ground,
Glaurung for a Devastate, the special a designator names) claim the
lowest-uid identity satisfying it first. Board outcomes never depend on the
choice -- only Layer-2 bookkeeping does. Twins that carry different numbers
of masks are told apart: where the game layer itself chooses which twin a
removal takes, it takes the one with the fewest masks (:meth:`Ledger.assign`
with ``masks_first``, and the winter removal), and the twins trade no centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable

    from entente.engine.map.types import Location, UnitKind
    from entente.engine.state import State, Unit

MORGOTH_UNIT = "morgoth-unit"
GLAURUNG = "glaurung"

SPECIAL_DESIGNATOR: dict[str, str] = {MORGOTH_UNIT: "M", GLAURUNG: "G"}
"""Identity tag -> the order-notation designator letter (rulebook ``3A(M)`` /
``3A(G)``). The inverse of ``orders``' parse: :meth:`LedgerUnit.spec` prints it
so events and snapshot keys name the special unambiguously."""

GroupKey = tuple[str, str | None, str, str, int]
"""(province, coast, power, kind value, strength) -- the fungibility class."""


def group_key_of(unit: Unit) -> GroupKey:
    """Return the fungibility class of a board unit."""
    return (
        unit.location.province,
        unit.location.coast,
        unit.power,
        unit.kind.value,
        unit.strength,
    )


@dataclass(slots=True)
class LedgerUnit:
    """One tracked identity.

    ``location`` is the board position, or the dislodgement origin while
    ``limbo`` (awaiting the retreat phase). ``special`` tags the Morgoth unit
    or Glaurung, ``masks`` counts the Dwarven masks carried, ``bound_to``
    names the supply centre a double is associated with, and ``supplied_by``
    the centre whose point supplies the unit (a double's own centre; None for
    the specials and for a unit no centre can take yet).
    """

    uid: int
    power: str
    kind: UnitKind
    strength: int
    location: Location
    special: str | None = None
    masks: int = 0
    bound_to: str | None = None
    supplied_by: str | None = None
    limbo: bool = False

    @property
    def group(self) -> GroupKey:
        """The fungibility class this identity belongs to at its current position."""
        return (
            self.location.province,
            self.location.coast,
            self.power,
            self.kind.value,
            self.strength,
        )

    def spec(self) -> str:
        """Return the unit in order notation, the special designator included."""
        prefix = str(self.strength) if self.strength != 1 else ""
        tag = SPECIAL_DESIGNATOR.get(self.special) if self.special is not None else None
        designator = f"({tag})" if tag is not None else ""
        return f"{prefix}{self.kind.value}{designator} {self.location}"


@dataclass(slots=True)
class Ledger:
    """All identities of one game.

    Killed identities are dropped, and their facts die with them -- masks
    included, as the rulebook says: *"Masks are lost with the Unit that
    carries them"*.
    """

    units: dict[int, LedgerUnit] = field(default_factory=dict)
    next_uid: int = 0

    def seed(self, state: State) -> None:
        """Create identities for an opening position, in canonical state order.

        The first ``3A`` MORGOTH occurrence is the Morgoth unit (the rulebook
        start has exactly one, in Angband); strength-2 units bind to their
        starting province (*"Double Units at the start of the game are tagged
        to a Supply Centre in which they start the game"*).
        """
        morgoth_seen = False
        for unit in state.units:
            special = None
            if unit.power == "MORGOTH" and unit.strength == 3 and not morgoth_seen:
                special = MORGOTH_UNIT
                morgoth_seen = True
            bound = unit.location.province if unit.strength == 2 else None
            self.create(unit, special=special, bound_to=bound)

    def create(
        self, unit: Unit, *, special: str | None = None, bound_to: str | None = None
    ) -> LedgerUnit:
        """Create a new identity for ``unit`` with the next uid."""
        entry = LedgerUnit(
            uid=self.next_uid,
            power=unit.power,
            kind=unit.kind,
            strength=unit.strength,
            location=unit.location,
            special=special,
            bound_to=bound_to,
        )
        self.units[entry.uid] = entry
        self.next_uid += 1
        return entry

    def kill(self, uid: int) -> LedgerUnit:
        """Drop the identity ``uid`` and return it."""
        return self.units.pop(uid)

    def alive(self) -> list[LedgerUnit]:
        """Return every live identity in ascending uid order."""
        return [self.units[uid] for uid in sorted(self.units)]

    def board(self) -> list[LedgerUnit]:
        """Return the live identities standing on the board."""
        return [u for u in self.alive() if not u.limbo]

    def limbo(self) -> list[LedgerUnit]:
        """Return the live identities dislodged and awaiting a retreat."""
        return [u for u in self.alive() if u.limbo]

    def special(self, tag: str) -> LedgerUnit | None:
        """Return the live special tagged ``tag``, or None once it is dead."""
        for unit in self.alive():
            if unit.special == tag:
                return unit
        return None

    def board_groups(self) -> dict[GroupKey, list[int]]:
        """Return the board identities' uids by fungibility class, ascending."""
        groups: dict[GroupKey, list[int]] = {}
        for unit in self.board():
            groups.setdefault(unit.group, []).append(unit.uid)
        return groups

    def limbo_groups(self) -> dict[GroupKey, list[int]]:
        """Return the limbo identities' uids by fungibility class, ascending."""
        groups: dict[GroupKey, list[int]] = {}
        for unit in self.limbo():
            groups.setdefault(unit.group, []).append(unit.uid)
        return groups

    def count_credentialed(self, uids: Iterable[int], need: Callable[[LedgerUnit], bool]) -> int:
        """Count the identities among ``uids`` that satisfy ``need``."""
        return sum(1 for uid in uids if need(self.units[uid]))

    def assign(
        self,
        uids: list[int],
        demands: list[Callable[[LedgerUnit], bool] | None],
        *,
        masks_first: bool = False,
        fewest_masks: Collection[int] = (),
    ) -> list[int | None]:
        """Bind one identity per demand slot, credentialed slots first.

        Slots with a credential predicate claim the lowest-uid satisfying
        identity first; the rest take ascending uids, or with ``masks_first``
        the identities that carry the most masks first (ties: ascending uid),
        so that the identities left over are those with the fewest. The plain
        slots whose position is in ``fewest_masks`` claim, before the other
        plain slots, the identity with the fewest masks (ties: ascending uid).
        Returns one uid per slot: None when a credentialed slot cannot be
        satisfied (the caller must have prevented that via claim
        preprocessing), while plain slots always bind while identities remain.
        """
        remaining = sorted(uids)
        chosen: list[int | None] = [None] * len(demands)
        for position, need in enumerate(demands):
            if need is None:
                continue
            for uid in remaining:
                if need(self.units[uid]):
                    chosen[position] = uid
                    remaining.remove(uid)
                    break
        for position in sorted(fewest_masks):
            if demands[position] is None and remaining:
                uid = min(remaining, key=lambda u: (self.units[u].masks, u))
                remaining.remove(uid)
                chosen[position] = uid
        if masks_first:
            remaining.sort(key=lambda uid: (-self.units[uid].masks, uid))
        for position, need in enumerate(demands):
            if need is None and chosen[position] is None and remaining:
                chosen[position] = remaining.pop(0)
        return chosen
