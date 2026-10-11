"""Ruleset protocol -- the seam that makes the adjudicator ruleset-generic.

The adjudicator is a pure function of ``(ruleset, state, order sheets)``.
Everything rule-specific is expressed through this protocol, along these axes
(the hook table is docs/adjudication.tex Sec. 7):

- **strength model** -- base unit strength plus composable modifiers (mountain
  barriers, fortresses), computed in exact rational arithmetic
  (`fractions.Fraction`); classic Diplomacy is the ``strength == 1``
  instantiation.
- **support-cut model** -- one shared numeric one-shot arithmetic (Sec. 3.2); the
  rulesets differ in the *protected-attacker scope* (province vs unit) and
  keep the same sum aggregation (binary cutting is the base == 1 degeneracy).
- **province capacity + entry rule** -- greedy admission of whole descending
  strength classes with tie-overflow bouncing; the classic bounce rule is the
  capacity-1 case (Lemma L3). Which capacities allow *implicit* displacement
  is ruleset policy; targeted displacement (Dislodge) is registry data.
- **order registry** -- which order types exist (Move/Support/Convoy/... plus
  variant orders such as targeted Dislodge and Bar) and how they join contests.
- **backup policy** -- resolution of paradoxical/cyclic components (circular
  movement, convoy paradoxes -> e.g. the Szykman rule), chosen per ruleset.

The interface is derived from exactly the two shipped rulesets, classic
Diplomacy and Beleriand, and it is **frozen**: every member exists because the
second ruleset proved it necessary, never speculatively, and a new member needs
a ruleset that needs it. ``strict_designation`` is the one knob that reads a
rulebook choice (4.C.2 choice C) rather than a calculus axis; its advisory
default keeps the standard column byte-identical.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from fractions import Fraction

    from entente.engine.map.graph import MapGraph
    from entente.engine.map.types import Province
    from entente.engine.orders import Move, Order
    from entente.engine.state import State, Unit


class AdmissionPolicy(Enum):
    """Entry rule for a province's candidate moves (docs/adjudication.tex Sec. 4.3).

    Both current instantiations use the same rule -- greedy admission of whole
    descending-prevent classes, stopping at the first class that does not fit
    (= the classic bounce rule at capacity 1, Lemma L3). The axis exists in Sec. 7
    because it is ruleset *data*, not engine behavior.
    """

    GREEDY_DESCENDING_CLASSES = "greedy-descending-classes"


class StrengthRole(Enum):
    """The pipeline role a strength value is computed for (Sec. 2/Sec. 4).

    ``MOVE`` covers every value a relocating order carries (attack, prevent,
    defend); ``SUPPORT`` is a support's contributed strength; ``HOLD`` is a
    stationary unit's resistance. Retention -- a failed mover defending its
    own province -- is *not* a role: it is unmodified base by Sec. 4.
    """

    MOVE = "move"
    SUPPORT = "support"
    HOLD = "hold"


class CutExceptionScope(Enum):
    """Sec. 3.1 condition 3 -- what a support is protected *from* (docs Sec. 3.1/Sec. 4.2)."""

    DIRECTED_PROVINCE = "directed-province"
    """Standard: not cut by an attack from the province the support is
    directed against (the supported move's destination; support-holds protect
    nothing) -- even when that attack arrives by convoy (DATC 4.A.4 choice A)."""

    DIRECTED_UNIT = "directed-unit"
    """Beleriand: not cut by the specific *unit* the support's subject order
    is directed against (a different unit attacking from the same Area does
    cut) -- the pseudo-province reduction derives this from the standard rule
    (Sec. 4.2)."""


class ResidueClass(Enum):
    """Classification of a residual undecided component (docs/adjudication.tex Sec. 5.2)."""

    PURE_CYCLE = "pure-cycle"
    CONVOY_ENTANGLED = "convoy-entangled"


class BackupAction(Enum):
    """Backup-policy verdict for a residual component (docs/adjudication.tex Sec. 5.2)."""

    ALL_MOVES_SUCCEED = "all-moves-succeed"
    """Circular-movement rule: the cycle's relocating orders all succeed."""

    SZYKMAN = "szykman"
    """Convoyed moves with undecided viability fail and deliver no cuts;
    their fleets are treated as holding (DATC 4.A.2 choice F)."""


@runtime_checkable
class Ruleset(Protocol):
    """A complete parameterization of the adjudication calculus.

    The full Sec. 7 table of docs/adjudication.tex: the hooks classic Diplomacy
    needs plus those the Beleriand ruleset needs (strength modifiers,
    protected-attacker scope, implicit-displacement regime, convoy capability,
    entry permission, variant order types) and the ``strict_designation``
    knob. The protocol is frozen: it changes only when a second ruleset
    proves a member necessary.
    """

    @property
    def name(self) -> str:
        """Canonical ruleset identifier, e.g. ``"standard"`` or ``"beleriand"``."""
        ...

    @property
    def power_symmetric(self) -> bool:
        """Whether outcomes are equivariant under relabeling powers.

        True for classic Diplomacy; False for rulesets with asymmetric powers or
        order-sheet-sequence-dependent targeting (e.g. Beleriand). Drives which
        property-based invariants the test suite asserts for this ruleset.
        """
        ...

    @property
    def order_registry(self) -> frozenset[type]:
        """Movement-phase order types this ruleset accepts (Sec. 1/Sec. 7).

        Orders outside the registry are invalid and become Hold in Phase 0.
        """
        ...

    @property
    def strict_designation(self) -> bool:
        """Phase-0 unit-designation strictness: the rulebook's reading of DATC 4.C.2.

        A power's sheet binds each order to its own unit occurrences by
        province, with the written kind/strength prefix narrowing the match
        exactly first. This knob decides what a prefix that matches *no* such
        occurrence means:

        - **False (advisory, DATC 4.C.2 choice b)** -- the prefix is advisory;
          the order still binds to the province (``F Armenia`` for an army is
          read as the army's order). Standard column, DATC 6.D.35.
        - **True (strict, the rulebook's DATC 4.C.2 choice C)** -- the prefix contradicts
          every unit of that power in the province, so the order is invalid and
          degrades to Hold (4.E.1); a correctly-designated order to the same
          occurrence still wins (4.D.3). Beleriand column.

        The advisory default makes the standard column byte-identical, so the
        knob is invisible to DATC.
        """
        ...

    @property
    def admission_policy(self) -> AdmissionPolicy:
        """Entry rule consuming prevent strengths and capacity (Sec. 4.3)."""
        ...

    @property
    def cut_exception_scope(self) -> CutExceptionScope:
        """Sec. 3.1 condition 3: whether protection is per province or per unit."""
        ...

    def base_strength(self, unit: Unit) -> Fraction:
        """Return the base strength of a unit (Sec. 2).

        Standard: ``1`` for every unit; Beleriand: the unit's own base
        strength (1/2/3). All strength arithmetic is exact rational arithmetic;
        no floats.
        """
        ...

    def modified_base(
        self,
        role: StrengthRole,
        unit: Unit,
        order: Order,
        graph: MapGraph,
        *,
        disembark: frozenset[str] | None = None,
    ) -> Fraction:
        """Return the unit's base strength after role modifiers (Sec. 2 ``mods``).

        Standard: the identity (no modifiers). Beleriand: a MOVE across a
        mountain-barrier edge in its penalized direction, or a SUPPORT through
        one in that direction, is halved (an exact rational 1/2; directional:
        only the marked crossing direction pays) -- a convoyed MOVE tolls its
        disembark crossings instead of its origin edge; HOLD gains +1 in a
        fortress province iff the unit's order is not relocating (a failed
        Move or Dislodge forfeits the bonus).

        ``disembark`` carries a convoyed MOVE's *ordered disembark seas*: the
        seas touching the landing province that the fleets ordered to convoy
        exactly this move link to the origin coast
        (``Tables.disembark_seas_of``). The adjudicator computes the set for
        every convoyed move under every ruleset and passes ``None`` for every
        other order; a hook the order's own fields cannot answer -- which sea
        is "the last Sea of the Convoy" -- reads it from here. Standard ignores
        it. Beleriand halves the move iff the set is non-empty and every sea
        in it is penalized into the landing province.

        Static per order sheet: the seas are read off the written convoy
        orders, and modifiers never depend on resolution.
        """
        ...

    def cut_aggregation(self, cutvals: Iterable[Fraction]) -> Fraction:
        """Combine qualifying cutters' values into one cut (Sec. 3.2).

        Both instantiations: the **sum** of all qualifying attacks (which
        degenerates to the classic binary rule at base == 1, Lemma L2).
        """
        ...

    def edge_contest(
        self, move_a: Move, move_b: Move, cap_a: int | None, cap_b: int | None
    ) -> bool:
        """Whether two opposed moves fight a head-to-head edge contest (Sec. 4.1).

        Standard: both endpoints displacement-governed (capacity 1) and
        neither move convoyed -- convoyed swaps are legal (DATC 6.G behavior).
        Beleriand: the same predicate; Areas are not displacement-governed,
        so cross-Area "swaps" are two independent entries (BATC certifies
        this).
        """
        ...

    def implicit_displacement(self, capacity: int | None) -> bool:
        """Whether the top entry candidate may free a stayer's slot by untargeted displacement.

        The contest is at a province of this capacity (Sec. 4.2/Sec. 4.3).
        Standard: yes (its maps are capacity-1 throughout). Beleriand: only
        at capacity 1 (ordinary provinces and off-map boxes behave
        classically); in Areas only a targeted Dislodge frees a slot.
        """
        ...

    def convoy_capability(self, fleet: Unit, cargo: Unit) -> bool:
        """Whether ``fleet`` may serve in a convoy chain carrying ``cargo`` (Sec. 3.5).

        Standard: any fleet. Beleriand: ``base(fleet) >= base(cargo)`` --
        convoying a Double Army requires Double Fleets (rulebook "Fleets").
        """
        ...

    def entry_permission(self, unit: Unit, province: Province) -> bool:
        """Whether ``unit`` may enter ``province`` at all (Sec. 4.3 gate; Sec. 7).

        Standard: always. Beleriand: Angband/Eriador are power-restricted and
        one-way (``Province.restricted_to`` / ``Province.one_way`` map data);
        re-entering Angband is reserved to Morgoth's strength-3 units (the
        rulebook's Morgoth-unit/Glaurung exception). Consulted for movement,
        retreat destinations, and legal-order enumeration.
        """
        ...

    def backup_policy(self, residue: ResidueClass) -> BackupAction:
        """Resolution policy for components with 0 or >=2 stable labelings (Sec. 5.2).

        Both rulesets (Beleriand's rulebook 4.A list): move cycles all
        succeed; convoy-entangled residues get the Szykman rule.
        """
        ...

    def update_supply_ownership(
        self, graph: MapGraph, state: State, ownership: Mapping[str, str]
    ) -> Mapping[str, str]:
        """Apply the post-Autumn supply-center ownership update (Sec. 9).

        Standard: every supply center occupied after the Fall retreats
        changes owner to the occupant's power; unoccupied centers keep their
        owner. Ruleset-parameterized because variant ownership rules differ
        (Beleriand's Layer-2 elven two-step ownership overrides this hook
        through ``DynamicBeleriandRuleset.update_supply_ownership``).
        """
        ...
