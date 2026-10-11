"""The standard ruleset: the Sec. 7 standard column of docs/adjudication.tex as data.

Classic Diplomacy is the <base == 1, no modifiers, cap == 1 (map data), implicit
displacement, edge contests on, numeric one-shot cuts with the province-scoped
exception, core registry, {all-succeed, Szykman} policies> instantiation of
the generalized calculus (Theorem T1). Everything here is a data/strategy
object; the adjudicator contains no ``if ruleset is standard`` branch --
DATC certifies the claim through the generalized core.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from entente.engine.orders import Convoy, Hold, Move, Support
from entente.engine.rules.protocol import (
    AdmissionPolicy,
    BackupAction,
    CutExceptionScope,
    ResidueClass,
    StrengthRole,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from entente.engine.map.graph import MapGraph
    from entente.engine.map.types import Province
    from entente.engine.orders import Order
    from entente.engine.state import State, Unit

_ONE = Fraction(1)


@dataclass(frozen=True, slots=True)
class StandardRuleset:
    """Classic Diplomacy as calculus parameters (docs/adjudication.tex Sec. 7)."""

    @property
    def name(self) -> str:
        """The registry key ``"standard"``."""
        return "standard"

    @property
    def power_symmetric(self) -> bool:
        """True: classic Diplomacy's outcomes are equivariant under relabeling the powers."""
        return True

    @property
    def order_registry(self) -> frozenset[type]:
        """Sec. 1/Sec. 7: the core movement-phase registry -- Hold, Move, Support, Convoy."""
        return frozenset({Hold, Move, Support, Convoy})

    @property
    def strict_designation(self) -> bool:
        """DATC 4.C.2 choice b: a written unit designation is advisory.

        ``F Armenia`` for an army still binds the army (6.D.35). Keeping the
        advisory default here is what makes the standard column byte-identical
        under the knob.
        """
        return False

    @property
    def admission_policy(self) -> AdmissionPolicy:
        """Sec. 4.3: greedy descending prevent classes (= classic bounce at cap 1, L3)."""
        return AdmissionPolicy.GREEDY_DESCENDING_CLASSES

    @property
    def cut_exception_scope(self) -> CutExceptionScope:
        """Sec. 3.1.3, standard column: protected from the directed-against *province*.

        That is the supported move's destination (support-holds protect
        nothing), overland or by convoy alike (DATC 4.A.4 choice A).
        """
        return CutExceptionScope.DIRECTED_PROVINCE

    def base_strength(self, unit: Unit) -> Fraction:
        """Sec. 2/Sec. 7: ``base == 1`` -- the degeneracy premise of Lemmas L2/L3."""
        return _ONE

    def modified_base(
        self,
        role: StrengthRole,
        unit: Unit,
        order: Order,
        graph: MapGraph,
        *,
        disembark: frozenset[str] | None = None,
    ) -> Fraction:
        """Sec. 2/Sec. 7: no modifiers -- every role sees the unmodified base.

        A convoyed move's ``disembark`` seas are offered to every ruleset and
        unread here: a classic landing pays no toll.
        """
        return _ONE

    def cut_aggregation(self, cutvals: Iterable[Fraction]) -> Fraction:
        """Sec. 3.2: the sum of all qualifying attacks (binary rule at base == 1, L2)."""
        return sum(cutvals, Fraction(0))

    def edge_contest(
        self, move_a: Move, move_b: Move, cap_a: int | None, cap_b: int | None
    ) -> bool:
        """Whether two opposed moves fight an edge contest (Sec. 4.1/Sec. 7).

        Both endpoints displacement-governed (capacity 1), neither move
        convoyed -- convoyed swaps are legal (DATC 6.G behavior).
        """
        return cap_a == 1 and cap_b == 1 and not move_a.via_convoy and not move_b.via_convoy

    def implicit_displacement(self, capacity: int | None) -> bool:
        """Whether the top entry candidate contests the defender: everywhere (Sec. 4.2/Sec. 7).

        Classic maps are capacity-1 by data.
        """
        return True

    def convoy_capability(self, fleet: Unit, cargo: Unit) -> bool:
        """Sec. 3.5/Sec. 7: any fleet may convoy any army."""
        return True

    def entry_permission(self, unit: Unit, province: Province) -> bool:
        """Sec. 4.3/Sec. 7: no restricted provinces on classic maps."""
        return True

    def backup_policy(self, residue: ResidueClass) -> BackupAction:
        """Sec. 5.2: move cycles all succeed (DATC 6.C); convoy-entangled residues get Szykman.

        The Szykman rule is DATC 4.A.2 choice F.
        """
        if residue is ResidueClass.PURE_CYCLE:
            return BackupAction.ALL_MOVES_SUCCEED
        return BackupAction.SZYKMAN

    def update_supply_ownership(
        self, graph: MapGraph, state: State, ownership: Mapping[str, str]
    ) -> Mapping[str, str]:
        """Apply the Sec. 9 standard rule: post-Autumn occupants claim their supply centers.

        Unoccupied centers keep their owner.
        """
        updated = dict(ownership)
        for unit in state.units:
            province = unit.location.province
            if graph.province(province).is_supply_center:
                updated[province] = unit.power
        return updated


STANDARD = StandardRuleset()
