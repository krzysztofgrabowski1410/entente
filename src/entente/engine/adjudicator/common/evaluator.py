"""``Eval``: the derived predicates of docs/adjudication.tex Sec. 5.1 (Sec. 3 to Sec. 5).

:class:`Evaluator` is a pure function of a move labeling: every stationary
status, every strength value and every contest verdict is *recomputed* from the
stored relocating labels. That is what makes Sec. 5.1's stability test
(``L == Eval(L)``) meaningful, and it is the reason nothing here memoizes
across calls -- a cache would have to be invalidated on each tentative label
the enumerator tries.

Both engines consume this class: the enumerator calls it under tentative
complete labelings to test stability, the resolver subclasses it
(``resolver._Resolver``) with three-valued variants that read the same tables
and fall back to these methods once statuses are crisp. The derived stationary
statuses and result assembly are therefore shared verbatim, which is what makes
``resolver == enumerator`` a test of the labeling search alone.

This module changes only with the calculus itself -- Sec. 3's cut arithmetic,
Sec. 4's contests, the twins rules -- and it is the package's largest and most
rule-dense module by design.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, assert_never

from entente.engine.adjudicator.common.geometry import sea_chain_exists
from entente.engine.adjudicator.common.phase1 import entry_province
from entente.engine.adjudicator.common.results import AdjudicationError
from entente.engine.adjudicator.common.tables import Tables, WalkGroup
from entente.engine.orders import (
    RELOCATING_TYPES,
    Bar,
    Convoy,
    Dislodge,
    Hold,
    Move,
    Support,
)
from entente.engine.rules.protocol import CutExceptionScope
from entente.engine.rules.strength import (
    effective_support,
    is_relocating,
    pipeline_value,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.adjudicator.common.results import OrderRecord
    from entente.engine.map.graph import MapGraph
    from entente.engine.rules.protocol import Ruleset


class Evaluator:
    """``Eval`` of Sec. 5.1: recomputes every derived predicate from move labels.

    The enumerator consults it under tentative complete labelings (stability
    checking); both engines consult it under their final labeling for the
    derived stationary statuses and result assembly. ``status`` holds
    relocating-order labels (None = undecided; stationary orders' statuses
    are always derived, never stored); ``szykmaned`` marks moves failed by
    the Szykman policy (treated as holding, delivering no cuts -- Sec. 5.2).

    Method names follow the spec's own symbols where the spec has one --
    ``eff`` (a support's effectiveness, Sec. 2), ``cutval`` (Sec. 3.2), and the
    four DATC role strengths ``hold`` / ``attack`` / ``defend`` / ``prevent``
    (Sec. 4) -- so a reader can go from a formula in docs/adjudication.tex to
    the method that computes it. Everything the spec leaves unnamed is spelled
    out in English.
    """

    def __init__(
        self,
        graph: MapGraph,
        ruleset: Ruleset,
        records: Sequence[OrderRecord],
        tables: Tables,
    ) -> None:
        self.graph = graph
        self.ruleset = ruleset
        self.records = records
        self.tables = tables
        self.status: list[bool | None] = [None] * len(records)
        self.szykmaned: set[int] = set()
        self._engaging: set[WalkGroup] = set()

    # -- label plumbing ------------------------------------------------------

    def succeeded(self, index: int) -> bool:
        """Return a relocating order's stored label; reading an undecided one breaks Lemma L1."""
        value = self.status[index]
        if value is None:  # pragma: no cover - guarded by L1/topological order
            raise AdjudicationError(f"read of undecided order {index} violates Lemma L1")
        return value

    def is_move(self, index: int) -> bool:
        """Whether the record's effective order is relocating (Move or Dislodge)."""
        return is_relocating(self.records[index].effective)

    def move_of(self, index: int) -> Move:
        """Return the record's effective order as a Move; the caller guarantees the kind."""
        order = self.records[index].effective
        assert isinstance(order, Move)
        return order

    def relocating_of(self, index: int) -> Move | Dislodge:
        """Return the record's effective order as a Move or Dislodge; the caller guarantees it."""
        order = self.records[index].effective
        assert isinstance(order, RELOCATING_TYPES)
        return order

    def entry_target(self, index: int) -> str:
        """Return the province the record's relocating order contests entry at."""
        target = entry_province(self.records[index].effective)
        assert target is not None
        return target

    def is_convoyed(self, index: int) -> bool:
        """Whether the record's effective order is a convoyed Move."""
        order = self.records[index].effective
        return isinstance(order, Move) and order.via_convoy

    # -- the Dislodge/victim clash (Sec. 4.1) ----------------------------------------

    def victim_clash(self, dislodge: int, victim: int | None) -> bool:
        """Whether a Dislodge's victim answers it with an order entering the Dislodge's province.

        An overland Move into that province, or a Dislodge of a unit standing
        there: such an order and the Dislodge are two opposite entries along one
        edge, the victim cannot slip past the unit that attacks it, and the two
        fight an edge contest whatever the capacities of the two provinces
        (Sec. 4.1).
        """
        if victim is None:
            return False
        home = self.records[dislodge].unit.location.province
        order = self.records[victim].effective
        if isinstance(order, Move):
            return not order.via_convoy and order.destination.province == home
        return isinstance(order, Dislodge) and self.entry_target(victim) == home

    def lost_clash(self, dislodge: int, victim: int | None) -> bool:
        """Whether a Dislodge met its victim head-to-head and did not win (Sec. 4.1).

        The mirror of the test ``eval_move`` makes on the victim's side: the
        Dislodge loses unless its defend strictly exceeds the victim's, so a tie
        loses too.
        """
        return (
            victim is not None
            and self.victim_clash(dislodge, victim)
            and self.defend_value(dislodge) <= self.defend_value(victim)
        )

    def beaten_in_clash(self, index: int) -> bool:
        """Whether the order is a Dislodge that lost or tied its clash with its victim.

        Such a Dislodge fails wholly, and the clash is the only contest it
        takes part in: it neither stands in the way of the units entering the
        Area, nor stands off another Dislodge of its victim, nor makes a
        standoff (Sec. 4.2).
        """
        if not isinstance(self.records[index].effective, Dislodge):
            return False
        return self.lost_clash(index, self.dislodge_victim(index))

    def dislodge_victim(self, index: int) -> int | None:
        """Return the unit a Dislodge takes: its Sec. 3.4 victim, or the twin the walk engages."""
        if index in self.tables.walk_group_of:
            return self._walk_assignment(index)
        return self.tables.victim_of.get(index)

    def clash_candidates(self, index: int) -> list[int]:
        """Return the Dislodges that could meet this relocating order head-to-head.

        Those that stand where it enters and are aimed at its origin; whether
        one of them does meet it depends on the victim it takes.
        """
        order = self.relocating_of(index)
        if isinstance(order, Move) and order.via_convoy:
            return []
        target = self.entry_target(index)
        return [
            d
            for d in self.tables.into.get(order.location.province, ())
            if d != index
            and isinstance(self.records[d].effective, Dislodge)
            and self.records[d].unit.location.province == target
        ]

    def wave_mates(self, index: int) -> tuple[int, ...]:
        """Return the members of this order's wave, itself included (twins rule 2)."""
        return self.tables.wave_members.get(self._champion(index), (index,))

    def _clashing_dislodges(self, index: int) -> list[int]:
        """Return the Dislodges this order meets head-to-head: those that name its wave.

        A named Dislodge meets the wave of its victim. An unnamed one names
        every twin of its stack, so it meets the wave of every twin ordered
        into its own province, whichever twin the engagement later gives it:
        the clash is read off the orders, before the engagement (Sec. 4.4,
        rule 6). A wave meets a Dislodge as one, every member answering it
        with the same Move at the champion's strength. A Dislodge is a wave
        of one.
        """
        mates = self.wave_mates(index)
        clashing: list[int] = []
        for d in self.clash_candidates(index):
            if not self.exists_as_attacker(d):
                continue
            position = self.tables.walk_group_of.get(d)
            if position is not None:
                pool = self.tables.walk_groups[position].pool
                if any(mate in pool for mate in mates):
                    clashing.append(d)
            elif self.tables.victim_of.get(d) in mates:
                clashing.append(d)
        return clashing

    # -- twin-stack machinery (the rulebook's "Indistinguishable Units (twins)" rules 1-6) --

    def _champion(self, index: int) -> int:
        """Return the wave champion this record's contest values delegate to.

        Twins rule 2: the wave's strength is its strongest move's -- the member
        every support already binds; non-members delegate to themselves.
        """
        return self.tables.wave_of.get(index, index)

    def _walk_taken(self, group: WalkGroup) -> set[int]:
        """Return the pool members already claimed before the unnamed walk starts.

        These are twins a successful *named* Dislodge took (twins rule 6: "a
        Dislodge which names an action takes the twin so named"). Reads only
        stored statuses -- never a win test -- so the walk terminates even
        through crossed support-hold walls.
        """
        taken: set[int] = set()
        for member in group.pool:
            if self.remains(member) and any(
                self.succeeded(named) for named in self.tables.dislodges_on.get(member, ())
            ):
                taken.add(member)
        return taken

    def _walk_engagement(self, group: WalkGroup) -> dict[int, int | None]:
        """Give each unnamed attack on a twin stack its victim, or none.

        Unnamed attacks take twins from the top of the attacked player's list,
        one per attack, winners first (twins rule 6). Concretely: unnamed
        attacks sort strongest-first -- equal strengths form a cohort,
        attacker canonical order within -- and each cohort spreads over the
        first still-standing twins in the victim's list order. A cohort larger
        than the twins left STANDS OFF: its attacks and every weaker attack
        engage nothing and fail (the rulebook's "bouncing when there is
        insufficient space ... the strength of a move", generalizing the
        classic lone-victim standoff). A twin repelling an attack keeps
        standing and keeps absorbing later engagements; a fallen or
        claimed-departed twin is consumed. Every value reads the given
        labeling -- Eval stays a pure function of L, and the fixpoint
        machinery supplies the iteration.

        The clash comes first: an attack meets every twin of the stack ordered
        into its own province, and one that such a twin beats or ties
        (``_stack_beater``) takes no part in the walk -- it is paired with
        that twin for its clash alone, consumes nothing and joins no cohort
        (Sec. 4.2). An attack that wins its clash walks like any other.
        """
        engagement: dict[int, int | None] = dict.fromkeys(group.attacks)
        if group in self._engaging:
            # Re-entered through a cross-group strength chain: opaque this
            # path -- the fixpoint (or residue selection) supplies the
            # missing statuses on a later evaluation.
            return engagement
        self._engaging.add(group)
        try:
            taken = self._walk_taken(group)
            available = [m for m in group.pool if m not in taken]
            live_attacks = []
            for attack in group.attacks:
                if not self.exists_as_attacker(attack):
                    continue
                beater = self._stack_beater(attack, group)
                if beater is not None:
                    engagement[attack] = beater
                else:
                    live_attacks.append(attack)
            # Attack values never include victim-power supports (role
            # filter), so a same-group strength recursion is impossible (the
            # beaten test above reads the twins' defend, whose supports may
            # come from twins of this stack: that path re-enters and reads
            # the group as opaque, like a cross-group chain);
            # consumption reads STORED statuses, never a win test, so the
            # wall chain (hold -> delivered_eff -> dislodged -> engagement)
            # terminates. At a stable labeling the stored status and the
            # win test coincide.
            strengths = {a: self.attack_value(a, group.power) for a in live_attacks}
            ordered = sorted(live_attacks, key=lambda a: strengths[a], reverse=True)
            consumed: set[int] = set()
            i = 0
            while i < len(ordered):
                cohort = [ordered[i]]
                while (
                    i + len(cohort) < len(ordered)
                    and strengths[ordered[i + len(cohort)]] == strengths[cohort[0]]
                ):
                    cohort.append(ordered[i + len(cohort)])
                i += len(cohort)
                targets = [m for m in available if m not in consumed]
                if len(cohort) > len(targets):
                    break  # the tied cohort overflows: it and every weaker attack fail
                for attack, member in zip(cohort, targets, strict=False):
                    engagement[attack] = member
                    if not self.remains(member) or self.succeeded(attack):
                        consumed.add(member)
            return engagement
        finally:
            self._engaging.discard(group)

    def _stack_beater(self, attack: int, group: WalkGroup) -> int | None:
        """Return the first twin of the stack that meets this unnamed attack and wins or ties."""
        return next((m for m in group.pool if self.lost_clash(attack, m)), None)

    def _walk_assignment(self, index: int) -> int | None:
        """Return the twin one unnamed attack engages, or None when the attack fails.

        None covers a standoff-blocked attack, no twin left, and a dead
        attacker.
        """
        group = self.tables.walk_groups[self.tables.walk_group_of[index]]
        return self._walk_engagement(group).get(index)

    def _stack_dislodger(self, index: int) -> int | None:
        """Return the successful unnamed walk attack that took this stack member, if any.

        Named attacks are covered by ``dislodges_on``.
        """
        position = self.tables.stack_group_of.get(index)
        if position is None:
            return None
        group = self.tables.walk_groups[position]
        for attack, member in self._walk_engagement(group).items():
            if member == index and self.remains(index) and self.succeeded(attack):
                return attack
        return None

    # -- Sec. 5.1 derived predicates ----------------------------------------------

    def remains(self, index: int) -> bool:
        """Whether the unit stays: stationary, or a failed/szykmaned relocation (Sec. 4.3)."""
        if not self.is_move(index):
            return True
        return index in self.szykmaned or not self.succeeded(index)

    def unit_dislodged(self, index: int) -> bool:
        """Whether a remaining unit is dislodged (Sec. 4.2).

        It is dislodged iff a successful targeted Dislodge names it, an unnamed
        walk attack took it (twins rule 6), or -- under the implicit regime for
        its province's capacity -- a foreign relocating order into its province
        succeeded (admission at capacity 1 makes that entrant the displacement
        winner).
        """
        if not self.remains(index):
            return False
        for dislodger in self.tables.dislodges_on.get(index, ()):
            if self.succeeded(dislodger):
                return True
        if self._stack_dislodger(index) is not None:
            return True
        record = self.records[index]
        province = record.unit.location.province
        if not self.ruleset.implicit_displacement(self.graph.capacity(province)):
            return False
        if self.graph.capacity(province) != 1:
            return False  # implicit displacement needs a single-slot defender (Sec. 4.2)
        return any(
            entrant != index
            and entrant not in self.szykmaned
            and self.records[entrant].power != record.power
            and isinstance(self.records[entrant].effective, Move)
            and self.succeeded(entrant)
            for entrant in self.tables.into.get(province, ())
        )

    def alive_chain_seas(self, index: int) -> set[str]:
        """Sea provinces of this move's convoying fleets that are not dislodged."""
        return {
            self.records[c].unit.location.province
            for c in self.tables.convoys_for.get(index, ())
            if not self.unit_dislodged(c)
        }

    def viable(self, index: int) -> bool:
        """Whether the convoyed move has a viable chain (Sec. 3.5).

        Some chain of undislodged capable fleets, each convoying exactly this
        move, links origin to destination.
        """
        move = self.move_of(index)
        alive = self.alive_chain_seas(index)
        if not alive:
            return False
        return sea_chain_exists(
            self.graph,
            alive,
            self.graph.sea_neighbors(move.location.province),
            self.graph.sea_neighbors(move.destination.province),
        )

    def exists_as_attacker(self, index: int) -> bool:
        """Whether the relocating order exists as an attacker (Sec. 3.1.4).

        Overland attackers exist; convoyed ones iff viable; a szykmaned move is
        treated as holding and never exists (Sec. 5.2); a mis-targeted Dislodge
        fails outright and never exists (Sec. 3.4). An unnamed walk attack on a
        twin stack always exists -- it attacks the stack and cuts even when it
        finds no twin left (twins rule 6).
        """
        if index in self.szykmaned:
            return False
        order = self.records[index].effective
        if isinstance(order, Dislodge):
            if index in self.tables.walk_group_of:
                return True
            return self.tables.victim_of.get(index) is not None
        return not self.move_of(index).via_convoy or self.viable(index)

    def _is_protected_attacker(self, support_index: int, attacker_index: int) -> bool:
        """Whether ``attacker_index`` is the attacker this support's cut exception protects.

        Sec. 3.1 condition 3, with the scope the ruleset chooses: the directed
        unit alone, or every attacker from the protected province (Sec. 4.2).
        """
        if self.ruleset.cut_exception_scope is CutExceptionScope.DIRECTED_UNIT:
            return self.tables.protected_record_of.get(support_index) == attacker_index
        province = self.tables.protected_province_of.get(support_index)
        return (
            province is not None and self.records[attacker_index].unit.location.province == province
        )

    def qualifying_cutters(self, support_index: int) -> list[int]:
        """Return the relocating orders whose attack cuts this support (Sec. 3.1).

        A plain Move targeting the supporter's province when its capacity is 1,
        a Dislodge whose Sec. 3.4 victim is the supporter, or an unnamed attack
        on a twin stack that the tables assigned this support to -- such
        attacks cut the attacked player's written supports from the first
        listed downward, one per attack (twins rule 6); foreign; not the
        protected attacker; existing (Sec. 3.1.4).
        """
        record = self.records[support_index]
        province = record.unit.location.province
        cutters: list[int] = []
        for attacker in self.tables.into.get(province, ()):
            if self.records[attacker].power == record.power:
                continue  # own-power attacks never cut
            # ``into`` holds relocating records only; the typed accessor makes
            # that assumption loud, so the else-branch below is a Move by type
            # rather than by omission.
            order = self.relocating_of(attacker)
            if isinstance(order, Dislodge):
                if attacker in self.tables.walk_group_of:
                    if self.tables.cut_target_of.get(attacker) != support_index:
                        continue
                elif self.tables.victim_of.get(attacker) != support_index:
                    continue
            elif self.graph.capacity(province) != 1:
                continue  # a plain move into an Area attacks no unit (Sec. 3.1)
            if self._is_protected_attacker(support_index, attacker):
                continue
            if not self.exists_as_attacker(attacker):
                continue
            cutters.append(attacker)
        return cutters

    def eff(self, index: int) -> Fraction:
        """Return a support's effectiveness: what survives its cut (Sec. 2, Sec. 3.2).

        The modified base minus the one-shot cut of its qualifying cutters,
        floored at zero. No dislodgement is read here: cutting depends on a
        labeling only through which convoyed attackers exist.
        """
        cutvals = [self.tables.cutval_of[a] for a in self.qualifying_cutters(index)]
        cut = self.ruleset.cut_aggregation(cutvals)
        return effective_support(self.tables.support_base_of[index], cut)

    def delivered_eff(self, index: int) -> Fraction:
        """Return what a support delivers to its subject (Sec. 2, Sec. 3.3).

        A support gives its effectiveness only while its own unit stands: a
        dislodged giver delivers nothing. The one exception is a support of a
        Bar, which delivers whatever survives its cut -- "in the same moment in
        which that Unit is Dislodged, it Supports the Bar, so both happen". For
        such a support the giver's fate is not asked at all, rather than asked
        and set aside: that question is answered by the very Dislodge the Bar
        gates, so asking it would make the Bar's strength depend on the contest
        it decides.
        """
        eff = self.eff(index)
        if index in self.tables.bar_supports:
            return eff
        if eff <= 0 or self.unit_dislodged(index):
            return Fraction(0)
        return eff

    # -- Sec. 4 role values --------------------------------------------------------

    def prevent_value(self, index: int) -> Fraction:
        """Return the prevent value a relocating order exerts at its destination.

        Wave members share the champion's value: "The wave's Strength is the
        Strength of its strongest Move" (twins rule 2), and the champion is the
        member every support was concentrated onto (twins rule 1).
        """
        index = self._champion(index)
        base = self.tables.move_base_of[index]
        effs = (self.delivered_eff(s) for s in self.tables.supports_of.get(index, ()))
        return pipeline_value(base, effs)

    def defend_value(self, index: int) -> Fraction:
        """Return the value a head-to-head opponent defends with: every matching support counts."""
        return self.prevent_value(index)  # Sec. 4: all matching supports count

    def admission_rank(self, index: int) -> Fraction:
        """Return the rank a relocating order exerts as an entry candidate (Sec. 4.3).

        An entry candidate ranks by the strength of its *move*, not by its bare
        unit strength: a non-champion wave member ranks at its own modified
        base -- every support was concentrated onto the champion (twins
        rule 1), leaving the siblings supportless -- so a supported
        champion strictly outranks its siblings at a too-small door; champions
        and lone movers rank at full prevent. Gates stay wave-wide (bar_stops
        consults the champion's concentrated value for every member), and
        Dislodge standoffs are untouched (Dislodge records never join waves).
        """
        if self.tables.wave_of.get(index, index) != index:
            return self.tables.move_base_of[index]
        return self.prevent_value(index)

    def attack_value(self, index: int, defender_power: str | None) -> Fraction:
        """Return the value an entrant attacks with against a ``defender_power`` occupant.

        Supports given by the defender's own power never add to an attack on it
        (the Sec. 3.3 role filter); ``None`` filters nothing, the shape of an
        empty destination.
        """
        index = self._champion(index)
        base = self.tables.move_base_of[index]
        effs = (
            self.delivered_eff(s)
            for s in self.tables.supports_of.get(index, ())
            if self.records[s].power != defender_power  # Sec. 3.3 role filter
        )
        return pipeline_value(base, effs)

    def hold_value(self, index: int) -> Fraction:
        """Return the value a unit defends its province with (Sec. 4).

        Stationary units hold with modified base (fortress included) plus
        hold-supports; a unit whose relocating order failed defends with
        unmodified base alone (retention: Sec. 2; twins rule 5: a twin ordered
        to relocate gains nothing from Support Holds). A twin-stack member
        defends with the stack-wide wall instead of its per-unit bindings
        ("its own Strength plus ALL such Support Holds").
        """
        if self.is_move(index):
            return self.ruleset.base_strength(self.records[index].unit)
        base = self.tables.hold_base_of[index]
        wall = self.tables.wall_supports_of.get(index)
        supports = wall if wall is not None else self.tables.hold_supports_of.get(index, ())
        # A support-hold never defends its own giver: a member's own record is
        # excluded from its wall ("the support doesn't add strength to the unit
        # that does it"; a stack of six mutual supporters defends at 6, not 7).
        effs = (self.delivered_eff(s) for s in supports if s != index)
        return pipeline_value(base, effs)

    def bar_value(self, index: int) -> Fraction:
        """Return a Bar's contest value: its unmodified base plus its supports (Sec. 4.2).

        Mountain and fortress modifiers never apply to a Bar. Its supports are
        cut like any others and are not lost when their givers are dislodged,
        so no dislodgement reaches this value -- neither the barring unit's nor
        a backer's: it is the same under every labeling that agrees on which
        convoyed attackers exist.
        """
        base = self.ruleset.base_strength(self.records[index].unit)
        effs = (self.delivered_eff(s) for s in self.tables.supports_of.get(index, ()))
        return pipeline_value(base, effs)

    def prevent_nulled(self, index: int, at_province: str) -> bool:
        """Whether the unit's dislodger came from ``at_province``, nulling its prevent there.

        Sec. 4.2 prevent-nulling: a unit dislodged by a non-convoyed attack from
        its own destination exerts no prevent there (DATC 4.A.7 choice B).

        The attack must be the unit's *dislodger*: a successful targeted
        Dislodge naming it, or -- implicit regime, capacity 1 -- the successful
        entrant into its province, which at one slot is unique and hence the
        displacement winner. An unrelated plain entry from ``at_province``
        while a Dislodge from elsewhere displaces the unit nulls nothing (in an
        Area, entry displaces nobody); BATC.C.12 and C.13 certify the negative
        and positive directions.
        """
        record = self.records[index]
        if not self.unit_dislodged(index):
            return False
        for d in self.tables.dislodges_on.get(index, ()):
            # Dislodges are never convoyed (Sec. 3.5), so no convoy exception here;
            # succeeded + the victim remaining (unit_dislodged above) means the
            # Dislodge displaced exactly this unit, not a vacated-victim entry.
            if self.succeeded(d) and self.records[d].unit.location.province == at_province:
                return True
        walker = self._stack_dislodger(index)
        if walker is not None and self.records[walker].unit.location.province == at_province:
            return True
        origin = record.unit.location.province
        if not self.ruleset.implicit_displacement(self.graph.capacity(origin)):
            return False
        if self.graph.capacity(origin) != 1:
            return False  # implicit displacement needs a single-slot defender (Sec. 4.2)
        return any(
            d != index
            and d not in self.szykmaned
            and self.succeeded(d)
            and self.records[d].power != record.power
            and self.records[d].unit.location.province == at_province
            and isinstance(self.records[d].effective, Move)
            and not self.is_convoyed(d)
            for d in self.tables.into.get(origin, ())
        )

    # -- the entry/edge/displacement contests (Sec. 4.1-Sec. 4.3) ----------------------

    def bar_stops(self, index: int) -> bool:
        """Whether some matching Bar stops the move (Sec. 4.2 Bar contests).

        The move is stopped iff some matching Bar has value >= the move's
        prevent -- the Bar wins ties (the rulebook: "a tie favours the Barring
        Order"). A Bar cannot be cut and takes effect even if the barring unit
        is dislodged in the same phase (twins rule 4: "All Orders happen at the
        same time"), and a unit backing the Bar keeps backing it when it is
        dislodged itself, for the same reason; so no dislodgement the judged
        entry inflicts can weaken the gate that judges it. Wave members
        consult the champion's Bars and value (rule 3: one Bar gates the whole
        wave; the wave must beat EACH power's standing gate).
        """
        for bar in self.tables.bars_on.get(self._champion(index), ()):
            if self.prevent_value(index) <= self.bar_value(bar):
                return True
        return False

    def _live_for_ranking(self, index: int, destination: str) -> bool:
        """Whether a relocating order contests slots at ``destination`` (Sec. 4.3).

        It must exist as an attacker, not be szykmaned, and not be
        prevent-nulled. Bounced, barred and edge-beaten movers still exert
        prevent (classic standoff semantics, which extend to barred moves). A
        Dislodge that lost or tied its clash with its victim (Sec. 4.1) takes
        no part in the Area's admission, whether its victim remains or has
        left. Any other Dislodge whose victim remains contests the victim's
        slot, not the pool (Sec. 4.2); one whose victim has left contests the
        pool as a Move does, so it loses its prevent where its own unit is
        dislodged from there.
        """
        if index in self.szykmaned or not self.exists_as_attacker(index):
            return False
        order = self.relocating_of(index)
        if isinstance(order, Dislodge):
            victim = self.dislodge_victim(index)
            if victim is None:
                return False  # no twin left: the attack fails, contests nothing
            if self.remains(victim) or self.lost_clash(index, victim):
                return False
        return not self.prevent_nulled(index, destination)

    def _area_admission(self, index: int, destination: str, capacity: int | None) -> bool:
        """Whether the candidate is admitted at a multi-slot or unbounded province (Sec. 4.3).

        Stayers keep their slots (a dislodged stayer's slot passes to its
        dislodger, never to the pool); the candidate classes, by strictly
        descending admission rank (member-level: a wave's supported champion
        outranks its supportless siblings), are admitted whole while they fit;
        the first class that does not wholly fit bounces, and admission stops.
        """
        if capacity is None:
            return True
        stayers = [
            occupant
            for occupant in self.tables.records_at.get(destination, ())
            if occupant != index and self.remains(occupant)
        ]
        free = capacity - len(stayers)
        if free <= 0:
            return False
        pool = [
            candidate
            for candidate in self.tables.into.get(destination, ())
            if candidate == index or self._live_for_ranking(candidate, destination)
        ]
        classes: dict[Fraction, list[int]] = {}
        for candidate in pool:
            classes.setdefault(self.admission_rank(candidate), []).append(candidate)
        for rank in sorted(classes, reverse=True):
            members = classes[rank]
            if len(members) > free:
                return False  # the first non-fitting class bounces; admission stops
            if index in members:
                return True
            free -= len(members)
        raise AdjudicationError(  # pragma: no cover - index is always pooled
            f"candidate {index} missing from its own admission pool at {destination}"
        )

    def _eval_dislodge(self, index: int) -> bool:
        """Whether a targeted Dislodge succeeds (Sec. 4.2).

        The pseudo-province reduction: the dislodger must strictly beat every
        rival dislodger's prevent (several can stand each other off) and
        strictly beat the remaining victim's hold; the winner takes the
        victim's freed slot. A vacated victim collapses the contest into plain
        Sec. 4.3 entry.

        Twin stacks replace the rival standoff: every attack fights the same
        defence -- a twin's own strength plus every Support Hold bound to any
        member of the stack (twins rule 5); an unnamed attack engages the
        topmost available twin of the attacked player's list, or fails when
        none is left, and a named attack takes the twin so named unless an
        earlier attacker-order named winner already did (twins rule 6).

        A matching Bar gates the Dislodge like any entry: a Bar guards the door
        against the unit it names whatever that unit's entry order is, so the
        dislodger must beat the victim's defence AND strictly beat the bar, or
        fail wholly with the victim standing.

        A victim moving into the dislodger's own province meets the Dislodge
        head-to-head (Sec. 4.1): it cannot slip past, so the Dislodge never
        collapses into a plain entry there; it must strictly beat the victim's
        Move and displace a victim that resists with that Move's strength.
        """
        if self.bar_stops(index):
            return False  # barred: the victim stands, the dislodger stays home
        if any(
            self.defend_value(index) <= self.defend_value(d)
            for d in self._clashing_dislodges(index)
        ):
            return False  # Sec. 4.1: a Dislodge of its own unit met it head-to-head and held
        destination = self.entry_target(index)
        if index in self.tables.walk_group_of:
            walk_victim = self._walk_assignment(index)
            if walk_victim is None:
                return False  # "a Dislodge which finds no twin left fails"
            walk_power = self.records[walk_victim].power
            if self.victim_clash(index, walk_victim):
                # Head-to-head: the victim stays unless it wins, and resists with
                # the full strength of its Move.
                resistance = self.defend_value(walk_victim)
                return (
                    self.defend_value(index) > resistance
                    and self.attack_value(index, walk_power) > resistance
                )
            if not self.remains(walk_victim):
                return self._area_admission(index, destination, self.graph.capacity(destination))
            return self.attack_value(index, walk_power) > self.hold_value(walk_victim)
        victim = self.tables.victim_of.get(index)
        assert victim is not None  # exists_as_attacker gates mis-targets
        victim_power = self.records[victim].power
        clash = self.victim_clash(index, victim)
        if clash and self.defend_value(index) <= self.defend_value(victim):
            return False  # Sec. 4.1: the clash is lost or tied
        if clash or self.remains(victim):
            my_attack = self.attack_value(index, victim_power)
            # Rival named attacks on the same twin are a lone-victim contest:
            # the strictly strongest wins and must beat the stack-wide defence
            # (the twin's strength plus every Support Hold bound to any member
            # -- twins rule 5); an exact top tie bounces everyone. Stack
            # members and lone victims share one shape. A rival beaten in its
            # own clash with the victim stands nobody off.
            for rival in self.tables.dislodges_on.get(victim, ()):
                if rival == index or not self.exists_as_attacker(rival):
                    continue
                if self.lost_clash(rival, victim):
                    continue  # beaten in its own clash, it stands nobody off
                if my_attack <= self.prevent_value(rival):
                    return False
            resistance = self.defend_value(victim) if clash else self.hold_value(victim)
            return my_attack > resistance
        return self._area_admission(index, destination, self.graph.capacity(destination))

    def eval_move(self, index: int) -> bool:
        """Whether this relocating order is admitted under the labeling (Sec. 4.1-Sec. 4.3).

        Dislodge orders resolve the Sec. 4.2 targeted contest. Moves fight the
        Sec. 4.1 edge contests -- against the opposite Move where both ends hold
        one unit, and against every Dislodge that takes a member of their wave
        from their destination -- then Bar contests, then entry. At capacity 1
        under the implicit regime, the DATC-exact contest shape (certified by
        6.E): the entrant's *attack* value -- role-filtered by the power of the
        remaining occupant, unfiltered when the destination empties -- must
        strictly exceed every rival's *prevent* (6.E.7: a tie enabled only by
        the defender's own support still bounces) and strictly exceed the
        defender's resistance: *defend* for its head-to-head opponent (6.E.6),
        *hold* for any other stayer. Multi-slot provinces run the Sec. 4.3
        greedy class admission instead (Lemma L3: the two coincide at
        capacity 1).
        """
        if index in self.szykmaned or not self.exists_as_attacker(index):
            return False
        order = self.records[index].effective
        if isinstance(order, Dislodge):
            return self._eval_dislodge(index)
        move = self.move_of(index)
        origin = move.location.province
        destination = move.destination.province

        counter = self.tables.counter_of.get(index)
        edge = (
            counter is not None
            and counter not in self.szykmaned
            and self.ruleset.edge_contest(
                move,
                self.move_of(counter),
                self.graph.capacity(origin),
                self.graph.capacity(destination),
            )
        )
        if edge and counter is not None and self.defend_value(index) <= self.defend_value(counter):
            return False  # Sec. 4.1: ties fail both; the loser's move fails
        clashing = self._clashing_dislodges(index)
        if any(self.defend_value(index) <= self.defend_value(d) for d in clashing):
            return False  # Sec. 4.1: a Dislodge met head-to-head; ties fail both

        if self.bar_stops(index):
            return False

        capacity = self.graph.capacity(destination)
        if capacity != 1 or not self.ruleset.implicit_displacement(capacity):
            return self._area_admission(index, destination, capacity)

        stayers = [
            occupant
            for occupant in self.tables.records_at.get(destination, ())
            if self.remains(occupant)
        ]
        defender_power = self.records[stayers[0]].power if stayers else None
        my_attack = self.attack_value(index, defender_power)

        for rival in self.tables.into.get(destination, ()):
            if rival == index or not self.exists_as_attacker(rival):
                continue
            if self.prevent_nulled(rival, destination):
                continue
            if my_attack <= self.admission_rank(rival):
                return False  # Sec. 4.3: one tie at the top bounces everyone

        if capacity - len(stayers) >= 1:
            return True
        # Capacity-1 data: the single remaining stayer defends (Lemma L3).
        defender = stayers[0]
        if self.records[defender].power == self.records[index].power:
            return False  # Sec. 4.2: no self-dislodgement, prevent still stood
        resistance = (
            self.defend_value(defender)
            if (edge and counter == defender) or defender in clashing
            else self.hold_value(defender)
        )
        return my_attack > resistance

    # -- derived stationary statuses (Sec. 5.1) -------------------------------------

    def convoy_route_used(self, index: int) -> bool:
        """Whether the cargo's chosen route runs through this fleet.

        Both halves of a viable chain (origin->here, here->destination) must
        exist among live fleets.
        """
        cargo_index = self.tables.move_record_of_cargo.get(index)
        if cargo_index is None or cargo_index in self.szykmaned:
            return False
        if not self.viable(cargo_index):
            return False
        move = self.move_of(cargo_index)
        my_sea = self.records[index].unit.location.province
        alive = self.alive_chain_seas(cargo_index)
        if my_sea not in alive:
            return False
        return sea_chain_exists(
            self.graph, alive, self.graph.sea_neighbors(move.location.province), frozenset({my_sea})
        ) and sea_chain_exists(
            self.graph,
            alive,
            self.graph.sea_neighbors(move.destination.province),
            frozenset({my_sea}),
        )

    def derived_status(self, index: int) -> bool:
        """Return the Sec. 5.1 derived status of any record under the current labeling."""
        record = self.records[index]
        order = record.effective
        # is_relocating(order), spelled as a literal tuple: the Hold tail's
        # assert_never needs the else-branch elimination that only a literal
        # isinstance provides (a variable tuple narrows the match branch only).
        if isinstance(order, (Move, Dislodge)):
            return index not in self.szykmaned and self.succeeded(index)
        if not record.valid:
            return False
        if isinstance(order, Support):
            return self.delivered_eff(index) > 0
        if isinstance(order, Convoy):
            return not self.unit_dislodged(index) and self.convoy_route_used(index)
        if isinstance(order, Bar):
            # A Bar cannot be cut and fires even if its unit is dislodged in
            # the same phase (twins rule 4); a same-power duplicate against
            # one wave is converted into a Hold Order by the tables (rule 3:
            # barred is None) and FAILS as written.
            barred = self.tables.barred_move_of.get(index)
            if barred is None:
                return False
            return not self.succeeded(barred)  # the bar delivered: its move failed
        if isinstance(order, Hold):
            return not self.unit_dislodged(index)
        assert_never(order)


def final_walk_victims(evaluator: Evaluator) -> dict[int, int]:
    """Return the twin each unnamed attack on a twin stack took, under the evaluator's labeling.

    It reads the rule-6 engagement itself (``Evaluator._walk_engagement``:
    strongest first, a cohort of equal attacks spread over the first twins
    not yet consumed, a cohort larger than the twins left standing off)
    rather than replaying it, so under a settled labeling
    (``resolver.settle``) it names exactly the twins the adjudication
    dislodged. Maps each walked attack to the stack member it engaged when
    the attack succeeded or that member had departed (the vacated-victim
    entry); an attack that failed against a standing twin, stood off or
    found no twin left is absent. The Beleriand identity ledger reads it,
    since ``Tables.victim_of`` carries no entry for walked attacks, whose
    victim depends on the labeling.
    """
    victims: dict[int, int] = {}
    for group in evaluator.tables.walk_groups:
        for attack, member in evaluator._walk_engagement(group).items():
            if member is None:
                continue
            if evaluator.succeeded(attack) or not evaluator.remains(member):
                victims[attack] = member
    return victims
