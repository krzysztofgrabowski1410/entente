"""The staged fixpoint resolver: docs/adjudication.tex Sec. 6, "Phases 2-4 as monotone stages".

Every relocating order carries a three-valued status (``bool | None``,
None = UNDECIDED) plus a Szykman flag; the stages S1 (convoy viability), S2
(one-shot supports/cuts) and S3 (edge, Bar, displacement -- targeted and
implicit -- entry with the vacancy iteration, failed-mover dislodgement) are
iterated in rounds until a round decides nothing. A round evaluates every
undecided order against the same statuses, takes the verdicts the stages
force, and then commits the vacancy cycles that are left (Sec. 6's vacancy
commit, which is the all-succeed policy applied in-line and is not monotone).
Propagation runs over all the orders of the phase at once; the
Phase-1 SCC structure (shared ``common`` machinery) governs the residue
step: UNDECIDED residue is grouped into sub-SCCs of the dependency graph
restricted to still-undecided orders, and the first residual SCC in
dependencies-first emission order is settled by the shared Sec. 5.1 selection
(``common.resolve_residue``: exact stable-labeling enumeration -- unique ->
committed, unanimity -> committed and re-enumerated, only genuine ambiguity
classified per Sec. 5.2 and handed to the ruleset's backup policy), then
propagation resumes (Sec. 6 "Residue -> selection -> resume"). The set that
propagation leaves undecided may be larger than the genuinely ambiguous
residue -- propagation cannot case-split
on a Dislodge victim's departure, so components with a *unique* stable
labeling can stall (the position W with its Dislodge also supported by a 2A,
tests/engine/test_residue_selection.py) -- which is why the residue
step recomputes the Sec. 5.1 selection exactly instead of applying the backup
policy outright.

Implementation choices where the spec underdetermines the letter (all
conformance-neutral -- the observable outcomes are fixed by Sec. 5/Sec. 6; each is
certified by ``resolver == enumerator`` over the full DATC + BATC and the
random differential):

- **Derived bounds, not stored intervals.** The state of propagation is the
  status vector alone; the Sec. 6 strength intervals ``[lo, hi]`` are recomputed
  per query as monotone functions of the decided statuses (endpoints: sums
  over supports whose delivery is forced/possible, Sec. 3 arithmetic over the
  statically modified bases). Narrowing is then automatic -- a decided status
  can only shrink every interval it feeds -- so no interval needs any
  bookkeeping. The stored state is finite (3^relocating), a round only turns
  None into True/False (asserted), and each selection decides every order of
  its residue: the T2 termination argument, which needs no monotonicity, with
  a hard round guard as an implementation-bug backstop.
- **Both-branch occupant analysis.** Where a capacity-1 admission depends on
  an occupant whose departure is UNDECIDED, the contest is evaluated under
  both branches (occupant stays: role-filtered attack, displacement contest;
  occupant leaves: unfiltered attack, free slot) and committed only when
  they agree; a targeted Dislodge with an undecided victim departure gets the
  same treatment (victim remains: Sec. 4.2 contest; victim gone: plain Sec. 4.3
  entry) -- sound because both are exactly the Sec. 6 requirement that
  propagation derive only forced facts.
- **Crisp-inputs area admission.** The Sec. 4.3 greedy class admission at
  capacity > 1 is decided exactly when every input is decided (stayer
  departures, candidate liveness, point prevent intervals); before that,
  three sound short-circuits fire -- everyone-fits (slack lower bound covers
  every possible candidate), no-slots (upper bound <= 0), and
  over-subscription (certainly-live candidates certainly ranked at-or-above
  this move already exceed the slot upper bound -- whether they fit or
  overflow, this move never enters). Everything else defers to a later
  round, the vacancy pass, or the residue policy.
- **Uncorrelated conservatism.** Interval comparisons and ``nulled3`` treat
  distinct uncertain inputs as independent (no case-splitting on a shared
  undecided convoy existence, no per-branch nulling): strictly conservative
  -- a fact is decided only if forced, so T3a soundness holds by
  construction; at worst a decidable case is deferred to a later round or
  the residue policy. The differential suite is the standing check that
  this conservatism never changes an outcome.
- **Vacancy optimism, gated (R8).** The Sec. 6 S3 vacancy iteration
  seeds only plain Moves whose admission is TRUE under the assumption that
  every undecided occupant-mover of the destination departs, with every
  other gate decided (exists TRUE, bar gate TRUE) and no edge contest (R8:
  corridor pairs are never optimistic -- they resolve as Sec. 4.1 contests).
  Multi-slot destinations join naturally: the assumption set holds all
  undecided leavers, and the Sec. 4.3 admission is re-evaluated under it.
  Downgrades cascade (survivors' assumed departures must be seed-internal or
  already decided); the surviving set commits to SUCCEEDS -- the Sec. 5.2 cycle
  policy applied in-line, per dependency component by construction. Targeted
  Dislodges are never seeded (their slot comes from the victim, not from
  vacancy; the vacated-victim entry case defers to propagation or residue).
- **Residual-SCC order.** "Topologically-first, ties by smallest canonical
  province index" is realized as first-emitted component of the deterministic
  dependencies-first Tarjan over canonically indexed records (identical
  tie-break data, one code path -- ``common.condensation_order``).
- **Szykman scope.** ``resolve_residue`` tracks the viability of every
  convoyed move of the chosen residual SCC -- including moves whose own
  status propagation already failed for other reasons (their cut delivery
  was still open; Sec. 5.2 "cuts those moves would have delivered do not
  happen") -- and the policy marks exactly those whose viability *varies
  across the residue's stable labelings*, the Sec. 5.2 scope computed exactly
  rather than from the interval fixpoint. Dislodges are never convoyed
  (Sec. 3.5), so Szykman never touches them.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from entente.engine.adjudicator.common import (
    AdjudicationError,
    Evaluator,
    PhaseResult,
    WalkGroup,
    assemble,
    build_tables,
    condensation_order,
    phase0,
    residue_adjacency,
    resolve_residue,
    sea_chain_exists,
)
from entente.engine.map.loader import load_map
from entente.engine.orders import Bar, Convoy, Dislodge, Move, Support
from entente.engine.state import State, check_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.adjudicator.common import OrderRecord, Tables
    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import Order
    from entente.engine.rules.protocol import Ruleset

_ZERO = Fraction(0)

Bounds = tuple[Fraction, Fraction]
"""A Sec. 6 strength interval [lo, hi]: every completion of the undecided
statuses yields a value inside it."""


def _and3(*values: bool | None) -> bool | None:
    """Kleene conjunction: False dominates, None is unknown."""
    result: bool | None = True
    for value in values:
        if value is False:
            return False
        if value is None:
            result = None
    return result


def _or3(*values: bool | None) -> bool | None:
    """Kleene disjunction: True dominates, None is unknown."""
    result: bool | None = False
    for value in values:
        if value is True:
            return True
        if value is None:
            result = None
    return result


def _not3(value: bool | None) -> bool | None:
    """Kleene negation: None stays unknown."""
    return None if value is None else not value


def _meet3(a: bool | None, b: bool | None) -> bool | None:
    """Return the consensus of two branches whose selector is unknown."""
    return a if a == b else None


def _gt3(a: Bounds, b: Bounds) -> bool | None:
    """Three-valued ``a > b`` on intervals: forced when the ranges separate."""
    if a[0] > b[1]:
        return True
    if a[1] <= b[0]:
        return False
    return None


class _Resolver(Evaluator):
    """Three-valued Sec. 5.1 facts over a *partial* labeling.

    Extends the shared two-valued ``Eval`` (used after the fixpoint decides
    everything, for derived statuses and assembly) with the ``*3``/``bounds``
    forms the stages propagate. Every method is monotone in the decided-status
    set: turning a None status into True/False can only turn None facts into
    True/False and shrink bounds, never the reverse (T2).

    ``_assumed_departed`` supports the vacancy pass: occupants in the set
    read as departed (``remains3`` False), letting the seed and cascade share
    the ordinary admission machinery under an explicit assumption.
    """

    _assumed_departed: frozenset[int] = frozenset()

    # -- three-valued primitive facts ------------------------------------------

    def status3(self, index: int) -> bool | None:
        """Return a relocating order's success; szykmaned moves are decided failures."""
        if index in self.szykmaned:
            return False
        if index in self._assumed_departed:
            return True
        return self.status[index]

    def remains3(self, index: int) -> bool | None:
        """Whether the unit stays, three-valued: stationary orders always do."""
        if not self.is_move(index):
            return True
        return _not3(self.status3(index))

    def _walk_taken3(self, group: WalkGroup) -> set[int] | None:
        """Return the pre-walk claims (named winners), or None while some claim is undecided.

        An undecided claim makes the whole walk opaque this round.
        """
        taken: set[int] = set()
        for member in group.pool:
            named = [self.status3(n) for n in self.tables.dislodges_on.get(member, ())]
            took = _and3(self.remains3(member), _or3(*named))
            if took is None:
                return None
            if took:
                taken.add(member)
        return taken

    def _walk_engagement3(
        self, group: WalkGroup, upto: int | None = None
    ) -> dict[int, int | None] | None:
        """Resolve the rule-6 engagement three-valued, mirroring ``Eval._walk_engagement``.

        The exact strength-first cohort spreading when every input is crisp --
        named claims decided, attacker existence decided, attack strengths
        point-valued, departures decided -- and ``None`` (opaque) otherwise.
        Consumption reads STAGED statuses, so the walk is well-founded in
        strength order: querying ``upto`` needs only strictly stronger
        cohorts' statuses (its own cohort's pairing is status-free), and each
        round decides one more cohort.
        """
        if group in self._engaging:
            return None  # re-entered through a cross-group strength chain
        self._engaging.add(group)
        try:
            taken = self._walk_taken3(group)
            if taken is None:
                return None
            live: list[int] = []
            beaten: list[int] = []
            beaters: dict[int, int] = {}
            for attack in group.attacks:
                exists = self.exists3(attack)
                if exists is None:
                    return None
                if not exists:
                    continue
                decided, beater = self._stack_beater3(attack, group)
                if not decided:
                    return None  # the clash with the stack is open: opaque this round
                if beater is None:
                    live.append(attack)
                else:
                    beaten.append(attack)
                    beaters[attack] = beater
            strengths: dict[int, Fraction] = {}
            for attack in live:
                lo, hi = self.attack_bounds(attack, group.power)
                if lo != hi:
                    return None
                strengths[attack] = lo
            # A cohort's pairing reads no departure; only consumption does, so
            # a query for one attack needs only the stronger cohorts' twins.
            available = [member for member in group.pool if member not in taken]
            engagement: dict[int, int | None] = dict.fromkeys(group.attacks)
            for attack in beaten:
                engagement[attack] = beaters[attack]  # its clash alone (Eval._walk_engagement)
            if upto is not None and upto in beaten:
                return engagement
            consumed: set[int] = set()
            ordered = sorted(live, key=lambda a: strengths[a], reverse=True)
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
                if upto is not None and upto in cohort:
                    return engagement
                for attack, member in zip(cohort, targets, strict=False):
                    remains = self.remains3(member)
                    if remains is None:
                        return None  # consumption undecided: opaque this round
                    if remains is False:
                        consumed.add(member)
                        continue
                    status = self.status3(attack)
                    if status is None:
                        return None  # consumption undecided: opaque this round
                    if status:
                        consumed.add(member)
            return engagement
        finally:
            self._engaging.discard(group)

    def _stack_beater3(self, attack: int, group: WalkGroup) -> tuple[bool, int | None]:
        """Return ``Evaluator._stack_beater`` three-valued, as ``(decided, twin)``.

        Decided once the first twin that beats or ties the attack is known,
        every twin before it in the pool decidedly not doing so.
        """
        for member in group.pool:
            lost = self._lost_clash3(attack, member)
            if lost is None:
                return (False, None)
            if lost:
                return (True, member)
        return (True, None)

    def _walk_assignment3(self, index: int) -> tuple[bool, int | None]:
        """Return the twin one unnamed attack engages, three-valued as ``(decided, member)``.

        An opaque engagement defers to the residue selection.
        """
        group = self.tables.walk_groups[self.tables.walk_group_of[index]]
        engagement = self._walk_engagement3(group, upto=index)
        if engagement is None:
            return (False, None)
        return (True, engagement.get(index))

    def _stack_dislodger3(self, index: int) -> tuple[bool, int | None]:
        """Whether the rule-6 engagement took this stack member, as ``(decided, attack)``.

        A None attack with ``decided`` True means decidedly not walk-taken.
        """
        position = self.tables.stack_group_of.get(index)
        if position is None:
            return (True, None)
        group = self.tables.walk_groups[position]
        engagement = self._walk_engagement3(group)
        if engagement is None:
            return (False, None)
        for attack, member in engagement.items():
            if member != index:
                continue
            status = self.status3(attack)
            remains = self.remains3(index)
            if status is None or remains is None:
                return (False, None)
            if status and remains:
                return (True, attack)
        return (True, None)

    def dislodged3(self, index: int) -> bool | None:
        """Whether the unit is dislodged, three-valued (Sec. 4.2).

        Remaining and (a targeted Dislodge on this unit succeeded, an unnamed
        walk attack took it -- twins rule 6 -- or, implicit regime at capacity
        1, some foreign entrant succeeded).
        """
        remains = self.remains3(index)
        if remains is False:
            return False
        record = self.records[index]
        province = record.unit.location.province
        sources: list[bool | None] = [
            self.status3(dislodger) for dislodger in self.tables.dislodges_on.get(index, ())
        ]
        decided, walker = self._stack_dislodger3(index)
        if not decided:
            sources.append(None)
        elif walker is not None:
            sources.append(True)
        capacity = self.graph.capacity(province)
        if capacity == 1 and self.ruleset.implicit_displacement(capacity):
            sources.extend(
                self.status3(entrant)
                for entrant in self.tables.into.get(province, ())
                if entrant != index
                and self.records[entrant].power != record.power
                and isinstance(self.records[entrant].effective, Move)
            )
        return _and3(remains, _or3(*sources))

    def viable3(self, index: int) -> bool | None:
        """Whether the convoyed move is viable, three-valued (Sec. 3.5, Sec. 6 S1).

        TRUE with a chain of safe fleets, FALSE when no chain survives even
        among possibly-safe fleets, else UNDECIDED.
        """
        move = self.move_of(index)
        seas_certain: set[str] = set()
        seas_possible: set[str] = set()
        for c in self.tables.convoys_for.get(index, ()):
            fate = self.dislodged3(c)
            sea = self.records[c].unit.location.province
            if fate is False:
                seas_certain.add(sea)
            if fate is not True:
                seas_possible.add(sea)
        anchors = self.graph.sea_neighbors(move.location.province)
        goals = self.graph.sea_neighbors(move.destination.province)
        if sea_chain_exists(self.graph, seas_certain, anchors, goals):
            return True
        if not sea_chain_exists(self.graph, seas_possible, anchors, goals):
            return False
        return None

    def exists3(self, index: int) -> bool | None:
        """Whether the relocating order exists as an attacker, three-valued (Sec. 3.1.4).

        Szykmaned never; a mis-targeted Dislodge never (Sec. 3.4, static); an
        unnamed walk attack always (it attacks the stack and cuts even
        victimless -- twins rule 6); overland always; convoyed iff viable.
        """
        if index in self.szykmaned:
            return False
        order = self.records[index].effective
        if isinstance(order, Dislodge):
            if index in self.tables.walk_group_of:
                return True
            return self.tables.victim_of.get(index) is not None
        if not self.move_of(index).via_convoy:
            return True
        return self.viable3(index)

    def nulled3(self, index: int, at_province: str) -> bool | None:
        """Whether the unit's prevent at ``at_province`` is nulled, three-valued (Sec. 4.2).

        Branch-uncorrelated (see the module docstring): dislodged and some
        successful *dislodging* attack from ``at_province`` -- a targeted
        Dislodge naming this unit, or (only under the capacity-1 implicit
        regime, where the unique successful entrant is the displacement
        winner) a plain foreign entrant. Mirrors the crisp
        ``Evaluator.prevent_nulled`` exactly.
        """
        record = self.records[index]
        dislodged = self.dislodged3(index)
        if dislodged is False:
            return False
        origin = record.unit.location.province
        attacker_statuses = [
            self.status3(d)
            for d in self.tables.dislodges_on.get(index, ())
            if self.records[d].unit.location.province == at_province
        ]
        decided, walker = self._stack_dislodger3(index)
        if not decided:
            attacker_statuses.append(None)
        elif walker is not None and self.records[walker].unit.location.province == at_province:
            attacker_statuses.append(True)
        if self.ruleset.implicit_displacement(self.graph.capacity(origin)) and (
            self.graph.capacity(origin) == 1
        ):
            attacker_statuses.extend(
                self.status3(d)
                for d in self.tables.into.get(origin, ())
                if d != index
                and self.records[d].power != record.power
                and self.records[d].unit.location.province == at_province
                and isinstance(self.records[d].effective, Move)
                and not self.is_convoyed(d)
            )
        return _and3(dislodged, _or3(*attacker_statuses))

    def _victim3(self, index: int) -> tuple[bool, int | None]:
        """Return the unit a Dislodge takes, three-valued as ``(decided, victim)``."""
        if index in self.tables.walk_group_of:
            return self._walk_assignment3(index)
        return (True, self.tables.victim_of.get(index))

    def _lost_clash3(self, dislodge: int, victim: int | None) -> bool | None:
        """Return ``Evaluator.lost_clash`` three-valued: the defend comparison may be open."""
        if victim is None or not self.victim_clash(dislodge, victim):
            return False
        return _not3(_gt3(self.defend_bounds(dislodge), self.defend_bounds(victim)))

    # -- Sec. 6 strength intervals ---------------------------------------------------

    def eff_bounds(self, index: int) -> Bounds:
        """Return a support's delivered effectiveness as an interval (Sec. 2, Sec. 3).

        Cut endpoints come from forced vs possible cutter existence (the
        qualifying predicate itself -- targeting, ownership, protection -- is
        static), then the Sec. 3.3 dislodgement void. A support of a Bar skips
        the void, as in ``Eval.delivered_eff``: it is not lost with its giver,
        so its interval is the cut interval and closes as soon as its cutters'
        existence is decided, whatever is still open about the giver's fate.
        Endpoints go through the ruleset's ``cut_aggregation`` hook (forced
        set -> high effectiveness, possible set -> low), interval-sound for
        any monotone aggregation -- both shipped rulesets sum (Sec. 7), so the
        numbers are unchanged; the hook is honored so a future non-sum ruleset
        degrades loudly in the differential rather than silently in these
        bounds.
        """
        record = self.records[index]
        province = record.unit.location.province
        forced: list[Fraction] = []
        possible: list[Fraction] = []
        for attacker in self.tables.into.get(province, ()):
            if self.records[attacker].power == record.power:
                continue
            attacker_order = self.records[attacker].effective
            if isinstance(attacker_order, Dislodge):
                if attacker in self.tables.walk_group_of:
                    if self.tables.cut_target_of.get(attacker) != index:
                        continue
                elif self.tables.victim_of.get(attacker) != index:
                    continue
            elif self.graph.capacity(province) != 1:
                continue  # a plain move into an Area attacks no unit (Sec. 3.1)
            if self._is_protected_attacker(index, attacker):
                continue
            exists = self.exists3(attacker)
            if exists is False:
                continue
            value = self.tables.cutval_of[attacker]
            possible.append(value)
            if exists is True:
                forced.append(value)
        cut_lo = self.ruleset.cut_aggregation(forced)
        cut_hi = self.ruleset.cut_aggregation(possible)
        base = self.tables.support_base_of[index]
        raw_lo = max(_ZERO, base - cut_hi)
        raw_hi = max(_ZERO, base - cut_lo)
        if index in self.tables.bar_supports:
            return (raw_lo, raw_hi)
        void = self.dislodged3(index)
        if void is True:
            return (_ZERO, _ZERO)
        if void is None:
            return (_ZERO, raw_hi)
        return (raw_lo, raw_hi)

    def _sum_bounds(self, base: Fraction, supports: Sequence[int]) -> Bounds:
        """Return ``base`` plus the summed effectiveness intervals of ``supports``."""
        lo = base
        hi = base
        for s in supports:
            s_lo, s_hi = self.eff_bounds(s)
            lo += s_lo
            hi += s_hi
        return (lo, hi)

    def prevent_bounds(self, index: int) -> Bounds:
        """Return the prevent interval of a relocating order (the wave's, for a member)."""
        index = self._champion(index)  # twins rule 2: the wave fights at champion strength
        base = self.tables.move_base_of[index]
        return self._sum_bounds(base, self.tables.supports_of.get(index, ()))

    def defend_bounds(self, index: int) -> Bounds:
        """Return the defend interval: every matching support counts (Sec. 4)."""
        return self.prevent_bounds(index)  # Sec. 4: all matching supports count

    def _admission_rank_bounds(self, index: int) -> Bounds:
        """Return ``Eval.admission_rank`` as an interval.

        A non-champion wave member ranks at its own modified base -- static,
        hence crisp -- while champions and lone movers rank at full prevent
        bounds.
        """
        if self.tables.wave_of.get(index, index) != index:
            base = self.tables.move_base_of[index]
            return (base, base)
        return self.prevent_bounds(index)

    def attack_bounds(self, index: int, defender_power: str | None) -> Bounds:
        """Return the attack interval against a ``defender_power`` occupant (Sec. 3.3 filter)."""
        index = self._champion(index)
        base = self.tables.move_base_of[index]
        supports = [
            s
            for s in self.tables.supports_of.get(index, ())
            if self.records[s].power != defender_power  # Sec. 3.3 role filter
        ]
        return self._sum_bounds(base, supports)

    def hold_bounds(self, index: int) -> Bounds:
        """Return the hold interval, mirroring ``Eval.hold_value``."""
        if self.is_move(index):
            base = self.ruleset.base_strength(self.records[index].unit)
            return (base, base)  # Sec. 4: a failed mover defends with unmodified base
        wall = self.tables.wall_supports_of.get(index)
        supports = wall if wall is not None else self.tables.hold_supports_of.get(index, ())
        # A support-hold never defends its own giver -- mirror of
        # Eval.hold_value.
        return self._sum_bounds(
            self.tables.hold_base_of[index], [s for s in supports if s != index]
        )

    def bar_bounds(self, index: int) -> Bounds:
        """Return the Sec. 4.2 Bar strength interval: unmodified base plus support bounds.

        The supports of a Bar are not lost with their givers, so each bound is
        a cut interval: the Bar's interval is open only while the existence of
        some convoyed cutter is, and never because a backer may yet fall.
        """
        base = self.ruleset.base_strength(self.records[index].unit)
        return self._sum_bounds(base, self.tables.supports_of.get(index, ()))

    # -- Sec. 6 S3: the admission predicate, three-valued ------------------------------

    def _edge_gate(self, index: int) -> tuple[bool | None, int | None]:
        """Return the Sec. 4.1 edge-contest gate as in ``eval_move``, plus the active counter."""
        move = self.move_of(index)
        counter = self.tables.counter_of.get(index)
        active = (
            counter is not None
            and counter not in self.szykmaned
            and self.ruleset.edge_contest(
                move,
                self.move_of(counter),
                self.graph.capacity(move.location.province),
                self.graph.capacity(move.destination.province),
            )
        )
        if not active or counter is None:
            return True, None
        return _gt3(self.defend_bounds(index), self.defend_bounds(counter)), counter

    def _bar_gate3(self, index: int) -> bool | None:
        """Whether the move passes every matching Bar, three-valued (Sec. 4.2 Bar contests).

        The move passes iff every matching Bar is strictly beaten (Bar wins
        ties). A Bar cannot be cut and fires even if its unit is dislodged
        (twins rule 4), and its backers keep backing it when they are
        dislodged, so no undecided dislodgement keeps the Bar's side of the
        comparison open; wave members consult the champion's gates (rule 3).
        """
        terms: list[bool | None] = []
        for bar in self.tables.bars_on.get(self._champion(index), ()):
            terms.append(_gt3(self.prevent_bounds(index), self.bar_bounds(bar)))
        return _and3(*terms)

    def _live_for_ranking3(self, index: int, destination: str) -> bool | None:
        """Whether the order contests slots at ``destination``, three-valued (Sec. 4.3)."""
        exists = self.exists3(index)
        if exists is False:
            return False
        order = self.records[index].effective
        if isinstance(order, Dislodge):
            decided, victim = self._victim3(index)
            if not decided:
                return None
            if victim is None:
                return False  # no twin left: contests nothing
            # A Dislodge that lost or tied its clash takes no part in the
            # admission, whether its victim remains or has left
            # (Evaluator._live_for_ranking).
            in_pool = _and3(_not3(self.remains3(victim)), _not3(self._lost_clash3(index, victim)))
            return _and3(exists, in_pool, _not3(self.nulled3(index, destination)))
        return _and3(exists, _not3(self.nulled3(index, destination)))

    def _area_admission3(self, index: int, destination: str, capacity: int | None) -> bool | None:
        """Whether the candidate is admitted at a multi-slot province, three-valued (Sec. 4.3).

        See the module docstring: a crisp-inputs exact run plus three sound
        short-circuits.
        """
        if capacity is None:
            return True
        occupants = [
            occupant
            for occupant in self.tables.records_at.get(destination, ())
            if occupant != index
        ]
        stay_lo = 0
        stay_hi = 0
        stays_crisp = True
        for occupant in occupants:
            remains = self.remains3(occupant)
            if remains is True:
                stay_lo += 1
                stay_hi += 1
            elif remains is None:
                stay_hi += 1
                stays_crisp = False
        free_lo = capacity - stay_hi
        free_hi = capacity - stay_lo
        if free_hi <= 0:
            return False

        rivals = [c for c in self.tables.into.get(destination, ()) if c != index]
        liveness = {c: self._live_for_ranking3(c, destination) for c in rivals}
        possible = [c for c in rivals if liveness[c] is not False]
        if free_lo >= len(possible) + 1:
            return True  # everyone who could contest fits regardless

        my_rank = self._admission_rank_bounds(index)
        certain_at_or_above = sum(
            1
            for c in possible
            if liveness[c] is True and self._admission_rank_bounds(c)[0] >= my_rank[1]
        )
        if certain_at_or_above >= free_hi:
            return False  # they fill or bounce every slot before/with this move

        if not stays_crisp or any(liveness[c] is None for c in possible):
            return None
        bounds = {c: self._admission_rank_bounds(c) for c in possible}
        if any(lo != hi for lo, hi in bounds.values()) or my_rank[0] != my_rank[1]:
            return None
        free = free_lo  # == capacity - exact stayer count when stays_crisp
        classes: dict[Fraction, list[int]] = {my_rank[0]: [index]}
        for c in possible:
            classes.setdefault(bounds[c][0], []).append(c)
        for rank in sorted(classes, reverse=True):
            members = classes[rank]
            if len(members) > free:
                return False
            if index in members:
                return True
            free -= len(members)
        raise AdjudicationError(  # pragma: no cover - index is always classed
            f"candidate {index} missing from its own admission classes at {destination}"
        )

    def _dislodge3(self, index: int) -> bool | None:
        """Whether a targeted Dislodge succeeds, three-valued (Sec. 4.2).

        Both-branch on the victim's departure (see the module docstring).
        Twin stacks follow rule 6: walk attacks engage their assigned twin
        against the rule-5 wall (no rival standoff); named attacks defer to an
        earlier attacker-order winner on the same twin. A matching Bar gates
        the Dislodge like any entry: beat the victim's defence AND strictly
        beat the bar, or fail wholly with the victim standing.
        """
        bar_ok = self._bar_gate3(index)
        if bar_ok is False:
            return False  # barred: the victim stands, the dislodger stays home
        return _and3(bar_ok, self._dislodge3_contest(index))

    def _dislodge3_contest(self, index: int) -> bool | None:
        """Return the displacement contest of ``_dislodge3`` without the Bar gate."""
        destination = self.entry_target(index)
        if index in self.tables.walk_group_of:
            decided, walk_victim = self._walk_assignment3(index)
            if not decided:
                return None
            if walk_victim is None:
                return False  # "a Dislodge which finds no twin left fails"
            walk_power = self.records[walk_victim].power
            if self.victim_clash(index, walk_victim):
                # Head-to-head (Sec. 4.1): no departure branch to read.
                resistance = self.defend_bounds(walk_victim)
                return _and3(
                    _gt3(self.defend_bounds(index), resistance),
                    _gt3(self.attack_bounds(index, walk_power), resistance),
                )
            remains = self.remains3(walk_victim)

            def walk_contest() -> bool | None:
                return _gt3(self.attack_bounds(index, walk_power), self.hold_bounds(walk_victim))

            if remains is True:
                return walk_contest()
            if remains is False:
                return self._area_admission3(index, destination, self.graph.capacity(destination))
            return _meet3(
                walk_contest(),
                self._area_admission3(index, destination, self.graph.capacity(destination)),
            )
        victim = self.tables.victim_of.get(index)
        assert victim is not None  # exists3 gates mis-targets
        victim_power = self.records[victim].power
        remains = self.remains3(victim)
        clash = self.victim_clash(index, victim)

        def contest() -> bool | None:
            # Rival named attacks on one twin are a lone-victim contest --
            # strictly strongest wins, ties bounce -- so stack members and lone
            # victims share one shape.
            my_attack = self.attack_bounds(index, victim_power)
            terms: list[bool | None] = []
            for rival in self.tables.dislodges_on.get(victim, ()):
                if rival == index or self.exists3(rival) is False:
                    continue
                # A rival beaten in its own clash with the victim stands nobody off.
                terms.append(
                    _or3(
                        self._lost_clash3(rival, victim),
                        _gt3(my_attack, self.prevent_bounds(rival)),
                    )
                )
            if clash:
                resistance = self.defend_bounds(victim)
                terms.append(_gt3(self.defend_bounds(index), resistance))
                terms.append(_gt3(my_attack, resistance))
            else:
                terms.append(_gt3(my_attack, self.hold_bounds(victim)))
            return _and3(*terms)

        if clash:
            return contest()  # head-to-head (Sec. 4.1): no departure branch to read
        if remains is True:
            return contest()
        if remains is False:
            return self._area_admission3(index, destination, self.graph.capacity(destination))
        return _meet3(
            contest(),
            self._area_admission3(index, destination, self.graph.capacity(destination)),
        )

    def _branch_admission(
        self, index: int, occupant: int | None, occupant_stays: bool, edge_counter: int | None
    ) -> bool | None:
        """Return the rival and defender contests of ``eval_move`` under one occupancy branch.

        The capacity-1 implicit-displacement shape (Lemma L3).
        """
        destination = self.move_of(index).destination.province
        defender_power = (
            self.records[occupant].power if occupant is not None and occupant_stays else None
        )
        my_attack = self.attack_bounds(index, defender_power)

        rival_terms: list[bool | None] = []
        for rival in self.tables.into.get(destination, ()):
            if rival == index:
                continue
            exists = self.exists3(rival)
            if exists is False:
                continue
            beats = _gt3(my_attack, self._admission_rank_bounds(rival))
            term = _or3(_not3(exists), self.nulled3(rival, destination), beats)
            rival_terms.append(term)
        rivals_ok = _and3(*rival_terms)
        if rivals_ok is False:
            return False

        stayer_count = 1 if occupant is not None and occupant_stays else 0
        capacity = self.graph.capacity(destination)
        if capacity is None or capacity - stayer_count >= 1:
            return rivals_ok
        assert occupant is not None  # capacity bound => a stayer occupies the slot
        if self.records[occupant].power == self.records[index].power:
            return False  # Sec. 4.2: no self-dislodgement, prevent still stood
        resistance = (
            self.defend_bounds(occupant) if edge_counter == occupant else self.hold_bounds(occupant)
        )
        return _and3(rivals_ok, _gt3(my_attack, resistance))

    def _clash_gate3(self, index: int) -> tuple[bool | None, frozenset[int]]:
        """Return the Dislodge/victim clash gate of ``Eval``, plus the clashing Dislodges.

        A relocating order passes iff it strictly beats every Dislodge that names
        its unit or a member of its wave (Sec. 4.1); an unnamed Dislodge names
        every twin of its stack, so whether it meets the Move is read off the
        orders, never off the engagement (``Eval._clashing_dislodges``).
        """
        mates = self.wave_mates(index)
        terms: list[bool | None] = []
        clashing: set[int] = set()
        for d in self.clash_candidates(index):
            exists = self.exists3(d)
            if exists is False:
                continue
            meets: bool | None
            position = self.tables.walk_group_of.get(d)
            if position is not None:
                pool = self.tables.walk_groups[position].pool
                meets = any(m in pool for m in mates)
            else:
                meets = self.tables.victim_of.get(d) in mates
            meets = _and3(exists, meets)
            if meets is False:
                continue
            if meets is None:
                # Whether the Dislodge meets the Move also decides which strength
                # its unit resists with, so nothing past this gate is known yet.
                terms.append(None)
                continue
            clashing.add(d)
            terms.append(_gt3(self.defend_bounds(index), self.defend_bounds(d)))
        return _and3(*terms), frozenset(clashing)

    def admission3(self, index: int) -> bool | None:
        """Return ``eval_move`` lifted to three values.

        Exists and edge and bar and (occupancy contest per regime). None only
        when some undecided input truly blocks the verdict.
        """
        exists = self.exists3(index)
        if exists is False:
            return False
        order = self.records[index].effective
        if isinstance(order, Dislodge):
            clash_ok, _ = self._clash_gate3(index)
            if clash_ok is False:
                return False
            return _and3(exists, clash_ok, self._dislodge3(index))

        edge_ok, edge_counter = self._edge_gate(index)
        if edge_ok is False:
            return False
        bar_ok = self._bar_gate3(index)
        if bar_ok is False:
            return False
        clash_ok, clashing = self._clash_gate3(index)
        if clash_ok is False:
            return False

        destination = self.move_of(index).destination.province
        capacity = self.graph.capacity(destination)
        if capacity != 1 or not self.ruleset.implicit_displacement(capacity):
            return _and3(
                exists,
                edge_ok,
                bar_ok,
                clash_ok,
                self._area_admission3(index, destination, capacity),
            )

        occupants = [i for i in self.tables.records_at.get(destination, ()) if i != index]
        occupant = occupants[0] if occupants else None

        branch: bool | None
        if occupant is None:
            branch = self._branch_admission(index, None, occupant_stays=False, edge_counter=None)
        elif occupant in clashing:
            # A Dislodge this Move beats head-to-head fails, so its unit stays
            # and resists with the full strength of the Dislodge.
            branch = self._branch_admission(
                index, occupant, occupant_stays=True, edge_counter=occupant
            )
        elif not self.is_move(occupant):
            branch = self._branch_admission(
                index, occupant, occupant_stays=True, edge_counter=edge_counter
            )
        else:
            departure = self.status3(occupant)
            if departure is True:
                branch = self._branch_admission(
                    index, occupant, occupant_stays=False, edge_counter=edge_counter
                )
            elif departure is False:
                branch = self._branch_admission(
                    index, occupant, occupant_stays=True, edge_counter=edge_counter
                )
            else:
                branch = _meet3(
                    self._branch_admission(
                        index, occupant, occupant_stays=False, edge_counter=edge_counter
                    ),
                    self._branch_admission(
                        index, occupant, occupant_stays=True, edge_counter=edge_counter
                    ),
                )
        return _and3(exists, edge_ok, bar_ok, clash_ok, branch)

    # -- Sec. 6 S3 vacancy iteration ----------------------------------------------------

    def _admission_assuming(self, index: int, departed: frozenset[int]) -> bool | None:
        """Return ``admission3`` under the assumption that ``departed`` occupants leave.

        The vacancy pass's optimistic hypothesis.
        """
        previous = self._assumed_departed
        self._assumed_departed = departed
        try:
            return self.admission3(index)
        finally:
            self._assumed_departed = previous

    def _vacancy_seed(self, undecided: Sequence[int]) -> dict[int, frozenset[int]]:
        """Return the seed-eligible moves mapped to the departure set the optimism assumes.

        See the module docstring for the gating.
        """
        seed: dict[int, frozenset[int]] = {}
        for index in undecided:
            if isinstance(self.records[index].effective, Dislodge):
                continue  # targeted displacement never rides vacancy optimism
            if self.exists3(index) is not True:
                continue
            _, edge_counter = self._edge_gate(index)
            if edge_counter is not None:
                continue  # R8: corridor pairs are never optimistic
            clash_ok, clashing = self._clash_gate3(index)
            if clashing or clash_ok is not True:
                continue  # a Move meeting a Dislodge head-to-head is no vacancy chain
            if self._bar_gate3(index) is not True:
                continue
            destination = self.move_of(index).destination.province
            departures = frozenset(
                occupant
                for occupant in self.tables.records_at.get(destination, ())
                if occupant != index and self.is_move(occupant) and self.status3(occupant) is None
            )
            if not departures:
                continue  # departure decided: plain propagation's business
            if self._admission_assuming(index, departures) is True:
                seed[index] = departures
        return seed

    def vacancy_pass(self, undecided: Sequence[int]) -> list[int]:
        """Run the Sec. 6 S3.3 vacancy iteration and return the committed survivors.

        Seed SUCCEEDS-unstable, cascade downgrades, commit the survivors (the
        Sec. 5.2 all-succeed cycle policy applied in-line).
        """
        seed = self._vacancy_seed(undecided)
        surviving = set(seed)
        changed = True
        while changed:
            changed = False
            for index in sorted(surviving):
                departed_ok = all(
                    source in surviving or self.status3(source) is True for source in seed[index]
                )
                if not departed_ok:
                    surviving.discard(index)
                    changed = True
        return sorted(surviving)

    # -- fixpoint driver ---------------------------------------------------------

    def _stationary_undecided(self, index: int) -> bool:
        """Whether a stationary record still carries undecided derived facts.

        The residue-subgraph membership test for non-moves (Sec. 6: the
        sub-SCCs are of D "restricted to undecided orders"; a support with an
        open interval, a fleet with an open fate, or a Bar with an open
        contest is still an undecided order even though its status is
        derived).
        """
        if self.dislodged3(index) is None:
            return True
        order = self.records[index].effective
        if isinstance(order, Support):
            lo, hi = self.eff_bounds(index)
            return lo != hi
        if isinstance(order, Convoy):
            cargo = self.tables.move_record_of_cargo.get(index)
            return (
                cargo is not None
                and cargo not in self.szykmaned  # a szykmaned cargo's viability is moot
                and self.viable3(cargo) is None
            )
        if isinstance(order, Bar):
            barred = self.tables.barred_move_of.get(index)
            return barred is not None and self.status3(barred) is None
        return False

    def _residue_components(self, adjacency: Sequence[Sequence[int]]) -> list[list[int]]:
        """Return the sub-SCCs of the dependency graph restricted to undecided orders.

        In dependencies-first emission order.
        """
        relevant: list[int] = []
        for record in self.records:
            index = record.index
            if self.is_move(index):
                # A decided mover with an open dislodgement fact stays a
                # bridging vertex (symmetric with _stationary_undecided): it
                # is never enumerated
                # (resolve_residue filters on undecided status) but keeps the
                # orders its open fact reads -- prevent-nulling entrants into
                # its origin -- inside the same component, so the enumeration's
                # crisp Eval never reads a cross-component undecided status.
                if (
                    self.status3(index) is None
                    or self.exists3(index) is None
                    or self.dislodged3(index) is None
                ):
                    relevant.append(index)
            elif self._stationary_undecided(index):
                relevant.append(index)
        position = {index: i for i, index in enumerate(relevant)}
        induced: list[list[int]] = [
            [position[j] for j in adjacency[index] if j in position] for index in relevant
        ]
        components = condensation_order(induced)
        return [[relevant[i] for i in component] for component in components]

    def propagate(self, round_guard: int) -> None:
        """Iterate S1-S3 and the vacancy commit until a round decides nothing (Sec. 6).

        Statuses only move None -> True/False (asserted); the guard bounds
        rounds per T2.
        """
        for _ in range(round_guard):
            decisions: dict[int, bool] = {}
            undecided = [m for m in self.tables.moves if self.status3(m) is None]
            if not undecided:
                return
            # S1/S2 facts are recomputed inside admission3 from the round's
            # statuses; S3 admissions all read the same snapshot (synchronous
            # within the stage), then commit together.
            for index in undecided:
                verdict = self.admission3(index)
                if verdict is not None:
                    decisions[index] = verdict
            for index, verdict in decisions.items():
                if self.status[index] is not None:  # pragma: no cover - T2: a round only decides
                    raise AdjudicationError("resolver attempted to re-decide a status")
                self.status[index] = verdict
            still_undecided = [m for m in undecided if m not in decisions]
            committed = self.vacancy_pass(still_undecided) if still_undecided else []
            for index in committed:
                if self.status[index] is not None:  # pragma: no cover - T2: a round only decides
                    raise AdjudicationError("vacancy pass attempted to re-decide a status")
                self.status[index] = True
            if not decisions and not committed:
                return  # a round that decides nothing ends propagation
        raise AdjudicationError(  # pragma: no cover - T2 guard
            "resolver exceeded the T2 round bound without reaching a fixpoint"
        )

    def solve(self, adjacency: Sequence[Sequence[int]]) -> None:
        """Propagate, then settle residual SCCs by the shared Sec. 5.1 selection until decided.

        While residue remains, ``common.resolve_residue`` runs on the first
        residual SCC and propagation resumes (Sec. 6). Each application
        decides every relocating member of its target (T2 amply satisfied), so
        the loop is bounded by twice the relocating-order count.
        """
        round_guard = 4 * len(self.records) + 8
        for _ in range(2 * len(self.tables.moves) + 2):
            self.propagate(round_guard)
            undecided_moves = [m for m in self.tables.moves if self.status3(m) is None]
            if not undecided_moves:
                return
            components = self._residue_components(adjacency)
            target = next(
                component
                for component in components
                if any(self.is_move(i) and self.status3(i) is None for i in component)
            )
            resolve_residue(self, target)
        raise AdjudicationError(  # pragma: no cover - T2: each application decides >= 1
            "residue selection failed to decide every order"
        )


def adjudicate(
    ruleset: Ruleset,
    state: State,
    sheets: Mapping[str, Sequence[Order]],
    *,
    graph: MapGraph | None = None,
) -> PhaseResult:
    """Adjudicate one movement phase by staged fixpoint -- a pure function.

    Observationally identical to the enumerator (docs/adjudication.tex Sec. 5
    and Sec. 6; certified by the differential suite). ``sheets`` maps power ->
    *ordered* order list (Sec. 1); units without orders hold. Deterministic in
    (ruleset, state, sheets), byte for byte.
    """
    if graph is None:
        graph = load_map(state.map_name)
    check_state(graph, state)
    parsed = phase0(graph, ruleset, state, sheets)
    tables = build_tables(graph, ruleset, parsed.records)
    return assemble(graph, state, sheets, parsed, settle(graph, ruleset, parsed.records, tables))


def settle(
    graph: MapGraph,
    ruleset: Ruleset,
    records: Sequence[OrderRecord],
    tables: Tables,
) -> Evaluator:
    """Return the ``Eval`` of the phase under the final labeling :func:`adjudicate` reports.

    Every relocating status is decided and the Szykman marks are set, so the
    returned evaluator answers every derived predicate exactly as the
    assembled result does. It is a pure function of its inputs: a caller that
    rebuilds the Phase-0 records and the tables of an adjudicated phase gets
    the very labeling that phase was assembled from -- which is how the
    Beleriand identity ledger reads the rule-6 engagement of a twin stack
    (``common.final_walk_victims``) instead of replaying it.
    """
    resolver = _Resolver(graph, ruleset, records, tables)
    resolver.solve(residue_adjacency(records, tables))
    return resolver
