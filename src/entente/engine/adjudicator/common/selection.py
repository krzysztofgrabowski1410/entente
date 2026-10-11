"""Residue selection and the backup policies: what happens to orders left undecided.

When propagation or forced facts leave orders undecided,
:func:`resolve_residue` enumerates the residue's stable labelings exactly,
commits a unique one or the statuses every one of them shares, and hands only
*genuine* ambiguity to the ruleset's backup policy -- circular movement
(:func:`commit_cycle_policy`) or Szykman. Both engines share the selection
itself, not just the policy, which is why they settle residues at the
identical point (docs/adjudication.tex Sec. 5.1 and Sec. 5.2).

This module changes only with the Sec. 5.2 policies and the residue-selection
rule -- distinct from ``Eval`` (which it only consults) and from result
assembly (which runs after it).
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from entente.engine.adjudicator.common.geometry import sea_chain_exists
from entente.engine.adjudicator.common.results import AdjudicationError
from entente.engine.orders import Move
from entente.engine.rules.protocol import BackupAction, ResidueClass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.adjudicator.common.evaluator import Evaluator


def commit_cycle_policy(evaluator: Evaluator, members: Sequence[int]) -> None:
    """Apply the ALL_MOVES_SUCCEED backup policy to a residue (Sec. 5.2).

    Shared verbatim by both engines: seed every undecided relocating member
    SUCCEEDS, then synchronously downgrade every member ``eval_move`` refutes
    under the tentative assignment, until fixed.

    A clean rotation keeps all members (the classic circular-movement rule).
    Anything else the conservative search left in the component -- tied
    rivals, over-capacity entries, and Beleriand's residues with *zero*
    stable labelings and no convoy in them (a Dislodge whose victim is
    leaving for the dislodger's own province out of a full Area, while a
    stronger entrant waits for the slot: the Dislodge succeeds iff the victim
    stays, and the victim stays iff the Dislodge fails) -- settles at the
    greatest eval-consistent point: every committed SUCCEEDS passes
    ``eval_move`` under the final assignment, so the result is admission- and
    capacity-consistent even where no stable labeling exists. Downgrades are
    synchronous over canonically sorted members: deterministic, monotone,
    terminating.

    ``members`` must be the Sec. 6 residue -- orders still undecided after
    propagation (resolver) or not unanimous across the stable labelings
    (enumerator), and, where there is no stable labeling, not forced
    (:func:`_forced`). Seeding an order that fails in every stable labeling
    poisons the cascade: read as alive, it can inflate a Sec. 4.3 class over
    the slot bound and drag genuine cycle members down in the same synchronous
    round (the ANF position, BATC.B.11).
    """
    undecided = [i for i in sorted(members) if evaluator.is_move(i) and evaluator.status[i] is None]
    for index in undecided:
        evaluator.status[index] = True
    changed = True
    while changed:
        changed = False
        refuted = [
            index
            for index in undecided
            if evaluator.status[index] is True and not evaluator.eval_move(index)
        ]
        for index in refuted:
            evaluator.status[index] = False
            changed = True


MAX_ENUMERATED = 20
"""Cap on one residue enumeration (2^20 evaluations): the knob, read by :func:`resolve_residue`.

Rebinding the copy that ``common/__init__.py`` re-exports changes nothing; set
``selection.MAX_ENUMERATED``. Raising the number is no fix -- an enumeration
costs 2^n evaluations (about 9 s at 20 orders, twice that with every further
order).

The resolver's fixpoint residues are far smaller (Sec. 6), but the
*enumerator's* residue is a whole dependency-SCC minus its forced facts, and a
legal position can rotate 21+ units in one cycle. Above the cap the selection
commits without enumerating in two cases and raises loudly in every other:
the all-succeed completion when it is stable (:func:`_all_succeed_is_stable`),
and a pure vacancy rotation (:func:`_pure_vacancy_rotation`).
"""


def _all_succeed_is_stable(
    evaluator: Evaluator, undecided: Sequence[int], verify: Sequence[int]
) -> bool:
    """Commit the all-succeed completion of an over-budget residue iff it is stable.

    Set every undecided order SUCCEEDS and check ``eval_move`` on every member
    of ``verify``; keep the assignment if all of them hold, otherwise put the
    orders back to undecided.

    This is exactly what the enumeration would select, provided no convoy is
    tracked (the caller's condition). The bound can only be exceeded on the
    loop's first pass, since ``undecided`` only shrinks, so there
    ``undecided == verify``. Let A be the all-succeed completion and suppose it
    is stable. (i) If it is the only stable completion, the selection takes it.
    (ii) If several are stable, every verdict they all share is A's, so the
    unanimity step commits only SUCCEEDS, and A stays stable on the smaller
    domain (stability is checked on all of ``verify``); this repeats until one
    stable completion is left -- A -- or several share nothing; then, with no
    tracked convoy, the residue is PURE_CYCLE, whose policy is
    ALL_MOVES_SUCCEED in both rulesets, and ``commit_cycle_policy`` seeds every
    order SUCCEEDS and downgrades only the refuted ones -- none, as A is
    stable. (iii) No stable completion is impossible, A being one. With a
    tracked convoy the unanimity step skips orders whose viability varies and
    the policy may be Szykman, so the caller does not take this shortcut.
    """
    for index in undecided:
        evaluator.status[index] = True
    if all(evaluator.eval_move(index) for index in verify):
        return True
    for index in undecided:
        evaluator.status[index] = None
    return False


def _pure_vacancy_rotation(evaluator: Evaluator, undecided: Sequence[int]) -> bool:
    """Whether an over-budget residue is a disjoint union of plain capacity-1 vacancy cycles.

    That is the one shape safely committable without enumeration. Conditions:
    every member is a non-convoyed plain Move between capacity-1 provinces, and
    the destination multiset equals the origin multiset. At capacity 1 origins
    are distinct, so the members decompose into disjoint simple cycles over
    provinces touched by no other member. Then: no convoy viability exists to
    vary, so Sec. 5.2 classifies PURE_CYCLE by structure alone; and each
    cycle's stable labelings are exactly {all-succeed, all-fail} (2-cycles:
    {all-fail} -- the Sec. 4.1 edge contest), pairwise province-disjoint, so
    ``commit_cycle_policy``'s greatest eval-consistent point *is* the Sec. 5.2
    per-cycle selection -- no unanimity extraction is skipped that could
    matter (contrast the ANF position, BATC.B.11: entangled admissions need
    the exact enumeration; they fail these conditions).
    """
    origins: list[str] = []
    destinations: list[str] = []
    for index in undecided:
        order = evaluator.records[index].effective
        if not isinstance(order, Move) or order.via_convoy:
            return False
        origin = order.location.province
        destination = order.destination.province
        if evaluator.graph.capacity(origin) != 1 or evaluator.graph.capacity(destination) != 1:
            return False
        origins.append(origin)
        destinations.append(destination)
    return sorted(origins) == sorted(destinations)


def _forced(evaluator: Evaluator, undecided: Sequence[int]) -> list[tuple[int, bool]]:
    """Return the members whose ``eval_move`` verdict is the same under every assignment.

    The assignments range over ``undecided``, everything else as decided.
    Such a verdict is one the rules settle on their own; where a residue has
    no stable labeling it is decided before a backup policy runs, so that the
    policy never reads it at a seeded value it cannot have (Sec. 5.1).
    """
    verdicts: dict[int, set[bool]] = {index: set() for index in undecided}
    for assignment in product((False, True), repeat=len(undecided)):
        for index, value in zip(undecided, assignment, strict=True):
            evaluator.status[index] = value
        for index in undecided:
            if len(verdicts[index]) < 2:
                verdicts[index].add(evaluator.eval_move(index))
    for index in undecided:
        evaluator.status[index] = None
    return [(index, next(iter(found))) for index, found in verdicts.items() if len(found) == 1]


def resolve_residue(evaluator: Evaluator, members: Sequence[int]) -> None:
    """Settle a residual order set by the Sec. 5.1 selection, shared verbatim by both engines.

    Enumerate the {SUCCEEDS, FAILS} assignments over the still-undecided
    relocating members and keep the stable ones (L = Eval(L)). Exactly one ->
    take it. Several -> commit every status all stable labelings share
    (Sec. 5.1 selects among what *remains*; unanimity is T3a computed exactly;
    a convoyed member is settled only when its *viability* is also constant
    across them, since its cut delivery stays Szykman's business whatever its
    own status) and re-enumerate the disagreement set. Zero -> commit every
    member whose ``eval_move`` verdict is the same under every assignment and
    re-enumerate, until none is (:func:`_forced`). Zero with nothing forced,
    or several sharing nothing -> classify per Sec. 5.2 (convoy-entangled iff some
    tracked member's viability varies, else pure cycle) and apply the
    ruleset's backup policy -- which therefore only ever sees genuine
    ambiguity, in both engines, by construction.

    ``members`` is the caller's residue estimate: the enumerator (``csp``)
    passes a component's undecided orders after its forced-facts pass; the
    resolver (``resolver``) its topologically-first residual sub-SCC after
    propagation. Both estimates may *overapproximate* the genuine Sec. 6
    residue -- a fixpoint can stall on a component with a unique stable
    labeling (conservative both-branch admission cannot case-split on a
    Dislodge victim's departure; the position W with its Dislodge also
    supported by a 2A, tests/engine/test_residue_selection.py) -- and
    committing the policy on such a set unravels uniquely-determined orders:
    the exact enumeration here is what closes that gap, and, where no stable
    labeling exists, the forced verdicts committed before the policy.

    Stability is verified over every member this call has settled, not just
    the current enumeration domain: an assignment stable on the shrunken
    domain may contradict a unanimity-settled member, and counting it would
    pollute the unanimity and Sec. 5.2 viability sets. Viability is tracked for
    every convoyed member of ``members``, including moves the resolver's
    propagation already failed for other reasons -- their cut delivery is
    still the residue's business while their viability varies (Sec. 5.2).
    Stationary members are ignored: their statuses are Sec. 5.1-derived.
    """
    ruleset = evaluator.ruleset
    undecided = [i for i in sorted(members) if evaluator.is_move(i) and evaluator.status[i] is None]
    tracked = [
        i
        for i in sorted(members)
        if evaluator.is_move(i) and evaluator.is_convoyed(i) and i not in evaluator.szykmaned
    ]
    verify = list(undecided)

    while undecided:
        if len(undecided) > MAX_ENUMERATED:
            if not tracked and _all_succeed_is_stable(evaluator, undecided, verify):
                return
            if not tracked and _pure_vacancy_rotation(evaluator, undecided):
                commit_cycle_policy(evaluator, undecided)
                return
            raise AdjudicationError(
                f"residue of {len(undecided)} undecided orders exceeds the enumeration bound"
            )
        stable: list[tuple[bool, ...]] = []
        viability_all: dict[int, set[bool]] = {m: set() for m in tracked}
        viability_stable: dict[int, set[bool]] = {m: set() for m in tracked}
        for assignment in product((False, True), repeat=len(undecided)):
            for index, value in zip(undecided, assignment, strict=True):
                evaluator.status[index] = value
            views = {m: evaluator.viable(m) for m in tracked}
            for m, v in views.items():
                viability_all[m].add(v)
            if all(evaluator.eval_move(index) == evaluator.status[index] for index in verify):
                stable.append(assignment)
                for m, v in views.items():
                    viability_stable[m].add(v)
        if len(stable) == 1:
            for index, value in zip(undecided, stable[0], strict=True):
                evaluator.status[index] = value
            return
        for index in undecided:
            evaluator.status[index] = None

        if not stable:
            forced = _forced(evaluator, undecided)
            if forced:
                for index, value in forced:
                    evaluator.status[index] = value
                decided = {index for index, _ in forced}
                undecided = [index for index in undecided if index not in decided]
                continue

        if len(stable) >= 2:
            settled: list[tuple[int, bool]] = []
            for position, index in enumerate(undecided):
                values = {labeling[position] for labeling in stable}
                if len(values) != 1:
                    continue
                if index in viability_stable and len(viability_stable[index]) != 1:
                    continue
                settled.append((index, values.pop()))
            if settled:
                for index, value in settled:
                    evaluator.status[index] = value
                decided = {index for index, _ in settled}
                undecided = [index for index in undecided if index not in decided]
                continue

        varying = viability_stable if len(stable) >= 2 else viability_all
        undecided_viability = sorted(m for m in tracked if len(varying[m]) == 2)
        residue = ResidueClass.CONVOY_ENTANGLED if undecided_viability else ResidueClass.PURE_CYCLE
        action = ruleset.backup_policy(residue)
        if action is BackupAction.SZYKMAN:
            if not undecided_viability:  # pragma: no cover - policy/classification mismatch
                raise AdjudicationError("Szykman policy with no undecided convoy viability")
            for index in undecided_viability:
                evaluator.szykmaned.add(index)
                if evaluator.status[index] is None:
                    evaluator.status[index] = False
            undecided = [m for m in undecided if m not in evaluator.szykmaned]
            tracked = [m for m in tracked if m not in evaluator.szykmaned]
            continue
        # ALL_MOVES_SUCCEED (Sec. 5.2): the shared seed-and-downgrade commit,
        # reached only with 0 stable labelings and nothing forced (in Beleriand
        # a residue can have none without any convoy in it) or with several
        # that disagree on every remaining member -- ``undecided`` is by now
        # exactly the Sec. 6 residue.
        commit_cycle_policy(evaluator, undecided)
        return


def possible_chain_for(evaluator: Evaluator, index: int) -> bool:
    """Whether any chain could carry this move even with every fleet alive.

    Constant over all labelings, so its negation is a forced FAILS.
    """
    move = evaluator.move_of(index)
    seas = {
        evaluator.records[c].unit.location.province
        for c in evaluator.tables.convoys_for.get(index, ())
    }
    if not seas:
        return False
    return sea_chain_exists(
        evaluator.graph,
        seas,
        evaluator.graph.sea_neighbors(move.location.province),
        evaluator.graph.sea_neighbors(move.destination.province),
    )
