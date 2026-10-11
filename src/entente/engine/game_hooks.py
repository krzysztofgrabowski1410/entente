"""The game-loop variant seam.

``GameHooks`` is the extension surface for **engine-level variant state and
events** (Beleriand Layer 2: knowledge facts, masks, timers, timed events,
per-power victory): everything that stays *out* of the adjudicator. The
:class:`~entente.engine.game.Game` facade calls these hooks at fixed points of
the phase machine and never branches on a ruleset name -- variants register a
factory in :data:`GAME_HOOKS`, and this base class *is* the standard game
(every hook a documented no-op), keeping the rule of two: standard = this
class, Beleriand = ``entente.variants.beleriand.layer2``.

The frozen ``Ruleset`` protocol is untouched: dynamic, game-state-dependent
rule data reaches the adjudicator only through :meth:`GameHooks.wrap_ruleset`
-- a per-game delegating wrapper that *implements* the same protocol.

Hooks own the game's only randomness (``seed`` -> one ``random.Random``
stream; the adjudicator stays RNG-free) and the variant's mutable state; one
hooks instance serves exactly one game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from entente.engine.adjudicator.common import OrderResolution, Resolution
from entente.engine.orders import ADJUDICATION_ORDER_TYPES, format_order
from entente.engine.phase import Phase, PhaseKind, Season

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from entente.engine.adjudicator.builds import AdjustmentResult
    from entente.engine.adjudicator.common import Dislodgement, PhaseResult
    from entente.engine.adjudicator.retreats import RetreatResult
    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import AnyOrder, GameOrder, Order
    from entente.engine.rules.protocol import Ruleset
    from entente.engine.state import State, Unit


@dataclass(frozen=True, slots=True)
class GrantRecord:
    """One knowledge transfer, recorded on the phase it precedes.

    Serialization replays it at the same point.
    """

    giver: str
    receiver: str
    fact: str


@dataclass(frozen=True, slots=True)
class SpecialUnit:
    """One of the variant's special units, as the orders name it.

    ``special`` is the variant's name for it and ``designator`` the letter its
    orders write; ``unit`` is where it stands, or while ``limbo`` the position
    it was dislodged from and has yet to retreat from.
    """

    special: str
    designator: str
    unit: Unit
    limbo: bool = False


@dataclass(frozen=True, slots=True)
class OathState:
    """Morgoth's standing Oath of the Second and its price.

    The rulebook's "Morgoth" section, "Victory": *"At any time Morgoth may
    swear a written Oath of the Second, naming one Power. An Oath is sworn or
    broken during a Phase, before Morgoth has placed his Orders for that
    Phase, and takes effect at once."* -- "at any time" is the Oath's freedom
    from any schedule of its own, and the sentence after it is the one gate it
    does carry, enforced by the variant through
    :meth:`GameHooks.validate_swear`'s ``pending``.

    ``power`` is the Power the standing Oath names (None when none stands) and
    ``year`` the Year it was sworn -- the rulebook records both, and both are
    public. ``broken`` counts the Oaths broken so far, each raising the victory
    threshold by one *for the rest of the game*; ``stabbed`` is the current
    Oath's free-break exemption, set once the sworn Power writes an attack on a
    Morgoth unit and cleared when a new Oath is sworn.

    The default instance is "no Oath was ever involved" -- what a classic game
    and every untouched variant game report, which is what keeps the whole
    surface inert (and every save byte-identical) until someone swears.
    """

    power: str | None = None
    year: int | None = None
    broken: int = 0
    stabbed: bool = False

    @property
    def is_default(self) -> bool:
        """Whether this is the never-touched state (nothing to serialize)."""
        return self == OathState()


@dataclass(frozen=True, slots=True)
class GameResult:
    """A variant's game-end verdict.

    ``winners`` may hold several powers (ties are joint); ``scores`` is the
    variant's scoring detail (Beleriand: net center gain over the original
    home endowment).

    ``second`` is the power a Morgoth win optionally places second (rulebook
    "If Morgoth wins, he may choose one Player to get second place. The rest
    of the Players are unplaced."): ``None`` until designated, and never set
    unless ``winners == (MORGOTH,)``. It records the presentation contract --
    under a Morgoth win the placements are Morgoth > ``second`` > everyone
    else unplaced -- while ``scores`` stays the raw center data. It defaults
    to ``None`` so a standard game and an undesignated or non-Morgoth
    Beleriand result carry no second at all.
    """

    winners: tuple[str, ...]
    reason: str
    scores: Mapping[str, int]
    second: str | None = None


@dataclass(slots=True)
class MovementPlan:
    """What :meth:`GameHooks.preprocess_movement` hands back to the loop.

    ``adjudicated`` is what the adjudicator sees (adjudication-order sheets
    only). ``deferred`` holds the extracted Layer-2 orders -- resolved by
    :meth:`postprocess_movement` -- and ``invalid`` the orders ruled out
    before adjudication (each reports FAILS, its unit stands unordered --
    mirroring the Phase-0 invalid-order treatment, rulebook 4.E.1); both are
    keyed by (power, sheet index) so the resolution merge realigns every
    written order with its outcome.

    ``reasons`` optionally annotates any of those keys with the human sentence
    naming the rule that voided the order. It is presentation only --
    :meth:`merge_resolutions` copies it onto the row and nothing else reads
    it -- so a variant may leave it empty and lose nothing but the wording.
    """

    adjudicated: dict[str, list[Order]]
    deferred: dict[tuple[str, int], GameOrder] = field(default_factory=dict)
    invalid: dict[tuple[str, int], GameOrder] = field(default_factory=dict)
    reasons: dict[tuple[str, int], str] = field(default_factory=dict)


class GameHooks:
    """The no-op instantiation: a classic game with no variant layer.

    Subclasses override selectively; every method documents its contract.
    ``seed`` is stored (and serialized) but unused here -- the standard game
    consumes no randomness.
    """

    def __init__(self, graph: MapGraph, seed: int | None = None) -> None:
        self.graph = graph
        self.seed = seed

    # -- construction ---------------------------------------------------------

    def initial_phase(self, start_year: int | None) -> Phase:
        """Return the phase a new game opens at.

        Standard: Spring 1901, years ascending. Variants may fix their own
        calendar (Beleriand: 25, descending -- rulebook "Valar's
        intervention").
        """
        year = start_year if start_year is not None else 1901
        return Phase(year, Season.SPRING, PhaseKind.MOVEMENT)

    def wrap_ruleset(self, base: Ruleset) -> Ruleset:
        """Return the ruleset the game adjudicates with.

        Variants return a per-game delegating wrapper whose data-driven hooks
        read game state; the base game adjudicates with the ruleset as
        registered.
        """
        return base

    def register_start(self, state: State, ownership: Mapping[str, str]) -> None:
        """Receive the game's opening position once, after any construction-time overrides.

        Variants seed their ledgers here.
        """

    # -- knowledge grants -------------------------------------------------------

    def validate_grant(self, giver: str, receiver: str, fact: str) -> str | None:
        """Return why the grant is illegal, or None if it may proceed.

        The base game has no transferable facts.
        """
        del giver, receiver
        return f"unknown fact {fact!r}"

    def apply_grant(self, giver: str, receiver: str, fact: str) -> None:
        """Record a validated grant in variant state.

        Unreachable in the base game (validation refuses everything).
        """
        raise NotImplementedError("the base game has no transferable facts")

    # -- movement phase ---------------------------------------------------------

    def on_phase_start(self, phase: Phase, state: State) -> State:
        """Apply the state effects the rulebook orders "before the Orders are placed".

        Ulmo annihilations and the Teleri disorder pick. Runs when a movement
        phase becomes current, before order intake. Returns the (possibly
        unchanged) state; events buffer in the hooks and flush into the next
        phase record.
        """
        del phase
        return state

    def preprocess_movement(
        self, phase: Phase, state: State, sheets: Mapping[str, Sequence[GameOrder]]
    ) -> MovementPlan:
        """Split submitted movement sheets into the adjudicated part and the variant part.

        The base game defers nothing and invalidates every Layer-2 order
        (out-of-registry semantics: FAILS, unit holds).
        """
        del phase, state
        plan = MovementPlan(adjudicated={})
        for power, sheet in sheets.items():
            kept: list[Order] = []
            for index, order in enumerate(sheet):
                if isinstance(order, ADJUDICATION_ORDER_TYPES):
                    kept.append(order)
                else:
                    plan.invalid[(power, index)] = order
            plan.adjudicated[power] = kept
        return plan

    def postprocess_movement(
        self,
        phase: Phase,
        pre_state: State,
        plan: MovementPlan,
        result: PhaseResult,
    ) -> tuple[State, Mapping[tuple[str, int], Resolution], tuple[Dislodgement, ...]]:
        """Apply the end-of-phase variant steps and update the variant ledger.

        The steps are transforms, devastation, masks and sailing. Returns the
        possibly amended post-movement state, the outcomes of the movement
        plan's deferred orders (keyed like the movement plan; absent deferred
        or invalid entries report FAILS), and any extra Dislodgement records a variant
        step produces beyond the adjudicator's own (a variant's Devastate can
        expel co-located units the calculus never touched) -- ``Game`` merges
        these into the movement result the retreat phase adjudicates from, so
        ``pending_dislodgements``, ``retreat_has_decisions``/
        ``retreat_has_effects`` and ``legal_retreats`` pick them up for free.
        The base game amends nothing and dislodges nothing extra.
        """
        del phase, pre_state, plan
        return result.new_state, {}, ()

    def preprocess_retreat(
        self,
        phase: Phase,
        sheets: Mapping[str, Sequence[AnyOrder]],
        movement: PhaseResult,
    ) -> tuple[Mapping[str, Sequence[AnyOrder]], list[tuple[str, AnyOrder]]]:
        """Gate retreat sheets for the variant (credential claims).

        ``movement`` is the movement result the retreats are adjudicated
        from, so a variant can ask which unit each line binds
        (``retreats.bind_retreat_orders``). Returns (sheets to adjudicate,
        invalidated orders).
        """
        del phase, movement
        return sheets, []

    def post_retreat(self, phase: Phase, result: RetreatResult) -> None:
        """Ledger flow for retreat outcomes (arrivals arm variant timers)."""
        del phase, result

    # -- adjustment phase ---------------------------------------------------------

    def adjudicate_adjustments(
        self,
        phase: Phase,
        state: State,
        ownership: Mapping[str, str],
        sheets: Mapping[str, Sequence[AnyOrder]],
    ) -> AdjustmentResult | None:
        """Run the variant's winter adjudication, or return None for the standard routine."""
        del phase, state, ownership, sheets
        return None

    def post_adjustment(self, phase: Phase, result: AdjustmentResult) -> None:
        """Flow the ledger through builds and removals."""
        del phase, result

    def adjustment_has_decisions(self, state: State, ownership: Mapping[str, str]) -> bool | None:
        """Whether some power faces a winter choice under variant rules, or None for standard."""
        del state, ownership
        return None

    def adjustment_has_effects(self, state: State, ownership: Mapping[str, str]) -> bool | None:
        """Whether an empty-sheet winter changes anything in the variant, or None for standard."""
        del state, ownership
        return None

    def legal_adjustment_orders(
        self, power: str, state: State, ownership: Mapping[str, str]
    ) -> tuple[AnyOrder, ...] | None:
        """Enumerate the variant's adjustment orders, or return None for standard."""
        del power, state, ownership
        return None

    def legal_extra_movement_orders(self, power: str, state: State) -> tuple[GameOrder, ...]:
        """Return the Layer-2 movement orders ``power`` could submit now.

        Enumerated at current positions; transfer directives reference
        end-of-phase spots, so enumeration offers the stationary cases.
        """
        del power, state
        return ()

    def special_designators(self, unit: Unit, *, limbo: bool = False) -> tuple[str, ...]:
        """Return the designator letters of the special units that ``unit``'s value stands for.

        A board unit is a value, and one value may stand for several units at
        once -- in Beleriand ``MORGOTH 3A ANF`` may be the Morgoth unit, or
        Glaurung, or both -- and only the variant's ledger knows which. The
        letters come back in a fixed order, one per special of that value on
        the board, or with ``limbo`` one per special dislodged from that
        position and awaiting its retreat, so an enumeration gives the k-th
        occurrence of the value the k-th letter and writes every special with
        its own designator (the retreat enumeration prefers the designator a
        dislodgement record carries, which ties each special to its own
        dislodger). The standard game has no specials and answers the empty
        tuple.
        """
        del unit, limbo
        return ()

    def special_units(self) -> tuple[SpecialUnit, ...]:
        """Return every special unit now in the game, on the board or awaiting its retreat.

        The phase record keeps this list as the phase left it: a board holds
        only values, so a past phase's board cannot say by itself which of its
        units, or which unit it dislodged or disbanded, was a special. The
        standard game has none.
        """
        return ()

    def build_quota(self, power: str, state: State, ownership: Mapping[str, str]) -> int | None:
        """Return the variant's build quota, or None for standard.

        Beleriand: special-unit exemptions and Eriador's counter.
        """
        del power, state, ownership
        return None

    # -- boundaries and victory ---------------------------------------------------

    def on_season_end(
        self, phase: Phase, state: State, ownership: Mapping[str, str]
    ) -> tuple[State, Mapping[str, str], bool]:
        """Run the boundary work when a season's retreat stage settles.

        Fires even when that phase was skipped: variant bookkeeping, timed
        events, and victory checks. Returns (state, ownership, game_over). The
        base game changes nothing and never ends the game here (the loop's
        standard majority check runs separately).
        """
        del phase
        return state, ownership, False

    @property
    def decides_victory(self) -> bool:
        """Whether this variant owns the victory decision; False keeps the strict-majority rule."""
        return False

    def game_result(self) -> GameResult | None:
        """Return the variant verdict once decided.

        None while the game runs, and always None when
        :attr:`decides_victory` is False.
        """
        return None

    def validate_second(self, power: str) -> str | None:
        """Return why ``power`` cannot be placed second, or None if it may.

        Only a variant with a Morgoth-style win offers second place (the
        rulebook's "Morgoth" section); the base game has none. The
        :class:`~entente.engine.game.Game` facade checks the game-is-over
        precondition before delegating here (the ``grant`` precedent,
        inverted).
        """
        del power
        return "this game has no second-place designation"

    def designate_second(self, power: str) -> None:
        """Record a validated second-place designation.

        Unreachable in the base game (validation refuses everything).
        """
        raise NotImplementedError("the base game has no second-place designation")

    # -- the Oath of the Second -----------------------------------------------------

    def validate_swear(self, power: str, pending: Collection[str]) -> str | None:
        """Return why an Oath naming ``power`` cannot be sworn now, or None if it may.

        ``pending`` is every power holding a sheet for the current phase; a
        variant whose Oath is gated on its swearer having not yet ordered reads
        itself out of it (the ``build_quota`` pattern -- the facade hands over
        the game context it owns and the variant decides what it means, so no
        power name ever enters the engine). The facade has already checked the
        game is running and the power exists. Only a variant with a
        Morgoth-style placement has an Oath; the base game has none.
        """
        del power, pending
        return "this game has no Oath of the Second"

    def apply_swear(self, power: str, year: int) -> None:
        """Record a validated Oath sworn in ``year``.

        Unreachable in the base game (validation refuses everything).
        """
        raise NotImplementedError("the base game has no Oath of the Second")

    def validate_break(self, pending: Collection[str]) -> str | None:
        """Return why the standing Oath cannot be broken now, or None if it may.

        ``pending`` as in :meth:`validate_swear`.
        """
        del pending
        return "this game has no Oath of the Second"

    def apply_break(self, year: int) -> None:
        """Break the standing Oath in ``year``, charging its price.

        Unreachable in the base game (validation refuses everything).
        """
        raise NotImplementedError("the base game has no Oath of the Second")

    def oath_state(self) -> OathState:
        """Return the standing Oath, its Year, the broken count and the free-break flag.

        The base game never has one, so the default rides through every
        reader (``Game.oath`` &c., serialization) unchanged.
        """
        return OathState()

    def validate_oath_state(self, state: OathState) -> str | None:
        """Return why ``state`` cannot be restored into this game, or None if it may.

        The loader-only replay input -- see :meth:`Game.restore_oath_state`.
        The base game accepts only the default: a classic save carrying any
        Oath at all is a corrupt document, not a silently ignored field.
        """
        return None if state.is_default else "this game has no Oath of the Second"

    def apply_oath_state(self, state: OathState) -> None:
        """Install a validated ``state``.

        Unreachable in the base game for anything but the default, which is
        already what it holds.
        """
        if not state.is_default:  # pragma: no cover - validation refuses first
            raise NotImplementedError("the base game has no Oath of the Second")

    # -- record plumbing -----------------------------------------------------------

    def drain_events(self) -> tuple[str, ...]:
        """Return the buffered variant event lines since the last drain.

        They attach to the next phase record (and thus the serialized
        history).
        """
        return ()

    def snapshot(self) -> dict[str, object] | None:
        """Return the serialized variant-state fragment, verified on replay.

        None when the game has no variant layer.
        """
        return None

    # -- shared helper -------------------------------------------------------------

    def merge_resolutions(
        self,
        sheets: Mapping[str, Sequence[GameOrder]],
        plan: MovementPlan,
        adjudicated: tuple[OrderResolution, ...],
        outcomes: Mapping[tuple[str, int], Resolution],
    ) -> tuple[OrderResolution, ...]:
        """Reassemble per-written-order resolutions over the *full* submitted sheets.

        Rows the adjudicator produced are consumed in its own order (it
        reports every written order it received per power, in sheet order);
        orders it never saw -- the movement plan's deferred and invalid entries
        -- take their row from ``outcomes``, keyed by (power, sheet index), absent
        meaning FAILS. Those rows also pick up ``plan.reasons`` where the
        variant filed one -- the only place a reason enters a resolution, so
        an unannotated key yields a row with no reason. The result preserves
        the sheet-order-per-sorted-power shape the adjudicator itself emits.
        """
        merged: list[OrderResolution] = []
        by_power: dict[str, list[OrderResolution]] = {}
        for row in adjudicated:
            by_power.setdefault(row.power, []).append(row)
        for power in sorted(sheets):
            queue = by_power.get(power, [])
            consumed = 0
            for index, order in enumerate(sheets[power]):
                if (power, index) in plan.deferred or (power, index) in plan.invalid:
                    resolution = outcomes.get((power, index), Resolution.FAILS)
                    merged.append(
                        OrderResolution(
                            power,
                            format_order(order),
                            resolution,
                            plan.reasons.get((power, index)),
                        )
                    )
                else:
                    merged.append(queue[consumed])
                    consumed += 1
            merged.extend(queue[consumed:])  # adjudicator rows beyond the sheet (defensive)
        return tuple(merged)


GAME_HOOKS: dict[str, Callable[[MapGraph, int | None], GameHooks]] = {}
"""Ruleset name -> hooks factory.

Absent names get :class:`GameHooks` itself (the standard no-op layer).
Variants register at import time, exactly like ``RULESETS`` (the engine never
imports variants).
"""


def hooks_for(name: str, graph: MapGraph, seed: int | None) -> GameHooks:
    """Return the registered hooks for ``name`` (or the no-op base)."""
    factory = GAME_HOOKS.get(name, GameHooks)
    return factory(graph, seed)


__all__ = [
    "GAME_HOOKS",
    "GameHooks",
    "GameResult",
    "GrantRecord",
    "MovementPlan",
    "OathState",
    "hooks_for",
]
