"""The Game facade: the phase machine over the pure adjudicators (docs/adjudication.tex Sec. 9).

``Game`` drives the phase machine over the pure adjudicators: movement
(:mod:`entente.engine.adjudicator`), retreats and adjustments (Sec. 9). It owns
everything the pure functions do not: the phase pointer, supply-center
ownership, pending order submissions, victory, and history.

Contract highlights:

- ``submit(power, orders)`` takes an *ordered* list (sequence carries meaning:
  duplicate orders to a unit, build/disband priority) and **replaces** any
  earlier submission by that power -- DATC 4.D.1 choice C, "only the latest
  order set counts". Orders must be of the current phase's kinds; anything
  else is an interface error and raises.
- Powers that submit nothing get the phase defaults: units hold (movement),
  dislodged units disband (retreat), builds are declined and removals fall to
  the civil-disorder rule (adjustment; DATC 4.D.8 on the standard map, or a
  variant's own law where its rulebook differs, as Beleriand's does).
- ``adjudicate()`` resolves the current phase, then advances the cycle:
  phases with no decisions but forced effects (trapped dislodged units,
  civil-disorder removals) auto-execute with empty sheets and get their own
  history record marked ``auto``; phases with neither effects nor decisions
  are skipped without a record (the three cases of :mod:`entente.engine.phase`).
  The Autumn ownership update runs at the Fall->Winter boundary through the
  ruleset hook.
- The game ends when a power owns a strict majority of supply centers
  (checked at each ownership update; 18 of 34 on the standard map, derived
  from map data) or when the board can never change again (no units and no
  possible build).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from entente.engine.adjudicator import PhaseResult, adjudicate
from entente.engine.adjudicator.builds import (
    adjudicate_adjustments,
    build_quota,
    legal_builds,
)
from entente.engine.adjudicator.common import (
    Dislodgement,
    OrderResolution,
    Resolution,
    potential_chain,
)
from entente.engine.adjudicator.retreats import (
    adjudicate_retreats,
    legal_retreats,
)
from entente.engine.game_hooks import GameHooks, GrantRecord, OathState, SpecialUnit, hooks_for
from entente.engine.map import Location, MapGraph, Terrain, UnitKind, load_map
from entente.engine.orders import (
    ADJUDICATION_ORDER_TYPES,
    LAYER2_ORDER_TYPES,
    AnyOrder,
    Bar,
    Build,
    Convoy,
    Disband,
    Dislodge,
    GameOrder,
    Hold,
    Move,
    Support,
    UnitRef,
    Waive,
    format_order,
    parse_unit_spec,
)
from entente.engine.phase import (
    Phase,
    PhaseKind,
    Season,
    adjustment_has_decisions,
    adjustment_has_effects,
    retreat_has_decisions,
    retreat_has_effects,
)
from entente.engine.rules import RULESETS, Ruleset
from entente.engine.state import State, Unit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class GameError(Exception):
    """A facade contract violation: unknown power, wrong-phase order kinds, or a finished game."""


_PHASE_ORDER_TYPES: dict[PhaseKind, tuple[type, ...]] = {
    # Retreat sheets admit the full movement grammar plus Disband: supports,
    # convoys and holds submitted during retreats are *game* orders the
    # retreat adjudicator must reject as illegal (DATC 6.H.1-6.H.4), not
    # interface errors. Variant orders (Dislodge/Bar) pass the phase gate the
    # same way -- whether a ruleset accepts them is registry data settled in
    # Phase 0 (out-of-registry orders become Hold, Sec. 1). Layer-2 orders
    # (Transform &c.) are movement-phase only: the game hooks resolve them --
    # or invalidate them under rulesets without a variant layer.
    PhaseKind.MOVEMENT: (*ADJUDICATION_ORDER_TYPES, *LAYER2_ORDER_TYPES),
    PhaseKind.RETREAT: (*ADJUDICATION_ORDER_TYPES, Disband),
    PhaseKind.ADJUSTMENT: (Build, Disband, Waive),
}


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    """One executed phase: what was submitted, what happened, what resulted.

    ``auto`` marks phases the game loop executed itself because no power had
    a decision (empty sheets). ``sheets`` holds the submissions in canonical
    notation, exactly as adjudicated -- the serialization replays these.
    ``grants`` are the knowledge transfers recorded since the previous
    record (replayed ahead of this phase's sheets); ``events`` are the variant
    layer's formatted event lines, including boundary events that occurred
    after the previous record. ``specials_after`` names the special units as
    the phase left them; with the previous record's it tells which of this
    record's units, and which of the units it dislodged or disbanded, were
    specials.
    """

    phase: str
    auto: bool
    sheets: Mapping[str, tuple[str, ...]]
    resolutions: tuple[OrderResolution, ...]
    state_after: State
    ownership_after: Mapping[str, str]
    dislodgements: tuple[Dislodgement, ...] = ()
    disbanded: tuple[Unit, ...] = ()
    built: tuple[Unit, ...] = ()
    civil_disorder: tuple[Unit, ...] = ()
    grants: tuple[GrantRecord, ...] = ()
    events: tuple[str, ...] = ()
    specials_after: tuple[SpecialUnit, ...] = ()


@dataclass(frozen=True, slots=True)
class _Seat:
    """One enumerated unit: a board value and the designator that names it.

    Twins of a plain value share one seat, since one text names any of them;
    each special unit has a seat of its own, because the designator on its
    orders is part of its name and the two specials are never twins.
    """

    unit: Unit
    special: str | None = None

    @property
    def prefix(self) -> int | None:
        """The strength an order writes for the unit: nothing at base strength 1."""
        return self.unit.strength if self.unit.strength != 1 else None


class Game:
    """A running game: engine state plus the phase machine.

    ``hooks`` carries the variant layer; None means the no-op standard layer.
    The hooks instance must be fully configured (ledger seeded, overrides
    applied) before construction -- the constructor immediately runs the
    phase-start hook when the game opens on a movement phase.
    """

    def __init__(
        self,
        ruleset: Ruleset,
        graph: MapGraph,
        state: State,
        ownership: Mapping[str, str],
        phase: Phase,
        *,
        hooks: GameHooks | None = None,
    ) -> None:
        self._hooks = hooks if hooks is not None else GameHooks(graph, None)
        self._ruleset = self._hooks.wrap_ruleset(ruleset)
        self._graph = graph
        self._state = state
        self._ownership = dict(ownership)
        self._phase = phase
        self._start_year = phase.year
        self._movement: PhaseResult | None = None
        self._pending: dict[str, list[GameOrder]] = {}
        self._pending_grants: list[GrantRecord] = []
        self._history: list[PhaseRecord] = []
        self._over = False
        if phase.kind is PhaseKind.MOVEMENT and state.units:
            started = self._hooks.on_phase_start(phase, state)
            if started is not state:
                self._state = State.of(state.map_name, phase.tag, started.units)

    @classmethod
    def new(
        cls,
        ruleset: Ruleset | str,
        map_name: str,
        *,
        start_year: int | None = None,
        seed: int | None = None,
    ) -> Game:
        """Start a fresh game from the map's starting position (rulebook data).

        ``start_year`` defaults to the ruleset's own calendar (standard:
        1901; Beleriand: 25, counting down). ``seed`` feeds the variant
        layer's RNG stream -- the engine is deterministic given it; rulesets
        without randomness ignore it.
        """
        if isinstance(ruleset, str):
            try:
                ruleset = RULESETS[ruleset]
            except KeyError:
                raise GameError(f"unknown ruleset {ruleset!r}") from None
        graph = load_map(map_name)
        hooks = hooks_for(ruleset.name, graph, seed)
        units = [
            Unit(power, *parse_unit_spec(spec))
            for power, specs in sorted(graph.starting_units.items())
            for spec in specs
        ]
        phase = hooks.initial_phase(start_year)
        state = State.of(map_name, phase.tag, units)
        ownership = {
            center: power
            for power, centers in sorted(graph.home_centers.items())
            for center in centers
        }
        hooks.register_start(state, ownership)
        return cls(ruleset, graph, state, ownership, phase, hooks=hooks)

    # -- read surface ---------------------------------------------------------

    @property
    def ruleset(self) -> Ruleset:
        """The ruleset the game adjudicates with (wrapped by the variant layer, if any)."""
        return self._ruleset

    @property
    def hooks(self) -> GameHooks:
        """The game's variant layer (the no-op base for classic games)."""
        return self._hooks

    @property
    def seed(self) -> int | None:
        """The variant-layer RNG seed this game was created with."""
        return self._hooks.seed

    @property
    def graph(self) -> MapGraph:
        """The board graph the game plays on."""
        return self._graph

    @property
    def state(self) -> State:
        """The current position (units only; ownership is separate)."""
        return self._state

    @property
    def ownership(self) -> Mapping[str, str]:
        """Supply-center province -> owning power, as a fresh dict."""
        return dict(self._ownership)

    @property
    def phase(self) -> str:
        """The current phase tag, e.g. ``"S1901M"``."""
        return self._phase.tag

    @property
    def year(self) -> int:
        """The current game year."""
        return self._phase.year

    @property
    def start_year(self) -> int:
        """The year the game opened at."""
        return self._start_year

    @property
    def phase_kind(self) -> PhaseKind:
        """The current phase's kind: movement, retreat or adjustment."""
        return self._phase.kind

    @property
    def powers(self) -> tuple[str, ...]:
        """Every power on the map, sorted: the home-center owners and the starting-unit owners."""
        return tuple(sorted(set(self._graph.home_centers) | set(self._graph.starting_units)))

    @property
    def history(self) -> tuple[PhaseRecord, ...]:
        """Every executed phase's record, oldest first."""
        return tuple(self._history)

    @property
    def pending_sheets(self) -> Mapping[str, tuple[str, ...]]:
        """Current submissions in canonical notation (for serialization)."""
        return {
            power: tuple(format_order(order) for order in sheet)
            for power, sheet in sorted(self._pending.items())
        }

    @property
    def pending_grants(self) -> tuple[GrantRecord, ...]:
        """Grants issued since the last adjudication (for serialization).

        :meth:`grant` applies the fact at once but files the record here
        until :meth:`adjudicate` flushes it into the phase record. The
        effect is live -- and so already visible in the hooks' snapshot --
        which is why a save taken in this window must carry these too, or
        its replay cannot reproduce its own ``"l2"`` fragment.
        """
        return tuple(self._pending_grants)

    @property
    def pending_dislodgements(self) -> tuple[Dislodgement, ...]:
        """The units awaiting retreat orders (empty outside retreat phases)."""
        if self._phase.kind is PhaseKind.RETREAT and self._movement is not None:
            return self._movement.dislodgements
        return ()

    @property
    def is_over(self) -> bool:
        """Whether the game has ended, by victory or a frozen board."""
        return self._over

    @property
    def winner(self) -> str | None:
        """The winning power, if the game has one.

        Classic games: the power owning a strict majority of supply centers
        (counted per ``supply_center_count`` -- multi-center special provinces
        weigh their count). Variant layers that decide victory themselves
        (Beleriand's per-power conditions) answer through their game result;
        joint winners report the first.
        """
        if self._hooks.decides_victory:
            result = self._hooks.game_result()
            return result.winners[0] if result is not None and result.winners else None
        total = sum(p.supply_center_count for p in self._graph.provinces.values())
        majority = total // 2 + 1
        counts: dict[str, int] = {}
        for center, owner in self._ownership.items():
            weight = self._graph.province(center).supply_center_count
            counts[owner] = counts.get(owner, 0) + weight
        for power in sorted(counts):
            if counts[power] >= majority:
                return power
        return None

    @property
    def winners(self) -> tuple[str, ...]:
        """Every winning power.

        Variant layers that decide victory report their full result (joint
        winners under a tie); classic games report the strict-majority power
        alone. Empty while the game runs and for a board that froze with no
        winner. :attr:`winner` is the first of these.
        """
        if self._hooks.decides_victory:
            result = self._hooks.game_result()
            return result.winners if result is not None else ()
        champion = self.winner
        return (champion,) if champion is not None else ()

    @property
    def second(self) -> str | None:
        """The power a Morgoth win placed second (rulebook "second place"), or None.

        None while the game runs, when none was designated, and for every
        classic game. Set either by :meth:`designate_second` after the win
        or, when an Oath stood at that moment, by the Oath itself.
        """
        if self._hooks.decides_victory:
            result = self._hooks.game_result()
            if result is not None:
                return result.second
        return None

    @property
    def oath(self) -> str | None:
        """The power Morgoth's standing Oath of the Second names, or None.

        None when no Oath stands or this game has none. Public by rule: every
        player may read it. See :meth:`swear_oath`.
        """
        return self._hooks.oath_state().power

    @property
    def oath_year(self) -> int | None:
        """The Year the standing Oath was sworn, or None.

        The rulebook: an Oath is "recorded with the Year in which it was
        sworn".
        """
        return self._hooks.oath_state().year

    @property
    def oaths_broken(self) -> int:
        """How many Oaths Morgoth has broken.

        Each one raises his victory threshold by one supply center for the
        rest of the game.
        """
        return self._hooks.oath_state().broken

    @property
    def oath_stabbed(self) -> bool:
        """Whether the standing Oath may be broken at no price.

        That is the case when the power it names has attacked a Morgoth unit
        since it was sworn (rulebook: "An Oath may be broken at no price if
        the Power named by it has Attacked Morgoth since the Oath was sworn,
        i. e. has Ordered a Move into a Province which is not an Area and in
        which a Unit of Morgoth stands, or a Dislodge of a Unit of Morgoth, or
        has Supported such an Order - whether or not the Order succeeded").
        """
        return self._hooks.oath_state().stabbed

    def supply_count(self, power: str) -> int:
        """Return the supply-center weight ``power`` owns; multi-center provinces count theirs."""
        return sum(
            self._graph.province(center).supply_center_count
            for center, owner in self._ownership.items()
            if owner == power
        )

    def build_quota(self, power: str) -> int:
        """Return owned centers minus units: positive = builds allowed, negative = removals due.

        Meaningful during adjustment phases. Variant layers may reshape the
        count (Beleriand's special-unit exemption and Eriador schedule).
        """
        self._require_power(power)
        variant = self._hooks.build_quota(power, self._state, self._ownership)
        if variant is not None:
            return variant
        return build_quota(self._graph, self._state, self._ownership, power)

    def grant(self, giver: str, receiver: str, fact: str) -> None:
        """Record a knowledge transfer between powers (the rulebook's "written statement").

        Effective immediately and replayed with the next executed phase.
        Raises :class:`GameError` when the variant layer refuses the transfer
        (or the game has none).
        """
        self._require_power(giver)
        self._require_power(receiver)
        if self._over:
            raise GameError("the game is over")
        error = self._hooks.validate_grant(giver, receiver, fact)
        if error is not None:
            raise GameError(error)
        self._hooks.apply_grant(giver, receiver, fact)
        self._pending_grants.append(GrantRecord(giver, receiver, fact))

    def swear_oath(self, power: str) -> None:
        """Swear the Oath of the Second naming ``power``.

        The rulebook: *"At any time Morgoth may swear a written Oath of the
        Second, naming one Power. An Oath is sworn or broken during a Phase,
        before Morgoth has placed his Orders for that Phase, and takes effect
        at once. The Oath is announced to all Players and recorded with the
        Year in which it was sworn."*

        A mid-game act on the ``grant`` channel's shape: free, effective at
        once, and -- like :meth:`break_oath`, which the book gates in the same
        sentence -- refused once the swearer's sheet for the phase is in. The
        gate is the variant's to read, so the facade only hands over the
        pending sheets. Raises :class:`GameError` when the variant layer
        refuses it (or the game has no Oath); classic games refuse outright.
        The Oath binds an eliminated power exactly as it binds a live one.
        """
        self._require_power(power)
        if self._over:
            raise GameError("the game is over")
        error = self._hooks.validate_swear(power, self._pending.keys())
        if error is not None:
            raise GameError(error)
        self._hooks.apply_swear(power, self._phase.year)

    def break_oath(self) -> None:
        """Break the standing Oath, charging its price.

        The rulebook: *"An Oath is sworn or broken during a Phase, before
        Morgoth has placed his Orders for that Phase, and takes effect at
        once."* Shares :meth:`swear_oath`'s timing gate -- what stops a
        buzzer-beater break once the winning count is already in hand -- and
        unlike swearing it is priced: the price rides on the victory threshold
        for the rest of the game. Raises :class:`GameError` on any refusal.
        """
        if self._over:
            raise GameError("the game is over")
        error = self._hooks.validate_break(self._pending.keys())
        if error is not None:
            raise GameError(error)
        self._hooks.apply_break(self._phase.year)

    def restore_oath_state(
        self, power: str | None, year: int | None, broken: int, *, stabbed: bool = False
    ) -> None:
        """Install a recorded Oath state wholesale: the LOADER'S door, not a play channel.

        Swearing and breaking are acts with a price and a phase gate, so a
        replay cannot reach this state by re-performing them: the gate has no
        phase to fire in outside the one the act was made in, and re-charging
        the price would move the very threshold the save is trying to
        reproduce. The recorded state is therefore a replay *input*, exactly
        as ``second`` is (see :mod:`entente.engine.serialization`). It
        validates loudly -- an unknown or forbidden power, a negative count,
        or any Oath at all in a game that has none is a corrupt document,
        never a silently dropped field.
        """
        if power is not None:
            self._require_power(power)
            if not isinstance(year, int) or isinstance(year, bool):
                raise GameError(f"an Oath naming {power} must record the year it was sworn")
        elif year is not None:
            raise GameError("no Oath stands, so no year can be recorded for one")
        if isinstance(broken, bool) or not isinstance(broken, int) or broken < 0:
            raise GameError(f"broken-Oath count must be a non-negative integer, not {broken!r}")
        state = OathState(power=power, year=year, broken=broken, stabbed=stabbed)
        error = self._hooks.validate_oath_state(state)
        if error is not None:
            raise GameError(error)
        self._hooks.apply_oath_state(state)

    def designate_second(self, power: str) -> None:
        """Name the power a Morgoth win places second.

        The rulebook: "If Morgoth wins, he may choose one Player to get second
        place. The rest of the Players are unplaced." A post-game act -- the
        ``grant`` channel's shape, inverted: valid ONLY once the game is over,
        exactly once, and only where the variant layer accepts it (Beleriand:
        after a Morgoth win, naming any other power, surviving or not). Raises
        :class:`GameError` on any refusal; classic games and every non-Morgoth
        Beleriand ending refuse outright. Optional and never required -- a
        Morgoth win with no second stands.
        """
        self._require_power(power)
        if not self._over:
            raise GameError("second place can only be designated after the game ends")
        error = self._hooks.validate_second(power)
        if error is not None:
            raise GameError(error)
        self._hooks.designate_second(power)

    # -- orders ----------------------------------------------------------------

    def legal_orders(self, power: str) -> tuple[AnyOrder, ...]:
        """Return every adjudication order ``power`` could sensibly submit this phase.

        Movement supports and convoys are *conditionally* valid -- they name a
        subject order another unit must actually receive; everything else is
        valid outright. The tuple is deterministic and canonically ordered
        (the neural-network order vocabularies enumerate exactly this).
        Variant layers' extra movement orders (Beleriand's Transform &c.) are
        a separate vocabulary: :meth:`legal_layer2_orders`.
        """
        self._require_power(power)
        if self._over:
            return ()
        kind = self._phase.kind
        if kind is PhaseKind.MOVEMENT:
            return tuple(self._movement_orders(power))
        if kind is PhaseKind.RETREAT:
            return tuple(self._retreat_orders(power))
        return tuple(self._adjustment_orders(power))

    def legal_layer2_orders(self, power: str) -> tuple[GameOrder, ...]:
        """Return the Layer-2 movement orders for ``power``.

        Empty for classic games and outside movement phases; the union is
        :data:`~entente.engine.orders.Layer2Order`.
        """
        self._require_power(power)
        if self._over or self._phase.kind is not PhaseKind.MOVEMENT:
            return ()
        return tuple(self._hooks.legal_extra_movement_orders(power, self._state))

    def submit(self, power: str, orders: Sequence[GameOrder]) -> None:
        """Record ``power``'s order sheet for the current phase.

        Replaces any earlier sheet by the same power (DATC 4.D.1 choice C).
        """
        self._require_power(power)
        if self._over:
            raise GameError("the game is over")
        allowed = _PHASE_ORDER_TYPES[self._phase.kind]
        for order in orders:
            if not isinstance(order, allowed):
                raise GameError(
                    f"{type(order).__name__} is not a {self._phase.kind.name.lower()}-phase order"
                )
        self._pending[power] = list(orders)

    def adjudicate(self) -> PhaseRecord:
        """Resolve the current phase with the submitted sheets and advance the machine.

        Decision-free phases are auto-executed or skipped on the way; the
        resolved phase's record is returned.
        """
        if self._over:
            raise GameError("the game is over")
        record = self._execute(self._pending, auto=False)
        self._advance()
        return record

    # -- machinery --------------------------------------------------------------

    def _require_power(self, power: str) -> None:
        if power not in self.powers:
            raise GameError(f"unknown power {power!r}")

    def _execute(self, sheets: Mapping[str, Sequence[GameOrder]], *, auto: bool) -> PhaseRecord:
        """Resolve the current phase with ``sheets`` and file its record.

        ``auto`` marks a phase the loop executed itself with empty sheets.
        """
        kind = self._phase.kind
        sheet_texts = {
            power: tuple(format_order(order) for order in sheet)
            for power, sheet in sorted(sheets.items())
            if sheet
        }
        dislodgements: tuple[Dislodgement, ...] = ()
        disbanded: tuple[Unit, ...] = ()
        built: tuple[Unit, ...] = ()
        civil: tuple[Unit, ...] = ()
        resolutions: tuple[OrderResolution, ...]
        new_units: tuple[Unit, ...]
        if kind is PhaseKind.MOVEMENT:
            plan = self._hooks.preprocess_movement(self._phase, self._state, sheets)
            result = adjudicate(self._ruleset, self._state, plan.adjudicated, graph=self._graph)
            amended, outcomes, extra_dislodgements = self._hooks.postprocess_movement(
                self._phase, self._state, plan, result
            )
            dislodgements = result.dislodgements + tuple(extra_dislodgements)
            if amended.units != result.new_state.units or extra_dislodgements:
                # the retreat stage must see the variant amendments
                # (transforms, departures, devastation expulsions) -- not the
                # raw adjudicator board or dislodgement list
                result = replace(
                    result,
                    new_state=State.of(
                        result.new_state.map_name, result.new_state.phase, amended.units
                    ),
                    dislodgements=dislodgements,
                )
            self._movement = result
            resolutions = self._hooks.merge_resolutions(sheets, plan, result.resolutions, outcomes)
            new_units = amended.units
        elif kind is PhaseKind.RETREAT:
            movement = self._movement
            if movement is None:  # pragma: no cover - machine invariant
                raise GameError("retreat phase without a movement result")
            # the submit gate keeps variant orders out of retreat sheets
            retreat_sheets = cast("Mapping[str, Sequence[AnyOrder]]", sheets)
            gated, invalid = self._hooks.preprocess_retreat(self._phase, retreat_sheets, movement)
            retreat = adjudicate_retreats(self._ruleset, movement, gated, graph=self._graph)
            self._movement = None
            disbanded = retreat.disbanded
            self._hooks.post_retreat(self._phase, retreat)
            resolutions = self._merge_invalid(retreat_sheets, invalid, retreat.resolutions)
            new_units = retreat.new_state.units
        else:
            # the submit gate keeps variant orders out of adjustment sheets
            winter_sheets = cast("Mapping[str, Sequence[AnyOrder]]", sheets)
            adjustment = self._hooks.adjudicate_adjustments(
                self._phase, self._state, self._ownership, winter_sheets
            )
            if adjustment is None:
                adjustment = adjudicate_adjustments(
                    self._ruleset, self._state, self._ownership, winter_sheets, graph=self._graph
                )
            disbanded = adjustment.disbanded
            built = adjustment.built
            civil = adjustment.civil_disorder
            self._hooks.post_adjustment(self._phase, adjustment)
            resolutions = adjustment.resolutions
            new_units = adjustment.new_state.units
        self._state = State.of(self._state.map_name, self._phase.tag, new_units)
        self._pending = {}
        record = PhaseRecord(
            phase=self._phase.tag,
            auto=auto,
            sheets=sheet_texts,
            resolutions=resolutions,
            state_after=self._state,
            ownership_after=dict(self._ownership),
            dislodgements=dislodgements,
            disbanded=disbanded,
            built=built,
            civil_disorder=civil,
            grants=tuple(self._pending_grants),
            events=self._hooks.drain_events(),
            specials_after=self._hooks.special_units(),
        )
        self._pending_grants = []
        self._history.append(record)
        return record

    def _merge_invalid(
        self,
        sheets: Mapping[str, Sequence[AnyOrder]],
        invalid: list[tuple[str, AnyOrder]],
        adjudicated: tuple[OrderResolution, ...],
    ) -> tuple[OrderResolution, ...]:
        """Append FAILS rows for orders the variant layer withheld from the retreat adjudicator.

        Row order stays (sorted power, sheet order).
        """
        if not invalid:
            return adjudicated
        del sheets  # alignment is by power grouping; withheld rows sort last per power
        extra: dict[str, list[OrderResolution]] = {}
        for power, order in invalid:
            extra.setdefault(power, []).append(
                OrderResolution(power, format_order(order), Resolution.FAILS)
            )
        merged: list[OrderResolution] = []
        powers = sorted({row.power for row in adjudicated} | set(extra))
        for power in powers:
            merged.extend(row for row in adjudicated if row.power == power)
            merged.extend(extra.get(power, []))
        return tuple(merged)

    def _advance(self) -> None:
        """Move to the next phase with decisions, auto-executing forced phases, skipping empty ones.

        The three cases are defined in :mod:`entente.engine.phase`. Season
        boundaries (a retreat stage settling, executed or skipped) run the
        ownership update (Autumn), then the variant boundary hook --
        bookkeeping, timed events, victory -- then the classic majority check
        for games without a variant victory.
        """
        while True:
            leaving = self._phase
            successor = leaving.successor()
            if leaving.kind is PhaseKind.RETREAT:
                if leaving.season is Season.FALL:
                    self._ownership = dict(
                        self._ruleset.update_supply_ownership(
                            self._graph, self._state, self._ownership
                        )
                    )
                state, ownership, over = self._hooks.on_season_end(
                    leaving, self._state, self._ownership
                )
                if state is not self._state:
                    self._state = State.of(self._state.map_name, self._state.phase, state.units)
                self._ownership = dict(ownership)
                if over:
                    self._over = True
                    return
                if (
                    leaving.season is Season.FALL
                    and not self._hooks.decides_victory
                    and self.winner is not None
                ):
                    self._over = True
                    return
            self._phase = successor
            self._state = State.of(self._state.map_name, successor.tag, self._state.units)
            kind = successor.kind
            if kind is PhaseKind.MOVEMENT:
                self._movement = None
                if self._state.units:
                    started = self._hooks.on_phase_start(successor, self._state)
                    if started is not self._state:
                        self._state = State.of(self._state.map_name, successor.tag, started.units)
                    if self._state.units:
                        return
            elif kind is PhaseKind.RETREAT:
                movement = self._movement
                if movement is not None and retreat_has_decisions(
                    self._ruleset, self._graph, movement
                ):
                    return
                if movement is not None and retreat_has_effects(movement):
                    self._execute({}, auto=True)
                else:
                    self._movement = None
            else:
                decisions = self._hooks.adjustment_has_decisions(self._state, self._ownership)
                if decisions is None:
                    decisions = adjustment_has_decisions(self._graph, self._state, self._ownership)
                if decisions:
                    return
                effects = self._hooks.adjustment_has_effects(self._state, self._ownership)
                if effects is None:
                    effects = adjustment_has_effects(self._graph, self._state, self._ownership)
                if effects:
                    self._execute({}, auto=True)
                elif not self._state.units:
                    self._over = True  # nothing can ever change again
                    return

    # -- per-phase order enumeration ---------------------------------------------

    def _movement_orders(self, power: str) -> list[AnyOrder]:
        graph = self._graph
        state = self._state
        sea_fleets: dict[str, list[Unit]] = {}
        for unit in state.units:
            if unit.kind is UnitKind.FLEET and graph.terrain(unit.location.province) is Terrain.SEA:
                sea_fleets.setdefault(unit.location.province, []).append(unit)
        # Every unit is enumerated as a seat: its value, plus the designator
        # that names it when it is one of the specials. A special's orders --
        # its own and every order that names it -- write that designator,
        # because a special named without it is named wrongly (Sec. 1); a
        # plain value's twins share one seat and one text.
        seats = self._seats(state)
        moves_of = {seat: self._unit_moves(seat, sea_fleets) for seat in seats}
        registry = self._ruleset.order_registry
        # Registry-gated targeted vocabularies, computed for EVERY unit before
        # any support: support-dislodge/support-bar subjects are exactly these
        # victim classes and bars, while the victim-order tails inside
        # _unit_dislodges name a plain Hold or a hold/move support only --
        # victim classes and bars, then supports, then tails, so the wire
        # nesting stays depth 1. None means "not in this ruleset": the
        # standard enumeration is byte-identical.
        victims_of = (
            {seat: self._dislodge_victims(seat, seats) for seat in seats}
            if Dislodge in registry
            else None
        )
        bars_of = (
            {seat: self._unit_bars(seat, seats) for seat in seats} if Bar in registry else None
        )
        orders: list[AnyOrder] = []
        for seat in seats:
            unit = seat.unit
            if unit.power != power:
                continue
            # A Hold names its unit's base strength exactly as a Move does, because
            # the rulebook reads a bare `A` as `1A`: without the prefix `A ANG H`
            # would do duty for EVERY base strength in a province and no strength class
            # could be bounded there. Strength 1 stays prefixless (the canonical
            # grammar has no `1A`), which keeps the standard column
            # byte-identical by construction.
            orders.append(Hold(unit.kind, unit.location, seat.prefix, seat.special))
            orders.extend(moves_of[seat])
            if victims_of is not None:
                orders.extend(self._unit_dislodges(seat, victims_of[seat], moves_of))
            if bars_of is not None:
                orders.extend(bars_of[seat])
            orders.extend(self._unit_supports(seat, moves_of, victims_of, bars_of))
            if unit.kind is UnitKind.FLEET and graph.terrain(unit.location.province) is Terrain.SEA:
                orders.extend(self._fleet_convoys(seat, moves_of))
        return orders

    def _seats(self, state: State) -> list[_Seat]:
        """Return one seat per unit occurrence, in the state's canonical order.

        The k-th occurrence of a value takes the k-th designator the hooks
        list for that value, so where both specials share a province each
        gets a seat and a set of orders of its own; every other occurrence
        is a plain seat.
        """
        seats: list[_Seat] = []
        seen: dict[Unit, int] = {}
        for unit in state.units:
            letters = self._hooks.special_designators(unit)
            occurrence = seen.get(unit, 0)
            seen[unit] = occurrence + 1
            seats.append(_Seat(unit, letters[occurrence] if occurrence < len(letters) else None))
        return seats

    def _seat_twins(self, seat: _Seat) -> int:
        """Count the units a seat's text may name: one for a special, the plain occurrences else."""
        if seat.special is not None:
            return 1
        unit = seat.unit
        occurrences = self._state.occupants(unit.location.province).count(unit)
        return occurrences - len(self._hooks.special_designators(unit))

    def _capable_seas(self, sea_fleets: Mapping[str, list[Unit]], cargo: Unit) -> frozenset[str]:
        """Sea provinces holding a fleet able to convoy ``cargo`` (Sec. 3.5)."""
        return frozenset(
            sea
            for sea, fleets in sea_fleets.items()
            if any(self._ruleset.convoy_capability(fleet, cargo) for fleet in fleets)
        )

    def _unit_moves(self, seat: _Seat, sea_fleets: Mapping[str, list[Unit]]) -> list[Move]:
        """Every Move the seat's unit could write, convoy attempts included for armies."""
        graph = self._graph
        unit = seat.unit
        strength = seat.prefix
        special = seat.special
        moves: list[Move] = []
        if unit.kind is UnitKind.ARMY:
            origin = unit.location.province
            for destination in sorted(graph.army_adjacency.get(origin, frozenset())):
                if not self._ruleset.entry_permission(unit, graph.province(destination)):
                    continue
                moves.append(
                    Move(
                        unit.kind,
                        unit.location,
                        Location(destination),
                        strength=strength,
                        special=special,
                    )
                )
            if graph.terrain(origin) is Terrain.COASTAL:
                capable = self._capable_seas(sea_fleets, unit)
                for destination in sorted(graph.provinces):
                    if (
                        destination != origin
                        and graph.terrain(destination) is Terrain.COASTAL
                        and self._ruleset.entry_permission(unit, graph.province(destination))
                        and potential_chain(graph, capable, origin, destination)
                    ):
                        moves.append(
                            Move(
                                unit.kind,
                                unit.location,
                                Location(destination),
                                via_convoy=True,
                                strength=strength,
                                special=special,
                            )
                        )
        else:
            for target in sorted(
                graph.fleet_adjacency.get(unit.location, frozenset()),
                key=lambda loc: (loc.province, loc.coast or ""),
            ):
                if not self._ruleset.entry_permission(unit, graph.province(target.province)):
                    continue
                moves.append(
                    Move(unit.kind, unit.location, target, strength=strength, special=special)
                )
        return moves

    def _dislodge_victims(self, seat: _Seat, seats: Sequence[_Seat]) -> list[tuple[_Seat, UnitRef]]:
        """Return the foreign victim classes the seat's unit could dislodge.

        One entry per distinct (owner, kind, strength, designator) class in
        each reachable, enterable multi-occupancy Area other than the unit's
        own (Sec. 4.2; victims are named by owner, kind and strength, and a
        special by its designator too -- sequence selection is submission-time
        business). Each entry pairs a representative seat of the class with
        the province-level written ref. Shared by :meth:`_unit_dislodges` (the
        Dislodge orders themselves) and :meth:`_unit_supports` (whose
        support-dislodge subjects are exactly these dislodges).
        """
        graph = self._graph
        unit = seat.unit
        victims: list[tuple[_Seat, UnitRef]] = []
        seen: set[tuple[str, str, UnitKind, int, str | None]] = set()
        for victim in seats:
            province = victim.unit.location.province
            if victim.unit.power == unit.power or graph.capacity(province) == 1:
                continue
            key = (
                province,
                victim.unit.power,
                victim.unit.kind,
                victim.unit.strength,
                victim.special,
            )
            if key in seen or province == unit.location.province:
                continue
            if unit.kind is UnitKind.ARMY:
                if graph.terrain(province) is Terrain.SEA or not graph.army_reachable(
                    unit.location.province, province
                ):
                    continue
            elif not graph.fleet_destinations(unit.location, province):
                continue
            if not self._ruleset.entry_permission(unit, graph.province(province)):
                continue
            seen.add(key)
            ref = UnitRef(
                victim.unit.power,
                victim.unit.kind,
                Location(province),
                victim.prefix,
                victim.special,
            )
            victims.append((victim, ref))
        return victims

    def _unit_dislodges(
        self,
        seat: _Seat,
        victims: list[tuple[_Seat, UnitRef]],
        moves_of: Mapping[_Seat, list[Move]],
    ) -> list[AnyOrder]:
        """Return the Dislodge orders the seat's unit could give.

        One per victim class from :meth:`_dislodge_victims`, plus, for each
        class, one victim-ORDER variant per support a unit of that class could
        give.

        The victim-order variant NAMES the victim's action to cut a specific
        support (Sec. 3.4/4.2 "the Dislodgment of a specific Unit, by naming
        its action"). The named action is drawn from the victim class's own
        legal supports -- the rulebook's cuttable actions -- plus one plain
        Hold per class (every unit can hold, and the wall excludes its giver,
        so among mixed hold/support twins ``DIS ... H`` picks a holding twin
        whose rule-5 defence differs from its supporting mates'), each with
        its unit spec normalized to the province-level victim ref so the
        canonical ``DIS ... <action>`` tail round-trips. Tails name HOLD/MOVE
        supports only: a support-dislodge/support-bar tail would nest the wire
        two deep, so the walk below asks for the un-widened support set (the
        tolerant parser still accepts a hand-written deeper form). The
        adjudicator consumes ``victim_order`` through
        :func:`common._resolve_victims`; the enumeration is byte-identical for
        standard (Dislodge is not in that registry). These variants are
        offered by the order list alone.
        """
        orders: list[AnyOrder] = []
        unit = seat.unit
        strength = seat.prefix
        for victim, ref in victims:
            orders.append(
                Dislodge(unit.kind, unit.location, ref, strength=strength, special=seat.special)
            )
            orders.append(
                Dislodge(
                    unit.kind,
                    unit.location,
                    ref,
                    Hold(ref.kind, ref.location, ref.strength, ref.special),
                    strength,
                    seat.special,
                )
            )
            for support in self._unit_supports(victim, moves_of):
                # the named action describes the REF unit (province-level, ref
                # strength and designator); the tail then re-parses to exactly
                # this order
                victim_order = replace(
                    support,
                    unit_kind=ref.kind,
                    location=ref.location,
                    strength=ref.strength,
                    special=ref.special,
                )
                orders.append(
                    Dislodge(unit.kind, unit.location, ref, victim_order, strength, seat.special)
                )
        return orders

    def _unit_bars(self, seat: _Seat, seats: Sequence[_Seat]) -> list[Bar]:
        """Return the Bar orders the seat's unit could give from inside a multi-occupancy Area.

        One per distinct foreign unit class that could move in (Sec. 4.2), a
        special being a class of its own under its designator. Doubles as the
        support-bar subject set of :meth:`_unit_supports`, so the concrete
        ``Bar`` type is kept.
        """
        graph = self._graph
        unit = seat.unit
        own_province = unit.location.province
        if graph.capacity(own_province) == 1:
            return []
        orders: list[Bar] = []
        strength = seat.prefix
        seen: set[tuple[str, str, UnitKind, int, str | None]] = set()
        for victim in seats:
            province = victim.unit.location.province
            if victim.unit.power == unit.power or province == own_province:
                continue
            key = (
                province,
                victim.unit.power,
                victim.unit.kind,
                victim.unit.strength,
                victim.special,
            )
            if key in seen:
                continue
            if victim.unit.kind is UnitKind.ARMY:
                if not graph.army_reachable(province, own_province):
                    continue
            elif not any(
                graph.fleet_destinations(loc, own_province)
                for loc in graph.fleet_locations(province)
            ):
                continue
            seen.add(key)
            ref = UnitRef(
                victim.unit.power,
                victim.unit.kind,
                Location(province),
                victim.prefix,
                victim.special,
            )
            orders.append(
                Bar(unit.kind, unit.location, ref, Location(own_province), strength, seat.special)
            )
        return orders

    def _unit_supports(
        self,
        seat: _Seat,
        moves_of: Mapping[_Seat, list[Move]],
        victims_of: Mapping[_Seat, list[tuple[_Seat, UnitRef]]] | None = None,
        bars_of: Mapping[_Seat, list[Bar]] | None = None,
    ) -> list[Support]:
        """Return the supports the seat's unit could give.

        Holds of reachable units (co-located Area-mates included), moves of
        any unit toward a reachable province (one per destination *province*
        -- an omitted coast matches any actual landing coast, DATC 6.B.4).
        Every row is gated by :meth:`_support_targets`, so a unit inside an
        Area is also offered the orders aimed at that Area: a Move entering it
        and a Dislodge against a unit standing in it, neither of which the
        supporter could reach. Subjects in multi-unit provinces (or with a
        base strength above 1) are fully specified -- owner and strength prefix --
        so the written support binds unambiguously (the rulebook's 4.C.5);
        a special subject writes its designator, so the two specials are
        two subjects; classic positions stay bare.

        A subject seat equal to ``seat`` is skipped only when its text names
        no other unit: with two fungible occurrences of a plain value in a
        multi-unit Area the text also denotes the other unit, whose moves are
        supportable -- the adjudicator binds the reference to the twin, never
        the supporter itself (a unit is never its own subject) -- while a
        special's text names that special alone.

        ``victims_of``/``bars_of`` (present exactly when Dislodge/Bar are in
        the ruleset registry -- the standard column never sees them) add the
        targeted subjects: the subject's GENERIC Dislodges when ``unit``
        targets the VICTIM's Area -- the support's target province, per
        :func:`common._validate_support`; entry permission is no
        supporter-side gate, a supporter co-located with the dislodger is
        legal, and so is one standing beside the victim -- and the subject's
        Bars when ``unit`` targets the subject's OWN Area (the rulebook's "a
        Unit in adjacent Province can Support a Barring Order" is permissive,
        not exclusive). Victim-order variants are never subjects: support matching
        compares victim refs only (Sec. 1), so a tailed subject would
        duplicate the generic row -- the tolerant parser still accepts a
        hand-written one.

        Reused by :meth:`_unit_dislodges` to enumerate a victim class's
        cuttable actions (which omits both mappings: tails stay hold/move,
        nesting stays depth 1), so the return type is the concrete
        ``Support``. A victim inside an Area may write a move support aimed
        at that Area, so the tails carry those too.
        """
        supports: list[Support] = []
        unit = seat.unit
        strength = seat.prefix
        special = seat.special
        for subject, moves in moves_of.items():
            if subject == seat and self._seat_twins(seat) < 2:
                continue
            subject_unit = subject.unit
            crowded = len(self._state.occupants(subject_unit.location.province)) > 1
            subject_power = subject_unit.power if crowded else None
            subject_strength = subject.prefix
            # A Bar's barred Area is its giver's own province, so the two share
            # one gate: the supporter reaches that province, or stands in it.
            defends_subject = self._support_targets(unit, subject_unit.location.province)
            if defends_subject:
                supports.append(
                    Support(
                        unit.kind,
                        unit.location,
                        Hold(
                            subject_unit.kind,
                            subject_unit.location,
                            subject_strength,
                            subject.special,
                        ),
                        strength,
                        subject_power,
                        special,
                    )
                )
            provinces: set[str] = set()
            for move in moves:
                province = move.destination.province
                if province not in provinces and self._support_targets(unit, province):
                    provinces.add(province)
                    supports.append(
                        Support(
                            unit.kind,
                            unit.location,
                            Move(
                                subject_unit.kind,
                                subject_unit.location,
                                Location(province),
                                strength=subject_strength,
                                special=subject.special,
                            ),
                            strength,
                            subject_power,
                            special,
                        )
                    )
            if victims_of is not None:
                for _, ref in victims_of[subject]:
                    if self._support_targets(unit, ref.location.province):
                        supports.append(
                            Support(
                                unit.kind,
                                unit.location,
                                Dislodge(
                                    subject_unit.kind,
                                    subject_unit.location,
                                    ref,
                                    strength=subject_strength,
                                    special=subject.special,
                                ),
                                strength,
                                subject_power,
                                special,
                            )
                        )
            if bars_of is not None and defends_subject:
                for bar in bars_of[subject]:
                    supports.append(
                        Support(unit.kind, unit.location, bar, strength, subject_power, special)
                    )
        return supports

    def _support_targets(self, unit: Unit, province: str) -> bool:
        """Whether ``unit`` may aim a support at ``province`` (the Sec. 1 reach rule).

        Reach, or standing in the province when it is an Area: a unit inside an
        Area backs every order aimed at it. Mirrors the Phase 0 gate in
        :func:`common._validate_support`, so the menu and the adjudicator
        cannot drift apart.
        """
        if self._support_reachable(unit, province):
            return True
        own = unit.location.province
        return province == own and self._graph.capacity(own) != 1

    def _support_reachable(self, unit: Unit, province: str) -> bool:
        graph = self._graph
        if unit.kind is UnitKind.ARMY:
            return graph.terrain(province) is not Terrain.SEA and graph.army_reachable(
                unit.location.province, province
            )
        return bool(graph.fleet_destinations(unit.location, province))

    def _fleet_convoys(self, seat: _Seat, moves_of: Mapping[_Seat, list[Move]]) -> list[AnyOrder]:
        """Return the convoys the seat's fleet could carry.

        Any army's potential convoyed move the fleet is capable of (Sec. 3.5;
        the fleet need not lie on the chain to give a valid order). The cargo
        move comes from the cargo's own seat, so it names a special cargo by
        its designator.
        """
        convoys: list[AnyOrder] = []
        fleet = seat.unit
        strength = seat.prefix
        for cargo, moves in moves_of.items():
            if cargo.unit.kind is not UnitKind.ARMY:
                continue
            if not self._ruleset.convoy_capability(fleet, cargo.unit):
                continue
            for move in moves:
                if move.via_convoy:
                    convoys.append(Convoy(fleet.kind, fleet.location, move, strength, seat.special))
        return convoys

    def _retreat_orders(self, power: str) -> list[AnyOrder]:
        movement = self._movement
        if movement is None:
            return []
        orders: list[AnyOrder] = []
        seen: dict[Unit, int] = {}
        for dislodgement in movement.dislodgements:
            if dislodgement.unit.power != power:
                continue
            unit = dislodgement.unit
            # a dislodged special writes its designator in every retreat order:
            # the one its dislodgement record carries, so that each special is
            # offered the retreats valid for its own dislodger (the retreat
            # binding requires it); a record without one takes the k-th
            # designator the hooks list for that value in limbo
            letters = self._hooks.special_designators(unit, limbo=True)
            occurrence = seen.get(unit, 0)
            seen[unit] = occurrence + 1
            special: str | None
            if dislodgement.special in letters:
                special = dislodgement.special
            else:
                special = letters[occurrence] if occurrence < len(letters) else None
            # a retreat-phase Disband names its unit's base strength like the
            # retreat moves beside it: two units dislodged from one Area are not
            # twins unless they share a base strength
            orders.append(
                Disband(
                    unit.kind, unit.location, unit.strength if unit.strength != 1 else None, special
                )
            )
            orders.extend(
                replace(move, special=special)
                for move in legal_retreats(self._ruleset, self._graph, movement, dislodgement)
            )
        return orders

    def _adjustment_orders(self, power: str) -> list[AnyOrder]:
        variant = self._hooks.legal_adjustment_orders(power, self._state, self._ownership)
        if variant is not None:
            return list(variant)
        quota = build_quota(self._graph, self._state, self._ownership, power)
        orders: list[AnyOrder] = []
        if quota > 0:
            orders.extend(legal_builds(self._graph, self._state, self._ownership, power))
            if orders:
                orders.append(Waive())
        elif quota < 0:
            for unit in self._state.units:
                if unit.power == power:
                    # the removal names its base strength too: disbanding the 2A in an
                    # Area is a different choice from disbanding its 1A neighbour
                    orders.append(
                        Disband(
                            unit.kind, unit.location, unit.strength if unit.strength != 1 else None
                        )
                    )
        return orders
