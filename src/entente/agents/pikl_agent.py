"""piKL agent: a BC checkpoint as anchor policy and value net over the search stack.

The search stack itself is torch-free; this module is the seam that hands it a
network (docs/pikl.tex Sec. 9). Wiring per movement decision: sample candidate
order-sets for **every** power from the BC policy (:class:`BCCandidatePolicy` ->
``search.candidates``), score joint profiles by stepping the engine one ply and
reading the BC value head on the successors (:func:`value_scores` ->
``search.rollout``), run anchored hedge over the subgame (``search.pikl``), then
play from the mean strategy. Non-movement phases delegate to a fallback agent,
mirroring :class:`~entente.agents.bc_agent.BCAgent` -- the anchor policy is only
defined where BC is.

Like ``bc_agent``, this module is constructed directly from a config (its
inputs -- checkpoint, k, N, lam -- do not fit the seed-only registry factory
shape). :func:`pikl_lockstep_policy` is the tournament path: the piKL seat is
searched per game; the six anchor opponents batch through
:func:`~entente.agents.bc_agent.bc_lockstep_policy` unchanged.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, cast

import torch

from entente.agents.bc_agent import _decode_orders, bc_lockstep_policy
from entente.agents.registry import build_agent
from entente.engine.game import Game
from entente.engine.orders import OrderSyntaxError, Waive, format_order, parse_order
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.state import State
from entente.eval.scoring import sos_scores
from entente.nn.features import _unit_token, encode_game
from entente.nn.model import BCModel
from entente.nn.order_vocab import legal_order_indices
from entente.search.candidates import CandidateSet, ScoredOrderSet, collect_candidates
from entente.search.pikl import PiklPlayer, run_pikl, sample_index
from entente.search.rollout import profile_utility_fn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from entente.agents.base import Agent
    from entente.engine.orders import AnyOrder, GameOrder
    from entente.search.config import PiklAgentConfig
    from entente.sim.vectorized import DecisionRequest, PolicyCallback


class BCCandidatePolicy:
    """A :class:`entente.search.candidates.CandidatePolicy` backed by a BC model.

    The board is encoded once, then ``count`` order-sets are decoded -- the
    greedy one first when asked, the rest sampled at ``temperature``/``top_p`` --
    each scored with its temperature-1 log-probability (docs/pikl.tex Sec. 5).
    """

    def __init__(
        self,
        model: BCModel,
        *,
        device: torch.device,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model = model
        self.device = device
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def sample_order_sets(
        self,
        game: Game,
        power: str,
        count: int,
        *,
        include_greedy: bool,
        rng: random.Random,
    ) -> list[ScoredOrderSet]:
        """Decode ``count`` scored order-sets for ``power`` in ``game``'s position."""
        model = self.model
        groups = legal_order_indices(game, power, model.vocab)
        units = sorted(
            (u for u in game.state.units if u.power == power), key=lambda u: str(u.location)
        )
        slots, legal_ids = [], []
        for unit in units:
            legal = groups.get(str(unit.location))
            if legal:
                slots.append(model.board.location_index(_unit_token(unit)))
                legal_ids.append(legal)
        if not slots:
            return [ScoredOrderSet((), 0.0)] * count
        encoding = model.encode(encode_game(game).unsqueeze(0).to(self.device))[0]
        enc_slots = encoding[slots]
        # generator device must match the model's, or multinomial raises on
        # CUDA; identical to the default on CPU.
        generator = torch.Generator(device=self.device).manual_seed(rng.getrandbits(63))
        results: list[ScoredOrderSet] = []
        for position in range(count):
            greedy = include_greedy and position == 0
            chosen, log_prob = model.policy.decode_scored(
                enc_slots,
                legal_ids,
                sample=not greedy,
                temperature=self.temperature,
                top_p=self.top_p,
                generator=generator,
            )
            # the order vocabulary enumerates standard-grammar orders only
            orders = cast(
                "tuple[AnyOrder, ...]",
                tuple(parse_order(model.vocab.order(idx)) for idx in chosen),
            )
            results.append(ScoredOrderSet(orders, log_prob))
        return results


def value_scores(
    model: BCModel, device: torch.device, games: Sequence[Game]
) -> list[dict[str, float]]:
    """Score live states with the BC value head and finished ones with the truth.

    This is the production :data:`entente.search.rollout.ValueFn`. A live board
    gets the value head's softmaxed 7-power SoS-share prediction; a decided one
    gets its exact Sum-of-Squares scores, because the net never sees a decided
    board at train time and there the truth is available anyway.
    """
    powers = model.board.powers
    results: list[dict[str, float] | None] = [None] * len(games)
    live: list[int] = []
    for i, game in enumerate(games):
        if game.is_over:
            counts = {p: game.supply_count(p) for p in powers}
            results[i] = sos_scores(counts, game.winner)
        else:
            live.append(i)
    if live:
        feats = torch.stack([encode_game(games[i]) for i in live]).to(device)
        shares = torch.softmax(model.value(model.encode(feats)), dim=-1).cpu()
        for row, i in enumerate(live):
            results[i] = dict(zip(powers, shares[row].tolist(), strict=True))
    return [r for r in results if r is not None]


class PiklAgent:
    """A complete piKL player: anchored search in movement, a fallback elsewhere.

    ``model`` may be passed in so that one loaded checkpoint is shared across
    many agent instances, which is what a lambda sweep needs.
    """

    def __init__(self, config: PiklAgentConfig, *, model: BCModel | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if model is None:
            model = BCModel.load(config.checkpoint, map_location=self.device)
        self.model = model.eval()
        self._fallback: Agent = build_agent(config.fallback, config.seed)
        self._candidates = BCCandidatePolicy(
            self.model,
            device=self.device,
            temperature=config.search.temperature,
            top_p=config.search.top_p,
        )

    @torch.no_grad()
    def orders(self, game: Game, power: str) -> Sequence[GameOrder]:
        """Return ``power``'s sheet: the searched strategy in movement, else the fallback."""
        if game.phase_kind is not PhaseKind.MOVEMENT:
            return self._fallback.orders(game, power)
        search = self.config.search
        # One deterministic stream per decision point (RandomAgent's replay
        # convention): candidate sampling, opponent sampling inside the
        # search, and the final draw from the mean strategy all share it.
        rng = random.Random(f"{self.config.seed}:{game.phase}:{power}")
        subgame_powers: list[str] = []
        candidate_sets: list[CandidateSet] = []
        for other in game.powers:
            candidates = collect_candidates(
                self._candidates,
                game,
                other,
                k=search.k,
                attempts=search.attempts,
                include_greedy=search.include_greedy,
                rng=rng,
            )
            if candidates.order_sets == ((),):  # power without units this phase
                continue
            subgame_powers.append(other)
            candidate_sets.append(candidates)
        if power not in subgame_powers:
            return ()
        oracle = profile_utility_fn(
            game,
            subgame_powers,
            [cs.order_sets for cs in candidate_sets],
            lambda states: value_scores(self.model, self.device, states),
        )
        result = run_pikl(
            [PiklPlayer(cs.anchor, search.lam) for cs in candidate_sets],
            oracle,
            iterations=search.iterations,
            mode="sampled",
            eta=search.eta.constant,
            adaptive_scale=search.eta.scale,
            rng=rng,
        )
        mine = subgame_powers.index(power)
        mean = result.mean_strategies[mine]
        if search.play == "argmax":
            pick = max(range(len(mean)), key=mean.__getitem__)
        else:
            pick = sample_index(mean, rng)
        return list(candidate_sets[mine].order_sets[pick])


def _order_location(order: GameOrder) -> str:
    """Return the acting unit's location for a movement-phase order.

    A Waive has no location and is never part of a per-unit sheet, so one here
    is a caller error rather than a case to handle.
    """
    if isinstance(order, Waive):
        raise ValueError("movement order expected, got a waive")
    return str(order.location)


class AnchorAgreement:
    """Per-game agreement between piKL's movement orders and the anchor's decode.

    Counting per unit how often the searched order equals the anchor's greedy
    decode gives a humanlikeness proxy that needs no dataset, which is what the
    lambda sweep records (docs/pikl.tex Sec. 7). The human metric proper needs
    held-out human orders; that is :func:`human_agreement`.
    """

    def __init__(self) -> None:
        self._matched: dict[int, int] = {}
        self._units: dict[int, int] = {}

    @torch.no_grad()
    def count(
        self,
        model: BCModel,
        device: torch.device,
        game: Game,
        power: str,
        orders: Sequence[GameOrder],
    ) -> None:
        """Fold one power's sheet for ``game`` into the running agreement counts."""
        encoding = model.encode(encode_game(game).unsqueeze(0).to(device))[0]
        greedy = _decode_orders(model, encoding, game, power)
        anchor_by_location = {_order_location(o): format_order(o) for o in greedy}
        key = id(game)
        for order in orders:
            self._units[key] = self._units.get(key, 0) + 1
            if anchor_by_location.get(_order_location(order)) == format_order(order):
                self._matched[key] = self._matched.get(key, 0) + 1

    def rate_for(self, game: Game) -> float | None:
        """Return ``game``'s agreement rate so far, or None if no unit was counted."""
        units = self._units.get(id(game), 0)
        if units == 0:
            return None
        return self._matched.get(id(game), 0) / units


def pikl_lockstep_policy(
    agent: PiklAgent,
    *,
    pikl_power: str,
    generator: torch.Generator | None = None,
    fallback: Agent | None = None,
    agreement: AnchorAgreement | None = None,
) -> PolicyCallback:
    """Build the sweep's :data:`~entente.sim.vectorized.PolicyCallback`.

    The piKL seat's movement decisions run the full search, one game at a time.
    Every other request -- the six anchor opponents sampling from the BC policy,
    and all non-movement phases -- forwards to :func:`bc_lockstep_policy` in one
    positional sub-batch.
    """
    anchor_policy = bc_lockstep_policy(
        agent.model,
        fallback=fallback,
        device=str(agent.device),
        sample=True,
        generator=generator,
    )

    def policy(requests: Sequence[DecisionRequest]) -> list[list[AnyOrder]]:
        sheets: list[list[AnyOrder] | None] = [None] * len(requests)
        forwarded: list[DecisionRequest] = []
        forwarded_at: list[int] = []
        for i, request in enumerate(requests):
            if request.power == pikl_power and request.game.phase_kind is PhaseKind.MOVEMENT:
                orders = list(agent.orders(request.game, request.power))
                if agreement is not None:
                    agreement.count(agent.model, agent.device, request.game, request.power, orders)
                sheets[i] = orders
            else:
                forwarded.append(request)
                forwarded_at.append(i)
        if forwarded:
            answered = anchor_policy(forwarded)
            for j, i in enumerate(forwarded_at):
                sheets[i] = list(answered[j])
        return [sheet or [] for sheet in sheets]

    return policy


def movement_game_from_row(
    graph_game: Game,
    units_by_power: Mapping[str, Sequence[str]],
    centers_by_power: Mapping[str, Sequence[str]],
    phase_tag: str,
) -> Game:
    """Rebuild a shard row's movement position on ``graph_game``'s map and ruleset.

    The reconstruction is literally the one the training set uses
    (:func:`entente.data.dataset.units_from_specs`, which also owns the dump's
    ``*`` dislodged-marker handling). The season is normalized to spring because
    movement legality is season-free.
    """
    from entente.data.dataset import units_from_specs

    units = units_from_specs(units_by_power)
    ownership = {
        province: owner for owner, provinces in centers_by_power.items() for province in provinces
    }
    phase = Phase(int(phase_tag[1:-1]), Season.SPRING, PhaseKind.MOVEMENT)
    state = State.of(graph_game.state.map_name, phase.tag, units)
    return Game(graph_game.ruleset, graph_game.graph, state, ownership, phase)


def human_agreement(
    agent: PiklAgent,
    shards_dir: str | Path,
    *,
    rows: int = 200,
    seed: int = 0,
    map_name: str = "standard",
    ruleset: str = "standard",
) -> tuple[float | None, int]:
    """Measure per-unit top-1 agreement of piKL play with recorded human orders.

    The sample is a deterministic draw of preprocessed shard rows. Point
    ``shards_dir`` at *held-out* shards: this function does not split them. A
    recorded order that is unparseable, or illegal in the reconstructed position,
    counts as a hold, which is exactly how the training labels treat it
    (:meth:`entente.data.dataset.ShardDataset._label_for`). Returns
    ``(rate, units)``, with the rate ``None`` when no unit was scored.
    """
    import json
    from pathlib import Path as _Path

    import pyarrow.parquet as pq

    from entente.engine.orders import Hold

    shard_paths = sorted(_Path(shards_dir).glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"no parquet shards under {shards_dir}")
    table_rows: list[dict[str, object]] = []
    for path in shard_paths:
        table_rows.extend(pq.read_table(path).to_pylist())
    random.Random(seed).shuffle(table_rows)
    template = Game.new(ruleset, map_name)
    matched = 0
    units = 0
    for row in table_rows[:rows]:
        game = movement_game_from_row(
            template,
            json.loads(str(row["units_json"])),
            json.loads(str(row["centers_json"])),
            str(row["phase"]),
        )
        power = str(row["power"])
        legal = set(game.legal_orders(power))
        recorded: dict[str, str] = {}
        for text in json.loads(str(row["orders_json"])):
            try:
                order = parse_order(text)
            except OrderSyntaxError:  # dataset rows may carry junk orders (notation drift only)
                continue
            if order in legal:
                recorded[_order_location(order)] = format_order(order)
        chosen = agent.orders(game, power)
        for order in chosen:
            location = _order_location(order)
            # the unit's DEFAULT hold, spelled the way the enumeration spells it
            # -- naming its base strength, since a bare token means base strength 1
            # and would never equal a stronger unit's own Hold text (the agreement
            # metric would then miss every such unit). Vacuous on the standard data
            # this reads, where every unit has base strength 1.
            target = recorded.get(
                location,
                format_order(
                    next(
                        Hold(u.kind, u.location, u.strength if u.strength != 1 else None)
                        for u in game.state.units
                        if u.power == power and str(u.location) == location
                    )
                ),
            )
            units += 1
            matched += format_order(order) == target
    return (matched / units if units else None), units


__all__ = [
    "AnchorAgreement",
    "BCCandidatePolicy",
    "PiklAgent",
    "human_agreement",
    "movement_game_from_row",
    "pikl_lockstep_policy",
    "value_scores",
]
