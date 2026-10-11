"""Behavioral-cloning agent: play a whole game from a trained checkpoint.

Network inference batches through :mod:`entente.sim.vectorized`. The learned
policy covers **movement** phases, which is where behavioral cloning is defined
and where its acceptance metric (per-unit top-1 agreement with the human order)
lives; retreat and adjustment phases delegate to a fallback heuristic (greedy by
default), so a :class:`BCAgent` is a complete player. Per-unit orders are
decoded greedily and masked to the engine's legal set
(:func:`entente.nn.order_vocab.legal_order_indices`), so the agent can only ever
submit legal orders.

Two entry points share the decoder:

- :class:`BCAgent` implements the :class:`~entente.agents.base.Agent` protocol
  for one (game, power) at a time -- the CLI/single-game path.
- :func:`bc_lockstep_policy` builds a :data:`~entente.sim.vectorized.PolicyCallback`
  that encodes every game needing a BC decision this tick in **one batched
  forward pass** -- the GPU-efficient path :func:`bc_vs_greedy` uses to run
  tournaments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING

import torch

from entente.agents.registry import build_agent
from entente.engine.orders import parse_order
from entente.engine.phase import PhaseKind
from entente.nn.features import _unit_token, encode_game
from entente.nn.model import BCModel
from entente.nn.order_vocab import legal_order_indices

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.agents.base import Agent
    from entente.engine.game import Game
    from entente.engine.orders import GameOrder
    from entente.nn.order_vocab import OrderVocabulary
    from entente.sim.vectorized import DecisionRequest, PolicyCallback


def _decode_orders(
    model: BCModel,
    encoding: torch.Tensor,
    game: Game,
    power: str,
    *,
    sample: bool = False,
    generator: torch.Generator | None = None,
) -> list[GameOrder]:
    """Decode ``power``'s movement orders from a board ``encoding`` of ``(L, d_model)``.

    Each unit takes one of its legal orders -- the argmax, or a draw when
    ``sample`` is set -- decoded autoregressively and mapped back to an engine
    order.
    """
    vocab: OrderVocabulary = model.vocab
    groups = legal_order_indices(game, power, vocab)
    units = sorted((u for u in game.state.units if u.power == power), key=lambda u: str(u.location))
    slots, legal_ids = [], []
    for unit in units:
        legal = groups.get(str(unit.location))
        if legal:
            slots.append(model.board.location_index(_unit_token(unit)))
            legal_ids.append(legal)
    if not slots:
        return []
    chosen = model.policy.decode(encoding[slots], legal_ids, sample=sample, generator=generator)
    return [parse_order(vocab.order(idx)) for idx in chosen]


class BCAgent:
    """A trained-checkpoint policy that still plays a whole game.

    Movement is the network; every other phase is handed to a fallback agent
    (greedy by default).
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        seed: int = 0,
        device: str = "cpu",
        fallback: str = "greedy",
    ) -> None:
        self.device = torch.device(device)
        self.model = BCModel.load(checkpoint, map_location=self.device).eval()
        self._fallback: Agent = build_agent(fallback, seed)

    @torch.no_grad()
    def orders(self, game: Game, power: str) -> Sequence[GameOrder]:
        """Return ``power``'s sheet: the network in movement, the fallback elsewhere."""
        if game.phase_kind is not PhaseKind.MOVEMENT:
            return self._fallback.orders(game, power)
        encoding = self.model.encode(encode_game(game).unsqueeze(0).to(self.device))[0]
        return _decode_orders(self.model, encoding, game, power)


def bc_lockstep_policy(
    model: BCModel,
    *,
    bc_power: str | None = None,
    fallback: Agent | None = None,
    device: str = "cpu",
    sample: bool = False,
    generator: torch.Generator | None = None,
) -> PolicyCallback:
    """Build a :data:`~entente.sim.vectorized.PolicyCallback` that batches BC decoding.

    Every request whose phase is movement and whose power is ``bc_power`` (or any
    power, if ``None``) is answered by the network; the games needing a decision
    are encoded together in one forward pass. All other requests fall to
    ``fallback`` (submitting nothing if it is ``None``). ``sample`` draws from the
    policy (variance for tournament CIs) rather than taking the argmax.
    """
    dev = torch.device(device)

    @torch.no_grad()
    def policy(requests: Sequence[DecisionRequest]) -> list[list[GameOrder]]:
        sheets: list[list[GameOrder] | None] = [None] * len(requests)
        bc_indices = [
            i
            for i, req in enumerate(requests)
            if req.game.phase_kind is PhaseKind.MOVEMENT
            and (bc_power is None or req.power == bc_power)
        ]
        if bc_indices:
            # encode each distinct game once (several powers may share a game),
            # then decode each request off its game's row -- the batched pass.
            row_of: dict[int, int] = {}
            distinct: list[Game] = []
            for i in bc_indices:
                game = requests[i].game
                if id(game) not in row_of:
                    row_of[id(game)] = len(distinct)
                    distinct.append(game)
            encoded = model.encode(torch.stack([encode_game(g) for g in distinct]).to(dev))
            for i in bc_indices:
                req = requests[i]
                sheets[i] = _decode_orders(
                    model,
                    encoded[row_of[id(req.game)]],
                    req.game,
                    req.power,
                    sample=sample,
                    generator=generator,
                )
        for i, req in enumerate(requests):
            if sheets[i] is None:
                sheets[i] = list(fallback.orders(req.game, req.power)) if fallback else []
        return [sheet or [] for sheet in sheets]

    return policy


@dataclass(frozen=True, slots=True)
class BCvsGreedyResult:
    """Aggregate of a BC-vs-greedy 1-vs-6 tournament (SoS shares)."""

    games: int
    bc_mean_sos: float
    bc_ci: tuple[float, float]
    greedy_mean_sos: float


def bc_vs_greedy(
    checkpoint: str,
    seeds: Sequence[int],
    *,
    max_year: int = 1910,
    device: str = "cpu",
    seed: int = 0,
    map_name: str = "standard",
    ruleset: str = "standard",
    output: str | Path | None = None,
) -> BCvsGreedyResult:
    """Run BC (sampled) in each seat against six greedy opponents.

    The ``7 * len(seeds)`` games batch through
    :class:`~entente.sim.vectorized.LockstepRunner`, and the result carries BC's
    mean SoS share with a bootstrap CI beside greedy's mean share.

    BC samples from its policy, so the games in a seat's seed-set diverge (argmax
    play would be identical every game); a fixed generator seed keeps the run
    reproducible. ``output`` writes one seated record per game in the tournament
    JSONL schema (:func:`entente.eval.tournament.write_seated_jsonl`), so the
    aggregates above regenerate from the committed file alone, exactly like the
    ProcessPool rungs' artifacts: every reported number is produced by a script
    reading a result file, never entered by hand. The record's ``seed`` field is
    the game's logical label ``seeds[i]``: lockstep games share one torch
    generator, so the reproducibility unit is the whole run (this function with
    the same arguments), not any single game.
    """
    from entente.eval.scoring import sos_scores
    from entente.eval.stats import bootstrap_ci
    from entente.eval.tournament import SeatedOutcome, write_seated_jsonl
    from entente.sim.parallel import GameOutcome
    from entente.sim.vectorized import LockstepRunner, new_games

    model = BCModel.load(checkpoint, map_location=torch.device(device)).eval()
    powers = model.board.powers
    # the generator's device must match the sampled tensors' (the model's),
    # or torch.multinomial raises on CUDA; on CPU this is identical to the
    # default generator.
    generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
    bc_shares: list[float] = []
    greedy_shares: list[float] = []
    seated: list[SeatedOutcome] = []
    for seat in powers:
        games = new_games(len(seeds), ruleset=ruleset, map_name=map_name)
        policy = bc_lockstep_policy(
            model,
            bc_power=seat,
            fallback=build_agent("greedy", seed),
            device=device,
            sample=True,
            generator=generator,
        )
        LockstepRunner(games, policy, max_year=max_year).run()
        for game_index, game in enumerate(games):
            counts = {p: game.supply_count(p) for p in powers}
            shares = sos_scores(counts, game.winner)
            bc_shares.append(shares[seat])
            greedy_shares.extend(shares[p] for p in powers if p != seat)
            if output is not None:
                seated.append(
                    SeatedOutcome(
                        outcome=GameOutcome(
                            seed=seeds[game_index],
                            ruleset=ruleset,
                            map_name=map_name,
                            max_year=max_year,
                            num_phases=len(game.history),
                            final_phase=game.phase,
                            winner=game.winner,
                            supply_counts=counts,
                        ),
                        seats={p: ("bc" if p == seat else "greedy") for p in powers},
                    )
                )
    if output is not None:
        write_seated_jsonl(seated, output)
    return BCvsGreedyResult(
        games=len(bc_shares),
        bc_mean_sos=fmean(bc_shares),
        bc_ci=bootstrap_ci(bc_shares),
        greedy_mean_sos=fmean(greedy_shares),
    )
