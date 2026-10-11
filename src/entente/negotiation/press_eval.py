"""Measure what press does to outcomes, over full self-play games.

Seven piKL agents share one BC checkpoint (anchor policy + value net) and
play a full self-play game; one designated **subject** seat's press
capability is toggled between conditions with the **same seeds** -- "no
press" (nobody negotiates) and "press" (the subject runs one mechanical
negotiation round per movement phase, module-level :func:`run_press_round`).
This is the single-scenario probe in ``scripts/probe_negotiation.py`` at
full-game scale, and it takes the project's standing 1-vs-6 evaluation shape
(``configs/tournament/README.md``) rather than a symmetric 7-way toggle.

**Why one subject seat, and not all seven at once.** All seven seats share the
identical policy, and a single game's Sum-of-Squares shares always sum to
exactly 1 by construction (:func:`entente.eval.scoring.sos_scores`). Average
*every* seat's share across *every* game under a symmetric all-seats-press
toggle and the mean is **mathematically forced to equal exactly 1/7** in
both conditions, regardless of what happens in the games -- a tautology, not
a measurement, and one that reports itself as identical means to four decimal
places rather than as an error. Rotating one
subject seat's press capability while the other six never negotiate breaks
that symmetry: the subject's own mean share is not fixed by construction, so
a press-vs-no-press delta is a real, checkable quantity. ``rotate_seats``
(mirroring :class:`entente.search.config.LambdaSweepConfig`) rotates the
subject through all seven powers so no single power's structural advantage or
disadvantage confounds the result.

**The negotiation round is `StubProvider`-only.** The subject generates a message via
:class:`entente.negotiation.generate.StubProvider`, the chosen ally parses it
back via :func:`entente.negotiation.parse.parse_message` and reweights its
own anchor via :func:`entente.negotiation.parse.reweight_from_intent`. Only
`StubProvider`'s output is ever parsed anywhere in the prototype, so a
press-vs-no-press delta measured this way isolates the *reweighting
mechanism's* effect on outcomes and cannot be read as a claim about any
model's prose. `LocalLoraProvider`/`AnthropicProvider` are demonstrated separately
(`entente negotiate-demo`'s `.lora.json`/`.anthropic.json` transcripts),
never wired into this loop.

**Proposal policy** (:func:`propose`): the subject suggests both sides play
their own current anchor-favorite ("most likely") action -- the simplest
non-adversarial proposal that still exercises reweighting for real (the
ally's belief in its own top candidate sharpens), consistent with how the
scripted toy scenarios (`scenario.py`) also use hand-picked, non-adversarial
proposals to test the mechanism rather than model strategic bargaining.

Non-movement phases fall back to a seeded heuristic agent (mirrors
:class:`entente.agents.pikl_agent.PiklAgent`'s own default) -- piKL and press
are both defined only where movement decisions happen.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict, Field

from entente.agents.pikl_agent import BCCandidatePolicy, value_scores
from entente.agents.registry import build_agent
from entente.engine.game import Game
from entente.engine.phase import PhaseKind
from entente.eval.press_results import PressRecord
from entente.eval.scoring import sos_scores
from entente.negotiation.generate import StubProvider
from entente.negotiation.intents import Intent
from entente.negotiation.parse import parse_message, reweight_from_intent
from entente.nn.model import BCModel
from entente.search.candidates import CandidateSet, collect_candidates
from entente.search.config import PiklAgentConfig, PiklSearchConfig
from entente.search.pikl import PiklPlayer, run_pikl, sample_index
from entente.search.rollout import profile_utility_fn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.orders import GameOrder


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PressEvalConfig(_Frozen):
    """Describe a full press-vs-no-press run.

    Both conditions share ``agent``'s checkpoint and search knobs and run over
    the same seed range, so the only difference between them is the press.
    """

    agent: PiklAgentConfig
    games: int = Field(default=5, ge=1)
    seed_start: int = 0
    subject_seat: str = "FRANCE"
    rotate_seats: bool = False
    max_year: int = Field(default=1904, ge=1901)
    ruleset: str = "standard"
    map_name: str = "standard"
    output: str = "docs/results/press_vs_no_press_smoke.jsonl"


def collect_subgame(
    policy: BCCandidatePolicy, game: Game, search: PiklSearchConfig, rng: random.Random
) -> tuple[list[str], list[CandidateSet]]:
    """Collect every power-with-units' candidate set for this phase from one policy.

    The negotiation round and the piKL solve both read this same subgame, so a
    proposal's reweighting lands on the exact candidates piKL then searches
    over. :class:`entente.agents.pikl_agent.PiklAgent` runs the same loop
    internally; it is factored out here so press can splice in between
    collecting the candidates and solving over them.
    """
    powers: list[str] = []
    candidate_sets: list[CandidateSet] = []
    for power in game.powers:
        candidates = collect_candidates(
            policy,
            game,
            power,
            k=search.k,
            attempts=search.attempts,
            include_greedy=search.include_greedy,
            rng=rng,
        )
        if candidates.order_sets == ((),):  # power without units this phase
            continue
        powers.append(power)
        candidate_sets.append(candidates)
    return powers, candidate_sets


def _top_action(candidates: CandidateSet) -> tuple[object, ...]:
    best = max(range(len(candidates.anchor)), key=candidates.anchor.__getitem__)
    return candidates.order_sets[best]


def propose(proposer: str, ally: str, phase: str, by_power: dict[str, CandidateSet]) -> Intent:
    """Build the subject's proposal: both sides' current anchor-favourite action."""
    return Intent(
        proposer=proposer,
        ally=ally,
        phase=phase,
        proposer_action=_top_action(by_power[proposer]),  # type: ignore[arg-type]
        ally_action=_top_action(by_power[ally]),  # type: ignore[arg-type]
        rationale="proposing our most likely moves",
    )


def run_press_round(
    powers: Sequence[str],
    candidate_sets: Sequence[CandidateSet],
    subject: str,
    phase: str,
    rng: random.Random,
) -> list[CandidateSet]:
    """Run one mechanical negotiation round, initiated by ``subject`` alone.

    Only the subject negotiates, which is what keeps the press-vs-no-press
    delta from collapsing into a tautology. This is a no-op when ``subject`` is
    not among ``powers`` this phase -- eliminated, say -- or when fewer than two
    powers remain. Returns candidate sets in the same order as ``powers``; only
    the chosen ally's is ever modified.
    """
    if subject not in powers or len(powers) < 2:
        return list(candidate_sets)
    by_power = dict(zip(powers, candidate_sets, strict=True))
    others = [power for power in powers if power != subject]
    ally = others[rng.randrange(len(others))]
    intent = propose(subject, ally, phase, by_power)
    message = StubProvider().generate(intent)
    parsed = parse_message(message)
    bias = reweight_from_intent(by_power[ally], parsed)
    reweighted = CandidateSet(
        power=ally,
        order_sets=by_power[ally].order_sets,
        log_probs=by_power[ally].log_probs,
        anchor=tuple(bias),
    )
    return [reweighted if power == ally else by_power[power] for power in powers]


def play_game(
    model: BCModel,
    device: torch.device,
    seed: int,
    *,
    press: bool,
    subject: str,
    search: PiklSearchConfig,
    fallback_name: str,
    max_year: int,
    ruleset: str = "standard",
    map_name: str = "standard",
) -> PressRecord:
    """Play one full self-play game among seven piKL agents sharing ``model``.

    Returns **only** ``subject``'s outcome, one record per rotated seat as in
    :mod:`entente.eval.lambda_sweep`: the other six seats are the context the
    subject played against, not separately reported results.
    """
    game = Game.new(ruleset, map_name)
    policy = BCCandidatePolicy(
        model, device=device, temperature=search.temperature, top_p=search.top_p
    )
    fallback = build_agent(fallback_name, seed)

    while not game.is_over and game.year <= max_year:
        if game.phase_kind is PhaseKind.MOVEMENT:
            rng = random.Random(f"{seed}:{press}:{subject}:{game.phase}")
            powers, candidate_sets = collect_subgame(policy, game, search, rng)
            if press:
                candidate_sets = run_press_round(powers, candidate_sets, subject, game.phase, rng)
            if powers:
                oracle = profile_utility_fn(
                    game,
                    powers,
                    [cs.order_sets for cs in candidate_sets],
                    lambda states: value_scores(model, device, states),
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
                for i, power in enumerate(powers):
                    mean = result.mean_strategies[i]
                    if search.play == "argmax":
                        pick = max(range(len(mean)), key=mean.__getitem__)
                    else:
                        pick = sample_index(mean, rng)
                    orders: list[GameOrder] = list(candidate_sets[i].order_sets[pick])
                    if orders:
                        game.submit(power, orders)
        else:
            for power in game.powers:
                orders = list(fallback.orders(game, power))
                if orders:
                    game.submit(power, orders)
        game.adjudicate()

    counts = {power: game.supply_count(power) for power in game.powers}
    shares = sos_scores(counts, game.winner)
    return PressRecord(
        press=press,
        seat=subject,
        seed=seed,
        sos=shares[subject],
        supply_centers=counts[subject],
        survived=counts[subject] > 0,
    )


def play_games(
    model: BCModel,
    device: torch.device,
    seeds: Sequence[int],
    *,
    press: bool,
    subject: str,
    search: PiklSearchConfig,
    fallback_name: str,
    max_year: int,
    ruleset: str = "standard",
    map_name: str = "standard",
) -> list[PressRecord]:
    """:func:`play_game` over every seed, one condition, one subject seat."""
    return [
        play_game(
            model,
            device,
            seed,
            press=press,
            subject=subject,
            search=search,
            fallback_name=fallback_name,
            max_year=max_year,
            ruleset=ruleset,
            map_name=map_name,
        )
        for seed in seeds
    ]


def run_press_eval(cfg: PressEvalConfig) -> list[PressRecord]:
    """Run both conditions over the same seed range and return the combined records.

    With ``cfg.rotate_seats`` the subject rotates through every power, so no
    single seat's structural position confounds the delta. Write the records
    with :func:`entente.eval.press_results.write_press_jsonl`.
    """
    device = torch.device(cfg.agent.device)
    model = BCModel.load(cfg.agent.checkpoint, map_location=device).eval()
    seeds = range(cfg.seed_start, cfg.seed_start + cfg.games)
    subjects = list(model.board.powers) if cfg.rotate_seats else [cfg.subject_seat]
    records: list[PressRecord] = []
    for subject in subjects:
        for press in (False, True):
            records.extend(
                play_games(
                    model,
                    device,
                    seeds,
                    press=press,
                    subject=subject,
                    search=cfg.agent.search,
                    fallback_name=cfg.agent.fallback,
                    max_year=cfg.max_year,
                    ruleset=cfg.ruleset,
                    map_name=cfg.map_name,
                )
            )
    return records


__all__ = [
    "PressEvalConfig",
    "collect_subgame",
    "play_game",
    "play_games",
    "propose",
    "run_press_eval",
    "run_press_round",
]
