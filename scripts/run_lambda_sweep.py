r"""Run the piKL lambda sweep and write the strength/agreement curve JSONL.

The sweep is the search's headline ablation: lambda trades anchor fidelity
against equilibrium play, and the curve is what shows the trade (docs/pikl.tex
Sec. 7).

    uv run --group ml python scripts/run_lambda_sweep.py \
        --config configs/pikl/lambda_sweep.yaml [--set key=value ...]

Per lambda: piKL sits in one seat (or rotates through all seven) against six
opponents sampling from the same BC anchor; every game becomes one
:class:`entente.eval.lambda_sweep.SweepRecord` (SoS share, survival,
per-unit anchor agreement, optional held-out human agreement). The JSONL is
the artifact -- the tables regenerate from it alone via
``entente.eval.lambda_sweep.aggregate``/``render_table``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import fmean

import torch

from entente.agents.pikl_agent import (
    AnchorAgreement,
    PiklAgent,
    human_agreement,
    pikl_lockstep_policy,
)
from entente.agents.registry import build_agent
from entente.eval.lambda_sweep import (
    SweepRecord,
    aggregate,
    append_sweep_jsonl,
    completed_seats,
    keep_completed_seats,
    read_sweep_jsonl,
    render_table,
    write_sweep_jsonl,
)
from entente.eval.scoring import sos_scores, survived
from entente.nn.model import BCModel
from entente.search.config import LambdaSweepConfig
from entente.sim.vectorized import LockstepRunner, new_games
from entente.training.config import load_config


def _resume_state(
    output: Path, games_per_seat: int
) -> tuple[list[SweepRecord], set[tuple[float, str]], dict[float, float | None]]:
    """Load prior progress from ``output`` and reconcile it against a fresh run.

    Only fully completed ``(lam, seat)`` seats are kept, so a seat left
    half-written by a killed run is dropped rather than resumed from a partial
    state; the file is rewritten canonically if anything was dropped. Returns
    the kept records, the set of seats to skip, and the per-lambda
    human-agreement value to reuse instead of recomputing.
    """
    if not output.exists():
        return [], set(), {}
    raw = list(read_sweep_jsonl(output))
    kept = keep_completed_seats(raw, games_per_seat)
    if len(kept) != len(raw):
        print(f"resume: dropping {len(raw) - len(kept)} record(s) from incomplete seats")
        tmp = output.with_suffix(output.suffix + ".tmp")
        write_sweep_jsonl(kept, tmp)
        tmp.replace(output)  # atomic -- a crash here leaves the original intact
    done = completed_seats(kept, games_per_seat)
    agreement = {r.lam: r.human_agreement for r in kept if r.human_agreement is not None}
    if done:
        print(f"resume: {len(done)} (lambda, seat) seat(s) already complete, skipping them")
    return kept, done, agreement


def run_sweep(cfg: LambdaSweepConfig) -> list[SweepRecord]:
    """Run every lambda and seat the config asks for, resuming what is done."""
    device = torch.device(cfg.agent.device)
    model = BCModel.load(cfg.agent.checkpoint, map_location=device).eval()
    powers = model.board.powers
    seats = list(powers) if cfg.rotate_seats else [cfg.seat]
    output = Path(cfg.output)
    # a game that never terminates would hang the whole sweep; cap ticks well
    # above any real game's phase count (<=5 phases/year to the year cap).
    max_ticks = (cfg.max_year - 1900) * 15 + 30

    records, done_seats, agreement_by_lam = _resume_state(output, cfg.games)
    for lam_index, lam in enumerate(cfg.lambdas):
        pending = [(i, s) for i, s in enumerate(seats) if (lam, s) not in done_seats]
        if not pending:
            continue
        search = cfg.agent.search.model_copy(update={"lam": lam})
        agent = PiklAgent(cfg.agent.model_copy(update={"search": search}), model=model)
        human_rate: float | None = agreement_by_lam.get(lam)
        if cfg.agreement_shards is not None and lam not in agreement_by_lam:
            human_rate, human_units = human_agreement(
                agent,
                cfg.agreement_shards,
                rows=cfg.agreement_rows,
                seed=cfg.agent.seed,
                map_name=cfg.map_name,
                ruleset=cfg.ruleset,
            )
            print(f"lam={lam}: human agreement {human_rate} over {human_units} units")
        for seat_index, seat in pending:
            games = new_games(cfg.games, ruleset=cfg.ruleset, map_name=cfg.map_name)
            agreement = AnchorAgreement()
            # generator device must match the model's, or multinomial raises
            # on CUDA; identical to the default on CPU.
            generator = torch.Generator(device=device).manual_seed(
                cfg.agent.seed * 1_000_003 + lam_index * 1_009 + seat_index
            )
            policy = pikl_lockstep_policy(
                agent,
                pikl_power=seat,
                generator=generator,
                fallback=build_agent(cfg.agent.fallback, cfg.agent.seed),
                agreement=agreement,
            )
            LockstepRunner(games, policy, max_year=cfg.max_year).run(max_ticks=max_ticks)
            seat_records = [
                SweepRecord(
                    lam=lam,
                    seat=seat,
                    seed=game_index,
                    sos=sos_scores({p: game.supply_count(p) for p in powers}, game.winner)[seat],
                    supply_centers=game.supply_count(seat),
                    survived=survived({p: game.supply_count(p) for p in powers})[seat],
                    anchor_agreement=agreement.rate_for(game),
                    human_agreement=human_rate,
                )
                for game_index, game in enumerate(games)
            ]
            append_sweep_jsonl(seat_records, output)  # durable per-seat write
            records.extend(seat_records)
            print(f"lam={lam} seat={seat}: +{len(seat_records)} records (resumable)")
        lam_records = [r for r in records if r.lam == lam]
        print(
            f"lam={lam}: {len(lam_records)} games, mean SoS {fmean(r.sos for r in lam_records):.4f}"
        )
    return records


def main(argv: list[str] | None = None) -> int:
    """Run the sweep named by the config and write its JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="LambdaSweepConfig YAML")
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=None,
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (e.g. --set agent.search.k=4)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(LambdaSweepConfig, args.config, overrides=args.overrides or ())
    records = run_sweep(cfg)
    # run_sweep already wrote every record durably as it went; report and render.
    print(f"{len(records)} records in {cfg.output}")
    print(render_table(aggregate(records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
