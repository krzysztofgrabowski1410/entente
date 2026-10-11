# Tournament ladder -- the full evaluation matrix

The evaluation protocol is fixed in advance, so no result is chosen after seeing
the numbers: **1-vs-6** with a game-year cap and Sum-of-Squares scoring, a
**fixed seed set reused across agents**, >=200 games for the heuristic baselines
and >=100 for the search agents, bootstrap CIs and permutation tests, and
**every reported number comes from committed JSONL** in `docs/results/`.

This directory holds the configs that are runnable today; the checkpoint-gated
rungs (BC, piKL) list their exact entry point and the single blocking
dependency below. `1` is the lone rotated seat, `6` the six opponents; "each
direction" means the pair is run once with each agent as the `1` so no result
depends on the 1-vs-6 seating.

## Two harnesses, by design

`random` and `greedy` are seed-only agents (`agents/registry.py`) and run
through the **ProcessPool** path -- `entente tournament <config>.yaml`
(`sim/parallel.py`, Windows-spawn-safe). BC and piKL are **checkpoint-driven**
(a trained anchor policy + value net; their inputs -- checkpoint, k, N, lambda -- do
not fit the seed-only factory shape, see the `bc_agent`/`pikl_agent` module
docstrings) and run through the **lockstep** path (`sim/vectorized.py`) so
network inference batches across the seven seats in one forward pass. They are
therefore evaluated by their own scripts, not `entente tournament`. This split
is architectural, not incidental: a torch model pickled into every spawn worker
would be both slow and un-batched.

## The matrix

| 1 -> 6 | Games | Harness / entry point | Status | Output JSONL |
|---|---:|---|---|---|
| greedy -> 6x random | 210 | `entente tournament configs/tournament/main.yaml` | **DONE** (real) | `docs/results/greedy_vs_random.jsonl` |
| random -> 6x greedy | 210 | `entente tournament configs/tournament/random_vs_greedy.yaml` | **DONE** (real) | `docs/results/random_vs_greedy.jsonl` |
| BC -> 6x greedy | 210 | `entente.agents.bc_agent.bc_vs_greedy` (RTX 2060) | **DONE** (BC > greedy, CI-separated) | `docs/results/bc_vs_greedy.jsonl` |
| piKL(lambda) -> 6x BC | 105/lambda | `scripts/run_lambda_sweep.py --config configs/pikl/lambda_sweep.yaml` | **DONE** (4 lambdas, piKL(lambda*) > 1/7) | `docs/results/pikl_lambda_sweep.jsonl` |

The piKL sweep **is** the piKL-vs-BC rung of the ladder. At full size four
lambdas `{0, 0.3, 1, 3}` ran; lambda=inf is BC playing itself through the anchor
short-circuit, which hit a GPU-numerics crash on the 2060 and is omitted, its SoS
taken as the analytic 1/7 that construction forces. The `piKL(lambda*) > BC` read-out is
per-lambda CI vs the 1/7~0.143 symmetric baseline (setup.md, "piKL" Sec. ) and it
HOLDS: lambda*=0 SoS 0.3018 CI [0.2590, 0.3451] clears it. A smoke-scale wiring
run of this exact pipeline is retained at
`docs/results/pikl_lambda_sweep_smoke.jsonl`.

## What these rungs need

A **CUDA-trained BC checkpoint** (`checkpoints/bc/bc.pt`). The primary
Windows-native box has CPU-only torch and no CUDA GPU, so BC training -- and with
it every BC and piKL rung -- runs on a second machine with an RTX 2060 (6 GiB).
The full command chain, dataset to preprocess to `train_bc.py` to `bc_vs_greedy`
and `run_lambda_sweep.py`, is in `docs/setup.md` under "Behavioral cloning" and
"piKL". Nothing here is extrapolated: the numbers are the runs' output as
measured, including the held-out top-1 miss at 0.361 recorded in
`docs/results/bc_plan_acceptance.jsonl`.

## Reproduce the runnable rungs

```sh
uv run entente tournament configs/tournament/main.yaml             # greedy -> 6x random
uv run entente tournament configs/tournament/random_vs_greedy.yaml # random -> 6x greedy
```

Both are deterministic in the seed set (`{0..29}` x 7 seats = 210 games,
`max_year: 1910`); re-running reproduces the committed JSONL byte-for-byte
(modulo the platform's line endings).
