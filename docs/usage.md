# User guide

Step-by-step from a clean machine: install, run each entry point, understand
how input works today, and regenerate every committed artifact. Every command
here was executed on the reference box (Windows 11, i7-1355U, CPU-only);
wall-clocks are noted for the slow ones. GPU-only commands are labeled as
such and quoted from the evidence of the runs that produced them, never
faked.

## 1. Install

You need [uv](https://docs.astral.sh/uv/) and any Python >= 3.11. uv manages
the interpreter, the virtualenv, and dependencies.

```sh
uv sync            # engine + dev tools; the stdlib-only core needs nothing else
uv run pytest -q   # full suite: DATC + BATC x both engines, properties
```

The engine core is dependency-free. Heavier stacks live in optional dependency
groups, each added only when a step needs it:

| Group | Install | Enables |
|:--|:--|:--|
| (default) | `uv sync` | engine, CLI, tournaments, the SVG renderer |
| `ml` | `uv sync --group ml` | behavioral cloning + piKL agents (torch, pyarrow) |
| `viz` | `uv sync --group viz` | the demo GIF (pillow) |
| `llm` | `uv sync --group llm` | real-model negotiation transcripts (the `anthropic` client) |
| `finetune` | `uv sync --group finetune` | the QLoRA dialogue fine-tune (transformers, peft) |
| `oracle` | `uv sync --group oracle` | the dev-only external differential oracle |

### The `--system-certs` note (this machine only)

The reference dev box intercepts TLS, so uv's bundled certificate store
rejects PyPI. On that machine every uv command needs `--system-certs` (or set
`UV_SYSTEM_CERTS=1` once). CI runners have plain network access and do not need
it. Full details, plus the `git`/`curl` analogues for dataset downloads, are
in [docs/setup.md](setup.md).

## 2. The command-line entry points

All are registered under the `entente` console script. Runs are deterministic
in their seed.

### Play a full game (agents only)

```sh
uv run entente play --agents greedy --seed 11 --max-year 1905   # ~29 s, exit 0
```

Prints a phase-by-phase log and a final result. `--agents` picks one agent for
every power (`random` or `greedy`); `--seed` fixes the agent RNG; `--max-year`
caps the game (calendar-aware -- see below); `--out FILE` saves the finished
game as versioned JSON; `--map`/`--ruleset` default to `standard`.

### Adjudicate a saved game

```sh
uv run entente play --agents random --seed 42 --max-year 1903 --out game.json
uv run entente adjudicate game.json --out advanced.json          # ~17 s each
```

`adjudicate` loads a serialized game, resolves its current phase, prints the
log, and writes the advanced game back (in place, or to `--out`).

### Run a tournament

```sh
uv run entente tournament configs/tournament/main.yaml           # ~1 min 37 s (210 games)
```

Runs the 1-vs-6 or round-robin tournament in the YAML config, prints a
per-agent Sum-of-Squares / survival / solo table with bootstrap CIs, and (if
the config sets `output`) writes every game as JSONL. Override any config value
inline: `--set max_year=1905` (repeatable, dotted keys).

### The Beleriand demo

```sh
uv run entente beleriand-demo --seed 0                            # ~18 s
```

A full seeded Beleriand game to the Valar's intervention,
choreographed to parade the Layer-2 mechanics (mask-making and a handover,
knowledge grants, Glaurung's devastations, Ulmo's wrath, the countdown).
`--out` saves it as format-v2 JSON.

### The negotiation demo (offline stub)

```sh
uv run entente negotiate-demo src/entente/negotiation/scenarios/s1.json   # ~17 s
```

Runs the negotiation prototype on a scripted scenario: the stub provider generates
a press message, it is parsed back to an intent, and the intent reweights
piKL's candidate beliefs. Prints the per-power move-prediction shift and writes
a stub transcript. With `ANTHROPIC_API_KEY` set (and `--group llm`) it also
writes a real-model `*.anthropic.json` transcript; with a fine-tuned adapter
configured, a `*.lora.json` one (both generated, never parsed back).

## 3. How input works today

- **Agents, not humans, on the CLI.** `entente play`, `tournament`, and
  `beleriand-demo` are batch runners: every power is an agent (`random` or
  `greedy`), and there is **no interactive human order entry on the command
  line**. A human-play mode, on the command line or in a GUI, is a possible
  future feature, not a silent omission -- it is recorded as such here rather
  than implied.
- **Configs are YAML + `--set` overrides.** Tournaments, the piKL sweep, and
  the press eval read a typed YAML config (pydantic-validated, unknown keys
  rejected); any value is overridable inline with `--set dotted.key=value`
  (repeatable). See `configs/`.
- **Seeds** make every run reproducible: `--seed` on the CLI, `seed:` in the
  configs. **Agent names** come from the registry (`random`, `greedy`; `bc`
  and `pikl` are constructed from checkpoints, not registry names).

## 4. The research scripts (standard ruleset)

These live under `scripts/` and target the ML/search reproduction. The CPU
smoke paths run anywhere; the plan-scale runs need a CUDA GPU (WSL2 on the
reference box -- `torch.cuda.is_available()` is False on Windows-native, so
full training runs there). Exact GPU commands and the dataset mirror are in
[docs/setup.md](setup.md).

### Behavioral cloning

```sh
uv sync --group ml --system-certs
# CPU smoke: synthesize + preprocess a few games, then overfit to prove the loop learns
uv run python scripts/preprocess_paquette.py --out data/processed_synth --synthetic-games 500
uv run python scripts/train_bc.py --config src/entente/training/configs/bc/smoke.yaml
```

Verified on CPU this pass (40 synthetic games -> 2,800 rows in ~5 s; a 20-step
train writes a checkpoint in ~35 s, printing `cuda_available=False device=cpu`).
The `>=50%` held-out top-1 target needs the `base.yaml` config on a GPU --
see docs/setup.md, "Behavioral cloning" step 3, and the measured outcome in
`docs/results/bc_plan_acceptance.jsonl`.

### piKL lambda sweep

```sh
# CPU wiring check (tiny k/attempts/iterations, one game per lambda):
uv run python scripts/run_lambda_sweep.py --config configs/pikl/lambda_sweep.yaml \
  --set games=1 --set rotate_seats=false --set max_year=1902 \
  --set agent.search.k=2 --set agent.search.attempts=4 --set agent.search.iterations=4 \
  --set 'lambdas=[0.0, 1.0, .inf]' --set agent.checkpoint=checkpoints/bc_smoke/bc_smoke.pt \
  --set output=sweep_wiring.jsonl
```

Verified on CPU this pass (~16 s; runs the full candidates -> 1-ply rollouts ->
anchored-hedge pipeline and writes the regenerable table). The plan-scale sweep
needs the GPU BC checkpoint (docs/setup.md "piKL"); it ran on an RTX 2060 with
105 games for each of lambda in `{0, 0.3, 1, 3}` (lambda = inf is left out: its
games crashed on the GPU, and its score is the 1/7 its construction forces), and
its records are `docs/results/pikl_lambda_sweep.jsonl`.

### Press-vs-no-press evaluation

`scripts/run_press_eval.py` measures the reweighting mechanism's SoS delta at
full-game scale; like the sweep it needs a BC checkpoint, so the plan-scale run
is GPU-only. It ran on the same RTX 2060, 105 games with press and 105 without,
and its records are `docs/results/press_vs_no_press.jsonl` (the smoke path is
exercised by `tests/negotiation/`). The
dialogue fine-tune (`scripts/train_dialogue_lora.py`) has a CPU smoke path
(`--group finetune`) and a GPU-only real run (docs/setup.md "QLoRA dialogue
fine-tune").

## 5. Where outputs land

| Output | Location |
|:--|:--|
| Saved games (`--out`) | wherever you point `--out` (versioned JSON) |
| Tournament JSONL | the config's `output:` (e.g. `docs/results/greedy_vs_random.jsonl`) |
| BC checkpoint + metrics | the config's `out_dir` (default `checkpoints/bc/`) |
| Sweep / press JSONL | the config's `output:` under `docs/results/` |
| Negotiation transcripts | `src/entente/negotiation/transcripts/` |

`data/`, `checkpoints/`, and `wandb/` are git-ignored (code + small committed
results only).

## 6. Regenerating committed artifacts

Every committed artifact rebuilds from a script, deterministically:

```sh
# the README demo GIF (byte-identical for a seed; encoder is determinism-tested)
uv run --group viz python scripts/make_demo_gif.py

# the typeset documents -- the two specifications, LaTeX -> PDF
# (needs a TeX distribution with latexmk + pdflatex; fails on an unresolved reference)
uv run python scripts/build_specs.py                                # docs/*.pdf

# the tournament results tables (the two baseline directions)
uv run entente tournament configs/tournament/main.yaml
uv run entente tournament configs/tournament/random_vs_greedy.yaml
```

The README numbers come from the committed JSONL in `docs/results/`, which the
commands above regenerate; this guide documents the commands, not the numbers
they produce.
