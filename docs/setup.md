# Development environment setup

## Requirements

- [uv](https://docs.astral.sh/uv/) -- manages Python, the venv, and dependencies.
  Any Python >=3.11 works locally; CI pins 3.11 and 3.12.

```sh
uv sync            # create .venv + install dev tools
uv run pytest
uv run ruff check .
uv run mypy src
uv run pre-commit install   # once, to enable the commit hooks
```

## Machine-specific notes (primary dev box, Windows 11)

- **TLS interception:** the network stack on this machine intercepts TLS, so
  uv's bundled certificate store rejects PyPI. Use the system store:
  `uv sync --system-certs`, or set it once per machine:
  `setx UV_SYSTEM_CERTS 1` (new shells pick it up). Not committed to project
  config because CI runners don't need it.
- **Training environment:** WSL2 Ubuntu 24.04 + CUDA is the
  supported path for PyTorch on this machine. The engine itself is
  cross-platform -- Windows stays in the CI matrix. GPU model/VRAM: this
  primary box has no CUDA-capable GPU (Iris Xe integrated), so plan-scale
  training is offloaded to a second machine -- an **NVIDIA GeForce
  RTX 2060 (6.0 GiB)**.
- **Installing a dependency from git (e.g. the `oracle` group's AGPL
  `diplomacy` package):** the TLS-intercepting network stack above is uv's
  problem to work around (`--system-certs`); a `git+https://` dependency is
  fetched by the system `git` binary instead, which has its own, separate TLS
  handling and hits the same interception with `SSL certificate problem:
  unable to get local issuer certificate`. Point that one command's git
  operations at the Windows certificate store (schannel already trusts
  whatever root CA the interception uses, same as uv's `--system-certs`)
  without touching global git config: prefix the command with
  `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.sslbackend GIT_CONFIG_VALUE_0=schannel`
  (bash) or the PowerShell equivalent
  (`$env:GIT_CONFIG_COUNT=1; $env:GIT_CONFIG_KEY_0='http.sslbackend'; $env:GIT_CONFIG_VALUE_0='schannel'`).
  This is a per-invocation override -- nothing is written to any `.git/config`
  or `~/.gitconfig`. Not needed in CI: GitHub Actions runners have plain,
  unintercepted network access.

## One-time setup: the DATC badge gist

The `datc` job in `ci.yml` computes the DATC pass/total count and pushes a
[shields.io endpoint JSON](https://shields.io/badges/endpoint-badge) to a
GitHub Gist via `schneegans/dynamic-badges-action`, so the README can embed a
live "DATC 352/352"-style badge without committing a static image. The badge
step no-ops (skips cleanly, no job failure) until this is set up -- do this
once, whenever it's convenient:

1. Create a new [GitHub Gist](https://gist.github.com/) (public, so
   shields.io can read it) with one file named `datc-badge.json` containing
   a placeholder shields endpoint document:
   ```json
   {"schemaVersion": 1, "label": "DATC", "message": "pending", "color": "lightgrey"}
   ```
   Note the gist ID from its URL (`gist.github.com/<user>/<GIST_ID>`).
2. Create a
   [fine-grained personal access token](https://github.com/settings/personal-access-tokens)
   scoped to **Gists: Read and write** only (no repo access needed), no
   expiry shorter than you're willing to redo this for.
3. In the repo's Settings -> Secrets and variables -> Actions:
   - add secret `GIST_SECRET` = the token from step 2;
   - add variable `DATC_GIST_ID` = the gist ID from step 1.
4. Push a commit through `ci.yml` once -- the `datc` job's badge step now
   updates the gist on every run. The live badge URL is
   `https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/<user>/<GIST_ID>/raw/datc-badge.json`;
   wiring it into the README is a separate step.

## Behavioral cloning: dataset, preprocessing, and training

The BC stack lives in the `ml` dependency group (torch, numpy, pyarrow, wandb),
which CI never installs -- the engine's cross-platform test matrix stays torch-free.
Install it locally to run the `ml`-marked tests and training:

```sh
uv sync --group ml --system-certs
```

### 1. Mirror the Paquette dataset

The no-press dump (156,468 games, ~2.24 GB JSONL zip) is hosted on a single
personal MinIO bucket -- `s3-public.billovia.com` (bucket `ppaquette.public`).
Place it under the **main checkout's** `data/` so
every worktree shares it:

```
C:\Users\krisi\repository\diplomacy\data\
```

> **The object key.** The bucket refuses listing (403), so only the full URL
> reaches the file. Verified by HEAD request: `200 OK`, `content-length:
> 2236684360` (2.24 GB), last modified 2025-11-16. Only the
> download + SHA-256 record remain. On this machine `curl`/`git` use the
> Windows certificate store and its revocation check rejects this host -- add
> `--ssl-no-revoke` (curl) alongside the usual `--system-certs` (uv):

```sh
# from the main checkout's data/ dir:
curl --ssl-no-revoke -L -o paquette_no_press.jsonl.zip \
  "https://s3-public.billovia.com/diplomacy/benchmarks/datasets/diplomacy-dataset.zip"

# record the SHA-256 so the mirror is verifiable (fill it in below):
sha256sum paquette_no_press.jsonl.zip                    # Git Bash / WSL2
certutil -hashfile paquette_no_press.jsonl.zip SHA256    # PowerShell
```

- **SHA-256 of the 2.24 GB zip:** c5d2ba695a16b752bc8e0c4f54fa6aea0926f574d2f6cc22f7388fe14f8332de
- **Restriction:** research/offline use only; may not power bots on Diplomacy
  websites without player consent (see `data/README.md`).

Without the download, the whole pipeline still runs on synthetic games (engine
self-play in the same schema -- the greedy-teacher fallback): pass no `--input`
to the preprocessing script (below).

### 2. Preprocess to Parquet shards

```sh
# real data (a 20k-game subset), from any worktree:
uv run --group ml python scripts/preprocess_paquette.py \
  --input "C:/Users/krisi/repository/diplomacy/data/paquette_no_press.jsonl.zip" \
  --out   "C:/Users/krisi/repository/diplomacy/data/processed" \
  --max-games 20000

# synthetic dry run (no download needed):
uv run --group ml python scripts/preprocess_paquette.py \
  --out "C:/Users/krisi/repository/diplomacy/data/processed_synth" --synthetic-games 500
```

### 3. Train

`torch.cuda.is_available()` is **False** on this Windows-native box (the PyPI
torch wheel is CPU-only), so full training runs in
**WSL2 Ubuntu + CUDA**. GPU model and VRAM of the second machine
the plan-scale run used:
**NVIDIA GeForce RTX 2060, 6.0 GiB** (`cuda_check`: `True NVIDIA GeForce RTX
2060 6.0 GiB`). The exact WSL2 commands:

```bash
# inside WSL2 Ubuntu, in the repo (or a worktree):
uv sync --group ml
uv run python scripts/preprocess_paquette.py \
  --input data/paquette_no_press.jsonl.zip --out data/processed --max-games 20000
uv run python scripts/train_bc.py --config src/entente/training/configs/bc/base.yaml \
  --set data.shards_dir=data/processed
# held-out per-unit top-1 and the BC-vs-greedy tournament print at the end;
# the JSONL metrics stream and checkpoint land under the config's out_dir.
```

The CPU smoke config (`configs/bc/smoke.yaml`) overfits a few synthetic games and
runs anywhere (`pytest -m "ml and slow"` exercises it); it exists to prove the
loop learns, not to hit the >=50% top-1 target -- that needs the GPU run above.

## piKL: the lambda sweep and the piKL > BC evaluation

The sweep (docs/pikl.tex Sec. 7) needs a trained BC checkpoint as anchor + value
net. Wiring is verified on CPU everywhere (`tests/agents/
test_lambda_sweep_smoke.py`, and the committed smoke-scale run in
`docs/results/pikl_lambda_sweep_smoke.jsonl`); the plan-scale numbers need
WSL2/CUDA and the BC base run above:

```bash
# inside WSL2 Ubuntu, after the BC base training run has produced
# checkpoints/bc/bc.pt (paths are the committed config defaults):
uv sync --group ml
# full sweep: lambda in {0, 0.3, 1, 3, inf}, 7 seats x 15 games = 105 games
# per lambda vs. six BC anchors, plus held-out human agreement per lambda.
# Point agreement_shards at a HELD-OUT shard directory (preprocess a game
# range disjoint from training, e.g. --max-games with a different slice).
uv run python scripts/run_lambda_sweep.py --config configs/pikl/lambda_sweep.yaml \
  --set agreement_shards=data/processed_holdout
# the JSONL lands at docs/results/pikl_lambda_sweep.jsonl; the printed table
# regenerates any time via entente.eval.lambda_sweep.aggregate/render_table.
```

The `piKL(lambda*) > BC` acceptance read-out: every sweep seat
plays against six BC opponents, so BC's expected share is 1/7 ~ 0.1429 by
symmetry; piKL(lambda*) beats BC when its per-lambda 95% bootstrap CI (in the
table above) lies above 1/7. GPU sizing: the search adds ~`7*(1 + attempts)`
policy decodes and `iterations` batched value calls per movement decision on
top of BC play -- plan-scale (k=12, N=128) fits comfortably wherever the BC
base config trains.

## Visualization: SVG boards and the demo GIF

`entente.viz.svg` is stdlib-only and works in the default environment; the GIF
rasterizer needs Pillow from the `viz` group:

```sh
uv sync --group viz --system-certs
# regenerate the committed README demo (byte-identical for a given seed --
# encoder determinism is pinned by tests/viz/test_gif.py):
uv run --group viz python scripts/make_demo_gif.py
# knobs: --seed --agent {greedy,random} --max-year --size --duration-ms --out
```

## The typeset documents: LaTeX

The two specifications (`docs/adjudication.tex`, `docs/pikl.tex`) build with
any TeX distribution that ships
`latexmk` and `pdflatex` (MiKTeX or TeX Live); the packages they use are
amsmath, amsthm, mathtools, booktabs, tabularx, longtable, enumitem, hyperref,
cleveref, microtype, titlesec, lmodern, geometry, graphicx and xcolor:

```sh
uv run python scripts/build_specs.py        # both PDFs; --keep-aux leaves the logs
```

Nothing in the test suite needs TeX; the committed PDFs are rebuilt in the
same commit as any source change. On this machine MiKTeX's on-the-fly package
installer runs into the TLS interception above, so a build that stalls is a
missing package rather than a hang: install it once from the MiKTeX console.

Map centroids are data (`"centroids"` in `standard.json`, integrity-tested in
`tests/viz/test_svg.py`); only the standard map ships them today -- rendering a
variant map means adding a centroids block and the tests pass judgment on the
layout (completeness, bounds, separation, nearest-marker-is-a-neighbour).

## LLM negotiation: real-model transcripts (optional)

The committed scenario transcripts are stub-generated (deterministic,
offline). The real-model demonstration is **closed**: the project's own
fine-tuned Provider generated the committed
`s{1,2,3}.lora.json` transcripts (see "QLoRA dialogue fine-tune" below). The
The `AnthropicProvider` route stays possible if an `ANTHROPIC_API_KEY` ever exists:

```sh
uv sync --group llm --system-certs
# per scenario (s1, s2, s3):
ANTHROPIC_API_KEY=... uv run entente negotiate-demo src/entente/negotiation/scenarios/s1.json
```

The default model is `claude-opus-4-8`; override via `ENTENTE_ANTHROPIC_MODEL`
(see `src/entente/negotiation/generate.py`).

## Negotiation dialogue data: the Peskov corpus

The dialogue fine-tune's training data is the Peskov et al. (2020) "It Takes
Two to Lie" corpus, CC-BY 4.0 --
17,289 messages, 246 real-game dialogues, ~3.2 MB total across three JSONL
files. Small enough to mirror directly (unlike the 2.24 GB Paquette dump);
full citation, SHA-256, and the CC-BY attribution requirement are in
`data/README.md`. Download into the **main checkout's** `data/` (shared
across worktrees, same convention as Paquette):

```sh
mkdir -p data/peskov
curl --ssl-no-revoke -L -o data/peskov/train.jsonl \
  "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/train.jsonl"
curl --ssl-no-revoke -L -o data/peskov/validation.jsonl \
  "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/validation.jsonl"
curl --ssl-no-revoke -L -o data/peskov/test.jsonl \
  "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/master/data/test.jsonl"
```

Preprocess each split to stance-conditioned instruction pairs (stdlib-only,
no `ml`/`finetune` group needed):

```sh
uv run python scripts/preprocess_peskov.py \
  --input "C:/Users/krisi/repository/diplomacy/data/peskov/train.jsonl" \
  --out   "C:/Users/krisi/repository/diplomacy/data/peskov_instructions/train.jsonl"
uv run python scripts/preprocess_peskov.py \
  --input "C:/Users/krisi/repository/diplomacy/data/peskov/validation.jsonl" \
  --out   "C:/Users/krisi/repository/diplomacy/data/peskov_instructions/validation.jsonl"
```

## QLoRA dialogue fine-tune (the secondary route)

```sh
uv sync --group finetune --system-certs
```

installs `transformers`/`peft`/`accelerate` (+`truststore`, needed to work
around this machine's TLS interception for `huggingface_hub` downloads --
verified directly: `AutoTokenizer.from_pretrained` fails with a wrapped SSL
error without it, exactly the class of problem `uv --system-certs`/
`curl --ssl-no-revoke` solve elsewhere on this box). `bitsandbytes` (4-bit
quantization) is restricted to Linux in the dependency group, so this
install never attempts it on Windows.

**CPU smoke** (proves the loop + `LocalLoraProvider` round-trip; no download
of the real model, no GPU, well under a minute):

```sh
uv run --group finetune python scripts/train_dialogue_lora.py \
  --config src/entente/negotiation/finetune/configs/smoke.yaml
```

**Real run** (`HuggingFaceTB/SmolLM3-3B`, Apache-2.0/ungated, 4-bit QLoRA):
needs a CUDA device (`load_in_4bit` raises loudly off-GPU rather than
silently loading full precision) -- WSL2 Ubuntu on this machine, whenever
its `nvidia-smi`/`torch.cuda.is_available()` check (docs/setup.md's
"Behavioral cloning" Sec. 3 pattern) finds a usable GPU:

```bash
# inside WSL2 Ubuntu, in a worktree, after preprocessing every
# split (see "Negotiation dialogue data" above):
uv sync --group finetune
uv run python scripts/train_dialogue_lora.py \
  --config src/entente/negotiation/finetune/configs/base.yaml
# writes the adapter under checkpoints/dialogue_lora/dialogue_lora/, loadable
# by entente.negotiation.generate.LocalLoraProvider(adapter_dir=...).
```

**Status (2026-07-22): DONE** -- executed on the second machine (RTX 2060
6 GiB, WSL2 Ubuntu; this box still has no CUDA): 1,000 steps, final loss
3.2935, curve committed at `docs/results/dialogue_lora_curve.jsonl`, adapter
kept git-ignored beside the training worktree. The plan-scale press-vs-no-press
evaluation ran in the same chain (`docs/results/press_vs_no_press.jsonl`).

## Reproducibility audit (release ritual)

The v1.0.0 acceptance run, from nothing to a verified tree -- target < 30
minutes on the reference laptop:

```sh
git clone <repo> && cd <repo>                    # or a fresh worktree
cd ..\entente-audit
uv sync --system-certs
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run pytest --cov=entente --cov-report=term -q
uv run entente play --agents greedy --seed 11 --max-year 1905
uv run entente tournament configs/tournament/main.yaml
```
