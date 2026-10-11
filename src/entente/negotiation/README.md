# `entente.negotiation` -- LLM negotiation prototype

A piKL candidate proposal becomes a structured **intent**, an LLM (or a
deterministic stub) turns that into a press message, an incoming message
parses back into an intent, and the intent **reweights** a power's piKL
candidate distribution. This is a prototype built to answer one question --
does conditioning on a proposal's stated intent improve prediction of that
power's next move? -- not a re-implementation of CICERO's dialogue-conditioned
planning loop.

## Capabilities

- **`intents.py`** -- `Intent`: a proposer/ally pair, a phase, and the two
  powers' proposed order-sets, serializable to/from dict and JSON via the
  engine's own order notation (`entente.engine.orders.format_order`/
  `parse_order`).
- **`generate.py`** -- `Provider` protocol; `StubProvider` (deterministic
  template, fully offline), `AnthropicProvider` (real model, gated on
  `ANTHROPIC_API_KEY`, deps in the `llm` group), and `LocalLoraProvider`
  (a locally fine-tuned LoRA adapter, gated on the adapter directory
  existing, deps in the `finetune` group -- see `data/` and `finetune/` below).
- **`data/`** -- `peskov.py` (reader for the Peskov et al. 2020 "It Takes Two
  to Lie" CC-BY 4.0 corpus) and `instructions.py` (conversations -> stance-
  conditioned instruction-tuning pairs; see the module docstring for why this
  is *not* the engine-grounded `Intent` schema).
- **`finetune/`** -- `config.py` (`DialogueLoraConfig`), `data.py` (tokenization
  + prompt-masking), `train.py` (the QLoRA/LoRA training loop
  `LocalLoraProvider` loads back from). CPU smoke config included; the real
  ~3B run is WSL2/CUDA-only (`docs/setup.md`); its committed loss curve is
  `docs/results/dialogue_lora_curve.jsonl`.
- **`parse.py`** -- `parse_message` reads `StubProvider`'s fixed grammar back
  into an `Intent`; `reweight_from_intent` matches the proposed action
  against a power's `entente.search.candidates.CandidateSet` and reweights
  its anchor through the one hook this prototype adds to piKL itself:
  `entente.search.pikl.reweight_anchor(anchor, bias)` -- a game-agnostic,
  opaque-numbers tilt-and-renormalize. piKL's core algorithm never sees an
  `Intent` or an `Order`.
- **`scenario.py`** -- loads a scripted JSON scenario into a two-power piKL
  subgame and runs it twice (baseline vs. intent-conditioned) over a
  torch-free, engine-grounded toy utility (see Limitations).
- **`cli.py`** -- `entente negotiate-demo <scenario.json>` runs one scenario
  end to end and writes a transcript under `transcripts/`. Wired into
  `entente.cli` through `importlib`, not a static import -- see below.
- **`press_eval.py`** -- the full-game press-vs-no-press evaluation:
  seven piKL agents share one BC checkpoint; a rotating **subject** seat's
  press capability toggles between conditions over the same seeds. See the
  module docstring for why a single subject seat (not a fully symmetric
  toggle) is what makes the SoS delta a real, non-tautological quantity.
- **`scripts/probe_negotiation.py`** (repo root) -- the quantitative probe:
  runs every committed scenario and writes `docs/results/negotiation_probe.jsonl`.
- **`scripts/run_press_eval.py`** (repo root) -- the full-game counterpart:
  runs `press_eval.py` from a config and writes
  `docs/results/press_vs_no_press*.jsonl`.

## Architecture: the one-way dependency rule

`entente.negotiation` may import the public APIs of a fixed, documented set of
core modules -- `entente.engine`, `entente.search`, `entente.agents`,
`entente.eval`, and `entente.training`'s generic config and tracking
utilities; **no module outside `entente.negotiation` may import it**.
`tests/negotiation/test_isolation.py` enforces this by grepping `src/entente`
for the import. The direction is what matters: a research prototype that does
not work out has to be removable without touching anything that does, so
cutting it is exactly `git rm -r src/entente/negotiation` plus dropping the
`llm` group -- nothing else references it, by construction.

The one deliberate wrinkle: `entente negotiate-demo` is a CLI subcommand, so
*something* in `entente.cli` has to reach this package. `entente/cli.py`
does it with `importlib.import_module("entente.negotiation.cli")` inside a
`try/except ModuleNotFoundError` -- a string, not an `import`/`from`
statement, so the isolation test's static grep doesn't (and shouldn't) flag
it, and deleting `negotiation/` makes the subcommand silently disappear
instead of leaving a dangling import. See `entente/cli.py`'s
`_register_optional_commands` and `entente/negotiation/cli.py`'s module
docstring.

## Usage

```sh
uv run entente negotiate-demo src/entente/negotiation/scenarios/s1.json
uv run python scripts/probe_negotiation.py
```

Both run fully offline (`StubProvider`, hand-authored candidates, real
engine adjudication for the utility -- no `ml`/`llm`/`finetune` group needed).
If `ANTHROPIC_API_KEY` is set, `negotiate-demo` additionally generates and
writes a `<scenario>.anthropic.json` transcript labeled `"provider":
"anthropic"`, using `AnthropicProvider` (deps: `uv sync --group llm
--system-certs`; model defaults to `claude-opus-4-8`, overridable via
`ENTENTE_ANTHROPIC_MODEL`). If `ENTENTE_DIALOGUE_LORA_DIR` points at a
trained adapter directory, it likewise writes a `<scenario>.lora.json`
transcript labeled `"provider": "lora"`, using `LocalLoraProvider` (deps:
`uv sync --group finetune --system-certs`; see `docs/setup.md` Sec. "QLoRA
dialogue fine-tune" for how to produce an adapter). Both real-model
transcripts are generated only, never parsed back -- see Limitations.

## Limitations (read before citing the probe numbers)

This is a small, honest prototype, not a scaled system. Specifically:

1. **Candidates are hand-authored, not BC-sampled.** A real piKL subgame
   (docs/pikl.tex Sec. 5) draws candidates from a trained behavioral-cloning
   policy via `entente.search.candidates.collect_candidates`. The scenarios
   here construct `CandidateSet` directly from a handful of verified-legal
   order-sets per power with hand-picked anchor probabilities, so the demo
   and probe never need a checkpoint, GPU, or the `ml` group. The mechanism
   under test (reweighting) is real; the anchor it starts from is not a
   learned prior.
2. **The utility oracle is order-success, not a value network.** piKL's
   production utility (docs/pikl.tex Sec. 6) scores a value net's prediction of
   the successor state. `scenario.make_utility_fn` instead scores the
   fraction of a power's own orders that reached their written effect
   (a move arrived, a hold wasn't displaced), computed from one real
   adjudication (`entente.search.rollout.branch_game`) per joint profile.
   It differentiates joint actions the same way a value net would need to
   (bounces depend on what the other power plays) without needing torch.
   Every scenario's candidates are Hold/Move only -- no Support -- so no
   candidate combination can produce a dislodgement, keeping the successor
   state simple to score correctly.
3. **`parse_message` parses one fixed grammar, not natural language.** It
   reads exactly the template `StubProvider` emits. This is a deliberate
   scope cut, not an oversight: this prototype's job is to test whether
   *correctly parsed* intent conditioning helps, not to build an NLU system.
   A message from `AnthropicProvider` or `LocalLoraProvider` (both free-form)
   is not fed back through `parse_message` anywhere in this codebase -- the
   full-game press-vs-no-press harness (`press_eval.py`) works under the same
   restriction, so no measured number rests on a model's prose.
4. **The reweighting boost is a fixed multiplier (`DEFAULT_BOOST = 4.0`),
   not a learned or calibrated trust weight.** A real system would want the
   boost to depend on the proposer's track record of following through --
   there is no such model here.
5. **The probe is 3 scripted scenarios.** Two are truthful (the proposer's
   announced move matches its scripted true move); one is deceptive
   (announced != true), included deliberately: conditioning on a lie
   *worsens* prediction of the true move, which is the honest complement to
   the two truthful results. n=3 supports no statistical claim -- the
   committed numbers are `docs/results/negotiation_probe.jsonl`.
6. **`AnthropicProvider.generate` and `LocalLoraProvider.generate` are not
   exercised by the default test suite or by CI** -- they need
   `ANTHROPIC_API_KEY`/the `llm` group, and a trained adapter/the `finetune`
   group plus network access, respectively. Everything else (prompt
   construction, construction-time validation) is unit-tested without either
   package installed; `LocalLoraProvider.generate` itself is covered by
   `finetune`-marked tests against a real (if tiny) trained adapter.
7. **The Peskov fine-tuning data conditions on a stance, not an `Intent`.**
   The corpus has no order data aligned to its messages (real players'
   actual moves were never captured alongside what they said), so
   `entente.negotiation.data.instructions` conditions on speaker/receiver/
   phase/truthfulness -- not a fabricated engine-grounded proposal. See that
   module's docstring for the full reasoning.

## References

- `docs/pikl.tex` -- the search this module conditions.
