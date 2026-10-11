"""Prototype language-model negotiation on top of the search agent.

The prototype is isolated by construction, because a research prototype that
fails should be removable without touching anything that works: this package
imports only the public APIs of :mod:`entente.search` and
:mod:`entente.engine`, and no module outside :mod:`entente.negotiation` may
import it, which ``tests/negotiation/test_isolation.py`` enforces. Cutting it
is ``git rm -r src/entente/negotiation`` plus dropping the ``llm`` and
``finetune`` dependency groups and the negotiation scripts (run_press_eval,
train_dialogue_lora, probe_negotiation, preprocess_peskov); nothing else
references it.

Modules: :mod:`intents` (the serializable proposal schema), :mod:`generate`
(intent -> natural-language message, behind a ``Provider`` protocol),
:mod:`parse` (message -> intent -> anchor reweighting, via the public hook in
:func:`entente.search.pikl.reweight_anchor`), :mod:`scenario` (loads the
scripted JSON scenarios under ``scenarios/`` into piKL-ready inputs),
:mod:`cli` (the negotiate-demo command), :mod:`press_eval` (the
press-vs-no-press harness), :mod:`data` (the Peskov corpus + instruction
formatting), and :mod:`finetune` (the QLoRA dialogue fine-tune).
"""

from __future__ import annotations
