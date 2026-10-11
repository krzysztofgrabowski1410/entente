"""Define the QLoRA dialogue fine-tune's configuration schema.

The models are pydantic v2 over the project's shared YAML and ``--set`` loader
(:mod:`entente.training.config`). Only that module's generic, torch-free
``load_yaml``, ``apply_overrides`` and ``load_config`` are used, and it never
imports ``entente.negotiation``, so the dependency stays one-way and the
prototype remains cuttable.

Torch-free by design (like :mod:`entente.training.config` itself): this
module loads and validates in the default env, so ``--help``/config errors
surface without the ``finetune`` group installed. The actual base-model load
(:mod:`entente.negotiation.finetune.train`) is where torch/transformers/peft
enter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LoraConfig(_Frozen):
    """Fix the LoRA adapter's geometry.

    The defaults target the usual attention projections. ``target_modules``
    depends on the base model's architecture, so each config names the
    projections of the model it trains.
    """

    r: int = Field(default=8, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


class DialogueLoraOptimConfig(_Frozen):
    """Set the optimizer and sequence-length knobs.

    The defaults are small on purpose: this is a short instruction-tuning run
    over a few thousand short messages, not a pretraining-scale schedule.
    """

    lr: float = Field(default=2e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    warmup_steps: int = Field(default=10, ge=0)
    max_steps: int = Field(default=200, ge=1)
    batch_size: int = Field(default=4, ge=1)
    grad_clip: float = Field(default=1.0, ge=0.0)
    max_seq_length: int = Field(default=256, ge=8)


class DialogueLoraConfig(_Frozen):
    """Describe a full QLoRA dialogue fine-tune run.

    ``load_in_4bit`` selects real QLoRA (bitsandbytes NF4 quantization,
    CUDA-only); the CPU smoke path trains a plain (non-quantized) LoRA
    adapter over a tiny model instead. The training *loop* is identical
    either way -- only the base model's load path differs
    (:func:`entente.negotiation.finetune.train.load_base_model`), so the
    smoke config is a real (if tiny) exercise of the same code the real run
    uses, not a separate mock path.
    """

    base_model: str = "HuggingFaceTB/SmolLM3-3B"
    load_in_4bit: bool = False
    lora: LoraConfig = LoraConfig()
    optim: DialogueLoraOptimConfig = DialogueLoraOptimConfig()
    data_path: str = "data/peskov_instructions/train.jsonl"
    seed: int = 0
    out_dir: str = "checkpoints/dialogue_lora"
    run_name: str = "dialogue_lora"
    device: str = "auto"
    log_every: int = Field(default=10, ge=1)


__all__ = ["DialogueLoraConfig", "DialogueLoraOptimConfig", "LoraConfig"]
