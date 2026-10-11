"""Run the QLoRA (or plain LoRA) dialogue fine-tuning loop.

:func:`load_base_model` is the one seam the real (4-bit, CUDA) and CPU-smoke
paths share: ``cfg.load_in_4bit`` selects ``bitsandbytes`` NF4 quantization
(GPU-only -- raises loudly on CPU rather than silently loading full
precision), otherwise a plain bf16-on-CUDA/fp32-on-CPU load. Everything after
that -- LoRA wrapping, the training loop, checkpoint save -- is identical
either way, so the CPU smoke config (a ~100k-parameter test model) exercises
the *same code* the real ~3B run does, not a separate mock.

``truststore.inject_into_ssl()`` works around this machine's TLS-intercepting
network stack for ``huggingface_hub``'s downloads (its HTTP client does not
pick up the OS certificate store the way ``uv --system-certs``/
``curl --ssl-no-revoke`` do elsewhere in this project -- verified directly:
``AutoTokenizer.from_pretrained`` fails with a wrapped SSL error without it).
Idempotent and safe to call unconditionally; not needed at all once a model
is already cached locally, but harmless either way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import truststore
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from entente.negotiation.data.instructions import read_instruction_jsonl
from entente.negotiation.finetune.config import DialogueLoraConfig
from entente.negotiation.finetune.data import PromptCompletionDataset, collate
from entente.training.tracking import MetricTracker

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class FinetuneResult:
    """Summarize a finished run.

    The full curve is in the JSONL metrics file next to the adapter, as in
    :class:`entente.training.bc.TrainResult`.
    """

    steps: int
    final_loss: float
    adapter_dir: Path
    metrics_path: Path


def resolve_device(spec: str) -> torch.device:
    """Resolve a device spec, where ``auto`` means CUDA if available and CPU otherwise.

    Any other spec is taken verbatim, as in
    :func:`entente.training.bc.resolve_device`.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_base_model(
    cfg: DialogueLoraConfig, device: torch.device
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load ``cfg.base_model`` and its tokenizer.

    The load is 4-bit only under ``load_in_4bit`` and plain otherwise. A pad
    token is guaranteed on the way out, because several small causal-LM
    checkpoints -- the CPU smoke model among them -- ship without one.
    """
    truststore.inject_into_ssl()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError(
                "load_in_4bit requires a CUDA device (bitsandbytes has no CPU kernel); "
                "use the CPU smoke config (load_in_4bit: false) off-GPU"
            )
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, quantization_config=quant_config, device_map={"": device.index or 0}
        )
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype).to(device)
    return model, tokenizer


def train(cfg: DialogueLoraConfig) -> FinetuneResult:
    """Run one dialogue fine-tune as ``cfg`` describes it.

    The adapter and its tokenizer are written under
    ``cfg.out_dir/cfg.run_name`` through ``peft``'s own ``save_pretrained``,
    which is exactly the layout
    :class:`entente.negotiation.generate.LocalLoraProvider` loads back.
    """
    _seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model, tokenizer = load_base_model(cfg, device)

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.train()

    examples = list(read_instruction_jsonl(cfg.data_path))
    dataset = PromptCompletionDataset(examples, tokenizer, max_seq_length=cfg.optim.max_seq_length)
    pad_id = tokenizer.pad_token_id
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, pad_token_id=pad_id),
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )

    tracker = MetricTracker(
        cfg.out_dir, cfg.run_name, config=cfg.model_dump(), wandb_project="entente-dialogue-lora"
    )
    step = 0
    last_loss = 0.0
    try:
        while step < cfg.optim.max_steps:
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                loss = output.loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                opt.step()
                step += 1
                last_loss = float(loss.item())
                if step % cfg.log_every == 0:
                    tracker.log(step, {"loss": last_loss})
                    print(f"step {step}: loss={last_loss:.4f}")
                if step >= cfg.optim.max_steps:
                    break
    finally:
        tracker.close()

    adapter_dir = Path(cfg.out_dir) / cfg.run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return FinetuneResult(
        steps=step, final_loss=last_loss, adapter_dir=adapter_dir, metrics_path=tracker.path
    )


__all__ = ["FinetuneResult", "load_base_model", "resolve_device", "train"]
