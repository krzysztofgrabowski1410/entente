"""Record run metrics to a local JSONL sink and, optionally, to Weights & Biases.

Every reported figure has to regenerate offline from committed files, which is
why the local sink is the primary one and the hosted service is the extra.

The JSONL sink is *always* written -- every run leaves a committed-quality curve
on disk with no account or network. Weights & Biases is opt-in and best-effort:
it activates only when ``WANDB_API_KEY`` is set and the package imports, and any
failure there degrades to JSONL-only rather than crashing a training run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class MetricTracker:
    """Append-only JSONL metrics, plus optional wandb mirroring."""

    def __init__(
        self,
        out_dir: str | Path,
        run_name: str,
        *,
        config: Mapping[str, Any] | None = None,
        wandb_project: str = "entente-bc",
    ) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / f"{run_name}.metrics.jsonl"
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        self._wandb = self._maybe_init_wandb(run_name, wandb_project, config)

    @staticmethod
    def _maybe_init_wandb(
        run_name: str, project: str, config: Mapping[str, Any] | None
    ) -> Any | None:
        if not os.environ.get("WANDB_API_KEY"):
            return None
        try:
            import wandb
        except ImportError:
            return None
        try:
            wandb.init(project=project, name=run_name, config=dict(config or {}))
        except Exception:
            return None
        return wandb

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Record one step's metrics to JSONL (and wandb, if active)."""
        record = {"step": step, **metrics}
        self._handle.write(json.dumps(record, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()
        if self._wandb is not None:
            try:
                self._wandb.log(dict(metrics), step=step)
            except Exception:  # the module docstring's promise: wandb is best-effort
                self._wandb = None

    def close(self) -> None:
        """Close the JSONL sink and finish the hosted run, if one was started."""
        self._handle.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # same degradation contract as log()
                self._wandb = None

    def __enter__(self) -> MetricTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
