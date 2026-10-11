"""piKL configuration models (docs/pikl.tex Sec. 7).

They are pydantic v2 models over the shared YAML and ``--set`` loader from
:mod:`entente.training.config`, so a search run has the same one override
mechanism every other config in the project has. Torch-free: the sweep script
and the agent read these; tests validate the committed ``configs/pikl/*.yaml``
files against them.

``lam`` accepts ``.inf`` in YAML (the anchor-only end of the sweep grid);
``NaN`` and negatives are rejected here rather than deep in the search.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EtaConfig(_Frozen):
    """Learning-rate schedule for the anchored-hedge update (docs/pikl.tex Sec. 7).

    The default is the paper's adaptive ``c/(sigma_t*sqrt(t))``; a constant rate
    is the option to take when a run has to reproduce exactly.
    """

    mode: Literal["adaptive", "constant"] = "adaptive"
    scale: float = Field(default=10.0 / 3.0, gt=0.0)
    value: float | None = Field(default=None, gt=0.0)

    def model_post_init(self, _ctx: object) -> None:
        """Reject a constant schedule that names no rate."""
        if self.mode == "constant" and self.value is None:
            raise ValueError("eta.value is required when eta.mode is 'constant'")

    @property
    def constant(self) -> float | None:
        """Return the ``eta`` argument :class:`entente.search.pikl.AnchoredHedge` takes.

        That is a float in constant mode and ``None`` in adaptive mode, where
        the learner picks the rate itself.
        """
        return self.value if self.mode == "constant" else None


class PiklSearchConfig(_Frozen):
    """One search invocation's knobs (docs/pikl.tex Sec. 7 table)."""

    k: int = Field(default=12, ge=1)
    attempts: int | None = Field(default=None, ge=1)
    iterations: int = Field(default=128, ge=1)
    lam: float = 1.0
    eta: EtaConfig = EtaConfig()
    temperature: float = Field(default=0.5, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    include_greedy: bool = True
    play: Literal["sample", "argmax"] = "sample"

    @field_validator("lam")
    @classmethod
    def _lam_domain(cls, value: float) -> float:
        if math.isnan(value) or value < 0.0:
            raise ValueError("lam must be >= 0 (inf allowed)")
        return value


class PiklAgentConfig(_Frozen):
    """Everything one complete piKL player needs.

    A BC checkpoint serving as both anchor policy and value net, the search
    knobs, and the fallback agent for the phases where the anchor is undefined,
    mirroring :class:`~entente.agents.bc_agent.BCAgent`.
    """

    checkpoint: str
    device: str = "cpu"
    fallback: str = "greedy"
    seed: int = 0
    search: PiklSearchConfig = PiklSearchConfig()


class LambdaSweepConfig(_Frozen):
    """The headline ablation: piKL(lam) in one seat against six BC-anchor opponents.

    The sweep runs the ``lambdas`` grid and writes the strength (SoS share) and
    agreement curves as JSONL, so every reported point regenerates from the
    committed file.

    ``agreement_shards`` points at preprocessed human shards for the held-out
    human-agreement metric; ``None`` records anchor-agreement only. The human
    metric is not approximated in that case: a number is measured or it is
    absent.
    """

    agent: PiklAgentConfig
    lambdas: tuple[float, ...] = Field(min_length=1)
    games: int = Field(default=15, ge=1)
    seat: str = "FRANCE"
    rotate_seats: bool = False
    max_year: int = Field(default=1910, ge=1901)
    ruleset: str = "standard"
    map_name: str = "standard"
    output: str = "docs/results/pikl_lambda_sweep.jsonl"
    agreement_shards: str | None = None
    agreement_rows: int = Field(default=200, ge=1)

    @field_validator("lambdas")
    @classmethod
    def _lambdas_domain(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        for value in values:
            if math.isnan(value) or value < 0.0:
                raise ValueError("every lambda must be >= 0 (inf allowed)")
        return values


__all__ = [
    "EtaConfig",
    "LambdaSweepConfig",
    "PiklAgentConfig",
    "PiklSearchConfig",
]
