"""Typed config loader for ``entente tournament``.

It rides on the project's one config system,
:mod:`entente.training.config`, so there is a single override mechanism to
learn: YAML reading, the non-mapping and malformed-file error surface, and
dotted ``key=value`` overrides are shared with the behavioral-cloning configs,
and :class:`TournamentConfig` is a frozen pydantic v2 model. The
tournament-specific shape -- the ``one_vs_six`` two-agent rule and
the ``{start, stop}`` seed-range shorthand -- stays here, validated with the same
:class:`~entente.training.config.ConfigError` messages the CLI reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from entente.training.config import ConfigError, apply_overrides, load_yaml

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = ["ConfigError", "TournamentConfig", "load_tournament_config"]

VALID_FORMATS = ("one_vs_six", "round_robin")


class TournamentConfig(BaseModel):
    """A validated tournament run specification.

    ``configs/tournament/main.yaml`` is the shipped file of this shape.
    """

    model_config = ConfigDict(frozen=True)

    format: str
    agents: tuple[str, ...]
    seeds: tuple[int, ...]
    max_year: int = 1910
    ruleset: str = "standard"
    map_name: str = "standard"
    max_workers: int | None = None
    output: str | None = None


def _require(data: dict[str, Any], key: str) -> Any:
    """Read a mandatory field, reporting its absence in the CLI's error shape."""
    if key not in data:
        raise ConfigError(f"missing required field {key!r}")
    return data[key]


def _seed_list(raw: Any) -> tuple[int, ...]:
    """Read a seed set: an explicit list of ints, or a ``{start, stop}`` range.

    The range form has ``stop`` exclusive, like :func:`range`, and is what keeps
    the config readable for a 200-seed sweep.
    """
    if isinstance(raw, list):
        return tuple(int(value) for value in raw)
    if isinstance(raw, dict) and "start" in raw and "stop" in raw:
        return tuple(range(int(raw["start"]), int(raw["stop"])))
    raise ConfigError("seeds must be a list of ints or a {start, stop} mapping")


_KNOWN_KEYS = frozenset(
    {"format", "agents", "seeds", "max_year", "ruleset", "map", "max_workers", "output"}
)


def _validate(data: dict[str, Any]) -> TournamentConfig:
    """Turn a raw mapping into a validated config.

    The tournament-specific checks stay explicit here rather than moving into
    the model, so their messages stay stable for the CLI to report.

    Unknown keys are rejected rather than dropped, matching the behavioral-
    cloning configs, which forbid extras (``training.config._Frozen``). A
    silently ignored ``max_yaer:``, or a misspelled ``--set`` override, would
    otherwise validate cleanly and change nothing about the run.
    """
    unknown = sorted(set(data) - _KNOWN_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys {unknown} (known: {sorted(_KNOWN_KEYS)})")
    fmt = str(_require(data, "format"))
    if fmt not in VALID_FORMATS:
        raise ConfigError(f"format must be one of {VALID_FORMATS}, got {fmt!r}")
    agents = tuple(str(a) for a in _require(data, "agents"))
    if fmt == "one_vs_six" and len(agents) != 2:
        raise ConfigError("one_vs_six needs exactly 2 agents: [agent_a, agent_b]")
    if not agents:
        raise ConfigError("agents must be non-empty")
    seeds = _seed_list(_require(data, "seeds"))
    if not seeds:
        raise ConfigError("seeds must be non-empty")
    max_workers = data.get("max_workers")
    output = data.get("output")
    return TournamentConfig(
        format=fmt,
        agents=agents,
        seeds=seeds,
        max_year=int(data.get("max_year", 1910)),
        ruleset=str(data.get("ruleset", "standard")),
        map_name=str(data.get("map", "standard")),
        max_workers=int(max_workers) if max_workers is not None else None,
        output=str(output) if output is not None else None,
    )


def load_tournament_config(path: str | Path, overrides: Iterable[str] = ()) -> TournamentConfig:
    """Parse, override and validate a tournament YAML config.

    ``overrides`` are dotted ``key=value`` strings, as in
    ``entente tournament --set max_year=1905``, and are applied through the
    shared config machinery before validation, so an override is checked exactly
    as a file value is.
    """
    data = load_yaml(path)
    overrides = list(overrides)
    if overrides:
        data = apply_overrides(data, overrides)
    return _validate(data)
