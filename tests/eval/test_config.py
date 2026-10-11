"""config.py: the typed YAML tournament-config loader.

Required fields, defaults, the seed-range shorthand, and the validation errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entente.eval.config import ConfigError, load_tournament_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_minimal_one_vs_six_config(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        format: one_vs_six
        agents: [greedy, random]
        seeds: [0, 1, 2]
        """,
    )

    config = load_tournament_config(path)

    assert config.format == "one_vs_six"
    assert config.agents == ("greedy", "random")
    assert config.seeds == (0, 1, 2)
    assert config.max_year == 1910  # default
    assert config.ruleset == "standard"
    assert config.map_name == "standard"
    assert config.max_workers is None
    assert config.output is None


def test_seed_range_shorthand(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        format: round_robin
        agents: [greedy, random]
        seeds: {start: 0, stop: 5}
        """,
    )

    config = load_tournament_config(path)

    assert config.seeds == (0, 1, 2, 3, 4)


def test_all_fields_override_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        format: one_vs_six
        agents: [greedy, random]
        seeds: [0]
        max_year: 1920
        ruleset: standard
        map: standard
        max_workers: 4
        output: docs/results/example.jsonl
        """,
    )

    config = load_tournament_config(path)

    assert config.max_year == 1920
    assert config.max_workers == 4
    assert config.output == "docs/results/example.jsonl"


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_tournament_config(tmp_path / "does-not-exist.yaml")


def test_missing_required_field_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "format: one_vs_six\nagents: [greedy, random]\n")
    with pytest.raises(ConfigError, match="seeds"):
        load_tournament_config(path)


def test_unknown_format_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "format: nonsense\nagents: [greedy, random]\nseeds: [0]\n")
    with pytest.raises(ConfigError, match="format"):
        load_tournament_config(path)


def test_one_vs_six_requires_exactly_two_agents(tmp_path: Path) -> None:
    path = _write(tmp_path, "format: one_vs_six\nagents: [greedy, random, greedy]\nseeds: [0]\n")
    with pytest.raises(ConfigError, match="exactly 2 agents"):
        load_tournament_config(path)


def test_round_robin_rejects_empty_agents(tmp_path: Path) -> None:
    # only round_robin can reach the generic "agents must be non-empty"
    # check -- one_vs_six's own arity check fires first for that format.
    path = _write(tmp_path, "format: round_robin\nagents: []\nseeds: [0]\n")
    with pytest.raises(ConfigError, match="agents must be non-empty"):
        load_tournament_config(path)


def test_rejects_empty_seeds_list(tmp_path: Path) -> None:
    path = _write(tmp_path, "format: round_robin\nagents: [greedy, random]\nseeds: []\n")
    with pytest.raises(ConfigError, match="seeds must be non-empty"):
        load_tournament_config(path)


def test_rejects_a_seeds_value_that_is_neither_list_nor_range(tmp_path: Path) -> None:
    path = _write(tmp_path, "format: round_robin\nagents: [greedy, random]\nseeds: 5\n")
    with pytest.raises(ConfigError, match="list of ints or a"):
        load_tournament_config(path)


def test_unknown_keys_are_rejected_not_dropped(tmp_path: Path) -> None:
    # A typo like `max_yaer: 1905` must fail rather than validate and change
    # nothing, which is why the loader takes the BC configs' extra="forbid"
    # stance.
    path = _write(
        tmp_path,
        "format: round_robin\nagents: [greedy, random]\nseeds: [0]\nmax_yaer: 1905\n",
    )
    with pytest.raises(ConfigError, match=r"unknown config keys.*max_yaer"):
        load_tournament_config(path)


def test_a_misspelled_set_override_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "format: round_robin\nagents: [greedy, random]\nseeds: [0]\n")
    with pytest.raises(ConfigError, match="unknown config keys"):
        load_tournament_config(path, overrides=["max_yaer=1905"])


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_tournament_config(path)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "format: [unterminated\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_tournament_config(path)
