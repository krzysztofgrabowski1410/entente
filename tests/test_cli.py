"""Drive every CLI subcommand end to end, in process through ``main``.

Calling ``main`` directly rather than spawning a process keeps these fast
enough to sit in the main gate, and still exercises argument parsing, the
subcommand wiring and the exit codes a shell would see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entente.cli import main
from entente.engine.game import Game
from entente.engine.orders import parse_adjustment_order, parse_order
from entente.engine.serialization import game_from_text, game_to_text


def test_play_runs_a_complete_random_game(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "game.json"
    code = main(
        ["play", "--agents", "random", "--seed", "5", "--max-year", "1902", "--out", str(out)]
    )
    captured = capsys.readouterr().out
    assert code == 0
    assert "== S1901M" in captured
    assert "RESULT:" in captured
    saved = game_from_text(out.read_text(encoding="utf-8"))
    assert saved.year > 1902 or saved.is_over


def test_play_rejects_unknown_agent(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["play", "--agents", "psychic"]) == 1
    assert "unknown agent" in capsys.readouterr().err


def test_play_max_year_caps_a_countdown_calendar(capsys: pytest.CaptureFixture[str]) -> None:
    # Beleriand counts 25 down to 0, so "stop after year 24" has to compare in
    # the calendar's own direction. A comparison that assumes years ascend
    # leaves the cap inert on a countdown ruleset, and the game runs to its
    # own end instead.
    code = main(
        [
            "play",
            "--agents",
            "random",
            "--seed",
            "0",
            "--ruleset",
            "beleriand",
            "--map",
            "beleriand",
            "--max-year",
            "24",
        ]
    )
    captured = capsys.readouterr().out
    assert code == 0
    assert "S24M" in captured  # year 24 played...
    assert "S23M" not in captured  # ...year 23 not started
    assert "no winner by 24" in captured


def test_adjudicate_advances_a_saved_game(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    game = Game.new("standard", "standard")
    game.submit("FRANCE", [parse_order("A PAR - BUR")])
    path = tmp_path / "game.json"
    path.write_text(game_to_text(game), encoding="utf-8")

    code = main(["adjudicate", str(path)])
    captured = capsys.readouterr().out
    assert code == 0
    assert "== S1901M" in captured
    assert "A PAR - BUR .. succeeds" in captured
    assert "next phase: F1901M" in captured

    advanced = game_from_text(path.read_text(encoding="utf-8"))
    assert advanced.phase == "F1901M"
    assert any(str(u) == "A BUR" for u in advanced.state.units)


def test_adjudicate_out_flag_leaves_the_input_untouched(tmp_path: Path) -> None:
    game = Game.new("standard", "standard")
    source = tmp_path / "in.json"
    target = tmp_path / "out.json"
    original = game_to_text(game)
    source.write_text(original, encoding="utf-8")
    assert main(["adjudicate", str(source), "--out", str(target)]) == 0
    assert source.read_text(encoding="utf-8") == original
    assert game_from_text(target.read_text(encoding="utf-8")).phase == "F1901M"


def test_adjudicate_reports_unreadable_and_unloadable_files(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["adjudicate", str(tmp_path / "missing.json")]) == 1
    assert "cannot read" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["adjudicate", str(bad)]) == 1
    assert "cannot load" in capsys.readouterr().err


def test_adjudicate_refuses_a_finished_game(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Declare France the winner at every Fall count: the game finishes after
    # its first year both when saved and when the CLI replays the file (the
    # patch spans both, keeping the save legitimate).
    monkeypatch.setattr(Game, "winner", property(lambda self: "FRANCE"))
    game = Game.new("standard", "standard")
    game.adjudicate()  # S1901M
    game.adjudicate()  # F1901M -> Fall count -> over
    assert game.is_over
    path = tmp_path / "done.json"
    path.write_text(game_to_text(game), encoding="utf-8")
    assert main(["adjudicate", str(path)]) == 1
    assert "game is over" in capsys.readouterr().err


def test_play_rejects_unknown_map(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["play", "--map", "atlantis"]) == 1
    assert capsys.readouterr().err.strip()


def test_phase_log_covers_retreats_and_winters(capsys: pytest.CaptureFixture[str]) -> None:
    # A scripted dislodgement, a retreat, and a winter with a build: every
    # branch of the phase log prints.
    from entente.cli import _print_record
    from entente.engine.orders import parse_retreat_order

    game = Game.new("standard", "standard")
    game.submit("GERMANY", [parse_order("A MUN - BUR"), parse_order("F KIE - HOL")])
    game.adjudicate()
    game.submit("FRANCE", [parse_order("A PAR - BUR"), parse_order("A MAR S A PAR - BUR")])
    game.adjudicate()
    game.submit("GERMANY", [parse_retreat_order("Disband A BUR")])
    game.adjudicate()
    assert game.phase == "W1901A"  # Germany took Holland: one build due
    game.submit("GERMANY", [parse_adjustment_order("Build A KIE")])
    game.adjudicate()
    for record in game.history:
        _print_record(game, record)
    out = capsys.readouterr().out
    assert "dislodged: GERMANY A BUR (attacked from PAR, overland)" in out
    assert "disbanded: GERMANY A BUR" in out
    assert "built: GERMANY A KIE" in out
    assert "centers: " in out


def _write_tournament_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tournament.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_tournament_runs_and_prints_a_summary_table(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = _write_tournament_config(
        tmp_path,
        "format: one_vs_six\nagents: [greedy, random]\nseeds: [0]\nmax_year: 1901\n",
    )

    code = main(["tournament", str(config)])

    out = capsys.readouterr().out
    assert code == 0
    assert "ran 7 games" in out
    assert "greedy" in out
    assert "random" in out
    assert "mean SoS" in out


def test_tournament_set_override_applies(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # seeds: {start,stop} is 3 games/seat * 7 seats = 21; --set narrows it to 1.
    config = _write_tournament_config(
        tmp_path,
        "format: one_vs_six\nagents: [greedy, random]\n"
        "seeds: {start: 0, stop: 3}\nmax_year: 1901\n",
    )

    code = main(["tournament", str(config), "--set", "seeds=[0]", "--set", "max_year=1901"])

    out = capsys.readouterr().out
    assert code == 0
    assert "ran 7 games" in out  # one seed * 7 seats, not three


def test_tournament_writes_output_jsonl_when_configured(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    config = _write_tournament_config(
        tmp_path,
        f"format: one_vs_six\nagents: [greedy, random]\nseeds: [0]\n"
        f"max_year: 1901\noutput: {results_path}\n",
    )

    assert main(["tournament", str(config)]) == 0

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7


def test_tournament_round_robin_format_works(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = _write_tournament_config(
        tmp_path,
        "format: round_robin\nagents: [greedy, random]\nseeds: [0, 1]\nmax_year: 1901\n",
    )

    code = main(["tournament", str(config)])

    out = capsys.readouterr().out
    assert code == 0
    assert "ran 4 games" in out


def test_tournament_reports_missing_config(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["tournament", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot load" in capsys.readouterr().err


def test_tournament_reports_unknown_agent(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = _write_tournament_config(
        tmp_path, "format: one_vs_six\nagents: [psychic, random]\nseeds: [0]\nmax_year: 1901\n"
    )
    assert main(["tournament", str(config)]) == 1
    assert "unknown agent" in capsys.readouterr().err


def test_negotiate_demo_is_registered_through_importlib(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``entente.negotiation.cli.register()`` wires this subcommand at
    # parser-build time through importlib, so nothing imports the negotiation
    # package to make the subcommand exist. This is the end-to-end proof that
    # the wiring reaches ``entente.cli.main``, and it does not import that
    # package itself, which would defeat the point of testing the seam.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    scenario = Path("src/entente/negotiation/scenarios/s1.json")
    code = main(["negotiate-demo", str(scenario), "--out-dir", str(tmp_path)])
    captured = capsys.readouterr().out
    assert code == 0
    assert "P(true move of GERMANY)" in captured
    assert (tmp_path / "s1.json").exists()


def test_civil_disorder_line_prints(capsys: pytest.CaptureFixture[str]) -> None:
    from entente.cli import _print_record
    from entente.engine.map import Location, UnitKind, load_map
    from entente.engine.phase import Phase, PhaseKind, Season
    from entente.engine.rules import STANDARD
    from entente.engine.state import State, Unit

    state = State.of(
        "standard",
        "F1901M",
        [
            Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
            Unit("GERMANY", UnitKind.ARMY, Location("MUN")),
        ],
    )
    game = Game(
        STANDARD,
        load_map("standard"),
        state,
        {"MUN": "GERMANY"},
        Phase(1901, Season.FALL, PhaseKind.MOVEMENT),
    )
    game.adjudicate()
    for record in game.history:
        _print_record(game, record)
    out = capsys.readouterr().out
    assert "== W1901A (auto)" in out
    assert "civil disorder removal: FRANCE A BUR" in out


def test_beleriand_demo_runs_start_to_intervention(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Play the demo game whole, from the opening dice to the intervention.

    A single seeded Beleriand game exercises the Layer-2 mechanics on the way
    through, so the demo doubles as an end-to-end check that the variant runs
    from the command line.
    """
    out = tmp_path / "beleriand.json"
    # The seed is chosen so the run walks the whole Layer-2 parade, Ulmo's
    # wrath included. Any change to the Ulmo list or to an order menu shifts
    # every later draw, so a seed that showed the wrath can stop showing it;
    # re-choosing one is then the fix, since the mechanic itself is covered
    # case by case elsewhere and what this asserts is that the demo prints it.
    code = main(["beleriand-demo", "--seed", "3", "--out", str(out)])
    captured = capsys.readouterr().out
    assert code == 0
    assert "Ulmo's dice" in captured
    assert "event: NAUGRIM A BEL makes 2 masks" in captured
    assert "passes a mask" in captured
    assert "grant: TELERI passes fleet-building to MEN" in captured
    assert "event: Glaurung arrives in Angband (available Spring 20)" in captured
    assert "event: Glaurung devastates" in captured
    assert "Ulmo's wrath annihilates" in captured
    assert "event: The Valar intervene" in captured
    assert "RESULT: the Valar intervene" in captured
    assert out.exists()
    # the saved v2 document replays: determinism + serialization in one check
    from entente.engine.serialization import game_from_text

    game = game_from_text(out.read_text(encoding="utf-8"))
    assert game.is_over and game.seed == 3


def test_beleriand_demo_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["beleriand-demo", "--seed", "11"]) == 0
    first = capsys.readouterr().out
    assert main(["beleriand-demo", "--seed", "11"]) == 0
    second = capsys.readouterr().out
    assert first == second
