"""PressRecord: the JSONL round-trip, the aggregation, the delta and the table.

The delta is the press-versus-no-press difference in the subject's mean SoS, and
the table has to regenerate from the records alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entente.eval.press_results import (
    PressRecord,
    read_press_jsonl,
    render_table,
    sos_delta,
    summarize,
    write_press_jsonl,
)


def _record(press: bool, sos: float, **kwargs: object) -> PressRecord:
    defaults: dict[str, object] = {
        "seat": "FRANCE",
        "seed": 0,
        "supply_centers": 4,
        "survived": True,
    }
    defaults.update(kwargs)
    return PressRecord(press=press, sos=sos, **defaults)  # type: ignore[arg-type]


def test_sos_delta_requires_both_conditions() -> None:
    # A delta over one condition is meaningless, so the guard refuses it in
    # the module's documented empty-input shape rather than letting the
    # division reach permutation_test and raise something less legible.
    with pytest.raises(ValueError, match="both conditions"):
        sos_delta([_record(True, 0.3)])
    with pytest.raises(ValueError, match="both conditions"):
        sos_delta([_record(False, 0.1)])


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = [_record(False, 0.14), _record(True, 0.20, seed=1, survived=False)]
    path = tmp_path / "press.jsonl"
    assert write_press_jsonl(records, path) == 2
    assert list(read_press_jsonl(path)) == records


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "press.jsonl"
    write_press_jsonl([_record(False, 0.1)], path)
    assert path.exists()


class TestSummarize:
    """Per-condition aggregates, and the absence of a condition that never ran."""

    def test_per_condition_aggregates(self) -> None:
        records = [
            _record(False, 0.10, seed=0),
            _record(False, 0.20, seed=1, survived=False),
            _record(True, 0.30, seed=0),
        ]
        summaries = summarize(records)
        assert summaries[False].games == 2
        assert summaries[False].mean_sos == pytest.approx(0.15)
        assert summaries[False].survival == pytest.approx(0.5)
        assert summaries[True].games == 1
        assert summaries[True].mean_sos == pytest.approx(0.30)
        assert summaries[True].survival == pytest.approx(1.0)
        assert summaries[False].sos_ci[0] <= summaries[False].mean_sos <= summaries[False].sos_ci[1]

    def test_missing_condition_is_absent_not_zero(self) -> None:
        summaries = summarize([_record(False, 0.1)])
        assert False in summaries
        assert True not in summaries


class TestSosDelta:
    """The headline delta and its p-value, both computed by hand here."""

    def test_hand_computed_delta_and_permutation_p_value(self) -> None:
        records = [
            _record(False, 0.10, seed=0),
            _record(False, 0.10, seed=1),
            _record(True, 0.20, seed=0),
            _record(True, 0.20, seed=1),
        ]
        delta, p_value = sos_delta(records)
        assert delta == pytest.approx(0.10)
        # Pooling {0.2, 0.2, 0.1, 0.1} into two groups of two has 6 equally
        # likely distinct splits (by which two of the four go together); only
        # the 2 that fully separate the pairs reach the observed gap of 0.1,
        # so the exact permutation p-value is 2/6 = 1/3 (matches
        # entente.eval.stats.permutation_test's own docstring semantics --
        # verified directly, not assumed: 10,000 Monte Carlo iterations give
        # ~0.333, not the 1.0 an "any shuffle is at least this extreme"
        # misreading would suggest).
        assert p_value == pytest.approx(1 / 3, abs=0.02)

    def test_negative_delta_is_reported_plainly(self) -> None:
        records = [_record(False, 0.30, seed=0), _record(True, 0.10, seed=0)]
        delta, _ = sos_delta(records)
        assert delta == pytest.approx(-0.20)


def test_render_table_regenerates_every_column() -> None:
    records = [
        _record(False, 0.10, seed=0),
        _record(False, 0.20, seed=1),
        _record(True, 0.30, seed=0, survived=False),
    ]
    table = render_table(records)
    lines = table.splitlines()
    assert lines[0].startswith("| Condition | Games | Mean SoS (subject) ")
    assert "| no press | 2 | 0.1500 " in table
    assert "| press | 1 | 0.3000 " in table
    assert "| 0.000 |" in table  # press condition's survival (the one game did not survive)
