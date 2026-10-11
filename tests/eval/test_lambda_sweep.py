"""SweepRecord: the JSONL round-trip, the aggregation and the results table.

The round-trip includes ``lam = inf``, which strict JSON cannot spell as a
number; the table has to regenerate from the records alone (docs/pikl.tex Sec. 7).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from entente.eval.lambda_sweep import (
    SweepRecord,
    aggregate,
    append_sweep_jsonl,
    completed_seats,
    keep_completed_seats,
    read_sweep_jsonl,
    render_table,
    write_sweep_jsonl,
)


def _record(lam: float, sos: float, **kwargs: object) -> SweepRecord:
    defaults: dict[str, object] = {
        "seat": "FRANCE",
        "seed": 0,
        "supply_centers": 4,
        "survived": True,
        "anchor_agreement": 0.5,
        "human_agreement": None,
    }
    defaults.update(kwargs)
    return SweepRecord(lam=lam, sos=sos, **defaults)  # type: ignore[arg-type]


def test_jsonl_round_trip_including_inf(tmp_path: Path) -> None:
    records = [
        _record(0.0, 0.25),
        _record(math.inf, 0.10, anchor_agreement=0.97, human_agreement=0.44),
    ]
    path = tmp_path / "sweep.jsonl"
    assert write_sweep_jsonl(records, path) == 2
    # strict-JSON safe: inf is the string "inf", not the Infinity literal
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["lam"] == "inf"
    back = list(read_sweep_jsonl(path))
    assert back == records


def test_append_accumulates_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "sweep.jsonl"
    assert append_sweep_jsonl([_record(0.0, 0.2, seed=0)], path) == 1  # creates the file
    append_sweep_jsonl([_record(0.0, 0.3, seed=1), _record(math.inf, 0.1)], path)
    back = list(read_sweep_jsonl(path))
    assert [r.seed for r in back] == [0, 1, 0]
    assert back[2].lam == math.inf  # inf survives the append/read round-trip


def test_completed_and_kept_seats_gate_on_full_batches() -> None:
    # two full seats (games_per_seat=2) plus a half-written third seat
    records = [
        _record(0.0, 0.2, seat="FRANCE", seed=0),
        _record(0.0, 0.3, seat="FRANCE", seed=1),
        _record(1.0, 0.2, seat="ITALY", seed=0),
        _record(1.0, 0.4, seat="ITALY", seed=1),
        _record(1.0, 0.1, seat="TURKEY", seed=0),  # partial: only 1 of 2
    ]
    done = completed_seats(records, games_per_seat=2)
    assert done == {(0.0, "FRANCE"), (1.0, "ITALY")}
    kept = keep_completed_seats(records, games_per_seat=2)
    assert len(kept) == 4  # the lone TURKEY record is dropped as a partial seat
    assert all((r.lam, r.seat) in done for r in kept)


def test_aggregate_orders_lambdas_and_averages() -> None:
    records = [
        _record(1.0, 0.2, seed=0),
        _record(1.0, 0.4, seed=1, survived=False, anchor_agreement=0.7),
        _record(0.0, 0.5, anchor_agreement=None),
        _record(math.inf, 0.1),
    ]
    points = aggregate(records)
    assert [p.lam for p in points] == [0.0, 1.0, math.inf]
    mid = points[1]
    assert mid.games == 2
    assert mid.mean_sos == pytest.approx(0.3)
    assert mid.survival == pytest.approx(0.5)
    assert mid.anchor_agreement == pytest.approx(0.6)
    assert mid.human_agreement is None
    assert mid.sos_ci[0] <= mid.mean_sos <= mid.sos_ci[1]
    # a lambda group with no agreement measurements reports None, not 0
    assert points[0].anchor_agreement is None


def test_render_table_regenerates_every_column() -> None:
    points = aggregate([_record(0.3, 0.25), _record(math.inf, 0.1, human_agreement=0.5)])
    table = render_table(points)
    lines = table.splitlines()
    assert lines[0].startswith("| lambda | Games | Mean SoS ")
    assert "| 0.3 | 1 | 0.2500 " in table
    assert "| inf | 1 | 0.1000 " in table
    assert "| 0.500 |" in table  # human agreement shown when present
    assert "| -- |" in table  # and em-dash when absent
