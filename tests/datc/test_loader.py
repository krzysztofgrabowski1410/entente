"""Check the conformance-case loader itself, which the whole suite trusts.

Every certified outcome reaches the runner through this loader, so a loader
that drops a case, or accepts one that asserts nothing, would hide exactly what
the suite exists to prove.
"""

from pathlib import Path

import pytest

from tests.datc.loader import CASES_DIR, ConformanceCase, _validate, load_all, load_file


@pytest.mark.datc
def test_loads_encoded_sections() -> None:
    # Marked `datc` so the badge job (`pytest -m datc`) runs this exact-count
    # guard: without it a valid-YAML edit that drops cases would publish a
    # green "N/N" badge with a smaller N.
    cases = load_all()
    by_section: dict[str, int] = {}
    for case in cases:
        by_section[case.id.rsplit(".", 1)[0]] = by_section.get(case.id.rsplit(".", 1)[0], 0) + 1
    # DATC v3.3 counts, complete: all 176 cases of 6.A-6.J. 6.B.14 is counted
    # under 6.I, because it is a build-phase case.
    assert by_section == {
        "6.A": 12,
        "6.B": 19,
        "6.C": 9,
        "6.D": 35,
        "6.E": 15,
        "6.F": 30,
        "6.G": 20,
        "6.H": 17,
        "6.I": 7,
        "6.J": 12,
    }


def test_case_fields_round_trip() -> None:
    case = load_file(CASES_DIR / "6a.yaml")[0]
    assert case.ruleset == "standard"
    assert case.section == "6.A Basic checks"
    assert case.phase == "S1901M"
    assert case.units == {"ENGLAND": ["F NTH"]}
    assert case.orders == {"ENGLAND": ["F NTH - PIC"]}
    assert case.expect["resolutions"] == {"F NTH - PIC": "fails"}


def test_pytest_id_is_spec_traceable() -> None:
    case = load_all()[0]
    assert case.pytest_id == "6.A.1-moving_to_a_province_that_is_not_a_neighbour"


def test_validation_allows_orders_for_power_without_units() -> None:
    # DATC 6.A.6: a power may submit orders for units it does not own; the
    # engine drops them. The loader must not reject such cases.
    case = ConformanceCase(
        id="X.1",
        title="foreign order",
        section="X",
        ruleset="standard",
        phase="S1901M",
        units={"FRANCE": ["A PAR"]},
        orders={"GERMANY": ["A PAR - BUR"]},
        expect={"resolutions": {"A PAR - BUR": "fails"}},
    )
    _validate(case, Path("inline.yaml"))


def test_validation_requires_resolutions() -> None:
    bad = ConformanceCase(
        id="X.2",
        title="bad",
        section="X",
        ruleset="standard",
        phase="S1901M",
        units={"FRANCE": ["A PAR"]},
        orders={"FRANCE": ["A PAR - BUR"]},
        expect={},
    )
    with pytest.raises(ValueError, match="resolutions"):
        _validate(bad, Path("inline.yaml"))
