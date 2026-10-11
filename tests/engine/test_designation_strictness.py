"""Pin what a written unit designation binds, on both sides of the knob.

``Ruleset.strict_designation`` decides what a written kind or strength prefix
that matches no unit of the ordering power in the named province means. The
standard game is advisory and binds the province anyway, which is DATC 4.C.2
choice (b); Beleriand is strict and the order becomes invalid, so it holds
under 4.E.1. This file pins:

* the knob's values, and that the standard column keeps advisory binding, so
  the classic results cannot move -- proved at the binding level here and by
  the whole DATC suite elsewhere;
* the observable flip and its controls: exact designations, bare
  province-only specs, duplicate-text k-th binding and the ``3A(M)``/``3A(G)``
  designators all keep their behaviour, because each still matches a unit;
* the rulebook's *"a correct order still takes place"* clause (4.D.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import Resolution, adjudicate
from entente.engine.adjudicator.common import _Phase0State, phase0
from entente.engine.map import load_map
from entente.engine.orders import Hold, Move, parse_order, parse_unit_spec
from entente.engine.rules import RULESETS, STANDARD
from entente.engine.state import State, Unit

BELERIAND = RULESETS["beleriand"]
GRAPH_STD = load_map("standard")
GRAPH_BEL = load_map("beleriand")


def _bel_state(units: Mapping[str, Sequence[str]]) -> State:
    return State.of(
        "beleriand",
        "S25M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )


def _std_state(units: Mapping[str, Sequence[str]]) -> State:
    return State.of(
        "standard",
        "S1901M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )


def _sheets(orders: Mapping[str, Sequence[str]]) -> dict[str, list[object]]:
    return {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}


def _bel_records(units, orders):  # type: ignore[no-untyped-def]
    return phase0(GRAPH_BEL, BELERIAND, _bel_state(units), _sheets(orders)).records  # type: ignore[arg-type]


def _record_written(records, order_text: str):  # type: ignore[no-untyped-def]
    target = parse_order(order_text)
    for record in records:
        if record.written == target:
            return record
    raise AssertionError(f"no record bound to {order_text!r}")


# -- the knob itself --------------------------------------------------------


def test_knob_values() -> None:
    # Standard keeps the advisory default (choice b); beleriand adopts choice C.
    assert STANDARD.strict_designation is False
    assert BELERIAND.strict_designation is True


def test_layer2_wrapper_delegates_the_knob() -> None:
    # The per-game Layer-2 delegate wraps the ruleset, so it has to forward the
    # knob or a hooked Beleriand game would silently fall back to advisory.
    from entente.variants.beleriand.layer2 import BeleriandHooks, DynamicBeleriandRuleset

    wrapper = DynamicBeleriandRuleset(BELERIAND, BeleriandHooks(GRAPH_BEL, seed=0))
    assert wrapper.strict_designation is True


def test_designation_contradicts_isolates_the_knob() -> None:
    # claim_slot's province-only selection is byte-identical for both settings;
    # only designation_contradicts fires, and only under strict. Same wrong
    # order, opposite verdicts by ruleset -- the knob in isolation.
    wrong = parse_order("3A ANF - LOT")
    correct = parse_order("2A ANF - LOT")
    strict = _Phase0State(GRAPH_BEL, BELERIAND, _bel_state({"DORIATH": ["2A ANF", "A ANF"]}))
    assert strict.designation_contradicts("DORIATH", wrong) is True
    assert strict.designation_contradicts("DORIATH", correct) is False
    advisory = _Phase0State(GRAPH_STD, STANDARD, _std_state({"FRANCE": ["A PAR"]}))
    assert advisory.designation_contradicts("FRANCE", parse_order("F PAR - BUR")) is False


# -- standard stays advisory (byte-identical guarantee) ---------------------


def test_standard_wrong_kind_still_binds_and_moves() -> None:
    # DATC 4.C.2 choice b / 6.D.35 shape: with the knob off the F is advisory
    # and the army takes the order. This is what the strict column flips.
    state = _std_state({"FRANCE": ["A PAR"]})
    result = adjudicate(STANDARD, state, _sheets({"FRANCE": ["F PAR - BUR"]}), graph=GRAPH_STD)  # type: ignore[arg-type]
    assert result.resolution_of("F PAR - BUR") is Resolution.SUCCEEDS
    assert sorted(str(u) for u in result.new_state.units) == ["A BUR"]


# -- strict: the flip and its controls --------------------------------------


def test_strict_wrong_strength_prefix_degrades_to_hold() -> None:
    records = _bel_records({"DORIATH": ["2A ANF", "A ANF"]}, {"DORIATH": ["3A ANF - LOT"]})
    record = _record_written(records, "3A ANF - LOT")  # bound an occurrence, then invalidated
    assert not record.valid
    assert isinstance(record.effective, Hold)


def test_strict_wrong_kind_degrades_to_hold() -> None:
    # The kind arm (matches no fleet), end to end.
    state = _bel_state({"FINROD": ["A NAR"]})
    result = adjudicate(BELERIAND, state, _sheets({"FINROD": ["F NAR - TAL"]}), graph=GRAPH_BEL)  # type: ignore[arg-type]
    assert result.resolution_of("F NAR - TAL") is Resolution.FAILS
    assert sorted(str(u) for u in result.new_state.units) == ["A NAR"]


def test_strict_correct_strength_designation_still_moves() -> None:
    records = _bel_records({"DORIATH": ["2A ANF", "A ANF"]}, {"DORIATH": ["2A ANF - LOT"]})
    record = _record_written(records, "2A ANF - LOT")
    # The 2A binds exactly and moves; strict never fires on a matching prefix.
    assert record.valid
    assert isinstance(record.effective, Move)
    assert record.effective.destination.province == "LOT"
    assert record.unit.strength == 2


def test_strict_bare_spec_names_the_single_strength_unit() -> None:
    # The rulebook treats a designation without a prefix as base strength 1 -- *"The convention
    # that A is any strength army is wrong. A is same as 1A."* -- so a
    # strengthless "A ANF" names a strength-1 army. With only a Double Army
    # present the designation contradicts every own occupant and the order
    # degrades to a Hold under strict binding. Beside a real strength-1 twin
    # the bare spec binds it exactly.
    records = _bel_records({"DORIATH": ["2A ANF"]}, {"DORIATH": ["A ANF - LOT"]})
    record = _record_written(records, "A ANF - LOT")
    assert not record.valid
    assert isinstance(record.effective, Hold)

    records = _bel_records({"DORIATH": ["2A ANF", "A ANF"]}, {"DORIATH": ["A ANF - LOT"]})
    record = _record_written(records, "A ANF - LOT")
    assert record.valid
    assert isinstance(record.effective, Move)
    assert record.unit.strength == 1


def test_strict_preserves_duplicate_text_kth_binding() -> None:
    # Two identical-text orders over two fungible twins bind to distinct
    # occurrences and both resolve: each "A ANF" matches an army, so strict
    # binding never fires.
    state = _bel_state({"MORGOTH": ["A ANF", "A ANF"]})
    orders = {"MORGOTH": ["A ANF - LOT", "A ANF - DNN"]}
    result = adjudicate(BELERIAND, state, _sheets(orders), graph=GRAPH_BEL)  # type: ignore[arg-type]
    assert result.resolution_of("A ANF - LOT") is Resolution.SUCCEEDS
    assert result.resolution_of("A ANF - DNN") is Resolution.SUCCEEDS
    assert sorted(str(u) for u in result.new_state.units) == ["A DNN", "A LOT"]


def test_strict_keeps_the_special_unit_designators() -> None:
    # ``3A(M)`` and ``3A(G)`` ride the ``special`` field, which the strict base-strength
    # check never reads, so a designated order whose value (3A) stands in the
    # province binds exactly as its plain twin would and strict never fires. A
    # designator naming a special that is not there is caught by the Layer-2
    # preprocessing, not here.
    state = _bel_state({"MORGOTH": ["3A ANF"]})
    result = adjudicate(
        BELERIAND, state, _sheets({"MORGOTH": ["3A(M) ANF - LOT"]}), graph=GRAPH_BEL
    )  # type: ignore[arg-type]
    assert result.resolution_of("3A(M) ANF - LOT") is Resolution.SUCCEEDS
    assert sorted(str(u) for u in result.new_state.units) == ["3A LOT"]


def test_strict_correct_order_still_takes_place() -> None:
    # Rulebook 4.C.2: "if an Order with correct designation exists, then the
    # correct Order is valid and takes place" -- the 4.D.3 first-legal-wins
    # rule. The sloppy fleet order is the ignored illegal claim; the army order
    # wins the occurrence.
    state = _bel_state({"MORGOTH": ["A ANF"]})
    orders = {"MORGOTH": ["F ANF - LOT", "A ANF - DNN"]}
    result = adjudicate(BELERIAND, state, _sheets(orders), graph=GRAPH_BEL)  # type: ignore[arg-type]
    assert result.resolution_of("F ANF - LOT") is Resolution.FAILS
    assert result.resolution_of("A ANF - DNN") is Resolution.SUCCEEDS
    assert sorted(str(u) for u in result.new_state.units) == ["A DNN"]


# -- the base strength a support names is binding too ------------------------
#
# The knob above governs an order's own spec. These pin the same law one step
# out, on what a unit reference writes. A bare ``A`` means base strength 1 wherever it
# is written, so ``S MORGOTH A DNN`` must not lend its strength to a 2A
# standing alone. The same knob decides it: under strict designation the
# reference binds only what it describes, exactly as ``designation_contradicts``
# does for own specs on that column, and where designation stays advisory the
# DATC-heritage province-only fallback applies -- which is what keeps the
# standard column's results fixed by construction rather than by coincidence.


def _survives_a_supported_defence(defender: str, ref: str, attacker_support: bool) -> bool:
    """Report whether a supported defender at DNN survives a 2A attack.

    DNN has capacity 1 and holds ``defender``; RIV supports it with the unit
    reference ``ref``; LAD attacks with a 2A, itself supported when
    ``attacker_support``. The defender lives exactly when the RIV support
    binds, which makes the return value a direct read of the reference.
    """
    units = {"MORGOTH": [f"{defender} DNN", "A RIV"], "NAUGRIM": ["2A LAD"]}
    orders = {
        "MORGOTH": [f"{defender} DNN H", f"A RIV S MORGOTH {ref} DNN"],
        "NAUGRIM": ["2A LAD - DNN"],
    }
    if attacker_support:
        units["NAUGRIM"].append("A AEL")
        orders["NAUGRIM"].append("A AEL S NAUGRIM 2A LAD - DNN")
    result = adjudicate(BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL)
    return not any(d.unit.location.province == "DNN" for d in result.dislodgements)


def test_a_named_base_strength_that_stands_there_still_lends_its_support() -> None:
    """Bind a support reference that names a base strength the province actually holds.

    These are the controls for the law below: a reference naming a present
    base strength binds, prefixless for base strength 1 and prefixed above it.
    """
    assert _survives_a_supported_defence("A", "A", attacker_support=False)
    assert _survives_a_supported_defence("2A", "2A", attacker_support=True)


def test_a_bare_reference_does_not_bind_a_stronger_unit_standing_alone() -> None:
    """Bind nothing when a bare reference meets only a stronger unit standing there.

    ``S MORGOTH A DNN`` names the strength-1 unit at DNN. With only a 2A
    standing there it names nothing at all, so it lends no strength and the 2A
    falls.
    """
    assert not _survives_a_supported_defence("2A", "A", attacker_support=True)


def test_a_written_base_strength_does_not_bind_a_weaker_unit_standing_alone() -> None:
    """Bind nothing when a written base strength names a base strength the province lacks."""
    assert not _survives_a_supported_defence("A", "2A", attacker_support=False)


def test_the_advisory_column_keeps_its_province_only_fallback() -> None:
    """Keep the province-only fallback on the advisory column.

    Where designation is advisory, DATC 4.C.2 choice (b) applies when no
    candidate of the written kind stands in the province, and that reading is
    what keeps the classic results where they are. England's fleet is named as
    an army and still binds.
    """
    units = {"ENGLAND": ["F NTH", "F ENG"], "FRANCE": ["A BEL"]}
    orders = {
        "ENGLAND": ["F NTH - BEL", "F ENG S ENGLAND A NTH - BEL"],
        "FRANCE": ["A BEL H"],
    }
    result = adjudicate(STANDARD, _std_state(units), _sheets(orders), graph=GRAPH_STD)
    assert result.resolution_of("F NTH - BEL") is Resolution.SUCCEEDS
    assert any(d.unit.location.province == "BEL" for d in result.dislodgements)


def test_a_convoy_cargo_province_can_never_hold_two_units() -> None:
    """Why the same law needs no counterpart for CONVOYS.

    A convoy's cargo reference binds by province alone (`find_subject_record`
    checks neither kind nor base strength), which would be the same ambiguity -- except
    that a cargo must be an army standing on a COASTAL province, and every
    coastal province on both shipped maps has capacity 1. At most one unit ever
    stands there, so the reference cannot be ambiguous by construction. This
    pins the structural fact, so a future map that gave a coastal province an
    Area's capacity fails here instead of silently reopening the question.
    """
    for graph in (GRAPH_STD, GRAPH_BEL):
        roomy = [
            province
            for province in graph.provinces
            if graph.terrain(province).value == "coastal" and graph.capacity(province) != 1
        ]
        assert not roomy, f"{graph.name}: coastal provinces with room for two: {roomy}"


# -- the sweep: every order type, in every phase ------------------------------
#
# A written base strength binds everywhere an order or a reference can carry one, not
# only where movement reads it. Four further sites bind a spec and are all
# hand-writable even where enumeration never offers the bare form: the retreat
# binder, the standard adjustment removal, Beleriand's own winter removal, and
# the Beleriand support-subject owner lookup.
# :func:`test_no_engine_site_reads_a_bare_spec_as_any_base_strength` is the guard that
# keeps them swept.


def test_a_bare_retreat_order_does_not_bind_a_stronger_unit() -> None:
    """Bind nothing when a bare retreat order names a base strength the pool lacks.

    ``_fitting_dislodged_units`` sees only a dislodged 2A, so the bare form,
    which enumeration never offers but a player can write, matches no unit.
    Under strict designation the province-only fallback cannot hand it back
    either.
    """
    from entente.engine.adjudicator.common import Dislodgement, PhaseResult
    from entente.engine.adjudicator.retreats import adjudicate_retreats
    from entente.engine.orders import parse_retreat_order

    strong = Unit("MORGOTH", parse_unit_spec("2A ANF")[0], parse_unit_spec("2A ANF")[1], 2)
    movement = PhaseResult(
        resolutions=(),
        new_state=State.of("beleriand", "S25M", [Unit("NAUGRIM", *parse_unit_spec("A LAD"))]),
        dislodgements=(Dislodgement(unit=strong, dislodger_origin="DNN", by_convoy=False),),
        embattled=frozenset(),
    )
    result = adjudicate_retreats(
        BELERIAND, movement, {"MORGOTH": [parse_retreat_order("A ANF - EIT")]}, graph=GRAPH_BEL
    )
    assert result.resolution_of("A ANF - EIT") is Resolution.FAILS
    assert not [u for u in result.new_state.units if u.power == "MORGOTH"]
    # the control: naming the base strength it actually has still retreats it
    ok = adjudicate_retreats(
        BELERIAND, movement, {"MORGOTH": [parse_retreat_order("2A ANF - EIT")]}, graph=GRAPH_BEL
    )
    assert ok.resolution_of("2A ANF - EIT") is Resolution.SUCCEEDS


def test_no_engine_site_reads_a_bare_spec_as_any_base_strength() -> None:
    """Keep the sweep swept by grepping the source for the wildcard shape.

    Every base-strength comparison in the engine and the variants normalises an
    unwritten prefix to 1 -- the ``(strength or 1)`` shape -- rather than
    treating ``None`` as a wildcard. ``strength is None`` in a matching
    position is exactly the reading the last four tests forbid, so this scan
    fails on it anywhere under ``src/entente``, in the same change that would
    introduce it.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "entente"
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "strength is None" in line
    ]
    assert not offenders, (
        "a bare spec must mean base strength 1, never any base strength (author 2026-07-20 "
        f"'A is same as 1A'); rewrite as (strength or 1) == ...: {offenders}"
    )
