"""The adjudication machinery both engines share: parsing, tables, ``Eval``, assembly.

Phase 0 parses and validates the sheets into canonical records, Phase 1 builds
the static order-dependency graph and its SCC condensation, ``Evaluator``
computes the derived predicates of docs/adjudication.tex Sec. 5.1, and the
result types and assembly turn a settled labeling into a ``PhaseResult``. Both
implementations -- the stable-labeling enumerator (``csp``) and the staged
resolver (``resolver``) -- consume exactly this package: they differ only in
how they *find* the move labeling; the derived stationary statuses and the
result assembly are the Sec. 5.1 definition, shared verbatim, which is what
makes the differential test a test of the labeling search.

The package is split by *role* -- what makes a piece change -- not by pipeline
stage, because the stages share too much to cut there: ``geometry`` and
``matching`` are used by Phase 0 and by resolution alike. Dependencies run one
way and there are no cycles:

    results, geometry, matching        (leaves: data and pure predicates)
        <- phase0                      (Sec. 1: sheets -> records)
        <- phase1                      (Sec. 6: dependency digraph, Tarjan)
        <- tables                      (Sec. 3/Sec. 4 static precomputation)
        <- evaluator                   (Sec. 5.1 Eval)
        <- selection, assembly         (Sec. 5.2 policy; PhaseResult)

This module re-exports the whole surface, so ``from
entente.engine.adjudicator.common import X`` works for every X, including the
underscore names the test suite reaches for. New code imports from the
submodules; the flat names are the compatibility contract that keeps the
package's internal layout free to change.
"""

from __future__ import annotations

from entente.engine.adjudicator.common.assembly import assemble
from entente.engine.adjudicator.common.evaluator import Evaluator, final_walk_victims
from entente.engine.adjudicator.common.geometry import (
    potential_chain,
    sea_chain_ends,
    sea_chain_exists,
)
from entente.engine.adjudicator.common.matching import ref_matches_unit
from entente.engine.adjudicator.common.phase0 import phase0
from entente.engine.adjudicator.common.phase1 import (
    condensation_order,
    dependency_graph,
    entry_province,
)
from entente.engine.adjudicator.common.results import (
    AdjudicationError,
    Dislodgement,
    OrderRecord,
    OrderResolution,
    Phase0,
    PhaseResult,
    Resolution,
)
from entente.engine.adjudicator.common.selection import (
    MAX_ENUMERATED,
    commit_cycle_policy,
    possible_chain_for,
    resolve_residue,
)
from entente.engine.adjudicator.common.tables import (
    Tables,
    WalkGroup,
    build_tables,
    find_subject_record,
    residue_adjacency,
)

# Underscore names the test suite imports through this module (it tests Phase-0
# internals directly). They are re-exported so the package's internal layout can
# change without breaking those imports; F401 is suppressed because "unused" is
# the point of a compatibility surface. New code imports from the submodule.
from entente.engine.adjudicator.common.matching import (  # noqa: F401  isort: skip
    _ref_special_ok,
    _spec_matches_unit,
    _unit_key,
)
from entente.engine.adjudicator.common.phase0 import (  # noqa: F401  isort: skip
    _kind_can_reach,
    _Phase0State,
    _resolve_subject_slot,
    _validate_bar,
    _validate_convoy,
    _validate_dislodge,
    _validate_move,
    _validate_support,
)
from entente.engine.adjudicator.common.results import (  # noqa: F401  isort: skip
    _UNORDERED_SHEET_POS,
)
from entente.engine.adjudicator.common.selection import (  # noqa: F401  isort: skip
    _pure_vacancy_rotation,
)
from entente.engine.adjudicator.common.tables import (  # noqa: F401  isort: skip
    _order_equivalent,
    _resolve_victims,
    _subject_record,
)

__all__ = [
    "MAX_ENUMERATED",
    "AdjudicationError",
    "Dislodgement",
    "Evaluator",
    "OrderRecord",
    "OrderResolution",
    "Phase0",
    "PhaseResult",
    "Resolution",
    "Tables",
    "WalkGroup",
    "assemble",
    "build_tables",
    "commit_cycle_policy",
    "condensation_order",
    "dependency_graph",
    "entry_province",
    "final_walk_victims",
    "find_subject_record",
    "phase0",
    "possible_chain_for",
    "potential_chain",
    "ref_matches_unit",
    "residue_adjacency",
    "resolve_residue",
    "sea_chain_ends",
    "sea_chain_exists",
]
