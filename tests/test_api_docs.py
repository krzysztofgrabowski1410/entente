"""Check that every name the API reference documents resolves to a real export.

Each page under docs/api/ lists the names it documents in a ``<!-- names ... -->``
block; importing every one of them makes a rename the docs miss fail here
instead of rotting.

A name whose module needs an absent optional dependency (torch, pyarrow,
fastapi, ...) is skipped -- but a wrong ``entente.*`` module path is a failure,
never a skip, so a typo in a page cannot hide.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[1] / "docs" / "api"
_NAMES_BLOCK = re.compile(r"<!--\s*names\s*(.*?)-->", re.DOTALL)
_NAME_LINE = re.compile(r"^([\w.]+):(\w+)$")


def _documented_names() -> list[tuple[str, str, str]]:
    """(page, module, attribute) for every name in every page's names block."""
    out: list[tuple[str, str, str]] = []
    for page in sorted(API_DIR.glob("*.md")):
        if page.name == "README.md":
            continue  # the index documents no names; its block is illustrative
        for block in _NAMES_BLOCK.findall(page.read_text(encoding="utf-8")):
            for raw in block.splitlines():
                line = raw.strip()
                if not line:
                    continue
                match = _NAME_LINE.match(line)
                assert match, f"{page.name}: malformed name line {line!r}"
                out.append((page.name, match.group(1), match.group(2)))
    return out


NAMES = _documented_names()


def test_pages_document_a_reasonable_number_of_names() -> None:
    # non-vacuous guard: the pages exist and carry real blocks
    assert len(NAMES) > 60
    assert {page for page, _, _ in NAMES} >= {"engine.md", "agents.md", "search.md"}


@pytest.mark.parametrize(
    ("module", "attr"),
    [(m, a) for _, m, a in NAMES],
    ids=[f"{m}:{a}" for _, m, a in NAMES],
)
def test_documented_name_is_importable(module: str, attr: str) -> None:
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing.split(".")[0] == "entente":
            raise AssertionError(f"documented module {module!r} does not exist") from error
        pytest.skip(f"optional dependency {missing!r} absent for {module}")
    assert hasattr(mod, attr), f"{module} has no public name {attr!r} (docs/api out of date)"


def test_engine_core_names_always_resolve() -> None:
    # The stdlib engine surface never needs an optional dependency, so these
    # must resolve in every environment (keeps the skip path from hiding a
    # core-API regression).
    from entente.engine.adjudicator import adjudicate  # noqa: F401
    from entente.engine.game import Game, GameError  # noqa: F401
    from entente.engine.orders import format_order, parse_order  # noqa: F401
