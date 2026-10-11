"""Package-level smoke tests."""

import entente
from entente.engine.rules import STANDARD, Ruleset


def test_version() -> None:
    assert entente.__version__ == "0.0.1"


def test_ruleset_protocol_is_runtime_checkable() -> None:
    # The protocol covers the whole hook surface a variant can change
    # (docs/adjudication.tex Sec. 7), so satisfying it takes a real ruleset: the
    # standard one passes, and a stub carrying only a name and a symmetry flag
    # does not.
    assert isinstance(STANDARD, Ruleset)

    class Incomplete:
        @property
        def name(self) -> str:
            return "fake"

        @property
        def power_symmetric(self) -> bool:
            return True

    assert not isinstance(Incomplete(), Ruleset)
