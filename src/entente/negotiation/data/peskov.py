"""Read the Peskov et al. (2020) "It Takes Two to Lie" deception corpus.

The corpus is CC-BY 4.0; its licence and provenance are recorded in
``data/README.md``, on the same discipline as :mod:`entente.data.paquette`.

The corpus ships three JSONL files (``train``/``validation``/``test``), one
line per full game-long *dialogue between two players*: parallel per-message
arrays, not per-message objects. The schema below is the documented subset
this module reads (the released files additionally carry ``game_score``,
``score_delta`` and ``absolute_message_index``, which this pipeline has no
use for and drops):

    {"game_id": 12, "players": ["germany", "russia"],
     "speakers":  ["russia", "germany", ...],
     "receivers": ["germany", "russia", ...],
     "messages":  ["Hey Germany! ...", "Hello! ...", ...],
     "sender_labels":   [true, false, ...],
     "receiver_labels": [true, "NOANNOTATION", ...],
     "seasons": ["Spring", "Winter", ...], "years": ["1901", "1901", ...],
     "relative_message_index": [0, 1, ...]}

``sender_labels`` is the sender's own truthful(``true``)/deceptive(``false``)
selection for that message (bool in the released files; the corpus's own
README notes it "can be bool or string format", so :func:`_coerce_bool`
accepts both). ``receiver_labels`` is the receiver's *perceived* truthfulness
of the same message, or the string ``"NOANNOTATION"`` when no perception was
recorded (~9% of messages) -- read here as ``bool | None``. Speaker/receiver
country names are upper-cased to match this project's own power-name
convention (``entente.engine`` powers are e.g. ``"GERMANY"``) even though
nothing here talks to the engine (module docstring of
:mod:`entente.negotiation.data.instructions`).

This module never fetches the corpus itself -- see ``data/README.md`` for the
download commands (system certs, like the Paquette mirror) and the CC-BY 4.0
attribution/restriction notes that must accompany any redistribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class PeskovConversation:
    """Hold one full game-long dialogue between two named players.

    Every field below is a parallel per-message tuple except ``game_id`` and
    ``players``, which describe the dialogue as a whole.
    """

    game_id: int
    players: tuple[str, ...]
    speakers: tuple[str, ...]
    receivers: tuple[str, ...]
    messages: tuple[str, ...]
    sender_truthful: tuple[bool, ...]
    receiver_truthful: tuple[bool | None, ...]
    seasons: tuple[str, ...]
    years: tuple[str, ...]
    relative_message_index: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        lengths = {
            len(self.speakers),
            len(self.receivers),
            len(self.messages),
            len(self.sender_truthful),
            len(self.receiver_truthful),
            len(self.seasons),
            len(self.years),
            len(self.relative_message_index),
        }
        if len(lengths) > 1:
            raise ValueError(
                f"conversation {self.game_id}: parallel arrays have mismatched lengths {lengths}"
            )

    def __len__(self) -> int:
        return len(self.messages)


def _coerce_bool(value: object, *, field_name: str) -> bool:
    """Read a label as a boolean.

    The released files use both a JSON bool and the string alternative the
    corpus README documents, ``"true"``/``"false"`` in any case.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field_name}: cannot interpret {value!r} as a boolean")


def _coerce_receiver_label(value: object) -> bool | None:
    if isinstance(value, str) and value.upper() == "NOANNOTATION":
        return None
    return _coerce_bool(value, field_name="receiver_labels")


def conversation_from_json(raw: dict[str, Any]) -> PeskovConversation:
    """Parse one saved-conversation object into a :class:`PeskovConversation`.

    Only the fields the module docstring lists are read; the rest are dropped.
    """
    messages = raw["messages"]
    relative_index = raw.get("relative_message_index")
    return PeskovConversation(
        game_id=int(raw["game_id"]),
        players=tuple(str(p).upper() for p in raw.get("players", [])),
        speakers=tuple(str(s).upper() for s in raw["speakers"]),
        receivers=tuple(str(r).upper() for r in raw["receivers"]),
        messages=tuple(str(m) for m in messages),
        sender_truthful=tuple(
            _coerce_bool(v, field_name="sender_labels") for v in raw["sender_labels"]
        ),
        receiver_truthful=tuple(_coerce_receiver_label(v) for v in raw["receiver_labels"]),
        seasons=tuple(str(s) for s in raw["seasons"]),
        years=tuple(str(y) for y in raw["years"]),
        relative_message_index=(
            tuple(int(i) for i in relative_index)
            if relative_index is not None
            else tuple(range(len(messages)))
        ),
    )


def iter_conversations(lines: Iterator[str]) -> Iterator[PeskovConversation]:
    """Parse a stream of JSONL lines into conversations, skipping blank lines.

    A malformed line raises. Callers that must survive one use
    :class:`PeskovJSONLSource`, which records and skips them instead.
    """
    for line in lines:
        stripped = line.strip()
        if stripped:
            yield conversation_from_json(json.loads(stripped))


class PeskovJSONLSource:
    """Read conversations from one ``.jsonl`` file.

    The corpus ships ``train``, ``validation`` and ``test`` as three separate
    plain files rather than one archive, so this takes a single path: point it
    at whichever split you need. Malformed lines are counted in :attr:`skipped`
    rather than raised, as in
    :class:`entente.data.paquette.PaquetteJSONLSource`.
    """

    def __init__(self, path: str | Path, *, limit: int | None = None) -> None:
        self.path = Path(path)
        self.limit = limit
        self.skipped: list[str] = []

    def __iter__(self) -> Iterator[PeskovConversation]:
        self.skipped = []
        count = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if self.limit is not None and count >= self.limit:
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    conversation = conversation_from_json(json.loads(stripped))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    self.skipped.append(f"{type(error).__name__}: {error}")
                    continue
                count += 1
                yield conversation


__all__ = [
    "PeskovConversation",
    "PeskovJSONLSource",
    "conversation_from_json",
    "iter_conversations",
]
