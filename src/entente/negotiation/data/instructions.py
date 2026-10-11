"""Turn Peskov conversations into instruction-tuning pairs.

**Honest scope decision (read before wiring this into anything else):** the
released corpus (:mod:`entente.negotiation.data.peskov`) has no order data
aligned to its messages -- real players' actual moves were never captured
alongside what they said. So the conditioning signal here is *not*
:class:`entente.negotiation.intents.Intent` (which requires concrete
``proposer_action``/``ally_action`` order-sets grounded in a piKL candidate
set): fabricating plausible-looking orders the corpus never recorded would put
invented data in a training set and let a later result rest on it. Instead each
example conditions on the one structured axis
the corpus *does* provide reliably for every message: speaker, receiver, the
game's own phase vocabulary (season + year, e.g. "Spring 1901"), and the
sender's own truthful/deceptive stance (``sender_labels`` -- exactly CICERO's
"intended deception" axis, and the reason this dataset exists). The
fine-tuned model therefore learns Diplomacy-flavored register and
stance-conditioning, not order semantics it was never shown.

At inference time, :class:`entente.negotiation.generate.LocalLoraProvider`
prompts the *same* fine-tuned adapter with the engine-grounded ``Intent``
instead (reusing ``generate._build_prompt``, exactly like
``AnthropicProvider``) -- so what transfers from fine-tuning to serving is
style and stance-steerability, not a promise that the model has "seen" the
kind of input it will actually receive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from entente.negotiation.data.peskov import PeskovConversation

_STANCE_PROMPT = (
    "You are drafting a short in-character Diplomacy press message. "
    "{speaker} is writing to {receiver} during {phase}. "
    "Write a {stance} message consistent with that stance -- 1-3 sentences, "
    "plausible in-game diplomatic tone, signed as {speaker}."
)


@dataclass(frozen=True, slots=True)
class InstructionExample:
    """Hold one instruction-tuning pair.

    The prompt names speaker, receiver, phase and the sender's own stance, and
    the completion is the real message text. ``truthful``, ``game_id`` and
    ``message_index`` are carried through for filtering and provenance; the
    training loop never reads them.
    """

    prompt: str
    completion: str
    truthful: bool
    game_id: int
    message_index: int


def _phase_label(season: str, year: str) -> str:
    return f"{season} {year}"


def conversation_to_examples(conversation: PeskovConversation) -> Iterator[InstructionExample]:
    """One example per message in ``conversation`` (module docstring)."""
    for position, message in enumerate(conversation.messages):
        truthful = conversation.sender_truthful[position]
        prompt = _STANCE_PROMPT.format(
            speaker=conversation.speakers[position],
            receiver=conversation.receivers[position],
            phase=_phase_label(conversation.seasons[position], conversation.years[position]),
            stance="truthful" if truthful else "deceptive",
        )
        yield InstructionExample(
            prompt=prompt,
            completion=message,
            truthful=truthful,
            game_id=conversation.game_id,
            message_index=conversation.relative_message_index[position],
        )


def to_instruction_examples(
    conversations: Iterable[PeskovConversation],
) -> Iterator[InstructionExample]:
    """Flatten many conversations into their instruction examples, in order."""
    for conversation in conversations:
        yield from conversation_to_examples(conversation)


def write_instruction_jsonl(examples: Iterable[InstructionExample], path: str | Path) -> int:
    """Write the examples as one sorted-key JSON object per line, and count them.

    Sorted keys and one record per line make the file diffable, which is the
    convention every result file in the project follows.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "prompt": example.prompt,
                        "completion": example.completion,
                        "truthful": example.truthful,
                        "game_id": example.game_id,
                        "message_index": example.message_index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
    return count


def read_instruction_jsonl(path: str | Path) -> Iterator[InstructionExample]:
    """Read back what :func:`write_instruction_jsonl` wrote, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            yield InstructionExample(**data)


__all__ = [
    "InstructionExample",
    "conversation_to_examples",
    "read_instruction_jsonl",
    "to_instruction_examples",
    "write_instruction_jsonl",
]
