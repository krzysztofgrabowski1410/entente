"""Pin the Peskov reader and the instruction-pair conversion built on it.

The schema is checked over the committed fixture: two real, CC-BY-licensed
conversations trimmed from the released corpus, recorded in ``data/README.md``.
Also covered are tolerance of malformed lines and the instruction-pair
conversion. Torch-free, so these run in the default environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entente.negotiation.data.instructions import (
    InstructionExample,
    conversation_to_examples,
    read_instruction_jsonl,
    to_instruction_examples,
    write_instruction_jsonl,
)
from entente.negotiation.data.peskov import (
    PeskovConversation,
    PeskovJSONLSource,
    conversation_from_json,
    iter_conversations,
)

FIXTURE = "tests/negotiation/fixtures/peskov_sample.jsonl"


def _fixture_conversations() -> list[PeskovConversation]:
    return list(PeskovJSONLSource(FIXTURE))


def _fixture_lines() -> list[str]:
    return Path(FIXTURE).read_text(encoding="utf-8").splitlines()


def _fixture_raw(index: int = 0) -> dict:  # type: ignore[type-arg]
    return json.loads(_fixture_lines()[index])


class TestPeskovJSONLSource:
    """Reading the released corpus files."""

    def test_reads_the_committed_fixture(self) -> None:
        conversations = _fixture_conversations()
        assert len(conversations) == 2
        source = PeskovJSONLSource(FIXTURE)
        list(source)
        assert not source.skipped

    def test_first_conversation_matches_the_released_record(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        assert game12.players == ("GERMANY", "RUSSIA")
        assert len(game12) == 6
        assert game12.speakers[0] == "RUSSIA"
        assert game12.receivers[0] == "GERMANY"
        assert game12.messages[0].startswith("Hey Germany!")

    def test_sender_truthful_carries_the_deceptive_message(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        # message index 1 is the one the sender themself flagged as a lie.
        assert game12.sender_truthful[1] is False
        assert all(t is True for i, t in enumerate(game12.sender_truthful) if i != 1)

    def test_receiver_truthful_maps_noannotation_to_none(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        assert game12.receiver_truthful[4] is None
        assert all(v is not None for i, v in enumerate(game12.receiver_truthful) if i != 4)

    def test_seasons_and_years_are_carried_through(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        assert game12.seasons == ("Spring", "Spring", "Winter", "Winter", "Spring", "Fall")
        assert game12.years == ("1901", "1901", "1901", "1901", "1902", "1902")

    def test_limit_caps_conversations(self) -> None:
        assert len(list(PeskovJSONLSource(FIXTURE, limit=1))) == 1

    def test_malformed_lines_are_skipped_not_raised(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        good = _fixture_raw()
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            json.dumps(good) + "\n" + "not json at all\n" + json.dumps({"game_id": 1}) + "\n",
            encoding="utf-8",
        )
        source = PeskovJSONLSource(path)
        conversations = list(source)
        assert len(conversations) == 1
        assert len(source.skipped) == 2

    def test_iter_conversations_parses_a_stream(self) -> None:
        lines = _fixture_lines()
        stream = iter([lines[0], "", lines[1]])
        assert len(list(iter_conversations(iter(stream)))) == 2


class TestBoolCoercion:
    """The two forms the corpus writes its labels in."""

    @pytest.mark.parametrize("value", ["true", "TRUE", "True"])
    def test_string_true_forms_accepted(self, value: str) -> None:
        raw = _fixture_raw()
        raw["sender_labels"] = [value] * len(raw["messages"])
        conversation = conversation_from_json(raw)
        assert all(conversation.sender_truthful)

    def test_invalid_sender_label_raises(self) -> None:
        raw = _fixture_raw()
        raw["sender_labels"] = ["maybe"] * len(raw["messages"])
        with pytest.raises(ValueError, match="sender_labels"):
            conversation_from_json(raw)


class TestConversationValidation:
    """What a malformed conversation record is rejected for."""

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="mismatched lengths"):
            PeskovConversation(
                game_id=1,
                players=("A", "B"),
                speakers=("A",),
                receivers=("B", "A"),
                messages=("hi", "there"),
                sender_truthful=(True, True),
                receiver_truthful=(True, True),
                seasons=("Spring", "Spring"),
                years=("1901", "1901"),
                relative_message_index=(0, 1),
            )


class TestInstructionExamples:
    """Turning conversations into instruction-tuning pairs."""

    def test_one_example_per_message(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        examples = list(conversation_to_examples(game12))
        assert len(examples) == 6
        assert all(isinstance(e, InstructionExample) for e in examples)

    def test_prompt_names_speaker_receiver_and_phase(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        first = next(conversation_to_examples(game12))
        assert "RUSSIA" in first.prompt
        assert "GERMANY" in first.prompt
        assert "Spring 1901" in first.prompt

    def test_stance_reflects_sender_truthful(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        examples = list(conversation_to_examples(game12))
        assert "deceptive" in examples[1].prompt
        assert examples[1].truthful is False
        assert "truthful" in examples[0].prompt
        assert examples[0].truthful is True

    def test_completion_is_the_verbatim_message(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        first = next(conversation_to_examples(game12))
        assert first.completion == game12.messages[0]

    def test_message_index_matches_relative_index(self) -> None:
        game12 = next(c for c in _fixture_conversations() if c.game_id == 12)
        examples = list(conversation_to_examples(game12))
        assert [e.message_index for e in examples] == list(game12.relative_message_index)

    def test_to_instruction_examples_flattens_every_conversation(self) -> None:
        examples = list(to_instruction_examples(_fixture_conversations()))
        assert len(examples) == 12
        assert {e.game_id for e in examples} == {7, 12}


class TestInstructionJsonlRoundTrip:
    """Writing instruction pairs out and reading them back."""

    def test_write_then_read_round_trips(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        examples = list(to_instruction_examples(_fixture_conversations()))
        path = tmp_path / "instructions.jsonl"
        count = write_instruction_jsonl(examples, path)
        assert count == len(examples)
        restored = list(read_instruction_jsonl(path))
        assert restored == examples

    def test_write_creates_parent_directories(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "nested" / "dir" / "instructions.jsonl"
        write_instruction_jsonl(list(to_instruction_examples(_fixture_conversations())), path)
        assert path.exists()
