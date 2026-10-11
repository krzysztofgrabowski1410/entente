"""Pin `entente negotiate-demo` end to end.

The end-to-end wiring through `entente.cli.main` (proving the `importlib`
registration actually works) is covered in `tests/test_cli.py`, which must
not import `entente.negotiation` itself -- see `test_isolation.py`. This file
tests `entente.negotiation.cli` directly, including error paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from entente.negotiation.cli import _cmd_negotiate_demo
from entente.negotiation.generate import StubProvider
from entente.negotiation.parse import parse_message

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "src/entente/negotiation/scenarios"


def _args(scenario: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(scenario=str(scenario), out_dir=str(out_dir))


class TestNegotiateDemo:
    """The demo subcommand, run end to end over the committed scenarios."""

    def test_writes_a_stub_transcript_with_the_expected_shape(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ENTENTE_DIALOGUE_LORA_DIR", raising=False)
        code = _cmd_negotiate_demo(_args(SCENARIOS_DIR / "s1.json", tmp_path))
        assert code == 0

        out = capsys.readouterr().out
        assert "P(true move of GERMANY)" in out
        assert "ANTHROPIC_API_KEY not set" in out
        assert "ENTENTE_DIALOGUE_LORA_DIR not set" in out

        transcript = json.loads((tmp_path / "s1.json").read_text(encoding="utf-8"))
        assert transcript["proposer"] == "GERMANY"
        assert transcript["ally"] == "RUSSIA"
        assert transcript["regenerated_message"]["provider"] == "stub"
        assert (
            transcript["prediction"]["conditioned_probability_of_true_move"]
            > (transcript["prediction"]["baseline_probability_of_true_move"])
        )
        # the regenerated message must parse back to the same intent as the original
        assert parse_message(transcript["regenerated_message"]["text"]) == parse_message(
            transcript["incoming_message"]
        )
        assert not (tmp_path / "s1.anthropic.json").exists()
        assert not (tmp_path / "s1.lora.json").exists()

    def test_reports_a_missing_scenario_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _cmd_negotiate_demo(_args(tmp_path / "missing.json", tmp_path))
        assert code == 1
        assert "cannot read" in capsys.readouterr().err

    def test_reports_an_unparseable_scenario_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scenario = json.loads((SCENARIOS_DIR / "s1.json").read_text(encoding="utf-8"))
        scenario["message"] = "not a valid proposal at all"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(scenario), encoding="utf-8")

        code = _cmd_negotiate_demo(_args(bad, tmp_path))
        assert code == 1
        assert "cannot parse scenario message" in capsys.readouterr().err

    def test_reports_a_malformed_scenario_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        code = _cmd_negotiate_demo(_args(bad, tmp_path))
        assert code == 1
        assert "cannot load" in capsys.readouterr().err

    def test_notes_a_failure_generating_the_real_model_transcript(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("anthropic is installed; a fake key would attempt a real network call")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-not-a-real-key")
        code = _cmd_negotiate_demo(_args(SCENARIOS_DIR / "s1.json", tmp_path))
        assert code == 1
        assert "ANTHROPIC_API_KEY is set but generation failed" in capsys.readouterr().err

    def test_notes_a_failure_generating_the_lora_transcript(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ENTENTE_DIALOGUE_LORA_DIR", str(tmp_path / "no-such-adapter"))
        code = _cmd_negotiate_demo(_args(SCENARIOS_DIR / "s1.json", tmp_path))
        assert code == 1
        assert "ENTENTE_DIALOGUE_LORA_DIR is set but generation failed" in capsys.readouterr().err


def test_stub_provider_is_the_only_thing_the_demo_needs_offline() -> None:
    # A cheap reminder that the demo's default path never touches `anthropic`.
    assert StubProvider().generate is not None


@pytest.mark.finetune
class TestLoraTranscript:
    """The demo's adapter transcript, written when an adapter directory is set.

    Needs the `finetune` group and network access for the tiny test model,
    cached after the first fetch. The provider itself is covered in
    tests/negotiation/test_generate_lora.py.
    """

    def test_writes_a_lora_transcript_when_configured(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from entente.negotiation.finetune.config import DialogueLoraConfig
        from entente.negotiation.finetune.train import train
        from entente.training.config import load_config

        cfg = load_config(
            DialogueLoraConfig,
            "src/entente/negotiation/finetune/configs/smoke.yaml",
            overrides=[f"out_dir={tmp_path / 'adapter_out'}", "optim.max_steps=4"],
        )
        result = train(cfg)

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ENTENTE_DIALOGUE_LORA_DIR", str(result.adapter_dir))
        code = _cmd_negotiate_demo(_args(SCENARIOS_DIR / "s1.json", tmp_path))
        assert code == 0
        assert "wrote" in capsys.readouterr().out

        transcript = json.loads((tmp_path / "s1.lora.json").read_text(encoding="utf-8"))
        assert transcript["provider"] == "lora"
        assert transcript["base_model"] == "hf-internal-testing/tiny-random-gpt2"
        assert isinstance(transcript["regenerated_message"], str)
