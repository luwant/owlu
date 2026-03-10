from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from owlu.llm_phrase_generator import (
    CandidatePhrase,
    LLMOutputError,
    LLMPhraseGenerator,
    OWLUConfig,
    get_api_key,
)
from owlu.semantic_matcher import SemanticMatcher


class _FakeMessage:
    def __init__(self, content: str | None):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, outputs: list[str | None]):
        self._outputs = outputs
        self.calls = 0

    def create(self, **kwargs):  # noqa: ANN003
        idx = self.calls
        self.calls += 1
        out = self._outputs[idx] if idx < len(self._outputs) else self._outputs[-1]
        return _FakeResponse(out)


class _FakeChat:
    def __init__(self, outputs: list[str | None]):
        self.completions = _FakeCompletions(outputs)


class _FakeClient:
    def __init__(self, outputs: list[str | None]):
        self.chat = _FakeChat(outputs)


@pytest.fixture()
def cfg_path() -> str:
    return str(Path("e:/lwt/workspace/owlu/configs/owlu.yaml"))


@pytest.fixture()
def cfg(cfg_path: str) -> OWLUConfig:
    return OWLUConfig.from_yaml(cfg_path)


def test_config_load(cfg: OWLUConfig):
    assert cfg.llm_base_url == "https://api.deepseek.com"
    assert cfg.llm_model == "deepseek-chat"
    assert cfg.merge_threshold == pytest.approx(0.80)
    assert cfg.multi_sample_k == 3


def test_env_key_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        get_api_key()


def test_generate_json_contract(cfg: OWLUConfig):
    content = '{"summary":"s","phrases":["cyber threat intel","malware family"],"evidence":["ev1"]}'
    gen = LLMPhraseGenerator(cfg, client=_FakeClient([content]))
    out = gen.generate("text", "doc-1")
    assert len(out) == 2
    assert out[0].summary == "s"
    assert out[0].source_doc_id == "doc-1"
    assert out[0].pass_id == 1


def test_generate_handles_invalid_json(cfg: OWLUConfig):
    gen = LLMPhraseGenerator(cfg, client=_FakeClient(["not-json"]))
    with pytest.raises(LLMOutputError):
        gen.generate("text", "doc-1")


def test_normalize_pipeline(cfg: OWLUConfig):
    matcher = SemanticMatcher(cfg)
    normalized = matcher.normalize("The   Running attacks in systems")
    assert normalized == "runn attack system"


def test_match_merge_threshold(cfg: OWLUConfig):
    matcher = SemanticMatcher(cfg)
    phrase = CandidatePhrase(
        text="malware analysis",
        raw_text="malware analysis",
        source_doc_id="d1",
        timestamp=datetime.now(timezone.utc),
        agreement=0.9,
    )
    labels = {"l1": "malware analysis", "l2": "network defense"}
    result = matcher.match(phrase, labels)
    assert result.action == "merge_pre"
    assert result.target_label == "l1"


def test_preliminary_decision_without_global_state(cfg: OWLUConfig):
    matcher = SemanticMatcher(cfg)
    assert matcher.preliminary_decide(0.40, 1.0) == "novel_pre"
    assert matcher.preliminary_decide(0.60, 1.0) == "hold_pre"
    assert matcher.preliminary_decide(0.90, 0.80) == "merge_pre"


def test_uncertain_trigger(cfg: OWLUConfig):
    gen = LLMPhraseGenerator(cfg, client=_FakeClient(['{"summary":"x","phrases":["a"],"evidence":[]}']))
    assert gen.should_trigger_uncertain(0.30, 0.10) is True
    assert gen.should_trigger_uncertain(0.90, 0.80) is True
    assert gen.should_trigger_uncertain(0.90, 0.70) is False


def test_multi_sample_agreement(cfg: OWLUConfig):
    outputs = [
        '{"summary":"s","phrases":["threat intel","malware"],"evidence":[]}',
        '{"summary":"s","phrases":["threat intel"],"evidence":[]}',
        '{"summary":"s","phrases":["threat intel","apt group"],"evidence":[]}',
    ]
    gen = LLMPhraseGenerator(cfg, client=_FakeClient(outputs))
    out = gen.multi_sample_aggregate("text", "doc-2", k=3)
    by_text = {c.text.lower(): c for c in out}
    assert "threat intel" in by_text
    assert by_text["threat intel"].agreement == pytest.approx(1.0)
    assert by_text["threat intel"].pass_id == 2


def test_gate_rule_with_agreement(cfg: OWLUConfig):
    matcher = SemanticMatcher(cfg)
    phrase = CandidatePhrase(
        text="threat intel",
        raw_text="threat intel",
        source_doc_id="d3",
        timestamp=datetime.now(timezone.utc),
        agreement=0.5,
    )
    labels = {"l1": "threat intel"}
    result = matcher.match(phrase, labels)
    assert result.similarity >= cfg.merge_threshold
    assert result.action == "hold_pre"
