from __future__ import annotations

import os
from pathlib import Path

import pytest

from owlu.llm_phrase_generator import LLMPhraseGenerator, OWLUConfig


@pytest.mark.integration
def test_live_generate_smoke():
    """Live DeepSeek API smoke test.

    Default behavior: skip unless OWLU_RUN_LIVE=1.
    """
    if os.getenv("OWLU_RUN_LIVE", "0") != "1":
        pytest.skip("Set OWLU_RUN_LIVE=1 to run live API test.")

    cfg = OWLUConfig.from_yaml(str(Path("e:/lwt/workspace/owlu/configs/owlu.yaml")))
    generator = LLMPhraseGenerator(cfg)
    text = (
        "Threat actors coordinate a multi-stage phishing campaign with custom loader "
        "malware and credential exfiltration."
    )
    results = generator.generate(text, "live-test-doc")
    assert len(results) >= 1
    assert all(r.text.strip() for r in results)
    assert all(r.pass_id == 1 for r in results)

