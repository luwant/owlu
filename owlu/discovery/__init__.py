"""Candidate Discovery Module — discovers label candidates from low-confidence samples.

Sub-modules:
    gate            — LTCE confidence gate (controls LLM invocation)
    phrase_generator — LLM-based phrase extraction
    matcher         — Lightweight semantic matching & preliminary decision
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common.types import (
    CandidatePhrase,
    GateDecision,
    MatchResult,
    OWLUConfig,
)
from .gate import LtceGate
from .matcher import SemanticMatcher
from .phrase_generator import LLMPhraseGenerator


class CandidateDiscovery:
    """Facade: from low-confidence documents, produce MatchResult candidates.

    Orchestrates three internal steps:
        1.  LtceGate — decide if a document needs LLM exploration
        2.  LLMPhraseGenerator — extract candidate phrases via DeepSeek
        3.  SemanticMatcher — match phrases against existing label inventory
    """

    def __init__(
        self,
        config: OWLUConfig,
        label_inventory: Mapping[str, str],
        *,
        llm_client: Any | None = None,
        fixed_threshold: float | None = None,
    ):
        self.config = config
        self.label_inventory = dict(label_inventory)

        self.gate = LtceGate(
            recognition_floor=config.recognition_floor,
            adaptive_percentile=config.adaptive_percentile,
            fixed_threshold=fixed_threshold,
        )
        self.generator = LLMPhraseGenerator(config, client=llm_client)
        self.matcher = SemanticMatcher(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate_gate(self, validation_logits: Sequence[Sequence[float]]) -> float:
        """Calibrate LTCE confidence gate from in-distribution logits."""
        return self.gate.calibrate(validation_logits)

    def discover(
        self,
        doc_id: str,
        text: str,
        logits: Sequence[float],
    ) -> list[MatchResult]:
        """End-to-end discovery: gate → extract → match.

        Returns
        -------
        list[MatchResult]
            Empty list when the gate does not trigger.
        """
        decision = self.gate.evaluate(logits)
        if not decision.should_invoke_llm:
            return []

        phrases = self.generator.generate(text, doc_id)
        return [
            self.matcher.match(phrase, self.label_inventory) for phrase in phrases
        ]

    def discover_uncertain(
        self,
        doc_id: str,
        text: str,
        logits: Sequence[float],
        top1_score: float,
        top2_score: float,
    ) -> list[MatchResult]:
        """Discovery with uncertainty-triggered multi-sample aggregation."""
        decision = self.gate.evaluate(logits)
        if not decision.should_invoke_llm:
            return []

        if self.generator.should_trigger_uncertain(top1_score, top2_score):
            phrases = self.generator.multi_sample_aggregate(text, doc_id)
        else:
            phrases = self.generator.generate(text, doc_id)

        return [
            self.matcher.match(phrase, self.label_inventory) for phrase in phrases
        ]

    def batch_discover(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        batch_logits: Sequence[Sequence[float]],
    ) -> dict[str, list[MatchResult]]:
        """Batch discovery across multiple documents.

        Returns
        -------
        dict[str, list[MatchResult]]
            Mapping from doc_id to its match results. Documents that are
            not triggered are omitted from the result dict.
        """
        triggered = self.gate.filter_for_llm(doc_ids, batch_logits)
        triggered_set = set(triggered)

        results: dict[str, list[MatchResult]] = {}
        for doc_id, text, logits in zip(doc_ids, texts, batch_logits):
            if doc_id not in triggered_set:
                continue
            phrases = self.generator.generate(text, doc_id)
            matches = [
                self.matcher.match(phrase, self.label_inventory)
                for phrase in phrases
            ]
            if matches:
                results[doc_id] = matches
        return results

    def update_label_inventory(self, label_inventory: Mapping[str, str]) -> None:
        """Hot-update the label inventory after Writer promotes new labels."""
        self.label_inventory = dict(label_inventory)
