"""Shared data structures used across all three OWLU modules.

This file centralises every dataclass / type alias so that
discovery, writer and absorption can import from a single location
without circular-dependency issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Sequence

import yaml


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Vector = list[float]
Matrix = list[Vector]
ClusterState = Literal["hold", "candidate", "promoted"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OWLUConfig:
    """Centralised config, loadable from owlu.yaml."""

    # LLM
    llm_base_url: str
    llm_model: str
    llm_api_key: str | None
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: float
    llm_max_phrases: int

    # Matching / decision thresholds
    merge_threshold: float
    novel_threshold: float
    agreement_threshold: float
    uncertain_top1_threshold: float
    uncertain_margin_threshold: float

    # Gate
    recognition_floor: float
    adaptive_percentile: float

    # Sampling
    multi_sample_k: int

    @classmethod
    def from_yaml(cls, path: str) -> "OWLUConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        llm = raw.get("llm", {})
        thresholds = raw.get("thresholds", {})
        gate = raw.get("gate", {})
        sampling = raw.get("sampling", {})
        return cls(
            llm_base_url=str(llm.get("base_url", "https://api.deepseek.com")),
            llm_model=str(llm.get("model", "deepseek-chat")),
            llm_api_key=(
                str(llm.get("api_key")).strip() if llm.get("api_key") else None
            ),
            llm_temperature=float(llm.get("temperature", 0.2)),
            llm_max_tokens=int(llm.get("max_tokens", 512)),
            llm_timeout_seconds=float(llm.get("timeout_seconds", 60.0)),
            llm_max_phrases=int(llm.get("max_phrases", 3)),
            merge_threshold=float(thresholds.get("merge", 0.80)),
            novel_threshold=float(thresholds.get("novel", 0.52)),
            agreement_threshold=float(thresholds.get("agreement", 0.67)),
            uncertain_top1_threshold=float(
                thresholds.get("uncertain_top1", 0.45)
            ),
            uncertain_margin_threshold=float(
                thresholds.get("uncertain_margin", 0.15)
            ),
            recognition_floor=float(gate.get("recognition_floor", 0.10)),
            adaptive_percentile=float(gate.get("adaptive_percentile", 0.05)),
            multi_sample_k=int(sampling.get("multi_sample_k", 3)),
        )


# ---------------------------------------------------------------------------
# Task-1 data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class CandidatePhrase:
    """Single phrase candidate produced by LLM."""

    text: str
    raw_text: str
    source_doc_id: str
    timestamp: datetime
    summary: str | None = None
    evidence: list[str] | None = None
    agreement: float = 1.0
    pass_id: int = 1
    source_count: int = 1
    cluster_id: str | None = None


@dataclass
class MatchResult:
    """Result of matching a CandidatePhrase against the label inventory."""

    phrase: CandidatePhrase
    action: str  # "merge_pre" | "novel_pre" | "hold_pre" | "discard"
    target_label: str | None
    similarity: float
    decision_reason: str
    normalized_phrase: str


# ---------------------------------------------------------------------------
# Task-2 data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class LabelInfo:
    """Mutable label metadata maintained by LabelBank."""

    label_id: str
    aliases: set[str] = field(default_factory=set)
    description: str = ""


@dataclass
class ProtoLabelCluster:
    """Cluster-level aggregation for candidate / hold tracking."""

    cluster_id: str
    representative_phrase: str
    phrases: dict[str, int] = field(default_factory=dict)
    source_docs: set[str] = field(default_factory=set)
    evidence_docs: set[str] = field(default_factory=set)
    freq: int = 0
    agreement_sum: float = 0.0
    agreement_count: int = 0
    nearest_label_id: str | None = None
    nearest_label_distance: float | None = None
    state: ClusterState = "hold"

    @property
    def agreement(self) -> float:
        if self.agreement_count == 0:
            return 0.0
        return self.agreement_sum / float(self.agreement_count)

    @property
    def source_doc_count(self) -> int:
        return len(self.source_docs)


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateDecision:
    """Result of the LTCE confidence gate evaluation."""

    should_invoke_llm: bool
    raw_max_prob: float
    recognition_threshold: float
    reason: str


# ---------------------------------------------------------------------------
# Sync / absorption data objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationSample:
    """Validation sample used for threshold recalibration."""

    embedding: Vector
    true_labels: set[str]
