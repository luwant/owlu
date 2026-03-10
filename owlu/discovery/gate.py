"""LTCE confidence gate — decides whether to invoke LLM for a given document.

When the LTCE model is confident about its predictions (high raw_max_prob),
calling the LLM is unnecessary.  When the model outputs low confidence
(potential OOD input), the gate triggers LLM phrase extraction.

Design rationale
----------------
Multi-label sigmoid classifiers output near-zero for *all* labels on truly
OOD inputs.  Other metrics (entropy, cosine, num_above_threshold) go the
wrong direction in this regime.  Therefore ``raw_max_prob`` — the maximum
sigmoid activation at T=1.0 before any temperature scaling — is the most
reliable recognition signal.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..common.types import GateDecision


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class LtceGate:
    """Confidence-based gate using LTCE model output to control LLM invocation.

    Parameters
    ----------
    recognition_floor : float
        Minimum allowed recognition threshold (default 0.10).
    adaptive_percentile : float
        Fraction of calibration scores to use as the adaptive baseline
        (default 0.05 = 5th percentile).
    fixed_threshold : float | None
        If provided, skip adaptive calibration and use this value directly.
    """

    def __init__(
        self,
        *,
        recognition_floor: float = 0.10,
        adaptive_percentile: float = 0.05,
        fixed_threshold: float | None = None,
    ):
        if not (0.0 <= recognition_floor <= 1.0):
            raise ValueError("recognition_floor must be in [0, 1]")
        if not (0.0 < adaptive_percentile < 1.0):
            raise ValueError("adaptive_percentile must be in (0, 1)")
        if fixed_threshold is not None and not (0.0 <= fixed_threshold <= 1.0):
            raise ValueError("fixed_threshold must be in [0, 1]")

        self.recognition_floor = recognition_floor
        self.adaptive_percentile = adaptive_percentile
        self._threshold: float | None = fixed_threshold

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError(
                "Gate has not been calibrated yet.  "
                "Call calibrate() with validation logits or set fixed_threshold."
            )
        return self._threshold

    # ------------------------------------------------------------------
    # Core gate logic
    # ------------------------------------------------------------------

    @staticmethod
    def raw_max_prob(logits: Sequence[float]) -> float:
        """Compute max sigmoid probability from raw logits (T=1.0)."""
        if not logits:
            return 0.0
        return max(_sigmoid(float(v)) for v in logits)

    def evaluate(self, logits: Sequence[float]) -> GateDecision:
        """Decide whether to invoke LLM for a single document."""
        rmp = self.raw_max_prob(logits)
        th = self.threshold
        invoke = rmp < th
        reason = (
            f"raw_max_prob={rmp:.4f} {'<' if invoke else '>='} "
            f"recognition_threshold={th:.4f}"
        )
        return GateDecision(
            should_invoke_llm=invoke,
            raw_max_prob=rmp,
            recognition_threshold=th,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Adaptive calibration
    # ------------------------------------------------------------------

    def calibrate(self, validation_logits: Sequence[Sequence[float]]) -> float:
        """Calibrate recognition threshold from in-distribution validation data."""
        if not validation_logits:
            self._threshold = self.recognition_floor
            return self._threshold

        scores = sorted(self.raw_max_prob(logits) for logits in validation_logits)
        idx = max(0, min(int(len(scores) * self.adaptive_percentile), len(scores) - 1))
        adaptive_value = scores[idx]
        self._threshold = max(adaptive_value, self.recognition_floor)
        return self._threshold

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def batch_evaluate(
        self, batch_logits: Sequence[Sequence[float]]
    ) -> list[GateDecision]:
        return [self.evaluate(logits) for logits in batch_logits]

    def filter_for_llm(
        self, doc_ids: Sequence[str], batch_logits: Sequence[Sequence[float]]
    ) -> list[str]:
        if len(doc_ids) != len(batch_logits):
            raise ValueError("doc_ids and batch_logits must have the same length")
        decisions = self.batch_evaluate(batch_logits)
        return [
            doc_id
            for doc_id, decision in zip(doc_ids, decisions)
            if decision.should_invoke_llm
        ]
