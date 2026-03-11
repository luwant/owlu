"""Lightweight semantic matching and preliminary decision logic."""

from __future__ import annotations

import math
import re
from typing import Callable, Mapping, Sequence

from ..common.types import CandidatePhrase, MatchResult, OWLUConfig


class SemanticMatcher:
    """Semantic matcher for preliminary decisions.

    Supports two operating modes:
        - **Dense** (preferred): uses a ``dense_encoder`` that maps text → ``list[float]``
          (e.g. ``SentenceTransformerEncoder.as_dense_encoder()``).
        - **BOW** (fallback): lightweight bag-of-words cosine — zero external deps.

    When *dense_encoder* is supplied it takes priority; the BOW path is only used
    if no dense encoder is provided.
    """

    def __init__(
        self,
        config: OWLUConfig,
        encoder: Callable[[str], dict[str, float]] | None = None,
        dense_encoder: Callable[[str], list[float]] | None = None,
        stopwords: set[str] | None = None,
    ):
        self.config = config
        self._dense_encoder = dense_encoder
        self._bow_encoder_fn = encoder or self._bow_encoder
        self.stopwords = stopwords or {
            "a", "an", "the", "and", "or", "to", "of", "for", "in", "on", "with",
        }
        # Label embedding cache (cleared on inventory update)
        self._label_cache: dict[str, list[float]] | None = None

    def _naive_lemmatize(self, token: str) -> str:
        if len(token) > 4 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 3 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def normalize(self, phrase: str) -> str:
        text = (phrase or "").lower().strip()
        tokens = re.findall(r"[a-z0-9]+", text)
        normalized_tokens: list[str] = []
        for tok in tokens:
            if tok in self.stopwords:
                continue
            lemma = self._naive_lemmatize(tok)
            if lemma and lemma not in self.stopwords:
                normalized_tokens.append(lemma)
        return " ".join(normalized_tokens[:8])

    def _bow_encoder(self, text: str) -> dict[str, float]:
        tokens = self.normalize(text).split()
        vec: dict[str, float] = {}
        for token in tokens:
            vec[token] = vec.get(token, 0.0) + 1.0
        return vec

    def _cosine_similarity(
        self, left: dict[str, float], right: dict[str, float]
    ) -> float:
        if not left or not right:
            return 0.0
        dot = 0.0
        for key, lv in left.items():
            dot += lv * right.get(key, 0.0)
        left_norm = math.sqrt(sum(v * v for v in left.values()))
        right_norm = math.sqrt(sum(v * v for v in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _dense_cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        ln = math.sqrt(sum(v * v for v in left))
        rn = math.sqrt(sum(v * v for v in right))
        if ln == 0.0 or rn == 0.0:
            return 0.0
        return dot / (ln * rn)

    def invalidate_label_cache(self) -> None:
        """Clear label embedding cache (call after updating label inventory)."""
        self._label_cache = None

    def preliminary_decide(self, s_max: float, agreement: float) -> str:
        if (
            s_max >= self.config.merge_threshold
            and agreement >= self.config.agreement_threshold
        ):
            return "merge_pre"
        if s_max < self.config.novel_threshold:
            return "novel_pre"
        return "hold_pre"

    def match(
        self, phrase: CandidatePhrase, labels: Mapping[str, str]
    ) -> MatchResult:
        normalized = self.normalize(phrase.text)

        if self._dense_encoder is not None:
            return self._match_dense(phrase, labels, normalized)
        return self._match_bow(phrase, labels, normalized)

    # ------------------------------------------------------------------
    # Dense path (sentence-transformer)
    # ------------------------------------------------------------------

    def _ensure_label_cache(self, labels: Mapping[str, str]) -> dict[str, list[float]]:
        if self._label_cache is not None:
            return self._label_cache
        assert self._dense_encoder is not None
        self._label_cache = {
            label_id: self._dense_encoder(label_text)
            for label_id, label_text in labels.items()
        }
        return self._label_cache

    def _match_dense(
        self,
        phrase: CandidatePhrase,
        labels: Mapping[str, str],
        normalized: str,
    ) -> MatchResult:
        assert self._dense_encoder is not None
        phrase_vec = self._dense_encoder(normalized)
        label_vecs = self._ensure_label_cache(labels)

        best_label: str | None = None
        best_similarity = -1.0
        for label_id, label_vec in label_vecs.items():
            sim = self._dense_cosine(phrase_vec, label_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_label = label_id

        similarity = max(best_similarity, 0.0)
        action = self.preliminary_decide(similarity, phrase.agreement)
        reason = (
            f"[dense] s_max={similarity:.4f}, agreement={phrase.agreement:.4f}, "
            f"merge_th={self.config.merge_threshold:.2f}, "
            f"novel_th={self.config.novel_threshold:.2f}"
        )
        return MatchResult(
            phrase=phrase,
            action=action,
            target_label=best_label if action == "merge_pre" else None,
            similarity=similarity,
            decision_reason=reason,
            normalized_phrase=normalized,
        )

    # ------------------------------------------------------------------
    # BOW path (fallback)
    # ------------------------------------------------------------------

    def _match_bow(
        self,
        phrase: CandidatePhrase,
        labels: Mapping[str, str],
        normalized: str,
    ) -> MatchResult:
        phrase_vec = self._bow_encoder_fn(normalized)

        best_label: str | None = None
        best_similarity = -1.0
        for label_id, label_text in labels.items():
            label_vec = self._bow_encoder_fn(label_text)
            sim = self._cosine_similarity(phrase_vec, label_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_label = label_id

        similarity = max(best_similarity, 0.0)
        action = self.preliminary_decide(similarity, phrase.agreement)
        reason = (
            f"[bow] s_max={similarity:.4f}, agreement={phrase.agreement:.4f}, "
            f"merge_th={self.config.merge_threshold:.2f}, "
            f"novel_th={self.config.novel_threshold:.2f}"
        )
        return MatchResult(
            phrase=phrase,
            action=action,
            target_label=best_label if action == "merge_pre" else None,
            similarity=similarity,
            decision_reason=reason,
            normalized_phrase=normalized,
        )
