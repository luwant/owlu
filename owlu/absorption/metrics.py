"""Shared scoring, inference, and threshold calibration utilities.

These pure functions are used by both fast_sync and slow_sync.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

from ..common.types import Matrix, ValidationSample, Vector


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------

def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def normalize(vec: Sequence[float]) -> Vector:
    nrm = _l2_norm(vec)
    if nrm == 0.0:
        return [0.0 for _ in vec]
    return [float(v) / nrm for v in vec]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions must match")
    ln = _l2_norm(left)
    rn = _l2_norm(right)
    if ln == 0.0 or rn == 0.0:
        return 0.0
    dot = sum(float(lv) * float(rv) for lv, rv in zip(left, right))
    return dot / (ln * rn)


def blend_and_normalize(
    base: Sequence[float], update: Sequence[float], eta: float
) -> Vector:
    if len(base) != len(update):
        raise ValueError("Vector dimensions must match")
    merged = [
        ((1.0 - eta) * float(b)) + (eta * float(u))
        for b, u in zip(base, update)
    ]
    return normalize(merged)


def mean_vector(vectors: Sequence[Sequence[float]]) -> Vector:
    if not vectors:
        raise ValueError("Cannot compute mean vector of empty input")
    dim = len(vectors[0])
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("All vectors must have the same dimension")
    acc = [0.0] * dim
    for vec in vectors:
        for idx, value in enumerate(vec):
            acc[idx] += float(value)
    return [value / float(len(vectors)) for value in acc]


# ---------------------------------------------------------------------------
# Text encoder fallback
# ---------------------------------------------------------------------------

def default_text_encoder(text: str, dim: int) -> Vector:
    """Deterministic lightweight encoder for tests and fallback runtime."""
    if dim <= 0:
        raise ValueError("dim must be > 0")
    tokens = [tok for tok in " ".join((text or "").lower().split()).split(" ") if tok]
    if not tokens:
        return [0.0] * dim

    vec = [0.0] * dim
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[bucket] += sign
    return normalize(vec)


# ---------------------------------------------------------------------------
# Scoring & inference
# ---------------------------------------------------------------------------

def score_document(embedding: Sequence[float], prototypes: Matrix) -> Vector:
    return [cosine_similarity(embedding, proto) for proto in prototypes]


def infer_topk(
    embedding: Sequence[float],
    model_state: Mapping[str, object],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    label_ids = list(model_state["label_ids"])
    prototypes = list(model_state["P"])
    scores = score_document(embedding, prototypes)
    pairs = list(zip(label_ids, scores))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[: max(1, int(top_k))]


def infer_above_threshold(
    embedding: Sequence[float],
    model_state: Mapping[str, object],
) -> list[str]:
    threshold = float(model_state.get("threshold", 0.5))
    scored = infer_topk(
        embedding, model_state, top_k=len(list(model_state["label_ids"]))
    )
    return [label_id for label_id, score in scored if score >= threshold]


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def _macro_f1(
    samples: Sequence[ValidationSample],
    label_ids: Sequence[str],
    prototypes: Matrix,
    threshold: float,
) -> float:
    if not samples:
        return 0.0

    label_f1: list[float] = []
    for label_idx, label_id in enumerate(label_ids):
        tp = fp = fn = 0
        for sample in samples:
            scores = score_document(sample.embedding, prototypes)
            pred_positive = scores[label_idx] >= threshold
            true_positive = label_id in sample.true_labels
            if pred_positive and true_positive:
                tp += 1
            elif pred_positive and not true_positive:
                fp += 1
            elif (not pred_positive) and true_positive:
                fn += 1

        denom = (2 * tp) + fp + fn
        label_f1.append((2 * tp) / float(denom) if denom > 0 else 0.0)

    return sum(label_f1) / float(len(label_f1)) if label_f1 else 0.0


def recalibrate_threshold(
    prototypes: Matrix,
    label_ids: Sequence[str],
    validation_set: Sequence[ValidationSample],
    current_threshold: float,
) -> float:
    if not validation_set:
        return float(current_threshold)

    best_threshold = float(current_threshold)
    best_score = -1.0

    grid = [round(0.10 + (i * 0.02), 2) for i in range(41)]
    for threshold in grid:
        score = _macro_f1(validation_set, label_ids, prototypes, threshold)
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = threshold
        elif abs(score - best_score) <= 1e-12:
            if abs(threshold - current_threshold) < abs(
                best_threshold - current_threshold
            ):
                best_threshold = threshold

    return float(best_threshold)
