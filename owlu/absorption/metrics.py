"""Shared scoring, inference, and threshold calibration utilities.

These pure functions are used by both fast_sync and slow_sync.

The module contains two layers:
    1. Pure-Python (list-based) utilities — for unit tests and lightweight usage.
    2. Torch-accelerated helpers — for operating on real LTCEModel buffers.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Mapping, Sequence

from ..common.types import Matrix, ValidationSample, Vector

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


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


# ---------------------------------------------------------------------------
# Torch-accelerated helpers (for real LTCEModel buffers)
# ---------------------------------------------------------------------------

def blend_and_normalize_torch(
    base: "torch.Tensor", update: "torch.Tensor", eta: float
) -> "torch.Tensor":
    """Blend + L2-normalise using torch ops.  Works on 1-D or 2-D tensors."""
    import torch
    import torch.nn.functional as F

    merged = (1.0 - eta) * base + eta * update
    merged = torch.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(merged, p=2, dim=-1)


def recalibrate_model_threshold(
    model: object,
    dataloader: "DataLoader",
    device: "torch.device",
    current_threshold: float,
) -> float:
    """Grid-search the best Macro-F1 threshold using actual model forward passes.

    Uses sigmoid(logits) as prediction scores — consistent with LTCE training.
    """
    import torch

    model.eval()  # type: ignore[union-attr]

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(  # type: ignore[operator]
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                sentence_map=batch.get("sentence_map"),
                labels=batch["labels"],
            )
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(batch["labels"].cpu())

    if not all_logits:
        return float(current_threshold)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).float()
    probs = torch.sigmoid(logits)

    num_labels = probs.shape[1]
    grid = [round(0.10 + i * 0.02, 2) for i in range(41)]

    best_threshold = float(current_threshold)
    best_score = -1.0

    for th in grid:
        preds = (probs >= th).float()
        per_label_f1: list[float] = []
        for j in range(num_labels):
            tp = (preds[:, j] * labels[:, j]).sum().item()
            fp = (preds[:, j] * (1.0 - labels[:, j])).sum().item()
            fn = ((1.0 - preds[:, j]) * labels[:, j]).sum().item()
            denom = 2.0 * tp + fp + fn
            per_label_f1.append((2.0 * tp / denom) if denom > 0.0 else 0.0)

        macro_f1 = sum(per_label_f1) / float(len(per_label_f1))

        if macro_f1 > best_score + 1e-12:
            best_score = macro_f1
            best_threshold = th
        elif abs(macro_f1 - best_score) <= 1e-12:
            if abs(th - current_threshold) < abs(best_threshold - current_threshold):
                best_threshold = th

    return float(best_threshold)
