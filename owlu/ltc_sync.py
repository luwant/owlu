"""Task 2 fast sync: semantic-only refresh without changing label dimensionality."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Mapping, Sequence

from .label_bank import LabelBank


Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class ValidationSample:
    """Validation sample used for threshold recalibration."""

    embedding: Vector
    true_labels: set[str]


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
    dot = 0.0
    for lv, rv in zip(left, right):
        dot += float(lv) * float(rv)
    return dot / (ln * rn)


def _blend_and_normalize(base: Sequence[float], update: Sequence[float], eta: float) -> Vector:
    if len(base) != len(update):
        raise ValueError("Vector dimensions must match")
    merged = [((1.0 - eta) * float(b)) + (eta * float(u)) for b, u in zip(base, update)]
    return normalize(merged)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> Vector:
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
    scored = infer_topk(embedding, model_state, top_k=len(list(model_state["label_ids"])))
    return [label_id for label_id, score in scored if score >= threshold]


def _macro_f1(samples: Sequence[ValidationSample], label_ids: Sequence[str], prototypes: Matrix, threshold: float) -> float:
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
            if abs(threshold - current_threshold) < abs(best_threshold - current_threshold):
                best_threshold = threshold

    return float(best_threshold)


def _resolve_label_semantic_text(label_id: str, label_bank: LabelBank) -> list[str]:
    aliases = label_bank.get_label_aliases(label_id)
    if aliases:
        return aliases

    fallback = " ".join(label_id.split("_"))
    return [fallback] if fallback else [label_id]


def fast_sync(
    model_state: Mapping[str, object],
    label_bank: LabelBank,
    *,
    eta_e: float = 0.2,
    eta_p: float = 0.1,
    validation_set: Sequence[ValidationSample] | None = None,
    text_encoder: Callable[[str, int], Vector] | None = None,
) -> dict[str, object]:
    """Update label embeddings/prototypes with semantic refresh while keeping shape stable."""

    if not (0.0 <= eta_e <= 1.0):
        raise ValueError("eta_e must be in [0, 1]")
    if not (0.0 <= eta_p <= 1.0):
        raise ValueError("eta_p must be in [0, 1]")

    label_ids = [str(v) for v in list(model_state["label_ids"])]
    E_old = [[float(x) for x in row] for row in list(model_state["E"])]
    P_old = [[float(x) for x in row] for row in list(model_state["P"])]

    if len(E_old) != len(label_ids) or len(P_old) != len(label_ids):
        raise ValueError("label_ids, E, and P row counts must match")
    if not E_old:
        raise ValueError("E cannot be empty")

    dim = len(E_old[0])
    if dim == 0:
        raise ValueError("Embedding dim must be > 0")

    for row in E_old + P_old:
        if len(row) != dim:
            raise ValueError("All embedding/prototype rows must share the same dim")

    encode = text_encoder or default_text_encoder

    E_new: Matrix = []
    P_new: Matrix = []

    label_aliases: dict[str, list[str]] = {}
    label_descriptions: dict[str, str] = {}

    for idx, label_id in enumerate(label_ids):
        semantic_texts = _resolve_label_semantic_text(label_id, label_bank)
        alias_embeddings = [encode(text, dim) for text in semantic_texts]
        alias_mean = normalize(_mean_vector(alias_embeddings))

        e_row = _blend_and_normalize(E_old[idx], alias_mean, eta=eta_e)
        p_row = _blend_and_normalize(P_old[idx], e_row, eta=eta_p)

        E_new.append(e_row)
        P_new.append(p_row)

        label_aliases[label_id] = semantic_texts
        label_descriptions[label_id] = label_bank.get_label_description(label_id)

    old_threshold = float(model_state.get("threshold", 0.5))
    new_threshold = recalibrate_threshold(
        P_new,
        label_ids,
        validation_set or [],
        current_threshold=old_threshold,
    )

    alignments = [cosine_similarity(e, p) for e, p in zip(E_new, P_new)]
    avg_alignment = sum(alignments) / float(len(alignments))

    updated = dict(model_state)
    updated["E"] = E_new
    updated["P"] = P_new
    updated["threshold"] = new_threshold
    updated["label_aliases"] = label_aliases
    updated["label_descriptions"] = label_descriptions
    updated["sync_report"] = {
        "eta_e": eta_e,
        "eta_p": eta_p,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "avg_embedding_prototype_alignment": avg_alignment,
        "num_labels": len(label_ids),
        "dim": dim,
    }
    return updated
