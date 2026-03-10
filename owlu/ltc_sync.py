"""OWLU sync module.

Task 2:
- fast_sync (semantic-only refresh, fixed label dimensionality)

Task 3:
- slow_sync (label expansion, replay + incremental tuning, constrained threshold search)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import random
from typing import Any, Callable, Mapping, Sequence
import uuid

from .label_bank import LabelBank


Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class ValidationSample:
    """Validation sample used for threshold recalibration."""

    embedding: Vector
    true_labels: set[str]


@dataclass(frozen=True)
class TrainingSample:
    """Training sample used by slow_sync replay + incremental update."""

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


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom <= 0:
        return 0.0
    return (2.0 * tp) / float(denom)


def _micro_f1(samples: Sequence[ValidationSample], label_ids: Sequence[str], prototypes: Matrix, threshold: float) -> float:
    if not samples:
        return 0.0

    tp_total = fp_total = fn_total = 0
    for sample in samples:
        scores = score_document(sample.embedding, prototypes)
        predicted = {label_ids[idx] for idx, score in enumerate(scores) if score >= threshold}
        true_set = set(sample.true_labels)

        tp_total += len(predicted & true_set)
        fp_total += len(predicted - true_set)
        fn_total += len(true_set - predicted)
    return _safe_f1(tp_total, fp_total, fn_total)


def _precision_at_3(
    samples: Sequence[ValidationSample],
    label_ids: Sequence[str],
    prototypes: Matrix,
    threshold: float,
) -> float:
    """P@3 in percentage points [0, 100]."""
    if not samples:
        return 0.0

    total = 0.0
    for sample in samples:
        scores = score_document(sample.embedding, prototypes)
        ranked = sorted(zip(label_ids, scores), key=lambda item: item[1], reverse=True)[:3]
        kept = [label for label, score in ranked if score >= threshold]
        hit = sum(1 for label in kept if label in sample.true_labels)
        total += hit / 3.0
    return (total / float(len(samples))) * 100.0


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


def recalibrate_threshold_with_constraints(
    prototypes: Matrix,
    label_ids: Sequence[str],
    validation_set: Sequence[ValidationSample],
    current_threshold: float,
    *,
    baseline_p_at_3: float | None = None,
    max_p_at_3_drop: float = 1.0,
) -> float:
    """Grid-search threshold with Task 3 constraints.

    - Grid: [0.10, 0.90], step 0.02
    - Objective: maximize Macro-F1
    - Constraint: P@3 drop vs baseline <= `max_p_at_3_drop` (percentage points)
    - Tie-break: closer to current threshold
    """
    if not validation_set:
        return float(current_threshold)

    if max_p_at_3_drop < 0.0:
        raise ValueError("max_p_at_3_drop must be >= 0")

    baseline = baseline_p_at_3
    if baseline is None:
        baseline = _precision_at_3(validation_set, label_ids, prototypes, current_threshold)

    best_threshold = float(current_threshold)
    best_score = -1.0
    found_valid = False

    # Fallback if no threshold satisfies constraints.
    fallback_threshold = float(current_threshold)
    fallback_score = -1.0

    grid = [round(0.10 + (i * 0.02), 2) for i in range(41)]
    for threshold in grid:
        macro = _macro_f1(validation_set, label_ids, prototypes, threshold)
        p_at_3 = _precision_at_3(validation_set, label_ids, prototypes, threshold)
        drop = float(baseline) - p_at_3

        if macro > fallback_score + 1e-12:
            fallback_score = macro
            fallback_threshold = threshold
        elif abs(macro - fallback_score) <= 1e-12:
            if abs(threshold - current_threshold) < abs(fallback_threshold - current_threshold):
                fallback_threshold = threshold

        if drop > max_p_at_3_drop + 1e-12:
            continue

        if macro > best_score + 1e-12:
            best_score = macro
            best_threshold = threshold
            found_valid = True
        elif abs(macro - best_score) <= 1e-12:
            if abs(threshold - current_threshold) < abs(best_threshold - current_threshold):
                best_threshold = threshold
                found_valid = True

    if found_valid:
        return float(best_threshold)
    return float(fallback_threshold)


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


def _coerce_training_samples(
    samples: Mapping[str, Sequence[TrainingSample | ValidationSample | Sequence[float]]] | None,
    *,
    only_labels: set[str] | None = None,
) -> list[TrainingSample]:
    if not samples:
        return []

    converted: list[TrainingSample] = []
    for label_id, rows in samples.items():
        if only_labels is not None and label_id not in only_labels:
            continue
        for row in rows:
            if isinstance(row, TrainingSample):
                if not row.true_labels:
                    converted.append(TrainingSample(embedding=list(row.embedding), true_labels={label_id}))
                else:
                    converted.append(TrainingSample(embedding=list(row.embedding), true_labels=set(row.true_labels)))
                continue
            if isinstance(row, ValidationSample):
                if not row.true_labels:
                    converted.append(TrainingSample(embedding=list(row.embedding), true_labels={label_id}))
                else:
                    converted.append(TrainingSample(embedding=list(row.embedding), true_labels=set(row.true_labels)))
                continue
            converted.append(TrainingSample(embedding=[float(v) for v in row], true_labels={label_id}))
    return converted


def _shape_for_classifier_row(vec: Sequence[float], dim: int) -> Vector:
    if dim <= 0:
        return []
    if len(vec) == dim:
        return [float(v) for v in vec]
    if len(vec) > dim:
        return [float(v) for v in vec[:dim]]
    out = [float(v) for v in vec]
    out.extend(0.0 for _ in range(dim - len(out)))
    return out


def _collect_label_metadata(label_ids: Sequence[str], label_bank: LabelBank) -> tuple[dict[str, list[str]], dict[str, str]]:
    aliases_map: dict[str, list[str]] = {}
    descriptions_map: dict[str, str] = {}
    for label_id in label_ids:
        aliases_map[label_id] = _resolve_label_semantic_text(label_id, label_bank)
        descriptions_map[label_id] = label_bank.get_label_description(label_id)
    return aliases_map, descriptions_map


def _initialize_vector_from_label_texts(
    *,
    label_id: str,
    label_bank: LabelBank,
    dim: int,
    text_encoder: Callable[[str, int], Vector],
) -> Vector:
    semantic_texts = label_bank.get_label_aliases(label_id)
    if not semantic_texts:
        promoted = label_bank.promoted_labels.get(label_id)
        if promoted and promoted.representative_phrase:
            semantic_texts = [promoted.representative_phrase]
    if not semantic_texts:
        semantic_texts = _resolve_label_semantic_text(label_id, label_bank)

    embeddings = [text_encoder(text, dim) for text in semantic_texts]
    return normalize(_mean_vector(embeddings))


def expand_model_state(
    model_state: Mapping[str, object],
    label_bank: LabelBank,
    *,
    new_label_ids: Sequence[str] | None = None,
    text_encoder: Callable[[str, int], Vector] | None = None,
) -> tuple[dict[str, object], dict[str, Vector]]:
    """Expand label dimension (L -> L') and rebuild optional classifier state."""
    label_ids_old = [str(v) for v in list(model_state["label_ids"])]
    e_old = [[float(x) for x in row] for row in list(model_state["E"])]
    p_old = [[float(x) for x in row] for row in list(model_state["P"])]

    if len(label_ids_old) != len(e_old) or len(label_ids_old) != len(p_old):
        raise ValueError("label_ids, E, and P row counts must match")
    if not e_old:
        raise ValueError("E cannot be empty")

    dim = len(e_old[0])
    for row in e_old + p_old:
        if len(row) != dim:
            raise ValueError("All embedding/prototype rows must share the same dim")

    encode = text_encoder or default_text_encoder
    requested_new = [str(v) for v in (new_label_ids or label_bank.promoted_labels.keys())]
    appended = [label_id for label_id in requested_new if label_id not in label_ids_old]

    expanded = dict(model_state)
    expanded_ids = list(label_ids_old)
    expanded_e = [list(row) for row in e_old]
    expanded_p = [list(row) for row in p_old]
    init_vectors: dict[str, Vector] = {}

    for label_id in appended:
        init_vec = _initialize_vector_from_label_texts(
            label_id=label_id,
            label_bank=label_bank,
            dim=dim,
            text_encoder=encode,
        )
        expanded_ids.append(label_id)
        expanded_e.append(list(init_vec))
        expanded_p.append(list(init_vec))
        init_vectors[label_id] = list(init_vec)

    expanded["label_ids"] = expanded_ids
    expanded["E"] = expanded_e
    expanded["P"] = expanded_p
    expanded["num_labels"] = len(expanded_ids)

    if "classifier_weight" in model_state:
        weights = [[float(v) for v in row] for row in list(model_state["classifier_weight"])]
        if len(weights) != len(label_ids_old):
            raise ValueError("classifier_weight row count must match old labels")
        weight_dim = len(weights[0]) if weights else 0
        for label_id in appended:
            init_vec = init_vectors[label_id]
            weights.append(_shape_for_classifier_row(init_vec, weight_dim))
        expanded["classifier_weight"] = weights

    if "classifier_bias" in model_state:
        bias = [float(v) for v in list(model_state["classifier_bias"])]
        if len(bias) != len(label_ids_old):
            raise ValueError("classifier_bias row count must match old labels")
        bias.extend(0.0 for _ in appended)
        expanded["classifier_bias"] = bias

    aliases, descriptions = _collect_label_metadata(expanded_ids, label_bank)
    expanded["label_aliases"] = aliases
    expanded["label_descriptions"] = descriptions

    return expanded, init_vectors


def build_replay_samples(
    old_samples: Sequence[TrainingSample],
    *,
    label_ids: Sequence[str],
    max_pos_per_label: int = 30,
    max_neg_per_label: int = 30,
    seed: int = 42,
) -> tuple[list[TrainingSample], dict[str, dict[str, int]]]:
    """Build stratified replay set with per-label positive/negative caps."""
    if max_pos_per_label < 0 or max_neg_per_label < 0:
        raise ValueError("max_pos_per_label and max_neg_per_label must be >= 0")
    if not old_samples:
        return [], {
            label_id: {"pos_total": 0, "neg_total": 0, "pos_selected": 0, "neg_selected": 0}
            for label_id in label_ids
        }

    rng = random.Random(seed)
    selected: set[int] = set()
    stats: dict[str, dict[str, int]] = {}

    for label_id in label_ids:
        pos_indices = [idx for idx, sample in enumerate(old_samples) if label_id in sample.true_labels]
        neg_indices = [idx for idx, sample in enumerate(old_samples) if label_id not in sample.true_labels]

        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        chosen_pos = pos_indices[:max_pos_per_label]
        chosen_neg = neg_indices[:max_neg_per_label]
        selected.update(chosen_pos)
        selected.update(chosen_neg)

        stats[label_id] = {
            "pos_total": len(pos_indices),
            "neg_total": len(neg_indices),
            "pos_selected": len(chosen_pos),
            "neg_selected": len(chosen_neg),
        }

    ordered = sorted(selected)
    return [old_samples[idx] for idx in ordered], stats


def compose_incremental_training_set(
    replay_samples: Sequence[TrainingSample],
    new_samples: Sequence[TrainingSample],
    *,
    old_new_ratio: tuple[int, int] = (2, 1),
    seed: int = 42,
) -> tuple[list[TrainingSample], dict[str, int]]:
    """Compose one-pass incremental set with fixed old:new ratio."""
    old_ratio, new_ratio = old_new_ratio
    if old_ratio < 0 or new_ratio <= 0:
        raise ValueError("old_new_ratio must be (>=0, >0)")

    rng = random.Random(seed)
    selected_old = list(replay_samples)
    selected_new = list(new_samples)

    if selected_new and selected_old:
        cap_old = int(math.ceil((len(selected_new) * float(old_ratio)) / float(new_ratio)))
        if len(selected_old) > cap_old:
            indices = list(range(len(selected_old)))
            rng.shuffle(indices)
            keep = sorted(indices[:cap_old])
            selected_old = [selected_old[idx] for idx in keep]

    combined = selected_old + selected_new
    rng.shuffle(combined)
    stats = {
        "old_selected": len(selected_old),
        "new_selected": len(selected_new),
        "total_selected": len(combined),
    }
    return combined, stats


def incremental_finetune(
    *,
    label_ids: Sequence[str],
    E: Matrix,
    P: Matrix,
    train_set: Sequence[TrainingSample],
    new_label_ids: set[str],
    old_lr: float = 0.05,
    new_lr: float = 0.20,
) -> tuple[Matrix, Matrix]:
    """Lightweight one-pass incremental tuning over E/P (centroid style)."""
    if not train_set:
        return [list(row) for row in E], [list(row) for row in P]

    if len(E) != len(label_ids) or len(P) != len(label_ids):
        raise ValueError("E/P row counts must match label_ids")
    dim = len(E[0]) if E else 0
    if dim <= 0:
        raise ValueError("Embedding dim must be > 0")
    for row in E + P:
        if len(row) != dim:
            raise ValueError("All E/P rows must share same dim")

    e_new = [list(row) for row in E]
    p_new = [list(row) for row in P]

    for sample in train_set:
        if len(sample.embedding) != dim:
            raise ValueError("Training sample embedding dim mismatch")

    for idx, label_id in enumerate(label_ids):
        positives = [sample.embedding for sample in train_set if label_id in sample.true_labels]
        negatives = [sample.embedding for sample in train_set if label_id not in sample.true_labels]
        if not positives:
            continue

        pos_mean = normalize(_mean_vector(positives))
        target = pos_mean
        if negatives:
            neg_mean = normalize(_mean_vector(negatives))
            contrast = [float(pv) - float(nv) for pv, nv in zip(pos_mean, neg_mean)]
            if _l2_norm(contrast) > 0.0:
                target = normalize(contrast)

        lr = float(new_lr if label_id in new_label_ids else old_lr)
        lr = max(0.0, min(1.0, lr))
        e_new[idx] = _blend_and_normalize(e_new[idx], target, eta=lr)
        p_eta = min(1.0, max(0.0, (lr * 0.5) + 0.05))
        p_new[idx] = _blend_and_normalize(p_new[idx], e_new[idx], eta=p_eta)

    return e_new, p_new


def _subset(
    label_ids: Sequence[str],
    prototypes: Matrix,
    subset_label_ids: Sequence[str],
) -> tuple[list[str], Matrix]:
    idx = {label_id: i for i, label_id in enumerate(label_ids)}
    selected_ids = [label_id for label_id in subset_label_ids if label_id in idx]
    selected_proto = [list(prototypes[idx[label_id]]) for label_id in selected_ids]
    return selected_ids, selected_proto


def _state_metrics(
    *,
    label_ids: Sequence[str],
    prototypes: Matrix,
    threshold: float,
    validation_set: Sequence[ValidationSample],
    subset_label_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    target_ids, target_prototypes = (
        _subset(label_ids, prototypes, subset_label_ids)
        if subset_label_ids is not None
        else ([str(v) for v in label_ids], [list(row) for row in prototypes])
    )
    if not target_ids or not validation_set:
        return {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "long_tail_macro_f1": 0.0,
            "coverage_rate": 0.0,
            "p_at_3": 0.0,
        }

    macro = _macro_f1(validation_set, target_ids, target_prototypes, threshold)
    micro = _micro_f1(validation_set, target_ids, target_prototypes, threshold)
    p3 = _precision_at_3(validation_set, target_ids, target_prototypes, threshold)

    predicted_any: set[str] = set()
    for sample in validation_set:
        scores = score_document(sample.embedding, target_prototypes)
        for idx, score in enumerate(scores):
            if score >= threshold:
                predicted_any.add(target_ids[idx])
    coverage = len(predicted_any) / float(len(target_ids)) if target_ids else 0.0

    freq = {label_id: 0 for label_id in target_ids}
    for sample in validation_set:
        for label_id in target_ids:
            if label_id in sample.true_labels:
                freq[label_id] += 1
    ordered = sorted(target_ids, key=lambda label_id: (freq[label_id], label_id))
    tail_size = max(1, len(ordered) // 3)
    long_tail_ids = ordered[:tail_size]
    long_tail_labels, long_tail_prototypes = _subset(target_ids, target_prototypes, long_tail_ids)
    long_tail_macro = _macro_f1(validation_set, long_tail_labels, long_tail_prototypes, threshold)

    return {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "long_tail_macro_f1": float(long_tail_macro),
        "coverage_rate": float(coverage),
        "p_at_3": float(p3),
    }


def slow_sync(
    model_state: Mapping[str, object],
    label_bank: LabelBank,
    *,
    old_training_samples: Sequence[TrainingSample] | None = None,
    new_label_samples: Mapping[str, Sequence[TrainingSample | ValidationSample | Sequence[float]]] | None = None,
    validation_set: Sequence[ValidationSample] | None = None,
    text_encoder: Callable[[str, int], Vector] | None = None,
    max_pos_per_label: int = 30,
    max_neg_per_label: int = 30,
    old_new_ratio: tuple[int, int] = (2, 1),
    seed: int = 42,
    old_lr: float = 0.05,
    new_lr: float = 0.20,
    max_p_at_3_drop: float = 1.0,
    run_id: str | None = None,
    sqlite_store: Any | None = None,
) -> dict[str, object]:
    """Task 3 slow sync.

    1) Read promoted labels from LabelBank.promoted_labels.
    2) Expand label space L -> L', initialize new E/P.
    3) Replay old samples + new label samples for one-pass incremental update.
    4) Recalibrate threshold on validation set under P@3 drop constraint.
    5) Emit traceable sync_report and optional SQLite persistence.
    """
    validation = list(validation_set or [])
    old_label_ids = [str(v) for v in list(model_state["label_ids"])]
    old_threshold = float(model_state.get("threshold", 0.5))
    old_prototypes = [[float(x) for x in row] for row in list(model_state["P"])]

    promoted_ids = [label_id for label_id in label_bank.promoted_labels.keys() if label_id not in old_label_ids]
    expanded_state, init_vectors = expand_model_state(
        model_state=model_state,
        label_bank=label_bank,
        new_label_ids=promoted_ids,
        text_encoder=text_encoder,
    )

    all_label_ids = [str(v) for v in list(expanded_state["label_ids"])]
    replay_samples, replay_stats = build_replay_samples(
        list(old_training_samples or []),
        label_ids=old_label_ids,
        max_pos_per_label=max_pos_per_label,
        max_neg_per_label=max_neg_per_label,
        seed=seed,
    )

    promoted_set = set(promoted_ids)
    new_samples = _coerce_training_samples(new_label_samples, only_labels=promoted_set)
    train_set, mix_stats = compose_incremental_training_set(
        replay_samples=replay_samples,
        new_samples=new_samples,
        old_new_ratio=old_new_ratio,
        seed=seed,
    )

    e_tuned, p_tuned = incremental_finetune(
        label_ids=all_label_ids,
        E=[[float(v) for v in row] for row in list(expanded_state["E"])],
        P=[[float(v) for v in row] for row in list(expanded_state["P"])],
        train_set=train_set,
        new_label_ids=promoted_set,
        old_lr=old_lr,
        new_lr=new_lr,
    )

    baseline_metrics = _state_metrics(
        label_ids=old_label_ids,
        prototypes=old_prototypes,
        threshold=old_threshold,
        validation_set=validation,
    )
    baseline_p3 = baseline_metrics["p_at_3"] if validation else None
    new_threshold = recalibrate_threshold_with_constraints(
        p_tuned,
        all_label_ids,
        validation,
        old_threshold,
        baseline_p_at_3=baseline_p3,
        max_p_at_3_drop=max_p_at_3_drop,
    )

    expanded_state["E"] = e_tuned
    expanded_state["P"] = p_tuned
    expanded_state["threshold"] = float(new_threshold)
    expanded_state["rollback_snapshot"] = {
        "label_ids": list(old_label_ids),
        "E": [[float(x) for x in row] for row in list(model_state["E"])],
        "P": [[float(x) for x in row] for row in list(model_state["P"])],
        "threshold": old_threshold,
    }

    after_metrics = _state_metrics(
        label_ids=all_label_ids,
        prototypes=p_tuned,
        threshold=new_threshold,
        validation_set=validation,
    )
    old_after_metrics = _state_metrics(
        label_ids=all_label_ids,
        prototypes=p_tuned,
        threshold=new_threshold,
        validation_set=validation,
        subset_label_ids=old_label_ids,
    )

    now = datetime.now(timezone.utc).isoformat()
    sync_run_id = run_id or f"slow_sync_{uuid.uuid4().hex[:10]}"
    sync_report = {
        "run_id": sync_run_id,
        "timestamp": now,
        "num_old_labels": len(old_label_ids),
        "num_new_labels": len(promoted_ids),
        "num_total_labels": len(all_label_ids),
        "new_label_ids": list(promoted_ids),
        "threshold_before": old_threshold,
        "threshold_after": float(new_threshold),
        "replay_stats": replay_stats,
        "train_mix_stats": mix_stats,
        "training_config": {
            "max_pos_per_label": int(max_pos_per_label),
            "max_neg_per_label": int(max_neg_per_label),
            "old_new_ratio": [int(old_new_ratio[0]), int(old_new_ratio[1])],
            "seed": int(seed),
            "old_lr": float(old_lr),
            "new_lr": float(new_lr),
            "max_p_at_3_drop": float(max_p_at_3_drop),
        },
        "init_vectors": {label_id: list(vec) for label_id, vec in init_vectors.items()},
        "metrics_before": baseline_metrics,
        "metrics_after": after_metrics,
        "old_label_metrics_after": old_after_metrics,
        "metrics_delta": {
            "micro_f1": float(after_metrics["micro_f1"] - baseline_metrics["micro_f1"]),
            "macro_f1": float(after_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
            "long_tail_macro_f1": float(after_metrics["long_tail_macro_f1"] - baseline_metrics["long_tail_macro_f1"]),
            "coverage_rate": float(after_metrics["coverage_rate"] - baseline_metrics["coverage_rate"]),
            "p_at_3": float(after_metrics["p_at_3"] - baseline_metrics["p_at_3"]),
        },
    }
    expanded_state["sync_report"] = sync_report

    if sqlite_store is not None:
        if hasattr(sqlite_store, "record_label_snapshot"):
            sqlite_store.record_label_snapshot(sync_run_id, "after", label_bank)
        if hasattr(sqlite_store, "record_promoted_labels"):
            sqlite_store.record_promoted_labels(sync_run_id, label_bank)
        if hasattr(sqlite_store, "record_slow_sync_run"):
            sqlite_store.record_slow_sync_run(sync_run_id, sync_report)

    return expanded_state
