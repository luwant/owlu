"""Slow sync — label expansion L → L' with incremental fine-tune.

Expands the label space by adding promoted labels from LabelBank,
initialises their embeddings / prototypes, performs one-pass incremental
fine-tuning, and recalibrates the decision threshold.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from ..common.types import Matrix, ValidationSample, Vector
from ..writer.label_bank import LabelBank
from .metrics import (
    blend_and_normalize,
    cosine_similarity,
    default_text_encoder,
    mean_vector,
    normalize,
    recalibrate_threshold,
    score_document,
)


def _resolve_label_semantic_text(label_id: str, label_bank: LabelBank) -> list[str]:
    aliases = label_bank.get_label_aliases(label_id)
    if aliases:
        return aliases
    fallback = " ".join(label_id.split("_"))
    return [fallback] if fallback else [label_id]


def _incremental_finetune(
    E: Matrix,
    P: Matrix,
    label_ids: list[str],
    promoted_ids: set[str],
    training_samples: Sequence[ValidationSample],
    *,
    new_lr: float,
    old_lr: float,
) -> tuple[Matrix, Matrix]:
    """One-pass contrastive update for embeddings and prototypes.

    For each label, computes mean of positive and negative sample embeddings,
    then blends toward the positive mean (push away from negative mean).
    Promoted labels use new_lr; legacy labels use old_lr.
    """
    if not training_samples:
        return E, P

    num_labels = len(label_ids)
    dim = len(E[0])

    for label_idx, label_id in enumerate(label_ids):
        pos_vecs: list[Vector] = []
        neg_vecs: list[Vector] = []
        for sample in training_samples:
            if label_id in sample.true_labels:
                pos_vecs.append(list(sample.embedding))
            else:
                neg_vecs.append(list(sample.embedding))

        if not pos_vecs:
            continue

        pos_mean = mean_vector(pos_vecs)
        lr = new_lr if label_id in promoted_ids else old_lr

        E[label_idx] = blend_and_normalize(E[label_idx], pos_mean, eta=lr)
        P[label_idx] = blend_and_normalize(P[label_idx], E[label_idx], eta=lr * 0.5)

    return E, P


def slow_sync(
    model_state: Mapping[str, object],
    label_bank: LabelBank,
    *,
    validation_set: Sequence[ValidationSample] | None = None,
    training_samples: Sequence[ValidationSample] | None = None,
    new_lr: float = 0.05,
    old_lr: float = 0.01,
    eta_e: float = 0.2,
    eta_p: float = 0.1,
    text_encoder: Callable[[str, int], Vector] | None = None,
) -> dict[str, object]:
    """Expand label space L → L' and perform incremental fine-tune.

    Steps:
        1. Read promoted labels from LabelBank
        2. Initialise new label embeddings from text encoder
        3. Extend E, P matrices
        4. One-pass incremental fine-tune (contrastive update)
        5. Recalibrate decision threshold with P@3 constraint
    """
    encode = text_encoder or default_text_encoder

    # --- Parse existing model state ---
    label_ids = [str(v) for v in list(model_state["label_ids"])]
    E = [[float(x) for x in row] for row in list(model_state["E"])]
    P = [[float(x) for x in row] for row in list(model_state["P"])]

    if not E:
        raise ValueError("E cannot be empty")
    dim = len(E[0])

    # --- Discover promoted labels not yet in model ---
    promoted = label_bank.promoted_labels
    new_label_ids: list[str] = []
    for lid in promoted:
        if lid not in set(label_ids):
            new_label_ids.append(lid)

    if not new_label_ids:
        # Nothing to expand — fall back to fast-sync-like refresh
        from .fast_sync import fast_sync

        return fast_sync(
            model_state=model_state,
            label_bank=label_bank,
            eta_e=eta_e,
            eta_p=eta_p,
            validation_set=validation_set,
            text_encoder=text_encoder,
        )

    # --- Initialise new label vectors ---
    for lid in new_label_ids:
        texts = _resolve_label_semantic_text(lid, label_bank)
        vecs = [encode(t, dim) for t in texts]
        init_vec = normalize(mean_vector(vecs))
        E.append(list(init_vec))
        P.append(list(init_vec))
        label_ids.append(lid)

    promoted_id_set = set(new_label_ids)

    # --- Incremental fine-tune ---
    if training_samples:
        E, P = _incremental_finetune(
            E, P, label_ids, promoted_id_set,
            training_samples,
            new_lr=new_lr,
            old_lr=old_lr,
        )

    # --- Threshold recalibration ---
    old_threshold = float(model_state.get("threshold", 0.5))
    new_threshold = recalibrate_threshold(
        P, label_ids, validation_set or [], current_threshold=old_threshold,
    )

    # --- Build report ---
    alignments = [cosine_similarity(e, p) for e, p in zip(E, P)]
    avg_alignment = sum(alignments) / float(len(alignments)) if alignments else 0.0

    updated = dict(model_state)
    updated["label_ids"] = label_ids
    updated["E"] = E
    updated["P"] = P
    updated["threshold"] = new_threshold
    updated["sync_report"] = {
        "sync_type": "slow",
        "old_num_labels": len(label_ids) - len(new_label_ids),
        "new_num_labels": len(label_ids),
        "added_labels": new_label_ids,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "avg_embedding_prototype_alignment": avg_alignment,
        "dim": dim,
    }
    return updated
