"""Fast sync — semantic-only refresh of label embeddings / prototypes.

Updates E and P by blending alias text embeddings while keeping
the label count (dimensionality) stable.
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
)


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
    """Update label embeddings / prototypes with semantic refresh (shape-stable).

    For each label:
        1. Encode all aliases → mean alias embedding
        2. Blend:  e'_y = (1 - eta_e) * e_y + eta_e * alias_mean
        3. Update: p'_y = (1 - eta_p) * p_y + eta_p * e'_y
        4. Recalibrate threshold on validation set (Macro-F1 grid search)
    """
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
            raise ValueError(
                "All embedding/prototype rows must share the same dim"
            )

    encode = text_encoder or default_text_encoder

    E_new: Matrix = []
    P_new: Matrix = []

    label_aliases: dict[str, list[str]] = {}
    label_descriptions: dict[str, str] = {}

    for idx, label_id in enumerate(label_ids):
        semantic_texts = _resolve_label_semantic_text(label_id, label_bank)
        alias_embeddings = [encode(text, dim) for text in semantic_texts]
        alias_mean = normalize(mean_vector(alias_embeddings))

        e_row = blend_and_normalize(E_old[idx], alias_mean, eta=eta_e)
        p_row = blend_and_normalize(P_old[idx], e_row, eta=eta_p)

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
        "sync_type": "fast",
        "eta_e": eta_e,
        "eta_p": eta_p,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "avg_embedding_prototype_alignment": avg_alignment,
        "num_labels": len(label_ids),
        "dim": dim,
    }
    return updated
