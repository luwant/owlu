"""Fast sync — semantic-only refresh of label embeddings / prototypes.

Updates E and P by blending alias text embeddings while keeping
the label count (dimensionality) stable.

Two entry points:
    fast_sync       — pure-Python, operates on model_state dict.
    fast_sync_model — torch-native, operates on LTCEModel registered buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

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


# ---------------------------------------------------------------------------
# Model-native variant — operates on LTCEModel registered buffers
# ---------------------------------------------------------------------------

def fast_sync_model(
    model: object,
    label_bank: "LabelBank",
    label_ids: Sequence[str],
    *,
    eta_e: float = 0.2,
    eta_p: float = 0.1,
    validation_loader: "DataLoader | None" = None,
    current_threshold: float = 0.45,
    text_encoder: Callable[[str, int], "Vector"] | None = None,
    device: "torch.device | None" = None,
) -> dict[str, object]:
    """Semantic-only refresh operating directly on LTCEModel registered buffers.

    In-place updates ``model.label_embeddings`` and ``model.label_prototypes``.
    Threshold is recalibrated using actual model forward passes when
    *validation_loader* is provided.

    Parameters
    ----------
    model : LTCEModel
        A loaded LTCE model with initialised label_embeddings / label_prototypes.
    label_bank : LabelBank
        Label bank containing the latest alias information.
    label_ids : Sequence[str]
        Ordered label names corresponding to model rows (dim-0 of E / P).
    eta_e : float
        Embedding blend weight (0 = keep original, 1 = full alias embedding).
    eta_p : float
        Prototype blend weight.
    validation_loader : DataLoader | None
        If given, used for Macro-F1 grid-search threshold recalibration.
    current_threshold : float
        The current decision threshold (sigmoid space).
    text_encoder : Callable | None
        ``(text, dim) -> Vector``.  Falls back to ``default_text_encoder``.
    device : torch.device | None
        Inferred from model parameters when ``None``.

    Returns
    -------
    dict with keys: ``threshold``, ``sync_report``.
    """
    import torch
    import torch.nn.functional as F

    from .metrics import (
        blend_and_normalize_torch,
        default_text_encoder,
        recalibrate_model_threshold,
    )

    if not (0.0 <= eta_e <= 1.0):
        raise ValueError("eta_e must be in [0, 1]")
    if not (0.0 <= eta_p <= 1.0):
        raise ValueError("eta_p must be in [0, 1]")

    E: torch.Tensor = model.label_embeddings  # type: ignore[union-attr]
    P: torch.Tensor = model.label_prototypes   # type: ignore[union-attr]
    num_labels, dim = E.shape

    if len(label_ids) != num_labels:
        raise ValueError(
            f"label_ids length ({len(label_ids)}) != model num_labels ({num_labels})"
        )

    if device is None:
        device = E.device
    encode = text_encoder or default_text_encoder

    label_aliases: dict[str, list[str]] = {}

    with torch.no_grad():
        for idx, label_id in enumerate(label_ids):
            semantic_texts = _resolve_label_semantic_text(label_id, label_bank)
            alias_vecs = [encode(text, dim) for text in semantic_texts]

            # Average alias embeddings → torch tensor
            alias_tensor = torch.tensor(alias_vecs, dtype=E.dtype, device=device)
            alias_mean = F.normalize(alias_tensor.mean(dim=0, keepdim=False), p=2, dim=0)

            E[idx] = blend_and_normalize_torch(E[idx], alias_mean, eta=eta_e)
            P[idx] = blend_and_normalize_torch(P[idx], E[idx],     eta=eta_p)

            label_aliases[label_id] = semantic_texts

    # --- Threshold recalibration ---
    old_threshold = float(current_threshold)
    if validation_loader is not None:
        new_threshold = recalibrate_model_threshold(
            model, validation_loader, device, old_threshold,
        )
    else:
        new_threshold = old_threshold

    # --- Alignment statistics ---
    with torch.no_grad():
        cos = F.cosine_similarity(E, P, dim=-1)
        avg_alignment = float(cos.mean().item())

    sync_report = {
        "sync_type": "fast",
        "eta_e": eta_e,
        "eta_p": eta_p,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "avg_embedding_prototype_alignment": avg_alignment,
        "num_labels": num_labels,
        "dim": dim,
    }

    return {
        "threshold": new_threshold,
        "label_aliases": label_aliases,
        "sync_report": sync_report,
    }
