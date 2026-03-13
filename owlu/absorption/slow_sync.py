"""Slow sync: label expansion with incremental fine-tuning."""

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
    """One-pass update for dict-mode prototype absorption."""
    if not training_samples:
        return E, P

    for label_idx, label_id in enumerate(label_ids):
        pos_vecs: list[Vector] = []
        for sample in training_samples:
            if label_id in sample.true_labels:
                pos_vecs.append(list(sample.embedding))

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
    """Expand label space in dict mode and run a lightweight update."""
    encode = text_encoder or default_text_encoder

    label_ids = [str(v) for v in list(model_state["label_ids"])]
    E = [[float(x) for x in row] for row in list(model_state["E"])]
    P = [[float(x) for x in row] for row in list(model_state["P"])]

    if not E:
        raise ValueError("E cannot be empty")
    dim = len(E[0])

    promoted = label_bank.promoted_labels
    new_label_ids: list[str] = []
    existing = set(label_ids)
    for lid in promoted:
        if lid not in existing:
            new_label_ids.append(str(lid))

    if not new_label_ids:
        from .fast_sync import fast_sync

        return fast_sync(
            model_state=model_state,
            label_bank=label_bank,
            eta_e=eta_e,
            eta_p=eta_p,
            validation_set=validation_set,
            text_encoder=text_encoder,
        )

    for lid in new_label_ids:
        texts = _resolve_label_semantic_text(lid, label_bank)
        vecs = [encode(text, dim) for text in texts]
        init_vec = normalize(mean_vector(vecs))
        E.append(list(init_vec))
        P.append(list(init_vec))
        label_ids.append(lid)

    if training_samples:
        E, P = _incremental_finetune(
            E,
            P,
            label_ids,
            set(new_label_ids),
            training_samples,
            new_lr=new_lr,
            old_lr=old_lr,
        )

    old_threshold = float(model_state.get("threshold", 0.5))
    new_threshold = recalibrate_threshold(
        P,
        label_ids,
        validation_set or [],
        current_threshold=old_threshold,
    )

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


def _expand_classifier(
    model: object,
    old_num_labels: int,
    new_num_labels: int,
    *,
    init_rows: Sequence[Sequence[float]] | None = None,
) -> None:
    """Expand the final LTCE linear classifier from L to L'."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seq: nn.Sequential = model.classifier  # type: ignore[union-attr]
    old_linear: nn.Linear = seq[-1]  # type: ignore[assignment]
    in_features = old_linear.in_features

    new_linear = nn.Linear(
        in_features,
        new_num_labels,
        bias=old_linear.bias is not None,
    )
    new_linear = new_linear.to(
        device=old_linear.weight.device,
        dtype=old_linear.weight.dtype,
    )

    with torch.no_grad():
        new_linear.weight[:old_num_labels] = old_linear.weight
        if old_linear.bias is not None and new_linear.bias is not None:
            new_linear.bias[:old_num_labels] = old_linear.bias

        new_rows = new_num_labels - old_num_labels
        if new_rows > 0:
            semantic_init = None
            if init_rows is not None and len(init_rows) == new_rows:
                semantic_init = torch.tensor(
                    init_rows,
                    device=old_linear.weight.device,
                    dtype=old_linear.weight.dtype,
                )
                if semantic_init.shape != (new_rows, in_features):
                    semantic_init = None

            if semantic_init is not None:
                semantic_init = F.normalize(semantic_init, p=2, dim=-1)
                semantic_init = torch.nan_to_num(
                    semantic_init, nan=0.0, posinf=0.0, neginf=0.0,
                )
                new_linear.weight[old_num_labels:] = semantic_init
            else:
                nn.init.xavier_uniform_(new_linear.weight[old_num_labels:])

            if new_linear.bias is not None:
                if old_linear.bias is not None and old_num_labels > 0:
                    fill_value = float(old_linear.bias.mean().item())
                    new_linear.bias[old_num_labels:].fill_(fill_value)
                else:
                    new_linear.bias[old_num_labels:].zero_()

    seq[-1] = new_linear


def _expand_buffer(
    model: object,
    buffer_name: str,
    old_num_labels: int,
    new_num_labels: int,
    hidden_size: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Re-register a model buffer from (L, H) to (L', H)."""
    import torch

    old_buf: torch.Tensor = getattr(model, buffer_name)
    new_buf = torch.zeros(new_num_labels, hidden_size, device=device, dtype=dtype)
    new_buf[:old_num_labels] = old_buf
    model.register_buffer(buffer_name, new_buf)  # type: ignore[union-attr]
    return getattr(model, buffer_name)


def _peek_loader_label_dim(loader: "DataLoader | None") -> int | None:
    if loader is None:
        return None

    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        return None

    labels = batch.get("labels")
    if labels is None:
        raise KeyError("Expected batch['labels'] in LTCE dataloader")
    if labels.ndim != 2:
        raise ValueError("Expected batch['labels'] to be a 2-D tensor")
    return int(labels.shape[1])


def _register_new_label_lr_hooks(
    linear: "torch.nn.Linear",
    old_num_labels: int,
    new_num_labels: int,
    *,
    new_lr: float,
    old_lr: float,
) -> list[object]:
    """Scale classifier gradients so new rows train with a larger effective LR."""
    import torch

    if old_lr <= 0.0:
        raise ValueError("old_lr must be > 0")

    row_scale = float(new_lr) / float(old_lr)
    if abs(row_scale - 1.0) <= 1e-12:
        return []

    handles: list[object] = []
    weight_mask = torch.ones_like(linear.weight)
    weight_mask[old_num_labels:new_num_labels] = row_scale
    handles.append(linear.weight.register_hook(lambda grad: grad * weight_mask))

    if linear.bias is not None:
        bias_mask = torch.ones_like(linear.bias)
        bias_mask[old_num_labels:new_num_labels] = row_scale
        handles.append(linear.bias.register_hook(lambda grad: grad * bias_mask))

    return handles


def _incremental_finetune_model(
    model: object,
    label_ids: list[str],
    promoted_ids: set[str],
    training_loader: "DataLoader",
    *,
    new_lr: float,
    old_lr: float,
    finetune_epochs: int,
    update_prototypes: bool,
    classifier_anchor_weight: float,
    device: "torch.device",
) -> None:
    """Incrementally fine-tune the real LTCE model on the expanded label space."""
    import torch
    import torch.nn.functional as F

    model.train()  # type: ignore[union-attr]

    classifier_last = model.classifier[-1]  # type: ignore[union-attr]
    old_num_labels = len(label_ids) - len(promoted_ids)
    optimizer = torch.optim.AdamW(
        model.parameters(),  # type: ignore[union-attr]
        lr=old_lr,
        weight_decay=1e-4,
    )

    hook_handles = _register_new_label_lr_hooks(
        classifier_last,
        old_num_labels,
        len(label_ids),
        new_lr=new_lr,
        old_lr=old_lr,
    )

    anchor_weight = max(float(classifier_anchor_weight), 0.0)
    if old_num_labels > 0 and anchor_weight > 0.0:
        old_weight_anchor = classifier_last.weight[:old_num_labels].detach().clone()
        old_bias_anchor = None
        if classifier_last.bias is not None:
            old_bias_anchor = classifier_last.bias[:old_num_labels].detach().clone()
    else:
        old_weight_anchor = None
        old_bias_anchor = None

    try:
        for _epoch in range(finetune_epochs):
            for batch in training_loader:
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                labels = batch["labels"].float()
                if labels.ndim != 2 or labels.shape[1] != len(label_ids):
                    raise ValueError(
                        "training_loader labels must have shape "
                        f"(B, {len(label_ids)}), got {tuple(labels.shape)}"
                    )

                outputs = model(  # type: ignore[operator]
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    sentence_map=batch.get("sentence_map"),
                    labels=labels,
                )
                loss = outputs["loss"]
                if loss is None:
                    raise RuntimeError("LTCE model did not return BCE loss")

                if update_prototypes and hasattr(model, "update_prototypes"):
                    model.update_prototypes(outputs["label_representations"], labels)  # type: ignore[union-attr]

                if old_weight_anchor is not None:
                    anchor_loss = F.mse_loss(
                        classifier_last.weight[:old_num_labels],
                        old_weight_anchor,
                    )
                    if old_bias_anchor is not None and classifier_last.bias is not None:
                        anchor_loss = anchor_loss + F.mse_loss(
                            classifier_last.bias[:old_num_labels],
                            old_bias_anchor,
                        )
                    loss = loss + (anchor_weight * anchor_loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # type: ignore[union-attr]
                optimizer.step()
    finally:
        for handle in hook_handles:
            handle.remove()

    model.eval()  # type: ignore[union-attr]


def slow_sync_model(
    model: object,
    label_bank: "LabelBank",
    label_ids: list[str],
    *,
    eta_e: float = 0.2,
    eta_p: float = 0.1,
    validation_loader: "DataLoader | None" = None,
    training_loader: "DataLoader | None" = None,
    current_threshold: float = 0.45,
    new_lr: float = 5e-5,
    old_lr: float = 1e-5,
    finetune_epochs: int = 3,
    update_prototypes: bool = True,
    classifier_anchor_weight: float = 0.05,
    text_encoder: Callable[[str, int], Vector] | None = None,
    device: "torch.device | None" = None,
) -> dict[str, object]:
    """Expand the real LTCE model from L to L' and fine-tune incrementally."""
    import torch
    import torch.nn.functional as F

    from .metrics import (
        blend_and_normalize_torch,
        recalibrate_model_threshold,
    )

    E: torch.Tensor = model.label_embeddings  # type: ignore[union-attr]
    P: torch.Tensor = model.label_prototypes  # type: ignore[union-attr]
    old_num_labels, hidden_size = E.shape

    if device is None:
        device = E.device
    encode = text_encoder or default_text_encoder

    existing_set = set(label_ids)
    new_label_ids: list[str] = []
    for lid in label_bank.promoted_labels:
        if lid not in existing_set:
            new_label_ids.append(str(lid))

    if not new_label_ids:
        from .fast_sync import fast_sync_model

        return fast_sync_model(
            model=model,
            label_bank=label_bank,
            label_ids=label_ids,
            eta_e=eta_e,
            eta_p=eta_p,
            validation_loader=validation_loader,
            current_threshold=current_threshold,
            text_encoder=text_encoder,
            device=device,
        )

    new_num_labels = old_num_labels + len(new_label_ids)

    new_label_init_vectors: list[torch.Tensor] = []
    for lid in new_label_ids:
        texts = _resolve_label_semantic_text(lid, label_bank)
        alias_vecs = [encode(text, hidden_size) for text in texts]
        alias_tensor = torch.tensor(alias_vecs, dtype=E.dtype, device=device)
        init_vec = F.normalize(alias_tensor.mean(dim=0), p=2, dim=0)
        init_vec = torch.nan_to_num(init_vec, nan=0.0, posinf=0.0, neginf=0.0)
        new_label_init_vectors.append(init_vec)

    _expand_buffer(
        model,
        "label_embeddings",
        old_num_labels,
        new_num_labels,
        hidden_size,
        device,
        E.dtype,
    )
    _expand_buffer(
        model,
        "label_prototypes",
        old_num_labels,
        new_num_labels,
        hidden_size,
        device,
        P.dtype,
    )
    _expand_classifier(
        model,
        old_num_labels,
        new_num_labels,
        init_rows=[vec.detach().cpu().tolist() for vec in new_label_init_vectors],
    )
    model.num_labels = new_num_labels  # type: ignore[union-attr]

    E = model.label_embeddings  # type: ignore[union-attr]
    P = model.label_prototypes  # type: ignore[union-attr]

    with torch.no_grad():
        for offset, lid in enumerate(new_label_ids):
            idx = old_num_labels + offset
            init_vec = new_label_init_vectors[offset]
            E[idx] = init_vec
            P[idx] = init_vec
            label_ids.append(lid)

    with torch.no_grad():
        for idx in range(old_num_labels):
            lid = label_ids[idx]
            texts = _resolve_label_semantic_text(lid, label_bank)
            alias_vecs = [encode(text, hidden_size) for text in texts]
            alias_tensor = torch.tensor(alias_vecs, dtype=E.dtype, device=device)
            alias_mean = F.normalize(alias_tensor.mean(dim=0), p=2, dim=0)
            alias_mean = torch.nan_to_num(alias_mean, nan=0.0, posinf=0.0, neginf=0.0)
            E[idx] = blend_and_normalize_torch(E[idx], alias_mean, eta=eta_e)
            P[idx] = blend_and_normalize_torch(P[idx], E[idx], eta=eta_p)

    observed_train_dim = _peek_loader_label_dim(training_loader)
    if observed_train_dim is not None and observed_train_dim != new_num_labels:
        raise ValueError(
            f"training_loader label dim ({observed_train_dim}) != expanded label dim ({new_num_labels})"
        )
    observed_validation_dim = _peek_loader_label_dim(validation_loader)
    if observed_validation_dim is not None and observed_validation_dim != new_num_labels:
        raise ValueError(
            "validation_loader label dim "
            f"({observed_validation_dim}) != expanded label dim ({new_num_labels})"
        )

    if training_loader is not None:
        _incremental_finetune_model(
            model,
            label_ids,
            set(new_label_ids),
            training_loader,
            new_lr=new_lr,
            old_lr=old_lr,
            finetune_epochs=finetune_epochs,
            update_prototypes=update_prototypes,
            classifier_anchor_weight=classifier_anchor_weight,
            device=device,
        )

    old_threshold = float(current_threshold)
    if validation_loader is not None:
        model.eval()  # type: ignore[union-attr]
        new_threshold = recalibrate_model_threshold(
            model,
            validation_loader,
            device,
            old_threshold,
        )
    else:
        new_threshold = old_threshold

    with torch.no_grad():
        cos = F.cosine_similarity(
            model.label_embeddings,  # type: ignore[union-attr]
            model.label_prototypes,  # type: ignore[union-attr]
            dim=-1,
        )
        avg_alignment = float(cos.mean().item())

    sync_report = {
        "sync_type": "slow",
        "old_num_labels": old_num_labels,
        "new_num_labels": new_num_labels,
        "added_labels": new_label_ids,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "avg_embedding_prototype_alignment": avg_alignment,
        "finetune_epochs": finetune_epochs,
        "new_lr": new_lr,
        "old_lr": old_lr,
        "update_prototypes": bool(update_prototypes),
        "classifier_anchor_weight": float(classifier_anchor_weight),
        "dim": hidden_size,
    }

    return {
        "threshold": new_threshold,
        "added_labels": new_label_ids,
        "label_ids": label_ids,
        "sync_report": sync_report,
    }
