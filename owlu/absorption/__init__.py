"""Prototype Absorption Module — absorbs approved labels into LTC-MPE representation space.

Sub-modules:
    metrics   — Shared scoring, inference, and threshold calibration utilities
    fast_sync — Semantic-only refresh (fixed label dimensionality)
    slow_sync — Label expansion + incremental fine-tune
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from ..common.types import Matrix, ValidationSample, Vector
from ..writer.label_bank import LabelBank
from .fast_sync import fast_sync, fast_sync_model
from .slow_sync import slow_sync
from .metrics import (
    score_document,
    infer_topk,
    infer_above_threshold,
    recalibrate_threshold,
    blend_and_normalize_torch,
    recalibrate_model_threshold,
)

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


class PrototypeAbsorption:
    """Facade: absorb new label information into the LTCE model's representation space.

    Two absorption strategies:
        fast_absorb — semantic-only E/P blending (label count unchanged)
        slow_absorb — full label expansion L→L' with incremental fine-tune
    """

    def __init__(
        self,
        label_bank: LabelBank,
        text_encoder: Callable[[str, int], Vector] | None = None,
    ):
        self.label_bank = label_bank
        self.text_encoder = text_encoder

    def fast_absorb(
        self,
        model_state: Mapping[str, object],
        *,
        eta_e: float = 0.2,
        eta_p: float = 0.1,
        validation_set: Sequence[ValidationSample] | None = None,
    ) -> dict[str, object]:
        """Semantic-only refresh: blend alias embeddings into E/P without changing label count.

        Typical trigger: after Writer adds new aliases to existing labels.
        """
        return fast_sync(
            model_state=model_state,
            label_bank=self.label_bank,
            eta_e=eta_e,
            eta_p=eta_p,
            validation_set=validation_set,
            text_encoder=self.text_encoder,
        )

    def slow_absorb(
        self,
        model_state: Mapping[str, object],
        *,
        validation_set: Sequence[ValidationSample] | None = None,
        training_samples: Sequence[ValidationSample] | None = None,
        new_lr: float = 0.05,
        old_lr: float = 0.01,
        eta_e: float = 0.2,
        eta_p: float = 0.1,
    ) -> dict[str, object]:
        """Full label expansion: L → L' with incremental fine-tune.

        Typical trigger: after Writer promotes new labels.
        """
        return slow_sync(
            model_state=model_state,
            label_bank=self.label_bank,
            validation_set=validation_set,
            training_samples=training_samples,
            new_lr=new_lr,
            old_lr=old_lr,
            eta_e=eta_e,
            eta_p=eta_p,
            text_encoder=self.text_encoder,
        )

    # ------------------------------------------------------------------
    # Model-native variant — operates directly on LTCEModel
    # ------------------------------------------------------------------

    def fast_absorb_model(
        self,
        model: object,
        label_ids: Sequence[str],
        *,
        eta_e: float = 0.2,
        eta_p: float = 0.1,
        validation_loader: "DataLoader | None" = None,
        current_threshold: float = 0.45,
        device: "torch.device | None" = None,
    ) -> dict[str, object]:
        """Semantic-only refresh operating on LTCEModel registered buffers.

        In-place updates ``model.label_embeddings`` and ``model.label_prototypes``.
        Returns ``{"threshold": float, "label_aliases": dict, "sync_report": dict}``.
        """
        return fast_sync_model(
            model=model,
            label_bank=self.label_bank,
            label_ids=label_ids,
            eta_e=eta_e,
            eta_p=eta_p,
            validation_loader=validation_loader,
            current_threshold=current_threshold,
            text_encoder=self.text_encoder,
            device=device,
        )
