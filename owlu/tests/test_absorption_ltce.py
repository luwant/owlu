from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from owlu.absorption.ltce_bridge import LtceArtifacts, build_ltce_incremental_loaders
from owlu.absorption.slow_sync import slow_sync_model
from owlu.common.types import LtceTextSample
from owlu.writer.label_bank import LabelBank


class FakeCollator:
    def __init__(self, tokenizer: object, num_labels: int, max_length: int):
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length

    def __call__(self, batch):
        labels = torch.zeros(len(batch), self.num_labels, dtype=torch.float32)
        for idx, sample in enumerate(batch):
            if sample.labels:
                labels[idx, sample.labels] = 1.0

        seq_len = min(self.max_length, 4)
        return {
            "input_ids": torch.ones(len(batch), seq_len, dtype=torch.long),
            "attention_mask": torch.ones(len(batch), seq_len, dtype=torch.long),
            "sentence_map": torch.zeros(len(batch), seq_len, dtype=torch.long),
            "labels": labels,
            "doc_ids": [sample.doc_id for sample in batch],
        }


class DummyLtceModel(torch.nn.Module):
    def __init__(self, num_labels: int, hidden_size: int = 4):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.encoder_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(hidden_size, num_labels),
        )
        self.register_buffer("label_embeddings", torch.eye(num_labels, hidden_size))
        self.register_buffer("label_prototypes", torch.eye(num_labels, hidden_size))
        self.prototype_updates = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        sentence_map: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        keep_ratio: float | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size = input_ids.shape[0]
        pooled = self.encoder_proj(torch.ones(batch_size, self.hidden_size, device=input_ids.device))
        logits = self.classifier(pooled)
        label_representations = pooled.unsqueeze(1).repeat(1, self.num_labels, 1)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
            "label_representations": label_representations,
        }

    def update_prototypes(
        self,
        label_representations: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        del label_representations
        del labels
        self.prototype_updates += 1


@dataclass(frozen=True)
class _Sample:
    doc_id: str
    text: str
    labels: list[int]


def _make_runtime() -> LtceArtifacts:
    train_samples = [_Sample(doc_id="train_0", text="old train", labels=[0])]
    val_samples = [_Sample(doc_id="val_0", text="old val", labels=[1])]
    test_samples = [_Sample(doc_id="test_0", text="old test", labels=[0, 1])]

    config = SimpleNamespace(
        data=SimpleNamespace(max_length=8),
        training=SimpleNamespace(batch_size=2, eval_batch_size=2),
    )

    return LtceArtifacts(
        label_gen_root=SimpleNamespace(),
        config=config,
        dataset_builder=SimpleNamespace(),
        tokenizer=object(),
        collator=FakeCollator(tokenizer=object(), num_labels=2, max_length=8),
        model=object(),
        device=torch.device("cpu"),
        label_ids=["label_a", "label_b"],
        train_loader=SimpleNamespace(dataset=SimpleNamespace(samples=train_samples)),
        validation_loader=SimpleNamespace(dataset=SimpleNamespace(samples=val_samples)),
        test_loader=SimpleNamespace(dataset=SimpleNamespace(samples=test_samples)),
    )


def test_build_ltce_incremental_loaders_expands_label_space() -> None:
    runtime = _make_runtime()
    label_bank = LabelBank()
    label_bank.register_label("label_a", "label a")
    label_bank.register_label("label_b", "label b")
    label_bank.register_label("label_c", "label c")
    label_bank.promoted_labels["label_c"] = object()

    promoted_samples = [
        LtceTextSample(
            doc_id="promoted_0",
            text="new label document",
            true_labels={"label_c"},
            split="train",
        )
    ]

    expanded = build_ltce_incremental_loaders(
        runtime,
        label_bank,
        runtime.label_ids,
        promoted_samples=promoted_samples,
        validation_doc_label_updates={"val_0": ["label_c"]},
        include_base_test=True,
    )

    assert expanded.label_ids == ["label_a", "label_b", "label_c"]
    assert expanded.added_labels == ["label_c"]

    train_batch = next(iter(expanded.train_loader))
    assert train_batch["labels"].shape[1] == 3
    assert train_batch["labels"][:, 2].sum().item() == 1.0

    validation_batch = next(iter(expanded.validation_loader))
    assert validation_batch["labels"].shape[1] == 3
    assert validation_batch["labels"][0, 2].item() == 1.0

    test_batch = next(iter(expanded.test_loader))
    assert test_batch["labels"].shape[1] == 3


def test_slow_sync_model_requires_expanded_loader_dim() -> None:
    label_bank = LabelBank()
    label_bank.register_label("label_a", "label a")
    label_bank.register_label("label_b", "label b")
    label_bank.register_label("label_c", "label c")
    label_bank.promoted_labels["label_c"] = object()

    model = DummyLtceModel(num_labels=2, hidden_size=4)
    label_ids = ["label_a", "label_b"]

    bad_loader = [
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "sentence_map": torch.zeros(1, 4, dtype=torch.long),
            "labels": torch.zeros(1, 2, dtype=torch.float32),
        }
    ]

    def text_encoder(text: str, dim: int) -> list[float]:
        del text
        return [1.0] + [0.0] * (dim - 1)

    try:
        slow_sync_model(
            model,
            label_bank,
            label_ids,
            training_loader=bad_loader,
            text_encoder=text_encoder,
        )
    except ValueError as exc:
        assert "training_loader label dim" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched expanded label dim")


def test_slow_sync_model_expands_and_trains_dummy_ltce() -> None:
    label_bank = LabelBank()
    label_bank.register_label("label_a", "label a")
    label_bank.register_label("label_b", "label b")
    label_bank.register_label("label_c", "label c", aliases=["new alias"])
    label_bank.promoted_labels["label_c"] = object()

    model = DummyLtceModel(num_labels=2, hidden_size=4)
    label_ids = ["label_a", "label_b"]

    loader = [
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "sentence_map": torch.zeros(2, 4, dtype=torch.long),
            "labels": torch.tensor(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        }
    ]

    def text_encoder(text: str, dim: int) -> list[float]:
        base = [0.0] * dim
        base[0] = 1.0
        if "alias" in text:
            base[1] = 1.0
        return base

    result = slow_sync_model(
        model,
        label_bank,
        label_ids,
        training_loader=loader,
        validation_loader=loader,
        text_encoder=text_encoder,
        finetune_epochs=2,
        new_lr=1e-3,
        old_lr=1e-4,
    )

    assert model.num_labels == 3
    assert model.classifier[-1].out_features == 3
    assert label_ids == ["label_a", "label_b", "label_c"]
    assert result["added_labels"] == ["label_c"]
    assert result["label_ids"] == ["label_a", "label_b", "label_c"]
    assert model.prototype_updates > 0
    assert model.label_embeddings.shape == (3, 4)
    assert model.label_prototypes.shape == (3, 4)
