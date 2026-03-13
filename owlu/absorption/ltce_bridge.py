"""Bridge utilities for integrating OWLU absorption with the real LTCE stack.

These helpers load the LTCE implementation from the sibling ``Label-gen``
workspace, build the AAPD-compatible data pipeline, and construct expanded
training / validation dataloaders for slow-sync label growth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import os
import sys

from ..common.types import LtceTextSample
from ..writer.label_bank import LabelBank


@dataclass(frozen=True)
class LtceArtifacts:
    """Loaded LTCE runtime objects needed by OWLU absorption."""

    label_gen_root: Path
    config: object
    dataset_builder: object
    tokenizer: object
    collator: object
    model: object
    device: object
    label_ids: list[str]
    train_loader: object
    validation_loader: object
    test_loader: object


@dataclass(frozen=True)
class LtceIncrementalLoaders:
    """Expanded LTCE dataloaders after appending promoted labels."""

    label_ids: list[str]
    added_labels: list[str]
    train_loader: object | None
    validation_loader: object | None
    test_loader: object | None


@dataclass(frozen=True)
class _ExpandedSample:
    doc_id: str
    text: str
    labels: list[int]


class _ExpandedLtceDataset:
    """Dataset wrapper matching the LTCE collator contract."""

    def __init__(self, samples: Sequence[_ExpandedSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> _ExpandedSample:
        return self.samples[index]


def _ensure_ltce_import_path(label_gen_root: Path) -> None:
    src_root = label_gen_root / "src"
    parent_root = label_gen_root.parent

    for candidate in (src_root, parent_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _resolve_label_gen_root(label_gen_root: str | os.PathLike[str] | None) -> Path:
    if label_gen_root:
        root = Path(label_gen_root).expanduser().resolve()
    else:
        env_root = os.environ.get("OWLU_LABEL_GEN_ROOT")
        if env_root:
            root = Path(env_root).expanduser().resolve()
        else:
            root = Path(__file__).resolve().parents[2] / "Label-gen"

    if not root.exists():
        raise FileNotFoundError(f"Label-gen root not found: {root}")
    return root


def _resolve_relative_path(base_root: Path, value: str | None) -> str | None:
    if not value:
        return value

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    candidate = (base_root / path).resolve()
    if candidate.exists():
        return str(candidate)

    return str(path)


def _normalise_ltce_config(cfg: object, label_gen_root: Path) -> object:
    cfg.data.dataset_root = str((label_gen_root / cfg.data.dataset_root).resolve())
    cfg.model.bert_model_name_or_path = _resolve_relative_path(
        label_gen_root,
        cfg.model.bert_model_name_or_path,
    )
    cfg.model.bert_cache_dir = _resolve_relative_path(
        label_gen_root,
        cfg.model.bert_cache_dir,
    )
    cfg.output_dir = _resolve_relative_path(label_gen_root, cfg.output_dir) or cfg.output_dir

    tfidf_path = getattr(cfg.model, "tfidf_idf_path", None)
    if tfidf_path:
        dataset_root = Path(cfg.data.dataset_root)
        resolved = (dataset_root / tfidf_path).resolve()
        if resolved.exists():
            cfg.model.tfidf_idf_path = str(resolved)
        else:
            cfg.model.tfidf_idf_path = _resolve_relative_path(label_gen_root, tfidf_path)

    return cfg


def _import_ltce_modules(label_gen_root: Path) -> dict[str, Any]:
    _ensure_ltce_import_path(label_gen_root)

    from ltce import ExperimentConfig
    from ltce.data import LtceBatchCollator, LtceDatasetBuilder
    from ltce.models import LTCEModel
    from ltce.training_utils import load_checkpoint, load_label_embeddings
    from transformers import AutoTokenizer
    import torch

    return {
        "ExperimentConfig": ExperimentConfig,
        "LtceBatchCollator": LtceBatchCollator,
        "LtceDatasetBuilder": LtceDatasetBuilder,
        "LTCEModel": LTCEModel,
        "load_checkpoint": load_checkpoint,
        "load_label_embeddings": load_label_embeddings,
        "AutoTokenizer": AutoTokenizer,
        "torch": torch,
    }


def _resolve_device(torch_mod: Any, requested_device: str | None, cfg_device: str) -> Any:
    device_name = requested_device or cfg_device
    if device_name.startswith("cuda") and not torch_mod.cuda.is_available():
        device_name = "cpu"
    return torch_mod.device(device_name)


def _load_label_ids(dataset_builder: object) -> list[str]:
    label_ids = [str(label_id) for label_id in getattr(dataset_builder, "label_list", [])]
    if label_ids:
        return label_ids

    num_labels = int(getattr(dataset_builder, "num_labels"))
    return [f"label_{idx}" for idx in range(num_labels)]


def load_ltce_artifacts(
    config_path: str | os.PathLike[str],
    *,
    checkpoint_path: str | os.PathLike[str] | None = None,
    label_gen_root: str | os.PathLike[str] | None = None,
    device: str | None = None,
    num_workers: int = 0,
) -> LtceArtifacts:
    """Load the real LTCE model, tokenizer, datasets, and dataloaders.

    ``config_path`` may be absolute or relative to ``Label-gen``.
    ``checkpoint_path`` is optional; when omitted, the returned model contains
    the configured label embeddings but no fine-tuned checkpoint weights.
    """
    root = _resolve_label_gen_root(label_gen_root)
    modules = _import_ltce_modules(root)
    torch_mod = modules["torch"]

    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (root / config_file).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"LTCE config not found: {config_file}")

    cfg = modules["ExperimentConfig"].from_yaml(str(config_file))
    cfg = _normalise_ltce_config(cfg, root)
    runtime_device = _resolve_device(torch_mod, device, cfg.device)

    dataset_builder = modules["LtceDatasetBuilder"](cfg.data, seed=cfg.seed)
    train_dataset, val_dataset, test_dataset = dataset_builder.build_datasets()

    tokenizer = modules["AutoTokenizer"].from_pretrained(
        cfg.model.bert_model_name_or_path,
        cache_dir=cfg.model.bert_cache_dir,
        use_fast=True,
    )
    collator = modules["LtceBatchCollator"](
        tokenizer=tokenizer,
        num_labels=dataset_builder.num_labels,
        max_length=cfg.data.max_length,
    )

    train_loader = torch_mod.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    validation_loader = torch_mod.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = torch_mod.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    model = modules["LTCEModel"](dataset_builder.num_labels, cfg.model)
    model.to(runtime_device)
    modules["load_label_embeddings"](
        model,
        dataset_builder,
        cfg.data.label_embeddings_path,
        tokenizer,
    )

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (root / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"LTCE checkpoint not found: {ckpt_path}")
        modules["load_checkpoint"](model, ckpt_path, map_location=runtime_device)

    return LtceArtifacts(
        label_gen_root=root,
        config=cfg,
        dataset_builder=dataset_builder,
        tokenizer=tokenizer,
        collator=collator,
        model=model,
        device=runtime_device,
        label_ids=_load_label_ids(dataset_builder),
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )


def _resolve_new_label_ids(label_ids: Sequence[str], label_bank: LabelBank) -> list[str]:
    existing = set(label_ids)
    return [str(label_id) for label_id in label_bank.promoted_labels if label_id not in existing]


def _normalise_true_labels(true_labels: Iterable[str]) -> set[str]:
    return {str(label_id) for label_id in true_labels}


def _augment_existing_labels(
    labels: list[int],
    doc_id: str,
    label_to_index: Mapping[str, int],
    doc_label_updates: Mapping[str, Sequence[str]] | None,
) -> list[int]:
    updated = set(int(label_idx) for label_idx in labels)
    if doc_label_updates and doc_id in doc_label_updates:
        for label_id in doc_label_updates[doc_id]:
            if label_id not in label_to_index:
                raise KeyError(f"Unknown label_id in doc_label_updates: {label_id}")
            updated.add(label_to_index[label_id])
    return sorted(updated)


def _build_expanded_samples(
    base_samples: Sequence[object],
    *,
    label_to_index: Mapping[str, int],
    doc_label_updates: Mapping[str, Sequence[str]] | None,
) -> list[_ExpandedSample]:
    samples: list[_ExpandedSample] = []
    for sample in base_samples:
        labels = _augment_existing_labels(
            list(getattr(sample, "labels")),
            str(getattr(sample, "doc_id")),
            label_to_index,
            doc_label_updates,
        )
        samples.append(
            _ExpandedSample(
                doc_id=str(getattr(sample, "doc_id")),
                text=str(getattr(sample, "text")),
                labels=labels,
            )
        )
    return samples


def _build_promoted_samples(
    samples: Sequence[LtceTextSample],
    *,
    label_to_index: Mapping[str, int],
) -> list[_ExpandedSample]:
    promoted: list[_ExpandedSample] = []
    for sample in samples:
        labels: list[int] = []
        for label_id in _normalise_true_labels(sample.true_labels):
            if label_id not in label_to_index:
                raise KeyError(f"Unknown label_id in LtceTextSample: {label_id}")
            labels.append(label_to_index[label_id])
        promoted.append(
            _ExpandedSample(
                doc_id=str(sample.doc_id),
                text=str(sample.text),
                labels=sorted(set(labels)),
            )
        )
    return promoted


def _split_promoted_samples(
    samples: Sequence[LtceTextSample] | None,
) -> dict[str, list[LtceTextSample]]:
    buckets: dict[str, list[LtceTextSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for sample in samples or []:
        buckets[str(sample.split)].append(sample)
    return buckets


def build_ltce_incremental_loaders(
    runtime: LtceArtifacts,
    label_bank: LabelBank,
    label_ids: Sequence[str],
    *,
    promoted_samples: Sequence[LtceTextSample] | None = None,
    train_doc_label_updates: Mapping[str, Sequence[str]] | None = None,
    validation_doc_label_updates: Mapping[str, Sequence[str]] | None = None,
    test_doc_label_updates: Mapping[str, Sequence[str]] | None = None,
    include_base_train: bool = True,
    include_base_validation: bool = True,
    include_base_test: bool = False,
    num_workers: int = 0,
) -> LtceIncrementalLoaders:
    """Expand LTCE dataloaders from ``L`` to ``L'`` for slow-sync training.

    Existing LTCE samples keep their legacy labels and receive zero-valued new
    label slots unless a ``*_doc_label_updates`` mapping is provided. New-label
    positive examples can be appended through ``promoted_samples``.
    """
    import torch

    added_labels = _resolve_new_label_ids(label_ids, label_bank)
    expanded_label_ids = [str(label_id) for label_id in label_ids] + added_labels
    label_to_index = {
        str(label_id): idx
        for idx, label_id in enumerate(expanded_label_ids)
    }

    promoted_buckets = _split_promoted_samples(promoted_samples)

    train_samples: list[_ExpandedSample] = []
    validation_samples: list[_ExpandedSample] = []
    test_samples: list[_ExpandedSample] = []

    if include_base_train:
        train_samples.extend(
            _build_expanded_samples(
                runtime.train_loader.dataset.samples,
                label_to_index=label_to_index,
                doc_label_updates=train_doc_label_updates,
            )
        )
    if include_base_validation:
        validation_samples.extend(
            _build_expanded_samples(
                runtime.validation_loader.dataset.samples,
                label_to_index=label_to_index,
                doc_label_updates=validation_doc_label_updates,
            )
        )
    if include_base_test:
        test_samples.extend(
            _build_expanded_samples(
                runtime.test_loader.dataset.samples,
                label_to_index=label_to_index,
                doc_label_updates=test_doc_label_updates,
            )
        )

    train_samples.extend(
        _build_promoted_samples(
            promoted_buckets["train"],
            label_to_index=label_to_index,
        )
    )
    validation_samples.extend(
        _build_promoted_samples(
            promoted_buckets["val"],
            label_to_index=label_to_index,
        )
    )
    test_samples.extend(
        _build_promoted_samples(
            promoted_buckets["test"],
            label_to_index=label_to_index,
        )
    )

    collator = type(runtime.collator)(
        tokenizer=runtime.tokenizer,
        num_labels=len(expanded_label_ids),
        max_length=runtime.config.data.max_length,
    )

    train_loader = None
    if train_samples:
        train_loader = torch.utils.data.DataLoader(
            _ExpandedLtceDataset(train_samples),
            batch_size=runtime.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
        )

    validation_loader = None
    if validation_samples:
        validation_loader = torch.utils.data.DataLoader(
            _ExpandedLtceDataset(validation_samples),
            batch_size=runtime.config.training.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
        )

    test_loader = None
    if test_samples:
        test_loader = torch.utils.data.DataLoader(
            _ExpandedLtceDataset(test_samples),
            batch_size=runtime.config.training.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
        )

    return LtceIncrementalLoaders(
        label_ids=expanded_label_ids,
        added_labels=added_labels,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )
