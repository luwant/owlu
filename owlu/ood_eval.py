"""OOD benchmark for comparing LTCE metrics before/after OWLU fast_sync."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Sequence

from .label_bank import LabelBank
from .ltc_sync import ValidationSample, fast_sync


@dataclass(frozen=True)
class OODExample:
    """Single out-of-domain evaluation sample."""

    text: str
    true_labels: set[str]


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom <= 0:
        return 0.0
    return (2.0 * tp) / float(denom)


def _compute_metrics(
    predictions: Sequence[set[str]],
    golds: Sequence[set[str]],
    selected_labels: Sequence[str],
) -> dict[str, float]:
    per_label_f1: list[float] = []
    micro_tp = micro_fp = micro_fn = 0

    for label in selected_labels:
        tp = fp = fn = 0
        for pred, gold in zip(predictions, golds):
            pred_pos = label in pred
            gold_pos = label in gold
            if pred_pos and gold_pos:
                tp += 1
            elif pred_pos and not gold_pos:
                fp += 1
            elif (not pred_pos) and gold_pos:
                fn += 1
        per_label_f1.append(_safe_f1(tp, fp, fn))
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    macro_f1 = sum(per_label_f1) / float(len(per_label_f1)) if per_label_f1 else 0.0
    micro_f1 = _safe_f1(micro_tp, micro_fp, micro_fn)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def build_default_ood_examples(target_labels: Sequence[str]) -> list[OODExample]:
    if len(target_labels) < 3:
        raise ValueError("Need at least 3 target labels")

    y0, y1, y2 = target_labels[0], target_labels[1], target_labels[2]
    return [
        OODExample(
            text="A ransomware crew used a custom loader and encrypted hospital servers overnight.",
            true_labels={y0},
        ),
        OODExample(
            text="Incident responders triaged phishing payloads and credential theft telemetry.",
            true_labels={y0},
        ),
        OODExample(
            text="SOC analysts linked zero-day exploit chains with malware reverse engineering notes.",
            true_labels={y0},
        ),
        OODExample(
            text="Superconducting qubits achieved better quantum error correction under new calibration pulses.",
            true_labels={y1},
        ),
        OODExample(
            text="A variational quantum eigensolver reduced circuit depth while preserving energy estimation quality.",
            true_labels={y1},
        ),
        OODExample(
            text="Photonic entanglement experiments reported improved fidelity in noisy quantum channels.",
            true_labels={y1},
        ),
        OODExample(
            text="CRISPR Cas9 editing reduced off-target mutations in a gene therapy safety study.",
            true_labels={y2},
        ),
        OODExample(
            text="Single-cell RNA sequencing profiled tumor microenvironment changes after immunotherapy.",
            true_labels={y2},
        ),
        OODExample(
            text="Cryo-EM protein folding analysis identified a stable binding pocket for drug design.",
            true_labels={y2},
        ),
    ]


def _build_label_bank(target_labels: Sequence[str]) -> LabelBank:
    if len(target_labels) < 3:
        raise ValueError("Need at least 3 target labels")

    y0, y1, y2 = target_labels[0], target_labels[1], target_labels[2]

    bank = LabelBank(min_freq=3, min_source_docs=2)
    bank.register_label(
        y0,
        "cyber threat operation",
        aliases=[
            "ransomware incident response",
            "phishing credential theft",
            "malware reverse engineering",
            "zero-day exploit detection",
        ],
        description="Security operation and threat detection semantics refresh.",
    )
    bank.register_label(
        y1,
        "quantum computing workflow",
        aliases=[
            "quantum error correction",
            "superconducting qubit calibration",
            "variational quantum eigensolver",
            "entanglement fidelity",
        ],
        description="Quantum algorithm and device calibration semantics refresh.",
    )
    bank.register_label(
        y2,
        "computational biomedicine",
        aliases=[
            "crispr gene editing",
            "single-cell rna sequencing",
            "tumor microenvironment analysis",
            "protein folding structure",
        ],
        description="Biomedical omics and molecular structure semantics refresh.",
    )
    return bank


def evaluate_ltce_fast_sync_ood_delta(
    *,
    checkpoint_dir: str | Path = "e:/lwt/workspace/Label-gen/outputs/aapd_ablation/aapd_full_seed22",
    bert_dir: str | Path = "e:/lwt/workspace/Label-gen/bert/bert-base-uncased",
    target_labels: Sequence[str] = ("label_0", "label_1", "label_2"),
    device: str = "cpu",
    eta_e: float = 0.2,
    eta_p: float = 0.1,
) -> dict[str, object]:
    """Evaluate metrics on OOD examples before and after fast_sync on a real LTCE model."""

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from ltce.config import ExperimentConfig
    from ltce.models import LTCEModel

    ckpt_dir = Path(checkpoint_dir)
    bert_path = Path(bert_dir)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not bert_path.exists():
        raise FileNotFoundError(f"BERT directory not found: {bert_path}")

    cfg = ExperimentConfig.from_yaml(ckpt_dir / "config.yaml")
    cfg.model.bert_model_name_or_path = str(bert_path)
    cfg.model.bert_cache_dir = str(bert_path)

    num_labels = int(cfg.data.num_labels or 0)
    if num_labels <= 0:
        raise ValueError("Invalid num_labels in checkpoint config")

    label_ids = [f"label_{i}" for i in range(num_labels)]
    selected_labels = list(target_labels)
    for lbl in selected_labels:
        if lbl not in label_ids:
            raise ValueError(f"Unknown label id in target_labels: {lbl}")

    model = LTCEModel(num_labels=num_labels, model_cfg=cfg.model)
    state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    # Some historical checkpoints store buffers but not Python-side init flags.
    model.label_embeddings_initialized = True
    model.label_prototypes_initialized = True
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(bert_path), local_files_only=True)

    examples = build_default_ood_examples(selected_labels)
    bank = _build_label_bank(selected_labels)
    threshold_before = float(cfg.training.label_threshold)

    old_E_tensor = model.label_embeddings.detach().clone()
    old_P_tensor = model.label_prototypes.detach().clone()

    def run_eval(current_threshold: float) -> tuple[list[set[str]], list[ValidationSample]]:
        predictions: list[set[str]] = []
        validation_samples: list[ValidationSample] = []

        for example in examples:
            encoded = tokenizer(
                [example.text],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }

            with torch.no_grad():
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                    labels=None,
                )

            probs = torch.sigmoid(outputs["logits"][0]).detach().cpu()
            pooled = outputs["pooled"][0].detach().cpu().tolist()

            pred_labels = {
                label_ids[idx]
                for idx in range(num_labels)
                if float(probs[idx]) >= current_threshold
            }
            predictions.append(pred_labels)
            validation_samples.append(
                ValidationSample(
                    embedding=pooled,
                    true_labels=set(example.true_labels),
                )
            )

        return predictions, validation_samples

    preds_before, validation_set = run_eval(threshold_before)
    golds = [set(example.true_labels) for example in examples]
    before_metrics = _compute_metrics(preds_before, golds, selected_labels)

    text_vec_cache: dict[str, list[float]] = {}
    fallback_pattern = re.compile(r"^label_(\\d+)$")
    old_E_norm = F.normalize(old_E_tensor.detach().cpu(), p=2, dim=-1)

    def encode_text_for_sync(text: str, dim: int) -> list[float]:
        cached = text_vec_cache.get(text)
        if cached is not None:
            return cached

        match = fallback_pattern.fullmatch(text)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < old_E_norm.shape[0]:
                vec = old_E_norm[idx].tolist()
                text_vec_cache[text] = vec
                return vec

        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.no_grad():
            enc_outputs = model.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
                return_dict=True,
            )
        vec_t = enc_outputs.last_hidden_state[:, 0, :][0]
        vec_t = F.normalize(vec_t, p=2, dim=-1)
        vec = vec_t.detach().cpu().tolist()

        if len(vec) != dim:
            raise ValueError(f"Encoded dim mismatch: expected {dim}, got {len(vec)}")

        text_vec_cache[text] = vec
        return vec

    model_state = {
        "label_ids": label_ids,
        "E": old_E_tensor.detach().cpu().tolist(),
        "P": old_P_tensor.detach().cpu().tolist(),
        "threshold": threshold_before,
    }

    synced_state = fast_sync(
        model_state=model_state,
        label_bank=bank,
        eta_e=eta_e,
        eta_p=eta_p,
        validation_set=validation_set,
        text_encoder=encode_text_for_sync,
    )

    with torch.no_grad():
        new_E = torch.tensor(synced_state["E"], dtype=model.label_embeddings.dtype, device=model.label_embeddings.device)
        new_P = torch.tensor(synced_state["P"], dtype=model.label_prototypes.dtype, device=model.label_prototypes.device)
        model.label_embeddings.copy_(new_E)
        model.label_prototypes.copy_(new_P)

    threshold_after = float(synced_state["threshold"])
    preds_after, _ = run_eval(threshold_after)
    after_metrics = _compute_metrics(preds_after, golds, selected_labels)

    delta_metrics = {
        key: float(after_metrics[key] - before_metrics[key])
        for key in before_metrics.keys()
    }

    report = {
        "checkpoint_dir": str(ckpt_dir),
        "num_examples": len(examples),
        "target_labels": selected_labels,
        "threshold_before": threshold_before,
        "threshold_after": threshold_after,
        "before": before_metrics,
        "after": after_metrics,
        "delta": delta_metrics,
        "sync_report": dict(synced_state.get("sync_report", {})),
    }

    # Sanity checks for runtime correctness.
    for value in [
        report["threshold_before"],
        report["threshold_after"],
        report["before"]["micro_f1"],
        report["before"]["macro_f1"],
        report["after"]["micro_f1"],
        report["after"]["macro_f1"],
    ]:
        if not math.isfinite(float(value)):
            raise ValueError("Non-finite metric detected in OOD fast_sync evaluation")

    return report
