from __future__ import annotations

import os
from pathlib import Path

import pytest

from owlu.label_bank import LabelBank
from owlu.ltc_sync import ValidationSample, fast_sync


@pytest.mark.integration
def test_fast_sync_ltce_model_integration():
    """Run fast_sync against a real LTCEModel instance and verify inference compatibility."""

    if os.getenv("OWLU_RUN_LTCE_INTEGRATION", "0") != "1":
        pytest.skip("Set OWLU_RUN_LTCE_INTEGRATION=1 to run LTCE integration test.")

    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torch.nn.functional")
    transformers = pytest.importorskip("transformers")

    from ltce.config import ModelConfig
    from ltce.models import LTCEModel

    bert_dir = Path("e:/lwt/workspace/Label-gen/bert/bert-base-uncased")
    if not bert_dir.exists():
        pytest.skip(f"Local BERT checkpoint not found: {bert_dir}")

    model_cfg = ModelConfig(
        bert_model_name_or_path=str(bert_dir),
        use_shsa=False,
        use_bilstm=False,
        use_co_attention=False,
        use_adaptive_fusion=False,
        use_prototype_fusion=True,
        use_gradient_checkpointing=False,
        word_filter_keep_ratio=0.4,
        word_filter_min_tokens=2,
    )

    label_ids = ["l1", "l2", "l3"]
    model = LTCEModel(num_labels=len(label_ids), model_cfg=model_cfg)
    model.eval()

    with torch.no_grad():
        init_embeddings = F.normalize(torch.randn(len(label_ids), model.hidden_size), p=2, dim=-1)
        model.update_label_embeddings(init_embeddings)

    old_E = model.label_embeddings.detach().clone()
    old_P = model.label_prototypes.detach().clone()

    bank = LabelBank(min_freq=3, min_source_docs=2)
    bank.register_label("l1", "malware analysis", aliases=["malware reverse engineering"])
    bank.register_label("l2", "network defense", aliases=["network hardening"])
    bank.register_label("l3", "threat intelligence", aliases=["threat intel"])

    validation = [
        ValidationSample(embedding=old_P[0].detach().cpu().tolist(), true_labels={"l1"}),
        ValidationSample(embedding=old_P[1].detach().cpu().tolist(), true_labels={"l2"}),
        ValidationSample(embedding=old_P[2].detach().cpu().tolist(), true_labels={"l3"}),
    ]

    model_state = {
        "label_ids": label_ids,
        "E": old_E.detach().cpu().tolist(),
        "P": old_P.detach().cpu().tolist(),
        "threshold": 0.50,
    }

    synced = fast_sync(
        model_state=model_state,
        label_bank=bank,
        eta_e=0.2,
        eta_p=0.1,
        validation_set=validation,
    )

    with torch.no_grad():
        new_E = torch.tensor(synced["E"], dtype=model.label_embeddings.dtype, device=model.label_embeddings.device)
        new_P = torch.tensor(synced["P"], dtype=model.label_prototypes.dtype, device=model.label_prototypes.device)
        model.label_embeddings.copy_(new_E)
        model.label_prototypes.copy_(new_P)

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    encoded = tokenizer(
        ["malware reverse engineering identifies loader behavior"],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    labels = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
            labels=labels,
        )

    assert outputs["logits"].shape == (1, len(label_ids))
    assert torch.isfinite(outputs["logits"]).all()
    assert model.classifier[-1].out_features == len(label_ids)
    assert not torch.allclose(old_E, model.label_embeddings)
    assert not torch.allclose(old_P, model.label_prototypes)

    alignment = torch.nn.functional.cosine_similarity(
        model.label_embeddings,
        model.label_prototypes,
        dim=-1,
    ).mean().item()
    assert alignment >= 0.85
    assert 0.10 <= float(synced["threshold"]) <= 0.90
