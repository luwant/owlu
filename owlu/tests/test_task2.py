from __future__ import annotations

from datetime import datetime, timezone

import pytest

from owlu.label_bank import LabelBank
from owlu.llm_phrase_generator import CandidatePhrase
from owlu.ltc_sync import ValidationSample, fast_sync, infer_above_threshold, infer_topk


def _phrase(text: str, doc_id: str, agreement: float = 1.0) -> CandidatePhrase:
    return CandidatePhrase(
        text=text,
        raw_text=text,
        source_doc_id=doc_id,
        timestamp=datetime.now(timezone.utc),
        agreement=agreement,
    )


def _base_model_state() -> dict[str, object]:
    return {
        "label_ids": ["l1", "l2", "l3"],
        "E": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "P": [
            [0.95, 0.05, 0.0, 0.0],
            [0.05, 0.95, 0.0, 0.0],
            [0.0, 0.0, 0.95, 0.05],
        ],
        "threshold": 0.5,
    }


@pytest.fixture()
def label_bank() -> LabelBank:
    bank = LabelBank(min_freq=3, min_source_docs=2)
    bank.register_label("l1", "malware analysis", aliases=["malware analytics"], description="malware")
    bank.register_label("l2", "network defense", aliases=["blue team"], description="defense")
    bank.register_label("l3", "threat intelligence", aliases=["threat intel"], description="intel")
    bank.add_alias("l1", "malware reverse engineering")
    bank.add_alias("l2", "network hardening")
    return bank


def test_add_candidate_accumulates_freq():
    bank = LabelBank(min_freq=3, min_source_docs=2)
    c1 = _phrase("supply chain attack", "doc-1", agreement=0.9)
    c2 = _phrase("supply chain attack", "doc-1", agreement=0.8)
    c3 = _phrase("supply chain attack", "doc-2", agreement=0.7)

    assert bank.add_candidate(c1) == "hold"
    assert bank.add_candidate(c2) == "hold"
    assert bank.add_candidate(c3) == "candidate"

    cluster = bank.proto_label_clusters["supply chain attack"]
    assert cluster.freq == 3
    assert cluster.source_doc_count == 2
    assert cluster.agreement == pytest.approx((0.9 + 0.8 + 0.7) / 3.0)


def test_hold_pool_insert_and_query():
    bank = LabelBank()
    phrase = _phrase("credential stuffing", "doc-hold", agreement=0.6)

    assert bank.add_hold(phrase) == "hold"
    cluster = bank.get_hold_cluster("credential stuffing")

    assert cluster is not None
    assert cluster.freq == 1
    assert cluster.source_doc_count == 1


def test_cluster_build_and_summary():
    bank = LabelBank(min_freq=3, min_source_docs=2)
    bank.add_candidate(_phrase("APT campaign", "d1", 0.8), cluster_id="apt cluster", nearest_label_distance=0.3)
    bank.add_candidate(_phrase("advanced persistent threat", "d2", 0.7), cluster_id="apt cluster", nearest_label_distance=0.2)
    bank.add_candidate(_phrase("APT campaign", "d3", 0.9), cluster_id="apt cluster", nearest_label_distance=0.4)

    summary = bank.summarize_cluster("apt cluster")
    packet = bank.build_review_packet("apt cluster")

    assert summary["state"] == "candidate"
    assert summary["freq"] == 3
    assert summary["source_docs"] == 3
    assert summary["agreement"] == pytest.approx((0.8 + 0.7 + 0.9) / 3.0)
    assert packet["representative_phrase"] == "apt campaign"
    assert packet["nearest_label_distance"] == pytest.approx(0.2)


def test_promote_cluster_state_transition():
    bank = LabelBank(min_freq=3, min_source_docs=2)
    cluster_id = "zero-day chain"
    bank.add_candidate(_phrase("zero-day chain", "d1"), cluster_id=cluster_id)
    bank.add_candidate(_phrase("zero-day chain", "d2"), cluster_id=cluster_id)
    bank.add_candidate(_phrase("zero-day chain", "d3"), cluster_id=cluster_id)

    assert cluster_id in bank.candidate_labels
    bank.promote_cluster(cluster_id, "new_label_001")

    assert cluster_id not in bank.proto_label_clusters
    assert cluster_id not in bank.candidate_labels
    assert "new_label_001" in bank.labels
    assert "new_label_001" in bank.promoted_labels


def test_fast_sync_shape_stable(label_bank: LabelBank):
    model_state = _base_model_state()
    validation = [
        ValidationSample(embedding=[1.0, 0.0, 0.0, 0.0], true_labels={"l1"}),
        ValidationSample(embedding=[0.0, 1.0, 0.0, 0.0], true_labels={"l2"}),
    ]

    updated = fast_sync(model_state, label_bank, validation_set=validation)

    assert len(updated["E"]) == len(model_state["E"])
    assert len(updated["P"]) == len(model_state["P"])
    assert len(updated["E"][0]) == len(model_state["E"][0])
    assert len(updated["P"][0]) == len(model_state["P"][0])


def test_fast_sync_inference_runs(label_bank: LabelBank):
    model_state = _base_model_state()
    updated = fast_sync(model_state, label_bank)

    ranked = infer_topk([0.9, 0.1, 0.0, 0.0], updated, top_k=2)
    predicted = infer_above_threshold([0.9, 0.1, 0.0, 0.0], updated)

    assert len(ranked) == 2
    assert ranked[0][0] in {"l1", "l2", "l3"}
    assert all(label_id in {"l1", "l2", "l3"} for label_id in predicted)


def test_fast_sync_updates_prototype_buffer(label_bank: LabelBank):
    model_state = _base_model_state()
    updated = fast_sync(model_state, label_bank)

    assert updated["P"] != model_state["P"]


def test_fast_sync_embedding_prototype_alignment(label_bank: LabelBank):
    model_state = _base_model_state()
    updated = fast_sync(model_state, label_bank)

    assert updated["sync_report"]["avg_embedding_prototype_alignment"] >= 0.85


def test_fast_sync_threshold_recalibration(label_bank: LabelBank):
    model_state = {
        "label_ids": ["l1", "l2"],
        "E": [[1.0, 0.0], [0.6, 0.8]],
        "P": [[1.0, 0.0], [0.6, 0.8]],
        "threshold": 0.50,
    }
    validation = [
        ValidationSample(embedding=[1.0, 0.0], true_labels={"l1"}),
        ValidationSample(embedding=[0.6, 0.8], true_labels={"l2"}),
    ]

    updated = fast_sync(model_state, label_bank, eta_e=0.0, eta_p=0.0, validation_set=validation)

    assert 0.10 <= updated["threshold"] <= 0.90
    assert updated["threshold"] != pytest.approx(0.50)
