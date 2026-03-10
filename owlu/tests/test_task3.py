from __future__ import annotations

from datetime import datetime, timezone

import owlu.ltc_sync as ltc_sync
from owlu.label_bank import LabelBank
from owlu.llm_phrase_generator import CandidatePhrase
from owlu.ltc_sync import (
    TrainingSample,
    ValidationSample,
    build_replay_samples,
    compose_incremental_training_set,
    recalibrate_threshold_with_constraints,
    slow_sync,
)
from owlu.task3_eval import run_ablation_suite, run_e2e_pipeline
from owlu.task3_sqlite import Task3SQLiteStore


def _base_model_state() -> dict[str, object]:
    return {
        "label_ids": ["l1", "l2"],
        "E": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        "P": [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
        ],
        "threshold": 0.5,
        "classifier_weight": [
            [0.6, 0.1, 0.0, 0.0],
            [0.1, 0.6, 0.0, 0.0],
        ],
        "classifier_bias": [0.0, 0.0],
    }


def _build_bank_with_promoted() -> LabelBank:
    bank = LabelBank(min_freq=1, min_source_docs=1)
    bank.register_label("l1", "malware analysis", aliases=["malware detection"])
    bank.register_label("l2", "network defense", aliases=["blue team"])

    promoted_phrase = CandidatePhrase(
        text="edge llm safety audit",
        raw_text="edge llm safety audit",
        source_doc_id="doc-promoted",
        timestamp=datetime.now(timezone.utc),
        agreement=0.9,
    )
    bank.add_candidate(promoted_phrase, cluster_id="edge llm safety audit")
    bank.promote_cluster("edge llm safety audit", "l3")

    # Keep one candidate cluster for ablation checks.
    candidate_phrase = CandidatePhrase(
        text="supply chain risk",
        raw_text="supply chain risk",
        source_doc_id="doc-candidate",
        timestamp=datetime.now(timezone.utc),
        agreement=0.8,
    )
    bank.add_candidate(candidate_phrase, cluster_id="supply chain risk")
    return bank


def _validation_set() -> list[ValidationSample]:
    return [
        ValidationSample(embedding=[1.0, 0.0, 0.0, 0.0], true_labels={"l1"}),
        ValidationSample(embedding=[0.0, 1.0, 0.0, 0.0], true_labels={"l2"}),
        ValidationSample(embedding=[0.0, 0.0, 1.0, 0.0], true_labels={"l3"}),
        ValidationSample(embedding=[0.8, 0.2, 0.0, 0.0], true_labels={"l1"}),
        ValidationSample(embedding=[0.2, 0.8, 0.0, 0.0], true_labels={"l2"}),
    ]


def _old_training_samples() -> list[TrainingSample]:
    rows: list[TrainingSample] = []
    for idx in range(80):
        if idx % 2 == 0:
            rows.append(TrainingSample(embedding=[1.0, 0.0, 0.0, 0.0], true_labels={"l1"}))
        else:
            rows.append(TrainingSample(embedding=[0.0, 1.0, 0.0, 0.0], true_labels={"l2"}))
    return rows


def _new_label_samples() -> dict[str, list[TrainingSample]]:
    return {
        "l3": [
            TrainingSample(embedding=[0.0, 0.0, 1.0, 0.0], true_labels={"l3"}),
            TrainingSample(embedding=[0.1, 0.0, 0.9, 0.0], true_labels={"l3"}),
            TrainingSample(embedding=[0.0, 0.1, 0.9, 0.0], true_labels={"l3"}),
            TrainingSample(embedding=[0.0, 0.0, 0.8, 0.2], true_labels={"l3"}),
            TrainingSample(embedding=[0.0, 0.0, 0.7, 0.3], true_labels={"l3"}),
            TrainingSample(embedding=[0.0, 0.0, 0.95, 0.05], true_labels={"l3"}),
        ]
    }


def test_slow_sync_label_expand():
    synced = slow_sync(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
        validation_set=_validation_set(),
    )

    assert "l3" in synced["label_ids"]
    assert len(synced["label_ids"]) == 3
    assert len(synced["E"]) == 3
    assert len(synced["P"]) == 3
    assert synced["sync_report"]["num_new_labels"] == 1


def test_slow_sync_model_rebuild():
    synced = slow_sync(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
        validation_set=_validation_set(),
    )

    assert len(synced["classifier_weight"]) == 3
    assert len(synced["classifier_bias"]) == 3
    assert int(synced["num_labels"]) == 3


def test_replay_sampler_ratio_and_cap():
    replay, replay_stats = build_replay_samples(
        _old_training_samples(),
        label_ids=["l1", "l2"],
        max_pos_per_label=30,
        max_neg_per_label=30,
        seed=42,
    )
    _, mix_stats = compose_incremental_training_set(
        replay_samples=replay,
        new_samples=_new_label_samples()["l3"],
        old_new_ratio=(2, 1),
        seed=42,
    )

    assert replay_stats["l1"]["pos_selected"] <= 30
    assert replay_stats["l1"]["neg_selected"] <= 30
    assert replay_stats["l2"]["pos_selected"] <= 30
    assert replay_stats["l2"]["neg_selected"] <= 30
    assert mix_stats["old_selected"] <= (mix_stats["new_selected"] * 2)


def test_threshold_grid_search_constraints():
    validation = [
        ValidationSample(embedding=[1.0, 0.0], true_labels={"l1"}),
        ValidationSample(embedding=[0.0, 1.0], true_labels={"l2"}),
    ]
    label_ids = ["l1", "l2"]
    prototypes = [[1.0, 0.0], [0.0, 1.0]]
    current = 0.5
    baseline = ltc_sync._precision_at_3(validation, label_ids, prototypes, current)

    threshold = recalibrate_threshold_with_constraints(
        prototypes,
        label_ids,
        validation,
        current_threshold=current,
        baseline_p_at_3=baseline,
        max_p_at_3_drop=0.0,
    )

    assert 0.10 <= threshold <= 0.90
    grid = [round(0.10 + (i * 0.02), 2) for i in range(41)]
    assert threshold in grid
    assert baseline - ltc_sync._precision_at_3(validation, label_ids, prototypes, threshold) <= 1e-12


def test_slow_sync_backward_compat():
    bank = LabelBank(min_freq=3, min_source_docs=2)
    bank.register_label("l1", "malware analysis")
    bank.register_label("l2", "network defense")

    synced = slow_sync(
        model_state=_base_model_state(),
        label_bank=bank,
        old_training_samples=_old_training_samples(),
        validation_set=_validation_set(),
    )

    assert synced["label_ids"] == ["l1", "l2"]
    assert synced["sync_report"]["num_new_labels"] == 0
    assert len(synced["classifier_weight"]) == 2
    assert len(synced["classifier_bias"]) == 2


def test_e2e_pipeline_smoke():
    report = run_e2e_pipeline(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        validation_set=_validation_set(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
        run_id="task3_smoke_run",
    )
    assert report["scenario"] == "baseline"
    assert "sync_report" in report
    assert report["sync_report"]["num_total_labels"] >= 3


def test_metrics_report_fields():
    report = run_e2e_pipeline(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        validation_set=_validation_set(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
    )
    required = {
        "coverage_rate",
        "micro_f1",
        "macro_f1",
        "long_tail_macro_f1",
        "candidate_pass_rate",
        "token_cost",
        "latency_ms",
        "p_at_3",
    }
    assert required.issubset(report.keys())


def test_ablation_runs():
    suite = run_ablation_suite(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        validation_set=_validation_set(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
        run_id="task3_ablation_run",
    )
    assert "baseline" in suite
    assert "ablations" in suite
    assert set(suite["ablations"].keys()) == {
        "no_agreement_gate",
        "no_uncertainty_revisit",
        "no_candidate_promotion",
    }
    assert suite["baseline"]["sync_report"]["num_new_labels"] >= 1
    assert suite["ablations"]["no_candidate_promotion"]["sync_report"]["num_new_labels"] == 0


def test_sqlite_roundtrip(tmp_path):
    db_path = tmp_path / "task3_roundtrip.db"
    store = Task3SQLiteStore(db_path)
    store.init_schema()

    report = run_e2e_pipeline(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        validation_set=_validation_set(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
        run_id="task3_sqlite_roundtrip",
        sqlite_store=store,
    )

    sync_run_id = report["sync_run_id"]
    slow_sync_payload = store.fetch_slow_sync_run(sync_run_id)
    e2e_payload = store.fetch_e2e_reports("task3_sqlite_roundtrip")
    promoted = store.fetch_promoted_labels(sync_run_id)
    labels = store.fetch_label_snapshot(sync_run_id, "after")

    assert slow_sync_payload is not None
    assert "baseline" in e2e_payload
    assert promoted
    assert labels


def test_sqlite_run_traceability(tmp_path):
    db_path = tmp_path / "task3_trace.db"
    store = Task3SQLiteStore(db_path)
    store.init_schema()

    run_e2e_pipeline(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        validation_set=_validation_set(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
        run_id="trace_run_a",
        sqlite_store=store,
    )
    run_e2e_pipeline(
        model_state=_base_model_state(),
        label_bank=_build_bank_with_promoted(),
        validation_set=_validation_set(),
        old_training_samples=_old_training_samples(),
        new_label_samples=_new_label_samples(),
        run_id="trace_run_b",
        sqlite_store=store,
    )

    run_ids = set(store.list_run_ids())
    assert "trace_run_a" in run_ids
    assert "trace_run_b" in run_ids
    assert store.fetch_slow_sync_run("trace_run_a:baseline") is not None
    assert store.fetch_slow_sync_run("trace_run_b:baseline") is not None
