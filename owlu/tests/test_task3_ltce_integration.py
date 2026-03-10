from __future__ import annotations

import os

import pytest

from owlu.task3_eval import evaluate_ltce_task3_pipeline


@pytest.mark.integration
def test_task3_ltce_bridge_smoke():
    if os.getenv("OWLU_RUN_TASK3_LTCE", "0") != "1":
        pytest.skip("Set OWLU_RUN_TASK3_LTCE=1 to run LTCE-backed Task 3 smoke test.")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    report = evaluate_ltce_task3_pipeline()
    bridge = report["ltce_bridge_check"]
    assert bridge["num_labels_after"] >= bridge["num_labels_before"]
    assert bridge["logits_shape"][0] == 1
    assert bridge["logits_shape"][1] == bridge["num_labels_after"]
