from __future__ import annotations

import os

import pytest

from owlu.ood_eval import evaluate_ltce_fast_sync_ood_delta


@pytest.mark.integration
def test_fast_sync_ood_metrics_delta_report():
    if os.getenv("OWLU_RUN_LTCE_OOD_COMPARE", "0") != "1":
        pytest.skip("Set OWLU_RUN_LTCE_OOD_COMPARE=1 to run OOD fast_sync comparison.")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    report = evaluate_ltce_fast_sync_ood_delta()

    assert report["num_examples"] >= 6
    assert 0.10 <= report["threshold_after"] <= 0.90
    assert "micro_f1" in report["before"]
    assert "macro_f1" in report["before"]
    assert "micro_f1" in report["after"]
    assert "macro_f1" in report["after"]

    # Require that fast_sync produces at least one measurable metric shift.
    delta = report["delta"]
    assert abs(float(delta["micro_f1"])) > 1e-12 or abs(float(delta["macro_f1"])) > 1e-12
