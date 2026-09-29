"""Final-test release, data integrity and conditional uncertainty boundaries."""

from pathlib import Path

import numpy as np
import pytest

from calolab_reco.confirmation import final_evaluation as final


def raw_events():
    return dict(
        deposits=np.ones((4, 30, 85), dtype=np.float32),
        targets=np.array([[2, 3, 4]] * 4, dtype=np.float32),
        source_ids=np.column_stack((np.zeros(4, dtype=int), np.arange(4))),
    )


def test_test_release_is_checked_before_any_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Data access before explicit release")

    monkeypatch.setattr(final, "load_json", forbidden)
    with pytest.raises(PermissionError, match="allow-test"):
        final.evaluate(*(Path("does-not-exist") for _ in range(5)))


@pytest.mark.parametrize("corruption", ["duplicate_ids", "nonfinite", "negative", "zero_truth"])
def test_final_data_are_rejected_rather_than_filtered(corruption):
    raw = raw_events()
    if corruption == "duplicate_ids":
        raw["source_ids"][1] = raw["source_ids"][0]
    elif corruption == "nonfinite":
        raw["targets"][0, 1] = np.nan
    elif corruption == "negative":
        raw["deposits"][0, 0, 0] = -1
    else:
        raw["targets"][0, 0] = 0
    with pytest.raises(ValueError):
        final.check_raw(raw)


def test_unphysical_energy_and_extreme_position_are_retained():
    truth = raw_events()["targets"]
    prediction = truth.copy()
    prediction[0] = [-1, 33, 44]
    result = final.extra_metrics(truth, prediction, "joint")
    assert result["nonpositive_energy_count"] == 1
    assert result["position_gt_10"] == 1
    assert result["position_rmse"] == 25


@pytest.mark.parametrize("task", ["energy", "position"])
def test_paired_bootstrap_uses_same_events_and_positive_favors_finetuning(task):
    truth = raw_events()["targets"].astype(float)
    direct, fine = truth.copy(), truth.copy()
    if task == "energy":
        direct[:, 0] *= 1.2
    else:
        direct[:, 1] += 0.2
    settings = dict(seed=1, replicates=100, confidence=0.95)
    result = final.paired_interval(truth, direct, fine, task, settings)
    assert result["direct_minus_finetuned"] == pytest.approx(0.2)
    assert result["interval"] == pytest.approx([0.2, 0.2])
    zero = final.paired_interval(truth, fine, fine, task, settings)
    assert zero["interval"] == [0, 0]
