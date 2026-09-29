"""Metric arithmetic and fixed-stratum boundaries on synthetic predictions."""

import json

import numpy as np
import pytest

from calolab_reco.metrics import regression_metrics, stratified_metrics


def test_regression_metrics_match_hand_computed_residuals():
    targets = np.array([[10, 10, 20], [20, 10, 20], [40, 10, 20]], dtype=float)
    predictions = np.array([[11, 13, 24], [18, 10, 20], [48, 4, 12]], dtype=float)
    result = regression_metrics(targets, predictions)
    assert result["count"] == 3
    assert result["energy_relative_bias"] == pytest.approx(0.2 / 3)
    assert result["energy_relative_resolution"] == pytest.approx(0.102)
    assert result["energy_mare"] == pytest.approx(0.4 / 3)
    assert result["energy_mae_gev"] == pytest.approx(11 / 3)
    assert result["position_bias_iphi"] == pytest.approx(-1)
    assert result["position_bias_ieta"] == pytest.approx(-4 / 3)
    assert result["position_distance_median"] == pytest.approx(5)
    assert result["position_distance_p68"] == pytest.approx(6.8)
    json.dumps(result, allow_nan=False)


def test_metrics_keep_negative_predictions_without_clipping():
    result = regression_metrics([[10, 10, 20]], [[-5, -1, -2]])
    assert result["energy_relative_bias"] == -1.5
    assert result["energy_mare"] == 1.5
    assert result["position_bias_iphi"] == -11
    assert result["position_bias_ieta"] == -22


def test_strata_boundaries_are_fixed_and_outside_targets_are_counted():
    energies = [1, 9.99, 10, 29.99, 30, 59.99, 60, 100, 100.0001, 0.5]
    positions = [[0, 42], [5, 5], [25, 80], [30, 85], [15, 40], [-1, 40], [15, 86]]
    positions += [[15, 40]] * 3
    targets = np.column_stack((energies, positions))
    report = stratified_metrics(targets, targets)
    assert [item["count"] for item in report["energy_bins"]] == [2, 2, 2, 2]
    assert report["energy_outside_bins"]["count"] == 2
    position = report["position_margin"]
    assert position["near_index_boundary"]["count"] == 2
    assert position["interior"]["count"] == 6
    assert position["outside_index_domain"]["count"] == 2
    assert position["interior"]["metrics"]["position_distance_median"] == 0
    json.dumps(report, allow_nan=False)


def test_empty_strata_report_none_instead_of_nonfinite_statistics():
    report = stratified_metrics([[2, 15, 40]], [[2, 15, 40]])
    for item in report["energy_bins"][1:]:
        assert item["count"] == 0
        assert item["metrics"] is None
    assert report["position_margin"]["near_index_boundary"] == {"count": 0, "metrics": None}
    empty = stratified_metrics(np.empty((0, 3)), np.empty((0, 3)))
    assert empty["count"] == 0
    assert all(item["metrics"] is None for item in empty["energy_bins"])
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_targets_or_predictions_are_rejected(invalid):
    good = np.array([[10, 10, 20]], dtype=float)
    bad = good.copy()
    bad[0, 2] = invalid
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(good, bad)
    with pytest.raises(ValueError, match="finite"):
        stratified_metrics(bad, good)


@pytest.mark.parametrize("energy", [0, -1])
def test_nonpositive_target_energy_is_rejected(energy):
    with pytest.raises(ValueError, match="strictly positive"):
        regression_metrics([[energy, 10, 20]], [[2, 10, 20]])


def test_metric_shape_mismatch_and_empty_overall_input_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        regression_metrics([[1, 2, 3]], [[1, 2]])
    with pytest.raises(ValueError, match="at least one"):
        regression_metrics(np.empty((0, 3)), np.empty((0, 3)))
