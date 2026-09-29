"""Hand-computed reference checks on synthetic events only."""

import copy
import json

import numpy as np
import pytest

from calolab_reco.baselines import fit_baselines, physical_features, predict_baselines


def _single_cell_events(energy, rows, columns):
    deposits = np.zeros((len(energy), 30, 85), dtype=np.float32)
    for index, (value, row, column) in enumerate(zip(energy, rows, columns, strict=True)):
        deposits[index, row, column] = value * 1000
    return deposits


def test_conversion_and_weighted_indices_are_separate_operations():
    deposits = np.zeros((1, 30, 85), dtype=np.float32)
    deposits[0, 0, 0] = 250
    deposits[0, 2, 4] = 750
    np.testing.assert_allclose(physical_features(deposits), [[1.0, 1.5, 3.0]])
    np.testing.assert_allclose(physical_features(deposits * 2), [[2.0, 1.5, 3.0]])


def test_affine_fit_recovers_three_independent_maps_and_serializes():
    deposits = _single_cell_events([1, 2, 4, 8], [1, 2, 4, 8], [5, 8, 13, 20])
    slopes = np.array([1.2, 0.7, 1.1])
    intercepts = np.array([0.3, 0.2, -1.0])
    targets = physical_features(deposits) * slopes + intercepts
    calibration = fit_baselines(deposits, targets)
    np.testing.assert_allclose(calibration["slopes"], slopes)
    np.testing.assert_allclose(calibration["intercepts"], intercepts)
    restored = json.loads(json.dumps(calibration, allow_nan=False))
    np.testing.assert_allclose(predict_baselines(deposits, restored)["calibrated"], targets)
    assert calibration["n_train"] == 4
    assert calibration["input_energy_divisor"] == 1000


def test_prediction_does_not_refit_or_mutate_calibration_or_inputs():
    train = _single_cell_events([1, 2, 4], [1, 2, 4], [5, 8, 13])
    targets = physical_features(train) * [1.2, 0.7, 1.1] + [0.3, 0.2, -1]
    calibration = fit_baselines(train, targets)
    saved = copy.deepcopy(calibration)
    held_out = _single_cell_events([70], [28], [80])
    original = held_out.copy()
    prediction = predict_baselines(held_out, calibration)
    np.testing.assert_allclose(prediction["raw"], [[70, 28, 80]])
    np.testing.assert_allclose(prediction["calibrated"], [[84.3, 19.8, 87]])
    np.testing.assert_array_equal(held_out, original)
    assert calibration == saved


def test_constant_features_use_the_training_target_mean():
    deposits = _single_cell_events([1, 1, 1], [2, 2, 2], [3, 3, 3])
    calibration = fit_baselines(deposits, [[1, 3, 6], [2, 4, 7], [3, 5, 8]])
    assert calibration["slopes"] == [0, 0, 0]
    assert calibration["intercepts"] == [2, 4, 7]
    assert calibration["constant_features"] == [True, True, True]


def test_predictions_are_not_clipped_to_positive_energy_or_coordinate_bounds():
    deposits = _single_cell_events([1, 2, 3], [1, 2, 3], [5, 8, 13])
    targets = physical_features(deposits)
    targets[:, 0] = 4 - targets[:, 0]
    calibration = fit_baselines(deposits, targets)
    held_out = _single_cell_events([10], [4], [20])
    assert predict_baselines(held_out, calibration)["calibrated"][0, 0] == pytest.approx(-6)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0])
def test_invalid_deposits_are_rejected_even_with_a_positive_sum(invalid):
    deposits = _single_cell_events([1], [1], [1])
    deposits[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite and nonnegative"):
        physical_features(deposits)


def test_zero_events_and_wrong_shapes_are_rejected():
    with pytest.raises(ValueError, match="strictly positive deposit sum"):
        physical_features(np.zeros((1, 30, 85)))
    with pytest.raises(ValueError, match="shape"):
        physical_features(np.zeros((1, 1, 30, 85)))


def test_prediction_rejects_changed_conversion_and_invalid_coefficients():
    deposits = _single_cell_events([1, 2], [1, 2], [5, 8])
    calibration = fit_baselines(deposits, physical_features(deposits))
    invalid = copy.deepcopy(calibration)
    invalid["input_energy_divisor"] = 978.65
    with pytest.raises(ValueError, match="unit conversion"):
        predict_baselines(deposits, invalid)
    invalid = copy.deepcopy(calibration)
    invalid["slopes"][0] = np.nan
    with pytest.raises(ValueError, match="coefficients must be finite"):
        predict_baselines(deposits, invalid)
