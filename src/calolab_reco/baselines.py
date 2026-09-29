"""Energy-sum and index-barycenter references with train-fitted calibration.

Deposits use the source array scale, explicitly converted from MeV to GeV by
division by 1000. Barycenters use integer array indices, not crystal centers.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

FEATURE_ORDER = ["deposit_sum_gev", "row_barycenter", "column_barycenter"]
TARGET_ORDER = ["energy_gev", "local_iphi", "local_ieta"]
INPUT_ENERGY_DIVISOR = 1000.0


def physical_features(deposits: ArrayLike) -> NDArray[np.float64]:
    """Return [sum in GeV, row barycenter, column barycenter] per event.

    Process bounded blocks so a memory-mapped dataset is not copied to float64
    in full. Zero total energy has no defined barycenter and is rejected.
    """
    values = np.asarray(deposits)
    if values.ndim != 3 or values.shape[1:] != (30, 85):
        raise ValueError("Deposits must have shape (events, 30, 85)")
    if values.dtype.kind not in "fiu":
        raise ValueError("Deposits must contain real numeric values")
    features = np.empty((len(values), 3), dtype=np.float64)
    rows = np.arange(30, dtype=np.float64)
    columns = np.arange(85, dtype=np.float64)
    for start in range(0, len(values), 512):
        block = values[start : start + 512]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError("Deposits must be finite and nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            row_sum = block.sum(axis=2, dtype=np.float64)
            column_sum = block.sum(axis=1, dtype=np.float64)
            total = row_sum.sum(axis=1, dtype=np.float64)
        if not np.isfinite(total).all() or (total <= 0).any():
            raise ValueError("Every event needs a finite, strictly positive deposit sum")
        stop = start + len(block)
        features[start:stop, 0] = total / INPUT_ENERGY_DIVISOR
        features[start:stop, 1] = (row_sum / total[:, None]) @ rows
        features[start:stop, 2] = (column_sum / total[:, None]) @ columns
    if not np.isfinite(features).all():
        raise ValueError("Physical features must remain finite")
    return features


def fit_baselines(deposits: ArrayLike, targets: ArrayLike) -> dict:
    """Fit three independent affine OLS maps using the supplied training data.

    The caller supplies train only. This function neither selects a partition nor
    reads validation or test data. A constant feature receives zero slope and the
    mean training target as intercept, a documented least-squares solution.
    """
    features = physical_features(deposits)
    target_values = np.asarray(targets)
    if target_values.shape != features.shape or target_values.dtype.kind not in "fiu":
        raise ValueError("Targets must be real values with shape (events, 3)")
    target_values = target_values.astype(np.float64, copy=False)
    if not len(features):
        raise ValueError("Calibration needs at least one training event")
    if not np.isfinite(target_values).all() or (target_values[:, 0] <= 0).any():
        raise ValueError("Targets must be finite with strictly positive incident energy")
    feature_mean = features.mean(axis=0)
    target_mean = target_values.mean(axis=0)
    centered_features = features - feature_mean
    denominator = np.square(centered_features).sum(axis=0)
    numerator = (centered_features * (target_values - target_mean)).sum(axis=0)
    slopes = np.divide(numerator, denominator, out=np.zeros(3), where=denominator > 0)
    intercepts = target_mean - slopes * feature_mean
    if not np.isfinite(slopes).all() or not np.isfinite(intercepts).all():
        raise ValueError("Fitted calibration coefficients must remain finite")
    return {
        "schema_version": 1,
        "feature_order": list(FEATURE_ORDER),
        "target_order": list(TARGET_ORDER),
        "input_energy_divisor": INPUT_ENERGY_DIVISOR,
        "fit_method": "independent univariate ordinary least squares on train",
        "slopes": slopes.tolist(),
        "intercepts": intercepts.tolist(),
        "constant_features": (denominator == 0).tolist(),
        "n_train": len(features),
        "coordinate_convention": "integer array indices, not physical crystal centers",
    }


def predict_baselines(deposits: ArrayLike, calibration: Mapping) -> dict[str, NDArray[np.float64]]:
    """Return raw features and calibrated predictions without fitting or clipping."""
    if calibration.get("schema_version") != 1:
        raise ValueError("Unsupported calibration schema")
    if (
        calibration.get("feature_order") != FEATURE_ORDER
        or calibration.get("target_order") != TARGET_ORDER
        or calibration.get("input_energy_divisor") != INPUT_ENERGY_DIVISOR
    ):
        raise ValueError("Calibration feature order or unit conversion does not match")
    slopes = np.asarray(calibration.get("slopes"), dtype=np.float64)
    intercepts = np.asarray(calibration.get("intercepts"), dtype=np.float64)
    if slopes.shape != (3,) or intercepts.shape != (3,):
        raise ValueError("Calibration needs exactly three slopes and three intercepts")
    if not np.isfinite(slopes).all() or not np.isfinite(intercepts).all():
        raise ValueError("Calibration coefficients must be finite")
    features = physical_features(deposits)
    with np.errstate(over="ignore", invalid="ignore"):
        calibrated = features * slopes + intercepts
    if not np.isfinite(calibrated).all():
        raise ValueError("Calibrated predictions must remain finite")
    return {"raw": features, "calibrated": calibrated}
