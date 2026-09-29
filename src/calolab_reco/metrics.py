"""Regression metrics in GeV and stored local iphi/ieta coordinate units.

Residuals are prediction minus target. Relative energy metrics are fractions,
not percentages. Stored-index strata do not describe physical detector edges.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

ENERGY_BIN_EDGES = (1.0, 10.0, 30.0, 60.0, 100.0001)
POSITION_BOUNDS = ((0.0, 30.0), (0.0, 85.0))
POSITION_MARGIN = 5.0


def _validated_arrays(
    targets: ArrayLike, predictions: ArrayLike, *, allow_empty: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    target_values = np.asarray(targets)
    predicted_values = np.asarray(predictions)
    if (
        target_values.ndim != 2
        or target_values.shape[1] != 3
        or predicted_values.shape != target_values.shape
    ):
        raise ValueError("Targets and predictions must share shape (events, 3)")
    if target_values.dtype.kind not in "fiu" or predicted_values.dtype.kind not in "fiu":
        raise ValueError("Targets and predictions must contain real numeric values")
    target_values = target_values.astype(np.float64, copy=False)
    predicted_values = predicted_values.astype(np.float64, copy=False)
    if not allow_empty and not len(target_values):
        raise ValueError("Metrics need at least one event")
    if not np.isfinite(target_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Targets and predictions must be finite")
    if (target_values[:, 0] <= 0).any():
        raise ValueError("Incident-energy targets must be strictly positive")
    return target_values, predicted_values


def regression_metrics(targets: ArrayLike, predictions: ArrayLike) -> dict:
    """Summarize errors on supplied events without selecting or fitting a model.

    Energy robust resolution is half the q84-q16 interval of relative residuals.
    Quantiles use linear interpolation. Position distances use the two stored
    coordinate units as supplied; they are not distances in millimeters.
    """
    target_values, predicted_values = _validated_arrays(targets, predictions)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = predicted_values - target_values
        relative_energy = residual[:, 0] / target_values[:, 0]
        distance = np.hypot(residual[:, 1], residual[:, 2])
    if (
        not np.isfinite(residual).all()
        or not np.isfinite(relative_energy).all()
        or not np.isfinite(distance).all()
    ):
        raise ValueError("Residuals and distances must remain finite")
    q16, q84 = np.quantile(relative_energy, [0.16, 0.84], method="linear")
    result = {
        "count": len(target_values),
        "energy_relative_bias": float(relative_energy.mean()),
        "energy_relative_resolution": float((q84 - q16) / 2),
        "energy_mare": float(np.abs(relative_energy).mean()),
        "energy_mae_gev": float(np.abs(residual[:, 0]).mean()),
        "position_bias_iphi": float(residual[:, 1].mean()),
        "position_bias_ieta": float(residual[:, 2].mean()),
        "position_distance_median": float(np.median(distance)),
        "position_distance_p68": float(np.quantile(distance, 0.68, method="linear")),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Aggregated metrics must remain finite")
    return result


def stratified_metrics(targets: ArrayLike, predictions: ArrayLike) -> dict:
    """Use fixed target-energy bins and stored-index margins without fitting bins.

    Energy bins are left-closed and right-open. The upper value 100.0001 includes
    100 GeV. Position membership uses target iphi in [0, 30] and ieta in [0, 85].
    Near-boundary means the minimum stored-index margin is strictly below five;
    a margin of exactly five belongs to the interior. Out-of-range targets are
    reported separately, never silently dropped. Empty strata have metrics=None.
    """
    target_values, predicted_values = _validated_arrays(targets, predictions, allow_empty=True)

    def summarize(selection: NDArray[np.bool_]) -> dict:
        count = int(selection.sum())
        return {
            "count": count,
            "metrics": (
                regression_metrics(target_values[selection], predicted_values[selection])
                if count
                else None
            ),
        }

    energy = target_values[:, 0]
    bins = []
    for lower, upper in zip(ENERGY_BIN_EDGES[:-1], ENERGY_BIN_EDGES[1:], strict=True):
        bins.append(
            {
                "lower_gev": lower,
                "upper_gev": upper,
                "interval": "[lower, upper)",
                **summarize((energy >= lower) & (energy < upper)),
            }
        )
    phi = target_values[:, 1]
    eta = target_values[:, 2]
    inside = (phi >= 0) & (phi <= 30) & (eta >= 0) & (eta <= 85)
    margin = np.minimum.reduce((phi, 30 - phi, eta, 85 - eta))
    return {
        "count": len(target_values),
        "energy_bins": bins,
        "energy_outside_bins": summarize(
            (energy < ENERGY_BIN_EDGES[0]) | (energy >= ENERGY_BIN_EDGES[-1])
        ),
        "position_margin": {
            "definition": "min(iphi, 30-iphi, ieta, 85-ieta) in stored index units",
            "interpretation": "Stored target-index ranges, not physical detector edges",
            "bounds": {"iphi": list(POSITION_BOUNDS[0]), "ieta": list(POSITION_BOUNDS[1])},
            "threshold": POSITION_MARGIN,
            "near_index_boundary": summarize(inside & (margin < POSITION_MARGIN)),
            "interior": summarize(inside & (margin >= POSITION_MARGIN)),
            "outside_index_domain": summarize(~inside),
        },
    }
