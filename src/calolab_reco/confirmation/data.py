"""Measured windows and train-only calibration, retained from the decision study."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from calolab_reco.metrics import regression_metrics, stratified_metrics

REGIMES = ("clean", "noise", "noise_cut")
TASKS = ("energy", "position", "joint")


def readout(deposits: np.ndarray, regime: str, seed: int, noise: dict) -> np.ndarray:
    """MeV stored deposits to GeV measurements; signed noise is never clipped.

    Identical seeds provide identical draws for noise and noise_cut. The seed is
    independent between train and validation. The cut is on measured CELL energy,
    never incident photon energy or the observed seed crystal.
    """
    if regime not in REGIMES:
        raise ValueError("Unknown readout regime")
    measured = np.asarray(deposits, dtype=np.float32).copy()
    if measured.ndim != 3 or measured.shape[1:] != (30, 85):
        raise ValueError("Expected (N,30,85) deposits")
    if not np.isfinite(measured).all() or (measured < 0).any():
        raise ValueError("Underlying simulated deposits must be nonnegative and finite")
    measured /= 1000.0
    if regime in {"noise", "noise_cut"}:
        rng = np.random.default_rng(seed)
        for start in range(0, len(measured), 256):
            block = measured[start : start + 256]
            sigma = np.sqrt(
                noise["electronic_gev"] ** 2
                + noise["stochastic"] ** 2 * block
                + (noise["constant"] * block) ** 2
            )
            block += (sigma * rng.standard_normal(block.shape)).astype(np.float32)
    if regime in {"cut", "noise_cut"}:
        measured[measured < noise["threshold_gev"]] = 0
    return measured


def observed_window(measured: np.ndarray) -> dict:
    """7x7 around measured maximum, with explicit edge validity and index origin."""
    values = np.asarray(measured, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (30, 85) or not np.isfinite(values).all():
        raise ValueError("Expected finite measured (N,30,85) array")
    peak = values.reshape(len(values), -1).argmax(axis=1)
    anchors = np.column_stack((peak // 85, peak % 85)).astype(np.float32)
    delta = np.arange(-3, 4)
    rows = anchors[:, 0].astype(int)[:, None, None] + delta[None, :, None]
    cols = anchors[:, 1].astype(int)[:, None, None] + delta[None, None, :]
    valid = (rows >= 0) & (rows < 30) & (cols >= 0) & (cols < 85)
    windows = values[np.arange(len(values))[:, None, None], rows.clip(0, 29), cols.clip(0, 84)]
    windows = np.where(valid, windows, 0).astype(np.float32)
    return {"values": windows, "valid": valid, "anchors": anchors}


def local_features(window: dict) -> np.ndarray:
    """Signed energy sum; positive-weight barycenter, falling back to measured anchor."""
    values, valid, anchors = window["values"], window["valid"], window["anchors"]
    signed_sum = np.where(valid, values, 0).sum(axis=(1, 2), dtype=np.float64)
    weights = np.maximum(values, 0) * valid
    total = weights.sum(axis=(1, 2), dtype=np.float64)
    delta = np.arange(-3, 4, dtype=np.float64)
    row = (weights * delta[None, :, None]).sum(axis=(1, 2), dtype=np.float64)
    col = (weights * delta[None, None, :]).sum(axis=(1, 2), dtype=np.float64)
    offset = np.column_stack((row, col)) / np.where(total > 0, total, 1)[:, None]
    return np.column_stack((signed_sum, anchors + offset))


def fit_affine(features: np.ndarray, targets: np.ndarray) -> dict:
    """Fit each measured coordinate to its corresponding label on train only."""
    slopes, intercepts = [], []
    for column in range(3):
        design = np.column_stack((features[:, column], np.ones(len(features))))
        slope, intercept = np.linalg.lstsq(design, targets[:, column], rcond=None)[0]
        slopes.append(float(slope))
        intercepts.append(float(intercept))
    return {"slopes": slopes, "intercepts": intercepts}


def affine_predict(features: np.ndarray, calibration: dict) -> np.ndarray:
    return features * np.asarray(calibration["slopes"]) + np.asarray(calibration["intercepts"])


def periodic_design(coordinate):
    phase = np.remainder(coordinate, 1.0)
    columns = [np.ones(len(phase)), coordinate]
    for harmonic in range(1, 4):
        angle = 2 * np.pi * harmonic * phase
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def fit_periodic(features, targets, affine):
    coefficients, ranks = [], []
    for column in (1, 2):
        coefficient, _, rank, _ = np.linalg.lstsq(
            periodic_design(features[:, column]), targets[:, column], rcond=None
        )
        coefficients.append(coefficient.tolist())
        ranks.append(int(rank))
    return {
        "affine": affine,
        "position_coefficients": coefficients,
        "position_design_ranks": ranks,
        "harmonics": 3,
        "fit_method": "train-only affine plus three fixed harmonics of measured barycenter",
        "rank_deficient": any(rank < 8 for rank in ranks),
    }


def periodic_predict(features, calibration):
    predictions = affine_predict(features, calibration["affine"])
    for column in (1, 2):
        predictions[:, column] = periodic_design(features[:, column]) @ np.asarray(
            calibration["position_coefficients"][column - 1]
        )
    return predictions


def fit_statistics(deposits, targets, *, anchors=None, linear_quantile=0.95) -> dict:
    """Every fitted statistic uses the actual selected training sample only."""
    positive = deposits[deposits > 0]
    if not len(positive):
        raise ValueError("Training deposits need positive signal")
    shifted = np.asarray(targets, dtype=np.float64).copy()
    if anchors is not None:
        shifted[:, 1:] -= anchors
    return {
        "log_scale_mev": float(np.median(positive)),
        "linear_scale_mev": max(
            float(np.quantile(deposits.max(axis=(1, 2)), linear_quantile)),
            np.finfo(np.float32).tiny,
        ),
        "target_mean": shifted.mean(axis=0).tolist(),
        "target_std": np.maximum(shifted.std(axis=0), 1e-6).tolist(),
        "position_output": "offset_from_observed_anchor" if anchors is not None else "direct",
    }


def task_loss(predictions, targets, task="joint"):
    """Relative Huber with fixed 10% energy scale and one index unit for position."""
    if task not in TASKS or predictions.shape != targets.shape or predictions.shape[1] != 3:
        raise ValueError("Expected matching (N,3) arrays and a known task")
    if not torch.isfinite(targets).all() or (targets[:, 0] <= 0).any():
        raise ValueError("Targets must be finite, with positive energy")
    energy_error = (predictions[:, 0] - targets[:, 0]) / (0.1 * targets[:, 0])
    position_error = predictions[:, 1:] - targets[:, 1:]
    energy = F.huber_loss(energy_error, torch.zeros_like(energy_error))
    position = F.huber_loss(position_error, torch.zeros_like(position_error))
    if task == "energy":
        return energy
    if task == "position":
        return position
    return 0.5 * (energy + position)


def aggregate_metrics(targets, predictions, task="joint") -> dict:
    values = regression_metrics(targets, predictions)
    return {
        key: value
        for key, value in values.items()
        if key == "count" or task == "joint" or key.startswith(task + "_")
    }


def task_subgroups(targets, predictions, task="joint") -> dict:
    def filter_metrics(value):
        if isinstance(value, list):
            return [filter_metrics(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "energy_mare" in value:
            return {
                key: item
                for key, item in value.items()
                if key == "count" or task == "joint" or key.startswith(task + "_")
            }
        return {key: filter_metrics(item) for key, item in value.items()}

    return filter_metrics(stratified_metrics(targets, predictions))


def selection_scores(metrics: dict, task: str) -> dict:
    result = {}
    if task in {"joint", "energy"}:
        result["energy"] = metrics["energy_mare"]
    if task in {"joint", "position"}:
        result["position"] = metrics["position_distance_median"]
    if task == "joint":
        result["joint"] = metrics["energy_mare"] / 0.1 + metrics["position_distance_median"]
    return result


def _tensor_split(inputs, raw, anchors=None):
    count = len(raw["targets"])
    return {
        "inputs": torch.from_numpy(np.asarray(inputs, dtype=np.float32)),
        "targets": torch.from_numpy(raw["targets"]),
        "raw_targets": raw["targets"],
        "source_ids": raw["source_ids"],
        "anchors": torch.from_numpy(
            anchors if anchors is not None else np.zeros((count, 2), dtype=np.float32)
        ),
    }


def shower_features(window) -> np.ndarray:
    """Seven observed summaries; no incident truth or calibrated prediction."""
    values = np.where(window["valid"], window["values"], 0).astype(np.float64)
    positive = np.maximum(values, 0)
    positive_sum = positive.sum(axis=(1, 2))
    denominator = np.maximum(positive_sum, 1e-12)
    delta = np.arange(-3, 4, dtype=np.float64)
    return np.column_stack(
        (
            values.sum(axis=(1, 2)),
            positive_sum,
            values[:, 3, 3],
            values[:, 2:5, 2:5].sum(axis=(1, 2)),
            (positive * delta[None, :, None] ** 2).sum(axis=(1, 2)) / denominator,
            (positive * delta[None, None, :] ** 2).sum(axis=(1, 2)) / denominator,
            window["valid"].mean(axis=(1, 2)),
        )
    )


def quadratic_design(features, mean, scale):
    normalized = np.clip((features - mean) / scale, -100, 100)
    columns = [np.ones(len(features)), *normalized.T]
    columns.extend(
        normalized[:, a] * normalized[:, b]
        for a in range(normalized.shape[1])
        for b in range(a, normalized.shape[1])
    )
    return np.column_stack(columns)


def fit_quadratic(features, energy):
    """Frozen degree-two relative least squares, SVD, no ridge or validation tuning."""
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    design = quadratic_design(features, mean, scale)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design / energy[:, None], np.ones(len(energy)), rcond=None
    )
    return dict(
        mean=mean.tolist(),
        scale=scale.tolist(),
        coefficients=coefficients.tolist(),
        rank=int(rank),
        solver="SVD relative least squares",
        ridge_penalty=0,
    )


def quadratic_predict(features, calibration):
    return (
        quadratic_design(
            features, np.asarray(calibration["mean"]), np.asarray(calibration["scale"])
        )
        @ calibration["coefficients"]
    )


def prepare(raw_train, raw_validation, regime, config):
    windows = {
        name: observed_window(
            readout(raw["deposits"], regime, config[f"{name}_noise_seed"], config["noise"])
        )
        for name, raw in [("train", raw_train), ("validation", raw_validation)]
    }
    stats = fit_statistics(
        raw_train["deposits"], raw_train["targets"], anchors=windows["train"]["anchors"]
    )
    splits = {}
    for name, raw in [("train", raw_train), ("validation", raw_validation)]:
        w = windows[name]
        x = np.stack(
            (w["values"] / (stats["linear_scale_mev"] / 1000), w["valid"].astype(np.float32)),
            axis=1,
        )
        splits[name] = _tensor_split(x, raw, w["anchors"])
    features = local_features(windows["train"])
    affine = fit_affine(features, raw_train["targets"])
    references = dict(
        affine=affine,
        periodic=fit_periodic(features, raw_train["targets"], affine),
        quadratic=fit_quadratic(shower_features(windows["train"]), raw_train["targets"][:, 0]),
    )
    return splits["train"], splits["validation"], stats, references


def evaluate_inputs(raw, regime, seed, config, stats):
    window = observed_window(readout(raw["deposits"], regime, seed, config["noise"]))
    x = np.stack(
        (window["values"] / (stats["linear_scale_mev"] / 1000), window["valid"].astype(np.float32)),
        axis=1,
    )
    return _tensor_split(x, raw, window["anchors"]), window


def reference_predictions(window, references):
    features = local_features(window)
    result = dict(
        affine=affine_predict(features, references["affine"]),
        periodic=periodic_predict(features, references["periodic"]),
    )
    quadratic = np.zeros((len(features), 3), dtype=np.float64)
    quadratic[:, 0] = quadratic_predict(shower_features(window), references["quadratic"])
    result["quadratic"] = quadratic
    return result
