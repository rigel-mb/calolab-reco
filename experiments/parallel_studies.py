"""Isolated, fixed-budget CNN and Transformer studies on train/validation.

The published comparison and reserved test are untouched. Readout studies use
signed Gaussian measurements before an optional 0.05 GeV cell threshold. The
noise model is a synthetic measurement assumption, not detector digitization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from calolab_reco.metrics import regression_metrics, stratified_metrics
from calolab_reco.pilot import configure, restore_rng, rng_state, sha256, synchronize, write_json
from calolab_reco.pilot_models import CNN, Transformer, _transformer_stack
from calolab_reco.training import canonical_hash, external_directory, load_data, save_checkpoint

PROJECT = Path(__file__).resolve().parents[1]
REGIMES = ("clean", "cut", "noise", "noise_cut")
TASKS = ("joint", "energy", "position")
CASE_NAMES = (
    "input_log_joint",
    "input_linear_joint",
    "task_energy",
    "task_position",
    "volume_7000",
    "volume_14000",
    "readout_clean",
    "readout_cut",
    "readout_noise",
    "readout_noise_cut",
)
DEFAULT_CONFIG = {
    "schema_version": 1,
    "study": "parallel_studies",
    "seed": 20260925,
    "sampling_seed": 20260926,
    "train_noise_seed": 20260928,
    "validation_noise_seed": 20260929,
    "updates": 4400,
    "evaluate_every": 220,
    "checkpoint_every": 220,
    "log_every": 50,
    "effective_batch_size": 128,
    "microbatch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "training_seconds_ceiling": 2700,
    "volume_counts": [7000, 14000],
    "noise": {
        "stochastic": 0.03,
        "constant": 0.0035,
        "electronic_gev": 0.167,
        "threshold_gev": 0.05,
    },
    "linear_scale_quantile": 0.95,
}
TRANSFORMER_CONFIG = {
    **DEFAULT_CONFIG,
    "study": "parallel_studies_transformer",
    "training_seconds_ceiling": 5400,
}


def validate_config(config: dict) -> None:
    expected_config = (
        TRANSFORMER_CONFIG
        if config.get("study") == "parallel_studies_transformer"
        else DEFAULT_CONFIG
    )
    if set(config) != set(expected_config):
        raise ValueError("Unexpected or missing study configuration keys")
    for key, expected in expected_config.items():
        if key == "training_seconds_ceiling":
            if (
                not isinstance(config[key], (int, float))
                or not 0 < config[key] <= expected
            ):
                raise ValueError(
                    f"Cumulative training/validation ceiling must be in (0,{expected}]"
                )
        elif config[key] != expected or type(config[key]) is not type(expected):
            raise ValueError(f"The bounded, predefined protocol requires {key}={expected!r}")


def array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def ids_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<i8").tobytes()).hexdigest()


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


def transform_input(deposits: np.ndarray, kind: str, statistics: dict) -> np.ndarray:
    if kind == "log":
        return np.log1p(deposits / statistics["log_scale_mev"]).astype(np.float32)
    if kind == "linear":
        return (deposits / statistics["linear_scale_mev"]).astype(np.float32)
    raise ValueError("Unknown input transformation")


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


class LocalCNN(nn.Module):
    """Local direct regression; deposit, edge mask and fixed relative indices.

    The observed anchor is supplied as known geometry, not a calibrated position.
    Output energy is predicted directly; there is no physical-sum correction.
    """

    def __init__(self):
        super().__init__()
        row, col = torch.meshgrid(torch.arange(-3, 4) / 3, torch.arange(-3, 4) / 3, indexing="ij")
        self.register_buffer("coordinates", torch.stack((row, col))[None])
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.GELU(),
        )
        self.energy = nn.Linear(66, 1)
        self.position = nn.Linear(66, 2)

    def forward(self, inputs, anchors):
        coordinates = self.coordinates.expand(len(inputs), -1, -1, -1)
        latent = self.encoder(torch.cat((inputs, coordinates), dim=1))
        features = torch.cat((latent, anchors / anchors.new_tensor([29, 84])), dim=1)
        return torch.cat((self.energy(features), self.position(features)), dim=1)


class LocalTransformer(nn.Module):
    """Cell-token attention over the same observed 7x7 window as LocalCNN.

    Tokens retain signed deposit, validity and fixed relative row/column indices.
    The observed anchor is appended only after token pooling, as for LocalCNN.
    """

    def __init__(self):
        super().__init__()
        row, col = torch.meshgrid(
            torch.arange(-3, 4) / 3, torch.arange(-3, 4) / 3, indexing="ij"
        )
        self.register_buffer("coordinates", torch.stack((row, col))[None])
        self.embedding = nn.Linear(4, 64)
        self.encoder = _transformer_stack(64, depth=3)
        for module in self.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.MultiheadAttention):
                module.dropout = 0.0
        self.energy = nn.Linear(66, 1)
        self.position = nn.Linear(66, 2)

    def token_features(self, inputs):
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (2, 7, 7):
            raise ValueError("Expected local deposits and validity with shape (batch,2,7,7)")
        coordinates = self.coordinates.to(dtype=inputs.dtype).expand(len(inputs), -1, -1, -1)
        return torch.cat((inputs, coordinates), dim=1).flatten(2).transpose(1, 2)

    def forward(self, inputs, anchors):
        if anchors.ndim != 2 or tuple(anchors.shape) != (len(inputs), 2):
            raise ValueError("Expected observed anchors with shape (batch,2)")
        latent = self.encoder(self.embedding(self.token_features(inputs))).mean(dim=1)
        features = torch.cat((latent, anchors / anchors.new_tensor([29, 84])), dim=1)
        return torch.cat((self.energy(features), self.position(features)), dim=1)


class DirectRegressor(nn.Module):
    def __init__(self, kind, statistics, architecture="cnn"):
        super().__init__()
        networks = {
            ("full", "cnn"): CNN,
            ("full", "transformer"): Transformer,
            ("local", "cnn"): LocalCNN,
            ("local", "transformer"): LocalTransformer,
        }
        if (kind, architecture) not in networks:
            raise ValueError("Expected a full/local CNN or Transformer")
        self.network = networks[(kind, architecture)]()
        if architecture == "transformer" and kind == "full":
            # The earlier controlled Transformer experiment selected zero dropout.
            for module in self.network.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.MultiheadAttention):
                    module.dropout = 0.0
        self.local = kind == "local"
        self.register_buffer("mean", torch.tensor(statistics["target_mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(statistics["target_std"], dtype=torch.float32))

    def forward(self, inputs, anchors):
        standardized = self.network(inputs, anchors) if self.local else self.network(inputs)
        prediction = standardized * self.std + self.mean
        if self.local:
            prediction = torch.cat((prediction[:, :1], prediction[:, 1:] + anchors), dim=1)
        return prediction


class CyclingSampler:
    """Exactly sized updates across deterministic permutations, with resumable cursor."""

    def __init__(self, count, seed):
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(count, generator=self.generator)
        self.cursor = 0
        self.cycles = 0

    def next(self, count):
        pieces = []
        while count:
            available = min(count, self.count - self.cursor)
            pieces.append(self.order[self.cursor : self.cursor + available])
            self.cursor += available
            count -= available
            if self.cursor == self.count:
                self.order = torch.randperm(self.count, generator=self.generator)
                self.cursor = 0
                self.cycles += 1
        return torch.cat(pieces)

    def state_dict(self):
        return {
            "count": self.count,
            "order": self.order.clone(),
            "cursor": self.cursor,
            "cycles": self.cycles,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if state["count"] != self.count:
            raise ValueError("Sampler count changed")
        self.order = state["order"].clone()
        self.cursor, self.cycles = state["cursor"], state["cycles"]
        self.generator.set_state(state["generator"])


def checked_checkpoint(path: Path) -> dict:
    digest = path.with_name(path.name + ".sha256")
    if not digest.exists() or digest.read_text().strip() != sha256(path):
        raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("kind") != "parallel_studies" or value.get("schema_version") != 1:
        raise ValueError("Checkpoint belongs to another study")
    return value


def verified_copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    shutil.copyfile(source, temporary)
    if sha256(source) != sha256(temporary):
        raise ValueError("Persistence copy checksum mismatch")
    temporary.replace(target)


def persist_case(case_dir: Path, backup_dir: Path | None):
    if backup_dir is None:
        return
    for path in sorted(case_dir.iterdir()):
        if path.is_file() and path.suffix != ".tmp":
            verified_copy(path, backup_dir / case_dir.name / path.name)


@torch.no_grad()
def predict(model, split, device):
    model.eval()
    outputs = []
    for start in range(0, len(split["targets"]), 256):
        selection = slice(start, start + 256)
        values = model(
            split["inputs"][selection].to(device), split["anchors"][selection].to(device)
        )
        if not torch.isfinite(values).all():
            raise RuntimeError("Nonfinite predictions")
        outputs.append(values.cpu().numpy())
    return np.concatenate(outputs)


def _snapshot(model, optimizer, scheduler, sampler, state):
    return {
        "schema_version": 1,
        "kind": "parallel_studies",
        **copy.deepcopy(state),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": rng_state(sampler.generator),
    }


def train_case(
    case,
    train,
    validation,
    statistics,
    config,
    output,
    provenance,
    *,
    device="cpu",
    resume=False,
    remaining_seconds=2700,
    backup_dir=None,
    stop_after=None,
):
    """Train one bounded case. stop_after is only for exact-resume verification."""
    hardware = configure(device, config["seed"])
    output = Path(output)
    case_dir = output / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "case": case,
        "config": config,
        "provenance": provenance,
        "statistics": statistics,
        "train_ids_sha256": ids_hash(train["source_ids"]),
        "validation_ids_sha256": ids_hash(validation["source_ids"]),
        "runtime": {key: hardware[key] for key in ("device", "torch", "cuda_runtime")},
    }
    model = DirectRegressor(
        case["kind"], statistics, architecture=case.get("architecture", "cnn")
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["updates"])
    sampler = CyclingSampler(len(train["targets"]), config["sampling_seed"])
    state = {
        "identity": identity,
        "update": 0,
        "history": [],
        "selected": {},
        "active_train_seconds": 0.0,
        "validation_seconds": 0.0,
        "partial_loss_sum": 0.0,
        "partial_event_count": 0,
    }
    last = case_dir / "last.pt"
    if last.exists():
        if not resume:
            raise ValueError("Existing run requires --resume; never overwrite scientific results")
        loaded = checked_checkpoint(last)
        if loaded["identity"] != identity:
            raise ValueError("Resume source, configuration, data, sample or runtime changed")
        for entry in loaded["selected"].values():
            prediction_path = (output / entry["prediction"]).resolve()
            best_path = (output / entry["checkpoint"]).resolve()
            if not prediction_path.is_relative_to(output.resolve()) or not best_path.is_relative_to(
                output.resolve()
            ):
                raise ValueError("Resume selection contains an unsafe artifact path")
            if (
                not prediction_path.exists()
                or sha256(prediction_path) != entry["prediction_sha256"]
            ):
                raise ValueError("Resume selection is incomplete; checkpoint copies do not match")
            selected_state = checked_checkpoint(best_path)
            if (
                selected_state["identity"] != identity
                or selected_state["update"] != entry["update"]
            ):
                raise ValueError("Resume best checkpoint and last checkpoint do not match")
        model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        sampler.load_state_dict(loaded["sampler"])
        restore_rng(loaded["rng"], sampler.generator)
        state = {key: loaded[key] for key in state}
    elif any(case_dir.iterdir()):
        raise ValueError("Incomplete case artifacts without last checkpoint; use a fresh output")
    np.savez(case_dir / "sample.npz", source_ids=train["source_ids"], targets=train["raw_targets"])
    initial_compute = state["active_train_seconds"] + state["validation_seconds"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def checkpoint():
        save_checkpoint(last, _snapshot(model, optimizer, scheduler, sampler, state))
        write_json(case_dir / "history.json", {"history": state["history"]})
        persist_case(case_dir, backup_dir)

    def exhausted():
        used = state["active_train_seconds"] + state["validation_seconds"] - initial_compute
        return used >= remaining_seconds

    for update in range(state["update"] + 1, config["updates"] + 1):
        if exhausted():
            checkpoint()
            break
        synchronize(device)
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = sampler.next(config["effective_batch_size"])
        for micro in indices.split(config["microbatch_size"]):
            prediction = model(
                train["inputs"][micro].to(device), train["anchors"][micro].to(device)
            )
            loss = task_loss(prediction, train["targets"][micro].to(device), case["task"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss * len(micro) / len(indices)).backward()
            state["partial_loss_sum"] += float(loss.detach()) * len(micro)
            state["partial_event_count"] += len(micro)
        finite_gradients = [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not bool(torch.stack(finite_gradients).all()):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        scheduler.step()
        synchronize(device)
        state["active_train_seconds"] += time.perf_counter() - start
        state["update"] = update
        if update % config["log_every"] == 0 or update == 1:
            print(
                f"{case['name']}: update {update}/{config['updates']}; "
                f"passes {sampler.cycles + sampler.cursor / sampler.count:.2f}; "
                f"train {state['active_train_seconds']:.1f}s",
                flush=True,
            )
        evaluate = update % config["evaluate_every"] == 0 or update == config["updates"]
        if evaluate:
            start = time.perf_counter()
            predictions = predict(model, validation, device)
            metrics = aggregate_metrics(validation["raw_targets"], predictions, case["task"])
            synchronize(device)
            state["validation_seconds"] += time.perf_counter() - start
            state["history"].append(
                {
                    "update": update,
                    "metrics": metrics,
                    "train_loss": state["partial_loss_sum"] / state["partial_event_count"],
                    "active_train_seconds": state["active_train_seconds"],
                    "validation_seconds": state["validation_seconds"],
                    "passes_through_train": sampler.cycles + sampler.cursor / sampler.count,
                }
            )
            state["partial_loss_sum"], state["partial_event_count"] = 0.0, 0
            for selection, score in selection_scores(metrics, case["task"]).items():
                best = state["selected"].get(selection)
                if best is None or score < best["score"]:
                    prediction_file = case_dir / f"predictions_{selection}.npz"
                    np.savez_compressed(
                        prediction_file,
                        predictions=predictions,
                        targets=validation["raw_targets"],
                        source_ids=validation["source_ids"],
                    )
                    entry = {
                        "score": score,
                        "update": update,
                        "metrics": metrics,
                        "prediction": f"{case['name']}/{prediction_file.name}",
                        "prediction_sha256": sha256(prediction_file),
                        "checkpoint": f"{case['name']}/best_{selection}.pt",
                        "subgroups": task_subgroups(
                            validation["raw_targets"], predictions, case["task"]
                        ),
                    }
                    if case["task"] != "position":
                        entry["nonpositive_energy_predictions"] = int(
                            (predictions[:, 0] <= 0).sum()
                        )
                    state["selected"][selection] = entry
                    save_checkpoint(
                        case_dir / f"best_{selection}.pt",
                        _snapshot(model, optimizer, scheduler, sampler, state),
                    )
            print(f"{case['name']}: validation {json.dumps(metrics)}", flush=True)
        if evaluate or update % config["checkpoint_every"] == 0:
            checkpoint()
        if stop_after is not None and update >= stop_after:
            checkpoint()
            break
    complete = state["update"] == config["updates"]
    selected = copy.deepcopy(state["selected"])
    for entry in selected.values():
        entry["checkpoint_sha256"] = sha256(output / entry["checkpoint"])
    active_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not (case["task"] == "energy" and "position" in name)
        and not (case["task"] == "position" and "energy" in name)
    )
    result = {
        "case": case,
        "complete": complete,
        "updates_completed": state["update"],
        "train_count": len(train["targets"]),
        "validation_count": len(validation["targets"]),
        "selected": selected,
        "history": state["history"],
        "transformations": statistics,
        "train_ids_sha256": identity["train_ids_sha256"],
        "validation_ids_sha256": identity["validation_ids_sha256"],
        "active_train_seconds": state["active_train_seconds"],
        "validation_seconds": state["validation_seconds"],
        "compute_seconds": state["active_train_seconds"] + state["validation_seconds"],
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "active_parameters": active_count,
        "hardware": hardware,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "provenance": provenance,
    }
    result["artifacts"] = {
        f"{case['name']}/{path.name}": sha256(path)
        for path in sorted(case_dir.iterdir())
        if path.is_file() and path.name != "result.json"
    }
    write_json(case_dir / "result.json", result)
    persist_case(case_dir, backup_dir)
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


def prepare_full(raw_train, raw_val, indices, transformation):
    selected = {key: value[indices] for key, value in raw_train.items()}
    statistics = fit_statistics(selected["deposits"], selected["targets"])
    train = _tensor_split(
        transform_input(selected["deposits"], transformation, statistics)[:, None], selected
    )
    validation = _tensor_split(
        transform_input(raw_val["deposits"], transformation, statistics)[:, None], raw_val
    )
    return train, validation, statistics, None


def prepare_readout(raw_train, raw_val, regime, config):
    windows = {}
    diagnostics = {}
    for split, raw in [("train", raw_train), ("validation", raw_val)]:
        measured = readout(raw["deposits"], regime, config[f"{split}_noise_seed"], config["noise"])
        windows[split] = observed_window(measured)
        zero = raw["deposits"] == 0
        diagnostics[split] = {
            "negative_measured_cells": int((measured < 0).sum()),
            "observed_max_below_0_5gev_count": int((measured.max(axis=(1, 2)) < 0.5).sum()),
            "mean_global_signed_sum_gev": float(measured.sum(axis=(1, 2), dtype=np.float64).mean()),
            "mean_measurement_in_true_zero_cells_gev": float(
                np.where(zero, measured, 0).sum(axis=(1, 2), dtype=np.float64).mean()
            ),
            "mean_window_signed_sum_gev": float(local_features(windows[split])[:, 0].mean()),
            "observed_anchor_differs_from_clean_max_fraction": float(
                np.mean(
                    measured.reshape(len(measured), -1).argmax(axis=1)
                    != raw["deposits"].reshape(len(measured), -1).argmax(axis=1)
                )
            ),
            "measurement_sha256": array_hash(measured),
        }
    statistics = fit_statistics(
        raw_train["deposits"], raw_train["targets"], anchors=windows["train"]["anchors"]
    )
    splits = {}
    for split, raw in [("train", raw_train), ("validation", raw_val)]:
        window = windows[split]
        inputs = np.stack(
            (
                window["values"] / (statistics["linear_scale_mev"] / 1000),
                window["valid"].astype(np.float32),
            ),
            axis=1,
        )
        splits[split] = _tensor_split(inputs, raw, window["anchors"])
    train_features, val_features = (
        local_features(windows["train"]),
        local_features(windows["validation"]),
    )
    calibration = fit_affine(train_features, raw_train["targets"])
    prediction = affine_predict(val_features, calibration)
    periodic_calibration = fit_periodic(train_features, raw_train["targets"], calibration)
    periodic_prediction = periodic_predict(val_features, periodic_calibration)
    baseline = {
        "calibration": calibration,
        "metrics": regression_metrics(raw_val["targets"], prediction),
        "subgroups": stratified_metrics(raw_val["targets"], prediction),
        "nonpositive_energy_predictions": int((prediction[:, 0] <= 0).sum()),
        "input_diagnostics": diagnostics,
        "predictions": prediction,
        "periodic": {
            "calibration": periodic_calibration,
            "metrics": regression_metrics(raw_val["targets"], periodic_prediction),
            "subgroups": stratified_metrics(raw_val["targets"], periodic_prediction),
            "nonpositive_energy_predictions": int((periodic_prediction[:, 0] <= 0).sum()),
            "predictions": periodic_prediction,
        },
    }
    return splits["train"], splits["validation"], statistics, baseline


def _verify_artifacts(output, result):
    for name, expected in result["artifacts"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != expected:
            raise ValueError("Completed case artifact was changed or lost")


def run_suite(data_dir, output, config, device="cpu", smoke=False, resume=False, backup_dir=None):
    validate_config(config)
    architecture = (
        "transformer" if config["study"] == "parallel_studies_transformer" else "cnn"
    )
    original_config = copy.deepcopy(config)
    config = copy.deepcopy(config)
    if smoke:
        config.update(
            updates=4,
            evaluate_every=2,
            checkpoint_every=2,
            log_every=1,
            effective_batch_size=8,
            microbatch_size=4,
            volume_counts=[8, 16],
            training_seconds_ceiling=120,
        )
    output = external_directory(Path(output))
    if backup_dir is not None:
        backup_dir = external_directory(Path(backup_dir))
        if (
            output == backup_dir
            or output.is_relative_to(backup_dir)
            or backup_dir.is_relative_to(output)
        ):
            raise ValueError("Output and persistence directories must be separate")
        if (
            resume
            and not (output / "summary.json").exists()
            and (backup_dir / "summary.json").exists()
        ):
            for path in sorted(backup_dir.rglob("*")):
                if path.is_file() and path.suffix not in {".tmp", ".partial"}:
                    verified_copy(path, output / path.relative_to(backup_dir))
    checked = load_data(Path(data_dir))
    provenance = copy.deepcopy(checked["provenance"])
    del checked
    for name in [
        "scripts/build_stage2_bundle.py",
        "experiments/parallel_studies.py",
        "experiments/transport.py",
        "experiments/plan.json",
    ]:
        provenance["source_files"][name] = sha256(PROJECT / name)
    provenance["code_sha256"] = canonical_hash(provenance["source_files"])
    provenance["config_sha256"] = canonical_hash(original_config)
    provenance["effective_config_sha256"] = canonical_hash(config)
    raw = {}
    for split in ["train", "validation"]:
        with np.load(Path(data_dir) / f"{split}.npz", allow_pickle=False) as archive:
            raw[split] = {
                name: archive[name].copy() for name in ["deposits", "targets", "source_ids"]
            }
        if smoke:
            limit = 32 if split == "train" else 16
            raw[split] = {key: value[:limit] for key, value in raw[split].items()}
    if len(raw["train"]["targets"]) < config["volume_counts"][-1]:
        raise ValueError("Not enough training events for the predefined nested samples")
    if not smoke and (
        len(raw["train"]["targets"]) != 28118 or len(raw["validation"]["targets"]) != 5947
    ):
        raise ValueError(
            "Scientific run requires the original 28,118/5,947 train/validation export"
        )
    order = np.random.default_rng(config["sampling_seed"]).permutation(len(raw["train"]["targets"]))
    summary_path = output / "summary.json"
    summary = {
        "schema_version": 1,
        "kind": "parallel_studies",
        "smoke": smoke,
        "test_used": False,
        "complete": False,
        "config": original_config,
        "effective_config": config,
        "provenance": provenance,
        "cases": {},
        "selected_input": None,
        "interpretation": "Exploratory validation; one seed; no reserved test accessed",
    }
    if summary_path.exists():
        if not resume:
            raise ValueError("Output already exists; use --resume for an identical run")
        stored = json.loads(summary_path.read_text())
        if any(
            stored[key] != summary[key]
            for key in ["smoke", "config", "effective_config", "provenance"]
        ):
            raise ValueError("Resume suite source, configuration or data changed")
        summary = stored

    def persist_summary():
        summary["compute_seconds"] = sum(
            item["compute_seconds"] for item in summary["cases"].values()
        )
        write_json(summary_path, summary)
        if backup_dir is not None:
            verified_copy(summary_path, backup_dir / "summary.json")

    persist_summary()
    for index, name in enumerate(CASE_NAMES):
        existing = summary["cases"].get(name)
        if existing and existing["complete"]:
            _verify_artifacts(output, existing)
            continue
        used_other = sum(
            item["compute_seconds"] for key, item in summary["cases"].items() if key != name
        )
        spent_here = existing["compute_seconds"] if existing else 0
        last = output / name / "last.pt"
        if last.exists():
            state = checked_checkpoint(last)
            spent_here = state["active_train_seconds"] + state["validation_seconds"]
        remaining = config["training_seconds_ceiling"] - used_other - spent_here
        if remaining <= 0:
            break
        if index < 2:
            selected_input = "log" if index == 0 else "linear"
        else:
            candidates = ["input_log_joint", "input_linear_joint"]
            winner = min(
                candidates, key=lambda key: summary["cases"][key]["selected"]["joint"]["score"]
            )
            selected_input = summary["cases"][winner]["case"]["input"]
            summary["selected_input"] = {
                "input": selected_input,
                "case": winner,
                "selection": "lowest joint score on fixed validation",
            }
        size = len(order)
        if name in {"volume_7000", "volume_14000"}:
            size = config["volume_counts"][0 if name == "volume_7000" else 1]
        case = {
            "name": name,
            "kind": "local" if name.startswith("readout_") else "full",
            "input": "linear" if name.startswith("readout_") else selected_input,
            "task": name.removeprefix("task_") if name.startswith("task_") else "joint",
            "regime": name.removeprefix("readout_") if name.startswith("readout_") else "clean",
            "train_size": size,
        }
        if architecture == "transformer":
            case["architecture"] = architecture
        if case["kind"] == "full":
            prepared = prepare_full(raw["train"], raw["validation"], order[:size], case["input"])
        else:
            prepared = prepare_readout(raw["train"], raw["validation"], case["regime"], config)
        train, validation, statistics, baseline = prepared
        print(
            f"Starting {name}: {size} train, {len(validation['targets'])} validation; "
            f"remaining compute {remaining:.1f}s; smoke={smoke}",
            flush=True,
        )
        result = train_case(
            case,
            train,
            validation,
            statistics,
            config,
            output,
            provenance,
            device=device,
            resume=resume,
            remaining_seconds=remaining,
            backup_dir=backup_dir,
        )
        if baseline:
            periodic = baseline.pop("periodic")
            predictions = baseline.pop("predictions")
            path = output / name / "predictions_baseline.npz"
            np.savez_compressed(
                path,
                predictions=predictions,
                targets=validation["raw_targets"],
                source_ids=validation["source_ids"],
            )
            baseline["prediction"] = f"{name}/{path.name}"
            baseline["prediction_sha256"] = sha256(path)
            result["baseline"] = baseline
            result["artifacts"][baseline["prediction"]] = sha256(path)
            periodic_path = output / name / "predictions_periodic.npz"
            np.savez_compressed(
                periodic_path,
                predictions=periodic.pop("predictions"),
                targets=validation["raw_targets"],
                source_ids=validation["source_ids"],
            )
            periodic["prediction"] = f"{name}/{periodic_path.name}"
            periodic["prediction_sha256"] = sha256(periodic_path)
            result["periodic_baseline"] = periodic
            result["artifacts"][periodic["prediction"]] = sha256(periodic_path)
        write_json(output / name / "result.json", result)
        persist_case(output / name, backup_dir)
        summary["cases"][name] = result
        persist_summary()
        del prepared, train, validation
        if not result["complete"]:
            break
    summary["complete"] = len(summary["cases"]) == len(CASE_NAMES) and all(
        item["complete"] for item in summary["cases"].values()
    )
    persist_summary()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny plumbing check, no scientific result"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    summary = run_suite(
        args.data_dir,
        args.output,
        json.loads(args.config.read_text()),
        args.device,
        args.smoke,
        args.resume,
        args.backup_dir,
    )
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "smoke": summary["smoke"],
                "compute_seconds": summary["compute_seconds"],
                "completed_cases": sum(item["complete"] for item in summary["cases"].values()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
