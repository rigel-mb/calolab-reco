"""Bounded CNN training and validation with verified epoch checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from calolab_reco.metrics import regression_metrics
from calolab_reco.pilot import configure, restore_rng, rng_state, sha256, synchronize, write_json
from calolab_reco.pilot_models import CNN, supervised_loss

PROJECT = Path(__file__).resolve().parents[2]
CONFIG_KEYS = {
    "seed",
    "epochs",
    "effective_batch_size",
    "microbatch_size",
    "learning_rate",
    "weight_decay",
    "eval_batch_size",
    "max_train_seconds",
}


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_fingerprints(bundle_manifest: Path | None = None) -> dict:
    """Match the executing package, including when it is installed in a container."""
    names = ["pyproject.toml", "uv.lock", "configs/cnn.toml"]
    names += [
        p.relative_to(PROJECT).as_posix()
        for p in sorted((PROJECT / "src/calolab_reco").glob("*.py"))
    ]
    files = {name: sha256(PROJECT / name) for name in names}
    code_hash = canonical_hash(files)
    if bundle_manifest is not None:
        bundle_manifest = Path(bundle_manifest).resolve()
        bundle = json.loads(bundle_manifest.read_text())
        for name, expected in bundle["files"].items():
            path = (bundle_manifest.parent / name).resolve()
            if not path.is_relative_to(bundle_manifest.parent) or sha256(path) != expected:
                raise ValueError(f"Bundle member fingerprint mismatch: {name}")
        bundled_code = {
            name: digest
            for name, digest in bundle["files"].items()
            if name in {"pyproject.toml", "uv.lock", "configs/cnn.toml"}
            or name.startswith("src/calolab_reco/")
        }
        if bundled_code != files or bundle["code_sha256"] != code_hash:
            raise ValueError("The executing code differs from the stage 2 bundle.")
    return {"code_sha256": code_hash, "source_files": files}


def validate_config(config: dict) -> None:
    if set(config) != CONFIG_KEYS:
        raise ValueError(f"Training configuration must contain exactly {sorted(CONFIG_KEYS)}")
    integer_keys = ["seed", "epochs", "effective_batch_size", "microbatch_size", "eval_batch_size"]
    if any(type(config[k]) is not int for k in integer_keys):
        raise ValueError("Seed, epoch and batch counts must be integers.")
    micro, effective = config["microbatch_size"], config["effective_batch_size"]
    if not 1 <= micro <= effective <= 128 or effective % micro:
        raise ValueError("Microbatch must divide the effective batch, bounded at 128.")
    if not 1 <= config["epochs"] <= 20 or not 1 <= config["eval_batch_size"] <= 512:
        raise ValueError("Epochs must be 1..20 and evaluation batch size 1..512.")
    for key in ["learning_rate", "weight_decay", "max_train_seconds"]:
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError("Optimizer and time-budget values must be finite.")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative.")
    if not 0 < config["max_train_seconds"] <= 2700:
        raise ValueError("Training and validation compute is limited to at most 2700 seconds.")


def load_data(data_dir: Path, bundle_manifest: Path | None = None, include_train=True) -> dict:
    data_dir = Path(data_dir).resolve()
    manifest_path = data_dir / "stage2_manifest.json"
    metadata = json.loads(manifest_path.read_text())
    if metadata["schema_version"] != 1 or metadata["input_selection"] != "train_validation_only":
        raise ValueError("Only a stage 2 train/validation export is accepted.")
    expected = {"train.npz", "validation.npz"}
    if set(metadata["data_sha256"]) != expected:
        raise ValueError("Both train and validation fingerprints are required; no test export.")
    for name, digest in metadata["data_sha256"].items():
        if sha256(data_dir / name) != digest:
            raise ValueError(f"Data fingerprint mismatch: {name}")
    scale = float(metadata["input_scale"])
    mean = np.asarray(metadata["target_mean"], dtype=np.float64)
    std = np.asarray(metadata["target_std"], dtype=np.float64)
    if (
        not math.isfinite(scale)
        or scale <= 0
        or mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid train-fitted transformation statistics.")
    output = {}
    ids_by_split = {}
    for split, count_key in [("train", "train_count"), ("validation", "val_count")]:
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            ids = archive["source_ids"].copy()
            count = metadata[count_key]
            if (
                type(count) is not int
                or count < 1
                or ids.shape != (count, 2)
                or ids.dtype.kind not in "iu"
                or len(np.unique(ids, axis=0)) != count
            ):
                raise ValueError("Invalid or repeated source identifiers.")
            ids_by_split[split] = {tuple(row) for row in ids.tolist()}
            if split == "train" and not include_train:
                continue
            x, raw_targets = archive["deposits"].copy(), archive["targets"].copy()
        if (
            x.shape != (count, 30, 85)
            or raw_targets.shape != (count, 3)
            or x.dtype != np.float32
            or raw_targets.dtype != np.float32
        ):
            raise ValueError("Expected raw float32 deposits (N,30,85) and targets (N,3).")
        if (
            not np.isfinite(x).all()
            or (x < 0).any()
            or not np.isfinite(raw_targets).all()
            or (raw_targets[:, 0] <= 0).any()
        ):
            raise ValueError("Inputs must be finite with nonnegative deposits and positive energy.")
        # Transform the in-memory copy only, avoiding a second complete deposit array.
        np.divide(x, scale, out=x)
        np.log1p(x, out=x)
        targets = ((raw_targets - mean) / std).astype(np.float32)
        if not np.isfinite(x).all() or not np.isfinite(targets).all():
            raise ValueError("Nonfinite transformed inputs or targets.")
        output[split] = {
            "inputs": torch.from_numpy(x[:, None]),
            "targets": torch.from_numpy(targets),
            "raw_targets": raw_targets,
            "source_ids": ids,
        }
    if ids_by_split["train"] & ids_by_split["validation"]:
        raise ValueError("Train and validation source identifiers overlap.")
    output["transformations"] = {
        "input_scale": scale,
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
    }
    output["metadata"] = metadata
    output["provenance"] = {
        "stage2_manifest_sha256": sha256(manifest_path),
        "prepared_manifest_sha256": metadata["prepared_manifest_sha256"],
        "prepared_output_sha256": metadata["prepared_output_sha256"],
        "data_sha256": metadata["data_sha256"],
        **source_fingerprints(bundle_manifest),
    }
    return output


def external_directory(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(PROJECT):
        raise ValueError("Training outputs and predictions must remain outside the repository.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    fingerprint = path.with_name(path.name + ".sha256")
    temporary_hash = fingerprint.with_name(fingerprint.name + ".tmp")
    temporary_hash.write_text(sha256(path) + "\n")
    temporary_hash.replace(fingerprint)


def load_checkpoint(path: Path) -> dict:
    path = Path(path)
    fingerprint = path.with_name(path.name + ".sha256")
    if not fingerprint.is_file() or fingerprint.read_text().strip() != sha256(path):
        raise ValueError("Checkpoint fingerprint is missing or does not match.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1 or state.get("model_kind") != "cnn":
        raise ValueError("Unsupported training checkpoint.")
    return state


def persist_files(output: Path, destination: Path | None) -> None:
    if destination is None:
        return
    for name in ["best.pt", "best.pt.sha256", "last.pt", "last.pt.sha256", "history.json"]:
        source = output / name
        if not source.exists():
            continue
        target = destination / name
        if source.resolve() == target.resolve():
            raise ValueError("Persistence requires a separate directory.")
        temporary = target.with_name(target.name + ".partial")
        shutil.copyfile(source, temporary)
        if sha256(source) != sha256(temporary):
            raise ValueError("Persistence copy fingerprint mismatch.")
        temporary.replace(target)
        if sha256(source) != sha256(target):
            raise ValueError("Persisted file fingerprint mismatch.")


@torch.no_grad()
def predict(model, split: dict, transformations: dict, batch_size: int, device: str, limit=None):
    model.eval()
    count = len(split["inputs"]) if limit is None else min(limit, len(split["inputs"]))
    predictions = np.empty((count, 3), dtype=np.float32)
    weighted_loss = 0.0
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        values = model(split["inputs"][start:stop].to(device))
        loss = supervised_loss(values, split["targets"][start:stop].to(device))
        if not torch.isfinite(values).all() or not torch.isfinite(loss):
            raise RuntimeError("Nonfinite validation predictions or loss.")
        weighted_loss += float(loss) * (stop - start)
        predictions[start:stop] = values.cpu().numpy()
    predictions = predictions.astype(np.float64) * np.asarray(
        transformations["target_std"]
    ) + np.asarray(transformations["target_mean"])
    return predictions, weighted_loss / count


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def train(
    data_dir,
    config,
    output,
    device,
    resume=None,
    stop_after_epochs=None,
    persist_dir=None,
    bundle_manifest=None,
) -> dict:
    start_wall = time.perf_counter()
    validate_config(config)
    if stop_after_epochs is not None and not 1 <= stop_after_epochs <= config["epochs"]:
        raise ValueError("stop_after_epochs must be within the configured epoch count.")
    hardware = configure(device, config["seed"])
    data = load_data(Path(data_dir), bundle_manifest)
    data["provenance"]["config_sha256"] = canonical_hash(config)
    output = external_directory(output)
    persisted = external_directory(persist_dir) if persist_dir is not None else None
    if persisted == output:
        raise ValueError("Persistence requires a separate directory.")
    if resume is None and any(
        (output / name).exists() for name in ["best.pt", "last.pt", "history.json"]
    ):
        raise FileExistsError("Existing training output requires explicit checkpoint resume.")
    model = CNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    updates_per_epoch = math.ceil(data["metadata"]["train_count"] / config["effective_batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates_per_epoch * config["epochs"]
    )
    sampler = torch.Generator().manual_seed(config["seed"])
    history, best_state, in_progress = [], None, None
    epoch = global_step = 0
    active_seconds = validation_seconds = prior_wall = 0.0
    best_loss = math.inf
    best_epoch = None
    prior_allocated = prior_reserved = 0
    if resume is not None:
        state = load_checkpoint(Path(resume))
        if (
            state["config"] != config
            or state["device"] != device
            or state["provenance"] != data["provenance"]
            or state["transformations"] != data["transformations"]
        ):
            raise ValueError("Resume code, data, device, configuration or transformations differ.")
        if (output / "last.pt").exists() and sha256(output / "last.pt") != sha256(Path(resume)):
            raise ValueError(
                "Output contains a different checkpoint; use a new external directory."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], sampler)
        epoch, global_step = state["epoch"], state["global_step"]
        history, in_progress = state["history"], state["in_progress"]
        best_loss, best_epoch = state["best_validation_loss"], state["best_epoch"]
        active_seconds = state["active_training_seconds"]
        validation_seconds, prior_wall = state["validation_seconds"], state["wall_seconds"]
        prior_allocated = state.get("peak_torch_allocated_bytes") or 0
        prior_reserved = state.get("peak_torch_reserved_bytes") or 0
        best_state = state.get("best_checkpoint")
        if best_state is None and epoch == best_epoch:
            best_state = {key: value for key, value in state.items() if key != "best_checkpoint"}
        if best_state is not None:
            save_checkpoint(output / "best.pt", best_state)
        if not (output / "last.pt").exists():
            save_checkpoint(output / "last.pt", state)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def resource_metrics():
        return {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "peak_torch_allocated_bytes": max(prior_allocated, torch.cuda.max_memory_allocated())
            if device == "cuda"
            else None,
            "peak_torch_reserved_bytes": max(prior_reserved, torch.cuda.max_memory_reserved())
            if device == "cuda"
            else None,
        }

    def snapshot():
        return _cpu_copy(
            {
                "schema_version": 1,
                "model_kind": "cnn",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng_state(sampler),
                "epoch": epoch,
                "global_step": global_step,
                "history": history,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "in_progress": in_progress,
                "config": config,
                "device": device,
                "hardware": hardware,
                "transformations": data["transformations"],
                "provenance": data["provenance"],
                "active_training_seconds": active_seconds,
                "validation_seconds": validation_seconds,
                "gpu_compute_seconds": active_seconds + validation_seconds,
                "wall_seconds": prior_wall + time.perf_counter() - start_wall,
                **resource_metrics(),
            }
        )

    target_epoch = stop_after_epochs or config["epochs"]
    status = "completed" if epoch == config["epochs"] else "stopped_after_epochs"
    while epoch < target_epoch:
        if in_progress is None:
            in_progress = {
                "order": torch.randperm(len(data["train"]["inputs"]), generator=sampler),
                "next_start": 0,
                "weighted_loss": 0.0,
                "seen": 0,
            }
        model.train()
        count = len(in_progress["order"])
        while in_progress["next_start"] < count:
            if active_seconds + validation_seconds >= config["max_train_seconds"]:
                status = "time_budget_exhausted"
                break
            synchronize(device)
            tick = time.perf_counter()
            start = in_progress["next_start"]
            batch_ids = in_progress["order"][start : start + config["effective_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for offset in range(0, len(batch_ids), config["microbatch_size"]):
                indices = batch_ids[offset : offset + config["microbatch_size"]]
                values = model(data["train"]["inputs"][indices].to(device))
                loss = supervised_loss(values, data["train"]["targets"][indices].to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss.")
                (loss * (len(indices) / len(batch_ids))).backward()
                update_loss += float(loss.detach()) * len(indices)
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise RuntimeError("Nonfinite training gradients.")
            optimizer.step()
            scheduler.step()
            synchronize(device)
            active_seconds += time.perf_counter() - tick
            global_step += 1
            in_progress["next_start"] += len(batch_ids)
            in_progress["seen"] += len(batch_ids)
            in_progress["weighted_loss"] += update_loss
        if active_seconds + validation_seconds >= config["max_train_seconds"]:
            status = "time_budget_exhausted"
        if status == "time_budget_exhausted":
            state = snapshot()
            state["best_checkpoint"] = best_state
            save_checkpoint(output / "last.pt", state)
            break
        synchronize(device)
        tick = time.perf_counter()
        predictions, validation_loss = predict(
            model, data["validation"], data["transformations"], config["eval_batch_size"], device
        )
        metrics = regression_metrics(data["validation"]["raw_targets"], predictions)
        synchronize(device)
        validation_seconds += time.perf_counter() - tick
        epoch += 1
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_count": in_progress["seen"],
                "training_loss": in_progress["weighted_loss"] / count,
                "validation_loss": validation_loss,
                "validation_metrics": metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "active_training_seconds": active_seconds,
                "validation_seconds": validation_seconds,
                "wall_seconds": prior_wall + time.perf_counter() - start_wall,
            }
        )
        in_progress = None
        improved = validation_loss < best_loss  # Strict inequality retains the earliest tie.
        if improved:
            best_loss, best_epoch = validation_loss, epoch
            best_state = snapshot()
            save_checkpoint(output / "best.pt", best_state)
        state = snapshot()
        state["best_checkpoint"] = best_state
        save_checkpoint(output / "last.pt", state)
        write_json(
            output / "history.json",
            {"schema_version": 1, "history": history, "provenance": data["provenance"]},
        )
        persist_files(output, persisted)
        print(
            f"Epoch {epoch}/{config['epochs']}: validation loss {validation_loss:.6f}", flush=True
        )
    if epoch == config["epochs"]:
        status = "completed"
    result = {
        "schema_version": 1,
        "model_kind": "cnn",
        "status": status,
        "completed": status == "completed",
        "epochs_completed": epoch,
        "train_count": data["metadata"]["train_count"],
        "val_count": data["metadata"]["val_count"],
        "global_step": global_step,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
        "active_training_seconds": active_seconds,
        "validation_seconds": validation_seconds,
        "gpu_compute_seconds": active_seconds + validation_seconds,
        "wall_seconds": prior_wall + time.perf_counter() - start_wall,
        **resource_metrics(),
        "hardware": hardware,
        "config": config,
        "provenance": data["provenance"],
        "transformations": data["transformations"],
        "test_used": False,
        "checkpoint_selection": "minimum validation weighted standardized MSE; earliest tie",
    }
    write_json(output / "history.json", result)
    persist_files(output, persisted)
    return result


def evaluate(data_dir, checkpoint, output, device="cpu", limit=None, bundle_manifest=None) -> dict:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Evaluation limit must be a positive integer.")
    state = load_checkpoint(Path(checkpoint))
    configure(device, state["config"]["seed"])
    data = load_data(Path(data_dir), bundle_manifest, include_train=False)
    data["provenance"]["config_sha256"] = canonical_hash(state["config"])
    if (
        state["provenance"] != data["provenance"]
        or state["transformations"] != data["transformations"]
    ):
        raise ValueError("Evaluation code, data or transformations differ from the checkpoint.")
    output = external_directory(output)
    if any((output / name).exists() for name in ["predictions.npz", "metrics.json"]):
        raise FileExistsError("Preserve existing evaluation outputs or choose a new directory.")
    model = CNN().to(device)
    model.load_state_dict(state["model"])
    predictions, loss = predict(
        model,
        data["validation"],
        data["transformations"],
        state["config"]["eval_batch_size"],
        device,
        limit,
    )
    count = len(predictions)
    targets = data["validation"]["raw_targets"][:count]
    source_ids = data["validation"]["source_ids"][:count]
    prediction_path = output / "predictions.npz"
    temporary = output / "predictions.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, source_ids=source_ids, targets=targets, predictions=predictions)
    temporary.replace(prediction_path)
    result = {
        "schema_version": 1,
        "model_kind": "cnn",
        "split": "validation",
        "count": count,
        "full_validation_count": len(data["validation"]["inputs"]),
        "limited": count < len(data["validation"]["inputs"]),
        "test_used": False,
        "device": device,
        "checkpoint_epoch": state["epoch"],
        "checkpoint_sha256": sha256(Path(checkpoint)),
        "provenance": data["provenance"],
        "source_ids_sha256": hashlib.sha256(source_ids.astype("<i8").tobytes()).hexdigest(),
        "predictions_sha256": sha256(prediction_path),
        "validation_loss": loss,
        "metrics": regression_metrics(targets, predictions),
    }
    write_json(output / "metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=["cpu", "cuda"], required=True)
        command.add_argument("--bundle-manifest", type=Path)
    training = commands.choices["train"]
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.add_argument("--stop-after-epochs", type=int)
    training.add_argument("--persist-dir", type=Path)
    evaluation = commands.choices["evaluate"]
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--limit", type=int)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "train":
        args["config"] = tomllib.loads(args["config"].read_text())
        result = train(**args)
        print(f"CNN training status: {result['status']}; test data unused.")
    else:
        result = evaluate(**args)
        print(f"Evaluated {result['count']} validation events on {result['device']}.")


if __name__ == "__main__":
    main()
