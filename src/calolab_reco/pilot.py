"""Bounded timing and fresh-process restart checks, never final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
import tomllib
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from calolab_reco.pilot_models import (
    CNN,
    MaskedAutoencoder,
    Transformer,
    masked_loss,
    supervised_loss,
)

KINDS = {"cnn": CNN, "transformer": Transformer, "masked_pretraining": MaskedAutoencoder}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configure(device: str, seed: int) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select a GPU runtime; no CPU fallback is used.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # One deterministic attention backend is used for timing and restart verification.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    hardware = {
        "device": device,
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": "float32; TF32 disabled",
        "attention_backend": "math",
        "deterministic_algorithms": True,
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        hardware.update(
            gpu_name=properties.name,
            gpu_total_bytes=properties.total_memory,
            compute_capability=[properties.major, properties.minor],
        )
        probe = torch.ones(8, device="cuda")
        assert (probe @ probe).item() == 8
        torch.cuda.synchronize()
    return hardware


def validate_config(config: dict) -> None:
    micro, effective = config["microbatch_size"], config["effective_batch_size"]
    if not 1 <= micro <= effective <= 128 or effective % micro:
        raise ValueError("Microbatch must divide the effective batch, which is at most 128.")
    if not 0 <= config["warmup_steps"] <= 10 or not 1 <= config["measured_steps"] <= 50:
        raise ValueError("Pilot update counts exceed their bounds.")
    if not 1 <= config["resume_prefix_steps"] <= 5:
        raise ValueError("Restart prefix must contain 1 to 5 updates.")
    if not 1 <= config["max_phase_seconds"] <= 240:
        raise ValueError("Each pilot phase is limited to at most 240 seconds.")


def load_inputs(data: Path, manifest_path: Path, config: dict, bundle_path: Path | None):
    """Use only the explicit train export and its recorded transformation statistics."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["input_selection"] != "train_only" or sha256(data) != manifest["data_sha256"]:
        raise ValueError("Train-only export or data fingerprint check failed.")
    with np.load(data, allow_pickle=False) as archive:
        x = archive["deposits"].copy()
        y = archive["targets"].copy()
        ids = archive["source_ids"].copy()
    if x.shape != (len(y), 30, 85) or y.shape != (len(x), 3) or ids.shape != (len(x), 2):
        raise ValueError("Unexpected pilot input dimensions.")
    if len(x) != manifest["subset_count"] or len(x) < config["microbatch_size"]:
        raise ValueError("Unexpected pilot sample size.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any():
        raise ValueError("Pilot inputs must be finite and deposits nonnegative.")
    scale = float(manifest["input_scale"])
    mean = np.asarray(manifest["target_mean"], dtype=np.float64)
    std = np.asarray(manifest["target_std"], dtype=np.float64)
    if (
        not np.isfinite(scale)
        or scale <= 0
        or mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid train-fitted transformation statistics.")
    if manifest["mask_ratio"] not in (0.25, 0.5):
        raise ValueError("Unexpected masking ratio.")
    x = torch.from_numpy(np.log1p(x / scale).astype(np.float32))[:, None]
    y = torch.from_numpy(((y - mean) / std).astype(np.float32))
    provenance = {
        "data_sha256": manifest["data_sha256"],
        "pilot_manifest_sha256": sha256(manifest_path),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
        "code_sha256": None,
    }
    if bundle_path:
        bundle = json.loads(bundle_path.read_text())
        for name, digest in bundle["files"].items():
            target = (bundle_path.parent / name).resolve()
            if not target.is_relative_to(bundle_path.parent.resolve()) or sha256(target) != digest:
                raise ValueError(f"Bundle fingerprint failed: {name}")
        provenance["code_sha256"] = bundle["code_sha256"]
    return x, y, manifest, provenance


def build(kind: str, config: dict, device: str):
    model = KINDS[kind]().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["scheduler_steps"]
    )
    return model, optimizer, scheduler


def update(kind, model, optimizer, scheduler, x, y, config, mask_ratio, device, sampler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = config["effective_batch_size"] // config["microbatch_size"]
    loss_value = 0.0
    for _ in range(accumulation):
        indices = torch.randint(len(x), (config["microbatch_size"],), generator=sampler)
        batch_x, batch_y = x[indices].to(device), y[indices].to(device)
        if kind == "masked_pretraining":
            result = model(batch_x, mask_ratio=mask_ratio)
            loss = masked_loss(result["predictions"], result["targets"], result["mask"])
        else:
            loss = supervised_loss(model(batch_x), batch_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {kind}")
        (loss / accumulation).backward()
        loss_value += float(loss.detach()) / accumulation
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError(f"Nonfinite gradients in {kind}")
    optimizer.step()
    scheduler.step()
    return loss_value


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def rng_state(sampler):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            torch.tensor(numpy_state[1].astype(np.int64)),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler.get_state(),
    }


def restore_rng(state, sampler):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler.set_state(state["sampler"])


def snapshot(model, optimizer, scheduler, sampler):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state(sampler),
    }


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return sha256(path)


def compare_state(actual, expected):
    """Strict replay comparison includes optimizer, schedule, RNG and model weights."""
    if isinstance(actual, torch.Tensor):
        return isinstance(expected, torch.Tensor) and torch.equal(actual.cpu(), expected.cpu())
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            compare_state(actual[k], expected[k]) for k in actual
        )
    if isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(
            compare_state(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def check_deadline(start, config):
    if time.perf_counter() - start >= config["max_phase_seconds"]:
        raise TimeoutError("Pilot phase reached its time ceiling; review before increasing it.")


def benchmark(x, y, manifest, config, device, hardware, provenance):
    rows = []
    start = time.perf_counter()
    for index, kind in enumerate(KINDS):
        configure(device, config["seed"] + index)
        model, optimizer, scheduler = build(kind, config, device)
        sampler = torch.Generator().manual_seed(config["seed"])
        for _ in range(config["warmup_steps"]):
            check_deadline(start, config)
            update(
                kind,
                model,
                optimizer,
                scheduler,
                x,
                y,
                config,
                manifest["mask_ratio"],
                device,
                sampler,
            )
        synchronize(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(config["measured_steps"]):
            check_deadline(start, config)
            synchronize(device)
            tick = time.perf_counter()
            update(
                kind,
                model,
                optimizer,
                scheduler,
                x,
                y,
                config,
                manifest["mask_ratio"],
                device,
                sampler,
            )
            synchronize(device)
            durations.append(time.perf_counter() - tick)
        median = float(np.median(durations))
        p95 = float(np.quantile(durations, 0.95))
        updates_per_epoch = math.ceil(manifest["full_train_count"] / config["effective_batch_size"])
        rows.append(
            {
                "workload": kind,
                "parameters": sum(p.numel() for p in model.parameters()),
                "measured_updates": len(durations),
                "seconds_per_update": durations,
                "median_seconds": median,
                "p95_seconds": p95,
                "events_per_second_at_median": config["effective_batch_size"] / median,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated()
                if device == "cuda"
                else None,
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()
                if device == "cuda"
                else None,
                "projected_minutes_median": median
                * updates_per_epoch
                * config["projected_epochs"]
                / 60,
                "projected_minutes_p95": p95 * updates_per_epoch * config["projected_epochs"] / 60,
            }
        )
        del model, optimizer, scheduler
        if device == "cuda":
            torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "hardware": hardware,
        "provenance": provenance,
        "config": config,
        "subset_count": len(x),
        "full_train_count": manifest["full_train_count"],
        "mask_ratio": manifest["mask_ratio"],
        "workloads": rows,
        "phase_wall_seconds": time.perf_counter() - start,
        "accuracy_evaluated": False,
        "test_used": False,
    }


def checkpoint_phase(x, y, manifest, config, device, provenance, output):
    start = time.perf_counter()
    checks = {}
    for index, kind in enumerate(KINDS):
        configure(device, config["seed"] + index)
        model, optimizer, scheduler = build(kind, config, device)
        sampler = torch.Generator().manual_seed(config["seed"])
        for _ in range(config["resume_prefix_steps"]):
            check_deadline(start, config)
            update(
                kind,
                model,
                optimizer,
                scheduler,
                x,
                y,
                config,
                manifest["mask_ratio"],
                device,
                sampler,
            )
        checkpoint = snapshot(model, optimizer, scheduler, sampler)
        checkpoint.update(
            kind=kind,
            step=config["resume_prefix_steps"],
            provenance=provenance,
            config=config,
            device=device,
            transformations={k: manifest[k] for k in ["input_scale", "target_mean", "target_std"]},
        )
        name = f"{kind}_checkpoint.pt"
        checks[name] = save_checkpoint(output / name, checkpoint)
        loss = update(
            kind, model, optimizer, scheduler, x, y, config, manifest["mask_ratio"], device, sampler
        )
        reference = snapshot(model, optimizer, scheduler, sampler)
        reference["loss"] = loss
        checks[f"{kind}_reference.pt"] = save_checkpoint(output / f"{kind}_reference.pt", reference)
        del model, optimizer, scheduler
        if device == "cuda":
            torch.cuda.empty_cache()
    write_json(output / "checkpoint_files.json", {"files": checks, "provenance": provenance})


def resume_phase(x, y, manifest, config, device, provenance, checkpoint_dir):
    start = time.perf_counter()
    stored = json.loads((checkpoint_dir / "checkpoint_files.json").read_text())
    expected_files = {f"{kind}_{role}.pt" for kind in KINDS for role in ["checkpoint", "reference"]}
    if set(stored["files"]) != expected_files:
        raise ValueError("All six checkpoint/reference fingerprints are required.")
    if stored["provenance"] != provenance:
        raise ValueError(
            "Checkpoint provenance differs from the current code, data or configuration."
        )
    for name, digest in stored["files"].items():
        target = (checkpoint_dir / name).resolve()
        if not target.is_relative_to(checkpoint_dir.resolve()) or sha256(target) != digest:
            raise ValueError("Checkpoint fingerprint mismatch.")
    checks = []
    for kind in KINDS:
        check_deadline(start, config)
        state = torch.load(
            checkpoint_dir / f"{kind}_checkpoint.pt", map_location="cpu", weights_only=True
        )
        reference = torch.load(
            checkpoint_dir / f"{kind}_reference.pt", map_location="cpu", weights_only=True
        )
        if (
            state["config"] != config
            or state["provenance"] != provenance
            or state["device"] != device
        ):
            raise ValueError("Checkpoint configuration, device or provenance mismatch.")
        model, optimizer, scheduler = build(kind, config, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler = torch.Generator()
        restore_rng(state["rng"], sampler)
        loss = update(
            kind, model, optimizer, scheduler, x, y, config, manifest["mask_ratio"], device, sampler
        )
        actual = snapshot(model, optimizer, scheduler, sampler)
        actual["loss"] = loss
        passed = compare_state(actual, reference)
        checks.append(
            {
                "workload": kind,
                "exact_replay": passed,
                "restored_step": state["step"],
                "next_loss_finite": math.isfinite(loss),
            }
        )
        if not passed:
            raise RuntimeError(f"Fresh-process replay differs for {kind}; do not start full runs.")
        del model, optimizer, scheduler
    return {
        "schema_version": 1,
        "provenance": provenance,
        "device": device,
        "checks": checks,
        "phase_wall_seconds": time.perf_counter() - start,
        "process_restart_checked": True,
        "colab_vm_restart_checked": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["benchmark", "checkpoint", "resume"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    validate_config(config)
    hardware = configure(args.device, config["seed"])
    x, y, manifest, provenance = load_inputs(args.data, args.manifest, config, args.bundle_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "benchmark":
        result = benchmark(x, y, manifest, config, args.device, hardware, provenance)
        write_json(args.output / "benchmark.json", result)
    elif args.phase == "checkpoint":
        checkpoint_phase(x, y, manifest, config, args.device, provenance, args.output)
    else:
        if args.checkpoint_dir is None:
            parser.error("resume requires --checkpoint-dir")
        result = resume_phase(x, y, manifest, config, args.device, provenance, args.checkpoint_dir)
        write_json(args.output / "resume.json", result)
    print(f"Pilot phase completed: {args.phase} ({args.device}); no accuracy evaluation.")


if __name__ == "__main__":
    main()
