"""Bounded local confirmation with one shared pretraining per training seed."""

from __future__ import annotations

import argparse
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from calolab_reco.pilot import configure, restore_rng, rng_state, synchronize

from . import data, models
from . import transport as io_tools


def cases(config):
    result = []
    for regime, seeds in [
        (config["primary_regime"], config["training_seeds"]),
        *[(r, config["training_seeds"][:1]) for r in config["control_regimes"]],
    ]:
        for seed in seeds:
            for arch in ("cnn", "transformer"):
                for task in ("energy", "position"):
                    result.append(
                        dict(
                            name=f"{regime}_{seed}_{arch}_{task}",
                            regime=regime,
                            seed=seed,
                            architecture=arch,
                            task=task,
                            phase="direct",
                            parent=None,
                        )
                    )
            if regime == config["primary_regime"]:
                name = f"{regime}_{seed}_pretraining"
                result.append(
                    dict(
                        name=name,
                        regime=regime,
                        seed=seed,
                        architecture="transformer",
                        task=None,
                        phase="pretraining",
                        parent=None,
                    )
                )
                for task in ("energy", "position"):
                    result.append(
                        dict(
                            name=f"{regime}_{seed}_finetuning_{task}",
                            regime=regime,
                            seed=seed,
                            architecture="transformer",
                            task=task,
                            phase="finetuning",
                            parent=name,
                        )
                    )
    return result


def tensor_hash(state):
    h = __import__("hashlib").sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def checkpoint_save(path, state, backup=None):
    b = io.BytesIO()
    torch.save(cpu_copy(state), b)
    io_tools.write_archive(
        path,
        {"state.pt": b.getvalue()},
        "confirmation_checkpoint",
        dict(identity=state["identity"], update=state["update"]),
    )
    if backup is not None:
        io_tools.copy_verified(path, backup)


def checkpoint_load(path):
    manifest, payload = io_tools.read_archive(path, "confirmation_checkpoint")
    if set(payload) != {"state.pt"}:
        raise ValueError("Unexpected checkpoint payload")
    state = torch.load(io.BytesIO(payload["state.pt"]), map_location="cpu", weights_only=True)
    if (
        state["identity"] != manifest["metadata"]["identity"]
        or state["update"] != manifest["metadata"]["update"]
    ):
        raise ValueError("Checkpoint metadata differs")
    return state


@torch.no_grad()
def predict(model, split, device):
    model.eval()
    parts = []
    for start in range(0, len(split["targets"]), 256):
        parts.append(
            model(
                split["inputs"][start : start + 256].to(device),
                split["anchors"][start : start + 256].to(device),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(parts)


def make_model(case, stats, device, parent=None):
    if case["phase"] == "pretraining":
        return models.LocalMaskedModel().to(device), None
    model = models.DirectRegressor("local", stats, case["architecture"])
    transfer = None
    if parent is not None:
        heads = {
            k: v
            for k, v in model.network.state_dict().items()
            if k.startswith(("energy.", "position."))
        }
        before = tensor_hash(heads)
        models.transfer_encoder(model, parent)
        transfer = dict(
            encoder_sha256=tensor_hash(parent),
            fresh_head_sha256=before,
            decoder_retained=False,
            full_encoder_trainable=True,
        )
    return model.to(device), transfer


def train_phase(
    case,
    train,
    validation,
    stats,
    config,
    path,
    identity,
    device="cpu",
    parent=None,
    backup=None,
    remaining_seconds=10800,
    stop_after=None,
):
    hardware = configure(device, case["seed"])
    identity = {
        **identity,
        "case": case,
        "statistics": stats,
        "parent_encoder_sha256": tensor_hash(parent) if parent is not None else None,
        "runtime": {k: hardware[k] for k in ("device", "torch", "cuda_runtime")},
    }
    model, transfer = make_model(case, stats, device, parent)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    total = config["pretraining_updates"] if case["phase"] == "pretraining" else config["updates"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    sampler = models.CyclingSampler(len(train["targets"]), config["sampling_seed"] + case["seed"])
    mask_rng = torch.Generator().manual_seed(config["mask_seed"] + case["seed"])
    state = dict(
        identity=identity,
        update=0,
        history=[],
        best=None,
        train_seconds=0.0,
        validation_seconds=0.0,
        partial_loss=0.0,
        partial_count=0,
        hardware=hardware,
        transfer=transfer,
        peak_gpu_bytes=0,
    )
    if case["phase"] == "pretraining":
        state["mask_audit"] = models.choose_mask(train["inputs"], config)
    if path.exists():
        state = checkpoint_load(path)
        if state["identity"] != identity:
            raise ValueError("Checkpoint input, protocol or runtime changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        mask_rng.set_state(state["mask_rng"])
        restore_rng(state["rng"], sampler.generator)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_compute = state["train_seconds"] + state["validation_seconds"]

    def save():
        state.update(
            model=cpu_copy(model.state_dict()),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            sampler=sampler.state_dict(),
            mask_rng=mask_rng.get_state(),
            rng=rng_state(sampler.generator),
        )
        if device == "cuda":
            state["peak_gpu_bytes"] = max(
                state["peak_gpu_bytes"], torch.cuda.max_memory_allocated()
            )
        checkpoint_save(path, state, backup)

    for update in range(state["update"] + 1, total + 1):
        if (
            state["train_seconds"] + state["validation_seconds"] - start_compute
            >= remaining_seconds
        ):
            save()
            break
        synchronize(device)
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = sampler.next(config["effective_batch_size"])
        mask = None
        counts = None
        if case["phase"] == "pretraining":
            mask = models.sample_mask(
                train["inputs"][indices], state["mask_audit"]["ratio"], mask_rng
            )
            target = train["inputs"][indices, 0].flatten(1)
            counts = [int((mask & (target > 0)).sum()), int((mask & (target == 0)).sum())]
        for start_micro in range(0, len(indices), config["microbatch_size"]):
            idx = indices[start_micro : start_micro + config["microbatch_size"]]
            x = train["inputs"][idx].to(device)
            if mask is not None:
                m = mask[start_micro : start_micro + len(idx)].to(device)
                loss = models.reconstruction_loss(model(x, m), x[:, 0].flatten(1), m, counts)
                weighted = loss
            else:
                p = model(x, train["anchors"][idx].to(device))
                loss = data.task_loss(p, train["targets"][idx].to(device), case["task"])
                weighted = loss * len(idx) / len(indices)
            if not torch.isfinite(weighted):
                raise RuntimeError("Nonfinite training loss")
            weighted.backward()
            state["partial_loss"] += float(weighted.detach())
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        scheduler.step()
        synchronize(device)
        state["train_seconds"] += time.perf_counter() - start
        state["update"] = update
        state["partial_count"] += 1
        if update == 1 or update % config["log_every"] == 0:
            print(
                f"{case['name']}: {update}/{total}; train {state['train_seconds']:.1f}s", flush=True
            )
        if update % config["evaluate_every"] == 0 or update == total:
            row = dict(
                update=update,
                train_loss=state["partial_loss"] / state["partial_count"],
                train_seconds=state["train_seconds"],
            )
            if case["phase"] != "pretraining":
                start = time.perf_counter()
                p = predict(model, validation, device)
                metrics = data.aggregate_metrics(validation["raw_targets"], p, case["task"])
                synchronize(device)
                state["validation_seconds"] += time.perf_counter() - start
                score = data.selection_scores(metrics, case["task"])[case["task"]]
                row["metrics"] = metrics
                if state["best"] is None or score < state["best"]["score"]:
                    state["best"] = dict(
                        score=score,
                        metrics=metrics,
                        update=update,
                        model=cpu_copy(model.state_dict()),
                    )
            row["validation_seconds"] = state["validation_seconds"]
            state["history"].append(row)
            state["partial_loss"] = 0.0
            state["partial_count"] = 0
            print(f"{case['name']}: {json.dumps(row)}", flush=True)
        if (
            update % config["checkpoint_every"] == 0
            or update == total
            or (stop_after is not None and update >= stop_after)
        ):
            save()
        if stop_after is not None and update >= stop_after:
            break
    if not path.exists():
        save()
    return state


def effective(protocol, smoke):
    c = copy.deepcopy(protocol)
    if smoke:
        c.update(
            updates=2,
            pretraining_updates=2,
            evaluate_every=1,
            checkpoint_every=1,
            log_every=1,
            effective_batch_size=8,
            microbatch_size=4,
        )
    return c


def record_prediction(run, summary, name, predictions, raw, task, metadata):
    predictions = np.asarray(predictions).copy()
    if task == "energy":
        predictions[:, 1:] = 0
    if task == "position":
        predictions[:, 0] = 0
    path = run / "predictions" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(
            f, predictions=predictions, targets=raw["targets"], source_ids=raw["source_ids"]
        )
    temp.replace(path)
    summary["predictions"][path.relative_to(run).as_posix()] = dict(
        **metadata,
        task=task,
        metrics=data.aggregate_metrics(raw["targets"], predictions, task),
        subgroups=data.task_subgroups(raw["targets"], predictions, task),
        sha256=io_tools.sha(path),
    )
    record = summary["predictions"][path.relative_to(run).as_posix()]
    if task in {"position", "joint"}:
        distance = np.linalg.norm(predictions[:, 1:] - raw["targets"][:, 1:], axis=1)
        record["position_tails"] = {
            "p95": float(np.quantile(distance, 0.95)),
            "p99": float(np.quantile(distance, 0.99)),
            "maximum": float(distance.max()),
            "above_one_count": int((distance > 1).sum()),
            "above_ten_count": int((distance > 10).sum()),
        }
    if task in {"energy", "joint"}:
        record["nonpositive_energy_count"] = int((predictions[:, 0] <= 0).sum())


def run(workspace, output, device="cuda", smoke=False, backup=None):
    workspace = Path(workspace)
    output = io_tools.external(output)
    raw, input_manifest = io_tools.load_raw(workspace, smoke)
    protocol = json.loads((workspace / "configs/confirmation.json").read_text())
    config = effective(protocol, smoke)
    input_identity = io_tools.digest(io_tools.canonical(input_manifest))
    identity = dict(input_identity=input_identity, protocol=config, smoke=smoke)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        if (
            summary["input_identity"] != input_identity
            or summary["smoke"] != smoke
            or summary["protocol"] != protocol
        ):
            raise ValueError("Use a separate run for changed inputs or settings")
    else:
        if any(output.iterdir()):
            raise ValueError("Unrecognized existing output directory")
        summary = dict(
            schema_version=1,
            input_identity=input_identity,
            protocol=protocol,
            smoke=smoke,
            test_used=False,
            complete=False,
            cases={},
            predictions={},
            references={},
            counts={k: len(v["targets"]) for k, v in raw.items()},
            measurement_noise_fixed=True,
        )
    io_tools.write_json(output / "protocol.json", protocol)
    io_tools.write_json(output / "summary.json", summary)
    prepared = {}
    references_built = set()
    for case in cases(protocol):
        regime = case["regime"]
        name = case["name"]
        if regime not in prepared:
            prepared.clear()  # bound memory; cases are grouped by readout regime
            prepared[regime] = data.prepare(raw["train"], raw["validation"], regime, config)
        train, validation, stats, references = prepared[regime]
        seeds = [config["validation_noise_seed"]] + (
            [] if regime == "clean" else config["validation_repeat_seeds"]
        )
        if regime not in references_built:
            for noise_seed in seeds:
                _, window = data.evaluate_inputs(
                    raw["validation"], regime, noise_seed, config, stats
                )
                for reference, p in data.reference_predictions(window, references).items():
                    task = "energy" if reference == "quadratic" else "joint"
                    record_prediction(
                        output,
                        summary,
                        f"{regime}_{reference}_{noise_seed}",
                        p,
                        raw["validation"],
                        task,
                        dict(
                            kind="reference",
                            reference=reference,
                            regime=regime,
                            noise_seed=noise_seed,
                        ),
                    )
            summary["references"][regime] = dict(calibrations=references, statistics=stats)
            references_built.add(regime)
            io_tools.write_json(output / "summary.json", summary)
        path = output / "cases" / name / "checkpoint.zip"
        parent = None
        if case["parent"] is not None:
            parent_state = checkpoint_load(output / "cases" / case["parent"] / "checkpoint.zip")
            pm = models.LocalMaskedModel()
            pm.load_state_dict(parent_state["model"])
            parent = models.encoder_state(pm)
        used = sum(r["train_seconds"] + r["validation_seconds"] for r in summary["cases"].values())
        # A committed checkpoint can be newer than a summary after interruption.
        previous = summary["cases"].get(name, {})
        current = checkpoint_load(path) if path.exists() else None
        if current:
            used += (
                current["train_seconds"]
                + current["validation_seconds"]
                - previous.get("train_seconds", 0)
                - previous.get("validation_seconds", 0)
            )
        remaining = config["compute_ceiling_seconds"] - used
        state = train_phase(
            case,
            train,
            validation,
            stats,
            config,
            path,
            identity,
            device,
            parent,
            None if backup is None else Path(backup) / "cases" / name / "checkpoint.zip",
            remaining_seconds=remaining,
        )
        total = (
            config["pretraining_updates"] if case["phase"] == "pretraining" else config["updates"]
        )
        record = {
            k: state[k]
            for k in [
                "update",
                "history",
                "train_seconds",
                "validation_seconds",
                "hardware",
                "transfer",
                "peak_gpu_bytes",
            ]
        }
        record.update(
            case=case, complete=state["update"] == total, checkpoint_sha256=io_tools.sha(path)
        )
        if case["phase"] == "pretraining":
            pm = models.LocalMaskedModel()
            pm.load_state_dict(state["model"])
            record.update(
                encoder_sha256=tensor_hash(models.encoder_state(pm)), mask_audit=state["mask_audit"]
            )
            from .reporting import reconstruction_figure

            reconstruction_figure(
                pm,
                train["inputs"][:3],
                stats,
                output / "figures" / f"{name}.png",
                {**config, "mask_ratio": state["mask_audit"]["ratio"]},
            )
        elif state["best"] is not None:
            model, _ = make_model(case, stats, device, parent)
            model.load_state_dict(state["best"]["model"])
            record["selected"] = {k: v for k, v in state["best"].items() if k != "model"}
            record["parameters"] = sum(p.numel() for p in model.parameters())
            unused = "position" if case["task"] == "energy" else "energy"
            record["active_parameters"] = sum(
                p.numel()
                for name, p in model.named_parameters()
                if not name.startswith(f"network.{unused}.")
            )
            for noise_seed in seeds:
                v, _ = data.evaluate_inputs(raw["validation"], regime, noise_seed, config, stats)
                record_prediction(
                    output,
                    summary,
                    f"{name}_{noise_seed}",
                    predict(model, v, device),
                    raw["validation"],
                    case["task"],
                    dict(kind="network", case=name, regime=regime, noise_seed=noise_seed),
                )
        summary["cases"][name] = record
        summary["complete"] = len(summary["cases"]) == len(cases(protocol)) and all(
            r["complete"] for r in summary["cases"].values()
        )
        io_tools.write_json(output / "summary.json", summary)
        if backup is not None:
            # Checkpoint is the authoritative resume state; smaller outputs are rebuilt if needed.
            io_tools.copy_verified(output / "protocol.json", Path(backup) / "protocol.json")
            io_tools.copy_verified(output / "summary.json", Path(backup) / "summary.json")
        if not record["complete"]:
            print(
                "Paused at the cumulative compute ceiling. Export partial results for review.",
                flush=True,
            )
            break
    from .reporting import report

    report(output)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--backup", type=Path)
    a = p.parse_args()
    result = run(a.workspace, a.output, a.device, a.smoke, a.backup)
    print(
        json.dumps(
            dict(complete=result["complete"], cases=len(result["cases"]), smoke=result["smoke"])
        )
    )


if __name__ == "__main__":
    main()
