"""Freeze selected estimators, then evaluate the explicitly released test sample."""

from __future__ import annotations

import argparse
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from calolab_reco.data import SPLIT_NAMES, load_prepared, verify_prepared
from calolab_reco.pilot import configure

from . import data, training, transport


def load_json(path):
    return json.loads(Path(path).read_text())


def freeze(bundle, result, data_root, output):
    """Only metadata and previously used train/validation are read at this gate."""
    if output.exists():
        raise FileExistsError("Keep the original freeze; do not overwrite it")
    verified = transport.verify_results(bundle, result)
    if not verified["complete"] or verified["smoke"]:
        raise ValueError("Final evaluation needs the complete scientific return")
    manifest, files = transport.read_archive(result, "confirmation_results")
    summary = json.loads(files["summary.json"])
    input_manifest, _ = transport.validate_input(bundle)
    prepared_path = data_root / "derived/audit_v1/manifest.json"
    prepared = load_json(prepared_path)
    scientific_code = {
        name: transport.sha(transport.ROOT / name)
        for name in input_manifest["files"]
        if name.startswith("src/") and name.endswith(".py")
    }
    if any(input_manifest["files"][name] != h for name, h in scientific_code.items()):
        raise ValueError("Scientific source changed since the frozen training bundle")
    specification = dict(
        schema_version=1,
        created_utc=datetime.now(UTC).isoformat(),
        stage="final_test_evaluation",
        input_bundle_sha256=transport.sha(bundle),
        result_archive_sha256=transport.sha(result),
        input_identity=manifest["metadata"]["input_identity"],
        prepared_manifest_sha256=transport.sha(prepared_path),
        prepared_file_sha256=prepared["output_sha256"],
        counts=prepared["split_counts"],
        protocol=summary["protocol"],
        test_noise_seeds=[20261010, 20261011, 20261012],
        selection="Use every validation-selected specialist; no test-based seed selection",
        selected={
            name: dict(
                case=r["case"],
                checkpoint_sha256=r["checkpoint_sha256"],
                selected_update=r["selected"]["update"],
            )
            for name, r in summary["cases"].items()
            if r["case"]["phase"] != "pretraining"
        },
        reference_sha256=transport.digest(transport.canonical(summary["references"])),
        scientific_code_sha256=scientific_code,
        evaluator_sha256=transport.sha(Path(__file__)),
        metrics=["energy_mare", "position_distance_median"],
        diagnostics="All existing metrics and energy/index strata; nonpositive energy; "
        "position p95/p99/RMSE and errors above 10 stored units; no clipping or removal",
        bootstrap=dict(
            seed=20261020,
            replicates=1000,
            confidence=0.95,
            interpretation="Paired events conditional on each fixed model, primary noise draw. "
            "Not training-seed uncertainty or simultaneous confidence intervals.",
        ),
        no_retraining=True,
        no_refitting=True,
    )
    transport.write_json(output, specification)
    return specification


def check_raw(raw):
    x, y, ids = (raw[k] for k in ("deposits", "targets", "source_ids"))
    if x.shape != (len(y), 30, 85) or y.shape != (len(x), 3) or ids.shape != (len(x), 2):
        raise ValueError("Unexpected final sample shapes")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or (x < 0).any()
        or (y[:, 0] <= 0).any()
    ):
        raise ValueError("Invalid final sample values; do not silently remove them")
    if len(set(map(tuple, ids.tolist()))) != len(ids):
        raise ValueError("Duplicate test source IDs")


def extra_metrics(targets, prediction, task):
    y, p = targets.astype(np.float64), prediction.astype(np.float64)
    result = {}
    if task in ("energy", "joint"):
        result["nonpositive_energy_count"] = int((p[:, 0] <= 0).sum())
        result["energy_absolute_relative_p99"] = float(
            np.quantile(abs(p[:, 0] / y[:, 0] - 1), 0.99)
        )
    if task in ("position", "joint"):
        d = np.linalg.norm(p[:, 1:] - y[:, 1:], axis=1)
        result.update(
            position_p95=float(np.quantile(d, 0.95)),
            position_p99=float(np.quantile(d, 0.99)),
            position_rmse=float(np.sqrt(np.mean(d**2))),
            position_gt_10=int((d > 10).sum()),
            position_max=float(d.max()),
        )
    return result


def paired_interval(y, direct, fine, task, settings):
    """Positive difference favors fine-tuning; sample events, not training runs."""
    if task == "energy":
        a, b = (np.abs(p[:, 0] / y[:, 0] - 1) for p in (direct, fine))
        reduce = np.mean
    else:
        a, b = (np.linalg.norm(p[:, 1:] - y[:, 1:], axis=1) for p in (direct, fine))
        reduce = np.median
    rng = np.random.default_rng(settings["seed"])
    diffs = []
    for _ in range(settings["replicates"]):
        indices = rng.integers(0, len(y), len(y))
        diffs.append(float(reduce(a[indices]) - reduce(b[indices])))
    alpha = (1 - settings["confidence"]) / 2
    return dict(
        direct_minus_finetuned=float(reduce(a) - reduce(b)),
        interval=np.quantile(diffs, [alpha, 1 - alpha]).tolist(),
    )


def validate_inputs(bundle, result, data_root, frozen):
    for path, key in [(bundle, "input_bundle_sha256"), (result, "result_archive_sha256")]:
        if transport.sha(path) != frozen[key]:
            raise ValueError(f"Changed frozen artifact: {key}")
    if transport.sha(Path(__file__)) != frozen["evaluator_sha256"]:
        raise ValueError("Evaluator changed after freeze")
    for name, h in frozen["scientific_code_sha256"].items():
        if transport.sha(transport.ROOT / name) != h:
            raise ValueError(f"Scientific code changed after freeze: {name}")
    root = data_root / "derived/audit_v1"
    if transport.sha(root / "manifest.json") != frozen["prepared_manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    prepared = verify_prepared(data_root)
    if prepared["output_sha256"] != frozen["prepared_file_sha256"]:
        raise ValueError("Prepared file identities changed")
    _, inputs = transport.validate_input(bundle)
    assignment = np.load(root / "split.npy", allow_pickle=False)
    arrays = {
        k: np.load(root / f"{k}.npy", mmap_mode="r", allow_pickle=False)
        for k in ("deposits", "targets", "source_ids")
    }
    used_ids = set()
    for split in ("train", "validation"):
        indices = np.flatnonzero(assignment == SPLIT_NAMES[split])
        with np.load(io.BytesIO(inputs[f"data/{split}.npz"]), allow_pickle=False) as z:
            for key, original in arrays.items():
                bundled = z[key]
                if len(bundled) != len(indices):
                    raise ValueError("Prepared and training split sizes differ")
                for start in range(0, len(indices), 1024):
                    if not np.array_equal(
                        original[indices[start : start + 1024]], bundled[start : start + 1024]
                    ):
                        raise ValueError("Prepared and training values/order differ")
            used_ids.update(map(tuple, z["source_ids"].tolist()))
    raw = load_prepared(data_root, "test", allow_test=True)
    check_raw(raw)
    if len(raw["targets"]) != frozen["counts"]["test"]:
        raise ValueError("Test count differs")
    if used_ids & set(map(tuple, raw["source_ids"].tolist())):
        raise ValueError("Test overlaps train/validation")
    return raw


def evaluate(bundle, result, data_root, freeze_path, output, allow_test=False):
    if not allow_test:
        raise PermissionError("Final test evaluation requires explicit --allow-test")
    frozen = load_json(freeze_path)
    output = transport.external(output)
    if output.exists():
        raise FileExistsError("Preserve previous evaluation output; do not overwrite it")
    raw = validate_inputs(bundle, result, data_root, frozen)
    _, files = transport.read_archive(result, "confirmation_results")
    source = json.loads(files["summary.json"])
    if transport.digest(transport.canonical(source["references"])) != frozen["reference_sha256"]:
        raise ValueError("Reference calibration changed")
    output.mkdir(parents=True)
    transport.write_json(output / "frozen_protocol.json", frozen)
    summary = dict(
        complete=False,
        test_used=True,
        count=len(raw["targets"]),
        frozen_protocol_sha256=transport.sha(freeze_path),
        source_ids_sha256=transport.digest(raw["source_ids"].tobytes()),
        hardware=configure("cpu", frozen["protocol"]["training_seeds"][0]),
        records={},
        anchor_checks=[],
        bootstrap=[],
    )
    predictions = {}

    def record(name, pred, task, details):
        if pred.shape != raw["targets"].shape or not np.isfinite(pred).all():
            raise ValueError("Invalid model predictions")
        path = output / "predictions" / (name + ".npz")
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            path, predictions=pred, targets=raw["targets"], source_ids=raw["source_ids"]
        )
        summary["records"][name] = dict(
            **details,
            task=task,
            sha256=transport.sha(path),
            metrics=data.aggregate_metrics(raw["targets"], pred, task),
            subgroups=data.task_subgroups(raw["targets"], pred, task),
            tails=extra_metrics(raw["targets"], pred, task),
        )
        predictions[name] = pred
        transport.write_json(output / "summary.json", summary)
        print(f"Evaluated {len(summary['records'])}/91: {name}", flush=True)

    for regime, reference in source["references"].items():
        noise_seeds = (
            frozen["test_noise_seeds"][:1] if regime == "clean" else frozen["test_noise_seeds"]
        )
        for noise_seed in noise_seeds:
            split, window = data.evaluate_inputs(
                raw, regime, noise_seed, source["protocol"], reference["statistics"]
            )
            far = np.linalg.norm(window["anchors"] - raw["targets"][:, 1:], axis=1) > 10
            summary["anchor_checks"].append(
                dict(
                    regime=regime,
                    noise_seed=noise_seed,
                    anchor_distance_gt_10=int(far.sum()),
                    incident_energy_range_gev=[
                        float(raw["targets"][far, 0].min()),
                        float(raw["targets"][far, 0].max()),
                    ]
                    if far.any()
                    else None,
                )
            )
            for ref, pred in data.reference_predictions(window, reference["calibrations"]).items():
                record(
                    f"{regime}_{ref}_{noise_seed}",
                    pred,
                    "energy" if ref == "quadratic" else "joint",
                    dict(kind="reference", family=ref, regime=regime, noise_seed=noise_seed),
                )
            for name, selected in frozen["selected"].items():
                case = selected["case"]
                if case["regime"] != regime:
                    continue
                payload = files[f"cases/{name}/checkpoint.zip"]
                if transport.digest(payload) != selected["checkpoint_sha256"]:
                    raise ValueError("Checkpoint changed")
                state = training.checkpoint_load(io.BytesIO(payload))
                if state["best"]["update"] != selected["selected_update"]:
                    raise ValueError("Wrong selected update")
                if state["identity"]["statistics"] != reference["statistics"]:
                    raise ValueError("Checkpoint preprocessing differs")
                configure("cpu", case["seed"])
                model, _ = training.make_model(case, reference["statistics"], "cpu")
                model.load_state_dict(state["best"]["model"])
                pred = training.predict(model, split, "cpu")
                record(
                    f"{name}_{noise_seed}",
                    pred,
                    case["task"],
                    dict(
                        kind="network",
                        family="finetuning"
                        if case["phase"] == "finetuning"
                        else case["architecture"],
                        regime=regime,
                        noise_seed=noise_seed,
                        training_seed=case["seed"],
                        selected_update=selected["selected_update"],
                        checkpoint_sha256=selected["checkpoint_sha256"],
                    ),
                )
    primary = source["protocol"]["primary_regime"]
    draw = frozen["test_noise_seeds"][0]
    for seed in source["protocol"]["training_seeds"]:
        for task in ("energy", "position"):
            fine = predictions[f"{primary}_{seed}_finetuning_{task}_{draw}"]
            for label, key in [
                ("direct_transformer", f"{primary}_{seed}_transformer_{task}_{draw}"),
                (
                    "reference",
                    f"{primary}_{'quadratic' if task == 'energy' else 'periodic'}_{draw}",
                ),
            ]:
                interval = paired_interval(
                    raw["targets"].astype(np.float64),
                    predictions[key].astype(np.float64),
                    fine.astype(np.float64),
                    task,
                    frozen["bootstrap"],
                )
                summary["bootstrap"].append(
                    dict(training_seed=seed, task=task, comparator=label, **interval)
                )
    if len(summary["records"]) != 91:
        raise ValueError("Incomplete final evaluation")
    summary["complete"] = True
    transport.write_json(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "evaluate"))
    for name in ("bundle", "result", "data-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--allow-test", action="store_true")
    a = p.parse_args()
    if a.command == "freeze":
        r = freeze(a.bundle, a.result, a.data_root, a.output)
        print(f"Frozen {len(r['selected'])} selected models before test access")
    else:
        r = evaluate(a.bundle, a.result, a.data_root, a.freeze, a.output, a.allow_test)
        print(
            json.dumps(
                dict(complete=r["complete"], count=r["count"], predictions=len(r["records"]))
            )
        )


if __name__ == "__main__":
    main()
