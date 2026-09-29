"""Verify final predictions and render a compact, held-out-test review."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calolab_reco.confirmation import data, transport
from calolab_reco.confirmation.final_evaluation import extra_metrics

LABELS = {"cnn": "CNN", "transformer": "Direct Transformer", "finetuning": "Pretrained Transformer"}
COLORS = {"cnn": "#3975a8", "transformer": "#1c8778", "finetuning": "#985ba5"}


def review(run, freeze_path, validation_path, output):
    summary = json.loads((run / "summary.json").read_text())
    frozen = json.loads(freeze_path.read_text())
    validation = json.loads(validation_path.read_text())
    if not summary["complete"] or not summary["test_used"] or len(summary["records"]) != 91:
        raise ValueError("Need the complete final evaluation")
    if summary["frozen_protocol_sha256"] != transport.sha(freeze_path):
        raise ValueError("Frozen protocol fingerprint differs")
    ids, targets = None, None
    for name, record in summary["records"].items():
        path = run / "predictions" / (name + ".npz")
        if transport.sha(path) != record["sha256"]:
            raise ValueError("Prediction fingerprint differs")
        with np.load(path, allow_pickle=False) as a:
            y, p, i = a["targets"], a["predictions"], a["source_ids"]
        if ids is None:
            ids, targets = i, y
        if not np.array_equal(ids, i) or not np.array_equal(targets, y):
            raise ValueError("Predictions do not describe the same ordered test sample")
        if transport.digest(i.tobytes()) != summary["source_ids_sha256"]:
            raise ValueError("Test identifiers differ")
        if data.aggregate_metrics(y, p, record["task"]) != record["metrics"]:
            raise ValueError("Final metrics do not reproduce")
        if extra_metrics(y, p, record["task"]) != record["tails"]:
            raise ValueError("Final tail metrics do not reproduce")
        if data.task_subgroups(y, p, record["task"]) != record["subgroups"]:
            raise ValueError("Final subgroup metrics do not reproduce")
    records = list(summary["records"].values())
    primary = frozen["protocol"]["primary_regime"]
    first_draw = frozen["test_noise_seeds"][0]
    main = []
    for family in LABELS:
        row = dict(family=family)
        for task, metric in (("energy", "energy_mare"), ("position", "position_distance_median")):
            selected = [
                r
                for r in records
                if r["family"] == family
                and r["regime"] == primary
                and r["noise_seed"] == first_draw
                and r["task"] == task
            ]
            values = [r["metrics"][metric] for r in selected]
            row[task] = dict(
                mean=float(np.mean(values)),
                min=min(values),
                max=max(values),
                by_seed={str(r["training_seed"]): r["metrics"][metric] for r in selected},
            )
        main.append(row)
    public = dict(
        **summary,
        main_accuracy=main,
        verification=dict(
            all_91_prediction_files_checked=True,
            metrics_and_subgroups_recomputed=True,
            frozen_protocol_sha256=transport.sha(freeze_path),
            source_summary_sha256=transport.sha(run / "summary.json"),
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, task, metric, factor in zip(
        axes,
        ("energy", "position"),
        ("energy_mare", "position_distance_median"),
        (100, 1),
        strict=True,
    ):
        for j, r in enumerate(main):
            vals = np.array(list(r[task]["by_seed"].values())) * factor
            color = COLORS[r["family"]]
            ax.scatter(j + np.linspace(-0.08, 0.08, len(vals)), vals, c=color, s=42, zorder=3)
            ax.plot([j - 0.18, j + 0.18], [r[task]["mean"] * factor] * 2, c=color, lw=2)
            val = next(v for v in validation["main_accuracy"] if v["family"] == r["family"])
            ax.scatter(
                j + 0.25,
                val[task]["mean"] * factor,
                marker="D",
                s=36,
                facecolors="none",
                edgecolors="#555",
                label="Validation mean" if j == 0 else None,
            )
        ref_family = "quadratic" if task == "energy" else "periodic"
        ref = next(
            r
            for r in records
            if r["family"] == ref_family
            and r["regime"] == primary
            and r["noise_seed"] == first_draw
        )
        ax.axhline(
            ref["metrics"][metric] * factor,
            color="#4b5563",
            linestyle="--",
            label=f"Test {ref_family} reference",
        )
        ax.set_xticks(range(3), ["CNN", "Direct\nTransformer", "Pretrained\nTransformer"])
        ax.set_ylabel(
            "Energy MARE (%)" if task == "energy" else "Median position error (stored units)"
        )
        ax.set_title(task.capitalize())
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Held-out test: 5,935 events, noise + 50 MeV cell cut\n"
        "Dots: three training seeds; lines: mean scores; lower is better",
        fontsize=11,
    )
    fig.savefig(output / "accuracy.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    families = ["periodic", "transformer", "finetuning"]
    names = ["Periodic\nreference", "Direct\nTransformer", "Pretrained\nTransformer"]
    for ax, metric, title in zip(
        axes,
        ("position_p99", "position_rmse"),
        ("Ordinary tail: 99th percentile", "Extreme failures dominate RMSE"),
        strict=True,
    ):
        for j, fam in enumerate(families):
            sel = [
                r
                for r in records
                if r["family"] == fam
                and r["regime"] == primary
                and r["noise_seed"] == first_draw
                and r["task"] in ("position", "joint")
            ]
            values = [r["tails"][metric] for r in sel]
            ax.bar(j, np.mean(values), width=0.55, color=COLORS.get(fam, "#68717d"))
            ax.scatter(j + np.linspace(-0.07, 0.07, len(values)), values, c="#222", s=20)
        ax.set_xticks(range(3), names)
        ax.set_ylabel("Position error (stored units)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Keep all events: seven misplaced anchors in the primary test draw\n"
        "Better reconstruction does not fix wrong-window selections",
        fontsize=11,
    )
    fig.savefig(output / "tails.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            dict(count=summary["count"], main_accuracy=main, verification=public["verification"]),
            indent=2,
        )
    )


def energy_median_review(run, freeze_path):
    """Post-hoc descriptive metric check on fixed predictions; never select a model."""
    summary = json.loads((run / "summary.json").read_text())
    frozen = json.loads(freeze_path.read_text())
    if not summary["complete"] or not summary["test_used"]:
        raise ValueError("Need the completed held-out evaluation")
    if summary["frozen_protocol_sha256"] != transport.sha(freeze_path):
        raise ValueError("Frozen protocol fingerprint differs")
    rows = []
    common_ids, common_targets = None, None
    for name, record in summary["records"].items():
        if record["task"] not in ("energy", "joint"):
            continue
        path = run / "predictions" / (name + ".npz")
        if transport.sha(path) != record["sha256"]:
            raise ValueError("Prediction fingerprint differs")
        with np.load(path, allow_pickle=False) as arrays:
            ids = arrays["source_ids"]
            targets = arrays["targets"].astype(np.float64)
            predictions = arrays["predictions"].astype(np.float64)
        if common_ids is None:
            common_ids, common_targets = ids.copy(), targets.copy()
        if not np.array_equal(ids, common_ids) or not np.array_equal(targets, common_targets):
            raise ValueError("Event ordering or targets differ")
        if transport.digest(ids.tobytes()) != summary["source_ids_sha256"]:
            raise ValueError("Test identifiers differ")
        if len(ids) != frozen["counts"]["test"]:
            raise ValueError("Test count differs")
        metrics = data.aggregate_metrics(targets, predictions, record["task"])
        if metrics != record["metrics"]:
            raise ValueError("Original metrics do not reproduce")
        error = np.abs((predictions[:, 0] - targets[:, 0]) / targets[:, 0])
        rows.append(dict(
            name=name, family=record["family"], regime=record["regime"],
            training_seed=record.get("training_seed"), noise_seed=record["noise_seed"],
            prediction_sha256=record["sha256"],
            mean_absolute_relative_error_pct=float(error.mean() * 100),
            median_absolute_relative_error_pct=float(np.median(error) * 100),
        ))
    return dict(
        analysis="Post-hoc energy median, no retraining or model selection",
        formula="100 * median(abs((prediction - truth) / truth))",
        count=len(common_ids), frozen_protocol_sha256=transport.sha(freeze_path),
        source_summary_sha256=transport.sha(run / "summary.json"),
        source_ids_sha256=summary["source_ids_sha256"], records=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "freeze"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("validation", "output"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--energy-median-only", action="store_true",
                   help="Print supplementary metrics without rewriting frozen reports")
    a = p.parse_args()
    if a.energy_median_only:
        print(json.dumps(energy_median_review(a.run, a.freeze), indent=2, allow_nan=False))
    else:
        if a.validation is None or a.output is None:
            p.error("--validation and --output are required for the full review")
        review(a.run, a.freeze, a.validation, a.output)
