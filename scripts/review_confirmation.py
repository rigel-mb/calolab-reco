"""Verify a returned archive and derive small, validation-only review artifacts."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calolab_reco.confirmation.data import observed_window, readout
from calolab_reco.confirmation.reporting import timing
from calolab_reco.confirmation.transport import read_archive, verify_results

LABELS = {"cnn": "CNN", "transformer": "Direct Transformer", "finetuning": "Pretrained Transformer"}
COLORS = {"cnn": "#3975a8", "transformer": "#1c8778", "finetuning": "#985ba5"}


def family(record):
    case = record["case"]
    return "finetuning" if case["phase"] == "finetuning" else case["architecture"]


def bounds(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def review(bundle, result, output):
    verification = verify_results(bundle, result)
    if not verification["complete"] or verification["smoke"]:
        raise ValueError("The public review requires a complete scientific run")
    result_manifest, files = read_archive(result, "confirmation_results")
    input_manifest, input_files = read_archive(bundle, "confirmation_input")
    summary = json.loads(files["summary.json"])
    protocol = summary["protocol"]
    primary = protocol["primary_regime"]
    noise_seed = protocol["validation_noise_seed"]
    cases = summary["cases"]
    records = []
    position_failures = {}
    for name, record in summary["predictions"].items():
        entry = {k: v for k, v in record.items() if k != "sha256"}
        if record["kind"] == "network":
            case = cases[record["case"]]
            entry.update(family=family(case), training_seed=case["case"]["seed"])
        else:
            entry.update(family=record["reference"], training_seed=None)
        with np.load(io.BytesIO(files[name]), allow_pickle=False) as archive:
            truth, prediction = archive["targets"], archive["predictions"]
        tail = {}
        if record["task"] in {"energy", "joint"}:
            rel = np.abs((prediction[:, 0] - truth[:, 0]) / truth[:, 0])
            tail["energy_absolute_relative_error_gt_20_percent"] = int((rel > 0.2).sum())
            tail["energy_absolute_relative_error_p99"] = float(np.quantile(rel, 0.99))
            tail["nonpositive_energy_count"] = int((prediction[:, 0] <= 0).sum())
        if record["task"] in {"position", "joint"}:
            dist = np.linalg.norm(prediction[:, 1:] - truth[:, 1:], axis=1)
            tail["position_distance_gt_1"] = int((dist > 1).sum())
            tail["position_distance_gt_10"] = int((dist > 10).sum())
            tail["position_distance_max"] = float(dist.max())
            tail["position_distance_p99"] = float(np.quantile(dist, 0.99))
            tail["position_distance_rmse"] = float(np.sqrt(np.mean(dist**2)))
            if record["regime"] == primary:
                position_failures.setdefault(record["noise_seed"], []).append(dist > 10)
            tail["position_catastrophe_energy_range_gev"] = (
                [float(truth[dist > 10, 0].min()), float(truth[dist > 10, 0].max())]
                if (dist > 10).any()
                else None
            )
        entry["tails"] = tail
        records.append(entry)
    # Post hoc diagnostic of a shared failure, without changing inputs or scores.
    anchor_checks = []
    with np.load(io.BytesIO(input_files["data/validation.npz"]), allow_pickle=False) as raw:
        deposits, targets = raw["deposits"], raw["targets"]
    for draw, failures in sorted(position_failures.items()):
        measured = readout(deposits, primary, draw, protocol["noise"])
        anchors = observed_window(measured)["anchors"]
        distances = np.linalg.norm(anchors - targets[:, 1:], axis=1)
        far = distances > 10
        anchor_checks.append(
            {
                "noise_seed": draw,
                "anchor_distance_gt_10": int(far.sum()),
                "fraction": float(far.mean()),
                "max_anchor_distance": float(distances.max()),
                "energy_range_gev": [float(targets[far, 0].min()), float(targets[far, 0].max())]
                if far.any()
                else None,
                "all_position_estimators_fail_on_exactly_these_events": all(
                    np.array_equal(f, far) for f in failures
                ),
                "estimators_checked": len(failures),
            }
        )
    main = []
    for fam in LABELS:
        row = {"family": fam}
        for task, metric in [("energy", "energy_mare"), ("position", "position_distance_median")]:
            selected = [
                r
                for r in records
                if r["regime"] == primary
                and r["family"] == fam
                and r["noise_seed"] == noise_seed
                and r["task"] == task
            ]
            row[task] = bounds([r["metrics"][metric] for r in selected])
            row[task]["by_seed"] = {str(r["training_seed"]): r["metrics"][metric] for r in selected}
        main.append(row)
    times = timing(summary)
    for row in times:
        prefix = f"{primary}_{row['seed']}"
        direct = [cases[prefix + f"_transformer_{task}"] for task in ("energy", "position")]
        row["direct_pair_train_seconds"] = sum(r["train_seconds"] for r in direct)
        row["direct_pair_train_validation_seconds"] = sum(
            r["train_seconds"] + r["validation_seconds"] for r in direct
        )
        row["combined_first_target_seconds"] = {
            "direct": sum(r["direct_first_seconds"] for r in row["tasks"].values()),
            "finetuning": sum(r["finetuning_first_seconds"] for r in row["tasks"].values()),
            "pretraining_plus_finetuning": row["pretraining_train_seconds"]
            + sum(r["finetuning_first_seconds"] for r in row["tasks"].values()),
        }
    report = dict(
        verification=verification,
        input_metadata=input_manifest["metadata"],
        result_metadata=result_manifest["metadata"],
        counts=summary["counts"],
        primary_noise_seed=noise_seed,
        training_seeds=protocol["training_seeds"],
        main_accuracy=main,
        time_to_target=times,
        train_seconds=sum(r["train_seconds"] for r in cases.values()),
        validation_seconds=sum(r["validation_seconds"] for r in cases.values()),
        hardware=next(iter(cases.values()))["hardware"],
        mask_audits={n: r["mask_audit"] for n, r in cases.items() if "mask_audit" in r},
        parameter_counts={
            n: {k: r[k] for k in ("parameters", "active_parameters")}
            for n, r in cases.items()
            if "parameters" in r
        },
        records=records,
        anchor_checks=anchor_checks,
        interpretation="Validation only; seed range is descriptive, not a confidence interval. "
        "Additional noise draws reuse the same events and trained weights. "
        "Subgroup and tail checks are post hoc diagnostics, not new selection rules.",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "learning.png").write_bytes(files["figures/learning.png"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, task, metric, factor in zip(
        axes,
        ("energy", "position"),
        ("energy_mare", "position_distance_median"),
        (100, 1),
        strict=True,
    ):
        for x, row in enumerate(main):
            vals = list(row[task]["by_seed"].values())
            ax.scatter(
                x + np.linspace(-0.08, 0.08, 3),
                np.array(vals) * factor,
                color=COLORS[row["family"]],
                s=48,
                zorder=3,
            )
            ax.plot(
                [x - 0.2, x + 0.2],
                [row[task]["mean"] * factor] * 2,
                color=COLORS[row["family"]],
                linewidth=2,
            )
        for ref, style in (
            [("affine", ":"), ("quadratic", "--")] if task == "energy" else [("periodic", "--")]
        ):
            r = next(
                r
                for r in records
                if r["regime"] == primary and r["family"] == ref and r["noise_seed"] == noise_seed
            )
            ax.axhline(
                r["metrics"][metric] * factor,
                color="#4b5563",
                linestyle=style,
                label=f"{ref.capitalize()} reference",
            )
        ax.set_xticks(range(3), ["CNN", "Direct\nTransformer", "Pretrained\nTransformer"])
        ax.set_ylabel(
            "Energy MARE (%)" if task == "energy" else "Median position error (stored units)"
        )
        ax.set_title("Energy" if task == "energy" else "Position")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Noise + 50 MeV cell cut: three training seeds, 5,947 validation events\n"
        "Dots: individual runs; short lines: seed means; lower is better",
        fontsize=11,
    )
    fig.savefig(output / "accuracy.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, task in zip(axes, ("energy", "position"), strict=True):
        for j, (key, label, color) in enumerate(
            [
                ("direct_first_seconds", "Direct training", COLORS["transformer"]),
                ("finetuning_first_seconds", "Fine-tuning only", COLORS["finetuning"]),
                ("including_shared_pretraining_seconds", "Pretraining + fine-tuning", "#bc7952"),
            ]
        ):
            ax.bar(
                np.arange(3) + (j - 1) * 0.24,
                [r["tasks"][task][key] / 60 for r in times],
                width=0.23,
                label=label,
                color=color,
            )
        ax.set_xticks(range(3), [str(r["seed"]) for r in times], fontsize=8)
        ax.set_ylabel("Active training minutes to target")
        ax.set_title(task.capitalize())
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Reach the same task/seed's selected direct checkpoint, with 2% tolerance\n"
        "First recorded crossing; validation every 220 updates",
        fontsize=11,
    )
    fig.savefig(output / "time_to_target.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {k: report[k] for k in ("main_accuracy", "train_seconds", "validation_seconds")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.bundle, args.result, args.output)
