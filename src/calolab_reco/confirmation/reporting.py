"""Compact interpretation and figures; results remain validation-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import sample_mask
from .transport import write_json


def reconstruction_figure(model, inputs, stats, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = sample_mask(
        inputs, config["mask_ratio"], torch.Generator().manual_seed(config["mask_seed"])
    )
    model.eval()
    with torch.no_grad():
        prediction = model(inputs, mask).reshape(-1, 7, 7).numpy()
    scale = stats["linear_scale_mev"] / 1000
    fig, axes = plt.subplots(len(inputs), 3, figsize=(8, 2.5 * len(inputs)), squeeze=False)
    for i in range(len(inputs)):
        target = inputs[i, 0].numpy() * scale
        hidden = np.ma.array(target, mask=mask[i].reshape(7, 7).numpy())
        recovered = np.where(mask[i].reshape(7, 7).numpy(), prediction[i] * scale, target)
        for j, (values, title) in enumerate(
            [
                (target, "Measured"),
                (hidden, "Hidden cells"),
                (recovered, "Observed + reconstructed\nhidden cells"),
            ]
        ):
            image = axes[i, j].imshow(
                values, vmin=0, vmax=max(float(target.max()), 1e-6), cmap="viridis"
            )
            axes[i, j].set_title(title)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            fig.colorbar(image, ax=axes[i, j], label="GeV", fraction=0.05)
    fig.suptitle("Training examples: reconstruct measured values, not clean truth")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def timing(summary):
    result = []
    cases = summary["cases"]
    protocol = summary["protocol"]
    for seed in protocol["training_seeds"]:
        prefix = f"{protocol['primary_regime']}_{seed}"
        pre = cases.get(prefix + "_pretraining")
        if pre is None or not pre["complete"]:
            continue
        pair = dict(seed=seed, pretraining_train_seconds=pre["train_seconds"], tasks={})
        for task, metric in [("energy", "energy_mare"), ("position", "position_distance_median")]:
            direct = cases.get(prefix + f"_transformer_{task}")
            fine = cases.get(prefix + f"_finetuning_{task}")
            if not direct or not fine or not direct.get("selected") or not fine.get("selected"):
                continue
            target = direct["selected"]["metrics"][metric] * (
                1 + protocol["time_target_relative_tolerance"]
            )

            def first(record, metric=metric, target=target):
                hits = [h for h in record["history"] if h["metrics"][metric] <= target]
                return hits[0]["train_seconds"] if hits else None

            ft = first(fine)
            pair["tasks"][task] = dict(
                target=target,
                direct_first_seconds=first(direct),
                finetuning_first_seconds=ft,
                including_shared_pretraining_seconds=None
                if ft is None
                else ft + pre["train_seconds"],
            )
        fine_cases = [cases.get(prefix + f"_finetuning_{task}") for task in ("energy", "position")]
        if all(fine_cases):
            pair["combined_train_seconds"] = pre["train_seconds"] + sum(
                r["train_seconds"] for r in fine_cases
            )
            pair["combined_train_validation_seconds"] = (
                pre["train_seconds"]
                + pre["validation_seconds"]
                + sum(r["train_seconds"] + r["validation_seconds"] for r in fine_cases)
            )
        result.append(pair)
    return result


def report(run):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    (run / "figures").mkdir(exist_ok=True)
    timings = timing(summary)
    summary["time_to_target"] = timings
    write_json(run / "summary.json", summary)
    primary = summary["protocol"]["validation_noise_seed"]
    lines = [
        "# Local reconstruction confirmation",
        "",
        "**Tiny smoke check, not scientific results.**"
        if summary["smoke"]
        else "**Validation results; the final test remains reserved.**",
        "",
        (
            f"Completed: {sum(c['complete'] for c in summary['cases'].values())} "
            f"phases. All complete: {summary['complete']}."
        ),
        "",
        "## Selected task checkpoints",
        "",
        (
            "| Case | Update | Energy MARE (%) | Position median (stored "
            "units) | Train + validation (s) |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, r in summary["cases"].items():
        if not r.get("selected"):
            continue
        m = r["selected"]["metrics"]
        e = m.get("energy_mare")
        p = m.get("position_distance_median")
        lines.append(
            (
                f"| {name} | {r['selected']['update']} | {100 * e:.3f} | - | "
                f"{r['train_seconds'] + r['validation_seconds']:.1f} |"
            )
            if e is not None
            else (
                f"| {name} | {r['selected']['update']} | - | {p:.5f} | "
                f"{r['train_seconds'] + r['validation_seconds']:.1f} |"
            )
        )
    lines += [
        "",
        "## Accuracy and cost",
        "",
        "![Task comparisons](figures/comparison.png)",
        "",
        "Each point is one selected training seed, not an event-bootstrap sample.",
        "References are fitted on train for each readout condition. Quadratic energy has no ridge.",
        "The clean and noise-only controls use one seed; the main condition uses three.",
        (
            "A fixed measurement-noise realization is shared by training "
            "seeds; extra noise draws are evaluation controls."
        ),
        "",
        "![Learning curves](figures/learning.png)",
        "",
        (
            "Time-to-target uses the selected direct Transformer for the "
            "same task and seed, with 2% tolerance."
        ),
        "The first qualifying validation observation is reported; finer timing is not established.",
        (
            "Pretraining is counted once in the combined two-task cost. A "
            "missing crossing means not reached."
        ),
        "",
        "```json",
        json.dumps(timings, indent=2),
        "```",
        "",
        "## Limits",
        "",
        "Use the exported prediction records for energy/region breakdowns and large-error rates.",
        (
            "Do not interpret stored position units as centimeters. Local "
            "anchoring can fail at low energy."
        ),
        (
            "Comparisons are conditional on prior exploratory selection; "
            "three seeds do not remove that limitation."
        ),
        "The shared pretrained representation is not a universal foundation model.",
    ]
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    regimes = ["clean", "noise", "noise_cut"]
    colors = {"cnn": "#3378ac", "transformer": "#258769", "finetuning": "#984ea3"}
    for panel, (task, metric) in enumerate(
        [("energy", "energy_mare"), ("position", "position_distance_median")]
    ):
        ax = axes[panel]
        scale = 100 if panel == 0 else 1
        for j, regime in enumerate(regimes):
            for family, offset in [("cnn", -0.2), ("transformer", 0), ("finetuning", 0.2)]:
                records = [
                    r
                    for r in summary["cases"].values()
                    if r["case"]["regime"] == regime
                    and r["case"]["task"] == task
                    and r.get("selected")
                    and (
                        r["case"]["phase"] == "finetuning"
                        if family == "finetuning"
                        else r["case"]["phase"] == "direct" and r["case"]["architecture"] == family
                    )
                ]
                values = [r["selected"]["metrics"][metric] * scale for r in records]
                if values:
                    ax.scatter(
                        np.full(len(values), j + offset),
                        values,
                        color=colors[family],
                        label=family if j == 2 else None,
                    )
            refs = [
                r
                for r in summary["predictions"].values()
                if r["kind"] == "reference"
                and r["regime"] == regime
                and r["noise_seed"] == primary
                and r["reference"] == ("quadratic" if panel == 0 else "periodic")
            ]
            if refs:
                ax.plot(
                    [j - 0.32, j + 0.32],
                    [refs[0]["metrics"][metric] * scale] * 2,
                    color="#555555",
                    linestyle="--",
                )
        ax.set_xticks(range(3), ["Provided deposits", "Noise", "Noise + cut"])
        ax.set_ylabel("Energy MARE (%)" if panel == 0 else "Median position error (stored units)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Local specialists; dashed lines: matched classical references")
    fig.tight_layout()
    fig.savefig(run / "figures/comparison.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    seed = summary["protocol"]["training_seeds"][0]
    for panel, (task, metric) in enumerate(
        [("energy", "energy_mare"), ("position", "position_distance_median")]
    ):
        for r in summary["cases"].values():
            c = r["case"]
            if c["seed"] != seed or c["regime"] != "noise_cut" or c["task"] != task:
                continue
            history = r["history"]
            family = "finetuning" if c["phase"] == "finetuning" else c["architecture"]
            axes[panel].plot(
                [h["train_seconds"] for h in history],
                [h["metrics"][metric] * (100 if panel == 0 else 1) for h in history],
                label=family,
                color=colors[family],
            )
        axes[panel].set_xlabel("Active supervised training seconds")
        axes[panel].set_ylabel("Energy MARE (%)" if panel == 0 else "Median position error")
        axes[panel].legend()
        axes[panel].grid(alpha=0.2)
    fig.suptitle("Primary seed: fine-tuning curve excludes pretraining cost")
    fig.tight_layout()
    fig.savefig(run / "figures/learning.png", dpi=140)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    report(p.parse_args().run)
