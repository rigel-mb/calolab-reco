"""Separate measured marginal adaptation time from hypothetical reuse economics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ("energy", "position")


def strict_break_even(pretraining_seconds: float, savings_seconds: float) -> int | None:
    """First integer N with P + N * F < N * D; a tie does not pay back."""
    if pretraining_seconds < 0 or not math.isfinite(pretraining_seconds):
        raise ValueError("Pretraining time must be finite and nonnegative")
    if not math.isfinite(savings_seconds):
        raise ValueError("Time savings must be finite")
    return None if savings_seconds <= 0 else math.floor(pretraining_seconds / savings_seconds) + 1


def build_report(source: Path) -> dict:
    raw = source.read_bytes()
    review = json.loads(raw)
    check = review["verification"]
    if not check["complete"] or check["smoke"] or not check["metrics_recomputed"]:
        raise ValueError("Use a complete, verified scientific confirmation review")
    accuracy = {row["family"]: row for row in review["main_accuracy"]}
    rows = []
    for entry in review["time_to_target"]:
        seed = entry["seed"]
        pretraining = entry["pretraining_train_seconds"]
        row = {"seed": seed, "pretraining_seconds": pretraining, "tasks": {}}
        for task in TASKS:
            t = entry["tasks"][task]
            direct, fine = t["direct_first_seconds"], t["finetuning_first_seconds"]
            if direct is None or fine is None or direct <= 0 or fine <= 0:
                raise ValueError("This analysis requires observed positive times for every target")
            saving = direct - fine
            row["tasks"][task] = {
                "validation_target": t["target"],
                "direct_seconds": direct,
                "finetuning_seconds": fine,
                "savings_seconds": saving,
                "savings_fraction": saving / direct,
                "hypothetical_strict_break_even_tasks": strict_break_even(pretraining, saving),
                "selected_direct_error": accuracy["transformer"][task]["by_seed"][str(seed)],
                "selected_finetuned_error": accuracy["finetuning"][task]["by_seed"][str(seed)],
            }
        pair_saving = sum(row["tasks"][task]["savings_seconds"] for task in TASKS)
        pair_threshold = strict_break_even(pretraining, pair_saving)
        row["hypothetical_balanced_mix"] = {
            "savings_per_energy_position_pair_seconds": pair_saving,
            "strict_break_even_pairs": pair_threshold,
            "strict_break_even_tasks_in_complete_pairs": (
                None if pair_threshold is None else pair_threshold * 2
            ),
        }
        row["actual_fixed_budget"] = {
            "updates_per_phase": 4400,
            "direct_pair_seconds": entry["direct_pair_train_seconds"],
            "finetuning_pair_seconds": entry["combined_train_seconds"] - pretraining,
            "pretraining_plus_pair_seconds": entry["combined_train_seconds"],
        }
        rows.append(row)
    mean_pre = float(np.mean([r["pretraining_seconds"] for r in rows]))
    means = {}
    for task in TASKS:
        direct = float(np.mean([r["tasks"][task]["direct_seconds"] for r in rows]))
        fine = float(np.mean([r["tasks"][task]["finetuning_seconds"] for r in rows]))
        means[task] = {
            "direct_seconds": direct,
            "finetuning_seconds": fine,
            "savings_seconds": direct - fine,
            "savings_fraction_of_mean_times": (direct - fine) / direct,
            "hypothetical_strict_break_even_tasks": strict_break_even(mean_pre, direct - fine),
        }
    pair_saving = sum(means[t]["savings_seconds"] for t in TASKS)
    pairs = strict_break_even(mean_pre, pair_saving)
    return {
        "schema_version": 1,
        "source_review_sha256": hashlib.sha256(raw).hexdigest(),
        "source_result_archive_sha256": check["archive_sha256"],
        "basis": "Validation learning histories only; no reserved-test data read",
        "timing": "Active training seconds, excluding validation, setup and I/O",
        "target": "Same task/seed selected direct error with 2% relative tolerance",
        "observation_interval_updates": 220,
        "comparison": "Retrospective first observed target crossing, not an executed stop rule",
        "break_even_definition": "Strict saving: sum(D_i - F_i) > P",
        "identical_task_formula": "floor(P / (D - F)) + 1 if D > F, otherwise no break-even",
        "means_are": "Means over three seeds with different accuracy targets; descriptive only",
        "mean_pretraining_seconds": mean_pre,
        "fixed_budget_mean_times": {
            name: float(np.mean([row["actual_fixed_budget"][name] for row in rows]))
            for name in (
                "direct_pair_seconds",
                "finetuning_pair_seconds",
                "pretraining_plus_pair_seconds",
            )
        },
        "mean_time_scenarios": means,
        "hypothetical_balanced_mix_mean_times": {
            "savings_per_energy_position_pair_seconds": pair_saving,
            "strict_break_even_pairs": pairs,
            "strict_break_even_tasks_in_complete_pairs": None if pairs is None else 2 * pairs,
        },
        "rows": rows,
        "conditions_for_reuse": [
            "A single existing pretrained encoder can initialize every added task",
            "Each added task reaches acceptable independent evaluation quality",
            "Each scenario repeats observed per-task costs on comparable hardware",
            "Training seeds are repetitions, not distinct reusable tasks",
            "No new tasks or prospective early-stopping policy were evaluated",
            "The fixed 4400-update runs themselves did not stop at target crossings",
        ],
    }


def make_figure(report: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    direct_color, fine_color = "#1c8778", "#985ba5"
    means = report["mean_time_scenarios"]
    ax = axes[0]
    for i, task in enumerate(TASKS):
        for method, offset, color in [
            ("direct", -0.17, direct_color),
            ("finetuning", 0.17, fine_color),
        ]:
            values = np.array([r["tasks"][task][method + "_seconds"] for r in report["rows"]]) / 60
            mean = values.mean()
            ax.bar(
                i + offset,
                mean,
                width=0.3,
                color=color,
                label={"direct": "Direct training", "finetuning": "Fine-tuning only"}[method]
                if i == 0
                else None,
            )
            ax.scatter(
                i + offset + np.linspace(-0.065, 0.065, len(values)),
                values,
                s=22,
                color="#202b36",
                zorder=3,
            )
            ax.text(i + offset, 0.15, f"{mean:.2f}", color="white", ha="center", weight="bold")
    ax.set_xticks([0, 1], ["Energy", "Position"])
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Active minutes to the validation target")
    ax.set_title("Measured: marginal adaptation cost", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.15)
    ax.text(
        0.5,
        -0.25,
        "Dots: 3 seeds; bars: means\nEach seed has its own direct-model target (+2%)",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    ax = axes[1]
    n = np.arange(0, 9)
    p = report["mean_pretraining_seconds"] / 60
    d = means["position"]["direct_seconds"] / 60
    f = means["position"]["finetuning_seconds"] / 60
    threshold = means["position"]["hypothetical_strict_break_even_tasks"]
    ax.plot(n, n * d, "o-", color=direct_color, label="N direct trainings")
    ax.plot(n, p + n * f, "o-", color=fine_color, label="One pretraining + N fine-tunings")
    ax.axvline(threshold, color="#64748b", linestyle=":")
    ax.annotate(
        f"First saving at N = {threshold}\nOnly if these costs repeat",
        xy=(threshold, p + threshold * f),
        xytext=(0.4, 23),
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
        fontsize=9,
    )
    ax.set_xticks(n)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Hypothetical position-like tasks")
    ax.set_ylabel("Total active training minutes")
    ax.set_title("Scenario: reuse with mean position costs", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.15)
    ax.text(
        0.5,
        -0.25,
        "These added tasks were NOT run.\nQuality and savings must hold on each new task.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    fig.suptitle(
        "Reuse can amortize pretraining, but marginal and total costs answer different questions"
    )
    fig.savefig(output / "amortization.png", dpi=150)
    plt.close(fig)


def write_markdown(report: dict, output: Path) -> None:
    p = report["mean_pretraining_seconds"] / 60
    fixed = report["fixed_budget_mean_times"]
    lines = [
        "# Fine-tuning cost and reuse",
        "",
        "**With an encoder already available, fine-tuning reaches the direct position model's",
        "validation level sooner on all three seeds.** Initial pretraining is a separate,",
        "one-off cost. Reuse can pay that cost back if enough useful tasks retain both",
        "the accuracy and the adaptation-time advantage.",
        "",
        "![Measured adaptation and conditional reuse](amortization.png)",
        "",
        "## Measured marginal time",
        "",
        "Times below are active training to the first observed validation crossing of the",
        "same task/seed direct model's selected error, allowing 2% relative tolerance.",
        "They exclude pretraining, validation, I/O and setup. The target is sampled every",
        "220 updates. The comparison uses the recorded learning curves retrospectively;",
        "it is not a tested early-stopping policy.",
        "",
        "| Seed | Task | Direct (min) | Fine-tuning (min) | Time saving |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        for task in TASKS:
            t = row["tasks"][task]
            lines.append(
                f"| {row['seed']} | {task.title()} | {t['direct_seconds'] / 60:.2f} | "
                f"{t['finetuning_seconds'] / 60:.2f} | {t['savings_fraction']:.1%} |"
            )
    for task in TASKS:
        t = report["mean_time_scenarios"][task]
        lines.append(
            f"| Seed mean | {task.title()} | {t['direct_seconds'] / 60:.2f} | "
            f"{t['finetuning_seconds'] / 60:.2f} | "
            f"{t['savings_fraction_of_mean_times']:.1%} |"
        )
    lines += [
        "",
        "These means pool timings, not predictions. Targets differ by seed: the weakest",
        "direct position run sets the easiest target and contributes the largest saving.",
        "Three repetitions do not establish a general acceleration factor.",
        "",
        f"Pretraining separately costs **{p:.2f} active minutes on average** (one Tesla T4).",
        "At the actual fixed budget of 4,400 updates, both approaches completed their",
        "full schedules. No early finish was exercised: the completed fine-tunings do",
        "not themselves demonstrate a reduced training bill. Faster target attainment",
        "would need a prospective stopping rule to realize that saving in a new run.",
        f"The full two-task schedules averaged {fixed['direct_pair_seconds'] / 60:.2f} min",
        f"direct versus {fixed['finetuning_pair_seconds'] / 60:.2f} min fine-tuning alone",
        "(approximately equal); pretraining is additional.",
        "",
        "## When would reuse pay back?",
        "",
        "Let P be pretraining time, D_i direct-training time and F_i fine-tuning time",
        "for task i at an acceptable, comparable quality level. Count pretraining once:",
        "",
        "```text",
        "Direct cost:       D_1 + D_2 + ... + D_N",
        "Reuse cost:    P + F_1 + F_2 + ... + F_N",
        "Reuse is faster when sum(D_i - F_i) > P.",
        "```",
        "",
        "If every added task repeats the same positive saving Delta = D - F, the first",
        "integer giving a strict saving is floor(P / Delta) + 1. If Delta is zero or",
        "negative, repeating that task profile never pays back the initial cost.",
        "",
        "| Hypothetical task profile | Using mean times | Per-seed scenarios |",
        "| --- | --- | --- |",
    ]
    for task in TASKS:
        mean = report["mean_time_scenarios"][task]["hypothetical_strict_break_even_tasks"]
        seeds = [r["tasks"][task]["hypothetical_strict_break_even_tasks"] for r in report["rows"]]
        text = ", ".join("no payback" if n is None else f"{n} tasks" for n in seeds)
        lines.append(f"| {task.title()}-like costs | {mean} tasks | {text} |")
    pairs = report["hypothetical_balanced_mix_mean_times"]
    per_seed = ", ".join(
        f"{r['hypothetical_balanced_mix']['strict_break_even_tasks_in_complete_pairs']} tasks"
        for r in report["rows"]
    )
    lines += [
        f"| Balanced energy/position pairs | {pairs['strict_break_even_pairs']} pairs = "
        f"{pairs['strict_break_even_tasks_in_complete_pairs']} tasks | {per_seed} |",
        "",
        "Per-seed order is 20260925, 20261001, 20261002. These are **conditional cost",
        "scenarios, not counts of demonstrated useful tasks or uncertainty intervals**.",
        "The two tasks actually studied do not pay back pretraining even at their",
        "first recorded target crossings. Repeating seeds does not create new tasks.",
        "The position-cost example crosses at four tasks using means, but the three",
        "seed-based scenarios span three to eight tasks. The exact break-even count",
        "for genuinely new tasks cannot be measured from this two-task study.",
        "",
        "## Quality is a separate requirement",
        "",
        "The strongest validation evidence is position: fine-tuning improves all three",
        "paired direct results and the periodic reference. Energy improves on average",
        "against direct training, but the quadratic reference remains better on average.",
        "A timing target based on the direct Transformer is not a guarantee of beating",
        "the best classical estimator, nor of reaching the best fine-tuned accuracy.",
        "The full-budget accuracy gains and the earlier target-crossing times describe",
        "different checkpoints; they must not be claimed simultaneously at the shorter time.",
        "",
        "A shared pretrained encoder can therefore have two distinct benefits: better",
        "quality on a demonstrated task and cheaper adaptation on future suitable tasks.",
        "Only the former and retrospective adaptation times are measured here. General",
        "reuse, new task quality and a deployment stopping rule remain untested. This",
        "small single-detector study does not establish a universal foundation model.",
        "",
        "## Reproduce",
        "",
        "This analysis reads only the verified validation review, not the reserved test.",
        "Its input hash, individual timings and formulas are in",
        "[amortization.json](amortization.json).",
        "",
        "```bash",
        "uv run --locked --extra cpu python scripts/review_amortization.py \\",
        "  --review reports/confirmation/review.json --output reports/final",
        "```",
        "",
    ]
    (output / "amortization.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("reports/confirmation/review.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/final"))
    args = parser.parse_args()
    report = build_report(args.review)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "amortization.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    make_figure(report, args.output)
    write_markdown(report, args.output)
    print(json.dumps(report["mean_time_scenarios"], indent=2))


if __name__ == "__main__":
    main()
