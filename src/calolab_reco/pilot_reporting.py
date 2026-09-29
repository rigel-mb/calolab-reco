"""Small, public reports for measured pilot runs."""

from __future__ import annotations

import math

WORKLOADS = {"cnn", "transformer", "masked_pretraining"}


def validate_report(report: dict, require_cuda: bool = True) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported pilot report schema.")
    benchmark = report["benchmark"]
    if require_cuda and benchmark["hardware"]["device"] != "cuda":
        raise ValueError("A CPU check cannot be published as the Colab GPU pilot.")
    if report["resume"]["device"] != benchmark["hardware"]["device"]:
        raise ValueError("Benchmark and restart devices differ.")
    if benchmark.get("test_used") is not False or benchmark.get("accuracy_evaluated") is not False:
        raise ValueError("A pilot must not claim test use or accuracy evaluation.")
    if benchmark["provenance"] != report["resume"]["provenance"]:
        raise ValueError("Benchmark and restart provenance differ.")
    rows = benchmark["workloads"]
    if len(rows) != 3 or {r["workload"] for r in rows} != WORKLOADS:
        raise ValueError("The three pilot workloads are required.")
    for row in rows:
        times = row["seconds_per_update"]
        if len(times) != row["measured_updates"] or not times:
            raise ValueError("Missing timing observations.")
        for number in [
            *times,
            row["median_seconds"],
            row["p95_seconds"],
            row["projected_minutes_median"],
            row["projected_minutes_p95"],
        ]:
            if not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError("Timing values must be finite and positive.")
        if require_cuda and any(
            not isinstance(row[key], int) or row[key] <= 0
            for key in ["peak_torch_allocated_bytes", "peak_torch_reserved_bytes"]
        ):
            raise ValueError("CUDA memory observations are required.")
    checks = report["resume"]["checks"]
    if len(checks) != 3 or {r["workload"] for r in checks} != WORKLOADS:
        raise ValueError("All workloads need restart verification.")
    if not all(r["exact_replay"] and r["next_loss_finite"] for r in checks):
        raise ValueError("A checkpoint replay did not pass.")
    if report["resume"].get("process_restart_checked") is not True:
        raise ValueError("Fresh-process replay was not checked.")
    if not all(
        report["storage_checks"].get(k) is True
        for k in [
            "bundle_roundtrip_verified",
            "checkpoint_roundtrip_verified",
            "report_roundtrip_verified",
        ]
    ):
        raise ValueError("Drive round-trip verification is incomplete.")
    if not report.get("bundle_sha256") or not benchmark["provenance"].get("code_sha256"):
        raise ValueError("Code and bundle fingerprints are required.")


def render_report(report: dict) -> str:
    validate_report(report)
    benchmark = report["benchmark"]
    hardware = benchmark["hardware"]
    config = benchmark["config"]
    rows = {r["workload"]: r for r in benchmark["workloads"]}
    lines = [
        "# Colab compute pilot",
        "",
        f"Measured on {hardware['gpu_name']} with PyTorch {hardware['torch']} "
        f"and CUDA runtime {hardware['cuda_runtime']}.",
        "",
        f"Pilot sample: {benchmark['subset_count']:,} train events. "
        f"Effective batch: {config['effective_batch_size']}; "
        f"microbatch: {config['microbatch_size']}.",
        f"Precision: {hardware['precision']}; attention backend: {hardware['attention_backend']}.",
        f"Warm-up: {config['warmup_steps']} updates; "
        f"measured: {config['measured_steps']} per workload.",
        "",
        "| Workload | Parameters | Median update (s) | P95 update (s) | Peak allocated (MiB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in benchmark["workloads"]:
        lines.append(
            f"| {row['workload']} | {row['parameters']:,} | {row['median_seconds']:.4f} "
            f"| {row['p95_seconds']:.4f} | {row['peak_torch_allocated_bytes'] / 2**20:.1f} |"
        )
    lines += [
        "",
        "## Compute-only projections",
        "",
        f"Extrapolation to {benchmark['full_train_count']:,} train events and "
        f"{config['projected_epochs']} epochs per workload. This is not a full run.",
        "",
        "| Stage | Median-based minutes | P95-based minutes | Proposed GPU ceiling (min) |",
        "| --- | ---: | ---: | ---: |",
    ]
    stage_rows = [
        ("CNN", ["cnn"], 45),
        ("Direct Transformer", ["transformer"], 45),
        ("Pretraining + fine-tuning", ["masked_pretraining", "transformer"], 105),
    ]
    for label, kinds, ceiling in stage_rows:
        median = sum(rows[k]["projected_minutes_median"] for k in kinds)
        p95 = sum(rows[k]["projected_minutes_p95"] for k in kinds)
        lines.append(f"| {label} | {median:.2f} | {p95:.2f} | {ceiling} |")
    lines += [
        "",
        "Validation, full-dataset input loading, periodic saves and final evaluation "
        "are not included in these projections. Pretraining's epoch count is a projection "
        "scenario, not a selected training budget. The two Transformer supervised runs "
        "use the same timing estimate. Review budget headroom before full runs.",
        "",
        "## Restart and persistence",
        "",
        "All three workloads reproduce the next update exactly in a new Python process, "
        "including model weights, optimizer, scheduler and random states. Checkpoints "
        "were read from verified Google Drive copies. A full Colab VM reset was not tested.",
        "",
        "Peak memory is measured by the PyTorch CUDA allocator; it is not total device usage. "
        "Pilot losses are only numerical checks. No reconstruction accuracy or reserved-test "
        "performance was evaluated.",
        "",
        "## Provenance",
        "",
        f"- Bundle SHA-256: `{report['bundle_sha256']}`",
        f"- Code SHA-256: `{benchmark['provenance']['code_sha256']}`",
        f"- Prepared manifest SHA-256: `{benchmark['provenance']['prepared_manifest_sha256']}`",
        f"- Pilot data SHA-256: `{benchmark['provenance']['data_sha256']}`",
        "",
        "The original data audit and this compute pilot answer different questions. "
        "Stage completion still requires review of the measured budget and remaining limitations.",
        "",
    ]
    return "\n".join(lines)
