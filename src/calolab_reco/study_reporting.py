"""Readable validation reports for the physical references and the CNN."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from calolab_reco.baselines import fit_baselines, predict_baselines
from calolab_reco.data import file_hash, load_prepared
from calolab_reco.metrics import regression_metrics, stratified_metrics


def baseline_study(data_root: Path):
    """Fit on train, predict validation, and retain public aggregate provenance."""
    root = Path(data_root) / "derived/audit_v1"
    manifest_hash = file_hash(root / "manifest.json")
    train = load_prepared(data_root, "train")
    calibration = fit_baselines(train["deposits"], train["targets"])
    train_count = len(train["targets"])
    del train
    validation = load_prepared(data_root, "validation")
    predictions = predict_baselines(validation["deposits"], calibration)
    if file_hash(root / "manifest.json") != manifest_hash:
        raise ValueError("The prepared sample changed during the reference calculation.")
    manifest = json.loads((root / "manifest.json").read_text())
    package = Path(__file__).resolve().parent
    code = {
        name: file_hash(package / name)
        for name in ["baselines.py", "metrics.py", "data.py", "study_reporting.py"]
    }
    report = {
        "schema_version": 1,
        "evaluation_sample": "validation",
        "train_count": train_count,
        "validation_count": len(validation["targets"]),
        "test_used": False,
        "calibration": calibration,
        "provenance": {
            "prepared_manifest_sha256": manifest_hash,
            "prepared_output_sha256": manifest["output_sha256"],
            "reference_code_files": code,
            "reference_code_sha256": hashlib.sha256(
                json.dumps(code, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "metrics": {
            key: regression_metrics(validation["targets"], value)
            for key, value in predictions.items()
        },
        "subgroups": {
            key: stratified_metrics(validation["targets"], value)
            for key, value in predictions.items()
        },
        "units": {
            "energy": "GeV, working interpretation supported by the source cross-check",
            "position": "stored local fractional iphi/ieta coordinate units",
        },
    }
    return report, validation, predictions


def metric_table(rows: dict[str, dict]) -> str:
    lines = [
        "| Estimator | Energy bias (%) | Robust resolution (%) | Energy MARE (%) "
        "| Energy MAE (GeV) "
        "| Position median | Position P68 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in rows.items():
        lines.append(
            f"| {name} | {100 * metrics['energy_relative_bias']:.3f} "
            f"| {100 * metrics['energy_relative_resolution']:.3f} "
            f"| {100 * metrics['energy_mare']:.3f} "
            f"| {metrics['energy_mae_gev']:.3f} "
            f"| {metrics['position_distance_median']:.4f} "
            f"| {metrics['position_distance_p68']:.4f} |"
        )
    return "\n".join(lines)


def baseline_markdown(report: dict) -> str:
    return "\n".join(
        [
            "# Physical references on validation",
            "",
            f"Calibration fitted on {report['train_count']:,} train events; "
            f"evaluation on {report['validation_count']:,} validation events. "
            "The final test was not used.",
            "",
            metric_table(report["metrics"]),
            "",
            "Raw energy is the deposit sum divided by exactly 1000. Raw position is "
            "the energy-weighted mean of integer row/column indices. Calibration "
            "fits a separate affine correction for each target, using train only.",
            "",
            "Energy relative residual is (prediction - target) / target. Bias is its "
            "mean; robust resolution is half the difference between its 84th and "
            "16th percentiles. Position distances use stored coordinate units, not "
            "millimeters. A calibrated offset does not identify physical cell centers.",
            "",
            "The raw barycenter's absolute position error depends on the index-origin "
            "convention. It is a diagnostic; use the calibrated reference for the "
            "main CNN comparison. Inputs contain no added noise or threshold.",
            "",
            "These results establish references, not a measured CNN advantage. "
            "Subgroup metrics and provenance are retained in the JSON report.",
            "",
            f"Prepared manifest SHA-256: `{report['provenance']['prepared_manifest_sha256']}`",
            "",
        ]
    )


def save_baseline_results(report, validation, predictions, external_output: Path) -> None:
    external_output = Path(external_output).expanduser().resolve()
    project = Path(__file__).resolve().parents[2]
    if external_output.is_relative_to(project):
        raise ValueError("Predictions must be stored outside the project repository.")
    external_output.mkdir(parents=True, exist_ok=True)
    (external_output / "baselines.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        external_output / "baseline_predictions.npz",
        targets=validation["targets"],
        source_ids=validation["source_ids"],
        **predictions,
    )
