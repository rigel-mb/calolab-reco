"""CPU replay checks and explicitly synthetic report fixtures, never GPU results."""

import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from calolab_reco import pilot
from calolab_reco.pilot_reporting import render_report, validate_report


@pytest.fixture(scope="module")
def cpu_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_pilot")
    rng = np.random.default_rng(19)
    deposits = rng.exponential(12, size=(8, 30, 85)).astype(np.float32)
    deposits[rng.random(deposits.shape) < 0.8] = 0
    targets = np.column_stack((np.arange(8) + 1, np.arange(8) / 2, np.arange(8) + 10))
    targets = targets.astype(np.float32)
    source_ids = np.column_stack((np.zeros(8), np.arange(8))).astype(np.int64)
    data_path = root / "train.npz"
    np.savez(data_path, deposits=deposits, targets=targets, source_ids=source_ids)
    config = {
        "seed": 29,
        "microbatch_size": 2,
        "effective_batch_size": 4,
        "warmup_steps": 0,
        "measured_steps": 1,
        "resume_prefix_steps": 1,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "scheduler_steps": 10,
        "projected_epochs": 2,
        "max_phase_seconds": 240,
    }
    config_path = root / "pilot.toml"
    config_path.write_text("".join(f"{key} = {value}\n" for key, value in config.items()))
    manifest = {
        "input_selection": "train_only",
        "data_sha256": pilot.sha256(data_path),
        "subset_count": 8,
        "full_train_count": 8,
        "input_scale": float(np.median(deposits[deposits > 0])),
        "target_mean": targets.mean(axis=0, dtype=np.float64).tolist(),
        "target_std": targets.std(axis=0, dtype=np.float64).tolist(),
        "mask_ratio": 0.5,
        "prepared_manifest_sha256": "1" * 64,
    }
    manifest_path = root / "transform_stats.json"
    manifest_path.write_text(json.dumps(manifest))
    pilot.validate_config(config)
    hardware = pilot.configure("cpu", config["seed"])
    x, y, metadata, provenance = pilot.load_inputs(data_path, manifest_path, config, None)
    benchmark = pilot.benchmark(x, y, metadata, config, "cpu", hardware, provenance)

    # Separate interpreters must recover the stochastic masked workload exactly.
    command = [
        sys.executable, "-B", "-m", "calolab_reco.pilot",
        "--data", str(data_path), "--manifest", str(manifest_path),
        "--config", str(config_path), "--device", "cpu",
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    checkpoints = root / "checkpoints"
    resumed = root / "resumed"
    for phase, arguments in [
        ("checkpoint", ["--output", str(checkpoints)]),
        ("resume", ["--output", str(resumed), "--checkpoint-dir", str(checkpoints)]),
    ]:
        result = subprocess.run(
            [*command, phase, *arguments], capture_output=True, text=True,
            env=environment, timeout=60, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return {
        "benchmark": benchmark,
        "resume": json.loads((resumed / "resume.json").read_text()),
        "checkpoints": checkpoints,
        "inputs": (x, y, metadata, config, "cpu", provenance),
    }


def test_cpu_benchmark_and_fresh_process_replay_cover_all_workloads(cpu_run):
    benchmark, resumed = cpu_run["benchmark"], cpu_run["resume"]
    assert benchmark["hardware"]["device"] == resumed["device"] == "cpu"
    assert benchmark["test_used"] is False
    assert benchmark["accuracy_evaluated"] is False
    assert benchmark["subset_count"] == 8
    assert benchmark["provenance"] == resumed["provenance"]
    assert {row["workload"] for row in benchmark["workloads"]} == set(pilot.KINDS)
    for row in benchmark["workloads"]:
        assert row["measured_updates"] == 1
        assert len(row["seconds_per_update"]) == 1
        assert math.isfinite(row["median_seconds"]) and row["median_seconds"] > 0
        assert row["median_seconds"] == row["p95_seconds"] == row["seconds_per_update"][0]
        assert row["parameters"] > 0
        assert row["peak_torch_allocated_bytes"] is None
        assert row["projected_minutes_median"] == pytest.approx(row["median_seconds"] * 4 / 60)
    assert resumed["process_restart_checked"] is True
    assert resumed["colab_vm_restart_checked"] is False
    assert {row["workload"] for row in resumed["checks"]} == set(pilot.KINDS)
    for check in resumed["checks"]:
        assert check["exact_replay"] is True
        assert check["next_loss_finite"] is True
        assert check["restored_step"] == 1


@pytest.mark.parametrize("corruption", ["checkpoint_bytes", "missing_fingerprint", "provenance"])
def test_resume_rejects_checkpoint_corruption_before_loading(cpu_run, tmp_path, corruption):
    checkpoint_dir = tmp_path / "corrupted"
    shutil.copytree(cpu_run["checkpoints"], checkpoint_dir)
    index_path = checkpoint_dir / "checkpoint_files.json"
    index = json.loads(index_path.read_text())
    if corruption == "checkpoint_bytes":
        with (checkpoint_dir / "cnn_checkpoint.pt").open("ab") as stream:
            stream.write(b"changed")
    elif corruption == "missing_fingerprint":
        del index["files"]["cnn_checkpoint.pt"]
        index_path.write_text(json.dumps(index))
    else:
        index["provenance"]["data_sha256"] = "0" * 64
        index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        pilot.resume_phase(*cpu_run["inputs"], checkpoint_dir)


@pytest.fixture
def synthetic_gpu_report():
    """Invented numbers for schema tests only; never written as a public report."""
    provenance = {
        "code_sha256": "a" * 64,
        "prepared_manifest_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "pilot_manifest_sha256": "d" * 64,
        "config_sha256": "e" * 64,
    }
    workloads = [
        {
            "workload": kind, "parameters": 10, "measured_updates": 1,
            "seconds_per_update": [0.25], "median_seconds": 0.25, "p95_seconds": 0.25,
            "projected_minutes_median": 0.1, "projected_minutes_p95": 0.1,
            "peak_torch_allocated_bytes": 1024, "peak_torch_reserved_bytes": 2048,
        }
        for kind in pilot.KINDS
    ]
    return {
        "schema_version": 1,
        "bundle_sha256": "f" * 64,
        "benchmark": {
            "hardware": {
                "device": "cuda", "gpu_name": "SYNTHETIC GPU FIXTURE ONLY",
                "torch": "synthetic", "cuda_runtime": "synthetic",
                "precision": "float32", "attention_backend": "math",
            },
            "config": {
                "effective_batch_size": 4, "microbatch_size": 2,
                "warmup_steps": 0, "measured_steps": 1, "projected_epochs": 2,
            },
            "subset_count": 8, "full_train_count": 8,
            "provenance": provenance, "test_used": False, "accuracy_evaluated": False,
            "workloads": workloads,
        },
        "resume": {
            "device": "cuda", "provenance": copy.deepcopy(provenance),
            "process_restart_checked": True, "colab_vm_restart_checked": False,
            "checks": [
                {"workload": kind, "exact_replay": True, "next_loss_finite": True}
                for kind in pilot.KINDS
            ],
        },
        "storage_checks": {
            "bundle_roundtrip_verified": True,
            "checkpoint_roundtrip_verified": True,
            "report_roundtrip_verified": True,
        },
    }


def test_gpu_report_rendering_uses_only_an_explicitly_synthetic_fixture(synthetic_gpu_report):
    validate_report(synthetic_gpu_report)
    text = render_report(synthetic_gpu_report)
    assert "SYNTHETIC GPU FIXTURE ONLY" in text
    assert "This is not a full run." in text
    assert "No reconstruction accuracy" in text
    assert "A full Colab VM reset was not tested." in text
    assert all(f"| {kind} |" in text for kind in pilot.KINDS)


@pytest.mark.parametrize("cpu_phase", ["benchmark", "resume"])
def test_gpu_report_rejects_cpu_measurements(synthetic_gpu_report, cpu_phase):
    if cpu_phase == "benchmark":
        synthetic_gpu_report["benchmark"]["hardware"]["device"] = "cpu"
    else:
        synthetic_gpu_report["resume"]["device"] = "cpu"
    with pytest.raises(ValueError):
        validate_report(synthetic_gpu_report)


@pytest.mark.parametrize("failed_check", [
    "bundle_roundtrip_verified", "checkpoint_roundtrip_verified", "report_roundtrip_verified",
])
def test_gpu_report_rejects_missing_persistence_evidence(synthetic_gpu_report, failed_check):
    synthetic_gpu_report["storage_checks"][failed_check] = False
    with pytest.raises(ValueError, match="round-trip"):
        validate_report(synthetic_gpu_report)


def test_gpu_report_rejects_mismatched_data_provenance(synthetic_gpu_report):
    synthetic_gpu_report["resume"]["provenance"]["data_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate_report(synthetic_gpu_report)


def test_report_import_conflict_preserves_both_existing_outputs(
    synthetic_gpu_report, tmp_path, monkeypatch,
):
    script = Path(__file__).resolve().parents[1] / "scripts" / "import_pilot_report.py"
    specification = importlib.util.spec_from_file_location("import_pilot_report", script)
    importer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(importer)
    provenance = synthetic_gpu_report["benchmark"]["provenance"]
    metadata = json.dumps({
        "data_sha256": provenance["data_sha256"],
        "prepared_manifest_sha256": provenance["prepared_manifest_sha256"],
    }).encode()
    provenance["pilot_manifest_sha256"] = hashlib.sha256(metadata).hexdigest()
    synthetic_gpu_report["resume"]["provenance"] = copy.deepcopy(provenance)
    bundle = tmp_path / "synthetic_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps({
            "code_sha256": provenance["code_sha256"],
        }))
        archive.writestr("data/pilot_manifest.json", metadata)
    synthetic_gpu_report["bundle_sha256"] = pilot.sha256(bundle)
    report = tmp_path / "synthetic_report.json"
    report.write_text(json.dumps(synthetic_gpu_report))
    output = tmp_path / "temporary_reports"
    output.mkdir()
    existing = output / "colab_pilot.md"
    existing.write_text("Preserve this previous synthetic report.\n")
    monkeypatch.setattr(sys, "argv", [
        str(script), str(report), "--bundle", str(bundle), "--output", str(output),
    ])
    with pytest.raises(FileExistsError):
        importer.main()
    assert existing.read_text() == "Preserve this previous synthetic report.\n"
    assert not (output / "colab_pilot.json").exists()
