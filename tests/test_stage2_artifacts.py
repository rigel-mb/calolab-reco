"""Synthetic result transport checks, using opaque checkpoint bytes without pickle."""

import io
import json
import zipfile

import numpy as np
import pytest

from calolab_reco import stage2_artifacts as transport
from calolab_reco.data import file_hash
from calolab_reco.metrics import regression_metrics


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(transport._canonical(value) + b"\n")


@pytest.fixture
def run(tmp_path):
    workspace, run_dir = tmp_path / "workspace", tmp_path / "run"
    workspace.mkdir()
    run_dir.mkdir()
    config = {"epochs": 2, "effective_batch_size": 4, "seed": 42}
    code = {
        "pyproject.toml": b"[project]\nname='fixture'\nversion='0.1.0'\n",
        "uv.lock": b"version=1\n",
        "configs/cnn.toml": b"epochs=2\neffective_batch_size=4\nseed=42\n",
        "src/calolab_reco/__init__.py": b"raise RuntimeError('Never execute bundled code')\n",
    }
    for name, content in code.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    validation_ids = np.array([[2, 10], [2, 11], [3, 1]], dtype=np.int64)
    targets = np.array([[5, 7, 12], [15, 10, 50], [80, 20, 70]], dtype=np.float32)
    data_dir = workspace / "data"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "train.npz",
        deposits=np.ones((8, 30, 85), dtype=np.float32),
        targets=np.ones((8, 3), dtype=np.float32),
        source_ids=np.column_stack((np.ones(8, dtype=np.int64), np.arange(8))),
    )
    np.savez_compressed(
        data_dir / "validation.npz",
        deposits=np.ones((3, 30, 85), dtype=np.float32),
        targets=targets,
        source_ids=validation_ids,
    )
    metadata = {
        "schema_version": 1,
        "input_selection": "train_validation_only",
        "train_count": 8,
        "val_count": 3,
        "input_scale": 2.0,
        "target_mean": [10.0, 12.0, 40.0],
        "target_std": [3.0, 4.0, 5.0],
        "prepared_manifest_sha256": "a" * 64,
        "prepared_output_sha256": {
            name: "b" * 64
            for name in ("deposits.npy", "targets.npy", "source_ids.npy", "split.npy")
        },
        "data_sha256": {
            f"{split}.npz": file_hash(data_dir / f"{split}.npz")
            for split in ("train", "validation")
        },
    }
    _write_json(data_dir / "stage2_manifest.json", metadata)
    (workspace / "ATTRIBUTION.txt").write_text("Synthetic fixture; no real dataset")
    file_names = [*code, "ATTRIBUTION.txt", *transport.BUNDLE_DATA_FILES]
    hashes = {name: file_hash(workspace / name) for name in file_names}
    code_hashes = {name: hashes[name] for name in code}
    manifest = {
        "schema_version": 1,
        "files": hashes,
        "code_sha256": transport._hash(transport._canonical(code_hashes)),
    }
    _write_json(workspace / "BUNDLE_MANIFEST.json", manifest)
    bundle = tmp_path / "original.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*hashes, "BUNDLE_MANIFEST.json"]:
            archive.write(workspace / name, name)
    provenance = {
        "stage2_manifest_sha256": hashes["data/stage2_manifest.json"],
        "prepared_manifest_sha256": metadata["prepared_manifest_sha256"],
        "prepared_output_sha256": metadata["prepared_output_sha256"],
        "data_sha256": metadata["data_sha256"],
        "source_files": code_hashes,
        "code_sha256": manifest["code_sha256"],
        "config_sha256": transport._hash(transport._canonical(config)),
    }
    for name in ("best.pt", "last.pt"):
        (run_dir / name).write_bytes(
            b"Opaque checkpoint fixture, deliberately not a pickle: " + name.encode()
        )
        (run_dir / f"{name}.sha256").write_text(file_hash(run_dir / name) + "\n")
    history = {
        "schema_version": 1,
        "model_kind": "cnn",
        "status": "completed",
        "completed": True,
        "epochs_completed": 2,
        "global_step": 4,
        "train_count": 8,
        "val_count": 3,
        "best_epoch": 1,
        "best_validation_loss": 0.2,
        "history": [
            {"epoch": 1, "global_step": 2, "train_count": 8, "validation_loss": 0.2},
            {"epoch": 2, "global_step": 4, "train_count": 8, "validation_loss": 0.3},
        ],
        "config": config,
        "test_used": False,
        "provenance": provenance,
        "transformations": {k: metadata[k] for k in ("input_scale", "target_mean", "target_std")},
    }
    _write_json(run_dir / "history.json", history)
    validation = run_dir / "validation"
    validation.mkdir()
    predictions = targets.astype(np.float64) + np.array([0.25, -0.1, 0.2])
    np.savez_compressed(
        validation / "predictions.npz",
        source_ids=validation_ids,
        targets=targets,
        predictions=predictions,
    )
    metrics = {
        "schema_version": 1,
        "model_kind": "cnn",
        "split": "validation",
        "count": 3,
        "full_validation_count": 3,
        "limited": False,
        "test_used": False,
        "checkpoint_epoch": 1,
        "checkpoint_sha256": file_hash(run_dir / "best.pt"),
        "provenance": provenance,
        "source_ids_sha256": transport._hash(validation_ids.tobytes()),
        "predictions_sha256": file_hash(validation / "predictions.npz"),
        "validation_loss": 0.2,
        "metrics": regression_metrics(targets, predictions),
    }
    _write_json(validation / "metrics.json", metrics)
    return {
        "run": run_dir,
        "workspace": workspace,
        "bundle": bundle,
        "artifact": tmp_path / "returned.zip",
        "output": tmp_path / "imported",
    }


def _export(run):
    return transport.export_results(
        run["run"],
        run["workspace"] / "BUNDLE_MANIFEST.json",
        file_hash(run["bundle"]),
        run["artifact"],
    )


def _import(run):
    return transport.import_results(run["artifact"], run["bundle"], run["output"])


def _rewrite_artifact(path, mutate, rehash=False):
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    mutate(files)
    if rehash:
        manifest = json.loads(files["RESULT_MANIFEST.json"])
        manifest["files"] = {
            name: transport._hash(content)
            for name, content in files.items()
            if name != "RESULT_MANIFEST.json"
        }
        files["RESULT_MANIFEST.json"] = transport._canonical(manifest)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_roundtrip_is_deterministic_idempotent_and_keeps_opaque_checkpoint_bytes(run):
    first = _export(run)
    original_zip = run["artifact"].read_bytes()
    assert _export(run) == first
    assert run["artifact"].read_bytes() == original_zip
    imported = _import(run)
    assert imported["completed"] is True and imported["test_used"] is False
    assert imported["result_sha256"] == file_hash(run["artifact"])
    assert _import(run) == imported
    for name in transport.RESULT_FILES:
        assert (run["output"] / name).read_bytes() == (run["run"] / name).read_bytes()
    with zipfile.ZipFile(run["artifact"]) as archive:
        assert set(archive.namelist()) == set(transport.RESULT_FILES) | {"RESULT_MANIFEST.json"}
        assert all(m.date_time == transport.ZIP_DATE for m in archive.infolist())


@pytest.mark.parametrize(
    "which,change",
    [
        ("history.json", {"completed": False}),
        ("history.json", {"status": "time_budget_exhausted"}),
        ("history.json", {"epochs_completed": 1}),
        ("history.json", {"train_count": 7}),
        ("validation/metrics.json", {"limited": True}),
        ("validation/metrics.json", {"test_used": True}),
        ("validation/metrics.json", {"full_validation_count": 4}),
    ],
)
def test_export_rejects_incomplete_or_nonprotocol_runs(run, which, change):
    path = run["run"] / which
    value = json.loads(path.read_text())
    value.update(change)
    _write_json(path, value)
    with pytest.raises(ValueError):
        _export(run)
    assert not run["artifact"].exists()


@pytest.mark.parametrize("name", ["source_ids", "targets", "predictions"])
def test_import_rejects_altered_validation_identity_or_nonfinite_predictions_even_rehashed(
    run, name
):
    _export(run)

    def alter(files):
        with np.load(io.BytesIO(files["validation/predictions.npz"]), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays[name][0, 0] = np.nan if name == "predictions" else arrays[name][0, 0] + 1
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        files["validation/predictions.npz"] = output.getvalue()
        metrics = json.loads(files["validation/metrics.json"])
        metrics["predictions_sha256"] = transport._hash(output.getvalue())
        metrics["source_ids_sha256"] = transport._hash(arrays["source_ids"].astype("<i8").tobytes())
        files["validation/metrics.json"] = transport._canonical(metrics)

    _rewrite_artifact(run["artifact"], alter, rehash=True)
    with pytest.raises(ValueError, match="original bundle|Nonfinite"):
        _import(run)
    assert not run["output"].exists()


def test_import_rejects_checkpoint_sidecar_mismatch_even_when_result_manifest_rehashed(run):
    _export(run)
    _rewrite_artifact(
        run["artifact"], lambda files: files.update({"last.pt": b"changed"}), rehash=True
    )
    with pytest.raises(ValueError, match="sidecar"):
        _import(run)


def test_import_rejects_wrong_original_bundle_and_altered_provenance(run):
    _export(run)
    with run["bundle"].open("ab") as stream:
        stream.write(b"different ZIP bytes")
    with pytest.raises(ValueError, match="provenance"):
        _import(run)


def test_import_rejects_tampered_file_without_updated_hash(run):
    _export(run)
    _rewrite_artifact(run["artifact"], lambda files: files.update({"last.pt": b"changed"}))
    with pytest.raises(ValueError, match="SHA256"):
        _import(run)


def test_import_refuses_different_existing_destination_without_changes(run):
    _export(run)
    _import(run)
    checkpoint = run["output"] / "best.pt"
    checkpoint.write_bytes(b"preserve existing work")
    with pytest.raises(FileExistsError, match="preserve"):
        _import(run)
    assert checkpoint.read_bytes() == b"preserve existing work"


def test_exact_allowlist_rejects_traversal_and_duplicate_members(run):
    _export(run)
    clean = run["artifact"].read_bytes()
    with zipfile.ZipFile(run["artifact"], "a") as archive:
        archive.writestr("../outside.txt", b"must not be extracted")
    with pytest.raises(ValueError, match="allowlist"):
        _import(run)
    assert not (run["output"].parent / "outside.txt").exists()
    run["artifact"].write_bytes(clean)
    with (
        pytest.warns(UserWarning, match="Duplicate"),
        zipfile.ZipFile(run["artifact"], "a") as archive,
    ):
        archive.writestr("best.pt", b"duplicate")
    with pytest.raises(ValueError, match="duplicates"):
        _import(run)


def test_results_size_limit_and_project_output_boundary(run, monkeypatch):
    _export(run)
    with pytest.raises(ValueError, match="outside"):
        transport.import_results(
            run["artifact"], run["bundle"], transport.PROJECT / "unsafe-output"
        )
    monkeypatch.setattr(transport, "MAX_RESULT_BYTES", 100)
    with pytest.raises(ValueError, match="size budget"):
        _import(run)


def test_reported_aggregates_must_match_predictions(run):
    path = run["run"] / "validation/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["metrics"]["energy_mae_gev"] += 1
    _write_json(path, metrics)
    with pytest.raises(ValueError, match="aggregate metrics"):
        _export(run)


def test_cli_roundtrip(run, capsys):
    transport.main(
        [
            "export",
            "--run-dir",
            str(run["run"]),
            "--bundle-manifest",
            str(run["workspace"] / "BUNDLE_MANIFEST.json"),
            "--bundle-sha256",
            file_hash(run["bundle"]),
            "--output",
            str(run["artifact"]),
        ]
    )
    assert json.loads(capsys.readouterr().out)["completed"] is True
    transport.main(
        [
            "import",
            "--artifact",
            str(run["artifact"]),
            "--bundle",
            str(run["bundle"]),
            "--output",
            str(run["output"]),
        ]
    )
    assert json.loads(capsys.readouterr().out)["test_used"] is False
