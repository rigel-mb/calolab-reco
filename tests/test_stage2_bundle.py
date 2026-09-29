"""Synthetic stage 2 export tests; no source dataset or final test is consumed."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from calolab_reco.data import file_hash

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_stage2_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_stage2_bundle", SCRIPT)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


@pytest.fixture
def inputs(tmp_path):
    project = tmp_path / "project"
    for name, content in {
        "pyproject.toml": '[project]\nname = "synthetic"\nversion = "0.1.0"\n',
        "uv.lock": "version = 1\n",
        "configs/cnn.toml": "seed = 20260922\n",
        "configs/pilot.toml": "excluded = true\n",
        "src/calolab_reco/__init__.py": "",
        "src/calolab_reco/train_cnn.py": "def run():\n    return 1\n",
        "LOCAL_NOTES.md": "PRIVATE CONTEXT MUST NOT BE EXPORTED",
        "notebooks/private.ipynb": "{}",
        "src/calolab_reco/__pycache__/private.pyc": "ignored cache",
    }.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    root = tmp_path / "prepared"
    prepared = root / "derived" / "audit_v1"
    prepared.mkdir(parents=True)
    indices = np.arange(12, dtype=np.float32)
    deposits = np.zeros((12, 30, 85), dtype=np.float32)
    deposits[:, 1, 3] = indices + 1
    deposits[:, 20, 67] = (indices + 1) / 10
    arrays = {
        "deposits": deposits,
        "targets": np.column_stack((indices + 1, indices / 2, 50 - indices)).astype(np.float32),
        "source_ids": np.column_stack((np.ones(12, dtype=np.int64), np.arange(12))),
        "split": np.array([0] * 8 + [1, 1, 2, 2], dtype=np.uint8),
    }
    arrays["deposits"][8:] *= 1e6
    arrays["targets"][8:] *= 1e6
    _save_prepared(prepared, arrays)
    return project, root, tmp_path / "export" / "stage2.zip", arrays


def _save_prepared(prepared, arrays):
    for name, array in arrays.items():
        np.save(prepared / f"{name}.npy", array, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "output_sha256": {f"{name}.npy": file_hash(prepared / f"{name}.npy") for name in arrays},
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _contents(path):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("data/stage2_manifest.json"))
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        partitions = {}
        for split in ("train", "validation"):
            with np.load(io.BytesIO(archive.read(f"data/{split}.npz")), allow_pickle=False) as data:
                partitions[split] = {name: data[name] for name in data.files}
        return metadata, manifest, partitions, set(archive.namelist())


def test_full_train_and_validation_export_preserves_raw_values_without_test(inputs, monkeypatch):
    project, root, output, arrays = inputs
    actual_load = bundle.load_prepared
    calls = []

    def no_test_loader(data_root, split):
        assert split in ("train", "validation")
        calls.append(split)
        return actual_load(data_root, split=split)

    monkeypatch.setattr(bundle, "load_prepared", no_test_loader)
    summary = bundle.build_bundle(project, root, output)
    metadata, manifest, partitions, names = _contents(output)
    assert calls == ["train", "validation"]
    for split, split_id in (("train", 0), ("validation", 1)):
        assert set(partitions[split]) == set(bundle.ARRAY_NAMES)
        for name, exported in partitions[split].items():
            expected = arrays[name][arrays["split"] == split_id]
            np.testing.assert_array_equal(exported, expected)
            assert exported.dtype == expected.dtype
    assert names == set(bundle.CODE_FILES) | set(bundle.DATA_FILES) | {
        "src/calolab_reco/__init__.py",
        "src/calolab_reco/train_cnn.py",
        "ATTRIBUTION.txt",
        "BUNDLE_MANIFEST.json",
    }
    assert metadata["schema_version"] == 1
    assert metadata["input_selection"] == "train_validation_only"
    assert metadata["train_count"] == 8
    assert metadata["val_count"] == 2
    train_deposits = arrays["deposits"][:8]
    assert metadata["input_scale"] == np.median(train_deposits[train_deposits > 0])
    np.testing.assert_allclose(metadata["target_mean"], arrays["targets"][:8].mean(0))
    np.testing.assert_allclose(
        metadata["target_std"], arrays["targets"][:8].std(0, dtype=np.float64)
    )
    assert metadata["target_std_ddof"] == 0
    assert metadata["source_record"] == "https://zenodo.org/records/18929909"
    assert metadata["energy_unit_interpretation"]["status"] == "inferred_not_producer_certified"
    assert metadata["prepared_manifest_sha256"] == file_hash(
        root / "derived/audit_v1/manifest.json"
    )
    assert (
        metadata["prepared_output_sha256"]
        == json.loads((root / "derived/audit_v1/manifest.json").read_text())["output_sha256"]
    )
    assert metadata["data_sha256"] == {
        f"{split}.npz": manifest["files"][f"data/{split}.npz"] for split in ("train", "validation")
    }
    assert str(root) not in json.dumps(metadata)
    assert summary["zip_sha256"] == file_hash(output)
    assert json.loads(output.with_suffix(".summary.json").read_text()) == summary
    bundle.verify_bundle(output)


def test_bundle_is_deterministic_and_refuses_different_overwrite(inputs):
    project, root, output, _ = inputs
    first = bundle.build_bundle(project, root, output)
    original_bytes = output.read_bytes()
    assert bundle.build_bundle(project, root, output) == first
    assert output.read_bytes() == original_bytes
    metadata, manifest, _, _ = _contents(output)
    code_files = {
        name: sha
        for name, sha in manifest["files"].items()
        if name in bundle.CODE_FILES or name.startswith("src/")
    }
    canonical = json.dumps(code_files, sort_keys=True, separators=(",", ":")).encode()
    assert manifest["code_sha256"] == hashlib.sha256(canonical).hexdigest()
    with zipfile.ZipFile(output) as archive:
        assert all(m.date_time == bundle.ZIP_DATE for m in archive.infolist())
        for split in ("train", "validation"):
            with zipfile.ZipFile(io.BytesIO(archive.read(f"data/{split}.npz"))) as data:
                assert all(m.date_time == bundle.ZIP_DATE for m in data.infolist())
    (project / "configs/cnn.toml").write_text("seed = 123\n")
    with pytest.raises(FileExistsError, match="--force"):
        bundle.build_bundle(project, root, output)
    assert output.read_bytes() == original_bytes
    replacement = bundle.build_bundle(project, root, output, force=True)
    assert replacement["zip_sha256"] != first["zip_sha256"]
    assert replacement["code_sha256"] != first["code_sha256"]
    assert _contents(output)[0]["data_sha256"] == metadata["data_sha256"]


@pytest.mark.parametrize("overlap", [False, True])
def test_rejects_duplicate_or_overlapping_source_ids(inputs, overlap):
    project, root, output, arrays = inputs
    arrays["source_ids"][8 if overlap else 1] = arrays["source_ids"][0]
    _save_prepared(root / "derived/audit_v1", arrays)
    with pytest.raises(ValueError, match="overlap" if overlap else "Duplicate"):
        bundle.build_bundle(project, root, output)
    assert not output.exists()


def _rewrite_archive(path, mutate, rehash=False):
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    mutate(files)
    if rehash:
        manifest = json.loads(files["BUNDLE_MANIFEST.json"])
        manifest["files"] = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in files.items()
            if name != "BUNDLE_MANIFEST.json"
        }
        code = {name: sha for name, sha in manifest["files"].items() if bundle._is_code_path(name)}
        manifest["code_sha256"] = hashlib.sha256(bundle._json_bytes(code)).hexdigest()
        files["BUNDLE_MANIFEST.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_verifier_rejects_changed_code_and_even_rehashed_private_files(inputs):
    project, root, output, _ = inputs
    bundle.build_bundle(project, root, output)
    _rewrite_archive(output, lambda files: files.update({"configs/cnn.toml": b"tampered"}))
    with pytest.raises(ValueError, match="SHA256"):
        bundle.verify_bundle(output)
    bundle.build_bundle(project, root, output, force=True)
    _rewrite_archive(
        output, lambda files: files.update({"LOCAL_NOTES.md": b"private"}), rehash=True
    )
    with pytest.raises(ValueError, match="allowlist"):
        bundle.verify_bundle(output)


def test_verifier_checks_partition_counts_inside_rehashed_manifest(inputs):
    project, root, output, _ = inputs
    bundle.build_bundle(project, root, output)

    def change_count(files):
        metadata = json.loads(files["data/stage2_manifest.json"])
        metadata["val_count"] += 1
        files["data/stage2_manifest.json"] = json.dumps(metadata).encode()

    _rewrite_archive(output, change_count, rehash=True)
    with pytest.raises(ValueError, match="shape"):
        bundle.verify_bundle(output)


def test_rejects_corrupted_prepared_arrays(inputs):
    project, root, output, _ = inputs
    with (root / "derived/audit_v1/targets.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="failed verification"):
        bundle.build_bundle(project, root, output)


def test_rejects_in_repo_output_or_symlinked_code(inputs):
    project, root, output, _ = inputs
    with pytest.raises(ValueError, match="outside"):
        bundle.build_bundle(project, root, project / "bad.zip")
    config = project / "configs/cnn.toml"
    config.unlink()
    config.symlink_to(project / "LOCAL_NOTES.md")
    with pytest.raises(ValueError, match="ordinary"):
        bundle.build_bundle(project, root, output)


def test_enforces_raw_size_budget_even_when_data_compress_well(inputs, monkeypatch):
    project, root, output, _ = inputs
    monkeypatch.setattr(bundle, "MAX_BUNDLE_BYTES", 100_000)
    with pytest.raises(ValueError, match="size budget"):
        bundle.build_bundle(project, root, output)
    assert not output.exists()


def test_npz_verification_does_not_load_deposit_arrays(inputs, monkeypatch):
    project, root, output, _ = inputs
    bundle.build_bundle(project, root, output)
    actual_read = np.lib.format.read_array

    def small_ids_only(stream, *args, **kwargs):
        result = actual_read(stream, *args, **kwargs)
        assert result.ndim == 2 and result.shape[1] == 2
        assert result.dtype.kind in "iu"
        return result

    monkeypatch.setattr(np.lib.format, "read_array", small_ids_only)
    bundle.verify_bundle(output)
