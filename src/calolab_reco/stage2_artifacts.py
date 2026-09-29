"""Exchange bounded CNN results without executing bundled code or loading checkpoints.

Checkpoint bytes and sidecars are fingerprinted, never deserialized. Returned
validation identities, targets and aggregate metrics are checked against the
original stage 2 bundle before any import destination is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import tomllib
import zipfile
from pathlib import Path

import numpy as np

from calolab_reco.data import file_hash
from calolab_reco.metrics import regression_metrics

PROJECT = Path(__file__).resolve().parents[2]
MAX_RESULT_BYTES = 64 * 1024**2
MAX_BUNDLE_BYTES = 512 * 1024**2
MAX_JSON_BYTES = 2 * 1024**2
RESULT_FILES = (
    "best.pt",
    "best.pt.sha256",
    "last.pt",
    "last.pt.sha256",
    "history.json",
    "validation/metrics.json",
    "validation/predictions.npz",
)
CODE_FILES = ("pyproject.toml", "uv.lock", "configs/cnn.toml")
BUNDLE_DATA_FILES = ("data/train.npz", "data/validation.npz", "data/stage2_manifest.json")
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stream_hash(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024**2), b""):
        digest.update(block)
    return digest.hexdigest()


def _digest(value) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


def _json(content: bytes) -> dict:
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("JSON metadata exceeds the size budget")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Nonfinite JSON number")
        return parsed

    value = json.loads(
        content,
        object_pairs_hook=unique_keys,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _external(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(PROJECT):
        raise ValueError("Result checkpoints and predictions must remain outside the repository")
    return path


def _ordinary(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Expected an ordinary file within its directory: {name}")
    if any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root)):
        raise ValueError(f"Symbolic-link parent is not allowed: {name}")
    return path


def _code_path(name: str) -> bool:
    return (
        name in CODE_FILES or re.fullmatch(r"src/calolab_reco/[A-Za-z_]\w*\.py", name) is not None
    )


def _bundle_manifest(manifest: dict) -> dict:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported original bundle manifest")
    files = manifest["files"]
    required = set(CODE_FILES) | set(BUNDLE_DATA_FILES) | {"ATTRIBUTION.txt"}
    if not required <= set(files) or not any(n.startswith("src/") for n in files):
        raise ValueError("Original bundle is missing required files")
    for name, digest in files.items():
        if not _code_path(name) and name not in (*BUNDLE_DATA_FILES, "ATTRIBUTION.txt"):
            raise ValueError(f"Original bundle member outside allowlist: {name}")
        _digest(digest)
    code = {name: digest for name, digest in files.items() if _code_path(name)}
    if _hash(_canonical(code)) != manifest["code_sha256"]:
        raise ValueError("Original bundle code fingerprint mismatch")
    return code


def _archive(archive: zipfile.ZipFile, limit: int, expected: set[str]) -> None:
    members = archive.infolist()
    names = [member.filename for member in members]
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("ZIP members differ from the exact allowlist or contain duplicates")
    if sum(member.file_size for member in members) > limit:
        raise ValueError("Uncompressed ZIP exceeds the size budget")
    if any(member.flag_bits & 1 or member.is_dir() for member in members):
        raise ValueError("Encrypted entries and directory entries are not allowed")


def _arrays(stream, count: int, predictions: bool) -> dict:
    """Bound inner ZIP/NPY headers before loading only labels, IDs and predictions."""
    limit = MAX_RESULT_BYTES if predictions else MAX_BUNDLE_BYTES
    shapes = {"targets": (count, 3), "source_ids": (count, 2)}
    shapes["predictions" if predictions else "deposits"] = (
        (count, 3) if predictions else (count, 30, 85)
    )
    with tempfile.TemporaryFile() as copied:
        shutil.copyfileobj(stream, copied, length=8 * 1024**2)
        if copied.tell() > limit:
            raise ValueError("NPZ exceeds the size budget")
        copied.seek(0)
        with zipfile.ZipFile(copied) as archive:
            _archive(archive, limit, {f"{name}.npy" for name in shapes})
            for name, shape in shapes.items():
                with archive.open(f"{name}.npy") as array_stream:
                    version = np.lib.format.read_magic(array_stream)
                    reader = {
                        (1, 0): np.lib.format.read_array_header_1_0,
                        (2, 0): np.lib.format.read_array_header_2_0,
                    }.get(version)
                    if reader is None:
                        raise ValueError("Unsupported NPY format")
                    actual_shape, _, dtype = reader(array_stream)
                    if actual_shape != shape or dtype.hasobject or dtype.kind not in "fiu":
                        raise ValueError("Unexpected NPZ shape or unsafe dtype")
                    if name in ("targets", "deposits") and dtype != np.float32:
                        raise ValueError("Raw targets/deposits must remain float32")
                    if name == "source_ids" and dtype.kind not in "iu":
                        raise ValueError("Source identifiers must be integers")
                    size = math.prod(shape) * dtype.itemsize
                    if array_stream.tell() + size != archive.getinfo(f"{name}.npy").file_size:
                        raise ValueError("NPY payload length differs from its header")
            output = {}
            for name in shapes:
                if name != "deposits":
                    with archive.open(f"{name}.npy") as array_stream:
                        output[name] = np.lib.format.read_array(array_stream, allow_pickle=False)
    if any(not np.isfinite(array).all() for array in output.values()):
        raise ValueError("Nonfinite validation arrays")
    ids = output["source_ids"]
    if (ids < 0).any() or len(np.unique(ids, axis=0)) != count:
        raise ValueError("Invalid or duplicate validation identifiers")
    return output


def _context(manifest: dict, code: dict, read, open_stream) -> dict:
    metadata = _json(read("data/stage2_manifest.json"))
    config = tomllib.loads(read("configs/cnn.toml").decode())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("input_selection") != "train_validation_only"
    ):
        raise ValueError("Original data manifest does not describe train/validation only")
    for key in ("train_count", "val_count"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ValueError("Original partition counts must be positive integers")
    expected_data = {
        f"{s}.npz": manifest["files"][f"data/{s}.npz"] for s in ("train", "validation")
    }
    if metadata["data_sha256"] != expected_data:
        raise ValueError("Original data fingerprints disagree")
    provenance = {
        "stage2_manifest_sha256": manifest["files"]["data/stage2_manifest.json"],
        "prepared_manifest_sha256": metadata["prepared_manifest_sha256"],
        "prepared_output_sha256": metadata["prepared_output_sha256"],
        "data_sha256": expected_data,
        "code_sha256": manifest["code_sha256"],
        "source_files": code,
        "config_sha256": _hash(_canonical(config)),
    }
    with open_stream("data/validation.npz") as stream:
        reference = _arrays(stream, metadata["val_count"], predictions=False)
    return {
        "metadata": metadata,
        "config": config,
        "provenance": provenance,
        "reference": reference,
        "manifest": manifest,
    }


def _workspace_context(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).expanduser().resolve()
    root = manifest_path.parent
    manifest = _json(manifest_path.read_bytes())
    code = _bundle_manifest(manifest)
    total = 0
    for name, expected in manifest["files"].items():
        path = _ordinary(root, name)
        total += path.stat().st_size
        if total > MAX_BUNDLE_BYTES or file_hash(path) != expected:
            raise ValueError(f"Original workspace size or fingerprint mismatch: {name}")
    return _context(
        manifest,
        code,
        lambda n: _ordinary(root, n).read_bytes(),
        lambda n: _ordinary(root, n).open("rb"),
    )


def _zip_context(path: Path) -> dict:
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Original bundle exceeds the size budget")
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo("BUNDLE_MANIFEST.json").file_size > MAX_JSON_BYTES:
            raise ValueError("Original bundle manifest exceeds the size budget")
        manifest = _json(archive.read("BUNDLE_MANIFEST.json"))
        code = _bundle_manifest(manifest)
        _archive(archive, MAX_BUNDLE_BYTES, set(manifest["files"]) | {"BUNDLE_MANIFEST.json"})
        for name, expected in manifest["files"].items():
            with archive.open(name) as stream:
                if _stream_hash(stream) != expected:
                    raise ValueError(f"Original bundle fingerprint mismatch: {name}")
        return _context(manifest, code, archive.read, archive.open)


def _validate_run(read, open_stream, hashes: dict, context: dict) -> tuple[dict, dict]:
    history, metrics = _json(read("history.json")), _json(read("validation/metrics.json"))
    for name in ("best.pt", "last.pt"):
        if read(f"{name}.sha256").decode().strip() != hashes[name]:
            raise ValueError(f"Checkpoint checksum sidecar mismatch: {name}")
    metadata, config, provenance = (context[k] for k in ("metadata", "config", "provenance"))
    if any(
        r.get("schema_version") != 1
        or r.get("model_kind") != "cnn"
        or r.get("test_used") is not False
        or r.get("provenance") != provenance
        for r in (history, metrics)
    ):
        raise ValueError("Run model, test-use declaration or provenance mismatch")
    if (
        history.get("status") != "completed"
        or history.get("completed") is not True
        or history.get("config") != config
        or history.get("epochs_completed") != config["epochs"]
        or history.get("train_count") != metadata["train_count"]
        or history.get("val_count") != metadata["val_count"]
    ):
        raise ValueError("Only a completed full-training run can be exported/imported")
    transformations = {key: metadata[key] for key in ("input_scale", "target_mean", "target_std")}
    if history.get("transformations") != transformations:
        raise ValueError("Run transformations differ from train-fitted statistics")
    epochs = history["history"]
    if len(epochs) != config["epochs"] or [e["epoch"] for e in epochs] != list(
        range(1, len(epochs) + 1)
    ):
        raise ValueError("Training history does not cover every configured epoch")
    steps = math.ceil(metadata["train_count"] / config["effective_batch_size"])
    for epoch in epochs:
        if (
            epoch["train_count"] != metadata["train_count"]
            or epoch["global_step"] != epoch["epoch"] * steps
            or not math.isfinite(epoch["validation_loss"])
            or epoch["validation_loss"] < 0
        ):
            raise ValueError("Training history counts, steps or losses are inconsistent")
    best = min(epochs, key=lambda epoch: epoch["validation_loss"])
    if (
        history["global_step"] != steps * config["epochs"]
        or history["best_epoch"] != best["epoch"]
        or history["best_validation_loss"] != best["validation_loss"]
    ):
        raise ValueError("Best checkpoint selection differs from the full history")
    count = metadata["val_count"]
    if (
        metrics.get("split") != "validation"
        or metrics.get("limited") is not False
        or metrics.get("count") != count
        or metrics.get("full_validation_count") != count
        or metrics.get("checkpoint_epoch") != best["epoch"]
        or metrics.get("checkpoint_sha256") != hashes["best.pt"]
        or metrics.get("predictions_sha256") != hashes["validation/predictions.npz"]
    ):
        raise ValueError("Evaluation must cover full validation using the selected best checkpoint")
    with open_stream("validation/predictions.npz") as stream:
        arrays = _arrays(stream, count, predictions=True)
    for name in ("source_ids", "targets"):
        reference = context["reference"][name]
        if arrays[name].dtype != reference.dtype or not np.array_equal(arrays[name], reference):
            raise ValueError(f"Returned validation {name} differ from the original bundle")
    if metrics["source_ids_sha256"] != _hash(arrays["source_ids"].astype("<i8").tobytes()):
        raise ValueError("Evaluation identifier fingerprint mismatch")
    recomputed = regression_metrics(arrays["targets"], arrays["predictions"])
    if set(metrics["metrics"]) != set(recomputed) or any(
        not np.isclose(metrics["metrics"][k], v, rtol=1e-9, atol=1e-12)
        for k, v in recomputed.items()
    ):
        raise ValueError("Returned aggregate metrics disagree with predictions")
    if not math.isfinite(metrics["validation_loss"]) or metrics["validation_loss"] < 0:
        raise ValueError("Invalid validation loss")
    return history, metrics


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def export_results(run_dir: Path, bundle_manifest: Path, bundle_sha256: str, output: Path) -> dict:
    """Export a completed run; bundle SHA is checked against the original ZIP on import."""
    run_dir, output = _external(run_dir), _external(output)
    if output.suffix != ".zip":
        raise ValueError("Result output must use the .zip suffix")
    _digest(bundle_sha256)
    context = _workspace_context(bundle_manifest)
    paths = {name: _ordinary(run_dir, name) for name in RESULT_FILES}
    if sum(p.stat().st_size for p in paths.values()) > MAX_RESULT_BYTES - MAX_JSON_BYTES:
        raise ValueError("Run files exceed the result size budget")
    hashes = {name: file_hash(path) for name, path in paths.items()}
    _validate_run(lambda n: paths[n].read_bytes(), lambda n: paths[n].open("rb"), hashes, context)
    result = {
        "schema_version": 1,
        "files": hashes,
        "bundle_sha256": bundle_sha256,
        "code_sha256": context["manifest"]["code_sha256"],
        "provenance": context["provenance"],
        "config_file_sha256": context["manifest"]["files"]["configs/cnn.toml"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-", dir=output.parent) as temporary:
        staged = Path(temporary) / "results.zip"
        with zipfile.ZipFile(staged, "w") as archive:
            for name in sorted((*RESULT_FILES, "RESULT_MANIFEST.json")):
                if name == "RESULT_MANIFEST.json":
                    archive.writestr(_zip_info(name), _canonical(result) + b"\n")
                else:
                    with paths[name].open("rb") as src, archive.open(_zip_info(name), "w") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024**2)
        _verify_results(staged, context, bundle_sha256)
        digest = file_hash(staged)
        if output.exists() and file_hash(output) != digest:
            raise FileExistsError(
                "Existing result ZIP differs; preserve it and choose another output"
            )
        size = staged.stat().st_size
        if not output.exists():
            os.replace(staged, output)
    return {
        "result_sha256": digest,
        "result_bytes": size,
        "bundle_sha256": bundle_sha256,
        "code_sha256": result["code_sha256"],
        "completed": True,
        "test_used": False,
    }


def _verify_results(path: Path, context: dict, bundle_sha256: str) -> dict:
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError("Compressed results exceed the size budget")
    with zipfile.ZipFile(path) as archive:
        _archive(archive, MAX_RESULT_BYTES, set(RESULT_FILES) | {"RESULT_MANIFEST.json"})
        if archive.getinfo("RESULT_MANIFEST.json").file_size > MAX_JSON_BYTES:
            raise ValueError("Result manifest exceeds the size budget")
        result = _json(archive.read("RESULT_MANIFEST.json"))
        if (
            result.get("schema_version") != 1
            or set(result.get("files", {})) != set(RESULT_FILES)
            or result.get("bundle_sha256") != bundle_sha256
            or result.get("code_sha256") != context["manifest"]["code_sha256"]
            or result.get("provenance") != context["provenance"]
            or result.get("config_file_sha256") != context["manifest"]["files"]["configs/cnn.toml"]
        ):
            raise ValueError("Result manifest provenance differs from the original bundle")
        for name, expected in result["files"].items():
            _digest(expected)
            with archive.open(name) as stream:
                if _stream_hash(stream) != expected:
                    raise ValueError(f"Returned file SHA256 mismatch: {name}")
        _validate_run(archive.read, archive.open, result["files"], context)
    return result


def import_results(artifact: Path, bundle: Path, output: Path) -> dict:
    """Verify the returned ZIP and atomically import to a separate external directory."""
    artifact, bundle, output = (
        Path(artifact).expanduser(),
        Path(bundle).expanduser(),
        _external(output),
    )
    bundle_sha256 = file_hash(bundle)
    context = _zip_context(bundle)
    if file_hash(bundle) != bundle_sha256:
        raise ValueError("Original bundle changed during verification")
    result = _verify_results(artifact, context, bundle_sha256)
    expected = {**result["files"]}
    with zipfile.ZipFile(artifact) as archive:
        manifest_bytes = archive.read("RESULT_MANIFEST.json")
        if _json(manifest_bytes) != result:
            raise ValueError("Result manifest changed during verification")
        expected["RESULT_MANIFEST.json"] = _hash(manifest_bytes)
        if output.exists():
            if not output.is_dir() or any(p.is_symlink() for p in output.rglob("*")):
                raise FileExistsError("Existing import destination must be preserved")
            present = {
                p.relative_to(output).as_posix(): file_hash(p)
                for p in output.rglob("*")
                if p.is_file()
            }
            directories = {
                p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_dir()
            }
            if present != expected or directories != {"validation"}:
                raise FileExistsError(
                    "Existing import differs; preserve it and choose another output"
                )
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".import-", dir=output.parent) as temporary:
                staged = Path(temporary) / "run"
                staged.mkdir()
                for name, digest in expected.items():
                    destination = staged / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024**2)
                    if file_hash(destination) != digest:
                        raise ValueError("Imported file copy failed SHA256 verification")
                if output.exists():
                    raise FileExistsError("Import destination appeared during verification")
                staged.rename(output)
    return {
        "result_sha256": file_hash(artifact),
        "bundle_sha256": bundle_sha256,
        "code_sha256": result["code_sha256"],
        "files": expected,
        "completed": True,
        "test_used": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--run-dir", type=Path, required=True)
    exporting.add_argument("--bundle-manifest", type=Path, required=True)
    exporting.add_argument("--bundle-sha256", required=True)
    exporting.add_argument("--output", type=Path, required=True)
    importing = commands.add_parser("import")
    importing.add_argument("--artifact", type=Path, required=True)
    importing.add_argument("--bundle", type=Path, required=True)
    importing.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = export_results(**args) if command == "export" else import_results(**args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
