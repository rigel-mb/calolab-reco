"""Package raw train/validation data and allowlisted code for stage 2 Colab runs.

Code identity is the SHA256 of the canonical UTF-8 JSON path/hash mapping:
sorted keys and compact separators. Data, attribution and the bundle manifest
are excluded from code identity. All archive timestamps and permissions are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from calolab_reco.data import PREPARED_FILES, file_hash, load_prepared
from calolab_reco.data import data_root as default_data_root

MAX_BUNDLE_BYTES = 512 * 1024**2
MAX_CODE_FILE_BYTES = 2 * 1024**2
CODE_FILES = ("pyproject.toml", "uv.lock", "configs/cnn.toml")
DATA_FILES = ("data/train.npz", "data/validation.npz", "data/stage2_manifest.json")
ARRAY_NAMES = ("deposits", "targets", "source_ids")
SOURCE_RECORD = "https://zenodo.org/records/18929909"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
ATTRIBUTION = """Dataset: GEANT4 Simulation Dataset for a CMS-Like PbWO4 Electromagnetic Calorimeter
Authors: Y. Maidannyk and M. O. Sahin
Source: https://zenodo.org/records/18929909 (version 0.0.1)
License: Creative Commons Attribution 4.0 International
https://creativecommons.org/licenses/by/4.0/
Associated article: https://doi.org/10.1140/epjc/s10052-026-16097-x

Changes: selected the prepared training and validation partitions; repackaged raw
float32 deposits and targets with source group/row identifiers. No numerical unit
conversion, noise, threshold or coordinate correction was applied. Test data are
excluded. Preprocessing statistics were fitted on the full training partition only.
"""


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stream_hash(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024**2), b""):
        digest.update(block)
    return digest.hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _is_code_path(name: str) -> bool:
    return (
        name in CODE_FILES or re.fullmatch(r"src/calolab_reco/[A-Za-z_][\w]*\.py", name) is not None
    )


def _code_payloads(root: Path) -> dict[str, bytes]:
    modules = sorted((root / "src/calolab_reco").glob("*.py"))
    if not modules:
        raise ValueError("No package modules found")
    files = {}
    for name in (*CODE_FILES, *(p.relative_to(root).as_posix() for p in modules)):
        path = root / name
        if not _is_code_path(name) or path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f"Bundle source must be an ordinary allowlisted project file: {name}")
        if any(parent.is_symlink() for parent in path.parents if parent != root):
            raise ValueError(f"Bundle source has a symbolic-link parent: {name}")
        if not path.is_file():
            raise ValueError(f"Required bundle source is missing: {name}")
        if path.stat().st_size > MAX_CODE_FILE_BYTES:
            raise ValueError(f"Code file exceeds the bundle budget: {name}")
        files[name] = path.read_bytes()
    return files


def _validate_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Unexpected data array keys")
    deposits, targets, ids = (arrays[name] for name in ARRAY_NAMES)
    count = len(deposits)
    if count == 0 or deposits.shape != (count, 30, 85) or targets.shape != (count, 3):
        raise ValueError("Expected nonempty grids (N,30,85) and targets (N,3)")
    if deposits.dtype != np.float32 or targets.dtype != np.float32:
        raise ValueError("Raw deposits and targets must retain float32 values")
    if ids.shape != (count, 2) or ids.dtype.kind not in "iu":
        raise ValueError("Expected integer source identifiers (N,2)")
    if not np.isfinite(targets).all() or (targets[:, 0] <= 0).any():
        raise ValueError("Targets must be finite with positive incident energy")
    for start in range(0, count, 512):
        block = deposits[start : start + 512]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError("Deposits must be finite and nonnegative")
        if not (block > 0).any(axis=(1, 2)).all():
            raise ValueError("All-zero events are not supported")
    return _unique_ids(ids)


def _unique_ids(ids: np.ndarray) -> np.ndarray:
    if ids.dtype.kind not in "iu" or ids.ndim != 2 or ids.shape[1] != 2 or (ids < 0).any():
        raise ValueError("Source identifiers must be nonnegative integer pairs")
    pairs = np.asarray(ids, dtype=np.uint64)
    keys = np.ascontiguousarray(pairs).view("V16").ravel()
    unique = np.unique(keys)
    if len(unique) != len(keys):
        raise ValueError("Duplicate source identifiers within a partition")
    return unique


def _training_metadata(train: dict[str, np.ndarray]) -> dict:
    deposits = train["deposits"]
    positive_parts = []
    for start in range(0, len(deposits), 512):
        block = deposits[start : start + 512]
        positive_parts.append(block[block > 0])
    positive = np.concatenate(positive_parts)
    del positive_parts
    scale = float(np.median(positive, overwrite_input=True))
    targets = train["targets"]
    mean = targets.mean(axis=0, dtype=np.float64)
    std = targets.std(axis=0, dtype=np.float64, ddof=0)
    if (std <= 0).any():
        raise ValueError("Every training target must have positive standard deviation")
    return {
        "input_scale": scale,
        "input_scale_method": "median of all strictly positive deposits in full train",
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "target_std_ddof": 0,
        "preprocessing_fit_selection": "full_train_only",
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(ARRAY_NAMES):
            with archive.open(_zip_info(f"{name}.npy"), "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, arrays[name], allow_pickle=False)


def _verify_npz(stream, expected_count: int) -> tuple[np.ndarray, int]:
    """Inspect bounded array headers and identifiers, without loading deposit grids."""
    with tempfile.TemporaryFile() as copied:
        shutil.copyfileobj(stream, copied, length=8 * 1024**2)
        copied.seek(0)
        with zipfile.ZipFile(copied) as archive:
            members = archive.infolist()
            names = [m.filename for m in members]
            if len(names) != 3 or set(names) != {f"{n}.npy" for n in ARRAY_NAMES}:
                raise ValueError("Unexpected or duplicate NPZ members")
            raw_bytes = sum(m.file_size for m in members)
            if raw_bytes > MAX_BUNDLE_BYTES:
                raise ValueError("NPZ raw payload exceeds the size budget")
            shapes = {
                "deposits.npy": (expected_count, 30, 85),
                "targets.npy": (expected_count, 3),
                "source_ids.npy": (expected_count, 2),
            }
            for member in members:
                with archive.open(member) as array_stream:
                    version = np.lib.format.read_magic(array_stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(array_stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(array_stream)
                    else:
                        raise ValueError("Unsupported NPY header version")
                    payload_size = int(np.prod(shape)) * dtype.itemsize
                    if shape != shapes[member.filename] or dtype.hasobject:
                        raise ValueError("NPZ array shape or dtype differs from its contract")
                    if member.filename != "source_ids.npy" and dtype != np.float32:
                        raise ValueError("NPZ deposits/targets must be raw float32")
                    if member.filename == "source_ids.npy" and dtype.kind not in "iu":
                        raise ValueError("NPZ source identifiers must be integers")
                    if array_stream.tell() + payload_size != member.file_size:
                        raise ValueError("NPZ payload length differs from its header")
            with archive.open("source_ids.npy") as array_stream:
                ids = np.lib.format.read_array(array_stream, allow_pickle=False)
    return _unique_ids(ids), raw_bytes


def _validate_metadata(metadata: dict) -> None:
    if metadata["schema_version"] != 1 or metadata["input_selection"] != "train_validation_only":
        raise ValueError("Unsupported stage 2 data manifest")
    if any(type(metadata[k]) is not int or metadata[k] <= 0 for k in ("train_count", "val_count")):
        raise ValueError("Partition counts must be positive integers")
    if not np.isfinite(metadata["input_scale"]) or metadata["input_scale"] <= 0:
        raise ValueError("Input scale must be finite and positive")
    mean, std = np.asarray(metadata["target_mean"]), np.asarray(metadata["target_std"])
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite([mean, std]).all():
        raise ValueError("Invalid target standardization metadata")
    if (std <= 0).any() or metadata["target_std_ddof"] != 0:
        raise ValueError("Invalid target standard deviations")
    if set(metadata["prepared_output_sha256"]) != set(PREPARED_FILES):
        raise ValueError("Prepared provenance must fingerprint all four prepared files")
    hashes = [metadata["prepared_manifest_sha256"], *metadata["prepared_output_sha256"].values()]
    if any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None for h in hashes):
        raise ValueError("Invalid prepared provenance digest")


def verify_bundle(path: Path) -> dict:
    """Check exact allowlist, hashes, bounded NPZ schemas and disjoint unique IDs."""
    path = Path(path)
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Compressed bundle exceeds the size budget")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(names) != len(set(names)) or sum(m.file_size for m in members) > MAX_BUNDLE_BYTES:
            raise ValueError("Duplicate ZIP members or oversized bundle payload")
        if archive.getinfo("BUNDLE_MANIFEST.json").file_size > MAX_CODE_FILE_BYTES:
            raise ValueError("Oversized bundle manifest")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported bundle schema")
        if set(names) != set(manifest["files"]) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("Bundle members differ from the fingerprinted allowlist")
        required = set(CODE_FILES) | set(DATA_FILES) | {"ATTRIBUTION.txt"}
        if not required <= set(names) or not any(n.startswith("src/") for n in names):
            raise ValueError("Required bundle files are missing")
        code_hashes = {}
        for name, expected in manifest["files"].items():
            if _is_code_path(name):
                code_hashes[name] = expected
            elif name not in (*DATA_FILES, "ATTRIBUTION.txt"):
                raise ValueError(f"Member outside bundle allowlist: {name}")
            if name not in DATA_FILES[:2] and archive.getinfo(name).file_size > MAX_CODE_FILE_BYTES:
                raise ValueError(f"Oversized code or metadata file: {name}")
            with archive.open(name) as stream:
                if _stream_hash(stream) != expected:
                    raise ValueError(f"Bundle member failed SHA256 verification: {name}")
        if _sha256(_json_bytes(code_hashes)) != manifest["code_sha256"]:
            raise ValueError("Bundle code identity failed verification")
        metadata = json.loads(archive.read("data/stage2_manifest.json"))
        _validate_metadata(metadata)
        expected_data = {
            f"{split}.npz": manifest["files"][f"data/{split}.npz"]
            for split in ("train", "validation")
        }
        if metadata["data_sha256"] != expected_data:
            raise ValueError("Stage 2 data identity failed verification")
        with archive.open("data/train.npz") as stream:
            train_ids, train_bytes = _verify_npz(stream, metadata["train_count"])
        with archive.open("data/validation.npz") as stream:
            val_ids, val_bytes = _verify_npz(stream, metadata["val_count"])
        if train_bytes + val_bytes > MAX_BUNDLE_BYTES:
            raise ValueError("Combined raw data exceed the size budget")
        if np.intersect1d(train_ids, val_ids).size:
            raise ValueError("Train and validation source identifiers overlap")
    return manifest


def build_bundle(project_root: Path, data_root: Path, output: Path, force: bool = False) -> dict:
    """Export full train and validation, fitting preprocessing only on train."""
    root = Path(project_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if root == output or root in output.parents:
        raise ValueError("The bundle must remain outside the project")
    if output.suffix != ".zip":
        raise ValueError("Bundle output must use the .zip suffix")
    code = _code_payloads(root)
    code_hashes = {name: _sha256(content) for name, content in code.items()}
    code_sha256 = _sha256(_json_bytes(code_hashes))
    prepared_path = Path(data_root).expanduser() / "derived/audit_v1/manifest.json"
    prepared_bytes = prepared_path.read_bytes()
    prepared_manifest = json.loads(prepared_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage2-", dir=output.parent) as temporary:
        staging = Path(temporary)
        data_hashes, partition_ids, counts = {}, {}, {}
        raw_bytes = 0
        metadata = {}
        for split in ("train", "validation"):
            arrays = load_prepared(data_root, split=split)
            raw_bytes += sum(array.nbytes for array in arrays.values())
            if raw_bytes + sum(map(len, code.values())) + 65536 > MAX_BUNDLE_BYTES:
                raise ValueError("Combined raw data and code exceed the size budget")
            partition_ids[split] = _validate_arrays(arrays)
            counts[split] = len(arrays["deposits"])
            if split == "train":
                metadata.update(_training_metadata(arrays))
            _write_npz(staging / f"{split}.npz", arrays)
            data_hashes[f"{split}.npz"] = file_hash(staging / f"{split}.npz")
            del arrays
        if np.intersect1d(partition_ids["train"], partition_ids["validation"]).size:
            raise ValueError("Train and validation source identifiers overlap")
        if prepared_path.read_bytes() != prepared_bytes:
            raise ValueError("Prepared manifest changed while loading data")
        metadata.update(
            {
                "schema_version": 1,
                "input_selection": "train_validation_only",
                "train_count": counts["train"],
                "val_count": counts["validation"],
                "prepared_manifest_sha256": _sha256(prepared_bytes),
                "prepared_output_sha256": prepared_manifest["output_sha256"],
                "data_sha256": data_hashes,
                "source_record": SOURCE_RECORD,
                "coordinate_names": ["local_fractional_iphi", "local_fractional_ieta"],
                "energy_unit_interpretation": {
                    "deposits": "MeV",
                    "incident_energy": "GeV",
                    "status": "inferred_not_producer_certified",
                    "values": "unchanged_as_stored",
                },
            }
        )
        _validate_metadata(metadata)
        payloads = {
            **code,
            "ATTRIBUTION.txt": ATTRIBUTION.encode(),
            "data/stage2_manifest.json": _json_bytes(metadata) + b"\n",
        }
        files = {name: _sha256(content) for name, content in payloads.items()}
        files.update({f"data/{name}": digest for name, digest in data_hashes.items()})
        manifest = {
            "schema_version": 1,
            "files": files,
            "code_sha256": code_sha256,
            "code_sha256_scope": "pyproject.toml, uv.lock, configs/cnn.toml, src/calolab_reco/*.py",
            "code_sha256_method": (
                "SHA256 of UTF-8 JSON path/hash map; sorted keys; compact separators"
            ),
        }
        payloads["BUNDLE_MANIFEST.json"] = _json_bytes(manifest) + b"\n"
        staged_zip = staging / "bundle.zip"
        with zipfile.ZipFile(staged_zip, "w") as archive:
            for name in sorted(set(files) | {"BUNDLE_MANIFEST.json"}):
                if name in payloads:
                    archive.writestr(_zip_info(name), payloads[name], compresslevel=6)
                else:
                    with (staging / Path(name).name).open("rb") as src:
                        with archive.open(_zip_info(name), "w", force_zip64=True) as dst:
                            shutil.copyfileobj(src, dst, length=8 * 1024**2)
        verify_bundle(staged_zip)
        digest = file_hash(staged_zip)
        existing = file_hash(output) if output.exists() else None
        if existing is not None and existing != digest and not force:
            raise FileExistsError("Existing bundle differs; inspect it or explicitly use --force")
        summary = {
            "schema_version": 1,
            "zip_filename": output.name,
            "zip_sha256": digest,
            "zip_bytes": staged_zip.stat().st_size,
            "raw_data_bytes": raw_bytes,
            "code_sha256": code_sha256,
            "input_selection": "train_validation_only",
            "train_count": counts["train"],
            "val_count": counts["validation"],
        }
        if existing != digest:
            os.replace(staged_zip, output)
        output.with_suffix(".summary.json").write_bytes(_json_bytes(summary) + b"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Data/cache/calolab-reco/colab/calolab-reco-stage2.zip",
    )
    parser.add_argument("--force", action="store_true", help="Replace a different existing bundle")
    args = parser.parse_args()
    print(
        json.dumps(
            build_bundle(
                Path(__file__).resolve().parents[1], args.data_root, args.output, args.force
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
