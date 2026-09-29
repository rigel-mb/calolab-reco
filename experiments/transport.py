"""Exchange the isolated parallel studies without opening the reserved test.

Only known code and train/validation archives enter the Colab bundle. Returned
checkpoints remain opaque bytes: import checks fingerprints and recomputes
metrics from bounded NumPy arrays without loading pickle or executing ZIP code.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from build_stage2_bundle import _validate_metadata, _verify_npz  # noqa: E402
from build_stage2_bundle import verify_bundle as verify_stage2  # noqa: E402

from calolab_reco.data import file_hash  # noqa: E402
from calolab_reco.metrics import regression_metrics, stratified_metrics  # noqa: E402
from calolab_reco.stage2_artifacts import (  # noqa: E402
    _arrays,
    _canonical,
    _digest,
    _json,
    _stream_hash,
)

MAX_BYTES = 512 * 1024**2
MAX_SMALL_BYTES = 2 * 1024**2
BUNDLE_MANIFEST = "PARALLEL_BUNDLE_MANIFEST.json"
SOURCE_MANIFEST = "SOURCE_STAGE2_MANIFEST.json"
RESULT_MANIFEST = "PARALLEL_RESULTS_MANIFEST.json"
BASE_FILES = ("pyproject.toml", "uv.lock", "configs/cnn.toml")
EXTRA_FILES = (
    "scripts/build_stage2_bundle.py",
    "experiments/parallel_studies.py",
    "experiments/transport.py",
    "experiments/plan.json",
)
DATA_FILES = ("data/train.npz", "data/validation.npz", "data/stage2_manifest.json")
CASES = (
    "input_log_joint",
    "input_linear_joint",
    "task_energy",
    "task_position",
    "volume_7000",
    "volume_14000",
    "readout_clean",
    "readout_cut",
    "readout_noise",
    "readout_noise_cut",
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def code_path(name: str) -> bool:
    return name in (*BASE_FILES, *EXTRA_FILES) or bool(
        re.fullmatch(r"src/calolab_reco/[A-Za-z_]\w*\.py", name)
    )


def safe_members(archive: zipfile.ZipFile, expected: set[str], limit=MAX_BYTES) -> None:
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("ZIP membership differs from its exact allowlist")
    if sum(entry.file_size for entry in entries) > limit:
        raise ValueError("ZIP uncompressed payload exceeds the size budget")
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in entry.filename
            or path.as_posix() != entry.filename
            or entry.is_dir()
            or entry.flag_bits & 1
            or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG)
        ):
            raise ValueError("ZIP members must be ordinary safe files")


def zip_info(name: str, *, stored=False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    return info


def _ordinary(root: Path, name: str, limit=MAX_SMALL_BYTES) -> Path:
    path = root / name
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or any(p.is_symlink() for p in path.parents if p.is_relative_to(root))
        or path.stat().st_size > limit
    ):
        raise ValueError(f"Expected an ordinary bounded file: {name}")
    return path


def _external(path: Path, root=PROJECT) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or candidate.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Heavy artifacts require an ordinary destination outside the project")
    return candidate.resolve()


def _plan_source(plan: Path) -> Path:
    candidate = Path(plan).expanduser().absolute()
    return _ordinary(
        Path(candidate.anchor), candidate.relative_to(candidate.anchor).as_posix()
    ).resolve()


def _local_code(root: Path, plan: Path | None = None) -> dict[str, bytes]:
    names = (
        *BASE_FILES,
        *EXTRA_FILES,
        *(p.relative_to(root).as_posix() for p in sorted((root / "src/calolab_reco").glob("*.py"))),
    )
    payloads = {name: _ordinary(root, name).read_bytes() for name in names}
    if plan is not None:
        payloads["experiments/plan.json"] = _plan_source(plan).read_bytes()
    return payloads


def _transformer_study(config: dict) -> bool:
    if config.get("study") not in {"parallel_studies", "parallel_studies_transformer"}:
        raise ValueError("Unsupported parallel study configuration")
    return config["study"] == "parallel_studies_transformer"


def _base_check(files: dict, source: dict) -> None:
    base = {
        name: value
        for name, value in files.items()
        if name in BASE_FILES or name.startswith("src/calolab_reco/")
    }
    expected = {
        name: value
        for name, value in source["files"].items()
        if name in BASE_FILES or name.startswith("src/calolab_reco/")
    }
    if base != expected or digest_bytes(_canonical(base)) != source.get("code_sha256"):
        raise ValueError("Frozen stage 2 source differs from the original bundle")
    if set(source.get("files", {})) != set(base) | set(DATA_FILES) | {"ATTRIBUTION.txt"}:
        raise ValueError("Unexpected original stage 2 membership")
    for name in (*DATA_FILES, "ATTRIBUTION.txt"):
        if files.get(name) != source["files"].get(name):
            raise ValueError("Frozen stage 2 data or attribution differs")


def verify_bundle(path: Path, project_root=None, expected_sha256=None, *, plan=None) -> dict:
    if plan is not None and project_root is None:
        raise ValueError("An alternative local plan requires a project root for verification")
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Compressed bundle exceeds the size budget")
    if expected_sha256 is not None and file_hash(path) != _digest(expected_sha256):
        raise ValueError("Bundle ZIP fingerprint mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo(BUNDLE_MANIFEST).file_size > MAX_SMALL_BYTES:
            raise ValueError("Bundle manifest exceeds the size budget")
        manifest = _json(archive.read(BUNDLE_MANIFEST))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "parallel_studies"
            or manifest.get("input_selection") != "train_validation_only"
        ):
            raise ValueError("Unsupported parallel bundle schema")
        files = manifest.get("files", {})
        required = set(BASE_FILES) | set(EXTRA_FILES) | set(DATA_FILES)
        required |= {"ATTRIBUTION.txt", SOURCE_MANIFEST}
        if not isinstance(files, dict) or not required <= files.keys():
            raise ValueError("Missing parallel bundle members")
        if any(not code_path(n) and n not in required for n in files):
            raise ValueError("Bundle file outside the source/data allowlist")
        safe_members(archive, set(files) | {BUNDLE_MANIFEST})
        for name, expected in files.items():
            if name not in DATA_FILES[:2] and archive.getinfo(name).file_size > MAX_SMALL_BYTES:
                raise ValueError("Oversized code or metadata")
            with archive.open(name) as stream:
                if _stream_hash(stream) != _digest(expected):
                    raise ValueError(f"Bundle member fingerprint mismatch: {name}")
        code = {name: value for name, value in files.items() if code_path(name)}
        if digest_bytes(_canonical(code)) != manifest.get("code_sha256"):
            raise ValueError("Bundle code fingerprint mismatch")
        config = _json(archive.read("experiments/plan.json"))
        if digest_bytes(_canonical(config)) != manifest.get("config_sha256"):
            raise ValueError("Bundle configuration fingerprint mismatch")
        _transformer_study(config)
        _base_check(files, _json(archive.read(SOURCE_MANIFEST)))
        metadata = _json(archive.read("data/stage2_manifest.json"))
        _validate_metadata(metadata)
        if metadata["data_sha256"] != {
            f"{s}.npz": files[f"data/{s}.npz"] for s in ("train", "validation")
        }:
            raise ValueError("Data metadata fingerprint mismatch")
        ids, raw_bytes = {}, 0
        for split, key in (("train", "train_count"), ("validation", "val_count")):
            with archive.open(f"data/{split}.npz") as stream:
                ids[split], size = _verify_npz(stream, metadata[key])
                raw_bytes += size
        if raw_bytes > MAX_BYTES or np.intersect1d(ids["train"], ids["validation"]).size:
            raise ValueError("Oversized raw arrays or overlapping train/validation IDs")
        _digest(manifest.get("source_stage2_zip_sha256"))
    if project_root is not None:
        actual = {
            name: digest_bytes(b)
            for name, b in _local_code(Path(project_root).resolve(), plan).items()
        }
        if code != actual:
            raise ValueError("Bundle code/config differs from the local project")
    return manifest


def build_bundle(
    project_root: Path, source_bundle: Path, output: Path, *, plan: Path | None = None
) -> dict:
    root, source = Path(project_root).resolve(), Path(source_bundle).expanduser().resolve()
    plan = _plan_source(plan) if plan is not None else None
    output = _external(output, root)
    summary_path = output.with_suffix(".summary.json")
    if output in {source, plan} or summary_path in {source, plan} or output.suffix != ".zip":
        raise ValueError("Choose a separate .zip output")
    if summary_path.is_symlink():
        raise ValueError("Bundle summary must be an ordinary file")
    stage2 = verify_stage2(source)
    payloads = _local_code(root, plan)
    with zipfile.ZipFile(source) as original:
        safe_members(original, set(stage2["files"]) | {"BUNDLE_MANIFEST.json"})
        for name in ("ATTRIBUTION.txt", "data/stage2_manifest.json"):
            payloads[name] = original.read(name)
        payloads[SOURCE_MANIFEST] = original.read("BUNDLE_MANIFEST.json")
        files = {name: digest_bytes(value) for name, value in payloads.items()}
        files.update({name: stage2["files"][name] for name in DATA_FILES[:2]})
        _base_check(files, stage2)
        code = {name: value for name, value in files.items() if code_path(name)}
        config = _json(payloads["experiments/plan.json"])
        metadata = _json(payloads["data/stage2_manifest.json"])
        manifest = {
            "schema_version": 1,
            "kind": "parallel_studies",
            "files": files,
            "code_sha256": digest_bytes(_canonical(code)),
            "config_sha256": digest_bytes(_canonical(config)),
            "source_stage2_zip_sha256": file_hash(source),
            "input_selection": "train_validation_only",
            "data_copy": "Original train/validation NPZ bytes copied unchanged",
        }
        payloads[BUNDLE_MANIFEST] = _canonical(manifest) + b"\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".parallel-build-", dir=output.parent) as tmp:
            staged = Path(tmp) / "bundle.zip"
            with zipfile.ZipFile(staged, "w") as archive:
                for name in sorted(set(files) | {BUNDLE_MANIFEST}):
                    if name in payloads:
                        archive.writestr(zip_info(name), payloads[name], compresslevel=6)
                    else:
                        with (
                            original.open(name) as src,
                            archive.open(zip_info(name, stored=True), "w") as dst,
                        ):
                            shutil.copyfileobj(src, dst, length=8 * 1024**2)
            verify_bundle(staged, root, plan=plan)
            if output.exists() and file_hash(output) != file_hash(staged):
                raise FileExistsError("Different bundle exists; choose a new output")
            if not output.exists():
                os.replace(staged, output)
    summary = {
        "schema_version": 1,
        "zip_filename": output.name,
        "zip_sha256": file_hash(output),
        "zip_bytes": output.stat().st_size,
        "code_sha256": manifest["code_sha256"],
        "config_sha256": manifest["config_sha256"],
        "source_stage2_zip_sha256": manifest["source_stage2_zip_sha256"],
        "train_count": metadata["train_count"],
        "validation_count": metadata["val_count"],
        "input_selection": "train_validation_only",
    }
    summary_bytes = _canonical(summary) + b"\n"
    if summary_path.exists() and summary_path.read_bytes() != summary_bytes:
        raise FileExistsError("Different bundle summary exists; choose a new output")
    if not summary_path.exists():
        with summary_path.open("xb") as stream:
            stream.write(summary_bytes)
    return summary


def _extract_payloads(payloads: dict[str, bytes], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("Extraction requires a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".parallel-extract-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "content"
        staged.mkdir()
        for name, content in payloads.items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        os.replace(staged, destination)


def extract_bundle(path: Path, destination: Path, expected_sha256: str) -> dict:
    destination = _external(destination)
    manifest = verify_bundle(path, expected_sha256=expected_sha256)
    with zipfile.ZipFile(path) as archive:
        payloads = {name: archive.read(name) for name in (*manifest["files"], BUNDLE_MANIFEST)}
    # The extraction uses the same bytes whose individual fingerprints were checked.
    for name, content in payloads.items():
        if name != BUNDLE_MANIFEST and digest_bytes(content) != manifest["files"][name]:
            raise ValueError("Bundle changed during extraction")
    _extract_payloads(payloads, destination)
    return manifest


def _result_path(name: str) -> bool:
    if name in {"summary.json", "progress.log"}:
        return True
    parts = PurePosixPath(name).parts
    if len(parts) != 2 or parts[0] not in CASES:
        return False
    return parts[1] in {
        "result.json",
        "history.json",
        "sample.npz",
        "last.pt",
        "last.pt.sha256",
        "predictions_baseline.npz",
        "predictions_periodic.npz",
        *(
            f"{kind}_{task}.{suffix}"
            for task in ("joint", "energy", "position")
            for kind, suffix in (("best", "pt"), ("best", "pt.sha256"), ("predictions", "npz"))
        ),
    }


def _aggregate_only(value) -> None:
    if isinstance(value, dict):
        if {"source_ids", "targets", "predictions", "deposits"} & value.keys():
            raise ValueError("Aggregate report contains per-event arrays")
        for child in value.values():
            _aggregate_only(child)
    elif isinstance(value, list):
        for child in value:
            _aggregate_only(child)


def export_results(run_dir: Path, output: Path) -> dict:
    root, output = Path(run_dir).expanduser().resolve(), _external(output)
    if output.is_relative_to(root) or output.suffix != ".zip":
        raise ValueError("Choose a .zip output outside the run directory")
    summary = _json(_ordinary(root, "summary.json").read_bytes())
    if summary.get("kind") != "parallel_studies" or summary.get("test_used") is not False:
        raise ValueError("Not a parallel train/validation run")
    payloads = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_dir() and not path.is_symlink():
            continue
        if path.suffix in {".tmp", ".partial"} and _result_path(name.removesuffix(path.suffix)):
            if path.is_symlink():
                raise ValueError("Temporary result files must not be symbolic links")
            # A terminated write is not a resumable checkpoint. Retain only the
            # last atomically committed files and their matching fingerprints.
            continue
        if not _result_path(name):
            raise ValueError(f"Run file outside result allowlist: {name}")
        payloads[name] = _ordinary(root, name, MAX_BYTES).read_bytes()
    if sum(map(len, payloads.values())) > MAX_BYTES:
        raise ValueError("Result payload exceeds the size budget")
    manifest = {
        "schema_version": 1,
        "kind": "parallel_studies",
        "complete": summary["complete"],
        "files": {name: digest_bytes(value) for name, value in payloads.items()},
    }
    payloads[RESULT_MANIFEST] = _canonical(manifest) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".parallel-export-", dir=output.parent) as tmp:
        staged = Path(tmp) / "results.zip"
        with zipfile.ZipFile(staged, "w") as archive:
            for name, value in sorted(payloads.items()):
                archive.writestr(zip_info(name), value, compresslevel=6)
        _read_results(staged)
        if output.exists() and file_hash(output) != file_hash(staged):
            raise FileExistsError("Different result ZIP exists; choose a new output")
        if not output.exists():
            os.replace(staged, output)
    return {
        "zip_sha256": file_hash(output),
        "zip_bytes": output.stat().st_size,
        "complete": summary["complete"],
    }


def _read_results(path: Path) -> tuple[dict, dict]:
    if Path(path).stat().st_size > MAX_BYTES:
        raise ValueError("Compressed results exceed the size budget")
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo(RESULT_MANIFEST).file_size > MAX_SMALL_BYTES:
            raise ValueError("Oversized result manifest")
        manifest = _json(archive.read(RESULT_MANIFEST))
        files = manifest.get("files", {})
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "parallel_studies"
            or type(manifest.get("complete")) is not bool
            or not isinstance(files, dict)
            or "summary.json" not in files
            or any(not _result_path(n) for n in files)
        ):
            raise ValueError("Invalid result manifest or result membership")
        safe_members(archive, set(files) | {RESULT_MANIFEST})
        payloads = {}
        for name, expected in files.items():
            limit = MAX_SMALL_BYTES if name.endswith((".json", ".sha256")) else MAX_BYTES
            if archive.getinfo(name).file_size > limit:
                raise ValueError("Oversized result member")
            content = archive.read(name)
            if digest_bytes(content) != _digest(expected):
                raise ValueError(f"Result fingerprint mismatch: {name}")
            payloads[name] = content
        for name, content in payloads.items():
            if name.endswith(".pt.sha256"):
                if (
                    name[:-7] not in payloads
                    or _digest(content.decode().strip()) != files[name[:-7]]
                ):
                    raise ValueError("Checkpoint sidecar fingerprint mismatch")
        for name in payloads:
            if name.endswith(".pt") and f"{name}.sha256" not in payloads:
                raise ValueError("Checkpoint sidecar is missing")
    return manifest, payloads


def _same(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Recomputed fields differ: {label}")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Recomputed list differs: {label}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same(left, right, f"{label}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (
            type(actual) not in (int, float)
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
        ):
            raise ValueError(f"Recomputed value differs: {label}")
    elif actual != expected:
        raise ValueError(f"Recomputed metadata differs: {label}")


def _metrics(targets, predictions, task):
    values = regression_metrics(targets, predictions)
    if task == "joint":
        return values
    return {
        key: value for key, value in values.items() if key == "count" or key.startswith(f"{task}_")
    }


def _subgroups(targets, predictions, task):
    def filtered(value):
        if isinstance(value, dict):
            if "energy_mare" in value:
                return {
                    key: child
                    for key, child in value.items()
                    if task == "joint" or key == "count" or key.startswith(f"{task}_")
                }
            return {key: filtered(child) for key, child in value.items()}
        if isinstance(value, list):
            return [filtered(child) for child in value]
        return value

    return filtered(stratified_metrics(targets, predictions))


def _sample(content, count):
    shapes = {"source_ids": (count, 2), "targets": (count, 3)}
    result = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        safe_members(archive, {f"{key}.npy" for key in shapes}, MAX_SMALL_BYTES)
        for key, shape in shapes.items():
            name = f"{key}.npy"
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }.get(version)
                if reader is None:
                    raise ValueError("Unsupported sample NPY version")
                actual, _, dtype = reader(stream)
                if (
                    actual != shape
                    or dtype.hasobject
                    or (key == "targets" and dtype != np.float32)
                    or (key == "source_ids" and dtype.kind not in "iu")
                    or stream.tell() + math.prod(shape) * dtype.itemsize
                    != archive.getinfo(name).file_size
                ):
                    raise ValueError("Invalid train sample header")
            with archive.open(name) as stream:
                result[key] = np.lib.format.read_array(stream, allow_pickle=False)
    if any(not np.isfinite(array).all() for array in result.values()):
        raise ValueError("Nonfinite train sample")
    return result


def verify_results(archive_path: Path, bundle_path: Path) -> dict:
    """Verify source/data identities and independently recompute validation errors."""
    bundle = verify_bundle(bundle_path)
    manifest, payloads = _read_results(archive_path)
    summary = _json(payloads["summary.json"])
    if (
        summary.get("schema_version") != 1
        or summary.get("kind") != "parallel_studies"
        or summary.get("test_used") is not False
        or summary.get("complete") != manifest["complete"]
        or type(summary.get("smoke")) is not bool
    ):
        raise ValueError("Invalid parallel run summary")
    _aggregate_only(summary)
    with zipfile.ZipFile(bundle_path) as archive:
        config = _json(archive.read("experiments/plan.json"))
        metadata = _json(archive.read("data/stage2_manifest.json"))
        arrays = {}
        for split, key in (("train", "train_count"), ("validation", "val_count")):
            with archive.open(f"data/{split}.npz") as stream:
                arrays[split] = _arrays(stream, metadata[key], predictions=False)
    transformer = _transformer_study(config)
    provenance = {
        "source_files": {n: v for n, v in bundle["files"].items() if code_path(n)},
        "code_sha256": bundle["code_sha256"],
        "config_sha256": bundle["config_sha256"],
        "data_sha256": metadata["data_sha256"],
        "stage2_manifest_sha256": bundle["files"]["data/stage2_manifest.json"],
        "prepared_manifest_sha256": metadata["prepared_manifest_sha256"],
        "prepared_output_sha256": metadata["prepared_output_sha256"],
    }
    effective = dict(config)
    if summary["smoke"]:
        effective.update(
            updates=4,
            evaluate_every=2,
            checkpoint_every=2,
            log_every=1,
            effective_batch_size=8,
            microbatch_size=4,
            volume_counts=[8, 16],
            training_seconds_ceiling=120,
        )
    provenance["effective_config_sha256"] = digest_bytes(_canonical(effective))
    if (
        summary.get("provenance") != provenance
        or summary.get("config") != config
        or summary.get("effective_config") != effective
    ):
        raise ValueError("Run source/config/data provenance mismatch")
    cases = summary.get("cases", {})
    if not isinstance(cases, dict) or set(cases) - set(CASES):
        raise ValueError("Unexpected experiment cases")
    if summary["complete"] and set(cases) != set(CASES):
        raise ValueError("Complete run lacks approved cases")
    reference = arrays["validation"]
    train = arrays["train"]
    if summary["smoke"]:
        reference = {key: value[:16] for key, value in reference.items()}
        train = {key: value[:32] for key, value in train.items()}
    train_ids = train["source_ids"]
    lookup = {tuple(row): index for index, row in enumerate(train_ids)}
    for name, case in cases.items():
        if case.get("case", {}).get("name") != name or type(case.get("complete")) is not bool:
            raise ValueError("Case identity or completion mismatch")
        spec = case["case"]
        if (transformer and spec.get("architecture") != "transformer") or (
            not transformer and "architecture" in spec
        ):
            raise ValueError("Case architecture differs from the approved study")
        if summary["complete"] and not case["complete"]:
            raise ValueError("Incomplete case in a complete suite")
        if case.get("provenance") != provenance:
            raise ValueError("Case source/config/data provenance mismatch")
        if not case["complete"]:
            # A hard interruption can leave newer atomic checkpoints than the
            # last suite summary. Their bytes/sidecars were checked above, but
            # no scientific result is inferred from an unfinished opaque state.
            continue
        if _json(payloads[f"{name}/result.json"]) != case:
            raise ValueError("Case result differs from suite summary")
        expected_artifacts = {
            path: value
            for path, value in manifest["files"].items()
            if path.startswith(f"{name}/") and not path.endswith("/result.json")
        }
        if case.get("artifacts") != expected_artifacts:
            raise ValueError("Case artifacts membership differs from the result ZIP")
        for path, expected in case.get("artifacts", {}).items():
            if not path.startswith(f"{name}/") or manifest["files"].get(path) != expected:
                raise ValueError("Case artifact fingerprint mismatch")
        _verify_case(
            name,
            case,
            effective,
            payloads,
            reference,
            train_ids,
            lookup,
            train["targets"],
            cases,
        )
    return {
        "summary": summary,
        "complete": summary["complete"],
        "payloads": payloads,
        "manifest": manifest,
        "smoke": summary["smoke"],
    }


def _verify_case(name, case, config, payloads, reference, train_ids, lookup, train_targets, cases):
    count = case.get("train_count")
    if type(count) is not int or not 1 <= count <= len(train_ids):
        raise ValueError("Invalid case training count")
    sample = _sample(payloads[f"{name}/sample.npz"], count)
    keys = [tuple(row) for row in sample["source_ids"]]
    if len(set(keys)) != count or any(key not in lookup for key in keys):
        raise ValueError("Train sample identifiers are duplicated or outside train")
    expected = train_targets[[lookup[key] for key in keys]]
    if not np.array_equal(sample["targets"], expected):
        raise ValueError("Train sample targets differ from original data")
    ids_hash = digest_bytes(np.ascontiguousarray(sample["source_ids"], dtype="<i8").tobytes())
    if case.get("train_ids_sha256") != ids_hash:
        raise ValueError("Train sample identifier fingerprint mismatch")
    expected_count = len(train_ids)
    if name.startswith("volume_"):
        expected_count = config["volume_counts"][0 if name == "volume_7000" else 1]
    if count != expected_count or case.get("validation_count") != len(reference["targets"]):
        raise ValueError("Case event counts differ from the declared experiment")
    local = name.startswith("readout_")
    if name in {"input_log_joint", "input_linear_joint"}:
        input_kind = name.split("_")[1]
    elif local:
        input_kind = "linear"
    else:
        winner = min(
            ("input_log_joint", "input_linear_joint"),
            key=lambda key: cases[key]["selected"]["joint"]["score"],
        )
        input_kind = cases[winner]["case"]["input"]
    expected_case = {
        "name": name,
        "kind": "local" if local else "full",
        "input": input_kind,
        "task": name.removeprefix("task_") if name.startswith("task_") else "joint",
        "regime": name.removeprefix("readout_") if local else "clean",
        "train_size": expected_count,
    }
    if _transformer_study(config):
        expected_case["architecture"] = "transformer"
    if case["case"] != expected_case:
        raise ValueError("Case configuration differs from the approved experiment")
    indices = np.random.default_rng(config["sampling_seed"]).permutation(len(train_ids))[:count]
    expected_ids = train_ids if local else train_ids[indices]
    if not np.array_equal(sample["source_ids"], expected_ids):
        raise ValueError("Train sample differs from deterministic sampling")
    val_hash = digest_bytes(np.ascontiguousarray(reference["source_ids"], dtype="<i8").tobytes())
    if case.get("validation_ids_sha256") != val_hash:
        raise ValueError("Validation identifier fingerprint mismatch")
    selected = case.get("selected", {})
    task = case["case"].get("task")
    selections = {"joint", "energy", "position"} if task == "joint" else {task}
    if task not in {"joint", "energy", "position"} or set(selected) != selections:
        raise ValueError("Case selection tasks differ")
    history = case.get("history", [])
    if not history or any(type(row.get("update")) is not int for row in history):
        raise ValueError("Invalid training history")
    updates = [row["update"] for row in history]
    if updates != sorted(set(updates)) or updates[0] < 1:
        raise ValueError("Nonconsecutive evaluation updates")
    if (
        updates[-1] != config["updates"]
        or case.get("updates_completed") != config["updates"]
        or updates
        != list(range(config["evaluate_every"], config["updates"] + 1, config["evaluate_every"]))
    ):
        raise ValueError("Complete case did not finish the update budget")
    if _json(payloads[f"{name}/history.json"]) != {"history": history}:
        raise ValueError("Case history differs from its independent file")
    for selection, result in selected.items():
        checkpoint = f"{name}/best_{selection}.pt"
        prediction = f"{name}/predictions_{selection}.npz"
        if result.get("checkpoint") != checkpoint or result.get("prediction") != prediction:
            raise ValueError("Selected checkpoint/prediction path mismatch")
        if checkpoint not in payloads:
            raise ValueError("Selected checkpoint is missing")
        if result.get("checkpoint_sha256") != digest_bytes(payloads[checkpoint]) or result.get(
            "prediction_sha256"
        ) != digest_bytes(payloads[prediction]):
            raise ValueError("Selected checkpoint/prediction fingerprint mismatch")
        arrays = _arrays(io.BytesIO(payloads[prediction]), case["validation_count"], True)
        for key in ("targets", "source_ids"):
            if not np.array_equal(arrays[key], reference[key][: len(arrays[key])]):
                raise ValueError(f"Validation {key} differ from original data")
        metrics = _metrics(arrays["targets"], arrays["predictions"], task)
        _same(result.get("metrics"), metrics, f"{name}.{selection}")
        _same(
            result.get("subgroups"),
            _subgroups(arrays["targets"], arrays["predictions"], task),
            f"{name}.{selection}.subgroups",
        )
        rows = {row["update"]: row for row in history}
        if result.get("update") not in rows:
            raise ValueError("Selected update is missing from history")
        _same(rows[result["update"]]["metrics"], metrics, f"{name}.selected_history")

        def score(row, selection=selection):
            metrics = row["metrics"]
            if selection == "joint":
                return metrics["energy_mare"] / 0.1 + metrics["position_distance_median"]
            return metrics["energy_mare" if selection == "energy" else "position_distance_median"]

        best = min(history, key=score)
        if result["update"] != best["update"]:
            raise ValueError("Selected checkpoint is not the earliest minimum validation score")
        _same(result.get("score"), score(best), "checkpoint selection score")
    for key, filename in (
        ("baseline", "predictions_baseline.npz"),
        ("periodic_baseline", "predictions_periodic.npz"),
    ):
        if local != (case.get(key) is not None):
            raise ValueError("Matched local references are required for every readout case")
        if not local:
            continue
        baseline = case[key]
        path = f"{name}/{filename}"
        if baseline.get("prediction") != path or baseline.get("prediction_sha256") != digest_bytes(
            payloads[path]
        ):
            raise ValueError("Baseline prediction path mismatch")
        arrays = _arrays(io.BytesIO(payloads[path]), case["validation_count"], True)
        for key in ("targets", "source_ids"):
            if not np.array_equal(arrays[key], reference[key][: len(arrays[key])]):
                raise ValueError("Baseline validation identity mismatch")
        _same(
            baseline.get("metrics"),
            _metrics(arrays["targets"], arrays["predictions"], task),
            f"{name}.{key}",
        )
        _same(
            baseline.get("subgroups"),
            _subgroups(arrays["targets"], arrays["predictions"], task),
            f"{name}.{key}.subgroups",
        )


def import_results(archive_path, bundle_path, output, report=None, expected_sha256=None) -> dict:
    output = _external(output)
    if expected_sha256 is not None and file_hash(Path(archive_path)) != _digest(expected_sha256):
        raise ValueError("Returned ZIP fingerprint mismatch")
    result = verify_results(Path(archive_path), Path(bundle_path))
    if report is not None and (not result["complete"] or result["smoke"]):
        raise ValueError("Only a complete scientific run can produce a public report")
    report = Path(report).expanduser() if report is not None else None
    if report is not None and (report.exists() or report.is_symlink()):
        raise FileExistsError("Report already exists; review before replacing")
    _extract_payloads(
        {**result["payloads"], RESULT_MANIFEST: _canonical(result["manifest"])}, output
    )
    if report is not None:
        summary = dict(result["summary"])
        summary["import_verification"] = {
            "archive_sha256": file_hash(Path(archive_path)),
            "bundle_sha256": file_hash(Path(bundle_path)),
            "validation_metrics_recomputed": True,
            "train_validation_ids_targets_verified": True,
            "checkpoint_bytes_verified": True,
            "checkpoints_deserialized": False,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("xb") as stream:
            stream.write(_canonical(summary) + b"\n")
    return {
        "complete": result["complete"],
        "smoke": result["smoke"],
        "report_written": report is not None,
        "archive_sha256": file_hash(Path(archive_path)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-bundle", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--plan", type=Path, help="Alternative plan to package as experiments/plan.json"
    )
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--sha256", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--bundle", type=Path, required=True)
    extract.add_argument("--sha256", required=True)
    extract.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    imp = commands.add_parser("import")
    for argument in ("archive", "bundle", "output"):
        imp.add_argument(f"--{argument}", type=Path, required=True)
    imp.add_argument("--report", type=Path)
    imp.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "build":
        result = build_bundle(PROJECT, args.source_bundle, args.output, plan=args.plan)
    elif args.command == "verify":
        result = verify_bundle(args.bundle, expected_sha256=args.sha256)
    elif args.command == "extract":
        result = extract_bundle(args.bundle, args.output, args.sha256)
    elif args.command == "export":
        result = export_results(args.run_dir, args.output)
    else:
        result = import_results(args.archive, args.bundle, args.output, args.report, args.sha256)
    print(_canonical(result).decode())


if __name__ == "__main__":
    main()
