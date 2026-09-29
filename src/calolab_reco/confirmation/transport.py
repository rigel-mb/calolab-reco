"""Checksummed, bounded Colab exchange; bundles contain train and validation only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA = "603c4be32c20a40edd0b3d0b0e8e20d5175e7c4b73a4d9706da0983a929b1933"
LIMIT = 768 * 1024**2


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def check_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or p.as_posix() != name:
        raise ValueError("Unsafe archive name")


def read_archive(path, kind):
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        names = z.namelist()
        if len(set(names)) != len(names) or "MANIFEST.json" not in names:
            raise ValueError("Missing manifest or duplicate archive member")
        if sum(x.file_size for x in entries) > LIMIT:
            raise ValueError("Archive exceeds uncompressed budget")
        for entry in entries:
            check_name(entry.filename)
            if (
                entry.is_dir()
                or entry.flag_bits & 1
                or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG)
            ):
                raise ValueError("Archive contains a nonregular member")
        manifest = json.loads(z.read("MANIFEST.json"))
        if manifest["kind"] != kind or set(names) != set(manifest["files"]) | {"MANIFEST.json"}:
            raise ValueError("Manifest membership or kind differs")
        payload = {k: z.read(k) for k in manifest["files"]}
        if any(digest(v) != manifest["files"][k] for k, v in payload.items()):
            raise ValueError("Archive checksum differs")
    return manifest, payload


def write_archive(path, payload, kind, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        kind=kind, files={k: digest(v) for k, v in payload.items()}, metadata=metadata or {}
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, value in {**payload, "MANIFEST.json": canonical(manifest)}.items():
                check_name(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, value)
        read_archive(temp, kind)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return manifest


def copy_verified(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".partial")
    shutil.copyfile(source, temp)
    if sha(source) != sha(temp):
        raise ValueError("Backup checksum differs")
    temp.replace(target)


def external(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("Keep data and results outside the repository")
    return path


def build(source, output):
    source = Path(source)
    output = external(output)
    if sha(source) != SOURCE_SHA:
        raise ValueError("Expected the verified methodology input bundle")
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(source) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("parallel.zip"))) as z:
            payload = {
                f"data/{split}.npz": z.read(f"data/{split}.npz")
                for split in ("train", "validation")
            }
            payload["ATTRIBUTION.txt"] = z.read("ATTRIBUTION.txt")
    code = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "configs/confirmation.json",
        *sorted((ROOT / "src/calolab_reco").glob("*.py")),
        *sorted((ROOT / "src/calolab_reco/confirmation").glob("*.py")),
    ]
    payload.update({p.relative_to(ROOT).as_posix(): p.read_bytes() for p in code})
    write_archive(
        output,
        payload,
        "confirmation_input",
        dict(source_bundle_sha256=SOURCE_SHA, test_used=False),
    )
    validate_input(output)
    return dict(sha256=sha(output), bytes=output.stat().st_size)


def validate_input(bundle):
    manifest, payload = read_archive(bundle, "confirmation_input")
    allowed = {
        "pyproject.toml",
        "uv.lock",
        "configs/confirmation.json",
        "ATTRIBUTION.txt",
        "data/train.npz",
        "data/validation.npz",
    }
    import re

    if any(
        k not in allowed
        and not re.fullmatch(r"src/calolab_reco/(?:confirmation/)?[A-Za-z_]\w*\.py", k)
        for k in payload
    ):
        raise ValueError("Unexpected input member")
    if not allowed.issubset(payload):
        raise ValueError("Input bundle is incomplete")
    ids = []
    for split in ("train", "validation"):
        with np.load(io.BytesIO(payload[f"data/{split}.npz"]), allow_pickle=False) as a:
            x, y, i = (a[k] for k in ("deposits", "targets", "source_ids"))
        if x.shape != (len(y), 30, 85) or y.shape != (len(x), 3) or i.shape != (len(x), 2):
            raise ValueError("Invalid input shapes")
        if (
            not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or (x < 0).any()
            or (y[:, 0] <= 0).any()
        ):
            raise ValueError("Invalid input values")
        pairs = set(map(tuple, i.tolist()))
        if len(pairs) != len(i):
            raise ValueError("Duplicate source IDs")
        ids.append(pairs)
    if ids[0] & ids[1]:
        raise ValueError("Train and validation IDs overlap")
    return manifest, payload


def unpack(bundle, workspace):
    manifest, payload = validate_input(bundle)
    workspace = external(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        if all(
            (workspace / k).is_file() and sha(workspace / k) == digest(v)
            for k, v in payload.items()
        ):
            return workspace
        raise FileExistsError("Existing workspace differs from the bundle")
    workspace.mkdir(parents=True, exist_ok=True)
    for name, value in payload.items():
        p = workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value)
    write_json(workspace / "INPUT_MANIFEST.json", manifest)
    return workspace


def load_raw(workspace, smoke=False):
    workspace = Path(workspace)
    manifest = json.loads((workspace / "INPUT_MANIFEST.json").read_text())
    for name, h in manifest["files"].items():
        if sha(workspace / name) != h:
            raise ValueError(f"Workspace changed: {name}")
    raw = {}
    for split, n in [("train", 32), ("validation", 16)]:
        with np.load(workspace / f"data/{split}.npz", allow_pickle=False) as a:
            raw[split] = {k: a[k].copy() for k in ("deposits", "targets", "source_ids")}
        if smoke:
            raw[split] = {k: v[:n] for k, v in raw[split].items()}
    return raw, manifest


def export_results(run, output):
    run = Path(run)
    output = external(output)
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((run / "summary.json").read_text())
    paths = [run / "summary.json", run / "protocol.json"]
    paths += sorted((run / "cases").glob("*/checkpoint.zip"))
    paths += sorted((run / "predictions").glob("*.npz"))
    paths += sorted((run / "figures").glob("*.png"))
    if (run / "RESULTS.md").exists():
        paths.append(run / "RESULTS.md")
    if (run / "recovery_verification.json").exists():
        paths.append(run / "recovery_verification.json")
    return write_archive(
        output,
        {p.relative_to(run).as_posix(): p.read_bytes() for p in paths},
        "confirmation_results",
        dict(
            input_identity=summary["input_identity"],
            complete=summary["complete"],
            smoke=summary["smoke"],
        ),
    )


def verify_results(bundle, result):
    from .data import aggregate_metrics
    from .training import cases

    input_manifest, payload = validate_input(bundle)
    manifest, files = read_archive(result, "confirmation_results")
    summary = json.loads(files["summary.json"])
    protocol = json.loads(payload["configs/confirmation.json"])
    if summary["input_identity"] != digest(canonical(input_manifest)) or summary["test_used"]:
        raise ValueError("Results use a different input or the reserved test")
    if json.loads(files["protocol.json"]) != protocol or summary["protocol"] != protocol:
        raise ValueError("Result protocol differs")
    expected = {c["name"]: c for c in cases(protocol)}
    if set(summary["cases"]) - set(expected):
        raise ValueError("Unexpected result case")
    with np.load(io.BytesIO(payload["data/validation.npz"]), allow_pickle=False) as a:
        targets, ids = a["targets"], a["source_ids"]
    if summary["smoke"]:
        targets, ids = targets[:16], ids[:16]
    for name, record in summary["cases"].items():
        if record["case"] != expected[name]:
            raise ValueError("Case identity differs")
        checkpoint = f"cases/{name}/checkpoint.zip"
        if checkpoint not in files or digest(files[checkpoint]) != record["checkpoint_sha256"]:
            raise ValueError("Missing or different checkpoint")
        updates = (
            2
            if summary["smoke"]
            else protocol[
                "pretraining_updates" if expected[name]["phase"] == "pretraining" else "updates"
            ]
        )
        if record["complete"] != (record["update"] == updates):
            raise ValueError("Update count differs")
        snapshot, _ = read_archive(io.BytesIO(files[checkpoint]), "confirmation_checkpoint")
        identity = snapshot["metadata"]["identity"]
        if (
            identity["case"] != expected[name]
            or identity["input_identity"] != summary["input_identity"]
        ):
            raise ValueError("Checkpoint input or case differs")
        if snapshot["metadata"]["update"] != record["update"]:
            raise ValueError("Checkpoint update differs")
        if expected[name]["phase"] == "finetuning":
            parent = summary["cases"].get(expected[name]["parent"])
            if not parent or not parent["complete"]:
                raise ValueError("Fine-tuning has no completed shared parent")
            if record["transfer"]["encoder_sha256"] != parent["encoder_sha256"]:
                raise ValueError("Fine-tuning encoder provenance differs")
        if expected[name]["phase"] != "pretraining" and record["complete"]:
            if not record.get("selected"):
                raise ValueError("Completed specialist has no selected checkpoint")
            metric = (
                "energy_mare" if expected[name]["task"] == "energy" else "position_distance_median"
            )
            best = min(record["history"], key=lambda h: h["metrics"][metric])
            if best["update"] != record["selected"]["update"]:
                raise ValueError("Checkpoint selection differs from validation history")
    expected_predictions = set()
    for regime in summary["references"]:
        if regime not in [protocol["primary_regime"], *protocol["control_regimes"]]:
            raise ValueError("Unknown reference condition")
        seeds = [protocol["validation_noise_seed"]] + (
            [] if regime == "clean" else protocol["validation_repeat_seeds"]
        )
        expected_predictions.update(
            f"predictions/{regime}_{ref}_{seed}.npz"
            for ref in ("affine", "periodic", "quadratic")
            for seed in seeds
        )
    for name, record in summary["cases"].items():
        if record.get("selected"):
            seeds = [protocol["validation_noise_seed"]] + (
                [] if record["case"]["regime"] == "clean" else protocol["validation_repeat_seeds"]
            )
            expected_predictions.update(f"predictions/{name}_{seed}.npz" for seed in seeds)
    if set(summary["predictions"]) != expected_predictions:
        raise ValueError("Missing or unexpected prediction records")
    for name, record in summary["predictions"].items():
        if name not in files or digest(files[name]) != record["sha256"]:
            raise ValueError("Missing prediction")
        with np.load(io.BytesIO(files[name]), allow_pickle=False) as a:
            y, p, i = a["targets"], a["predictions"], a["source_ids"]
        if not np.array_equal(y, targets) or not np.array_equal(i, ids) or not np.isfinite(p).all():
            raise ValueError("Validation identity or prediction values differ")
        actual = aggregate_metrics(y, p, record["task"])
        if set(actual) != set(record["metrics"]) or any(
            not np.isclose(actual[k], record["metrics"][k], rtol=1e-9, atol=1e-9) for k in actual
        ):
            raise ValueError("Prediction metrics do not reproduce")
    complete = len(summary["cases"]) == len(expected) and all(
        r["complete"] for r in summary["cases"].values()
    )
    if summary["complete"] != complete:
        raise ValueError("Incorrect completion claim")
    return dict(
        complete=complete,
        smoke=summary["smoke"],
        cases=len(summary["cases"]),
        expected_cases=len(expected),
        predictions=len(summary["predictions"]),
        metrics_recomputed=True,
        test_used=False,
        archive_sha256=sha(result),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "unpack", "export", "verify"])
    p.add_argument("--source", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--run", type=Path)
    p.add_argument("--result", type=Path)
    a = p.parse_args()
    if a.command == "build":
        result = build(a.source, a.output)
    elif a.command == "unpack":
        result = str(unpack(a.bundle, a.output))
    elif a.command == "export":
        manifest = export_results(a.run, a.output)
        result = dict(
            **manifest["metadata"],
            files=len(manifest["files"]),
            sha256=sha(a.output),
            bytes=a.output.stat().st_size,
        )
    else:
        result = verify_results(a.bundle, a.result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
