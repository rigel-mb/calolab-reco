"""Bounded inspection and reproducible sampling of the one-photon NPZ files.

Values are retained as stored. Unit and coordinate interpretation is a separate,
explicit audit decision, never an implicit conversion in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ARCHIVES = {
    "calo": ("1photon_calo.npz", "X", (30, 85)),
    "energy": ("1photon_en.npz", "en", ()),
    "position": ("1photon_yc.npz", "yc", (2,)),
}
MAX_MEMBER_BYTES = 512 * 1024**2
SPLIT_NAMES = {"train": 0, "validation": 1, "test": 2}
PREPARED_FILES = ("deposits.npy", "targets.npy", "source_ids.npy", "split.npy")


def data_root() -> Path:
    default = Path.home() / "Data" / "public" / "calolab-reco"
    return Path(os.environ.get("CALOLAB_DATA_ROOT", default)).expanduser().resolve()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inventory_archives(raw_dir: Path, max_member_bytes: int = MAX_MEMBER_BYTES) -> dict:
    """Validate NPY headers and event alignment without materializing arrays."""
    raw_dir = Path(raw_dir)
    descriptions = {}
    for role, (name, prefix, tail_shape) in ARCHIVES.items():
        groups = {}
        with zipfile.ZipFile(raw_dir / name) as archive:
            members = archive.infolist()
            if len({member.filename for member in members}) != len(members):
                raise ValueError(f"Duplicate ZIP member name in {name}")
            for member in members:
                match = re.fullmatch(rf"{prefix}_(\d+)\.npy", member.filename)
                if match is None:
                    raise ValueError(f"Unexpected member in {name}: {member.filename}")
                group_id = int(match.group(1))
                if group_id in groups:
                    raise ValueError(f"Duplicate numeric group in {name}: {group_id}")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"Unsupported NPY header version: {version}")
                if dtype.hasobject or dtype.kind not in "fiu":
                    raise ValueError(f"Unsupported array dtype in {member.filename}: {dtype}")
                if len(shape) != 1 + len(tail_shape) or shape[1:] != tail_shape or shape[0] <= 0:
                    raise ValueError(f"Unexpected shape in {member.filename}: {shape}")
                payload_bytes = math.prod(shape) * dtype.itemsize
                if payload_bytes > max_member_bytes or member.file_size > max_member_bytes + 65536:
                    raise ValueError(f"Member exceeds memory budget: {member.filename}")
                groups[group_id] = {
                    "key": member.filename.removesuffix(".npy"),
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "fortran_order": bool(fortran_order),
                    "payload_bytes": payload_bytes,
                    "compressed_bytes": member.compress_size,
                }
        if not groups:
            raise ValueError(f"Empty archive: {name}")
        descriptions[role] = groups
    group_ids = set(descriptions["calo"])
    if any(set(groups) != group_ids for groups in descriptions.values()):
        raise ValueError("Archive group identifiers do not match")
    aligned = []
    for group_id in sorted(group_ids):
        sizes = {groups[group_id]["shape"][0] for groups in descriptions.values()}
        if len(sizes) != 1:
            raise ValueError(f"Event counts do not match in group {group_id}")
        aligned.append(
            {
                "group_id": group_id,
                "n_events": sizes.pop(),
                "keys": {role: groups[group_id]["key"] for role, groups in descriptions.items()},
                "arrays": {role: groups[group_id] for role, groups in descriptions.items()},
            }
        )
    return {
        "groups": aligned,
        "n_events": sum(group["n_events"] for group in aligned),
        "grid_shape_stored": [30, 85],
        "largest_array_bytes": max(
            array["payload_bytes"] for groups in descriptions.values() for array in groups.values()
        ),
        "total_decompressed_array_bytes": sum(
            array["payload_bytes"] for groups in descriptions.values() for array in groups.values()
        ),
    }


def _quantiles(values: np.ndarray) -> dict | None:
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    q = np.quantile(values, [0, 0.01, 0.5, 0.99, 1])
    return dict(zip(["min", "p01", "median", "p99", "max"], q.tolist(), strict=True))


def _quality_and_observations(deposits: np.ndarray, targets: np.ndarray, split: np.ndarray):
    quality = {
        "nonfinite_deposit_values": 0,
        "negative_deposit_values": 0,
        "all_zero_events": 0,
        "nonfinite_target_values": int((~np.isfinite(targets)).sum()),
        "nonpositive_incident_energy_events": int((targets[:, 0] <= 0).sum()),
    }
    totals = np.empty(len(deposits), dtype=np.float64)
    occupancies = np.empty(len(deposits), dtype=np.int64)
    positive_min = math.inf
    for start in range(0, len(deposits), 1000):
        block = deposits[start : start + 1000]
        finite = np.isfinite(block)
        quality["nonfinite_deposit_values"] += int((~finite).sum())
        quality["negative_deposit_values"] += int((block < 0).sum())
        quality["all_zero_events"] += int((block == 0).all(axis=(1, 2)).sum())
        totals[start : start + len(block)] = block.sum(axis=(1, 2), dtype=np.float64)
        occupancies[start : start + len(block)] = (block > 0).sum(axis=(1, 2))
        train_block = block[split[start : start + len(block)] == 0]
        positive = train_block[np.isfinite(train_block) & (train_block > 0)]
        if positive.size:
            positive_min = min(positive_min, float(positive.min()))
    train = split == 0
    valid_ratio = train & np.isfinite(targets[:, 0]) & (targets[:, 0] > 0)
    observations = {
        "population": "training partition only; no reconstruction model evaluated",
        "incident_energy_as_stored": _quantiles(targets[train, 0]),
        "coordinate_0_as_stored": _quantiles(targets[train, 1]),
        "coordinate_1_as_stored": _quantiles(targets[train, 2]),
        "deposit_sum_as_stored": _quantiles(totals[train]),
        "deposit_sum_over_incident_as_stored": _quantiles(
            totals[valid_ratio] / targets[valid_ratio, 0]
        ),
        "positive_cells_per_event": _quantiles(occupancies[train]),
        "smallest_positive_deposit_as_stored": None if math.isinf(positive_min) else positive_min,
    }
    return quality, observations


def prepare_sample(
    data_root: Path,
    sample_size: int = 40000,
    sample_seed: int = 20260920,
    split_seed: int = 20260921,
    max_member_bytes: int = MAX_MEMBER_BYTES,
) -> dict:
    """Retain raw values, sample globally and keep exact duplicate maps together."""
    root = Path(data_root).expanduser().resolve()
    package_root = Path(__file__).resolve().parents[2]
    if root == package_root or package_root in root.parents:
        raise ValueError("The data directory must be outside the project")
    raw = root / "raw"
    inventory = inventory_archives(raw, max_member_bytes=max_member_bytes)
    if any(
        array["dtype"] != "float32"
        for group in inventory["groups"]
        for array in group["arrays"].values()
    ):
        raise ValueError("Preparation currently requires float32 sources to preserve stored values")
    if not 0 < sample_size <= inventory["n_events"]:
        raise ValueError(
            "sample_size must be positive and no greater than the available population"
        )
    max_derived_bytes = 1024**3
    required_bytes = sample_size * (30 * 85 * 4 + 3 * 4 + 2 * 8 + 1)
    if required_bytes > max_derived_bytes:
        raise ValueError("Sample exceeds the one GiB derived-data budget")
    parameters = {"sample_size": sample_size, "sample_seed": sample_seed, "split_seed": split_seed}
    source_hashes = {name: file_hash(raw / name) for name, _, _ in ARCHIVES.values()}
    download_manifest = raw / "download_manifest.json"
    if download_manifest.exists():
        verified = json.loads(download_manifest.read_text())
        expected = {entry["name"]: entry["sha256"] for entry in verified["files"]}
        if source_hashes != expected:
            raise ValueError("Sources no longer match the verified download manifest")
    destination = root / "derived" / "audit_v1"
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["parameters"] != parameters or manifest["source_sha256"] != source_hashes:
            raise ValueError(
                "Existing prepared data use different parameters or sources; not overwritten"
            )
        return verify_prepared(root)
    staging = destination.with_name("audit_v1.partial")
    if staging.exists():
        raise ValueError("Partial preparation exists; inspect it before restarting")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required_bytes + 1024**3:
        raise RuntimeError("Insufficient disk space for the prepared sample")
    staging.mkdir()
    selected = np.sort(
        np.random.default_rng(sample_seed).choice(inventory["n_events"], sample_size, replace=False)
    )
    deposits = np.lib.format.open_memmap(
        staging / "deposits.npy", mode="w+", dtype="<f4", shape=(sample_size, 30, 85)
    )
    targets = np.lib.format.open_memmap(
        staging / "targets.npy", mode="w+", dtype="<f4", shape=(sample_size, 3)
    )
    identifiers = np.lib.format.open_memmap(
        staging / "source_ids.npy", mode="w+", dtype="<i8", shape=(sample_size, 2)
    )
    # Numeric group ordering is used explicitly; ZIP order is never an event identifier.
    with ExitStack() as stack:
        archives = {
            role: stack.enter_context(np.load(raw / name, allow_pickle=False))
            for role, (name, _, _) in ARCHIVES.items()
        }
        offset = 0
        for group in inventory["groups"]:
            left, right = np.searchsorted(selected, [offset, offset + group["n_events"]])
            local_indices = selected[left:right] - offset
            if len(local_indices):
                for role in ARCHIVES:
                    values = archives[role][group["keys"][role]]
                    chosen = values[local_indices]
                    if role == "calo":
                        deposits[left:right] = chosen
                    elif role == "energy":
                        targets[left:right, 0] = chosen
                    else:
                        targets[left:right, 1:] = chosen
                    del values, chosen
                identifiers[left:right, 0] = group["group_id"]
                identifiers[left:right, 1] = local_indices
            offset += group["n_events"]
    for array in (deposits, targets, identifiers):
        array.flush()
    row_hashes = np.array(
        [hashlib.blake2b(row.tobytes(order="C"), digest_size=16).digest() for row in deposits],
        dtype="V16",
    )
    _, inverse, counts = np.unique(row_hashes, return_inverse=True, return_counts=True)
    group_splits = np.random.default_rng(split_seed).choice(
        3, size=len(counts), p=[0.70, 0.15, 0.15]
    )
    split = group_splits[inverse].astype(np.uint8)
    np.save(staging / "split.npy", split, allow_pickle=False)
    conflicting_groups = 0
    for group_id in np.flatnonzero(counts > 1):
        labels = targets[inverse == group_id]
        conflicting_groups += int(not np.all(labels == labels[0]))
    quality, observations = _quality_and_observations(deposits, targets, split)
    del deposits, targets, identifiers
    manifest = {
        "schema_version": 1,
        "parameters": parameters,
        "source_sha256": source_hashes,
        "inventory": inventory,
        "target_columns": ["incident_energy_raw", "coordinate_0", "coordinate_1"],
        "source_id_columns": ["npz_group_id", "row_in_group"],
        "source_id_limit": "Storage provenance, not a certified physical-event identifier",
        "split_counts": {name: int((split == value).sum()) for name, value in SPLIT_NAMES.items()},
        "duplicate_extra_events": int((counts - 1).sum()),
        "duplicate_groups_with_different_targets": conflicting_groups,
        "duplicate_control_limit": "Exact stored deposit maps within the selected sample only",
        "quality": quality,
        "observations": observations,
        "transformations": "None: source float32 values retained in stored axis and target order",
        "training_gate": {
            "ready": False,
            "reasons": [
                "Review stored energy units and coordinate conventions before learning",
                "Review the stage 1 notebook and data quality with the user",
                "Colab GPU and timing pilot have not been verified",
            ],
        },
        "output_sha256": {name: file_hash(staging / name) for name in PREPARED_FILES},
    }
    _write_json(staging / "manifest.json", manifest)
    staging.rename(destination)
    return manifest


def verify_prepared(data_root: Path) -> dict:
    """Reject incomplete or changed numerical files before consuming any partition."""
    root = Path(data_root).expanduser() / "derived" / "audit_v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("output_sha256", {})
    if set(hashes) != set(PREPARED_FILES):
        raise ValueError("Prepared manifest must fingerprint all four expected numerical files")
    for name in PREPARED_FILES:
        if not (root / name).is_file() or file_hash(root / name) != hashes[name]:
            raise ValueError(f"Prepared output failed verification: {name}")
    return manifest


def load_prepared(data_root: Path, split: str = "train", allow_test: bool = False) -> dict:
    """Read one named partition. Final-test access requires an explicit opt-in."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    if split == "test" and not allow_test:
        raise PermissionError("The final test is reserved; explicit allow_test=True is required")
    verify_prepared(data_root)
    root = Path(data_root).expanduser() / "derived" / "audit_v1"
    assignment = np.load(root / "split.npy", mmap_mode="r", allow_pickle=False)
    selected = assignment == SPLIT_NAMES[split]
    return {
        key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False)[selected]
        for key in ("deposits", "targets", "source_ids")
    }
