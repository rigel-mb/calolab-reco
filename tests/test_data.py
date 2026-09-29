"""Synthetic checks for archive alignment, sampling and reserved-test protection."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from calolab_reco.data import inventory_archives, load_prepared, prepare_sample

GroupArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


def _make_groups() -> dict[int, GroupArrays]:
    """Use distinguishable raw deposits, labels and deliberately unordered groups."""
    groups = {}
    for group_id, count in ((10, 80), (2, 40)):
        index = np.arange(count, dtype=np.float32)
        deposits = np.zeros((count, 30, 85), dtype=np.float32)
        deposits[:, 3, 4] = group_id + 1 + index / 8
        deposits[:, 20, 77] = (index + 1) / 32
        energy = 20 + group_id + index / 2
        position = np.column_stack((index / 10, -index / 5)).astype(np.float32)
        groups[group_id] = deposits, energy, position
    return groups


def _write_archives(raw_dir: Path, groups: dict[int, GroupArrays]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, prefix, column in (
        ("1photon_calo.npz", "X", 0),
        ("1photon_en.npz", "en", 1),
        ("1photon_yc.npz", "yc", 2),
    ):
        arrays = {f"{prefix}_{group_id}": values[column] for group_id, values in groups.items()}
        np.savez_compressed(raw_dir / filename, **arrays)


def _read_outputs(data_root: Path) -> dict[str, np.ndarray]:
    output_dir = data_root / "derived" / "audit_v1"
    return {
        name: np.load(output_dir / f"{name}.npy", allow_pickle=False)
        for name in ("deposits", "targets", "source_ids", "split")
    }


def _assert_raw_values_preserved(
    outputs: dict[str, np.ndarray], groups: dict[int, GroupArrays]
) -> None:
    assert outputs["deposits"].shape[1:] == (30, 85)
    assert outputs["targets"].shape == (len(outputs["deposits"]), 3)
    assert outputs["source_ids"].shape == (len(outputs["deposits"]), 2)
    assert np.issubdtype(outputs["source_ids"].dtype, np.integer)
    for row, (group_id, source_row) in enumerate(outputs["source_ids"]):
        deposits, energy, position = groups[int(group_id)]
        np.testing.assert_array_equal(outputs["deposits"][row], deposits[source_row])
        np.testing.assert_array_equal(
            outputs["targets"][row], np.r_[energy[source_row], position[source_row]]
        )


def _raw_checksums(raw_dir: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_dir.iterdir()}


def test_inventory_reads_headers_and_sorts_groups_numerically(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    _write_archives(raw_dir, _make_groups())

    def forbid_array_loading(*args, **kwargs):
        pytest.fail("Inventory must inspect headers without loading full arrays")

    monkeypatch.setattr(np, "load", forbid_array_loading)
    groups = inventory_archives(raw_dir)["groups"]

    assert [group["group_id"] for group in groups] == [2, 10]
    assert [group["n_events"] for group in groups] == [40, 80]
    assert groups[0]["keys"] == {"calo": "X_2", "energy": "en_2", "position": "yc_2"}


@pytest.mark.parametrize("filename", ["1photon_calo.npz", "1photon_en.npz", "1photon_yc.npz"])
def test_inventory_requires_all_three_archives(tmp_path, filename):
    _write_archives(tmp_path, _make_groups())
    (tmp_path / filename).unlink()

    with pytest.raises((FileNotFoundError, ValueError)):
        inventory_archives(tmp_path)


@pytest.mark.parametrize("invalid_kind", ["grid_shape", "energy_length", "position_shape"])
def test_inventory_rejects_misaligned_shapes(tmp_path, invalid_kind):
    groups = _make_groups()
    deposits, energy, position = groups[2]
    if invalid_kind == "grid_shape":
        deposits = deposits[:, :, :-1]
    elif invalid_kind == "energy_length":
        energy = energy[:-1]
    else:
        position = position[:, :1]
    groups[2] = deposits, energy, position
    _write_archives(tmp_path, groups)

    with pytest.raises(ValueError):
        inventory_archives(tmp_path)


@pytest.mark.parametrize("invalid_kind", ["missing_group", "wrong_prefix", "duplicate_group_id"])
def test_inventory_rejects_ambiguous_or_misaligned_keys(tmp_path, invalid_kind):
    groups = _make_groups()
    _write_archives(tmp_path, groups)
    arrays = {"en_2": groups[2][1], "en_10": groups[10][1]}
    if invalid_kind == "missing_group":
        arrays.pop("en_2")
    elif invalid_kind == "wrong_prefix":
        arrays["energy_2"] = arrays.pop("en_2")
    else:
        arrays["en_02"] = groups[2][1]
    np.savez_compressed(tmp_path / "1photon_en.npz", **arrays)

    with pytest.raises(ValueError):
        inventory_archives(tmp_path)


def test_inventory_rejects_duplicate_zip_members(tmp_path):
    _write_archives(tmp_path, _make_groups())
    archive_path = tmp_path / "1photon_en.npz"
    with ZipFile(archive_path) as archive:
        member = archive.read("en_2.npy")
    with ZipFile(archive_path, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("en_2.npy", member)

    with pytest.raises(ValueError):
        inventory_archives(tmp_path)


@pytest.mark.parametrize("dtype", [object, np.complex64, "U12"])
def test_inventory_rejects_nonreal_numeric_arrays(tmp_path, dtype):
    groups = _make_groups()
    deposits, energy, position = groups[2]
    groups[2] = deposits, energy.astype(dtype), position
    _write_archives(tmp_path, groups)

    with pytest.raises((TypeError, ValueError)):
        inventory_archives(tmp_path)


def test_inventory_checks_uncompressed_member_budget(tmp_path):
    _write_archives(tmp_path, _make_groups())
    # Mostly zero grids compress well but must still respect their decoded size.
    with pytest.raises(ValueError):
        inventory_archives(tmp_path, max_member_bytes=4096)


def test_sampling_is_reproducible_unique_and_preserves_raw_values(tmp_path):
    groups = _make_groups()
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    for root in (first_root, second_root):
        _write_archives(root / "raw", groups)
        prepare_sample(root, sample_size=100, sample_seed=41, split_seed=73)

    first, second = _read_outputs(first_root), _read_outputs(second_root)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert len(first["deposits"]) == 100
    assert len(np.unique(first["source_ids"], axis=0)) == 100
    assert set(first["source_ids"][:, 0]) == {2, 10}
    assert set(first["split"]) == {0, 1, 2}
    _assert_raw_values_preserved(first, groups)


def test_preparation_preserves_raw_archives_and_reuses_identical_request(tmp_path):
    _write_archives(tmp_path / "raw", _make_groups())
    before = _raw_checksums(tmp_path / "raw")
    prepare_sample(tmp_path, sample_size=80)
    first_outputs = _read_outputs(tmp_path)

    prepare_sample(tmp_path, sample_size=80)

    assert _raw_checksums(tmp_path / "raw") == before
    for name, values in _read_outputs(tmp_path).items():
        np.testing.assert_array_equal(values, first_outputs[name])


@pytest.mark.parametrize("changed_kwargs", [{"sample_size": 60}, {"sample_seed": 99}])
def test_preparation_rejects_incompatible_overwrite(tmp_path, changed_kwargs):
    _write_archives(tmp_path / "raw", _make_groups())
    prepare_sample(tmp_path, sample_size=80)
    original = _read_outputs(tmp_path)
    kwargs = {"sample_size": 80, **changed_kwargs}

    with pytest.raises((FileExistsError, ValueError)):
        prepare_sample(tmp_path, **kwargs)

    for name, values in _read_outputs(tmp_path).items():
        np.testing.assert_array_equal(values, original[name])


def test_preparation_rejects_sample_larger_than_available(tmp_path):
    _write_archives(tmp_path / "raw", _make_groups())
    with pytest.raises(ValueError):
        prepare_sample(tmp_path, sample_size=121)


def test_duplicate_deposits_stay_in_one_partition_even_when_labels_differ(tmp_path):
    groups = _make_groups()
    deposits, energy, position = groups[10]
    other_deposits, other_energy, other_position = groups[2]
    deposits[0] = other_deposits[0]
    energy[0], position[0] = other_energy[0], other_position[0]
    deposits[1] = other_deposits[1]
    assert energy[1] != other_energy[1]
    _write_archives(tmp_path / "raw", groups)
    prepare_sample(tmp_path, sample_size=120)
    outputs = _read_outputs(tmp_path)

    partitions_by_deposit = {}
    for deposit, partition in zip(outputs["deposits"], outputs["split"], strict=True):
        partitions_by_deposit.setdefault(deposit.tobytes(), set()).add(int(partition))
    assert len(partitions_by_deposit) == 118
    assert all(len(partitions) == 1 for partitions in partitions_by_deposit.values())
    assert len(np.unique(outputs["source_ids"], axis=0)) == 120
    _assert_raw_values_preserved(outputs, groups)


def test_quality_failures_are_reported_without_silent_event_removal(tmp_path):
    groups = _make_groups()
    deposits, energy, position = groups[2]
    deposits[0, 3, 4] = -1
    deposits[1, 3, 4] = np.nan
    deposits[2] = 0
    energy[3] = 0
    position[4, 0] = np.nan
    _write_archives(tmp_path / "raw", groups)

    manifest = prepare_sample(tmp_path, sample_size=120)
    outputs = _read_outputs(tmp_path)

    assert len(outputs["deposits"]) == 120
    _assert_raw_values_preserved(outputs, groups)
    assert manifest["quality"]["nonfinite_deposit_values"] == 1
    assert manifest["quality"]["negative_deposit_values"] == 1
    assert manifest["quality"]["all_zero_events"] == 1
    assert manifest["quality"]["nonfinite_target_values"] == 1
    assert manifest["quality"]["nonpositive_incident_energy_events"] == 1
    assert manifest["training_gate"]["ready"] is False
    saved = json.loads((tmp_path / "derived" / "audit_v1" / "manifest.json").read_text())
    assert saved["quality"] == manifest["quality"]


@pytest.mark.parametrize("name", ["deposits", "targets", "source_ids", "split"])
def test_loading_rejects_modified_prepared_payload(tmp_path, name):
    _write_archives(tmp_path / "raw", _make_groups())
    prepare_sample(tmp_path, sample_size=120)
    output_path = tmp_path / "derived" / "audit_v1" / f"{name}.npy"
    values = np.load(output_path, mmap_mode="r+", allow_pickle=False)
    if name == "split":
        # A valid-looking reassignment must not expose a reserved test event to train.
        test_row = np.flatnonzero(values == 2)[0]
        values[test_row] = 0
    else:
        # Keep the NPY header, dtype and shape valid so only integrity checks catch it.
        values.flat[0] += 1
    values.flush()
    del values

    with pytest.raises(ValueError):
        load_prepared(tmp_path, split="train")


def test_loading_requires_a_fingerprint_for_every_prepared_output(tmp_path):
    _write_archives(tmp_path / "raw", _make_groups())
    prepare_sample(tmp_path, sample_size=120)
    manifest_path = tmp_path / "derived" / "audit_v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"].pop("targets.npy")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        load_prepared(tmp_path, split="train")


def test_load_prepared_limits_access_to_requested_partition(tmp_path):
    _write_archives(tmp_path / "raw", _make_groups())
    prepare_sample(tmp_path, sample_size=120)
    outputs = _read_outputs(tmp_path)

    for split_name, split_code in (("train", 0), ("validation", 1), ("test", 2)):
        loaded = load_prepared(tmp_path, split=split_name, allow_test=split_name == "test")
        for key in ("deposits", "targets", "source_ids"):
            np.testing.assert_array_equal(loaded[key], outputs[key][outputs["split"] == split_code])

    with pytest.raises((PermissionError, ValueError)):
        load_prepared(tmp_path, split="test")


def test_load_prepared_rejects_unknown_partition(tmp_path):
    _write_archives(tmp_path / "raw", _make_groups())
    prepare_sample(tmp_path, sample_size=120)
    with pytest.raises(ValueError):
        load_prepared(tmp_path, split="all")
