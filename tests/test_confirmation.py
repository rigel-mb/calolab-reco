"""Numerical continuity, information boundaries and exact phase recovery."""

import copy
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from calolab_reco.confirmation import data, models, reporting, training, transport

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/confirmation.json").read_text())


def events(n, group=0):
    rng = np.random.default_rng(711 + group)
    x = np.zeros((n, 30, 85), np.float32)
    row = rng.integers(0, 30, n)
    col = rng.integers(0, 85, n)
    energy = rng.uniform(1, 60, n).astype(np.float32)
    x[np.arange(n), row, col] = energy * 900
    x[np.arange(n), (row + 1).clip(0, 29), col] += energy * 60
    return dict(
        deposits=x,
        targets=np.column_stack((energy, row + 0.2, col + 0.3)).astype(np.float32),
        source_ids=np.column_stack((np.full(n, group), np.arange(n))).astype(np.int64),
    )


def test_matrix_one_shared_pretraining_per_seed():
    cases = training.cases(PROTOCOL)
    assert len(cases) == 29
    pre = [c for c in cases if c["phase"] == "pretraining"]
    assert len(pre) == 3
    for p in pre:
        children = [c for c in cases if c["parent"] == p["name"]]
        assert {c["task"] for c in children} == {"energy", "position"}
    assert all(
        c["seed"] == PROTOCOL["training_seeds"][0] for c in cases if c["regime"] != "noise_cut"
    )


@pytest.mark.parametrize("architecture", ["cnn", "transformer"])
def test_supervised_model_and_inputs_match_exploratory_pipeline(architecture):
    spec = importlib.util.spec_from_file_location(
        "frozen_parallel", ROOT / "experiments/parallel_studies.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    raw, val = events(16), events(8, 1)
    new = data.prepare(raw, val, "noise_cut", PROTOCOL)
    original = old.prepare_readout(raw, val, "noise_cut", PROTOCOL)
    assert new[2] == original[2]
    torch.testing.assert_close(new[0]["inputs"], original[0]["inputs"], rtol=0, atol=0)
    torch.manual_seed(11)
    m = models.DirectRegressor("local", new[2], architecture)
    torch.manual_seed(11)
    o = old.DirectRegressor("local", original[2], architecture)
    x, a = new[1]["inputs"], new[1]["anchors"]
    torch.testing.assert_close(m(x, a), o(x, a), rtol=0, atol=0)


def test_validation_truth_does_not_change_input_or_calibration():
    raw, val = events(16), events(8, 1)
    first = data.prepare(raw, val, "noise_cut", PROTOCOL)
    val["targets"] *= 3
    second = data.prepare(raw, val, "noise_cut", PROTOCOL)
    assert first[2:] == second[2:]
    torch.testing.assert_close(first[1]["inputs"], second[1]["inputs"], rtol=0, atol=0)
    torch.testing.assert_close(first[1]["anchors"], second[1]["anchors"], rtol=0, atol=0)


def test_mask_padding_hidden_value_and_zero_distinction():
    x = torch.zeros(3, 2, 7, 7)
    x[:, 1] = 1
    x[0, 1, :3] = 0
    x[:, 0, 4, 4] = 1
    mask = models.sample_mask(x, 0.5, torch.Generator().manual_seed(17))
    assert not (mask & ~x[:, 1].flatten(1).bool()).any()
    model = models.LocalMaskedModel().eval()
    changed = x.clone()
    changed[:, 0].view(3, -1)[mask] = 123
    with torch.no_grad():
        a = model(x, mask)
        b = model(changed, mask)
        zero = model(x, torch.zeros_like(mask))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, zero)


def test_balanced_mask_loss_is_microbatch_invariant():
    targets = torch.tensor([[0.0, 1.0, 3.0], [0.0, 0.0, 4.0], [0.0, 2.0, 0.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    p = torch.randn_like(targets, requires_grad=True)
    full = models.reconstruction_loss(p, targets, mask)
    full.backward()
    gradient = p.grad.clone()
    p.grad = None
    counts = [int((mask & (targets > 0)).sum()), int((mask & (targets == 0)).sum())]
    split = sum(
        models.reconstruction_loss(p[i : i + 1], targets[i : i + 1], mask[i : i + 1], counts)
        for i in range(3)
    )
    split.backward()
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(p.grad, gradient)


def test_one_encoder_transfer_preserves_task_heads():
    stats = data.fit_statistics(events(8)["deposits"], events(8)["targets"])
    pretrained = models.LocalMaskedModel()
    state = models.encoder_state(pretrained)
    for task in ("energy", "position"):
        case = dict(phase="finetuning", architecture="transformer", task=task)
        torch.manual_seed(91)
        direct, _ = training.make_model({**case, "phase": "direct"}, stats, "cpu")
        torch.manual_seed(91)
        fine, record = training.make_model(case, stats, "cpu", state)
        for head in ("energy", "position"):
            torch.testing.assert_close(
                getattr(direct.network, head).weight,
                getattr(fine.network, head).weight,
                rtol=0,
                atol=0,
            )
        assert training.tensor_hash(models.encoder_state(fine)) == record["encoder_sha256"]
        assert all(p.requires_grad for p in fine.parameters())
        assert not any("decoder" in k for k in fine.state_dict())


@pytest.mark.parametrize("phase", ["direct", "pretraining", "finetuning"])
def test_interrupted_resume_matches_all_training_state(tmp_path, phase):
    config = training.effective(PROTOCOL, True)
    config.update(updates=4, pretraining_updates=4, evaluate_every=2, checkpoint_every=1)
    tr, va, stats, _ = data.prepare(events(16), events(8, 1), "noise_cut", config)
    case = next(
        c
        for c in training.cases(PROTOCOL)
        if c["phase"] == phase and c["architecture"] == "transformer"
    )
    parent = models.encoder_state(models.LocalMaskedModel()) if phase == "finetuning" else None
    full = training.train_phase(
        case, tr, va, stats, config, tmp_path / "full.zip", {}, parent=parent
    )
    training.train_phase(
        case, tr, va, stats, config, tmp_path / "resume.zip", {}, parent=parent, stop_after=2
    )
    actual = training.train_phase(
        case, tr, va, stats, config, tmp_path / "resume.zip", {}, parent=parent
    )
    assert training.tensor_hash(full["model"]) == training.tensor_hash(actual["model"])
    assert full["scheduler"] == actual["scheduler"]
    assert [h["train_loss"] for h in full["history"]] == [
        h["train_loss"] for h in actual["history"]
    ]
    assert torch.equal(full["mask_rng"], actual["mask_rng"])
    for k, v in full["optimizer"]["state"].items():
        for name, value in v.items():
            torch.testing.assert_close(value, actual["optimizer"]["state"][k][name], rtol=0, atol=0)
    if phase != "pretraining":
        assert full["best"]["metrics"] == actual["best"]["metrics"]
    changed = copy.deepcopy(config)
    changed["learning_rate"] *= 2
    # The caller includes effective configuration in identity, not only the state contents.
    with pytest.raises(ValueError, match="changed"):
        training.train_phase(
            case,
            tr,
            va,
            stats,
            changed,
            tmp_path / "resume.zip",
            {"config": changed},
            parent=parent,
        )


def test_archive_rejects_traversal_and_corruption(tmp_path):
    with pytest.raises(ValueError):
        transport.write_archive(tmp_path / "bad.zip", {"../escape": b"x"}, "test")
    path = tmp_path / "good.zip"
    transport.write_archive(path, {"x": b"abc"}, "test")
    import zipfile

    with zipfile.ZipFile(path, "a") as z:
        z.writestr("unexpected", b"x")
    with pytest.raises(ValueError):
        transport.read_archive(path, "test")


def synthetic_bundle(path):
    payload = {
        "configs/confirmation.json": transport.canonical(PROTOCOL),
        "ATTRIBUTION.txt": b"synthetic",
        "pyproject.toml": b"",
        "uv.lock": b"",
    }
    for split, raw in [("train", events(32)), ("validation", events(16, 1))]:
        b = io.BytesIO()
        np.savez_compressed(b, **raw)
        payload[f"data/{split}.npz"] = b.getvalue()
    transport.write_archive(path, payload, "confirmation_input")


def test_full_smoke_export_reimport_and_drive_style_rebuild(tmp_path):
    bundle = tmp_path / "input.zip"
    synthetic_bundle(bundle)
    workspace = transport.unpack(bundle, tmp_path / "workspace")
    result = training.run(
        workspace, tmp_path / "run", device="cpu", smoke=True, backup=tmp_path / "drive"
    )
    assert result["complete"] and len(result["cases"]) == 29
    export = tmp_path / "result.zip"
    transport.export_results(tmp_path / "run", export)
    verification = transport.verify_results(bundle, export)
    assert verification["complete"] and verification["predictions"] == 91
    before = {k: v["checkpoint_sha256"] for k, v in result["cases"].items()}
    # Drive contains authoritative checkpoints, but intentionally no predictions.
    recovered = training.run(workspace, tmp_path / "drive", device="cpu", smoke=True)
    assert before == {k: v["checkpoint_sha256"] for k, v in recovered["cases"].items()}
    transport.export_results(tmp_path / "drive", tmp_path / "recovered.zip")
    assert transport.verify_results(bundle, tmp_path / "recovered.zip")["complete"]
    assert len(reporting.timing(recovered)) == 3
    from calolab_reco.confirmation.evaluate import evaluate

    case_name = "noise_cut_20260925_finetuning_position"
    evaluated = evaluate(workspace, tmp_path / "drive", case_name, tmp_path / "native.json")
    assert evaluated["metrics"] == recovered["cases"][case_name]["selected"]["metrics"]
    manifest, files = transport.read_archive(export, "confirmation_results")
    broken = json.loads(files["summary.json"])
    removed = next(iter(broken["predictions"]))
    broken["predictions"].pop(removed)
    files.pop(removed)
    files["summary.json"] = transport.canonical(broken)
    transport.write_archive(
        tmp_path / "missing.zip", files, "confirmation_results", manifest["metadata"]
    )
    with pytest.raises(ValueError, match="prediction records"):
        transport.verify_results(bundle, tmp_path / "missing.zip")
