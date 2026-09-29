"""Real CNN updates on synthetic events, with exact fresh-process continuation."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from calolab_reco import training
from calolab_reco.metrics import regression_metrics
from calolab_reco.pilot import compare_state, configure, sha256
from calolab_reco.pilot_models import CNN, supervised_loss


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("cnn_training")
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(78)
    arrays = {}
    for split, count, group in [("train", 9, 0), ("validation", 5, 1)]:
        deposits = rng.exponential(5, size=(count, 30, 85)).astype(np.float32)
        deposits[rng.random(deposits.shape) < 0.85] = 0
        targets = np.column_stack(
            (rng.uniform(1, 10, count), rng.uniform(5, 20, count), rng.uniform(15, 70, count))
        ).astype(np.float32)
        source_ids = np.column_stack((np.full(count, group), np.arange(count))).astype(np.int64)
        arrays[split] = {"deposits": deposits, "targets": targets, "source_ids": source_ids}
        np.savez(data / f"{split}.npz", **arrays[split])
    manifest = {
        "schema_version": 1,
        "input_selection": "train_validation_only",
        "train_count": 9,
        "val_count": 5,
        "input_scale": float(
            np.median(arrays["train"]["deposits"][arrays["train"]["deposits"] > 0])
        ),
        "target_mean": arrays["train"]["targets"].mean(axis=0, dtype=np.float64).tolist(),
        "target_std": arrays["train"]["targets"].std(axis=0, dtype=np.float64).tolist(),
        "prepared_manifest_sha256": "1" * 64,
        "prepared_output_sha256": {
            name: "2" * 64
            for name in ["deposits.npy", "targets.npy", "source_ids.npy", "split.npy"]
        },
        "data_sha256": {
            f"{split}.npz": sha256(data / f"{split}.npz") for split in ["train", "validation"]
        },
    }
    (data / "stage2_manifest.json").write_text(json.dumps(manifest))
    config = {
        "seed": 32,
        "epochs": 2,
        "effective_batch_size": 4,
        "microbatch_size": 2,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "eval_batch_size": 3,
        "max_train_seconds": 240,
    }
    config_path = root / "cnn.toml"
    config_path.write_text("".join(f"{key} = {value}\n" for key, value in config.items()))
    return {
        "root": root,
        "data": data,
        "arrays": arrays,
        "config": config,
        "config_path": config_path,
    }


def run_cli(arguments):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "calolab_reco.training", *map(str, arguments)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def trained(synthetic):
    full, first, continued, persistent = [
        synthetic["root"] / name for name in ["full", "first", "continued", "persistent"]
    ]
    common = [
        "train",
        "--data-dir",
        synthetic["data"],
        "--config",
        synthetic["config_path"],
        "--device",
        "cpu",
    ]
    run_cli([*common, "--output", full])
    run_cli([*common, "--output", first, "--stop-after-epochs", "1", "--persist-dir", persistent])
    partial = json.loads((first / "history.json").read_text())
    assert partial["status"] == "stopped_after_epochs" and partial["completed"] is False
    assert partial["epochs_completed"] == 1
    for name in ["best.pt", "best.pt.sha256", "last.pt", "last.pt.sha256", "history.json"]:
        assert sha256(first / name) == sha256(persistent / name)
    run_cli([*common, "--output", continued, "--resume", persistent / "last.pt"])
    return {"full": full, "continued": continued, "persistent": persistent}


def without_times(history):
    return [
        {key: value for key, value in row.items() if not key.endswith("seconds")} for row in history
    ]


def test_full_epochs_and_fresh_process_resume_are_identical(synthetic, trained):
    full = training.load_checkpoint(trained["full"] / "last.pt")
    resumed = training.load_checkpoint(trained["continued"] / "last.pt")
    for key in [
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "epoch",
        "global_step",
        "best_epoch",
        "best_validation_loss",
        "config",
        "transformations",
        "provenance",
    ]:
        assert compare_state(full[key], resumed[key]), key
    assert without_times(full["history"]) == without_times(resumed["history"])
    assert full["global_step"] == 6 and full["epoch"] == 2
    assert [row["train_count"] for row in full["history"]] == [9, 9]
    assert full["history"][-1]["learning_rate"] == pytest.approx(0)
    assert full["active_training_seconds"] > 0 and full["validation_seconds"] > 0
    configure("cpu", synthetic["config"]["seed"])
    initial = CNN().state_dict()
    assert any(not torch.equal(initial[key], full["model"][key]) for key in initial)
    for output in trained.values():
        assert training.load_checkpoint(output / "best.pt")["epoch"] >= 1
    complete = json.loads((trained["continued"] / "history.json").read_text())
    assert complete["completed"] is True and complete["test_used"] is False


def test_evaluation_reload_retains_ids_and_metrics(synthetic, trained):
    output = synthetic["root"] / "evaluation"
    run_cli(
        [
            "evaluate",
            "--data-dir",
            synthetic["data"],
            "--checkpoint",
            trained["continued"] / "best.pt",
            "--device",
            "cpu",
            "--output",
            output,
        ]
    )
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        targets, predictions, source_ids = [
            archive[name] for name in ["targets", "predictions", "source_ids"]
        ]
    np.testing.assert_array_equal(targets, synthetic["arrays"]["validation"]["targets"])
    np.testing.assert_array_equal(source_ids, synthetic["arrays"]["validation"]["source_ids"])
    report = json.loads((output / "metrics.json").read_text())
    assert report["metrics"] == regression_metrics(targets, predictions)
    assert report["count"] == 5 and report["limited"] is False and report["test_used"] is False
    assert report["checkpoint_sha256"] == sha256(trained["continued"] / "best.pt")
    state = training.load_checkpoint(trained["continued"] / "best.pt")
    assert report["validation_loss"] == state["best_validation_loss"]
    limited = training.evaluate(
        synthetic["data"],
        trained["continued"] / "best.pt",
        synthetic["root"] / "limited",
        "cpu",
        limit=2,
    )
    assert limited["count"] == 2 and limited["limited"] is True


@pytest.mark.parametrize("kind", ["checkpoint_bytes", "missing_hash", "changed_config"])
def test_resume_rejects_corruption_and_configuration_drift(synthetic, trained, tmp_path, kind):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes((trained["persistent"] / "last.pt").read_bytes())
    fingerprint = checkpoint.with_name("last.pt.sha256")
    fingerprint.write_text(sha256(checkpoint))
    config = copy.deepcopy(synthetic["config"])
    if kind == "checkpoint_bytes":
        with checkpoint.open("ab") as stream:
            stream.write(b"corrupt")
    elif kind == "missing_hash":
        fingerprint.unlink()
    else:
        config["learning_rate"] *= 2
    with pytest.raises(ValueError):
        training.train(synthetic["data"], config, tmp_path / "output", "cpu", resume=checkpoint)


def test_changed_data_is_rejected_before_training(synthetic, tmp_path):
    for path in synthetic["data"].iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    with (tmp_path / "validation.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Data fingerprint"):
        training.load_data(tmp_path)


def test_time_budget_stop_is_incomplete_and_preserves_partial_epoch(synthetic, tmp_path):
    config = {**synthetic["config"], "max_train_seconds": 1e-9}
    result = training.train(synthetic["data"], config, tmp_path / "output", "cpu")
    assert result["status"] == "time_budget_exhausted" and result["completed"] is False
    state = training.load_checkpoint(tmp_path / "output/last.pt")
    assert state["global_step"] == 1 and state["epoch"] == 0
    assert state["in_progress"]["seen"] == 4
    assert len(torch.unique(state["in_progress"]["order"])) == 9
    resumed = training.train(
        synthetic["data"], config, tmp_path / "resumed", "cpu", resume=tmp_path / "output/last.pt"
    )
    assert resumed["global_step"] == 1 and resumed["status"] == "time_budget_exhausted"


def test_source_fingerprint_requires_the_executing_code(synthetic, tmp_path):
    files = training.source_fingerprints()["source_files"]
    for name in files:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((training.PROJECT / name).read_bytes())
    manifest = {"files": files, "code_sha256": training.canonical_hash(files)}
    manifest_path = tmp_path / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    assert training.source_fingerprints(manifest_path) == training.source_fingerprints()
    module = tmp_path / "src/calolab_reco/training.py"
    module.write_text(module.read_text() + "\n# Different source revision.\n")
    manifest["files"]["src/calolab_reco/training.py"] = sha256(module)
    manifest["code_sha256"] = training.canonical_hash(manifest["files"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="executing code"):
        training.source_fingerprints(manifest_path)


@pytest.mark.parametrize(
    "changes", [{"epochs": 21}, {"microbatch_size": 3}, {"max_train_seconds": 2701}]
)
def test_training_bounds_are_explicit(synthetic, changes):
    with pytest.raises(ValueError):
        training.validate_config({**synthetic["config"], **changes})


def test_final_single_event_has_undiluted_gradient(synthetic, tmp_path, monkeypatch):
    observed = {}
    original_step = torch.optim.AdamW.step
    original_model = training.CNN

    def create_model():
        observed["model"] = original_model()
        return observed["model"]

    def capture_step(optimizer, *args, **kwargs):
        observed["steps"] = observed.get("steps", 0) + 1
        if observed["steps"] == 3:
            observed["weights"] = copy.deepcopy(observed["model"].state_dict())
            observed["gradients"] = [p.grad.clone() for p in observed["model"].parameters()]
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(training, "CNN", create_model)
    monkeypatch.setattr(torch.optim.AdamW, "step", capture_step)
    config = {**synthetic["config"], "epochs": 1}
    training.train(synthetic["data"], config, tmp_path, "cpu")
    reference = original_model()
    reference.load_state_dict(observed["weights"])
    data = training.load_data(synthetic["data"])["train"]
    order = torch.randperm(9, generator=torch.Generator().manual_seed(config["seed"]))
    index = order[-1:]
    supervised_loss(reference(data["inputs"][index]), data["targets"][index]).backward()
    for parameter, actual in zip(reference.parameters(), observed["gradients"], strict=True):
        torch.testing.assert_close(actual, parameter.grad, rtol=1e-6, atol=1e-7)


def test_storage_boundary_prevents_repository_outputs():
    with pytest.raises(ValueError, match="outside"):
        training.external_directory(training.PROJECT / "artifacts/not_allowed")
