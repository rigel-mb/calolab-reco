"""The portability check must reject changed events and numerical disagreement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from calolab_reco.confirmation import transport

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_confirmation_docker_check.py"
SPEC = importlib.util.spec_from_file_location("confirmation_docker_check", SCRIPT)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def pair(tmp_path, change=None):
    for label in ("native", "docker"):
        folder = tmp_path / label
        folder.mkdir()
        predictions = np.array([[10.0, 2.0, 5.0], [20.0, 3.0, 6.0]])
        targets = predictions.copy()
        ids = np.array([[0, 1], [0, 2]], dtype=np.int64)
        report = {
            "split": "validation",
            "test_used": False,
            "limit": 2,
            "input_identity": "input",
            "sources": {"source.py": "source"},
            "environment": {"device": "cpu"},
            "cases": {
                "model": {
                    "case": {"task": "energy"},
                    "checkpoint_sha256": "checkpoint",
                    "metrics": {"count": 2, "energy_mare": 0.1},
                }
            },
        }
        if label == "docker":
            if change == "predictions":
                predictions[0, 0] += 0.01
            elif change == "targets":
                targets[0, 0] += 0.01
            elif change == "source_ids":
                ids[0, 1] += 100
            elif change == "metrics":
                report["cases"]["model"]["metrics"]["energy_mare"] += 0.01
            elif change == "checkpoint":
                report["cases"]["model"]["checkpoint_sha256"] = "other"
            elif change == "sources":
                report["sources"]["source.py"] = "other"
        np.savez(folder / "model.npz", predictions=predictions, targets=targets, source_ids=ids)
        transport.write_json(folder / "evaluation.json", report)
    return tmp_path / "native", tmp_path / "docker"


def test_identical_cpu_predictions_pass(tmp_path):
    result = check.compare(*pair(tmp_path))
    assert result["comparison_passed"]
    assert result["cases"]["model"]["max_absolute_difference_by_column"] == [0.0] * 3


@pytest.mark.parametrize("change", ["predictions", "targets", "source_ids", "metrics"])
def test_changed_evaluation_fails(tmp_path, change):
    with pytest.raises(AssertionError):
        check.compare(*pair(tmp_path, change))


@pytest.mark.parametrize("change", ["checkpoint", "sources"])
def test_changed_provenance_fails(tmp_path, change):
    with pytest.raises(ValueError):
        check.compare(*pair(tmp_path, change))
