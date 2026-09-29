"""CPU evaluation of a selected validation checkpoint, for native/Docker comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calolab_reco.pilot import configure

from . import data, training, transport


def evaluate(workspace, run, case_name, output, limit=None):
    summary = json.loads((run / "summary.json").read_text())
    if (
        case_name not in summary["cases"]
        or summary["cases"][case_name]["case"]["phase"] == "pretraining"
    ):
        raise ValueError("Select a supervised or fine-tuned case")
    raw, manifest = transport.load_raw(workspace, summary["smoke"])
    state = training.checkpoint_load(run / "cases" / case_name / "checkpoint.zip")
    if state["identity"]["input_identity"] != transport.digest(transport.canonical(manifest)):
        raise ValueError("Checkpoint belongs to another input bundle")
    case = state["identity"]["case"]
    stats = state["identity"]["statistics"]
    config = summary["protocol"]
    configure("cpu", case["seed"])
    split, _ = data.evaluate_inputs(
        raw["validation"], case["regime"], config["validation_noise_seed"], config, stats
    )
    if limit is not None:
        if not 1 <= limit <= len(split["targets"]):
            raise ValueError("Invalid validation sample limit")
        split = {k: v[:limit] for k, v in split.items()}
    model, _ = training.make_model(case, stats, "cpu")
    model.load_state_dict(state["best"]["model"])
    prediction = training.predict(model, split, "cpu")
    result = dict(
        case=case,
        metrics=data.aggregate_metrics(split["raw_targets"], prediction, case["task"]),
        checkpoint_sha256=transport.sha(run / "cases" / case_name / "checkpoint.zip"),
        input_identity=summary["input_identity"],
        test_used=False,
        smoke=summary["smoke"],
    )
    transport.write_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    print(json.dumps(evaluate(a.workspace, a.run, a.case, a.output, a.limit), indent=2))
