"""Compare all six primary-seed local specialists on bounded validation data."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from calolab_reco.confirmation import data, training, transport
from calolab_reco.pilot import configure

PROJECT = transport.ROOT
RTOL, ATOL = 1e-5, 1e-6


def evaluate(workspace, run, output, limit):
    """Use frozen preprocessing and selected weights, without any fitting."""
    if not 1 <= limit <= 1024:
        raise ValueError("The validation limit must be between 1 and 1024")
    summary = json.loads((run / "summary.json").read_text())
    if not summary["complete"] or summary["smoke"] or summary["test_used"]:
        raise ValueError("Expected a complete scientific train/validation return")
    raw, manifest = transport.load_raw(workspace)
    source_hashes = {}
    for name, expected in manifest["files"].items():
        if name.startswith("src/"):
            actual = transport.sha(transport.ROOT / name)
            if actual != expected:
                raise ValueError(f"Evaluation source differs from the frozen input: {name}")
            source_hashes[name] = actual
    if (workspace / "data/test.npz").exists():
        raise ValueError("Use the train/validation bundle only")
    config = summary["protocol"]
    seed = config["training_seeds"][0]
    prefix = f"{config['primary_regime']}_{seed}_"
    names = [
        prefix + family + "_" + task
        for family in ("cnn", "transformer", "finetuning")
        for task in ("energy", "position")
    ]
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in names:
        checkpoint = run / "cases" / name / "checkpoint.zip"
        state = training.checkpoint_load(checkpoint)
        if state["identity"]["input_identity"] != transport.digest(transport.canonical(manifest)):
            raise ValueError("Checkpoint belongs to a different input bundle")
        case = state["identity"]["case"]
        if case != summary["cases"][name]["case"]:
            raise ValueError("Checkpoint and summary case differ")
        stats = state["identity"]["statistics"]
        configure("cpu", seed)
        split, _ = data.evaluate_inputs(
            raw["validation"], case["regime"], config["validation_noise_seed"], config, stats
        )
        split = {key: value[:limit] for key, value in split.items()}
        model, _ = training.make_model(case, stats, "cpu")
        model.load_state_dict(state["best"]["model"])
        prediction = training.predict(model, split, "cpu")
        prediction_file = output / f"{name}.npz"
        np.savez_compressed(
            prediction_file,
            predictions=prediction,
            targets=split["raw_targets"],
            source_ids=split["source_ids"],
        )
        results[name] = dict(
            case=case,
            checkpoint_sha256=transport.sha(checkpoint),
            prediction_sha256=transport.sha(prediction_file),
            metrics=data.aggregate_metrics(split["raw_targets"], prediction, case["task"]),
        )
        print(f"Evaluated {name}: {len(prediction)} validation events", flush=True)
    report = dict(
        split="validation",
        test_used=False,
        limit=limit,
        input_identity=summary["input_identity"],
        sources=source_hashes,
        environment=dict(
            system=platform.system(),
            machine=platform.machine(),
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device="cpu",
            threads=torch.get_num_threads(),
        ),
        cases=results,
    )
    transport.write_json(output / "evaluation.json", report)
    return report


def compare(native, container):
    reports = [json.loads((path / "evaluation.json").read_text()) for path in (native, container)]
    for key in ("split", "test_used", "limit", "input_identity", "sources"):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"Evaluation provenance differs: {key}")
    if set(reports[0]["cases"]) != set(reports[1]["cases"]):
        raise ValueError("Evaluation case sets differ")
    comparisons = {}
    for name, first in reports[0]["cases"].items():
        second = reports[1]["cases"][name]
        for key in ("case", "checkpoint_sha256"):
            if first[key] != second[key]:
                raise ValueError(f"Case provenance differs: {name}, {key}")
        if first["metrics"].keys() != second["metrics"].keys():
            raise ValueError("Metric names differ")
        for key, value in first["metrics"].items():
            np.testing.assert_allclose(value, second["metrics"][key], rtol=RTOL, atol=ATOL)
        with (
            np.load(native / f"{name}.npz", allow_pickle=False) as a,
            np.load(container / f"{name}.npz", allow_pickle=False) as b,
        ):
            for key in ("source_ids", "targets"):
                np.testing.assert_array_equal(a[key], b[key])
            if not np.isfinite(a["predictions"]).all() or not np.isfinite(b["predictions"]).all():
                raise ValueError("Nonfinite predictions")
            np.testing.assert_allclose(a["predictions"], b["predictions"], rtol=RTOL, atol=ATOL)
            comparisons[name] = dict(
                checkpoint_sha256=first["checkpoint_sha256"],
                source_ids_sha256=transport.digest(a["source_ids"].astype("<i8").tobytes()),
                max_absolute_difference_by_column=np.max(
                    np.abs(a["predictions"] - b["predictions"]), axis=0
                ).tolist(),
                native_metrics=first["metrics"],
                docker_metrics=second["metrics"],
                native_predictions_sha256=transport.sha(native / f"{name}.npz"),
                docker_predictions_sha256=transport.sha(container / f"{name}.npz"),
            )
    return dict(
        comparison_passed=True,
        cases=comparisons,
        environments={
            key: report["environment"]
            for key, report in zip(("native", "docker"), reports, strict=True)
        },
        frozen_sources=reports[0]["sources"],
        input_identity=reports[0]["input_identity"],
    )


def run_check(args):
    from run_docker_check import docker_cli, read_command_json

    if not 1 <= args.limit <= 1024:
        raise ValueError("The validation limit must be between 1 and 1024")
    bundle = args.bundle.expanduser().resolve(strict=True)
    run = args.run.expanduser().resolve(strict=True)
    output = transport.external(args.output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; previous attempts are preserved")
    if any("," in str(path) for path in (run, output, Path(__file__).resolve())):
        raise ValueError("Docker mount paths must not contain commas")
    cli = docker_cli(args.docker_cli)
    version = read_command_json([cli, "version", "--format", "{{json .}}"])
    if not version.get("Server", {}).get("Version"):
        raise RuntimeError("Docker engine unavailable")
    if args.build:
        subprocess.run(
            [cli, "build", "--platform", args.platform, "-t", args.image, str(PROJECT)],
            check=True,
        )
    image = read_command_json([cli, "image", "inspect", args.image])[0]
    if f"{image['Os']}/{image['Architecture']}" != args.platform:
        raise ValueError("Image platform differs from requested platform")
    output.mkdir(parents=True)
    workspace = transport.unpack(bundle, output / "workspace")
    runner = Path(__file__).resolve()
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--evaluate",
            "--workspace",
            str(workspace),
            "--run",
            str(run),
            "--output",
            str(output / "native"),
            "--limit",
            str(args.limit),
        ],
        check=True,
    )
    docker_command = [
        cli,
        "run",
        "--rm",
        "--platform",
        args.platform,
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=128m",
        "--entrypoint",
        "/opt/venv/bin/python",
        "--mount",
        f"type=bind,src={workspace},dst=/inputs,readonly",
        "--mount",
        f"type=bind,src={run},dst=/run,readonly",
        "--mount",
        f"type=bind,src={runner},dst=/runner.py,readonly",
        "--mount",
        f"type=bind,src={output},dst=/output",
        image["Id"],
        "/runner.py",
        "--evaluate",
        "--workspace",
        "/inputs",
        "--run",
        "/run",
        "--output",
        "/output/docker",
        "--limit",
        str(args.limit),
    ]
    subprocess.run(docker_command, check=True)
    report = compare(output / "native", output / "docker")
    report.update(
        schema_version=1,
        created_utc=datetime.now(UTC).isoformat(),
        split="validation",
        test_used=False,
        count=args.limit,
        training_seed=20260925,
        input_bundle_sha256=transport.sha(bundle),
        runner_sha256=transport.sha(runner),
        tolerances=dict(relative=RTOL, absolute=ATOL),
        image=dict(id=image["Id"], platform=args.platform),
        docker_versions={key.lower(): version[key]["Version"] for key in ("Client", "Server")},
        limits=dict(cpus=2, memory_gib=2, network="none", root="read-only", tmpfs_mib=128),
        inputs_and_checkpoints="read-only mounts",
        build_requested=args.build,
    )
    transport.write_json(output / "docker_check.json", report)
    if args.report is not None:
        transport.write_json(args.report, report)
    print(
        json.dumps(
            {"comparison_passed": True, "cases": len(report["cases"]), "count": args.limit},
            indent=2,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--image", default="calolab-reco:confirmation")
    parser.add_argument("--docker-cli")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--platform",
        default="linux/arm64" if platform.machine() == "arm64" else "linux/amd64",
        choices=("linux/arm64", "linux/amd64"),
    )
    args = parser.parse_args()
    if args.evaluate:
        if args.workspace is None:
            parser.error("--evaluate needs --workspace")
        evaluate(args.workspace, args.run, args.output, args.limit)
    else:
        if args.bundle is None:
            parser.error("--bundle is required")
        run_check(args)


if __name__ == "__main__":
    main()
