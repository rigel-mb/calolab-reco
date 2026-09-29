"""Fresh-process interruption/replay check on tiny train/validation subsets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from . import data, models, training, transport


def state_signature(value):
    if isinstance(value, torch.Tensor):
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=transport.digest(value.detach().cpu().contiguous().numpy().tobytes()),
        )
    if isinstance(value, dict):
        return {str(k): state_signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [state_signature(v) for v in value]
    return value


def worker(workspace, output, device, phase, mode):
    raw, manifest = transport.load_raw(workspace, True)
    protocol = json.loads((workspace / "configs/confirmation.json").read_text())
    config = training.effective(protocol, True)
    config.update(updates=4, pretraining_updates=4, evaluate_every=2, checkpoint_every=1)
    tr, va, stats, _ = data.prepare(raw["train"], raw["validation"], "noise_cut", config)
    case = next(
        c
        for c in training.cases(protocol)
        if c["phase"] == phase and c["architecture"] == "transformer"
    )
    torch.manual_seed(101)
    parent = models.encoder_state(models.LocalMaskedModel()) if phase == "finetuning" else None
    path = output / phase / ("full.zip" if mode == "full" else "resumed.zip")
    state = training.train_phase(
        case,
        tr,
        va,
        stats,
        config,
        path,
        dict(
            input_identity=transport.digest(transport.canonical(manifest)),
            protocol=config,
            verification_only=True,
        ),
        device,
        parent,
        stop_after=2 if mode == "prefix" else None,
    )
    fields = ["model", "optimizer", "scheduler", "sampler", "rng", "mask_rng", "update", "best"]
    signature = state_signature({k: state[k] for k in fields})
    transport.write_json(path.with_suffix(".json"), signature)


def verify(workspace, output, device):
    output = transport.external(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for phase in ("pretraining", "finetuning"):
        for mode in ("full", "prefix", "resume"):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "calolab_reco.confirmation.verification",
                    "--workspace",
                    str(workspace),
                    "--output",
                    str(output),
                    "--device",
                    device,
                    "--worker",
                    phase,
                    "--mode",
                    mode,
                ],
                check=True,
            )
        a = json.loads((output / phase / "full.json").read_text())
        b = json.loads((output / phase / "resumed.json").read_text())
        if a != b:
            raise RuntimeError(f"Fresh-process restart differs for {phase}")
        result[phase] = dict(
            exact=True,
            updates=4,
            interrupted_after=2,
            checked=["weights", "optimizer", "scheduler", "sampling", "RNG", "selected checkpoint"],
        )
    transport.write_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--worker", choices=["pretraining", "finetuning"])
    p.add_argument("--mode", choices=["full", "prefix", "resume"])
    args = p.parse_args()
    if args.worker:
        worker(args.workspace, args.output, args.device, args.worker, args.mode)
    else:
        print(json.dumps(verify(args.workspace, args.output, args.device), indent=2))
