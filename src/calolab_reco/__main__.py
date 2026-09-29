"""Audit, prepare, train and evaluate the bounded reconstruction study."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

from calolab_reco.data import data_root, inventory_archives, prepare_sample


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"train", "evaluate"}:
        from calolab_reco.training import main as training_main

        training_main(sys.argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "prepare", "train", "evaluate"])
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    settings = {
        "sample_size": 40000,
        "sample_seed": 20260920,
        "split_seed": 20260921,
        "max_member_bytes": 536870912,
    }
    if args.config:
        config = tomllib.loads(args.config.read_text(encoding="utf-8"))
        if config.keys() - settings.keys():
            parser.error("Unsupported configuration keys")
        if any(type(value) is not int for value in config.values()):
            parser.error("Audit configuration values must be integers")
        settings.update(config)
    for key in ("sample_size", "sample_seed", "split_seed"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.command == "audit":
        report = inventory_archives(
            args.data_root.expanduser() / "raw", settings["max_member_bytes"]
        )
        report = {key: value for key, value in report.items() if key != "groups"} | {
            "group_count": len(report["groups"]),
            "first_group": report["groups"][0],
        }
    else:
        report = prepare_sample(args.data_root, **settings)
        report = {key: value for key, value in report.items() if key != "inventory"}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
