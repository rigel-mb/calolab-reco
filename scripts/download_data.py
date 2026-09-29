"""Download and verify the three one-photon archives, outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

FILES = ("1photon_en.npz", "1photon_yc.npz", "1photon_calo.npz")
API_URL = "https://zenodo.org/api/records/18929909"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(root: Path) -> None:
    root = root.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("The data directory must be outside the repository.")
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    metadata = json.loads(
        subprocess.check_output(
            ["curl", "--fail", "--location", "--silent", "--show-error", API_URL],
            text=True,
        )
    )
    license_id = metadata["metadata"]["license"]["id"]
    if license_id != "cc-by-4.0":
        raise ValueError(f"Unexpected license: {license_id}")
    entries = {entry["key"]: entry for entry in metadata["files"]}
    expected_total = sum(entries[name]["size"] for name in FILES)
    if expected_total != 1_690_168_100:
        raise ValueError("File sizes changed; review the source record before downloading.")
    if shutil.disk_usage(raw).free < expected_total + 3 * 1024**3:
        raise RuntimeError("Insufficient free space for archives, derived data and safety margin.")
    (raw / "zenodo_record.json").write_text(json.dumps(metadata, indent=2) + "\n")
    verified = []
    for name in FILES:
        entry = entries[name]
        target = raw / name
        algorithm, expected_hash = entry["checksum"].split(":", 1)
        if target.exists():
            if target.stat().st_size != entry["size"] or digest(target, algorithm) != expected_hash:
                raise RuntimeError(f"Existing {name} failed verification; preserve it for review.")
            print(f"Already verified: {name}", flush=True)
        else:
            partial = raw / f"{name}.part"
            print(f"Downloading {name}: {entry['size']:,} bytes", flush=True)
            url = f"https://zenodo.org/records/18929909/files/{name}?download=1"
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "3",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    url,
                ],
                check=True,
            )
            if (
                partial.stat().st_size != entry["size"]
                or digest(partial, algorithm) != expected_hash
            ):
                raise RuntimeError(f"Downloaded {name} failed verification; kept .part file.")
            partial.rename(target)
        verified.append(
            {
                "name": name,
                "bytes": entry["size"],
                "checksum": entry["checksum"],
                "sha256": digest(target, "sha256"),
            }
        )
        print(f"Verified {name}", flush=True)
    manifest = {
        "record_id": "18929909",
        "source": API_URL,
        "license": license_id,
        "files": verified,
    }
    (raw / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All three one-photon archives verified. No other dataset file requested.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CALOLAB_DATA_ROOT", Path.home() / "Data/public/calolab-reco")),
    )
    download(parser.parse_args().data_root)
