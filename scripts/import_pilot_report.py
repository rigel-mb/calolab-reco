"""Validate a downloaded Colab report before adding lightweight public results."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from calolab_reco.pilot_reporting import render_report, validate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    validate_report(report)
    bundle_hash = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    if bundle_hash != report["bundle_sha256"]:
        raise ValueError("Report does not correspond to this pilot bundle.")
    with zipfile.ZipFile(args.bundle) as archive:
        bundle = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        metadata_bytes = archive.read("data/pilot_manifest.json")
        metadata = json.loads(metadata_bytes)
    provenance = report["benchmark"]["provenance"]
    expected = {
        "code_sha256": bundle["code_sha256"],
        "data_sha256": metadata["data_sha256"],
        "prepared_manifest_sha256": metadata["prepared_manifest_sha256"],
        "pilot_manifest_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }
    if any(provenance.get(k) != value for k, value in expected.items()):
        raise ValueError("Code or data provenance mismatch.")
    # Import only defined public fields; discard any unrelated notebook or account metadata.
    public = {
        k: report[k]
        for k in ["schema_version", "bundle_sha256", "benchmark", "resume", "storage_checks"]
    }
    content = json.dumps(public, indent=2, allow_nan=False) + "\n"
    if any(term in content for term in ["/Users/", "/content/drive/", "@gmail.com"]):
        raise ValueError("Report contains personal paths or account information.")
    outputs = [("colab_pilot.json", content), ("colab_pilot.md", render_report(public))]
    for name, value in outputs:
        path = args.output / name
        if path.exists() and path.read_text() != value:
            raise FileExistsError(f"Preserve the existing report before replacing {name}.")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in outputs:
        path = args.output / name
        path.write_text(value)
    print("Verified GPU report imported; review it before closing stage 1 or committing.")


if __name__ == "__main__":
    main()
