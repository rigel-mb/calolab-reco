# Reproduce the study

This is the optional technical path. Start with the [main reading guide](../README.md) for the science, the [methodology](methodology.md) for the fixed protocol and the [final report](../reports/final/README.md) for measured results.

## Install and read

Clone the repository, then use Python 3.12 through uv:

```bash
git clone https://github.com/rigel-mb/calolab-reco.git
cd calolab-reco
uv sync --locked --extra cpu --group notebook
```

The reading notebooks are `01_data_audit.ipynb`, `02_methods_and_decisions.ipynb` and `04_final_results.ipynb`. Their saved outputs can be read without training. Methods and final results read the lightweight reports included here. Executing the audit additionally requires the prepared data described below. Notebook 03 and the Kaggle runner are optional training interfaces, not extra reading steps.

Large data, model weights and per-event predictions are external. Set `CALOLAB_DATA_ROOT` to the data directory, normally `~/Data/public/calolab-reco`. Its prepared sample is `derived/audit_v1/`. Keep checkpoints and run exports under `~/Data/models/calolab-reco`, and exchange bundles under `~/Data/cache/calolab-reco` or the remote runtime's storage.

## Required external artifacts

| Operation | Additional inputs |
| --- | --- |
| Read results or run synthetic CPU tests | None |
| Execute the data-audit notebook | Prepared sample and manifest under the data root |
| Reproduce the frozen GPU training | Original `calolab-reco-confirmation.zip` input bundle, containing train and validation only |
| Docker prediction comparison | Data and six checkpoints from the release archive below |
| Reproduce held-out inference | Original input/result archives, prepared data including test, and the existing evaluation freeze |

Data and weights stay outside Git. The downloadable Docker archive supplies the original train/validation input bundle and six selected checkpoints. The complete training-result export and reserved test are not included; reproducing the full held-out evaluation still requires those separate artifacts. Data provenance is documented in the [data audit](data_audit.md).

For the commands below, `CONFIRMATION_BUNDLE` is the original input ZIP, `RETURNED_ZIP` is the completed training export, `RUN` is its extracted run directory, and `OUTPUT` is a new external output directory. Paths are shell variables to set to actual files, not files shipped in Git. The notebooks require input SHA256 `0e758bf20eda5cde456ff77ca218a396705437bff24cff63c718484e24509fe9`.

## Prepare the data audit from the public source

This optional path downloads about 1.69 GB and produces about 0.409 GB of derived arrays outside Git. Existing prepared data can be transferred instead. These commands do not train a network or evaluate test performance:

```bash
export CALOLAB_DATA_ROOT="$HOME/Data/public/calolab-reco"
uv run --locked --extra cpu python scripts/download_data.py --data-root "$CALOLAB_DATA_ROOT"
uv run --locked --extra cpu python -m calolab_reco audit --config configs/audit.toml
uv run --locked --extra cpu python -m calolab_reco prepare --config configs/audit.toml
```

The raw source checksums and preparation manifest record provenance. Recreating a sample does not by itself guarantee byte-identical frozen bundles: use the recorded artifacts for exact checkpoint/evaluation reproduction. Do not overwrite an existing preparation to bypass a fingerprint mismatch.

## Software checks

```bash
uv lock --check
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu pytest
```

Tests use small synthetic examples to cover data boundaries, metrics, gradients, checkpoint recovery and evaluation consistency. They do not establish model superiority or run the full GPU experiments. The workflow in `.github/workflows/cpu.yml` runs CPU checks on pushes and pull requests, and validates notebook structure and code syntax without executing scientific notebooks.

## Optional GPU training

The same frozen package and configuration are used by both runners. The completed scientific run used one Tesla T4 and recorded 110.53 minutes of active training and checkpoint selection; installation, file transfers, checkpoint writes and other overhead are additional. Free GPU availability is not guaranteed.

### Colab

1. Open [03_colab_confirmation.ipynb](../notebooks/03_colab_confirmation.ipynb) and select a GPU runtime.
2. Run cells in order, authorize Drive persistence and upload the original input ZIP when prompted.
3. Read the runtime and recovery checks, then run training. Progress is printed during each phase.
4. Download the exported results ZIP. A new session with the same notebook and bundle restores committed checkpoints from Drive.

### Kaggle alternative

1. Import [kaggle_confirmation.ipynb](../notebooks/runners/kaggle_confirmation.ipynb). Enable GPU and Internet; account verification may be required.
2. Upload the original input ZIP as a private dataset, renaming its extension to `.bundle` to prevent automatic extraction. Do not change its bytes. Attach it to the notebook.
3. The verified run used a T4 x2 allocation but only one GPU. Run the notebook interactively or as a saved background version; do not launch both at once.
4. Download the exported `calolab-reco-confirmation-results-*.zip` from saved outputs or `/kaggle/working`.

Kaggle working files are not an independent backup. Save outputs and download the archive. To resume from saved outputs, attach them as input and set `PREVIOUS_RUN` in the first cell to the run directory containing `protocol.json` and `cases/`. Runtime environment and inputs must match the checkpoint. The runner sets a noninteractive plotting backend for its isolated Python process.

Both runners exclude reserved-test data. They verify a short interruption/replay before full training. Completed phases and partial runs are exported explicitly; changing the compute ceiling or protocol is a new experiment, not a presentation change.

## Verify returned results and rebuild reports

```bash
uv run --locked --extra cpu python -m calolab_reco.confirmation.transport verify \
  --bundle "$CONFIRMATION_BUNDLE" --result "$RETURNED_ZIP"

uv run --locked --extra cpu python scripts/review_confirmation.py \
  --bundle "$CONFIRMATION_BUNDLE" --result "$RETURNED_ZIP" --output "$OUTPUT"
```

Use an external output directory for regenerated reports and review differences before replacing published aggregates. Returned archives supply data and weights, not Python to execute. The verification checks archive integrity, event alignment and recomputed metrics using the project's metric functions; this is not an independent implementation of every formula.

## Docker CPU evaluation

Start an available Docker engine after the installation above. Download [calolab-reco-docker-demo.zip](https://github.com/rigel-mb/calolab-reco/releases/download/v0.1.0/calolab-reco-docker-demo.zip) (36.2 MB) and keep its contents outside the repository. The archive supplies the unchanged original train/validation bundle, six primary-seed checkpoints and their provenance. It includes more data than the check uses so that the original input fingerprints remain valid; reserved-test data are excluded.

From the repository root, download and extract it, then run the existing comparison command:

```bash
DEMO_DIR="$HOME/Data/cache/calolab-reco/docker-demo"
mkdir -p "$DEMO_DIR"
curl --fail --location \
  https://github.com/rigel-mb/calolab-reco/releases/download/v0.1.0/calolab-reco-docker-demo.zip \
  --output "$DEMO_DIR/calolab-reco-docker-demo.zip"
unzip -n "$DEMO_DIR/calolab-reco-docker-demo.zip" -d "$DEMO_DIR"

CONFIRMATION_BUNDLE="$DEMO_DIR/calolab-reco-docker-demo/calolab-reco-confirmation.zip"
RUN="$DEMO_DIR/calolab-reco-docker-demo/run"
OUTPUT="$DEMO_DIR/comparison-$(date +%Y%m%d-%H%M%S)"
uv run --locked --extra cpu python scripts/run_confirmation_docker_check.py \
  --bundle "$CONFIRMATION_BUNDLE" --run "$RUN" --output "$OUTPUT" \
  --limit 256 --build --report "$OUTPUT/docker_check.json"
```

The expected result is `comparison_passed: true`, `cases: 6`, `count: 256`; details are saved in `$OUTPUT/docker_check.json`. The script verifies input/checkpoint fingerprints, builds the existing Dockerfile and compares native and container predictions on the same 256 validation events. No training occurs. Inputs and checkpoints are read-only; the container uses two CPUs, 2 GiB memory and no network during evaluation. The [recorded comparison](../reports/final/docker_check.json) passed on native macOS ARM64 and Linux ARM64, with exact IDs/targets and prediction/metric tolerances `rtol=1e-5`, `atol=1e-6`. Windows/WSL and Linux AMD64 have not yet been verified. This checks bounded numerical portability, not the full held-out results.

## Reproduce the fixed final evaluation

Keep the existing `configs/final_evaluation.json`; do not create a new freeze or select models after inspecting the test. `FINAL_RUN` must be a new external output directory. The explicit test flag authorizes only inference under that existing protocol:

```bash
uv run --locked --extra cpu python -m calolab_reco.confirmation.final_evaluation evaluate \
  --bundle "$CONFIRMATION_BUNDLE" --result "$RETURNED_ZIP" \
  --data-root "$CALOLAB_DATA_ROOT" --freeze configs/final_evaluation.json \
  --output "$FINAL_RUN" --allow-test

uv run --locked --extra cpu python scripts/review_final.py \
  --run "$FINAL_RUN" --freeze configs/final_evaluation.json \
  --validation reports/confirmation/review.json --output "$OUTPUT"
```

The final inference covers 5,935 test events. Saved aggregate evidence is in `reports/final/results.json`. The separate adaptation-cost calculation uses validation learning histories only:

```bash
uv run --locked --extra cpu python scripts/review_amortization.py \
  --review reports/confirmation/review.json --output "$OUTPUT"
```

That utility also generates a standalone cost note for an external run; the public interpretation is consolidated in the final report.

## Repository map and historical scope

| Location | Responsibility |
| --- | --- |
| `notebooks/` | Three reading notebooks and two optional GPU runners |
| `docs/` | Data conventions, fixed method, decisions and this reproduction guide |
| `src/calolab_reco/confirmation/` | Final local models, training and evaluation |
| `configs/` | Exact settings and the final evaluation freeze |
| `reports/` | Final/validation results and compact exploratory evidence |
| `scripts/` | Data preparation support, review and reproduction commands |
| `tests/` | Checks for the retained code and final protocol |

Frozen source files keep their original paths and bytes because the original bundles and checkpoints record their fingerprints. Some earlier modules remain required for imports, compatibility or source verification. The small `experiments/` subset supports the continuity comparison and its export dependencies; it is not an additional visitor route. The [decision record](decisions.md) explains earlier experiments. Their intermediate notebooks and dedicated utilities are not all distributed in this compact repository; the current public code is the reproduction path for the selected method, not a claim that every exploratory run can be regenerated from this checkout.
