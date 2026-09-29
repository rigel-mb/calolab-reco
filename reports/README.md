# Recorded evidence

The [final notebook](../notebooks/04_final_results.ipynb) is the reading entry. This directory keeps lightweight evidence, not raw data or checkpoints.

| Stage | Read | Numerical record |
| --- | --- | --- |
| Final held-out results and reuse cost | [Final report](final/README.md) | [Results](final/results.json), [timings](final/amortization.json) |
| Three-seed validation | [Validation report](confirmation/README.md) | [Validation review](confirmation/review.json) |
| Methodology selection on validation | [36-configuration review](experiments/methodology_review/README.md) | [Evidence](experiments/methodology_review/evidence.json) |

## Earlier evidence

These records support the [decision history](../docs/decisions.md); they are not additional final-model results. Keep original and corrected calibrations distinct.

- Initial CNN and references: `cnn_validation.json`, `baselines.json`; controlled reconstruction: `experiments/neural_summary.json` and `experiments/calibration.json`.
- Input, task and volume controls: `experiments/parallel_results.json`, `experiments/transformer_parallel_results.json` and `experiments/task_controls_review/comparison.json`.
- Earlier correction-model pretraining: `pretraining/pretraining_results.json` and `pretraining/paired_comparison.json`.
- Runtime and portability evidence: `colab_pilot.json` and `final/docker_check.json`.

Historical numerical records retain original fields and fingerprints. Their former standalone presentations and intermediate execution notebooks are not all part of this compact checkout. Consult the reproduction guide for the selected method; historical code references inside recorded JSON are provenance, not promises of files currently shipped here.
