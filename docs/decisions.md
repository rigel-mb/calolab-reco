# From experiments to the selected method

This is the short decision record. Detailed evidence lives in the linked reports; raw predictions, checkpoints and execution archives remain outside Git.

| Question | Experiment and result | Decision | Meaning and limitation |
| --- | --- | --- | --- |
| Why did the first networks fail? | Initial direct CNN/Transformer errors greatly exceeded the physical reference; input and loss controls followed. | Keep the failure as motivation, not as the final result. | A small model with one representation does not diagnose an entire architecture family. |
| Should models correct a reference? | Correction models improved energy on clean deposits; prior pretraining had similar accuracy and higher total cost. | Retain this as an earlier viable approach; use direct local models in the selected study. | The question becomes what measured local structure alone can support. |
| Does more training data solve it? | The earlier fixed-update 7k/14k/28k comparison showed limited gain. | Keep 28,118 training events; no new volume sweep. | This is not evidence that more data can never help. |
| Full grid or local input? | Local specialists improved over the tested full-grid pipelines; median contained clean energy was 99.52%. | Use the measured-maximum 7 x 7 window. | Models/tokenization also changed; this is a pipeline comparison, not a pure crop ablation. |
| Joint or separate tasks? | Under noise plus cut, Transformer position median fell from 0.39694 to 0.05775; energy MARE from 2.463% to 2.388%. | Use separate downstream training for energy and position. | Specialist loss weight is one instead of one half; the experiment does not isolate task interference. |
| Add noise and a threshold? | Position advantage appears under imperfect measurements; clean references remain stronger. | Main condition: matched noisy training/evaluation with a 50 MeV cell cut; retain clean and noise-only controls. | Published detector-response motivation, not a noise setting tuned to maximize neural advantage. |
| Are references strong enough? | Removing an excessive quadratic-calibration ridge penalty improved noise-plus-cut MARE from 17.563% to 2.051%, better than the Transformer. | Freeze the SVD quadratic reference with no ridge, alongside affine energy and periodic position. | The repair was post hoc and train-only; retain both original and corrected evidence. |
| Does pretraining transfer? | Validation motivated the final evaluation. On held-out test, mean position error falls from 0.07040 to 0.04643; mean energy MARE from 2.554% to 2.389%. | Keep one shared pretrained encoder with two independent fine-tunings; conclude a position benefit. | Same downstream architecture and update budget, with additional pretraining compute. Energy remains behind the 2.104% quadratic reference on average. |
| Are the gains stable? | All three primary paired position runs improve on test. Energy improves in one of three; the other two differences have event-bootstrap intervals spanning zero. | Report all seeds, noise draws and tails, including the weak direct-position run. | Earlier validation energy gains were not consistently retained on test. Three seeds and repeated noise on the same events do not establish general robustness. |
| Does fine-tuning save adaptation time? | At the first observed validation target crossing, mean position fine-tuning takes 2.03 min versus 4.01 min direct; initial pretraining costs 7.21 min. | Present marginal fine-tuning as the adaptation cost, with initial pretraining separately amortized over useful tasks. | These retrospective timings do not prove early-stopping savings: all runs completed 4,400 updates, and full fine-tuning schedules cost approximately as much as direct schedules. |
| When could shared pretraining pay back? | Mean position-like savings imply four adaptations; the per-seed scenarios span three to eight. | Keep this as a conditional reuse calculation, with comparable quality required on every task counted. | Only two tasks were studied, and only position shows a consistent final quality gain. New tasks, savings and the stopping rule remain untested. |
| Is the final evaluation reproducible? | All 5,935 held-out events were evaluated after freezing settings and checkpoints. Six local specialists passed native macOS ARM / Linux ARM64 Docker parity on 256 validation events. | Close the technical evaluation without test-based changes. | Bounded portability is demonstrated; Linux AMD64 and real detector data remain untested. |

## Evidence and scope

The [36-configuration validation review](../reports/experiments/methodology_review/README.md) records the final methodology-selection comparison, including the original and repaired energy calibration. The [validation report](../reports/confirmation/README.md) then checks the selected method over three training seeds; the [final report](../reports/final/README.md) gives held-out outcomes and reuse costs. Earlier compact numerical records are indexed in [reports](../reports/README.md). These stages are distinct: exploratory choices used validation, while the final test was opened only after freezing the method and weights.

Intermediate execution notebooks and dedicated utilities are not all included in this compact checkout. The decision history and aggregate evidence remain public; full earlier execution artifacts are preserved separately. Current code reproduces the selected method.

## Physical interpretation


An electromagnetic shower spreads the photon's energy over neighboring crystals. A measured local window captures most of that shower and avoids asking a small network to locate it across a mostly empty grid. Position is learned as an offset from the observed window origin, not as a correction to a calibrated barycenter.

Added noise models imperfect readout. The 50 MeV cut suppresses some measurements but removes genuine deposits too. It is below the 167 MeV electronic noise scale; removing negative fluctuations introduces a positive measurement bias. References must therefore be calibrated on the same readout and local window.

The window can select the wrong maximum: ten of 5,947 validation events failed badly in the primary exploratory noise draw, all around 1 GeV. The final primary test draw has seven distant maximum selections among 5,935 events, between 1.059 and 1.300 GeV. They remain in all scores and tail reports; no post-test energy cut is introduced. The few padded windows do not establish general robustness at physical detector edges.

Sources: [DeepCluster, section 3.2](https://link.springer.com/article/10.1140/epjc/s10052-024-12978-1) and [ClusTEX, sections 3.4-3.5](https://arxiv.org/html/2603.18172v2#S3.SS4). Their full candidate-building and reference algorithms are more elaborate than this bounded, one-photon study.
