# Data and detector conventions

The initial audit and source cross-check were performed on 20 September 2026. The three single-photon archives from [Zenodo record 18929909](https://zenodo.org/records/18929909), version 0.0.1, are by Y. Maidannyk and M. O. Sahin and licensed CC BY 4.0. Their sizes and checksums were verified. This document consolidates the storage audit and follow-up; source arrays, prepared values and the original manifest remain unchanged. Final evaluation is now complete; audit-time checks below are data diagnostics, not model performance results.

## Verified storage structure

| File | Keys | Shape per group | Type |
| --- | --- | --- | --- |
| `1photon_calo.npz` | `X_0` through `X_219` | `(10000, 30, 85)` | float32 |
| `1photon_en.npz` | `en_0` through `en_219` | `(10000,)` | float32 |
| `1photon_yc.npz` | `yc_0` through `yc_219` | `(10000, 2)` | float32 |

Matching numeric group suffixes and row counts support the documented association between the files. There is no certified physical-event identifier in these arrays. A `(group, row)` pair identifies storage provenance; matching keys alone cannot prove physical correspondence if the producer reordered rows. The direct ROOT comparison subsequently checked all values for source group 0, with no differences after float32 conversion. The other 219 groups were not checked against ROOT.

The files contain 2,200,000 events in 220 groups. Compressed files total 1,690,168,100 bytes. Decoded array payloads total 22,466,400,000 bytes, but the largest individual array is 102,000,000 bytes. Preparation reads one group array at a time; it never extracts the full corpus. Reading a compressed NPZ member still decompresses that member in memory.

## Sampling and quality

The sample contains 40,000 events drawn without replacement across the full indexed population with NumPy generator seed `20260920`. All 220 groups are represented. Source values are preserved without normalization, conversion, thresholding or selection based on targets.

Exact stored deposit maps are hashed with BLAKE2b-128. Identical hashes are kept in the same partition, even if their labels differ. With seed `20260921`, unique map groups are assigned with probabilities 0.70, 0.15 and 0.15:

| Partition | Events |
| --- | ---: |
| Train | 28,118 |
| Validation | 5,947 |
| Test | 5,935 |

The selected sample contains zero extra exact duplicate maps, zero nonfinite deposit/target values, zero negative deposits, zero all-zero events and zero nonpositive incident-energy labels. These checks do not certify unselected data. They do not detect approximately repeated showers or hidden links between events.

All selected labels are retained. Descriptive distributions and example plots use train only. Integrity checks cover the whole selected sample; test remained reserved until the final frozen evaluation; it was never used for model selection.

Derived numerical payloads occupy 409,160,000 bytes, plus small headers and a manifest. They remain under the external data root. The sample and partitions are reproducible and fingerprinted. The current protocol is described in the [methodology](methodology.md).

## Direct ROOT to NPZ comparison

A bounded HTTP range request retrieved bytes 0 through 33,554,431 of the 4,861,249,776-byte `ntuples.tgz` archive in [Zenodo record 18929909](https://zenodo.org/records/18929909). The response was HTTP 206 with the requested content range. Only the first complete member, `data/ntuple_0.root` (29,893,280 bytes), was read from this prefix; the full archive was neither downloaded nor extracted.

The member contains one tree, `CaloH`, with 10,000 entries. Its SHA-256 is `6adf0eab62e3ccc11215ca9a2dd03c0a7124a7aec4c4f15e889ce694a9b8504a`. Uproot 5.7.6 and NumPy 2.5.3 were used in a temporary environment, without changing the project's dependencies.

For every event in source group 0, the following reconstruction matches the NPZ arrays exactly after float32 conversion:

```python
en_0 = float32(InitialMomentum)
yc_0 = float32([InitialRow + 15, InitialColumn])
X_0[event, Row, Column - 85] = float32(EnergyVector)
```

The deposit assignment uses entries with positive energy and starts from a zero array. In this member, all positive entries have integer `Row` in 0..29 and `Column` in 85..169; no negative deposits were found. The original energy vectors have 5,100 entries, including zero entries with sentinel coordinates; they must not simply be reshaped into the NPZ grid.

| Comparison | Values checked | Differences |
| --- | ---: | ---: |
| Deposits | 25,500,000 | 0 |
| Incident energy | 10,000 | 0 |
| Impact coordinates | 20,000 | 0 |

The producer describes `InitialRow` as fractional iphi and `InitialColumn` as fractional ieta. Combined with the verified mapping, this identifies `coordinate_0` as local fractional iphi and `coordinate_1` as local fractional ieta. These are detector coordinates expressed through cell indexing, not automatically phi in radians, eta itself, or distances in centimeters.

This is a storage and alignment check, not a performance evaluation. Of these 10,000 source events, 114 belong to the prepared train subset. Correspondence for the other 219 source groups has not been checked against ROOT.

## Cell centers: the remaining geometry question

The current ROOT branches identify cells and impact coordinates but do not specify where each integer cell's center lies in the continuous target system. The verified +15 target offset and -85 deposit-column offset do not determine whether a center is at index or index + 0.5.

The earlier DeepCluster implementation uses `floor(y)` to identify the impact cell and index + 0.5 for its center: [preprocessing at a pinned revision](https://github.com/psimkina/DeepCluster/blob/2f32b9805389a58723e119710ac590b00b8c8b15/src/utils/data_preprocessing.py#L61-L67). That code concerns the older 51 x 51 toy calorimeter, so extending its convention to the current geometry remains a hypothesis.

Implications for this study:

- Direct regression of the stored coordinates can be evaluated in their stored units without selecting a center offset.
- A constant offset in a barycenter can be absorbed by a fitted affine intercept: if b' = b + c, then A b + a = A b' + (a - A c). Good calibrated performance therefore does not establish the physical coordinate origin.
- The raw barycenter bias, impact overlays, subcell boundaries and distances to detector edges require an explicit geometry convention. Their interpretation must wait for confirmation or be labeled as an assumed convention.
- Conversion to millimeters also requires the geometry mapping; multiplying every coordinate difference by a nominal crystal width is not sufficient evidence.

No current public simulator or ROOT conversion script defining this convention was found in the article, author pages, linked code or targeted branch-name searches. No author was contacted.

## Silicon thickness

The [first article version](https://arxiv.org/html/2603.18172v1) gives 33 mm in section 3.2 and the equivalent 3.3 cm in appendix 7. The [second version](https://arxiv.org/html/2603.18172v2), [published article](https://link.springer.com/article/10.1140/epjc/s10052-026-16097-x) and [author's LHCP page](https://ymaidannyk.web.cern.ch/lhcp/) agree on 33 mm.

The figure provides an independent numerical consistency check: [figure 11](https://arxiv.org/html/2603.18172v1/figures/Fig11.png) marks about 0.35 radiation lengths at eta = 0 and 0.81 near |eta| = 1.48. The [PDG silicon table](https://pdg.lbl.gov/2025/AtomicNuclearProperties/HTML/silicon_Si.html) gives a radiation length of 9.370 cm. For a radial cylindrical layer traversed from the origin, t / X0 multiplied by cosh(eta) gives:

| Radial silicon thickness | eta = 0 | Absolute eta = 1.479 |
| --- | ---: | ---: |
| 33 mm | 0.3522 | 0.8129 |
| 33 cm | 3.5219 | 8.1292 |

Thus 33 cm is very likely a factor-ten error in the Zenodo notice. This establishes the intended documented geometry strongly; the exact configuration that produced the deposited files has not been independently certified. No numerical correction to the arrays follows from correcting this description.

The article describes 1.25 million events, while these arrays contain 2.2 million. The difference in event count remains unexplained; this study does not claim exact source reproduction.

## Energy units, noise and threshold

For all 10,000 ROOT events, the following relation holds with a maximum absolute numerical discrepancy of 1.85e-13:

```text
sum(EnergyVector) / 1000 = TotalEn
```

This is a comparison of two representations of deposited energy, not a fit to the incident photon energy. The NPZ deposits exactly preserve `EnergyVector` after float32 rounding, and the NPZ energy label preserves `InitialMomentum`.

The generated photon range is documented as 1 to 100 GeV in the [Maidannyk presentation, page 6](https://indico.cern.ch/event/1655754/contributions/7178759/attachments/3338071/5981775/MODE_clustering.pdf). The exact internal scale relation, source descriptions and numeric energy range strongly support MeV deposits and GeV energy labels. Geant4 also uses MeV as a [default internal energy unit](https://geant4.web.cern.ch/documentation/dev/bfad_html/ForApplicationDevelopers/Fundamentals/unitSystem.html), although that convention alone would not prove an exported file's units. The ROOT branch titles contain no explicit unit annotations.

For the training example at source group 0, row 21, ROOT stores incident energy 82.379... and total deposited energy 79.117755785..., while the NPZ deposit sum is approximately 79,117.754635. The small last-digit difference is float32 rounding. Division by exactly 1000 expresses the deposits on the same scale; it does not force deposited energy to equal incident energy.

Section 3.4 and figure 1 of the published article distinguish simulated deposits from the signal after readout smearing and a 50 MeV cut. These are separate processing stages. The loaded arrays are not the final thresholded signal under the supported unit interpretation.

Checks on the 28,118 prepared training events give:

| Quantity | Result |
| --- | ---: |
| Positive cells | 4,671,047 |
| Positive cells below 50 stored units | 4,095,951 (87.6881%) |
| Fraction of total stored energy in those cells | 1.4478% |
| Smallest positive stored deposit | 7.91624e-9 |

Small deposits comprise many cells but a small fraction of total energy. These statistics rule out a final blanket cut at 50 stored units. They do not recover the complete processing history or prove the absence of every possible noise operation. Source semantics and exact ROOT correspondence support the interpretation of pre-readout simulated deposits.

Applying only a cut would not reproduce the documented chain. Adding noise and then thresholding changes the input distribution, especially its small deposits, zeros and spatial structure; it must be a deliberate protocol choice. Source and prepared arrays remain unchanged. The notebook's train-only diagnostic examines removal at 50 stored units: affected cells, energy fraction and relative barycenter displacement by energy. A constant center offset cancels in that displacement; it is not a physical position-resolution measurement.

The median Euclidean displacement is 0.0057 index units on all train, with P95 0.0234. For the 2,568 events between 1 and less than 10 GeV under the supported interpretation, the cut removes 90.73% of positive cells and 5.99% of deposited energy; median displacement is 0.0225 and P95 is 0.0575 index units. Fractions are pooled within each population, not averaged per event. No training event becomes empty. These are input diagnostics, not measured model degradation.

The final study explicitly trains and evaluates under matched clean, noise-only and noise-plus-cut conditions. Its selected noise law and seeds are in the [methodology](methodology.md); this supersedes the earlier proposal to study only perturbations of already-trained clean models.

## Relationship to CaloFound

The [official CaloFound slides](https://indico.cern.ch/event/1655754/contributions/7178905/attachments/3338097/5981814/CaloFound_Mode.pdf) show the 30 x 85 grid and link to the Maidannyk article on PDF page 7. On page 13 they describe dynamic noise:

```text
sigma(E)^2 = 0.167^2 + 0.03^2 E + (0.0035 E)^2
E_observed = E + alpha sigma(E) epsilon
alpha ~ Uniform(0, 2) per event
epsilon ~ Normal(0, 1) independently per crystal
crystal cut: 0.05 GeV
```

The slides therefore corroborate noise as an explicit processing operation; they do not say it has already been applied to our NPZ files. Their event counts also differ from the public single-photon corpus, and there is no direct file manifest linking their selection to these NPZ files.

Page 14 uses `log(1 + E / s)` with s = 1 in its input units. This is a concrete source for the initially explored transformation family; fitting s on our training sample was an adaptation, not a requirement of that source or a demonstrated improvement.

The final input is linearly scaled, not logarithmic. CaloFound is scientific context, not an exact reproduction target.

## Compute feasibility

The [recorded T4 pilot](../reports/colab_pilot.json) checked compute cost and next-update replay on small workloads; it was not an accuracy result. The full selected GPU training and final evaluation are reported separately. The original prepared manifest retains its audit-time training gate as provenance; it is not the current project status.

The ROOT excerpt and numerical cross-check artifacts remain in external data storage. Physical coordinate origins, full export processing history and exact production geometry remain limitations. The [data notebook](../notebooks/01_data_audit.ipynb) provides the illustrated entry point.
