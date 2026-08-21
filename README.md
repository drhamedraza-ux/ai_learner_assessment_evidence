# PCDT-P2

**A Construct-Validity Audit and Gap-Stratified Evaluation Framework for Persistent Cognitive Digital Twins**

This repository contains the complete experimental pipeline for the study, covering four pre-registered research questions across three large-scale learning-interaction datasets, three random seeds, and eight learner models. The codebase supports construct-validity auditing, model training, cross-session persistence analysis, and evaluation-protocol comparison, enabling researchers to reproduce and extend the results reported in the manuscript.

**Every empirical result in this study is a null or a negative finding.** The pipeline is designed to make that verifiable rather than to make a method look good — see [Results summary](#results-summary).

---

## Overview

Learner models are conventionally benchmarked on pooled predictive accuracy within a single session. This study asks whether the multidimensional learner-state representation such models are assumed to carry survives empirical audit, whether a persistence operator confers any measurable advantage across sessions, and whether the evaluation protocol itself changes which model is selected.

The framework proceeds in four stages. A construct-validity audit tests each candidate state dimension against pre-registered reliability, convergent-validity, and recovery floors. Eight models are then trained under strict parameter matching on a learner-level, gap-stratified split. Predictions are stratified by the elapsed gap preceding each target item, and models are compared both within session and after gaps of one day, one week, and four weeks. A pre-registered decision rule finally tests whether the two protocols select different models.

## Key features

- **Pre-registered thresholds** — reliability ≥ 0.60, |convergent r| ≥ 0.30, simulator recovery ≥ 0.50, minimum meaningful ΔAUC of 0.010, adequacy floor of 2,000 scored points per gap bucket
- **Capacity-matched comparison** — every model is sized to 350,000 parameters ± 5% before the learning-rate grid, so no result can be attributed to model size
- **Gap-stratified evaluation** — five horizons (within-session, same-day, 1 day, 1 week, 4 weeks) derived from the actual inter-item interval
- **Temporal alignment gate** — refuses to train if any model lags behind or leaks ahead of the others (see [Corrections](#corrections))
- **Enforced run isolation** — each notebook owns one output folder; a run that writes into another run's folder raises rather than completing
- **Full provenance** — every output and every consumed input is recorded with a SHA-256 checksum in a per-run manifest
- **Independent verification harness** — recomputes every reported table from the raw saved predictions

## Repository structure

```
PCDT-P2/
├── pcdt_p2.py                          # Canonical library: loaders, audit, split builder,
│                                       #   8 models, training harness, metrics, statistics
├── p2_runtime.py                       # RunContext: per-run folders, manifests, isolation
│
├── 01_RQ1__<dataset>__seed<n>.ipynb    # ×9  Construct-validity audit           (CPU)
├── 02_RQ2RQ3__<dataset>__seed<n>.ipynb # ×9  Training, prediction, persistence   (GPU)
├── 03_RQ4__<dataset>.ipynb             # ×3  Protocol-divergence decision rule   (CPU)
├── 04_SUMMARY__all.ipynb               #     Cross-dataset, cross-seed synthesis (CPU)
│
├── 05_DKVMN_REFIT__<dataset>__seed<n>.ipynb  # ×9  Single-model refit (see Corrections)
├── 06_RQ4_REFIT__<dataset>.ipynb             # ×3  Rule re-applied to the refit
│
├── verification/
│   ├── verify_runs.py                  # Six-layer independent verification harness
│   ├── 02_dependency_probe.py          # Model temporal-alignment probe
│   ├── regen_all.py                    # Regenerates all notebooks by anchored patching
│   └── HOW_TO_VERIFY.md
│
├── PCDT_P2_runs/runs/<RUN_ID>/         # Outputs: tables/ figures/ artifacts/ logs/ manifest.json
├── AUDIT_FINDINGS.md                   # Code audit: six findings and their resolution
└── README.md
```

Dataset and seed are **hardcoded in every notebook**. There is nothing to edit before running, and no notebook can silently be run against the wrong data.

### `pcdt_p2.py`

The single canonical implementation. Dataset loaders, the construct-validity audit, the learner-level gap-stratified split builder, all eight model architectures, the capacity-matching and training harness, the metric definitions, and the paired-bootstrap statistics. It replaces roughly 8,000 lines of duplicated notebook code from earlier pipeline versions, so a change to a metric or a split rule takes effect everywhere at once.

### `p2_runtime.py`

`RunContext` gives each notebook its own output folder and writes a manifest recording configuration, environment, runtime, every output with its SHA-256 checksum, every input consumed with its checksum, and an isolation verdict. Isolation is enforced rather than promised: the context snapshots sibling run folders at start, re-scans at finish, and raises if a **finished** run was modified. Concurrent runs writing into their own folders are classified by execution-window overlap and are never treated as violations, so the nine GPU runs can be executed in parallel.

### Notebooks 01–04 — the main pipeline

| Notebook | Research question | Compute |
|---|---|---|
| `01_RQ1` | Which learner-state dimensions survive a construct-validity audit? | CPU, ~1 min |
| `02_RQ2RQ3` | Does the persistent representation predict better than tuned baselines, and does it retain that across sessions? | GPU, ~1.5–3.5 h |
| `03_RQ4` | Do within-session and post-gap protocols select different models? | CPU, ~10 min |
| `04_SUMMARY` | Cross-dataset and cross-seed synthesis | CPU, < 1 min |

RQ2 and RQ3 share one notebook deliberately. Both need the same eight trained models on the same split; separating them would double the GPU cost and produce two AUC tables that disagree at the fourth decimal for the models that are not bit-reproducible. The governing principle throughout is **one number per quantity, not one notebook per question**.

### Notebooks 05–06 — targeted refit

Retrain a single model and substitute its predictions into an existing run, then recompute every downstream table. Used to repair the DKVMN defect described below at roughly an eighth of the cost of a full re-execution. Notebook 05 refuses to proceed unless the rebuilt split reproduces the archived bucket counts exactly, and asserts afterwards that no model other than the refitted one moved.

## Models

| Family | Models |
|---|---|
| Knowledge tracing baselines | DKT, SAKT, AKT, DKVMN |
| Time-aware baseline | DKT-F (receives the elapsed gap as an explicit feature) |
| Proposed | PCDT (persistence operator Ψ), PCDT-noPsi (ablation) |
| Control | Session-resetting control |

All are capacity-matched to 350,000 parameters ± 5% and tuned over the same learning-rate grid {5e-4, 1e-3, 2e-3}.

## Datasets

| Dataset | Learners | Interactions | KCs | Behavioural signals |
|---|---|---|---|---|
| ASSISTments 2020-21 | 15,406 | 1,145,941 | 396 | latency, hint, attempt |
| EdNet-KT1 (flat) | 15,324 | 4,934,433 | 142 | latency only |
| Junyi Academy | 16,921 | 4,482,120 | 1,326 | latency, hint, attempt |

Counts are for the primary seed. Each seed draws its own 20,000-learner subsample before the minimum-sequence-length filter, so learner and interaction counts differ slightly between seeds — the notebooks state this explicitly and record the seed alongside the counts.

Set the local paths in each notebook's configuration cell. See [Data availability](#data-availability) below.

## Installation

```bash
git clone https://github.com/<username>/PCDT-P2.git
cd PCDT-P2

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python 3.13, PyTorch (CUDA build), NumPy, pandas, scikit-learn, matplotlib, Jupyter.

Reference environment: Python 3.13.9, PyTorch 2.12.0.dev, NumPy 2.3.5, scikit-learn 1.7.2, on an NVIDIA RTX 5080. The full pipeline takes roughly 20 GPU-hours; the nine `02_RQ2RQ3` runs are mutually independent and parallel-safe.

## Usage

Run in numerical order. Each notebook is Run-All safe from a fresh kernel.

```
1.  01_RQ1__*        ×9   audit runs; produces the retained-dimension verdicts
2.  02_RQ2RQ3__*     ×9   training and evaluation; the expensive stage
3.  03_RQ4__*        ×3   applies the pre-registered decision rule
4.  04_SUMMARY__all       cross-dataset synthesis
```

Verify a completed run tree without moving it:

```bash
python verification/verify_runs.py \
    --runs     PCDT_P2_runs/runs \
    --baseline /path/to/preserved/baseline/runs \
    --lib      . \
    --out      ./verification_output
```

The harness checks manifest integrity, configuration and environment drift, model temporal alignment, full recomputation of every table from the saved predictions, re-derivation of the RQ1 and RQ4 decision logic, and comparison against a preserved baseline. It emits a report plus a small evidence bundle containing manifests, tables and logs only — around 0.5 MB for 34 runs, so verification can be shared without transferring model weights or prediction arrays.

## Results summary

| Question | Outcome |
|---|---|
| RQ1 — construct validity | Three of six audited dimensions retained; three rejected. Verdicts unanimous across all datasets and seeds |
| RQ2 — predictive advantage | **Not supported** on any dataset. Largest margin over the strongest tuned baseline is +0.0034 AUC, below the pre-registered 0.010 threshold |
| RQ3 — cross-session persistence | **Negative** against the strongest baseline: the persistent model loses ground as the gap grows |
| RQ4 — protocol divergence | **Not confirmed** on any of the three datasets |

Two disclosures are carried in the manuscript rather than hidden here. The `I_t` retain verdict is **circular** — its reliability is borrowed from another dimension and its convergent validity is numerically the same statistic as its recovery score. And two audit statistics are estimator artefacts: one reliability of exactly 1.000 arises from a deterministic construction, and one split-half of exactly 0.000 is a floor on a negative correlation. All three are flagged in the audit output itself.

## Corrections

The pipeline was audited after its first full execution, and one substantive defect was found and repaired.

**DKVMN read its value memory before writing the current interaction.** Its prediction for target *j* was formed from interactions 0…*j*−2 while every other model used 0…*j*−1, leaving one comparator under-informed by a single interaction. The handicap cost more within session than after a gap, which is precisely where the RQ4 decision rule looks. Repairing it raised DKVMN by +0.0031 to +0.0213 AUC and **changed the RQ4 verdict from confirmed to not confirmed on two of the three datasets**.

The fix sits behind `Config.dkvmn_legacy_lag` (default `False`, canonical ordering; `True` reproduces the pre-correction runs). Because the correction is parameter-neutral, the capacity search is unaffected.

`P.check_alignment()` now runs before training in every `02_RQ2RQ3` notebook. It perturbs a single response, records which predictions move, and refuses to continue if any model lags behind or leaks ahead of the others — seconds of work ahead of hours of GPU time. It is verified against both failure modes.

`AUDIT_FINDINGS.md` documents this and five further findings in full.

## Reproducibility

Six of eight models reproduce to full precision across runs. **AKT and SAKT do not** — AKT has been observed to drift by up to 0.0021 AUC between otherwise identical executions. No conclusion in the study depends on either model. Bootstrap confidence intervals are seeded but have not been verified bit-for-bit across NumPy versions.

Every number reported in the manuscript is recomputed programmatically from the run artifacts; none is transcribed by hand.

## Data availability

The source datasets — **ASSISTments 2020-21**, **EdNet-KT1**, and **Junyi Academy** — are publicly available from their original providers and are not redistributed in this repository.

The processed data derived in this study — the per-seed learner subsamples, the gap-stratified learner-level splits, the constructed behavioural features, and the complete run artifacts including saved test predictions — are available upon reasonable request. Please contact Syed Hamed Raza (Email: dr.hamedraza@gmail.com / sp24-pcs-007@cuilahore.edu.pk) or the corresponding author for data access.

## Citation

```bibtex
@article{raza_pcdt_p2,
  title   = {},
  author  = {Raza, Syed Hamed and Sohail, Abid},
  journal = {},
  year    = {},
  doi     = {}
}
```

## Author

**Syed Hamed Raza**
Department of Computer Science, COMSATS University Islamabad, Lahore Campus, Pakistan

Supervised by **Dr. Abid Sohail**.

## Contact

**Syed Hamed Raza**
dr.hamedraza@gmail.com · sp24-pcs-007@cuilahore.edu.pk

**Corresponding Author: Dr. Abid Sohail**
abidbhutta@cuilahore.edu.pk

## License
This repository is intended for academic and non-commercial research purposes.
<!-- Choose a license before the repository is made public; MIT is common for academic code. -->
