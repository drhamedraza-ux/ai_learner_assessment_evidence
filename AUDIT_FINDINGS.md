# P02 verification and correction report

**Scope.** Independent validation of the 22-run P02 package: execution correctness,
reproducibility, dataset and configuration correctness, leakage, and consistency between
methodology, code, results, tables and figures. Every claim below was produced by running
something, not by reading code alone.

**What could not be done here.** The raw datasets live at `/home/hamed/Downloads/Research/dataset/`
and are not in the archive; there is no GPU in this environment. The nine RQ2/RQ3 training
runs (22.3 GPU-hours) therefore could not be re-executed. Everything downstream of training
*could* be, because `predictions_test.npz` is archived for all nine runs.

---

## 1. Findings

### Finding A — DKVMN is denied the most recent interaction (HIGH; RQ3 and RQ4 affected)

`DKVMN.forward` reads the value memory **before** writing interaction *t*. Its prediction
for target *t+1* is therefore formed from interactions 0…*t*−1, while all seven other
models use 0…*t*. This is not the Zhang et al. (2017) ordering, in which (q_t, a_t) is
written before q_{t+1} is read.

Established by perturbation probe (`02_dependency_probe.py`): flip the response at one
position, observe which output positions move.

| model | prediction for target *j* uses |
|---|---|
| DKT, SAKT, AKT, DKT-F, PCDT, PCDT-noPsi, Control | 0 … *j*−1 ✓ |
| **DKVMN** | **0 … *j*−2** ✗ |

**Why it matters.** The omitted interaction is most informative when the next item follows
minutes later and least informative after a week, so the handicap costs more within session
than post-gap — which is precisely the pattern RQ4 reports. Leg 1 of the pre-registered
rule is `within_rank > 3`: the comparator must rank *outside* the top 3 within session.

**The margins are small enough for this to be decisive.**

| dataset | seed | DKVMN within-session AUC | rank | AUC gain needed to enter the top 3 |
|---|---|---|---|---|
| Junyi | 20260204 | 0.7428 | 4 | **+0.0016** (Control at 0.7443) |
| EdNet-KT1 | 20260204 | 0.7110 | 5 | **+0.0021** (DKT at 0.7130) |

Recovering the most recent interaction plausibly buys more than that. Note also that the
rule confirms on the two datasets where the comparator is DKVMN and fails on the one where
it is a correctly implemented model (PCDT-noPsi on ASSISTments).

**Direction of effect on each research question**

| RQ | exposure |
|---|---|
| RQ2 (H1 not supported) | **Safe, arguably strengthened.** Correcting DKVMN can only shrink PCDT's margin over the strongest baseline. |
| RQ3 vs Control, vs DKT-F | **Unaffected.** Neither comparator is DKVMN. |
| RQ3 vs DKVMN on Junyi (9 of 9 negative) | **Contaminated, overstating the negative.** The difference-in-differences subtracts the within-session gap — the term the defect inflates. Correction moves the result toward zero. |
| RQ4 on EdNet and Junyi | **Not defensible until DKVMN is retrained.** Both confirmations rest on DKVMN's within-session rank. |
| Model-independence claim (EdNet DKT 3rd→4th vs DKVMN 5th→1st) | **Contaminated** — it is a DKVMN inversion. |

**Fixed** in `pcdt_p2.py` behind `Config.dkvmn_legacy_lag` (default `False` = canonical;
`True` reproduces the archived runs and prints a warning). The fix is parameter-neutral,
verified: 79,121 parameters either way, so the capacity search lands on the same width.

**Resolution path**: `05_DKVMN_REFIT__*` (9 notebooks) — see §4.

---

### Finding B — `summary_rq4_by_dataset` is blank, and the anti-cherry-picking warning never fired (HIGH; fixed and re-run)

`04_SUMMARY__all` reads `verdict_rq4.json` and looks for `primary_confirmed` at the top
level. The file stores it one level down, under `"verdict"`. Every field resolved to
`None`, so the table read:

```
assistments_2021,complete,-,-,-,-
ednet_flat,complete,-,-,-,-
junyi,complete,-,-,-,-
```

The downstream guard `done = RQ4[RQ4.confirmed.isin(["CONFIRMED","NOT CONFIRMED"])]` was
consequently empty, so this block never printed:

> *** The effect is DATASET-DEPENDENT. A claim about the field's default evaluation
> protocol cannot rest on the subset where it replicated — report the datasets where it
> did not. ***

The actual pattern is 2 of 3 confirmed — exactly the case the warning exists to catch.
A second bug sat behind the first: `horizons_used` read `v["adequate_horizons"]`, a key
notebook 03 never writes.

**Fixed and re-executed against the real artifacts.** Corrected output:

```
dataset,status,comparator,confirmed,secondary,horizons_used,failing_leg
assistments_2021,complete,PCDT-noPsi,NOT CONFIRMED,not passed,"d1, d7",leg1: comparator ranks inside the top 3 within session
ednet_flat,complete,DKVMN,CONFIRMED,passed,"d1, d7",none
junyi,complete,DKVMN,CONFIRMED,passed,"d1, d7, d28",none
```

and the dataset-dependence warning now fires. `horizons_used` is derived from the
confirmatory seed's `table9` (the file the rule was evaluated on) rather than from a key
that may be absent; `03_RQ4__*` now also records `adequate_horizons` for future runs.

---

### Finding C — a genuine "CI includes zero" result becomes NaN on re-read (MEDIUM; fixed and re-run)

`summary_rq3_cross_seed.csv` labelled inconclusive rows with the literal string `null`.
`pandas.read_csv` treats `null` as a missing value by default, so 18 of 55 rows come back
as `NaN` — indistinguishable from the "bootstrap could not be formed" case that the same
line of code exists to keep separate. The conflation the comment warns against, in the
opposite direction.

Fixed: the label is now `inconclusive (CI spans zero)`. Re-run confirms 22 positive /
15 negative / 18 inconclusive / **0 NaN**.

---

### Finding D — one integrity check was structurally vacuous (LOW; fixed)

`finalize()` hashes every output, then writes the run's completion banner *into*
`logs/console.log`. The recorded hash could therefore never match the file on disk. I
confirmed on all 22 runs that the manifest hash matches a prefix of the log, so nothing was
tampered with — but that one check could not have detected tampering. `p2_runtime.py` now
records `sha256: null` with an explanatory note instead of a hash that is guaranteed wrong.

---

### Finding E — each seed is a different learner sample, not just a different split (MEDIUM; reporting)

`_subsample_learners(..., max_learners=20000)` runs **before** the `min_seq_len ≥ 10`
filter, and the raw datasets have more than 20,000 learners. Each seed therefore draws a
different 20,000 and the surviving population differs:

| dataset | learners (seeds 0/1/2) | interactions | KCs |
|---|---|---|---|
| ASSISTments 2020-21 | 15,406 / 15,378 / 15,325 | 1,145,941 / 1,122,656 / 1,135,067 | 396 / 398 / 394 |
| EdNet-KT1 | 15,324 / 15,287 / 15,235 | 4,934,433 / 4,888,578 / 5,071,150 | 142 |
| Junyi | 16,921 / 16,908 / 16,844 | 4,482,120 / 4,475,927 / 4,408,280 | 1,326 |

Not a defect — cross-seed agreement is a *stronger* robustness statement when the sample
varies too. But the manuscript's Table 1 characteristics are seed-0 specific and are
presented as properties of the dataset, and any wording along the lines of "three random
splits" understates what actually varies. State it as *three independent learner samples,
each with its own split*.

---

### Finding F — the RQ4 comparator is selected on the seed the rule is tested on (MEDIUM; design)

`03_RQ4__*` names the comparator as `argmax(auc_mean_postgap)` on the **confirmatory**
seed, then evaluates leg 2 ("first at every adequate gap") on that same seed. Since the
comparator is defined as the model with the highest *mean* post-gap AUC there, leg 2 is
substantially favoured by construction. Pre-registration does not repair this — the rule
was fixed in advance, but selection and confirmation share a data set. Either name the
comparator on the exploratory seeds and confirm on seed 20260204, or state the dependency
explicitly in the Threats section.

---

## 2. What was verified and passed

| check | result |
|---|---|
| Manifest integrity: sha256 of every recorded output | **409 files, 0 mismatches** |
| Consumer provenance: sha256 of every recorded input | **48 inputs, 0 mismatches** — RQ4 and SUMMARY provably read what the producers wrote |
| Files on disk vs files recorded | 0 unrecorded, 0 missing, sizes match |
| Every RQ2/RQ3 table recomputed independently from `predictions_test.npz` (tables 1, 2, 5, 6, 9; all metrics; per row+column, never by set membership) | **4,983 cell checks across all 9 runs, 0 mismatches** |
| RQ4 primary rule re-derived from the raw `table9` files | **60 checks, 0 mismatches** |
| RQ1 verdicts re-derived from the pre-registered floors | **54 cells, 0 mismatches** |
| RQ1 verdict/convergent matrices and `audited_subset_*.json` vs the audit tables | **117 checks, 0 mismatches** |
| SUMMARY tables re-derived from the 21 source runs | **277 checks, 0 mismatches** (the 24 remaining are Findings B and C) |
| Label leakage: does any model's prediction for target *j* use position ≥ *j*? | **No, for all 8 models** |
| Preprocessing leakage | **None found.** Learner-level split; skill vocabulary from train only (unseen → index 0); behaviour median/μ/σ from train only; `dt` differenced within learner; gap stratification uses a covariate, not an outcome |
| Bootstrap unit | Resamples **learners**, not interactions — correct for repeated measures |
| Config drift across the 18 producer runs | **None** — identical apart from dataset and seed |
| Environment drift across all 22 runs | **None** — one environment |
| Tracebacks or warnings in the 22 console logs | **None** beyond the intended `I_t` circularity disclosure |
| Isolation | `isolation_ok=True` on all 22, v2 window-overlap classifier, concurrent runs correctly benign |
| Manuscript claim set vs artifacts | RQ1 correlations, dataset characteristics, RQ2 deltas, RQ3 cells and RQ4 verdicts all match the archived tables |

**Reproducibility statement supported by the evidence**: the recorded results are exactly
what the archived predictions produce; the archived predictions cannot be re-derived here
without the raw data and a GPU.

---

## 3. What remains unverified

- **Bootstrap confidence intervals** (`table3`, `table5`, `table12`). Point estimates all
  reproduce. Whether the percentile CIs reproduce bit-for-bit under a different NumPy
  (1.26 here vs 2.3.5 on the run machine) was still computing when this report was written;
  `audit/boot_check.py` completes the check.
- **RQ1 correlation values themselves.** The audit statistics are computed from the raw
  interaction logs, which are absent. Verdict logic, matrices, subset files and cross-seed
  stability are verified; the underlying correlations are not independently recomputed.
- **Figures.** Not re-rendered or inspected.

---

## 4. Corrective package

**All 34 notebooks regenerated.** See `README.md` for the change table, the two recovery
paths and the run order.

```
pcdt_p2.py                          DKVMN ordering fixed; dkvmn_legacy_lag; check_alignment()
p2_runtime.py                       console.log hash marked non-hashable
01_RQ1__<ds>__seed<n>         (9)   library guard; table0 declares the seed's sample (E)
02_RQ2RQ3__<ds>__seed<n>      (9)   library guard; TEMPORAL ALIGNMENT GATE before training
03_RQ4__<ds>                  (3)   adequate_horizons; selection-sensitivity cell (B, F)
04_SUMMARY__all                     Findings B and C fixed; RE-EXECUTED, output above
05_DKVMN_REFIT__<ds>__seed<n> (9)   retrain DKVMN only, substitute, recompute
06_RQ4_REFIT__<ds>            (3)   re-apply the UNCHANGED rule to the corrected table9
```

The 22 originals were **patched at explicit anchors, never rewritten from memory**; every
anchor is asserted to match exactly once, so no patch can silently no-op and no
methodology can drift in unnoticed. `verification/regen_all.py` reproduces them.

### The alignment gate — the check that would have caught Finding A

`02_RQ2RQ3` now runs `P.check_alignment()` before any training. It perturbs one response
and records which predictions move, for all eight models, in seconds. It refuses to
continue if any model **LAGS** (target *j* does not use position *j*−1) or **LEAKS**
(target *j* uses position *j* or later). Verified against both failure modes: it catches
the legacy DKVMN and a deliberately leaking model, and does not false-positive on
`Control`, which legitimately zeroes its state at a session boundary.

**Why refitting one model is valid.** The other seven are independent of DKVMN; the split
is a deterministic function of dataset and seed; the fix is parameter-neutral. Notebook 05
proves the premise rather than assuming it: cell 4 is a **split gate** that raises unless
the rebuilt split reproduces the archived bucket counts exactly, and after substitution it
asserts that no model other than DKVMN moved. This avoids producing a second value for any
quantity the archived runs already report.

**Cost.** DKVMN only, not the full eight-model grid — roughly one eighth of the 22.3
GPU-hours. The nine refits are mutually independent and parallel-safe.

**Verified by execution, on a fixture** (real data absent):

- 3 × `05_DKVMN_REFIT` and 1 × `06_RQ4_REFIT` execute clean, Run All from a fresh kernel
- after substitution: 7 models bit-identical, DKVMN changed, legacy column retained
- **negative test**: an archived bucket count altered by +7 → the split gate fires and
  names the failing cell; file restored afterwards
- **the full regenerated chain executed end to end on fixture data**: `01` ×3 → `02` ×3 →
  `03` → `04`. The alignment gate fires and passes; the selection-sensitivity diagnostic
  runs; the corrected RQ4 summary populates every field the archived code left as `-`
- three bugs in this package's own first draft were found this way and fixed: `int(rank)`
  on a NaN rank; `idxmax` on an all-NaN column; and an alignment test that checked
  membership rather than the maximum, which flagged `Control`'s legitimate session reset
  as a lag. The first sat before `save_tables` and would have destroyed a completed
  training run — the same defect class the original audit fixed in notebook 02. Tables are
  now written before anything is printed or plotted.

The fixture's DKVMN delta (+0.0005) is **not** an estimate of the real effect: one epoch on
synthetic data. Only the real refit can size it.

## 5. Recommended order

1. Drop in the corrected `pcdt_p2.py` and `p2_runtime.py`.
2. Run `05_DKVMN_REFIT__junyi__seed2` first — Junyi is the only dataset with a real 4-week
   horizon and seed 20260204 is the confirmatory seed, so this single run settles most of
   the exposure. Check the split gate passes and read the DKVMN within/post-gap movement.
3. Run the remaining eight refits, then the three `06_RQ4_REFIT` notebooks.
4. Re-run `04_SUMMARY__all` (corrected).
5. Decide what the paper claims **before** reading the new verdicts, and record that
   decision. If RQ4 no longer confirms, that is the finding — and the archived result must
   not be reported as it stands.
