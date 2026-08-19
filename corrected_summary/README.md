# P02 — corrected package

Every notebook in the suite, regenerated against the corrected library. 34 notebooks:
the 22 originals plus 12 that recover the archived results without a full re-run.

Read `AUDIT_FINDINGS.md` first — it says what was wrong and what the evidence is.

---

## How these were produced

The 22 originals were **patched at explicit anchors, not rewritten**. Every anchor is
asserted to match exactly once, so a patch cannot silently fail and no methodology can
drift in unnoticed. `verification/regen_all.py` is the generator; run it against the
archived package to reproduce these files byte-for-byte.

Changes per notebook, measured against the archived original:

| notebook | code cells | lines +/− |
|---|---|---|
| `01_RQ1__*` ×9 | 11 → 11 | +23 / −2 |
| `02_RQ2RQ3__*` ×9 | 18 → **19** | +21 / −0 |
| `03_RQ4__*` ×3 | 10 → **11** | +61 / −0 |
| `04_SUMMARY__all` | 8 → 8 | +41 / −3 |

Nothing else in the methodology moved: same split, same capacity budget, same lr grid,
same pre-registered floors, same rule, same seeds.

---

## What changed

### Library

`pcdt_p2.py` (52 lines) — **`DKVMN.forward` now writes interaction *t* before reading**
(Finding A). Behind `Config.dkvmn_legacy_lag`, default `False` = canonical; `True`
reproduces the archived runs and prints a warning. Parameter-neutral (79,121 either way),
so the capacity search lands on the same width. Adds `check_alignment()`.

`p2_runtime.py` (11 lines) — `console.log` is recorded with `sha256: null` and a note
rather than a hash that could never match (Finding D).

### Every notebook — library guard

Refuses to run against a pre-audit `pcdt_p2.py`. Without it, a stale copy on the path
silently reinstates the DKVMN defect and every table is quietly wrong again.

### `02_RQ2RQ3` — temporal alignment gate (new cell, before training)

Runs in seconds, ahead of ~2.5 h of GPU work. Perturbs one response and checks which
predictions move, for all eight models. Refuses to continue if any model **LAGS** (target
*j* does not use position *j*−1 — under-informed relative to its competitors) or **LEAKS**
(target *j* uses position *j* or later — can see its own answer).

This is the check that would have caught Finding A on day one. Verified against both
failure modes: it catches the legacy DKVMN, and it catches a deliberately leaking model.
It does not false-positive on `Control`, which legitimately zeroes its state at a session
boundary.

### `01_RQ1` — the learner sample is seed-specific (Finding E)

`table0` gains `seed` and `sampled_from_max_learners`, and the notebook states plainly
that these counts belong to this seed's sample, not to the dataset. **The regenerated
`table0` therefore has two more columns than the archived one — by design.**

### `03_RQ4` — records `adequate_horizons`, adds a selection-sensitivity cell (Findings B, F)

The pre-registered rule and its verdict are **unchanged and remain primary**. A new cell
re-runs the identical rule with the comparator named from the *exploratory* seeds instead
of the confirmatory one, and flags disagreement. Saved as `table13_selection_sensitivity`.

### `04_SUMMARY__all` — Findings B and C

Reads the verdict at its actual nesting; derives `horizons_used` from the confirmatory
seed's `table9`; carries `secondary` and `selection_sensitive`; writes
`inconclusive (CI spans zero)` instead of `null`, which `pandas.read_csv` silently
converts back to `NaN`.

---

## Two recovery paths

The archived results were produced by the defective DKVMN. Pick one.

### Path 1 — targeted refit (~1/8 the GPU time) — `05` and `06`

Retrain DKVMN only, substitute its column into the archived prediction matrix, recompute
everything downstream. Valid because the other seven models are independent of it, the
split is a deterministic function of (dataset, seed), and the fix is parameter-neutral.
`05` does not assume that: **cell 4 is a split gate** that raises unless the rebuilt split
reproduces the archived bucket counts exactly, and after substitution it asserts that no
model but DKVMN moved.

```
05_DKVMN_REFIT__junyi__seed2          <- run this first (see below)
05_DKVMN_REFIT__<ds>__seed<n>         x9, mutually independent, parallel-safe
06_RQ4_REFIT__<ds>                    x3
04_SUMMARY__all
```

Produces no second value for any quantity the archived runs already report. RQ1 is
untouched by Finding A and does not need re-running.

### Path 2 — full re-run (~22.3 GPU-h) — `01` … `04`

One coherent suite, no substitution provenance to explain to a reviewer. Outputs will not
be byte-identical to the archived ones: the seven non-DKVMN models are retrained from the
same seeds and should land in the same place, but `table0` gains two columns and floating
point is not promised to be bitwise reproducible across runs.

```
01_RQ1__<ds>__seed<n>                 x9   (CPU, ~1 min each)
02_RQ2RQ3__<ds>__seed<n>              x9   (GPU, ~2.5 h each)
03_RQ4__<ds>                          x3
04_SUMMARY__all
```

**Path 1 is the better scientific choice** unless you have reason to distrust the archived
predictions — and §2 of `AUDIT_FINDINGS.md` gives 4,983 independent cell checks saying you
should not.

---

## Run `05_DKVMN_REFIT__junyi__seed2` first

Junyi is the only dataset with a real 4-week horizon, and 20260204 is the confirmatory
seed. That single run settles most of the exposure. Check that the split gate passes, then
read DKVMN's within-session and post-gap movement. The margin that decides the Junyi RQ4
verdict is **+0.0016 AUC**.

Decide what the paper claims *before* you read the new verdicts, and write that decision
down. If RQ4 no longer confirms, that is the finding.

---

## Verification

Everything here was run, not just written.

- All 34 notebooks: AST-parsed, unique `RUN_ID`s, library guard present, `finalize()`
  present, zero stale outputs.
- **Full chain executed end to end on fixture data**: `01` ×3 → `02` ×3 → `03` → `04`.
  The alignment gate fires and passes; the sensitivity diagnostic runs; the corrected RQ4
  summary populates every field the archived code left as `-`.
- `05` ×3 and `06` executed on fixture: after substitution, seven models bit-identical,
  DKVMN changed, legacy column retained.
- **Negative tests**: an archived bucket count altered by +7 → the split gate fires and
  names the failing cell. The alignment gate catches the legacy DKVMN and a deliberately
  leaking model.

Three bugs in this package's own first draft were found by running it, and fixed: `int()`
on a NaN rank, `idxmax` on an all-NaN column, and an alignment test that used membership
instead of the maximum and so flagged `Control`'s legitimate session reset as a lag. The
first sat before `save_tables` and would have destroyed a completed training run.

`verification/` holds the scripts behind the audit: `00_integrity.py` (manifest and
provenance hashing), `01_recompute.py` (independent recomputation from saved predictions),
`02_dependency_probe.py` (the probe that found Finding A), `boot_check.py` (bootstrap CI
reproducibility), `regen_all.py` (this regeneration).
