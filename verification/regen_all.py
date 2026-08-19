#!/usr/bin/env python3
"""Regenerate the full 22-notebook suite against the corrected library.

Method: the archived notebooks are PATCHED at explicit anchors, never rewritten from
memory. Every anchor is asserted to match exactly once, so a patch cannot silently
no-op and no methodology can drift in unnoticed. A per-notebook diff summary is printed.
"""
import difflib, json, sys
from pathlib import Path

SRC = Path("/home/claude/work/P02")
OUT = Path("/home/claude/work/out/P02_corrections")
DATASETS = ["assistments_2021", "ednet_flat", "junyi"]
SEEDS = [0, 1, 2]

LIB_GUARD = '''print("p2_runtime:", R.__file__)
# --- AUDIT GUARD (CORRECTIONS.md) -------------------------------------------------
# Refuse to run against the pre-audit library. Without this a stale pcdt_p2.py on the
# path silently reinstates the DKVMN memory-ordering defect and every table below is
# quietly wrong again.
assert hasattr(P.Config(), "dkvmn_legacy_lag"), (
    "the pcdt_p2.py on this path is the PRE-AUDIT copy: it has no dkvmn_legacy_lag "
    "field, so DKVMN still reads its value memory before writing the current "
    "interaction. Replace pcdt_p2.py and p2_runtime.py with the corrected versions.")
assert hasattr(P, "check_alignment"), "corrected pcdt_p2.py is missing check_alignment"'''


def cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def patch(nb, old, new, label, expect=1):
    """Replace `old` with `new` in exactly `expect` code cells."""
    hits = 0
    for c in cells(nb):
        s = "".join(c["source"])
        if old in s:
            n = s.count(old)
            if n != 1:
                raise AssertionError(f"{label}: anchor appears {n}x in one cell")
            c["source"] = s.replace(old, new).splitlines(keepends=True)
            hits += 1
    if hits != expect:
        raise AssertionError(f"{label}: anchor matched {hits} cell(s), expected {expect}")


def insert_after(nb, anchor, new_source, label):
    """Insert a new code cell immediately after the cell containing `anchor`."""
    idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and anchor in "".join(c["source"]):
            if idx is not None:
                raise AssertionError(f"{label}: anchor cell is not unique")
            idx = i
    if idx is None:
        raise AssertionError(f"{label}: anchor cell not found")
    nb["cells"].insert(idx + 1, dict(cell_type="code", metadata={}, execution_count=None,
                                     outputs=[], source=new_source.splitlines(keepends=True)))


def clean(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    return nb


def save(nb, name):
    (OUT / name).write_text(json.dumps(nb, indent=1))


def summarise(name):
    a = json.loads((SRC / name).read_text())
    b = json.loads((OUT / name).read_text())
    ta = [l for c in cells(a) for l in "".join(c["source"]).splitlines()]
    tb = [l for c in cells(b) for l in "".join(c["source"]).splitlines()]
    d = list(difflib.unified_diff(ta, tb, lineterm=""))
    add = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
    return f"{name:46s} code cells {len(cells(a))}->{len(cells(b))}  +{add} -{rem}"


report = []

# =============================================================================
# 01_RQ1 x9 — library guard + Finding E (the learner sample is seed-specific)
# =============================================================================
for ds in DATASETS:
    for si in SEEDS:
        name = f"01_RQ1__{ds}__seed{si}.ipynb"
        nb = json.loads((SRC / name).read_text())
        patch(nb, 'print("p2_runtime:", R.__file__)', LIB_GUARD, f"{name}/guard")
        patch(nb,
              """                     hint=P.DATASET_SPECS[ds]["has_hint"], attempt=P.DATASET_SPECS[ds]["has_attempt"]))
T0 = pd.DataFrame(rows)
print("Table 0. Dataset characteristics (computed from the data).\\n")
print(T0.to_string(index=False))""",
              """                     hint=P.DATASET_SPECS[ds]["has_hint"], attempt=P.DATASET_SPECS[ds]["has_attempt"],
                     # AUDIT FIX (CORRECTIONS.md, Finding E): these characteristics belong
                     # to THIS SEED's learner sample, not to the dataset. _subsample_learners
                     # draws max_learners BEFORE the min_seq_len filter and every source
                     # dataset holds more than that, so each seed keeps a different
                     # population and the counts below move with it.
                     seed=cfg.seed, sampled_from_max_learners=cfg.max_learners))
T0 = pd.DataFrame(rows)
print("Table 0. Dataset characteristics for THIS SEED's learner sample.\\n")
print(T0.to_string(index=False))
print("\\nThese are not fixed properties of the datasets. Seeds 20260202/20260203/20260204 "
      "draw different samples, so learners, interactions, items, KCs and correct-rate all "
      "differ between them. Report Table 1 of the manuscript for the PRIMARY seed and say "
      "so, or report the three side by side — do not present one seed's counts as the "
      "dataset's. Cross-seed agreement elsewhere is correspondingly STRONGER than a "
      "split-only resampling would give, and is worth stating in those terms.")""",
              f"{name}/findingE")
        save(clean(nb), name)
        report.append(summarise(name))

# =============================================================================
# 02_RQ2RQ3 x9 — library guard + temporal alignment gate before training
# =============================================================================
GATE = '''# ============================================================================
# TEMPORAL ALIGNMENT GATE — runs in seconds, before ~2.5 h of training
# ============================================================================
# The archived runs trained for 22 GPU-hours against a DKVMN that read its value memory
# BEFORE writing interaction t, so its prediction for target j used interactions 0..j-2
# while the other seven models used 0..j-1. Nothing in the loss curve, the capacity match
# or the manifest could reveal that. This gate perturbs one response and checks which
# predictions move, for every model, and refuses to continue if any model either lags
# behind its competitors or can see the answer it is being asked to predict.
ALIGN = P.check_alignment(CLS, CFG)
print(f"\\nDKVMN memory ordering: "
      f"{'LEGACY (reads before writing) — REPRODUCTION ONLY' if CFG.dkvmn_legacy_lag else 'canonical'}")'''

for ds in DATASETS:
    for si in SEEDS:
        name = f"02_RQ2RQ3__{ds}__seed{si}.ipynb"
        nb = json.loads((SRC / name).read_text())
        patch(nb, 'print("p2_runtime:", R.__file__)', LIB_GUARD, f"{name}/guard")
        insert_after(nb, 'print("models:", P.MODEL_ORDER)', GATE, f"{name}/gate")
        patch(nb,
              '''CTX.finalize(''',
              '''CTX.finalize(''',
              f"{name}/finalize-anchor")   # existence check only
        save(clean(nb), name)
        report.append(summarise(name))

# =============================================================================
# 03_RQ4 x3 — guard, adequate_horizons, and the Finding F sensitivity diagnostic
# =============================================================================
SENSITIVITY = '''# ============================================================================
# SENSITIVITY (NOT the pre-registered rule) — CORRECTIONS.md, Finding F
# ============================================================================
# The comparator above is named as argmax(auc_mean_postgap) on the CONFIRMATORY seed, and
# leg 2 ("first at every adequate gap") is then tested on that same seed. Leg 2 is
# therefore substantially favoured by construction: the comparator is defined as the model
# with the highest mean post-gap AUC on exactly the data the leg is evaluated against.
# Pre-registration fixes the rule in advance; it does not make selection and confirmation
# independent.
#
# This cell re-runs the identical rule with the comparator named from the EXPLORATORY
# seeds only. The pre-registered verdict above is unchanged and remains primary. If the
# two disagree, the finding is selection-dependent and must be reported as such.
SENS = {}
if "s3" in T9 and all(k in T9 for k in ("s1", "s2")):
    _mean = (pd.concat([T9["s1"][["model", "auc_mean_postgap"]],
                        T9["s2"][["model", "auc_mean_postgap"]]])
             .groupby("model")["auc_mean_postgap"].mean())
    if _mean.notna().any():
        COMP_EXPL = _mean.idxmax()
        t = T9["s3"]
        wr = int(t.loc[t.model == COMP_EXPL, "rank_within"].iloc[0])
        adq = [h for h in REAL if t[f"auc_{h}"].notna().any()]
        ldrs = {h: t.loc[t[f"auc_{h}"].idxmax(), "model"] for h in adq}
        l1, l2 = wr > 3, (bool(adq) and all(v == COMP_EXPL for v in ldrs.values()))
        SENS = dict(comparator_from_exploratory=COMP_EXPL,
                    comparator_from_confirmatory=VERDICT.get("comparator"),
                    same_comparator=bool(COMP_EXPL == VERDICT.get("comparator")),
                    within_rank_on_s3=wr, leg1=l1, leg2=l2, divergence=bool(l1 and l2))
        print(f"comparator named from exploratory seeds : {COMP_EXPL}")
        print(f"comparator named from confirmatory seed : {VERDICT.get('comparator')}")
        print(f"rule on s3 with the exploratory comparator: "
              f"{'CONFIRMED' if SENS['divergence'] else 'NOT CONFIRMED'} "
              f"(leg1={l1}, leg2={l2}, within rank={wr})")
        if SENS["divergence"] != bool(VERDICT.get("primary_confirmed")):
            print("*** SELECTION-DEPENDENT: the verdict changes with how the comparator "
                  "is named. Report both and do not present the pre-registered verdict "
                  "alone. ***")
        elif not SENS["same_comparator"]:
            print("Verdict agrees, but the two procedures name different comparators — "
                  "state which model the claim is about.")
        else:
            print("Comparator and verdict both agree: the finding is not an artefact of "
                  "naming the comparator on the seed it is tested on.")
    VERDICT["selection_sensitivity"] = SENS
else:
    print("sensitivity not evaluable (needs all three seeds)")'''

for ds in DATASETS:
    name = f"03_RQ4__{ds}.ipynb"
    nb = json.loads((SRC / name).read_text())
    patch(nb, 'print("p2_runtime:", R.__file__)', LIB_GUARD, f"{name}/guard")
    patch(nb,
          '''    s3row = PRIMARY[PRIMARY.seed == "s3"].iloc[0]
    VERDICT["primary_confirmed"] = bool(s3row.divergence)''',
          '''    s3row = PRIMARY[PRIMARY.seed == "s3"].iloc[0]
    VERDICT["primary_confirmed"] = bool(s3row.divergence)
    # AUDIT FIX (CORRECTIONS.md, Finding B): the horizons the rule was actually tested on
    # were never recorded, so the cross-dataset summary had no way to state them.
    VERDICT["adequate_horizons"] = adequate3''',
          f"{name}/horizons")
    insert_after(nb, 'VERDICT["primary_confirmed"] = bool(s3row.divergence)',
                 SENSITIVITY, f"{name}/sensitivity")
    patch(nb,
          '''tabs = {"table10_rankings_by_seed": RANKS}''',
          '''tabs = {"table10_rankings_by_seed": RANKS}
if SENS:
    tabs["table13_selection_sensitivity"] = pd.DataFrame([SENS])''',
          f"{name}/table13")
    save(clean(nb), name)
    report.append(summarise(name))

# =============================================================================
# 04_SUMMARY x1 — already corrected in place; re-apply from the archived original
# =============================================================================
name = "04_SUMMARY__all.ipynb"
nb = json.loads((SRC / name).read_text())
patch(nb, 'print("p2_runtime:", R.__file__)', LIB_GUARD, f"{name}/guard")
patch(nb,
      '''    v = json.load(open(f)) if f else man.get("verdict", {}) or {}
    pc = v.get("primary_confirmed")
    rows.append(dict(dataset=ds, status=man.get("status", "run"),
                     comparator=v.get("comparator", "-"),
                     confirmed=("-" if pc is None else
                                ("CONFIRMED" if pc else "NOT CONFIRMED")),
                     horizons_used=", ".join(v.get("adequate_horizons", [])) or "-",
                     failing_leg=v.get("failing_leg", "-")))''',
      '''    # AUDIT FIX (CORRECTIONS.md, Finding B): verdict_rq4.json stores the verdict one
    # level down, under "verdict". Reading the top level returned None for every field, so
    # this table printed "-" for all three datasets and the dataset-dependence warning
    # below could never fire — on results that are in fact 2-of-3 confirmed.
    rec = json.load(open(f)) if f else {}
    v = rec.get("verdict", rec) or man.get("verdict", {}) or {}
    pc = v.get("primary_confirmed")
    # AUDIT FIX: "adequate_horizons" was never written by the archived notebook 03, so this
    # field stayed blank even with the nesting corrected. Fall back to the confirmatory
    # seed's table9 — the file the rule itself was evaluated on.
    horizons = v.get("adequate_horizons") or []
    if not horizons:
        b23 = CTX.sibling(RQ2RQ3_ID(ds, max(SEEDS)))
        if b23:
            t9 = os.path.join(b23, "tables", "table9_postgap_auc_by_model.csv")
            if os.path.exists(t9):
                _t = pd.read_csv(t9)
                horizons = [h for h in P.REAL_HORIZONS
                            if f"auc_{h}" in _t.columns and _t[f"auc_{h}"].notna().any()]
    _sens = (v.get("selection_sensitivity") or {})
    rows.append(dict(dataset=ds, status=man.get("status", "run"),
                     comparator=v.get("comparator", "-"),
                     confirmed=("-" if pc is None else
                                ("CONFIRMED" if pc else "NOT CONFIRMED")),
                     secondary=("-" if v.get("secondary_confirmed") is None
                                else ("passed" if v.get("secondary_confirmed")
                                      else "not passed")),
                     selection_sensitive=("-" if not _sens else
                                          str(bool(_sens.get("divergence") != pc))),
                     horizons_used=", ".join(horizons) or "-",
                     failing_leg=v.get("failing_leg", "-")))''',
      f"{name}/findingB")
patch(nb,
      '''            sig = ("n/a (no CI)" if (np.isnan(lo) or np.isnan(hi))
                   else "positive" if lo > 0 else "negative" if hi < 0 else "null")''',
      '''            # AUDIT FIX (CORRECTIONS.md, Finding C): the label was the literal string
            # "null", which pandas.read_csv converts back to NaN by default. A genuine
            # "CI includes zero" result was therefore indistinguishable on re-read from a
            # bootstrap that could not be formed — the exact conflation the comment above
            # exists to prevent, in the opposite direction.
            sig = ("n/a (no CI)" if (np.isnan(lo) or np.isnan(hi))
                   else "positive" if lo > 0 else "negative" if hi < 0
                   else "inconclusive (CI spans zero)")''',
      f"{name}/findingC")
save(clean(nb), name)
report.append(summarise(name))

print("\n".join(report))
print(f"\nregenerated {len(report)} notebooks from the archived originals by anchored patching")
