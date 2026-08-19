"""
pcdt_p2.py — the single canonical implementation for the Paper 2 experiments.

Everything the three notebooks need lives here exactly once: dataset loaders, the
construct-validity audit, the split builder, the eight models, the training harness,
the metrics, and the statistical tests.

WHY THIS FILE EXISTS
--------------------
The original P02 package carried the same code many times over: `pcdt_datasets.py`
was embedded verbatim inside 12 notebooks, the five model classes appeared as *string
literals* inside a `_CELLS` list in 9 notebooks, and three pipeline notebooks were
byte-identical to each other apart from one dataset name. Duplicated code drifts, and
it did: a fix applied to one copy (the §15 educational-utility comparator) never
reached another (see AUDIT_REPORT.md, Finding 3).

PROVENANCE
----------
Semantics are carried over from `P2_Revision_Pipeline_v3_*.ipynb`, which produced the
V3 / seed-2 / seed-3 results the manuscript reports. Where the original had two
divergent implementations of the same operation, the one that produced the reported
numbers was kept and the divergence is recorded in AUDIT_REPORT.md. No model, metric,
threshold, seed or split rule has been altered.
"""

from __future__ import annotations

import ast
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 1. PATHS  —  PRESERVED EXACTLY AS FOUND IN THE ORIGINAL NOTEBOOKS
# ============================================================================
# Taken verbatim from P2_Revision_Pipeline_v3_*.ipynb (the notebooks that produced
# the reported V3 results). Do not "normalise" these.
#
# NOTE ON A DISCREPANCY THE AUDIT FOUND (reported, not silently resolved):
#   the v3 pipeline and the three P2_RQ*_junyi notebooks use  .../dataset/junyi/...
#   the eight other P2_RQ* notebooks carry  .../dataset/Junyi/...  (capital J)
#   in a DATA_PATHS dict entry they never read. On a case-sensitive filesystem only
#   one can exist. The lowercase form is used here because it is the form that was
#   actually executed. See AUDIT_REPORT.md, Finding 6.

DATA_ROOT = "/home/hamed/Downloads/Research/dataset"

PATHS = {
    "assistments_2021": dict(plogs=f"{DATA_ROOT}/feb_to_apr/plogs.csv",
                             details=f"{DATA_ROOT}/feb_to_apr/pdets.csv"),
    "ednet_flat":       dict(flat=f"{DATA_ROOT}/EdNet/ednet_kt1_flat.csv"),
    "junyi":            dict(log=f"{DATA_ROOT}/junyi/Log_Problem.csv",
                             content=f"{DATA_ROOT}/junyi/Info_Content.csv"),
}

# Output root. Deliberately NOT the V3 cache: the existing
# /home/hamed/Downloads/Research/P02/PCDT_P2_cache_rev_V3 holds the verified results the
# manuscript cites, and a clean re-run must not overwrite them.
OUTPUT_ROOT = "/home/hamed/Downloads/Research/P02/PCDT_P2_clean"

STD = ["learner_id", "item_id", "skill_id", "start_time", "correct",
       "response_time", "hint_frac", "attempt_count"]
BEHAV = ["response_time", "hint_frac", "attempt_count"]

DATASET_SPECS = {
    "assistments_2021": dict(has_hint=True,  has_attempt=True,  has_latency=True),
    "junyi":            dict(has_hint=True,  has_attempt=True,  has_latency=True),
    "ednet_flat":       dict(has_hint=False, has_attempt=False, has_latency=True),
}

DATASETS = ["assistments_2021", "ednet_flat", "junyi"]


# ============================================================================
# 2. CONFIGURATION  —  every pre-registered constant in one place
# ============================================================================
@dataclass(frozen=True)
class Config:
    dataset: str = "junyi"

    # ---- paths ----
    root: str = DATA_ROOT
    out_root: str = OUTPUT_ROOT

    # ---- population ----
    max_learners: int = 20_000
    min_seq_len: int = 10
    max_seq_len: int = 200

    # ---- split (revision design: 40% test, stratified on largest gap) ----
    train_frac: float = 0.50
    val_frac: float = 0.10
    test_frac: float = 0.40
    split_mode: str = "learner"

    # ---- capacity + optimisation ----
    param_budget: int = 350_000
    param_tolerance: float = 0.05
    batch_size: int = 64
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 5
    grad_clip: float = 5.0
    search_lrs: Sequence[float] = (5e-4, 1e-3, 2e-3)

    # ---- persistence ----
    use_persistence: bool = True
    # AUDIT FIX (see CORRECTIONS.md, Finding A). False = canonical DKVMN: the value
    # memory is written with interaction t BEFORE the read that predicts t+1, so DKVMN
    # sees the same history as every other model. True reproduces the archived runs, in
    # which the read preceded the write and DKVMN was denied the most recent interaction.
    dkvmn_legacy_lag: bool = False
    tau0_seconds: float = 3600.0
    session_gap_s: float = 20 * 60

    # ---- pre-registered thresholds (unchanged since the original pre-registration) ----
    reliability_floor: float = 0.60
    convergent_floor: float = 0.30
    simulator_recovery_floor: float = 0.50
    min_meaningful_auc_delta: float = 0.010
    warm_min_history: int = 5

    # ---- adequacy floors for a horizon to be reported on REAL data ----
    min_learners_bucket: int = 300
    min_points_bucket: int = 2_000

    # ---- educational utility ----
    utility_thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7)

    # ---- evaluation ----
    n_bootstrap: int = 2_000
    ece_bins: int = 15
    seeds: Sequence[int] = (20260202, 20260203, 20260204)
    seed_index: int = 0

    # ---- fixture guard ----
    # A missing dataset RAISES. The original 12-notebook suite silently substituted
    # synthetic data and produced a complete set of tables from it. Set this True only
    # to smoke-test the pipeline; every artefact is then written under a FIXTURE_ prefix.
    allow_fixture: bool = False

    @property
    def seed(self) -> int:
        return self.seeds[self.seed_index]

    @property
    def seed_tag(self) -> str:
        return "" if self.seed_index == 0 else f"_seed{self.seed_index + 1}"

    @property
    def paths(self) -> dict:
        p = {
            "assistments_2021": dict(plogs=f"{self.root}/feb_to_apr/plogs.csv",
                                     details=f"{self.root}/feb_to_apr/pdets.csv"),
            "ednet_flat":       dict(flat=f"{self.root}/EdNet/ednet_kt1_flat.csv"),
            "junyi":            dict(log=f"{self.root}/junyi/Log_Problem.csv",
                                     content=f"{self.root}/junyi/Info_Content.csv"),
        }
        return p[self.dataset]


def make_dirs(cfg: Config, tag: str) -> dict:
    """Create and return the output folders for one run."""
    prefix = "FIXTURE_" if cfg.allow_fixture and not dataset_available(cfg.dataset, cfg.paths) else ""
    out = os.path.join(cfg.out_root, "results", f"{prefix}{tag}")
    d = dict(out=out, fig=os.path.join(out, "figures"), tab=os.path.join(out, "tables"))
    for v in d.values():
        os.makedirs(v, exist_ok=True)
    return d


def resolve_run_dir(out_root: str, folder: str) -> Optional[str]:
    """Locate a run folder written by Notebook 2.

    Checks the plain name first, then the FIXTURE_ variant, so a smoke-test run is found
    without special-casing and a real run is never confused with a fixture one.
    Returns None if neither exists — the caller must treat that as an error, never as an
    empty result.
    """
    for cand in (os.path.join(out_root, "results", folder),
                 os.path.join(out_root, "results", "FIXTURE_" + folder)):
        if os.path.isdir(cand):
            return cand
    return None


def seed_everything(s: int):
    import torch
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)


# ============================================================================
# 3. DATASET LOADERS
# ============================================================================
def _finalize(df, verbose, name):
    df = df.dropna(subset=["learner_id", "item_id", "correct", "start_time"]).copy()
    df["learner_id"] = df["learner_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(str).fillna("UNK").replace({"nan": "UNK", "": "UNK"})
    df["correct"] = df["correct"].astype(np.float32)
    for c in ("response_time", "hint_frac", "attempt_count"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df[STD]
    if verbose:
        span = (df.groupby("learner_id")["start_time"]
                  .agg(lambda s: (s.max() - s.min()).total_seconds()) / 86400.0)
        print(f"[{name}] {df['learner_id'].nunique():,} learners | {len(df):,} interactions | "
              f"{df['item_id'].nunique():,} items | {df['skill_id'].nunique():,} skills | "
              f"correct {df['correct'].mean():.3f} | median span {span.median():.1f} d")
        have = [c for c in ("response_time", "hint_frac", "attempt_count") if df[c].notna().mean() > 0.01]
        print(f"        behavioural signals present: {have or ['(none beyond correctness)']}")
    return df


def _subsample_learners(df, id_col, max_learners, rng):
    if max_learners is None:
        return df
    ids = df[id_col].unique()
    if len(ids) <= max_learners:
        return df
    keep = set(rng.choice(ids, size=max_learners, replace=False))
    return df[df[id_col].isin(keep)]


def load_ednet_flat(paths, max_learners=None, min_seq_len=1, rng=None, verbose=True):
    """ednet_kt1_flat.csv: user_id, item_id, skill_id, correct, timestamp_ms, latency_ms."""
    rng = rng or np.random.default_rng(0)
    df = pd.read_csv(paths["flat"])
    need = {"user_id", "item_id", "skill_id", "correct", "timestamp_ms", "latency_ms"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"ednet_flat: missing columns {missing}; present {sorted(df.columns)}")
    df = _subsample_learners(df, "user_id", max_learners, rng)
    if min_seq_len > 1:
        cnt = df.groupby("user_id")["item_id"].transform("size")
        df = df[cnt >= min_seq_len]
    out = pd.DataFrame({
        "learner_id":    df["user_id"],
        "item_id":       df["item_id"],
        "skill_id":      df["skill_id"],
        "start_time":    pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True),
        "correct":       df["correct"],
        "response_time": pd.to_numeric(df["latency_ms"], errors="coerce") / 1000.0,
        "hint_frac":     np.nan,     # EdNet-KT1 has no hint signal
        "attempt_count": np.nan,     # EdNet-KT1 has no attempt signal
    })
    return _finalize(out, verbose, "ednet_flat")


def load_junyi(paths, max_learners=None, min_seq_len=1, rng=None, verbose=True):
    """Junyi Academy. skill_id defaults to `ucid` (~1,330 distinct).

    The KC default matters: an earlier version mapped ucid -> level1_id, and because Junyi
    is mathematics-only, level1_id is essentially constant. That collapsed the whole dataset
    to ONE knowledge component, which made C_t's convergent validity 1.000 by construction
    and degenerated the per-skill Bayesian baseline. A coarser level is opt-in and falls back
    to ucid if it turns out degenerate.
    """
    rng = rng or np.random.default_rng(0)
    usecols = ["timestamp_TW", "uuid", "ucid", "upid", "is_correct",
               "total_sec_taken", "total_attempt_cnt", "used_hint_cnt"]
    log = pd.read_csv(paths["log"], usecols=lambda c: c in usecols)
    miss = set(usecols) - set(log.columns)
    if {"uuid", "upid", "is_correct"} & miss:
        raise KeyError(f"junyi log: missing essential columns {miss}; present {sorted(log.columns)}")
    log = _subsample_learners(log, "uuid", max_learners, rng)
    if min_seq_len > 1:
        cnt = log.groupby("uuid")["upid"].transform("size")
        log = log[cnt >= min_seq_len]

    skill_level = str(paths.get("skill_level", "ucid"))
    skill = log["ucid"].astype(str)
    if skill_level != "ucid" and paths.get("content") and os.path.exists(paths["content"]):
        cont = pd.read_csv(paths["content"], usecols=lambda c: c in ("ucid", skill_level))
        if skill_level in cont.columns:
            cont = cont.dropna(subset=["ucid"]).drop_duplicates(subset=["ucid"], keep="first")
            m = pd.Series(dict(zip(cont["ucid"].astype(str), cont[skill_level].astype(str))))
            mapped = log["ucid"].astype(str).map(m)
            if int(mapped.nunique()) < 5:
                print(f"        [junyi] skill_level='{skill_level}' yields <5 KCs; falling back to 'ucid'")
            else:
                skill = mapped.fillna(log["ucid"].astype(str))

    att = pd.to_numeric(log.get("total_attempt_cnt"), errors="coerce")
    hints = pd.to_numeric(log.get("used_hint_cnt"), errors="coerce")
    hint_frac = (hints / att.replace(0, np.nan)).clip(0, 1)

    out = pd.DataFrame({
        "learner_id":    log["uuid"],
        "item_id":       log["upid"],
        "skill_id":      skill.values,
        "start_time":    pd.to_datetime(log["timestamp_TW"], errors="coerce", utc=True),
        "correct":       log["is_correct"].astype(float),
        "response_time": pd.to_numeric(log.get("total_sec_taken"), errors="coerce"),
        "hint_frac":     hint_frac.values if hasattr(hint_frac, "values") else hint_frac,
        "attempt_count": att.values if hasattr(att, "values") else att,
    })
    return _finalize(out, verbose, "junyi")


def _parse_skills(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return [] if (v is None or (isinstance(v, float) and np.isnan(v))) else [v]


def load_assistments_2021(paths, max_learners=None, min_seq_len=1, rng=None, verbose=True):
    """ASSISTments 2020-2021 problem logs joined to problem details for skill_id."""
    rng = rng or np.random.default_rng(0)
    plog_cols = ["student_id", "problem_id", "start_time", "time_on_task",
                 "fraction_of_hints_used", "attempt_count", "correct"]
    plogs = pd.read_csv(paths["plogs"], usecols=lambda c: c in plog_cols)
    ess = {"student_id", "problem_id", "correct", "start_time"} - set(plogs.columns)
    if ess:
        raise KeyError(f"assistments plogs: missing {ess}; present {sorted(plogs.columns)}")
    plogs = _subsample_learners(plogs, "student_id", max_learners, rng)
    if min_seq_len > 1:
        cnt = plogs.groupby("student_id")["problem_id"].transform("size")
        plogs = plogs[cnt >= min_seq_len]

    corr = plogs["correct"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0,
                                 "1": 1.0, "0": 0.0, 1: 1.0, 0: 0.0})
    corr = pd.to_numeric(corr.fillna(plogs["correct"]), errors="coerce")

    skill = pd.Series("UNK", index=plogs.index)
    det_path = paths.get("details") or paths.get("pdets")
    if det_path and os.path.exists(det_path):
        det = pd.read_csv(det_path, usecols=lambda c: c in ("problem_id", "skills"))
        if "skills" in det.columns:
            # drop_duplicates matters: duplicate problem_id keys otherwise blow up the join
            det = det.dropna(subset=["problem_id"]).drop_duplicates(subset=["problem_id"], keep="first")
            det["skills"] = det["skills"].map(_parse_skills)
            det["skill_id"] = det["skills"].map(lambda s: str(s[0]) if isinstance(s, list) and s else "UNK")
            m = dict(zip(det["problem_id"], det["skill_id"]))
            skill = plogs["problem_id"].map(pd.Series(m)).fillna("UNK")

    out = pd.DataFrame({
        "learner_id":    plogs["student_id"],
        "item_id":       plogs["problem_id"],
        "skill_id":      skill.values if hasattr(skill, "values") else skill,
        "start_time":    pd.to_datetime(plogs["start_time"], errors="coerce", utc=True),
        "correct":       corr.values,
        "response_time": pd.to_numeric(plogs.get("time_on_task"), errors="coerce"),
        "hint_frac":     pd.to_numeric(plogs.get("fraction_of_hints_used"), errors="coerce"),
        "attempt_count": pd.to_numeric(plogs.get("attempt_count"), errors="coerce"),
    })
    return _finalize(out, verbose, "assistments_2021")


_LOADERS = dict(ednet_flat=load_ednet_flat, junyi=load_junyi,
                assistments_2021=load_assistments_2021)


def dataset_available(name, paths) -> bool:
    need = {"ednet_flat": ["flat"], "junyi": ["log"], "assistments_2021": ["plogs"]}[name]
    return all(paths.get(k) and os.path.exists(paths[k]) for k in need)


def load_dataset(cfg: Config, verbose=True) -> pd.DataFrame:
    """Load cfg.dataset. RAISES if the files are absent unless cfg.allow_fixture."""
    name, paths = cfg.dataset, cfg.paths
    if dataset_available(name, paths):
        return _LOADERS[name](paths, max_learners=cfg.max_learners,
                              min_seq_len=cfg.min_seq_len,
                              rng=np.random.default_rng(cfg.seed), verbose=verbose)
    msg = (f"dataset '{name}' not found at {paths}. "
           "Fix Config.root rather than proceeding — a silent fixture substitution is how "
           "synthetic numbers get into a manuscript.")
    if not cfg.allow_fixture:
        raise FileNotFoundError(msg)
    print("!" * 78)
    print(f"! FIXTURE MODE. {msg}")
    print("! Every artefact from this run is prefixed FIXTURE_ and is NOT a result.")
    print("!" * 78)
    return _fixture(cfg)


def _fixture(cfg: Config) -> pd.DataFrame:
    """Synthetic data with realistic gap structure. Smoke-testing only."""
    rng = np.random.default_rng(cfg.seed)
    spec = DATASET_SPECS[cfg.dataset]
    skills = [f"S{i}" for i in range(40)]
    diff = {s: rng.normal(0, .8) for s in skills}
    rows, t0 = [], pd.Timestamp("2024-01-01", tz="UTC")
    for u in range(min(cfg.max_learners, 400)):
        ab = rng.normal(0, 1)
        mast = {s: rng.normal(-.3, .3) for s in skills}
        last = {s: None for s in skills}
        t = t0 + pd.Timedelta(days=int(rng.integers(0, 30)))
        for _ in range(int(rng.integers(4, 10))):
            uu = rng.random()
            gap = (rng.uniform(0, 1) if uu < .4 else rng.uniform(1, 7) if uu < .7
                   else rng.uniform(7, 28) if uu < .9 else rng.uniform(28, 70))
            t += pd.Timedelta(days=float(gap))
            for _ in range(int(rng.integers(5, 20))):
                s = skills[int(rng.integers(len(skills)))]
                if last[s] is not None:
                    dd = (t - last[s]).total_seconds() / 86400
                    mast[s] *= (1 + dd) ** (-0.5)
                p = 1 / (1 + np.exp(-(ab + mast[s] - diff[s])))
                c = int(rng.random() < np.clip(p, .02, .98))
                mast[s] += .2 if c else .06
                last[s] = t
                rows.append(dict(
                    learner_id=f"u{u}", item_id=f"q{rng.integers(0, 500)}", skill_id=s,
                    start_time=t, correct=float(c),
                    response_time=float(np.clip(rng.lognormal(3, .4), 2, 900)),
                    hint_frac=(float(np.clip(rng.beta(2, 5), 0, 1)) if spec["has_hint"] else np.nan),
                    attempt_count=(float(1 + rng.poisson(.5)) if spec["has_attempt"] else np.nan)))
                t += pd.Timedelta(seconds=int(rng.integers(20, 240)))
    return _finalize(pd.DataFrame(rows), True, f"FIXTURE::{cfg.dataset}")


# ============================================================================
# 4. VALIDATION GATES  —  each one caught a real defect at some point
# ============================================================================
def validate(df: pd.DataFrame) -> dict:
    assert set(STD) <= set(df.columns), f"missing columns: {set(STD) - set(df.columns)}"
    assert df["correct"].isin([0.0, 1.0]).all(), "correct must be binary"
    assert df["start_time"].notna().all(), "unparsed timestamps"
    n_kc = df["skill_id"].nunique()
    assert n_kc >= 5, (f"DEGENERATE KC SPACE: {n_kc} distinct skills. Per-KC mastery collapses to a "
                       "per-learner average and C_t validity becomes ~1.0 by construction.")
    span = (df.groupby("learner_id")["start_time"]
              .agg(lambda s: (s.max() - s.min()).total_seconds()) / 86400)
    stats = dict(n_learners=int(df["learner_id"].nunique()), n_interactions=int(len(df)),
                 n_kc=int(n_kc), n_items=int(df["item_id"].nunique()),
                 correct_rate=float(df["correct"].mean()),
                 median_span_days=float(span.median()), mean_span_days=float(span.mean()))
    print(f"  OK  {stats['n_learners']:,} learners | {stats['n_interactions']:,} interactions | "
          f"{n_kc:,} KCs")
    print(f"      correct {stats['correct_rate']:.3f} | median span {stats['median_span_days']:.1f} d "
          f"| mean span {stats['mean_span_days']:.1f} d")
    if stats["median_span_days"] < 7:
        print("      NOTE short median span; quote the mean and the gap distribution too — a short "
              "median with a long mean means a bimodal population, not an unusable dataset.")
    return stats


def temporal_profile(df: pd.DataFrame):
    """Span and largest-gap distribution. Answers 'how many learners reach each horizon'."""
    d = df.sort_values(["learner_id", "start_time"])
    g = d.groupby("learner_id")["start_time"]
    span_d = (g.agg(lambda s: (s.max() - s.min()).total_seconds()) / 86400).rename("span_days")
    gaps = g.diff().dt.total_seconds()
    maxgap_d = (gaps.groupby(d["learner_id"]).max() / 86400)
    rows = [dict(statistic=k, value=float(v))
            for k, v in span_d.describe(percentiles=[.25, .5, .75, .9, .95]).items()]
    for wk in (1, 2, 4, 8):
        rows.append(dict(statistic=f"% learners with span >= {wk} week(s)",
                         value=float((span_d >= 7 * wk).mean() * 100)))
        rows.append(dict(statistic=f"% learners with a GAP >= {wk} week(s)",
                         value=float((maxgap_d >= 7 * wk).mean() * 100)))
    return pd.DataFrame(rows), span_d, maxgap_d


# ============================================================================
# 5. RQ1  —  CONSTRUCT-VALIDITY AUDIT
# ============================================================================
CANDIDATE_DIMS = ["I_t (identity)", "C_t (cognition)", "B_t (behaviour)",
                  "M_t (metacognition)", "G_t (goals)", "H_t (historical memory)"]
# U_t is NOT audited as a state component: estimation uncertainty is a property of the
# model's belief about the learner, not a coordinate of the learner.


def build_proxies(df: pd.DataFrame, spec: dict):
    """Per-interaction proxy columns + a per-learner split-half ability frame.

    Every proxy is computed from history STRICTLY BEFORE the current response, so no
    proxy can see the outcome it is later correlated against.
    """
    df = df.sort_values(["learner_id", "start_time"]).copy()
    g = df.groupby("learner_id")

    # --- C_t: running per-(learner, KC) correctness before the current item ---
    df["kc_key"] = df["learner_id"].astype(str) + "|" + df["skill_id"].astype(str)
    kc = df.groupby("kc_key")
    df["kc_prior_n"] = kc.cumcount()
    df["kc_prior_sum"] = kc["correct"].cumsum() - df["correct"]
    df["C_proxy"] = (df["kc_prior_sum"] / df["kc_prior_n"].replace(0, np.nan)).fillna(0.5)

    # --- independent mastery signal used as the convergent-validity target ---
    df["glob_prior_n"] = g.cumcount()
    df["glob_prior_sum"] = g["correct"].cumsum() - df["correct"]
    df["mastery_signal"] = (df["glob_prior_sum"] / df["glob_prior_n"].replace(0, np.nan)).fillna(0.5)

    # --- B_t: struggle signal from whichever of hint / attempt / latency the dataset has ---
    rt = df["response_time"].copy()
    df["rt_log"] = np.log1p(rt.clip(lower=0))
    mu = g["rt_log"].transform("mean")
    sd = g["rt_log"].transform("std").replace(0, 1.0).fillna(1.0)
    df["rt_z"] = ((df["rt_log"] - mu) / sd).fillna(0.0)
    df["rt_slow"] = (df["rt_z"] > 1.0).astype(float)
    df["rt_kc_std"] = kc["rt_log"].transform("std").fillna(0.0)
    parts = [df["rt_z"]]
    if spec.get("has_attempt") and df["attempt_count"].notna().mean() > 0.01:
        a = pd.to_numeric(df["attempt_count"], errors="coerce")
        df["att_z"] = ((a - a.mean()) / (a.std() or 1.0)).fillna(0.0)
        parts.append(df["att_z"])
    if spec.get("has_hint") and df["hint_frac"].notna().mean() > 0.01:
        h = pd.to_numeric(df["hint_frac"], errors="coerce")
        df["hint_z"] = ((h - h.mean()) / (h.std() or 1.0)).fillna(0.0)
        parts.append(df["hint_z"])
    df["B_proxy"] = sum(parts) / len(parts)
    has_struggle = len(parts) > 1

    # --- M_t: does a slow response precede a correction? ---
    df["prev_correct"] = g["correct"].shift(1)
    df["is_correction"] = ((df["prev_correct"] == 0) & (df["correct"] == 1)).astype(float)
    df["M_proxy"] = g["rt_z"].shift(1).fillna(0.0)

    # --- H_t: recency-weighted prior correctness (sufficient-statistic candidate) ---
    def recency(s, half=10):
        w = 0.5 ** (np.arange(len(s))[::-1] / half)
        num, den = np.cumsum(s.values * w), np.cumsum(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.5)
        return pd.Series(np.concatenate([[0.5], val[:-1]]), index=s.index)
    df["H_proxy"] = g["correct"].apply(recency).values

    # --- G_t: curricular-progress fraction. No dataset here carries a goal annotation. ---
    df["G_proxy"] = g.cumcount() / g["learner_id"].transform("count")

    # --- I_t: split-half baseline ability ---
    rows = []
    for lid, gg in df.groupby("learner_id"):
        h = len(gg) // 2
        if h >= 3:
            rows.append((lid, gg["correct"].iloc[:h].mean(), gg["correct"].iloc[h:].mean()))
    trait = pd.DataFrame(rows, columns=["learner_id", "ability_h1", "ability_h2"])
    return df, trait, has_struggle


def cronbach_alpha(items):
    X = np.asarray(items, float)
    X = X[~np.isnan(X).any(axis=1)]
    k = X.shape[1]
    if k < 2 or len(X) < 10:
        return np.nan
    vt = X.sum(axis=1).var(ddof=1)
    if vt == 0:
        return np.nan
    return float(k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / vt))


def split_half_reliability(df, col):
    """Spearman-Brown corrected correlation of a proxy's two per-learner history halves."""
    rows = []
    for _, gg in df.groupby("learner_id"):
        h = len(gg) // 2
        if h >= 3 and gg[col].notna().sum() >= 6:
            rows.append((gg[col].iloc[:h].mean(), gg[col].iloc[h:].mean()))
    if len(rows) < 10:
        return np.nan
    a = np.array(rows)
    if a[:, 0].std() == 0 or a[:, 1].std() == 0:
        return np.nan
    r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    # A negative half-half correlation means the signal is not internally consistent.
    # Spearman-Brown is undefined there, so it is floored at 0 and reported as such.
    if not np.isfinite(r) or r <= 0:
        return 0.0
    return float(min(max(2 * r / (1 + r), 0.0), 1.0))


def convergent(df, col, target="mastery_signal", warm=True, min_hist=5):
    """Correlate a proxy with the target on the WARM portion.

    A running estimate over 0-2 responses is a cold-start artefact rather than a
    measurement; including it dilutes every convergent-validity estimate toward zero.
    """
    sub = df
    if warm and "glob_prior_n" in sub.columns:
        sub = sub[sub["glob_prior_n"] >= min_hist]
    x, y = sub[col].values, sub[target].values
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 50 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def incremental_validity_H(prox):
    """Partial correlation of H_proxy with correctness, controlling for C_proxy.

    If H adds nothing over current-KC mastery it is redundant as a separate coordinate.
    """
    d = prox[["H_proxy", "C_proxy", "correct"]].dropna()
    if len(d) < 200:
        return np.nan
    H, C, y = d["H_proxy"].values, d["C_proxy"].values, d["correct"].values
    if H.std() == 0 or C.std() == 0 or y.std() == 0:
        return np.nan

    def resid(a, b):
        B = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef
    rH, ry = resid(H, C), resid(y, C)
    if rH.std() == 0 or ry.std() == 0:
        return 0.0
    return abs(float(np.corrcoef(rH, ry)[0, 1]))


def audit_dimensions(prox, trait, has_struggle, cfg: Config, dataset: str) -> pd.DataFrame:
    """Run the pre-registered audit on all six candidate dimensions."""
    def trait_r():
        if len(trait) >= 10 and trait["ability_h1"].std() > 0:
            return abs(float(np.corrcoef(trait["ability_h1"], trait["ability_h2"])[0, 1]))
        return np.nan

    out = []
    for dim in CANDIDATE_DIMS:
        r = dict(dimension=dim, proxy="", alpha=np.nan, split_half=np.nan,
                 convergent=np.nan, sim_recovery=np.nan, note="")
        if dim == "C_t (cognition)":
            r["proxy"] = "running per-KC correctness (mastery estimate)"
            r["split_half"] = split_half_reliability(prox, "C_proxy")
            r["convergent"] = convergent(prox, "C_proxy", "mastery_signal",
                                         min_hist=cfg.warm_min_history)
        elif dim == "B_t (behaviour)":
            r["proxy"] = ("hint+attempt+latency struggle signal" if has_struggle
                          else "latency-only (no hint/attempt in this dataset)")
            cols = [c for c in ["rt_z", "att_z", "hint_z", "rt_slow", "rt_kc_std"] if c in prox.columns]
            r["alpha"] = cronbach_alpha(prox[cols].values) if len(cols) >= 2 else np.nan
            r["split_half"] = split_half_reliability(prox, "B_proxy")
            r["convergent"] = convergent(prox, "B_proxy", "mastery_signal",
                                         min_hist=cfg.warm_min_history)
        elif dim == "M_t (metacognition)":
            r["proxy"] = "latency-accuracy coupling (prev_rt_z vs correction)"
            r["split_half"] = split_half_reliability(prox, "M_proxy")
            r["convergent"] = convergent(prox, "M_proxy", "is_correction", warm=False)
            r["sim_recovery"] = abs(r["convergent"]) if np.isfinite(r["convergent"]) else np.nan
            if r["split_half"] == 0.0:
                r["note"] = ("split-half floored at 0: the two history halves correlate NEGATIVELY, "
                             "so this is 'not internally consistent', not 'reliability measured as 0'")
        elif dim == "G_t (goals)":
            r["proxy"] = f"curricular-progress fraction (no goal annotation in {dataset})"
            r["split_half"] = split_half_reliability(prox, "G_proxy")
            r["convergent"] = convergent(prox, "G_proxy", "mastery_signal",
                                         min_hist=cfg.warm_min_history)
            if np.isfinite(r["split_half"]) and r["split_half"] > 0.999:
                r["note"] = ("reliability is 1.000 BY CONSTRUCTION: the proxy is a deterministic "
                             "0->1 ramp in sequence position, so its halves correlate perfectly. "
                             "Read this as 'not a measurement', not as 'perfectly reliable'")
        elif dim == "H_t (historical memory)":
            r["proxy"] = "recency-weighted prior correctness; retained only if it adds OVER C_t"
            r["split_half"] = split_half_reliability(prox, "H_proxy")
            r["convergent"] = convergent(prox, "H_proxy", "mastery_signal",
                                         min_hist=cfg.warm_min_history)
            r["sim_recovery"] = incremental_validity_H(prox)
            r["note"] = "sim_recovery column holds INCREMENTAL validity over C_t, not simulator recovery"
        elif dim == "I_t (identity)":
            r["proxy"] = "split-half baseline-ability trait"
            r["split_half"] = split_half_reliability(prox, "C_proxy")
            r["convergent"] = trait_r()
            r["sim_recovery"] = trait_r()
            r["note"] = ("CIRCULARITY WARNING: reliability is borrowed from C_proxy, and convergent "
                         "validity is the SAME statistic as sim_recovery (corr(ability_h1, ability_h2)) "
                         "-- i.e. I_t is validated against its own stability. Treat the 'retain' verdict "
                         "as computational persistence of ability, NOT as an independently validated "
                         "construct. See AUDIT_REPORT.md Finding 2")

        # ---- pre-registered verdict ----
        reliab = r["alpha"] if np.isfinite(r["alpha"]) else r["split_half"]
        conv, sim = r["convergent"], r["sim_recovery"]
        if all(not np.isfinite(v) for v in (reliab, conv, sim)):
            r["verdict"] = "reject (no signal)"
        elif np.isfinite(reliab) and reliab >= cfg.reliability_floor and \
                np.isfinite(conv) and abs(conv) >= cfg.convergent_floor:
            r["verdict"] = "retain"
        elif np.isfinite(conv) and abs(conv) >= cfg.convergent_floor:
            r["verdict"] = "proxy-only"
        elif np.isfinite(sim) and sim >= cfg.simulator_recovery_floor:
            r["verdict"] = "retain (simulator only)"
        else:
            r["verdict"] = "reject"
        out.append(r)
    return pd.DataFrame(out)


def write_audited_subset(audit: pd.DataFrame, cfg: Config, path: str) -> dict:
    key = lambda d: d.split()[0]
    retained = [key(d) for d, v in zip(audit.dimension, audit.verdict) if v.startswith("retain")]
    proxy_only = [key(d) for d, v in zip(audit.dimension, audit.verdict) if v == "proxy-only"]
    dims = [d for d in ["C_t", "B_t", "M_t", "G_t", "H_t", "I_t"] if d in set(retained + proxy_only)]
    if not dims:
        dims = ["C_t"]
    sub = dict(dataset=cfg.dataset, seed=cfg.seed,
               thresholds=dict(reliability=cfg.reliability_floor,
                               convergent=cfg.convergent_floor,
                               simulator_recovery=cfg.simulator_recovery_floor,
                               warm_min_history=cfg.warm_min_history),
               retained=retained, proxy_only=proxy_only, model_dimensions=dims,
               uncertainty="belief-level (Sigma_t), not a state coordinate",
               per_dimension={key(row.dimension): {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                                                   for k, v in row._asdict().items() if k not in ("Index", "dimension")}
                              for row in audit.itertuples()})
    with open(path, "w") as fh:
        json.dump(sub, fh, indent=2, default=str)
    return sub


# ============================================================================
# 6. SPLIT CONSTRUCTION AND GAP BUCKETS
# ============================================================================
BUCKETS = [("within", 0, 20 * 60), ("sameday", 20 * 60, 86400), ("d1", 86400, 7 * 86400),
           ("d7", 7 * 86400, 28 * 86400), ("d28", 28 * 86400, np.inf)]
REAL_HORIZONS = ["d1", "d7", "d28"]


def build_packs(df, cfg: Config, rng, maxgap_d=None):
    """Learner-level split, stratified on each learner's largest gap.

    Stratifying on gap length does not leak outcome information — the gap is a property of
    WHEN a learner appeared, not of how they performed — but it is a deviation from simple
    random assignment and is recorded in the provenance so a reader can judge it.
    """
    df = df.sort_values(["learner_id", "start_time"]).copy()
    keep = df.groupby("learner_id").size()
    df = df[df.learner_id.isin(set(keep[keep >= cfg.min_seq_len].index))].copy()
    learners = np.sort(df.learner_id.unique())

    if cfg.split_mode == "learner" and maxgap_d is not None:
        gap = maxgap_d.reindex(learners).fillna(0).values
        strat = np.digitize(gap, [1, 7, 28])          # 0:<1d 1:1-7d 2:7-28d 3:>=28d
        split_of = {}
        for s in np.unique(strat):
            ids = learners[strat == s].copy()
            rng.shuffle(ids)
            n = len(ids)
            ntr, nva = int(cfg.train_frac * n), int(cfg.val_frac * n)
            for l in ids[:ntr]:
                split_of[l] = "train"
            for l in ids[ntr:ntr + nva]:
                split_of[l] = "val"
            for l in ids[ntr + nva:]:
                split_of[l] = "test"
    else:
        learners = learners.copy()
        rng.shuffle(learners)
        n = len(learners)
        ntr, nva = int(cfg.train_frac * n), int(cfg.val_frac * n)
        split_of = {**{l: "train" for l in learners[:ntr]},
                    **{l: "val" for l in learners[ntr:ntr + nva]},
                    **{l: "test" for l in learners[ntr + nva:]}}
    df["split"] = df.learner_id.map(split_of)

    # skill vocabulary derived from TRAIN only; unseen skills map to index 0
    tsk = df.loc[df.split == "train", "skill_id"].unique()
    s2i = {s: i + 1 for i, s in enumerate(sorted(tsk, key=str))}
    df["skill_idx"] = df.skill_id.map(s2i).fillna(0).astype(int)
    n_skills = len(s2i) + 1

    # behaviour imputation + scaling on TRAIN statistics only
    med = df[df.split == "train"][BEHAV].median()
    df[BEHAV] = df[BEHAV].fillna(med)
    df["response_time"] = np.log1p(df["response_time"].clip(lower=0))
    tr = df[df.split == "train"]
    mu, sd = tr[BEHAV].mean(), tr[BEHAV].std().replace(0, 1.0)
    df[BEHAV] = ((df[BEHAV] - mu) / sd).fillna(0.0)
    df["dt_sec"] = df.groupby("learner_id")["start_time"].diff().dt.total_seconds().fillna(0).clip(lower=0)

    packs = {}
    for sp in ("train", "val", "test"):
        sub = df[df.split == sp]
        gids = sub.learner_id.unique()
        L = cfg.max_seq_len
        S = np.zeros((len(gids), L), np.int64)
        A = np.zeros((len(gids), L), np.float32)
        Fb = np.zeros((len(gids), L, len(BEHAV)), np.float32)
        D = np.zeros((len(gids), L), np.float32)
        M = np.zeros((len(gids), L), np.float32)
        for i, l in enumerate(gids):
            g = sub[sub.learner_id == l].tail(L)
            k = len(g)
            S[i, :k] = g.skill_idx.values
            A[i, :k] = g.correct.values
            Fb[i, :k] = g[BEHAV].values
            D[i, :k] = g.dt_sec.values
            M[i, :k] = 1.0
        packs[sp] = dict(skill=S, correct=A, behav=Fb, dt=D, mask=M, learner_id=gids)
        print(f"  {sp:5s} {len(gids):,} learners | {int(M.sum()):,} interactions")
    return packs, n_skills


def bucket_of(pack):
    """Bucket id per prediction position.

    Position t predicts t+1, so the gap that matters is dt[t+1] — the elapsed time
    CROSSED to reach the predicted item.
    """
    dtp = np.zeros_like(pack["dt"])
    dtp[:, :-1] = pack["dt"][:, 1:]
    out = np.full(dtp.shape, -1, np.int64)
    for i, (_, lo, hi) in enumerate(BUCKETS):
        out[(dtp >= lo) & (dtp < hi)] = i
    return out


def bucket_counts(packs, test_bucket):
    m = packs["test"]["mask"][:, 1:] > 0
    rows = []
    for i, (k, _, _) in enumerate(BUCKETS):
        sel = (test_bucket[:, :-1] == i) & m
        rows.append(dict(horizon=k, points=int(sel.sum()),
                         learners=int(np.unique(np.where(sel)[0]).size)))
    return pd.DataFrame(rows)


# ============================================================================
# 7. MODELS
# ============================================================================
# Imported lazily so that the RQ1 audit notebook (CPU, no training) does not need torch.
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    return torch, nn, Fn


def build_model_classes():
    """Return the five model classes. Defined inside a function so torch is imported lazily.

    All five are carried over unchanged from P2_Revision_Pipeline_v3. The single alias
    `Fn` is used throughout: the original had BOTH `import torch.nn.functional as Fn` and
    `... as F` because classes pasted in from an older notebook referenced `F`, and a
    missing alias produced a NameError mid-run.
    """
    torch, nn, Fn = _torch()

    class PackDS(torch.utils.data.Dataset):
        def __init__(self, p):
            self.p = p

        def __len__(self):
            return self.p["skill"].shape[0]

        def __getitem__(self, i):
            return dict(skill=torch.from_numpy(self.p["skill"][i]),
                        correct=torch.from_numpy(self.p["correct"][i]),
                        behav=torch.from_numpy(self.p["behav"][i]),
                        dt=torch.from_numpy(self.p["dt"][i]),
                        mask=torch.from_numpy(self.p["mask"][i]), idx=i)

    class DKT(nn.Module):
        """Recurrent KT. use_time=True gives DKT-F, the forgetting-aware comparator.

        NOTE (see AUDIT_REPORT.md Finding 1): with use_time=True the model receives BOTH
        the gap before position t AND the gap before t+1 — the gap crossed to reach the
        item being predicted. PCDT's Psi below decays by the FORMER only. The retention
        analysis buckets predictions by the LATTER. That asymmetry is preserved here
        because changing it would change every reported number; it is flagged, not fixed.
        """
        def __init__(self, n_skills, hidden, dropout=0.2, use_time=False):
            super().__init__()
            self.n_skills, self.use_time = n_skills, use_time
            self.emb = nn.Embedding(2 * n_skills + 1, hidden, padding_idx=0)
            self.lstm = nn.LSTM(hidden + (2 if use_time else 0), hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, n_skills)

        def forward(self, b):
            s, a, dt = b["skill"], b["correct"], b["dt"]
            x = self.emb((s + a.long() * self.n_skills).clamp(max=self.emb.num_embeddings - 1)[:, :-1])
            if self.use_time:
                g = torch.log1p(dt[:, :-1] / 60.0).unsqueeze(-1)
                gn = torch.log1p(dt[:, 1:] / 60.0).unsqueeze(-1)
                x = torch.cat([x, g, gn], dim=-1)
            h, _ = self.lstm(x)
            return self.out(self.drop(h)).gather(-1, s[:, 1:].unsqueeze(-1)).squeeze(-1)

    class SAKT(nn.Module):
        """Self-attentive KT: query = target exercise, key/value = past interactions."""
        def __init__(self, n_skills, hidden, n_heads=4, dropout=0.2, max_len=200):
            super().__init__()
            # embed_dim must be divisible by num_heads, else the capacity search hits
            # invalid widths, the constructor raises, and the search silently collapses
            # downward leaving SAKT far below the parameter budget.
            hidden = max(n_heads, (hidden // n_heads) * n_heads)
            self.inter_emb = nn.Embedding(2 * n_skills + 1, hidden, padding_idx=0)
            self.ex_emb = nn.Embedding(n_skills, hidden, padding_idx=0)
            self.pos_emb = nn.Embedding(max_len, hidden)
            self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, 1)
            self.n_skills = n_skills

        def forward(self, b):
            s, a = b["skill"], b["correct"]
            inter = (s + a.long() * self.n_skills).clamp(max=self.inter_emb.num_embeddings - 1)
            L = s.shape[1] - 1
            pos = torch.arange(L, device=s.device).unsqueeze(0)
            k = self.inter_emb(inter[:, :-1]) + self.pos_emb(pos)
            q = self.ex_emb(s[:, 1:])
            causal = torch.triu(torch.ones(L, L, device=s.device, dtype=torch.bool), diagonal=1)
            att, _ = self.attn(q, k, k, attn_mask=causal)
            h = self.ln1(q + self.drop(att))
            h = self.ln2(h + self.drop(self.ffn(h)))
            return self.out(h).squeeze(-1)

    class AKT(nn.Module):
        """Attentive KT: Rasch-style embeddings + monotonic distance-decayed attention."""
        def __init__(self, n_skills, hidden, n_heads=4, dropout=0.2, max_len=200):
            super().__init__()
            self.c_emb = nn.Embedding(n_skills, hidden, padding_idx=0)
            self.d_emb = nn.Embedding(n_skills, hidden, padding_idx=0)
            self.diff = nn.Embedding(n_skills, 1, padding_idx=0)
            self.resp_emb = nn.Embedding(3, hidden, padding_idx=2)
            self.q_proj, self.k_proj, self.v_proj = (nn.Linear(hidden, hidden) for _ in range(3))
            self.theta = nn.Parameter(torch.tensor(0.5))
            self.ln = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.drop = nn.Dropout(dropout)
            self.scale = hidden ** 0.5

        def _rasch(self, s):
            return self.c_emb(s) + self.diff(s) * self.d_emb(s)

        def forward(self, b):
            s, a = b["skill"], b["correct"]
            L = s.shape[1] - 1
            x_past = self._rasch(s[:, :-1]) + self.resp_emb(a[:, :-1].long())
            q = self.q_proj(self._rasch(s[:, 1:]))
            k, v = self.k_proj(x_past), self.v_proj(x_past)
            scores = (q @ k.transpose(-2, -1)) / self.scale
            pos = torch.arange(L, device=s.device)
            dist = (pos.view(-1, 1) - pos.view(1, -1)).float().clamp(min=0)
            scores = scores - Fn.softplus(self.theta) * dist
            causal = torch.triu(torch.ones(L, L, device=s.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))
            ctx = self.drop(torch.softmax(scores, dim=-1)) @ v
            return self.ffn(torch.cat([self.ln(ctx), q], dim=-1)).squeeze(-1)

    class DKVMN(nn.Module):
        """Memory-augmented KT: static key memory + dynamic value memory, erase-add writes.

        AUDIT FIX (CORRECTIONS.md, Finding A). The archived implementation READ the value
        memory before WRITING interaction t, so the prediction for target t+1 was formed
        from interactions 0..t-1 while every other model in the suite used 0..t. DKVMN was
        therefore denied the single most recent interaction. That is not the Zhang et al.
        (2017) formulation, in which (q_t, a_t) is written before q_{t+1} is read.

        legacy_lag=True restores the archived behaviour so a run can be reproduced exactly.
        The default is the corrected ordering.
        """
        def __init__(self, n_skills, hidden, n_concepts=32, dropout=0.2, legacy_lag=False):
            super().__init__()
            self.legacy_lag = legacy_lag
            self.N, self.dk, self.dv = n_concepts, hidden, hidden
            self.k_emb = nn.Embedding(n_skills, self.dk, padding_idx=0)
            self.v_emb = nn.Embedding(2 * n_skills + 1, self.dv, padding_idx=0)
            self.Mk = nn.Parameter(torch.randn(self.N, self.dk) * 0.1)
            self.Mv0 = nn.Parameter(torch.randn(self.N, self.dv) * 0.1)
            self.erase, self.add = nn.Linear(self.dv, self.dv), nn.Linear(self.dv, self.dv)
            self.ffn = nn.Sequential(nn.Linear(self.dv + self.dk, hidden), nn.Tanh(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.n_skills = n_skills

        def forward(self, b):
            s, a = b["skill"], b["correct"]
            B, L = s.shape
            inter = (s + a.long() * self.n_skills).clamp(max=self.v_emb.num_embeddings - 1)
            Mv = self.Mv0.unsqueeze(0).expand(B, -1, -1).contiguous()
            outs = []
            for t in range(L - 1):
                if self.legacy_lag:
                    # archived ordering: read, predict, then write interaction t
                    kq = self.k_emb(s[:, t + 1])
                    w = torch.softmax(kq @ self.Mk.t(), dim=-1)
                    r = torch.einsum("bn,bnd->bd", w, Mv)
                    outs.append(self.ffn(torch.cat([r, kq], dim=-1)))
                    vt = self.v_emb(inter[:, t])
                    wv = w
                else:
                    # corrected ordering: write interaction t, then read for target t+1
                    kv = self.k_emb(s[:, t])
                    wv = torch.softmax(kv @ self.Mk.t(), dim=-1)
                    vt = self.v_emb(inter[:, t])
                e, ad = torch.sigmoid(self.erase(vt)), torch.tanh(self.add(vt))
                Mv = Mv * (1 - wv.unsqueeze(-1) * e.unsqueeze(1)) + wv.unsqueeze(-1) * ad.unsqueeze(1)
                if not self.legacy_lag:
                    kq = self.k_emb(s[:, t + 1])
                    w = torch.softmax(kq @ self.Mk.t(), dim=-1)
                    r = torch.einsum("bn,bnd->bd", w, Mv)
                    outs.append(self.ffn(torch.cat([r, kq], dim=-1)))
            return torch.cat(outs, dim=1)

    class PCDT(nn.Module):
        """Persistent representation with the time-gap persistence operator Psi.

        reset_sessions=True gives the session-resetting control (no persistence at all).

        psi_uses_target_gap is a DIAGNOSTIC SWITCH, default False = original behaviour.
        False: Psi decays by dt[t], the gap before the OBSERVED item (what was run).
        True : Psi decays by dt[t+1], the gap crossed to the PREDICTED item — the same
               quantity DKT-F receives as a feature and the same quantity the retention
               buckets are defined on. Flipping it changes every number, so it is off by
               default and exists only so the asymmetry in AUDIT_REPORT.md Finding 1 can
               be tested without editing the model.
        """
        def __init__(self, n_skills, hidden, n_behav=3, tau0=3600., dropout=0.2,
                     use_persistence=True, reset_sessions=False, session_gap_s=1200.,
                     psi_uses_target_gap=False):
            super().__init__()
            self.n_skills, self.tau0 = n_skills, tau0
            self.use_persistence = use_persistence
            self.reset_sessions, self.session_gap_s = reset_sessions, session_gap_s
            self.psi_uses_target_gap = psi_uses_target_gap
            self.state = hidden
            self.emb = nn.Embedding(2 * n_skills + 1, hidden, padding_idx=0)
            self.ex = nn.Embedding(n_skills, hidden, padding_idx=0)
            self.bproj = nn.Linear(n_behav, hidden)
            self.cell = nn.GRUCell(2 * hidden, self.state)
            self.log_lambda = nn.Parameter(torch.tensor(-1.0))
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(self.state + hidden, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

        def psi(self, h, dt):
            if not self.use_persistence:
                return h
            lam = Fn.softplus(self.log_lambda)
            return h * torch.exp(-lam * torch.log1p(dt.clamp(min=0) / self.tau0)).unsqueeze(-1)

        def forward(self, b):
            s, a, f, dt = b["skill"], b["correct"], b["behav"], b["dt"]
            B, L = s.shape
            inter = (s + a.long() * self.n_skills).clamp(max=self.emb.num_embeddings - 1)
            h = torch.zeros(B, self.state, device=s.device)
            outs = []
            for t in range(L - 1):
                x = torch.cat([self.emb(inter[:, t]), self.bproj(f[:, t])], dim=-1)
                if self.reset_sessions:
                    h = h * (1.0 - (dt[:, t] >= self.session_gap_s).float().unsqueeze(-1))
                else:
                    h = self.psi(h, dt[:, t])
                h = self.cell(x, h)
                hh = self.psi(h, dt[:, t + 1]) if self.psi_uses_target_gap else h
                outs.append(self.head(torch.cat([self.drop(hh), self.ex(s[:, t + 1])], dim=-1)))
            return torch.cat(outs, dim=1)

    return dict(PackDS=PackDS, DKT=DKT, SAKT=SAKT, AKT=AKT, DKVMN=DKVMN, PCDT=PCDT)


# ============================================================================
# 8. TRAINING HARNESS
# ============================================================================
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def check_alignment(cls, cfg: "Config", n_skills: int = 7, L: int = 12, B: int = 4,
                    verbose: bool = True):
    """Assert every model predicts target j from interactions 0..j-1 — no more, no less.

    Added after the audit of the archived runs, in which DKVMN read its value memory
    BEFORE writing interaction t and so formed its prediction for target j from 0..j-2
    while the other seven models used 0..j-1. The defect is invisible in the loss curve,
    survives capacity matching, and cost the comparator between one and two thousandths
    of AUC in exactly the bucket a ranking claim was built on.

    The test needs no data and no training: it perturbs the response at one position and
    records which output positions move, which is a property of the forward pass alone.

    Two failures are distinguished:
      LAGS    - target j does not use position j-1: the model is under-informed relative
                to its competitors and any comparison between them is confounded.
      LEAKS   - target j uses position j or later: the model can see the answer it is
                being asked to predict.
    """
    torch, _, _ = _torch()
    g = torch.Generator().manual_seed(1)
    skill = torch.randint(1, n_skills, (B, L), generator=g)
    correct = torch.randint(0, 2, (B, L), generator=g).float()
    behav = torch.randn(B, L, len(BEHAV), generator=g)
    dt = torch.rand(B, L, generator=g) * 5000
    mask = torch.ones(B, L)
    batch = lambda c: dict(skill=skill, correct=c, behav=behav, dt=dt, mask=mask)

    rows = []
    for name, build in model_specs(n_skills, cfg, cls).items():
        torch.manual_seed(7)
        m = build(16).eval()
        with torch.no_grad():
            base = m(batch(correct))
        dep = {u: [] for u in range(L - 1)}
        for t in range(L - 1):
            c2 = correct.clone()
            c2[:, t] = 1.0 - c2[:, t]
            with torch.no_grad():
                alt = m(batch(c2))
            moved = (alt - base).abs().max(dim=0).values > 1e-6
            for u in torch.nonzero(moved).flatten().tolist():
                dep[u].append(t)
        # Output index u is the prediction for target position u+1, so the most recent
        # interaction it may use is position u. Test the MAXIMUM, not membership: a model
        # that legitimately drops older history (Control zeroes the state at a session
        # boundary) still consumes position u, and must not be flagged for that.
        highest = {u + 1: (max(ts) if ts else None) for u, ts in dep.items()}
        leaks = any(v is not None and v > j - 1 for j, v in highest.items())
        lags = any(v is None or v < j - 1 for j, v in highest.items())
        rows.append(dict(model=name, leaks=leaks, lags=lags,
                         highest_position_used=highest,
                         verdict="LEAKS" if leaks else "LAGS" if lags else "ok"))
    df = pd.DataFrame(rows)
    if verbose:
        print("Temporal alignment gate — target j must use interactions 0..j-1\n")
        for r in df.itertuples():
            ex = ", ".join(f"{j}:{'-' if v is None else v}"
                           for j, v in list(r.highest_position_used.items())[:6])
            print(f"  {r.model:12s} {r.verdict:6s}  highest position used per target: {ex} ...")
    bad = df[df.verdict != "ok"]
    if len(bad):
        raise AssertionError(
            "TEMPORAL ALIGNMENT GATE FAILED for "
            + ", ".join(f"{r.model} ({r.verdict})" for r in bad.itertuples())
            + ". Training would produce a comparison confounded by history length, not "
              "by representation. Fix the forward pass before spending GPU time.")
    if verbose:
        print("\nGATE PASSED — all eight models see identical history.")
    return df


def match_capacity(builder, budget, tol, lo=8, hi=512):
    """Binary-search the hidden width so total parameters land closest to the budget."""
    best, bg = None, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            n = count_params(builder(mid))
        except Exception:
            hi = mid - 1
            continue
        if abs(n - budget) < bg:
            best, bg = (mid, n), abs(n - budget)
        if n < budget:
            lo = mid + 1
        else:
            hi = mid - 1
    return best[0], best[1], abs(best[1] - budget) / budget <= tol


def train_one(builder, hidden, lr, packs, cfg: Config, cls, device, seed):
    """Identical harness for every model: same optimiser, grid, patience, early stopping."""
    torch, nn, Fn = _torch()
    seed_everything(seed)
    model = builder(hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    tr = torch.utils.data.DataLoader(cls["PackDS"](packs["train"]), batch_size=cfg.batch_size, shuffle=True)
    va = torch.utils.data.DataLoader(cls["PackDS"](packs["val"]), batch_size=cfg.batch_size, shuffle=False)

    def masked_bce(logits, y, m):
        l = Fn.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (l * m).sum() / m.sum().clamp(min=1.)

    best, bstate, bad = -1e9, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        for b in tr:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss = masked_bce(model(b), b["correct"][:, 1:], b["mask"][:, 1:])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for b in va:
                b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                m = b["mask"][:, 1:] > 0
                P.append(torch.sigmoid(model(b))[m].cpu().numpy())
                Y.append(b["correct"][:, 1:][m].cpu().numpy())
        p, y = np.concatenate(P), np.concatenate(Y)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        if auc > best + 1e-5:
            best, bad = auc, 0
            bstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if bstate is not None:
        model.load_state_dict(bstate)
    return model, best


def predict(model, pack, cfg: Config, cls, device):
    """Dense [n_learners, L-1] probability matrix aligned with mask[:,1:] and bucket[:,:-1]."""
    torch, _, _ = _torch()
    model.eval()
    dl = torch.utils.data.DataLoader(cls["PackDS"](pack), batch_size=cfg.batch_size, shuffle=False)
    out = np.zeros((pack["skill"].shape[0], pack["skill"].shape[1] - 1), np.float32)
    with torch.no_grad():
        for b in dl:
            idx = b["idx"].numpy()
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out[idx] = torch.sigmoid(model(b)).cpu().numpy()
    return out


MODEL_ORDER = ["DKT", "SAKT", "AKT", "DKVMN", "DKT-F", "PCDT", "PCDT-noPsi", "Control"]
BASELINES = ["DKT", "SAKT", "AKT", "DKVMN", "DKT-F"]


def model_specs(n_skills, cfg: Config, cls):
    """The eight models. PCDT-noPsi and Control are the two ablations of the operator."""
    C = cls
    nb = len(BEHAV)
    return {
        "DKT":        lambda h: C["DKT"](n_skills, h, use_time=False),
        "SAKT":       lambda h: C["SAKT"](n_skills, h, max_len=cfg.max_seq_len),
        "AKT":        lambda h: C["AKT"](n_skills, h),
        "DKVMN":      lambda h: C["DKVMN"](n_skills, h, legacy_lag=cfg.dkvmn_legacy_lag),
        "DKT-F":      lambda h: C["DKT"](n_skills, h, use_time=True),
        "PCDT":       lambda h: C["PCDT"](n_skills, h, n_behav=nb, tau0=cfg.tau0_seconds),
        "PCDT-noPsi": lambda h: C["PCDT"](n_skills, h, n_behav=nb, use_persistence=False),
        "Control":    lambda h: C["PCDT"](n_skills, h, n_behav=nb, reset_sessions=True,
                                          use_persistence=False, session_gap_s=cfg.session_gap_s),
    }


def train_all(packs, n_skills, cfg: Config, cls, device):
    """Capacity-match, grid-search lr, train and predict for all eight models."""
    specs = model_specs(n_skills, cfg, cls)
    if cfg.dkvmn_legacy_lag:
        print("*** DKVMN LEGACY LAG IS ON: DKVMN is denied the most recent interaction that "
              "every other model receives. Reproduction mode only — not a fair comparison. ***")
    info, pred = {}, {}
    for name, b in specs.items():
        t0 = time.time()
        hh, npar, ok = match_capacity(b, cfg.param_budget, cfg.param_tolerance)
        best = (-1e9, None, None)
        for lr in cfg.search_lrs:
            mdl, auc = train_one(b, hh, lr, packs, cfg, cls, device, cfg.seed)
            if auc > best[0]:
                best = (auc, lr, mdl)
        info[name] = dict(model=best[2], lr=best[1], val_auc=float(best[0]), hidden=int(hh),
                          params=int(npar), capacity_matched=bool(ok),
                          minutes=round((time.time() - t0) / 60, 2))
        pred[name] = predict(best[2], packs["test"], cfg, cls, device)
        print(f"  {name:12s} hidden={hh:4d} params={npar:8,} lr={best[1]:<7g} "
              f"val AUC={best[0]:.4f} {'' if ok else '*** CAPACITY OUTSIDE TOLERANCE ***'}")
    if not all(v["capacity_matched"] for v in info.values()):
        print("*** WARNING: at least one model is outside the capacity tolerance. Any comparison "
              "involving it confounds representation with capacity. ***")
    return info, pred


# ============================================================================
# 9. METRICS AND STATISTICS
# ============================================================================
def ece(p, y, bins=15):
    e, edges = 0., np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    for b in range(bins):
        s = idx == b
        if s.sum():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)


def evaluate(p, y, bins=15):
    fin = np.isfinite(p) & np.isfinite(y)
    p, y = p[fin], y[fin]
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), auc=float(roc_auc_score(y, p)), acc=float(((p >= .5) == y).mean()),
                rmse=float(np.sqrt(((p - y) ** 2).mean())),
                nll=float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
                ece=ece(p, y, bins), brier=float(brier_score_loss(y, p)))


def paired_bootstrap(pa, pb, y, rows_idx, n_boot, seed):
    """Paired bootstrap resampling LEARNERS, not interactions.

    Resampling interactions would treat repeated measures on one learner as independent
    and shrink the interval spuriously.
    """
    rg = np.random.default_rng(seed)
    learners = np.unique(rows_idx)
    by = {l: np.where(rows_idx == l)[0] for l in learners}
    d = []
    for _ in range(n_boot):
        s = rg.choice(learners, len(learners), replace=True)
        ii = np.concatenate([by[l] for l in s])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return (float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), float(lo), float(hi), float(min(pv, 1)))


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def did_retention(pa, pb, bucket_id, packs, test_bucket, Y, cfg: Config, ref_id=0):
    """Difference-in-differences retention advantage at one horizon vs the within-session ref.

    Reports adequate=False rather than a number when the pre-registered floor is not met.
    No horizon is silently substituted by a simulated value.
    """
    Mm = packs["test"]["mask"][:, 1:] > 0
    B = test_bucket[:, :-1]
    selh, selr = (B == bucket_id) & Mm, (B == ref_id) & Mm
    nl, npt = int(np.unique(np.where(selh)[0]).size), int(selh.sum())
    if nl < cfg.min_learners_bucket or npt < cfg.min_points_bucket:
        return dict(adequate=False, n_learners=nl, n_points=npt)

    def auc(sel, p):
        y = Y[sel]
        return roc_auc_score(y, p[sel]) if len(np.unique(y)) > 1 else np.nan

    ra = (auc(selh, pa) - auc(selh, pb)) - (auc(selr, pa) - auc(selr, pb))
    rg = np.random.default_rng(cfg.seed)
    ls = np.unique(np.where(selh | selr)[0])
    boot = []
    for _ in range(cfg.n_bootstrap):
        keep = np.zeros(selh.shape[0], bool)
        keep[rg.choice(ls, len(ls), replace=True)] = True
        mh, mr = selh & keep[:, None], selr & keep[:, None]
        if mh.sum() < 50 or mr.sum() < 50:
            continue
        try:
            boot.append((auc(mh, pa) - auc(mh, pb)) - (auc(mr, pa) - auc(mr, pb)))
        except Exception:
            pass
    boot = np.asarray([b for b in boot if np.isfinite(b)])
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
    return dict(adequate=True, n_learners=nl, n_points=npt, retention_advantage=float(ra),
                ci_lo=float(lo), ci_hi=float(hi),
                auc_pcdt=float(auc(selh, pa)), auc_comparator=float(auc(selh, pb)))


def utility(p, y, thr):
    """Flag-for-review decision at a probability threshold.

    A learner is flagged when predicted success falls below thr; 'needs review' is a
    genuinely incorrect next response. Precision and false alarms are reported ALONGSIDE
    correct flags: a model can buy recall with precision and look better on counts alone.
    """
    flag, need = (p < thr).astype(int), (y == 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(need, flag, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec) else np.nan
    return dict(threshold=thr, flagged=int(tp + fp), correctly_flagged=int(tp), false_alarms=int(fp),
                missed=int(fn), precision=prec, recall=rec, specificity=spec, f1=f1,
                balanced_accuracy=(rec + spec) / 2)


def postgap_auc_by_model(pred, packs, test_bucket, Y, cfg: Config):
    """Per-model AUC in every gap bucket. This is RQ4's raw evidence.

    In the original package this was computed only inside the seed-confirmation notebooks,
    so the main pipeline never produced the table its most novel claim rests on.
    """
    Mm = packs["test"]["mask"][:, 1:] > 0
    B = test_bucket[:, :-1]
    rows = []
    for name, p in pred.items():
        rec = dict(model=name)
        for i, (k, _, _) in enumerate(BUCKETS):
            sel = (B == i) & Mm
            y = Y[sel]
            rec[f"auc_{k}"] = (float(roc_auc_score(y, p[sel]))
                               if sel.sum() >= cfg.min_points_bucket and len(np.unique(y)) > 1 else np.nan)
            rec[f"n_{k}"] = int(sel.sum())
        real = [rec[f"auc_{k}"] for k in REAL_HORIZONS if np.isfinite(rec.get(f"auc_{k}", np.nan))]
        rec["auc_mean_postgap"] = float(np.mean(real)) if real else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["rank_within"] = df["auc_within"].rank(ascending=False, method="min")
    df["rank_mean_postgap"] = df["auc_mean_postgap"].rank(ascending=False, method="min")
    return df.sort_values("auc_within", ascending=False).reset_index(drop=True)


def environment() -> dict:
    env = dict(python=sys.version.split()[0], platform=platform.platform(),
               numpy=np.__version__, pandas=pd.__version__)
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                   gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    except Exception:
        env.update(torch=None)
    return env


def save_tables(tables: dict, tab_dir: str, latex=True):
    for name, t in tables.items():
        t.to_csv(os.path.join(tab_dir, f"{name}.csv"), index=False)
        if latex:
            with open(os.path.join(tab_dir, f"{name}.tex"), "w") as fh:
                fh.write(t.to_latex(index=False, float_format="%.4f", escape=True))
    print(f"tables -> {tab_dir}: {sorted(tables)}")
