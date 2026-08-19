#!/usr/bin/env python3
"""Layer 0 — package integrity.

For every run: does the manifest describe what is actually on disk?
Recompute sha256 of every recorded output and every recorded input.
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path("/home/claude/work/P02/PCDT_P2_runs/runs")
# manifests record absolute paths from the user's machine; map them onto ours
USER_ROOT = "/home/hamed/Downloads/Research/P02/PCDT_P2_runs"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def localise(p: str) -> Path:
    p = str(p)
    if p.startswith(USER_ROOT):
        return Path("/home/claude/work/P02/PCDT_P2_runs") / p[len(USER_ROOT):].lstrip("/")
    return Path(p)


rows = []
problems = []
all_out_hashes = {}

for run_dir in sorted(ROOT.iterdir()):
    if not run_dir.is_dir():
        continue
    mf = run_dir / "manifest.json"
    if not mf.exists():
        problems.append(f"{run_dir.name}: NO manifest.json")
        continue
    m = json.load(open(mf))
    n_out, n_bad, n_missing = 0, 0, 0
    for o in m.get("outputs", []):
        rel = o.get("path") or o.get("file")
        f = localise(rel)
        if not f.exists():
            f = run_dir / rel
        if not f.exists():
            n_missing += 1
            problems.append(f"{run_dir.name}: recorded output MISSING on disk: {rel}")
            continue
        n_out += 1
        rec = o.get("sha256")
        if rec:
            got = sha256(f)
            all_out_hashes[str(f.resolve())] = got
            if got != rec:
                n_bad += 1
                problems.append(
                    f"{run_dir.name}: SHA MISMATCH {rel}\n    manifest={rec}\n    ondisk  ={got}")
    # inputs
    n_in, n_in_bad = 0, 0
    for i in m.get("inputs", []):
        rel = i.get("path") or i.get("file")
        f = localise(rel)
        if not f.exists():
            problems.append(f"{run_dir.name}: recorded INPUT missing on disk: {rel}")
            continue
        n_in += 1
        rec = i.get("sha256")
        if rec and sha256(f) != rec:
            n_in_bad += 1
            problems.append(f"{run_dir.name}: INPUT SHA MISMATCH {rel}")

    # files on disk NOT recorded in the manifest
    recorded = set()
    for o in m.get("outputs", []):
        rel = o.get("path") or o.get("file")
        recorded.add(str(localise(rel).resolve()))
    on_disk = set()
    for f in run_dir.rglob("*"):
        if f.is_file() and f.name not in ("manifest.json", ".runinfo.json",
                                          "manifest_v1_backup.json"):
            on_disk.add(str(f.resolve()))
    unrecorded = sorted(on_disk - recorded)

    rows.append(dict(
        run=run_dir.name,
        status=m.get("status"),
        fixture=m.get("fixture"),
        trustworthy=m.get("trustworthy"),
        iso_ok=m.get("isolation_ok"),
        iso_ver=m.get("isolation_check_version"),
        n_foreign=len(m.get("foreign_writes", [])),
        n_unattr=len(m.get("unattributed_writes", [])),
        n_out=n_out, sha_bad=n_bad, out_missing=n_missing,
        n_in=n_in, in_bad=n_in_bad,
        n_missing_inputs=len(m.get("missing_inputs", [])),
        unrecorded=len(unrecorded),
        runtime_min=m.get("runtime_minutes"),
    ))
    for u in unrecorded:
        problems.append(f"{run_dir.name}: file on disk not recorded in manifest: "
                        f"{Path(u).relative_to(run_dir.resolve())}")

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 50)
print(df.to_string(index=False))
print()
print(f"=== {len(problems)} PROBLEM(S) ===")
for p in problems:
    print(" *", p)
json.dump(all_out_hashes, open("/home/claude/work/audit/out_hashes.json", "w"))
