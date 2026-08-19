"""
p2_runtime.py — one isolated, self-documenting output folder per notebook.

    {OUT_ROOT}/runs/{RUN_ID}/
        tables/  figures/  artifacts/  logs/console.log
        manifest.json     provenance + isolation verdict
        .runinfo.json     this run's [start, finish] window

ISOLATION, v2
-------------
v1 flagged ANY sibling file whose mtime changed during a run. That was wrong: when two
notebooks run in parallel — which RUN_ORDER.md recommends — each legitimately writes into
its OWN folder while the other is executing, and v1 reported that as a violation. Two
completed 92-minute runs were failed for doing exactly what they were told to do.

v2 distinguishes the two cases by run window. Every run records [started, finished] in
.runinfo.json. When a sibling's files change, this run reads that sibling's window:

  * windows OVERLAP  -> concurrent execution. Benign. Counted and reported, never fatal.
  * windows DISJOINT -> a finished run was modified after the fact. That is the real
                        violation, and it raises.
  * no .runinfo.json -> unattributable. Raises, because a legitimate run always writes one.
                        (repair_manifests.py backfills it for folders created before v2.)

Still guaranteed: no notebook writes outside its own folder, and any post-hoc modification
of a completed run is caught and named.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from typing import Optional

RUNS_SUBDIR = "runs"
RUNINFO = ".runinfo.json"


class _Tee:
    """Mirror stdout/stderr into the run's log, flushing every write so a crashed run
    still leaves a complete log."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        self._stream.write(s)
        try:
            self._fh.write(s)
            self._fh.flush()
        except ValueError:
            pass
        return len(s)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except ValueError:
            pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def __getattr__(self, k):
        return getattr(self._stream, k)


def _sha256(path: str, cap: int = 2_000_000_000) -> Optional[str]:
    h, n = hashlib.sha256(), 0
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
                n += len(chunk)
                if n > cap:
                    return "too-large-to-hash"
    except OSError:
        return None
    return h.hexdigest()


def read_runinfo(run_dir: str) -> Optional[dict]:
    p = os.path.join(run_dir, RUNINFO)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


class RunContext:
    """Owns one notebook's output directory. Nothing else writes there; it writes
    nowhere else."""

    def __init__(self, run_id: str, out_root: str, cfg=None, fixture: bool = False,
                 description: str = "", inputs_expected: Optional[list] = None):
        if "/" in run_id or "\\" in run_id:
            raise ValueError(f"run_id must be a single folder name, got {run_id!r}")
        self.run_id = ("FIXTURE_" + run_id) if fixture else run_id
        self.out_root = os.path.abspath(out_root)
        self.runs_root = os.path.join(self.out_root, RUNS_SUBDIR)
        self.root = os.path.join(self.runs_root, self.run_id)
        self.dirs = dict(out=self.root,
                         tab=os.path.join(self.root, "tables"),
                         fig=os.path.join(self.root, "figures"),
                         art=os.path.join(self.root, "artifacts"),
                         log=os.path.join(self.root, "logs"))
        for v in self.dirs.values():
            os.makedirs(v, exist_ok=True)

        self.t0 = self.started_at = time.time()
        self.description = description
        self.inputs: list = []
        self.missing_inputs: list = []
        self.inputs_expected = list(inputs_expected or [])
        self.cfg = cfg
        self.fixture = fixture

        self._write_runinfo(None)
        self._snap = self._scan_siblings()

        self._logfh = open(os.path.join(self.dirs["log"], "console.log"), "w",
                           encoding="utf-8")
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_out, self._logfh)
        sys.stderr = _Tee(self._orig_err, self._logfh)

        print("=" * 78)
        print(f"RUN {self.run_id}")
        if description:
            print(description)
        print(f"started   {time.strftime('%Y-%m-%d %H:%M:%S')}   pid {os.getpid()}")
        print(f"writes to {self.root}")
        print("reads     " + (", ".join(self.inputs_expected) if self.inputs_expected
                              else "nothing — this notebook is a producer"))
        live = self._live_siblings()
        if live:
            print(f"concurrent runs detected: {', '.join(live)}")
            print("  (parallel execution is supported and expected — see RUN_ORDER.md)")
        if fixture:
            print("*** FIXTURE MODE — outputs are NOT results ***")
        print("=" * 78)

    # -------------------------------------------------------------- run window
    def _write_runinfo(self, finished: Optional[float]):
        try:
            with open(os.path.join(self.root, RUNINFO), "w") as fh:
                json.dump(dict(run_id=self.run_id, pid=os.getpid(), host=platform.node(),
                               started_epoch=self.started_at, finished_epoch=finished,
                               started=time.strftime("%Y-%m-%d %H:%M:%S",
                                                     time.localtime(self.started_at)),
                               finished=(None if finished is None else
                                         time.strftime("%Y-%m-%d %H:%M:%S",
                                                       time.localtime(finished)))),
                          fh, indent=2)
        except OSError:
            pass

    def _live_siblings(self) -> list:
        out = []
        if not os.path.isdir(self.runs_root):
            return out
        for name in sorted(os.listdir(self.runs_root)):
            if name == self.run_id:
                continue
            info = read_runinfo(os.path.join(self.runs_root, name))
            if info and info.get("finished_epoch") is None:
                out.append(name)
        return out

    # -------------------------------------------------------------- paths
    def path(self, kind: str, *parts) -> str:
        if kind not in self.dirs:
            raise KeyError(f"unknown output kind {kind!r}; use {sorted(self.dirs)}")
        return os.path.join(self.dirs[kind], *parts)

    def sibling(self, run_id: str) -> Optional[str]:
        for cand in (run_id, "FIXTURE_" + run_id):
            p = os.path.join(self.runs_root, cand)
            if os.path.isdir(p):
                return p
        return None

    def read_input(self, path: Optional[str], label: str = "",
                   required: bool = True) -> Optional[str]:
        if path is None or not os.path.exists(path):
            self.inputs.append(dict(label=label, path=path, present=False))
            self.missing_inputs.append(label or str(path))
            if required:
                print(f"  MISSING INPUT  {label or path}")
            return None
        st = os.stat(path)
        self.inputs.append(dict(label=label, path=os.path.abspath(path), present=True,
                                sha256=_sha256(path), bytes=st.st_size,
                                mtime=time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime(st.st_mtime))))
        print(f"  read input     {label or os.path.basename(path)}")
        return path

    # -------------------------------------------------------------- isolation
    def _scan_siblings(self) -> dict:
        snap = {}
        if not os.path.isdir(self.runs_root):
            return snap
        for name in os.listdir(self.runs_root):
            if name == self.run_id:
                continue
            d = os.path.join(self.runs_root, name)
            if not os.path.isdir(d):
                continue
            for dp, _, fns in os.walk(d):
                for fn in fns:
                    p = os.path.join(dp, fn)
                    try:
                        snap[p] = os.stat(p).st_mtime_ns
                    except OSError:
                        pass
        return snap

    def _classify_foreign(self, my_end: float):
        now = self._scan_siblings()
        changed = [p for p, m in now.items() if self._snap.get(p) != m]
        by_owner = defaultdict(list)
        for p in changed:
            owner = os.path.relpath(p, self.runs_root).split(os.sep)[0]
            by_owner[owner].append(p)

        violations, unattributed, concurrent = [], [], {}
        for owner, files in by_owner.items():
            info = read_runinfo(os.path.join(self.runs_root, owner))
            if not info or info.get("started_epoch") is None:
                unattributed.extend(files)
                continue
            s = float(info["started_epoch"])
            e = info.get("finished_epoch")
            e = float("inf") if e is None else float(e)
            if s < my_end and e > self.started_at:       # windows overlap -> concurrent
                concurrent[owner] = len(files)
            else:                                         # a finished run was modified
                violations.extend(files)
        return sorted(violations), sorted(unattributed), concurrent

    # -------------------------------------------------------------- finish
    def finalize(self, status: str = "complete", notes: str = "",
                 extra: Optional[dict] = None):
        my_end = time.time()
        violations, unattributed, concurrent = self._classify_foreign(my_end)

        outputs = []
        for dp, _, fns in os.walk(self.root):
            for fn in sorted(fns):
                if fn in ("manifest.json", RUNINFO):
                    continue
                p = os.path.join(dp, fn)
                outputs.append(dict(path=os.path.relpath(p, self.root),
                                    bytes=os.path.getsize(p), sha256=_sha256(p)))

        env = dict(python=platform.python_version(), platform=platform.platform())
        for m in ("numpy", "pandas", "torch", "sklearn", "scipy", "matplotlib"):
            try:
                env[m] = __import__(m).__version__
            except Exception:
                pass
        try:
            import torch
            env["cuda"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                env["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        fixture_inputs = [i["label"] for i in self.inputs
                          if i.get("present") and "FIXTURE_" in str(i.get("path", ""))]
        if fixture_inputs:
            status = f"{status} [CONSUMED FIXTURE INPUTS — NOT RESULTS]"

        ok = not violations and not unattributed
        man = dict(run_id=self.run_id, description=self.description, status=status,
                   notes=notes, fixture=self.fixture,
                   started=time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(self.started_at)),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(my_end)),
                   runtime_minutes=round((my_end - self.t0) / 60, 2),
                   output_dir=self.root, environment=env,
                   config=(self.cfg if isinstance(self.cfg, dict) else None),
                   inputs=self.inputs, missing_inputs=self.missing_inputs,
                   n_outputs=len(outputs), outputs=outputs,
                   concurrent_runs=concurrent,
                   foreign_writes=violations, unattributed_writes=unattributed,
                   isolation_ok=ok, isolation_check_version=2,
                   fixture_inputs=fixture_inputs,
                   trustworthy=(ok and not self.fixture and not fixture_inputs))
        if extra:
            man.update(extra)
        # AUDIT FIX (CORRECTIONS.md, Finding D): the completion banner below is written
        # INTO console.log after the outputs are hashed, so the recorded hash could never
        # match the file on disk and that one integrity check was vacuous. The log entry is
        # now marked non-hashable rather than carrying a hash that is guaranteed wrong.
        for _o in man.get("outputs", []):
            if _o.get("path", "").endswith("logs/console.log"):
                _o["sha256"] = None
                _o["note"] = ("not hashed: the run's own completion banner is appended to "
                              "this file after the manifest is assembled")
        with open(os.path.join(self.root, "manifest.json"), "w") as fh:
            json.dump(man, fh, indent=2, default=str)
        self._write_runinfo(my_end)

        print("\n" + "=" * 78)
        print(f"RUN {self.run_id} — {status}")
        print(f"  runtime        {man['runtime_minutes']} min")
        print(f"  files written  {len(outputs)}")
        print(f"  inputs read    {sum(1 for i in self.inputs if i.get('present'))}"
              f" ({len(self.missing_inputs)} missing)")
        if concurrent:
            print("  concurrent     " + ", ".join(f"{k} ({v} files)"
                                                  for k, v in sorted(concurrent.items())))
            print("                 ran alongside this one and wrote only into its own "
                  "folder — expected, not a problem")
        if fixture_inputs:
            print(f"  *** consumed {len(fixture_inputs)} FIXTURE input(s); outputs are "
                  f"NOT results ***")
        if violations or unattributed:
            print(f"  *** ISOLATION VIOLATED — {len(violations) + len(unattributed)} "
                  f"file(s) in a FINISHED run were modified during this run:")
            for p in (violations + unattributed)[:10]:
                print(f"        {p}")
            if unattributed:
                print(f"      ({len(unattributed)} of them in a folder with no {RUNINFO} "
                      f"— run repair_manifests.py if these predate v2)")
            print("  *** Do not trust this run. ***")
        else:
            print("  isolation      OK — no finished run was modified")
        print(f"  manifest       {os.path.join(self.root, 'manifest.json')}")
        print("=" * 78)

        sys.stdout, sys.stderr = self._orig_out, self._orig_err
        try:
            self._logfh.close()
        except Exception:
            pass
        if violations or unattributed:
            raise RuntimeError(
                f"isolation violated: {len(violations) + len(unattributed)} file(s) in a "
                f"finished run modified during this run")
        return man

    def fail(self, exc: BaseException):
        print("\n*** RUN FAILED ***")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        return self.finalize(status="FAILED", notes=f"{type(exc).__name__}: {exc}")


def load_manifest(runs_root: str, run_id: str) -> Optional[dict]:
    for cand in (run_id, "FIXTURE_" + run_id):
        p = os.path.join(runs_root, cand, "manifest.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None
