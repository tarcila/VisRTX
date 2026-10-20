#!/usr/bin/env python3
"""Run paired, unprofiled pilots; retain raw output and per-process medians.

Build one identical harness against ABI-compatible base/branch libraries. The
environment is expected to already resolve the CUDA driver; only the device
library under test is injected. Not a CI timing assertion. First iteration is reported but excluded from warm
statistics; each trial is a fresh process. Device-resident inputs are a separate
branch-only measurement, not an unsupported base comparison.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("executable")
p.add_argument("base_library_dir")
p.add_argument("branch_library_dir")
p.add_argument("output", type=Path)
p.add_argument("--trials", type=int, default=3)
p.add_argument("--repeats", type=int, default=21)
p.add_argument("--case", action="append", default=[], metavar="NAME",
               help="run only the named default case; repeatable")
p.add_argument("--extra-case", action="append", default=[],
               metavar="NAME:COUNT:GEOMETRIES:MODE:FRAMES",
               help="append a custom case; repeatable")
a = p.parse_args()
if a.trials < 2 or a.repeats < 3:
    p.error("need at least two trials and three repetitions")
# name, primitive count per geometry, distinct geometries, update mode, frames
cases = [
    ("small-recommit", 32, 64, "recommit", 0),
    ("small-replace", 32, 64, "replace", 0),
    ("large-replace", 262144, 1, "replace", 0),
    ("unchanged-frame", 32, 64, "unchanged", 1),
    ("two-frames-replace", 32, 64, "replace", 2),
]
if a.case:
    known = {c[0] for c in cases}
    unknown = [n for n in a.case if n not in known]
    if unknown:
        p.error(f"unknown case(s) {unknown}; known: {sorted(known)}")
    cases = [c for c in cases if c[0] in a.case]
for spec in a.extra_case:
    try:
        name, count, geometries, mode, frames = spec.split(":")
        cases.append((name, int(count), int(geometries), mode, int(frames)))
    except ValueError:
        p.error(f"bad --extra-case {spec!r}; want NAME:COUNT:GEOMETRIES:MODE:FRAMES")
records = []
for name, count, geometries, mode, frames in cases:
    for trial in range(a.trials):
        order = ["base", "branch"] if trial % 2 == 0 else ["branch", "base"]
        for revision in order:
            env = os.environ.copy()
            lib = a.base_library_dir if revision == "base" else a.branch_library_dir
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                filter(None, [lib, env.get("LD_LIBRARY_PATH")]))
            command = [a.executable, "cylinder", str(count), "0",
                       str(a.repeats), "1", str(geometries), mode, str(frames), "1"]
            run = subprocess.run(command, env=env, capture_output=True, text=True)
            samples = [(int(i), float(ms)) for i, ms in re.findall(
                r"sample iteration=(\d+) ms=([\d.]+)", run.stdout)]
            record = dict(case=name, trial=trial, revision=revision,
                          command=command, library_dir=lib, returncode=run.returncode,
                          stdout=run.stdout, stderr=run.stderr)
            records.append(record)
            a.output.write_text(json.dumps(records, indent=2) + "\n")
            if run.returncode or len(samples) != a.repeats:
                raise SystemExit(f"failed {name} {revision}: {run.stderr}")
            record["warm_median_ms"] = statistics.median(ms for i, ms in samples if i)
            a.output.write_text(json.dumps(records, indent=2) + "\n")
    medians = {rev: statistics.median(r["warm_median_ms"] for r in records
                                    if r["case"] == name and r["revision"] == rev)
               for rev in ["base", "branch"]}
    print(name, medians, "branch/base=", medians["branch"] / medians["base"], flush=True)
