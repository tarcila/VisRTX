#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare frozen optimization runs; split-QMC errors are diagnostics, not gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_quality_baseline import blocks, metrics, overlap_reference, receiver_reference


def timing(folder):
    rows = list(csv.DictReader((folder / "timings.csv").open()))[2:]
    times = np.array([float(row["gpu_ms"]) for row in rows])
    return {"median_ms": float(np.median(times)), "max_ms": float(times.max()),
            "min_ms": float(times.min()),
            "gpu_wide_used_mib": max(int(row["gpu_used_bytes"]) for row in rows) / 2**20}


def convergence(folder):
    samples = blocks(folder)
    return {"split_256_vs_256": metrics(samples[:8].mean(0), samples[8:].mean(0)),
            "mean_rgb": samples[..., :3].mean(axis=(0, 1, 2)).tolist(),
            "median_batch_ms": timing(folder)["median_ms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="quality-opt evidence directory")
    args = parser.parse_args()
    root = args.root
    result = {"caveat": "Disjoint QMC blocks are not independent replicates or convergence goldens.",
              "rounds": {}, "analytic": {}, "paired_image_checks": {}, "hashes": {}}
    for folder in sorted(root.glob("round*")):
        record = {"timing": {}, "convergence": {}}
        for run in sorted(folder.glob("perf-*")):
            record["timing"][run.name] = timing(run)
        for run in sorted(folder.glob("convergence-*")):
            record["convergence"][run.name] = convergence(run)
        result["rounds"][folder.name] = record
        for binary in sorted((folder / "library").glob("*.so")):
            result["hashes"][str(binary.relative_to(root))] = hashlib.sha256(binary.read_bytes()).hexdigest()
    receiver = receiver_reference(128, 20)
    for scene in ("receiver-native", "receiver-matte", "overlap"):
        b = blocks(root / "round4" / f"convergence-{scene}")
        result["analytic"][scene] = metrics(b.mean(0), overlap_reference(128) if scene == "overlap" else receiver)
    for before, after, scenes in (("round1", "round2", ("cloud", "plume")),
                                  ("round1", "round2c", ("plume",)),
                                  ("round3", "round4", ("cloud", "plume"))):
        for scene in scenes:
            result["paired_image_checks"][f"{before}-{after}-{scene}"] = metrics(
                blocks(root / after / f"convergence-{scene}").mean(0),
                blocks(root / before / f"convergence-{scene}").mean(0))
    importance = root.parent / "quality-importance"
    result["importance_vs_final"] = {}
    for scene in ("cloud", "plume", "receiver-native", "receiver-matte", "overlap", "support"):
        before = convergence(importance / f"convergence-{scene}")
        after = convergence(root / "round4" / f"convergence-{scene}")
        a = before["split_256_vs_256"]["normalized_rms"]
        b = after["split_256_vs_256"]["normalized_rms"]
        result["importance_vs_final"][scene] = {"before": before, "after": after,
            "diagnostic_squared_split_error_cost_ratio":
                (b*b*after["median_batch_ms"])/(a*a*before["median_batch_ms"]) if a else None}
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"analytic": result["analytic"],
                      "final_timing": result["rounds"]["round4"]["timing"],
                      "importance_vs_final": result["importance_vs_final"]}, indent=2))


if __name__ == "__main__":
    main()
