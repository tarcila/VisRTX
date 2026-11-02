#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Measure an exact-graph optimization against all frozen candidate replicates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from analyze_quality_replicates import estimate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("library", type=Path)
    args = p.parse_args()
    baseline, output, library = (v.resolve() for v in (args.baseline, args.output, args.library))
    plan = json.loads((baseline / "plan.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    contract = {"baseline": str(baseline), "baseline_plan_sha256": hashlib.sha256((baseline / "plan.json").read_bytes()).hexdigest(),
                "library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "gate": "every retained candidate accumulation image must match byte-for-byte; no new reference rendering needed only if this holds"}
    path = output / "comparison-plan.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise RuntimeError("refusing to change frozen comparison")
    path.write_text(json.dumps(contract, indent=2) + "\n")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(library.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    results = {}
    for scene in plan["scenes"]:
        costs = []
        for index, seed in enumerate(plan["candidate"]["seeds"]):
            target = output / scene / f"candidate-{index}"
            source = baseline / scene / f"candidate-{index}"
            target.mkdir(parents=True, exist_ok=True)
            with (target / "run.log").open("w") as log:
                subprocess.run([plan["runner"], "visrtx_principle_volume", "quality", scene,
                                plan["assets"], str(target), "128", "32", "18", "32", "1", "static", str(seed)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            for frame in range(2, 18):
                if (target / f"frame-{frame}.rgba").read_bytes() != (source / f"frame-{frame}.rgba").read_bytes():
                    raise RuntimeError(f"exact graph gate failed: {scene} replicate {index} frame {frame}")
            _, before_gpu, before_wall = estimate(source, 512, 128)
            _, after_gpu, after_wall = estimate(target, 512, 128)
            costs.append({"before_gpu_seconds": before_gpu, "after_gpu_seconds": after_gpu,
                          "before_wall_seconds": before_wall, "after_wall_seconds": after_wall})
        results[scene] = {"identical_retained_frames": 128, "replicates": costs,
                          "gpu_error_cost_ratio": float(np.mean([x["after_gpu_seconds"] for x in costs]) / np.mean([x["before_gpu_seconds"] for x in costs])),
                          "wall_error_cost_ratio": float(np.mean([x["after_wall_seconds"] for x in costs]) / np.mean([x["before_wall_seconds"] for x in costs]))}
        (output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
        print(scene, results[scene]["gpu_error_cost_ratio"], flush=True)


if __name__ == "__main__":
    main()
