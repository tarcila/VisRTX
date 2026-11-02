#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Run a frozen, seed-randomized Quality study. Never select seeds from results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(group, index):
    text = f"visrtx-quality-convergence-v1/{group}/{index}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("library", type=Path, help="frozen shared library")
    p.add_argument("runner", type=Path, help="quality_baseline executable")
    p.add_argument("assets", type=Path)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    root, library, runner, assets = (x.resolve() for x in
                                    (args.output, args.library, args.runner, args.assets))
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": 1, "size": 128, "spp_per_launch": 32, "depth": 32,
        "excluded_initial_launches": 2,
        "scenes": ["cloud", "plume", "support", "overlap", "receiver-matte", "receiver-native"],
        "candidate": {"spp": 512, "seeds": [seed("candidate", i) for i in range(8)]},
        "reference": {"spp": 4096, "seeds": [seed("reference", i) for i in range(8)]},
        "library": str(library), "library_sha256": sha(library),
        "runner": str(runner), "runner_sha256": sha(runner),
        "assets": str(assets),
        "asset_sha256": {str(path.relative_to(assets)): sha(path) for path in
                         [assets / "cloud-128/density.nvdb", assets / "plume-128/density.nvdb",
                          assets / "plume-128/temperature.nvdb", assets / "plume-256/emission_offset.nvdb"]},
        "analysis_contract": {
            "prefix_spp": [32, 64, 128, 256, 512],
            "mean_consistency_gate": "6 estimated replicate standard errors + 5e-5 absolute, per RGB channel",
            "reference_error": "report variance of independent reference mean; no heterogeneous golden claim",
            "performance": "128-square timings measure error-cost only, not 640-square FPS",
            "no_seed_or_tolerance_retuning": True,
        },
    }
    path = root / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise RuntimeError("refusing to alter a frozen study plan")
    else:
        path.write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(library.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    for scene in plan["scenes"]:
        for group in ("candidate", "reference"):
            launches = 2 + plan[group]["spp"] // 32
            for index, key in enumerate(plan[group]["seeds"]):
                folder = root / scene / f"{group}-{index}"
                if (folder / "complete.json").exists():
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                cmd = [str(runner), "visrtx_principle_volume", "quality", scene,
                       str(assets), str(folder), "128", "32", str(launches), "32", "1", "static", str(key)]
                print(scene, group, index, key, flush=True)
                with (folder / "run.log").open("w") as log:
                    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=600)
                (folder / "complete.json").write_text(json.dumps({"seed": key, "launches": launches}) + "\n")


if __name__ == "__main__":
    main()
