#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Replicate error-cost estimates with explicit finite-reference uncertainty."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from analyze_quality_baseline import overlap_reference, receiver_reference


def estimate(folder, spp, size):
    rows = list(csv.DictReader((folder / "timings.csv").open()))
    def image(index):
        a = np.fromfile(folder / f"frame-{index}.rgba", np.float32)
        a = a.reshape(size, size, 4).astype(np.float64)
        if not np.isfinite(a).all():
            raise RuntimeError(f"nonfinite frame: {folder} {index}")
        return a
    base = int(rows[1]["samples"])
    end = 1 + spp // 32
    count = int(rows[end]["samples"])
    if count - base != spp:
        raise RuntimeError(f"unexpected sample counts: {folder}")
    result = (count * image(end) - base * image(1)) / spp
    gpu = sum(float(row["gpu_ms"]) for row in rows[2:end+1]) / 1000
    wall = sum(float(row["wall_ms"]) for row in rows[2:end+1]) / 1000
    return result, gpu, wall


def consistency(a, b):
    # Arrays of replicate RGB image means, not per-pixel iid assumptions.
    delta = a.mean(0) - b.mean(0)
    se = np.sqrt(a.var(0, ddof=1)/len(a) + b.var(0, ddof=1)/len(b))
    tolerance = 6 * se + 5e-5
    return {"difference": delta.tolist(), "replicate_standard_error": se.tolist(),
            "tolerance": tolerance.tolist(), "pass": bool(np.all(np.abs(delta) <= tolerance))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    root = args.output
    plan = json.loads((root / "plan.json").read_text())
    size = plan["size"]
    report = {"caveat": "Seed-randomized replicate estimates; heterogeneous references have measured residual noise, not certified convergence.", "scenes": {}}
    failures = []
    for scene in plan["scenes"]:
        refs = np.stack([estimate(root / scene / f"reference-{i}", plan["reference"]["spp"], size)[0]
                         for i in range(len(plan["reference"]["seeds"]))])
        reference = refs.mean(0)
        reference_variance = refs.var(0, ddof=1) / len(refs)
        power = np.square(reference[..., :3]).sum()
        reference_noise = reference_variance[..., :3].sum()
        result = {"reference_mean_rgb": reference[..., :3].mean((0, 1)).tolist(),
                  "estimated_reference_noise_nrms": float(np.sqrt(reference_noise / power)),
                  "prefixes": {}}
        for spp in plan["analysis_contract"]["prefix_spp"]:
            runs = [estimate(root / scene / f"candidate-{i}", spp, size)
                    for i in range(len(plan["candidate"]["seeds"]))]
            images = np.stack([run[0] for run in runs])
            mse = np.square(images[..., :3] - reference[..., :3]).sum((1, 2, 3))
            corrected = float(mse.mean() - reference_noise)
            gpu = float(np.mean([run[1] for run in runs]))
            wall = float(np.mean([run[2] for run in runs]))
            result["prefixes"][str(spp)] = {
                "raw_reference_difference_nrms": float(np.sqrt(mse.mean()/power)),
                "reference_noise_subtracted_nrms": float(np.sqrt(corrected/power)) if corrected >= 0 else None,
                "signed_corrected_normalized_mse": corrected/power,
                "replicate_mse_standard_error_conditioned_on_reference": float(mse.std(ddof=1)/np.sqrt(len(mse))/power),
                "mean_gpu_seconds": gpu, "mean_wall_seconds": wall,
                "corrected_mse_times_gpu_seconds": corrected/power*gpu,
                "corrected_mse_times_wall_seconds": corrected/power*wall,
            }
        a = images[..., :3].mean((1, 2))
        b = refs[..., :3].mean((1, 2))
        result["candidate_reference_mean_consistency"] = consistency(a, b)
        if not result["candidate_reference_mean_consistency"]["pass"]:
            failures.append(scene + ": candidate/reference means")
        if scene == "overlap" or scene.startswith("receiver-"):
            analytic = overlap_reference(size) if scene == "overlap" else receiver_reference(size, 20)
            target = analytic[..., :3].mean((0, 1))
            # An exact integral has zero replicate variance.
            gate = consistency(b, np.stack([target, target]))
            result["analytic_reference_mean_consistency"] = gate
            result["analytic_candidate_nrms"] = float(np.sqrt(
                np.square(images[..., :3] - analytic[..., :3]).sum((1, 2, 3)).mean()
                / np.square(analytic[..., :3]).sum()))
            if not gate["pass"]:
                failures.append(scene + ": analytic mean")
        report["scenes"][scene] = result
    report["failed_predeclared_mean_checks"] = failures
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for scene, result in report["scenes"].items():
        final = result["prefixes"][str(plan["candidate"]["spp"])]
        print(scene, "candidate NRMS", final["reference_noise_subtracted_nrms"],
              "reference noise", result["estimated_reference_noise_nrms"],
              "gpu seconds", final["mean_gpu_seconds"])
    if failures:
        raise RuntimeError("predeclared mean consistency failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
