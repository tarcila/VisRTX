#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-work baseline diagnostics. Held-out estimates are not convergence goldens."""
import argparse
import csv
import collections
import json
from pathlib import Path
import numpy as np


def image(folder, index, size=128):
    data = np.fromfile(folder / f"frame-{index}.rgba", np.float32).reshape(size, size, 4).astype(np.float64)
    if not np.isfinite(data).all():
        raise RuntimeError(f"nonfinite image: {folder} {index}")
    return data


def blocks(folder):
    rows = list(csv.DictReader((folder / "timings.csv").open()))
    counts = [int(r["samples"]) for r in rows]
    # Both warm launches are excluded. Reconstruct subsequent disjoint batches
    # from the actual public accumulation counts, not nominal frame indices.
    out = []
    previous = image(folder, 1)
    for i in range(2, 18):
        current = image(folder, i)
        if counts[i] - counts[i-1] != 32:
            raise RuntimeError(f"unexpected accumulation reset/count: {folder}")
        out.append((counts[i]*current-counts[i-1]*previous)/32)
        previous = current
    return np.stack(out)


def metrics(a, b):
    delta = a[..., :3]-b[..., :3]
    den = np.abs(b[..., :3]).sum()
    power = np.square(b[..., :3]).sum()
    return dict(relative_l1=float(np.abs(delta).sum()/den) if den else None,
                normalized_rms=float(np.sqrt(np.square(delta).sum()/power)) if power else None,
                alpha_max=float(np.abs(a[..., 3]-b[..., 3]).max()),
                rgb_mean_a=a[..., :3].mean(axis=(0, 1)).tolist(),
                rgb_mean_b=b[..., :3].mean(axis=(0, 1)).tolist())


def overlap_reference(size):
    a = -np.log(np.array([.7, .5, .9], np.float32).astype(float))
    b = -np.log(np.array([.6, .8, .4], np.float32).astype(float))
    ja = np.array([.2, .4, .1], np.float32).astype(float)
    jb = np.array([.3, .05, .2], np.float32).astype(float)
    t = np.ones(3)
    color = np.zeros(3)
    for sigma, source, length in ((a, ja, .5), (a+b, ja+jb, 1.5), (b, jb, .5)):
        color += t*source*(-np.expm1(-sigma*length))/sigma
        t *= np.exp(-sigma*length)
    color += .05*t
    rgba = np.r_[color, 1-np.dot(t, [.2126, .7152, .0722])]
    return np.broadcast_to(rgba, (size, size, 4)).copy()


def receiver_reference(size, order):
    # Independent nonsingular volume integral, Lambertian K=rho*cos/pi.
    # Integrate over each pixel footprint with a 2x2 Gauss rule too.
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = 1 + .25*nodes
    y = .25*nodes
    z = 1 + .25*nodes
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    ww = np.einsum("i,j,k->ijk", weights, weights, weights)*(.25**3)
    xx, yy, zz, ww = [v.ravel() for v in (xx, yy, zz, ww)]
    px, py = np.meshgrid((np.arange(size)+.5)/size-.5,
                         (np.arange(size)+.5)/size-.5)
    px, py = px.ravel(), py.ravel()
    result = np.zeros(size*size)
    for start in range(0, len(px), 128):
        end = start+128
        value = np.zeros(len(px[start:end]))
        for dx in (-1/np.sqrt(3), 1/np.sqrt(3)):
            for dy in (-1/np.sqrt(3), 1/np.sqrt(3)):
                r2 = ((xx[:, None]-px[None, start:end]-dx/(2*size))**2
                      +(yy[:, None]-py[None, start:end]-dy/(2*size))**2+zz[:, None]**2)
                value += np.sum((ww*zz)[:, None]/(r2*np.sqrt(r2)), axis=0)/4
        result[start:end] = (.7/np.pi)*value
    rgb = np.repeat(result.reshape(size, size, 1), 3, axis=2)
    return np.concatenate((rgb, np.ones((size, size, 1))), axis=2)


def profile_metrics(root, scene):
    folder = root / f"profile-{scene}"
    details = {r['Metric Name']: r['Metric Value'] for r in csv.DictReader((folder / 'details.csv').open())}
    raw = list(csv.DictReader((root / f'fp64-{scene}' / 'metrics.csv').open()))[-1]
    functions = collections.Counter()
    doubles = local = total = 0
    name = ''
    header = []
    for row in csv.reader((folder / 'source.csv').open()):
        if row and row[0] == 'Kernel Name':
            name = row[1]
        elif row and row[0] == 'Address':
            header = row
        elif row and row[0].startswith('0x'):
            values = dict(zip(header, row))
            samples = int(values.get('# Samples', '0') or 0)
            functions[name] += samples
            total += samples
            tokens = values['Source'].strip().split()
            op = tokens[1] if tokens[0].startswith('@') else tokens[0]
            if op in ('DADD', 'DMUL', 'DFMA'):
                doubles += samples
            if op.startswith(('LDL', 'STL')):
                local += int(values.get('Predicated-On Thread Instructions Executed', '0') or 0)
    return dict(fp64_active_percent=float(raw['sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed']),
                achieved_occupancy_percent=float(details['Achieved Occupancy']),
                dram_percent=float(details['DRAM Throughput']),
                sampled_double_instruction_percent=100*doubles/total,
                local_load_store_thread_instructions=local,
                sampled_function_percent={name:100*count/total for name, count in functions.most_common()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    report = {"timing": {}, "convergence": {}, "analytic": {}, "depth": {}}
    for folder in sorted(root.glob("perf-*")):
        if not folder.is_dir():
            continue
        rows = list(csv.DictReader((folder / "timings.csv").open()))
        times = np.array([float(r["gpu_ms"]) for r in rows[2:]])
        final = image(folder, 11, 640)
        report["timing"][folder.name] = dict(last_image_rgb_max=float(final[..., :3].max()),
            gpu_ms_median=float(np.median(times)),
            gpu_ms_min=float(times.min()), gpu_ms_max=float(times.max()),
            steady_gpu_used_mib=max(int(r["gpu_used_bytes"]) for r in rows[2:])/2**20,
            cold_gpu_ms=float(rows[0]["gpu_ms"]), cold_wall_ms=float(rows[0]["wall_ms"]))
    sampled = {}
    for scene in ("cloud", "plume", "support", "overlap", "receiver-matte", "receiver-native"):
        b = blocks(root / f"convergence-{scene}")
        sampled[scene] = b
        reference = b[8:].mean(axis=0)
        power = np.square(reference[..., :3]).sum()
        noise = np.var(b[8:, ..., :3], axis=0, ddof=1).sum()/8
        report["convergence"][scene] = dict(
            prefix_vs_disjoint_256spp_tail={str(32*n):metrics(b[:n].mean(axis=0), reference) for n in (1, 2, 4, 8)},
            estimated_tail_noise_nrms=float(np.sqrt(noise/power)) if power else None,
            full_512spp_rgb_max=float(b.mean(axis=0)[..., :3].max()),
            blocks_rgb_means=b[..., :3].mean(axis=(1, 2)).tolist())
        mean = b.mean(axis=0)
        np.asarray(mean, np.float32).tofile(root / f"{scene}-512spp.rgba")
    overlap = overlap_reference(128)
    report["analytic"]["overlap"] = metrics(sampled["overlap"].mean(axis=0), overlap)
    overlap_means = sampled["overlap"][..., :3].mean(axis=(1, 2))
    report["analytic"]["overlap"]["block_mean_standard_error"] = (overlap_means.std(axis=0, ddof=1)/4).tolist()
    small = receiver_reference(128, 12)
    large = receiver_reference(128, 20)
    agreement = metrics(small, large)
    if agreement["relative_l1"] > 1e-6 or agreement["normalized_rms"] > 1e-6:
        raise RuntimeError("receiver reference not resolved")
    report["analytic"]["receiver_reference_refinement"] = agreement
    large.astype(np.float32).tofile(root / "receiver-reference.rgba")
    for scene in ("receiver-matte", "receiver-native"):
        report["analytic"][scene] = metrics(sampled[scene].mean(axis=0), large)
        expected_mean = large[..., :3].mean(axis=(0, 1))
        bmeans = sampled[scene][..., :3].mean(axis=(1, 2))
        report["analytic"][scene]["rgb_mean_bias"] = (bmeans.mean(axis=0)-expected_mean).tolist()
        report["analytic"][scene]["block_mean_standard_error"] = (bmeans.std(axis=0, ddof=1)/4).tolist()
    for scene in ("cloud", "plume"):
        main_mean = sampled[scene].mean(axis=0)
        for depth in (12, 64):
            other = blocks(root / f"depth-{scene}-{depth}").mean(axis=0)
            report["depth"][f"{scene}-{depth}-vs-32"] = metrics(other, main_mean)
    low = blocks(root / "support-preview-2").mean(axis=0)
    high = blocks(root / "support-preview-4").mean(axis=0)
    support = sampled["support"].mean(axis=0)
    report["support_crosscheck"] = dict(preview_refinement=metrics(low, high),
        quality_vs_preview=metrics(support, high),
        bright_zero_alpha_pixels=int(np.count_nonzero((support[..., :3].max(axis=2)>.06)
                                                     & (np.abs(support[..., 3])<1e-7))))
    available = [scene for scene in ('cloud', 'plume')
                 if (root / f'profile-{scene}' / 'source.csv').exists()
                 and (root / f'fp64-{scene}' / 'metrics.csv').exists()]
    report['profiles'] = {scene:profile_metrics(root, scene) for scene in available}
    report['profiles_missing'] = [scene for scene in ('cloud', 'plume') if scene not in available]
    (root / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    for name, row in report["timing"].items():
        print(name, row["gpu_ms_median"], "ms", row["steady_gpu_used_mib"], "MiB GPU-wide")
    for name, row in report["convergence"].items():
        print(name, "256 vs 256 NRMS", row["prefix_vs_disjoint_256spp_tail"]["256"]["normalized_rms"],
              "estimated tail noise", row["estimated_tail_noise_nrms"])
    for section in ("analytic", "depth", "support_crosscheck"):
        print(section, json.dumps(report[section]))

if __name__ == "__main__":
    main()
