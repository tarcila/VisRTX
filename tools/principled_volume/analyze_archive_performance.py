#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Analyze archive captures, paired randomizations, and canonical controls."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from analyze_quality_baseline import blocks, metrics


def rows(folder):
    return [line.split(',') for line in (folder/'run.log').read_text().splitlines()
            if line.startswith('CAPTURE,')]


def image(folder, index):
    a = np.fromfile(folder/f'frame-{index}.rgba', np.float32).reshape(128, 128, 4).astype(float)
    if not np.isfinite(a).all():
        raise RuntimeError(f'nonfinite HDR: {folder}')
    return a


def accumulated(folder):
    r = rows(folder)
    first, last = int(r[1][2]), int(r[17][2])
    if last-first != 512:
        raise RuntimeError(f'unexpected accumulation counts: {folder}')
    return (last*image(folder, 17)-first*image(folder, 1))/(last-first)


def timing(folder):
    if (folder/'timings.csv').exists():
        r = list(csv.DictReader((folder/'timings.csv').open()))[2:]
        gpu, wall, memory = ([float(x[key]) for x in r] for key in ('gpu_ms', 'wall_ms', 'gpu_used_bytes'))
    else:
        r = rows(folder)[2:]
        gpu, wall, memory = ([float(x[i]) for x in r] for i in (3, 4, 5))
    return {'gpu_median_ms': float(np.median(gpu)), 'gpu_max_ms': max(gpu),
            'wall_median_ms': float(np.median(wall)), 'wall_max_ms': max(wall),
            'gpu_wide_memory_MiB': max(memory)/2**20}


def profile(folder):
    details = {r['Metric Name']: r['Metric Value'] for r in csv.DictReader((folder/'details.csv').open())}
    functions = collections.Counter()
    name, header = '', []
    for row in csv.reader((folder/'source.csv').open()):
        if row and row[0] == 'Kernel Name':
            name = row[1]
        elif row and row[0] == 'Address':
            header = row
        elif row and row[0].startswith('0x'):
            functions[name] += int(dict(zip(header, row)).get('# Samples', '0') or 0)
    total = sum(functions.values())
    return {'details': details, 'sampled_function_percent': {n: 100*v/total for n, v in functions.most_common(12)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    root = args.root
    plan = json.loads((root/'qualification-plan.json').read_text())
    result = {'timing': {}, 'profiles': {}, 'fidelity': {}}
    for version in ('baseline', 'final'):
        result['timing'][version] = {}
        for pattern in ('qualified-*', 'repeat-*', 'perf-*', 'memory-check', 'viewer-*', 'diagnostic-*'):
            for folder in sorted((root/version).glob(pattern)):
                if folder.is_dir():
                    result['timing'][version][folder.name] = timing(folder)
        for renderer in ('interactive', 'quality'):
            result['profiles'][f'{version}-{renderer}'] = profile(root/version/f'{renderer}-ncu')
    a, b = [np.stack([accumulated(root/v/f'hdr-quality-{i}') for i in range(len(plan['seeds']))])
            for v in ('baseline', 'final')]
    delta = (b-a)[..., :3].mean((1, 2))
    se = delta.std(0, ddof=1)/np.sqrt(len(delta))
    gate = 6*se+5e-5
    variance = [v[..., :3].var(0, ddof=1).sum() for v in (a, b)]
    costs = [np.mean([sum(float(x[3]) for x in rows(root/v/f'hdr-quality-{i}')[2:])/1000
                     for i in range(len(plan['seeds']))]) for v in ('baseline', 'final')]
    result['fidelity']['quality'] = {'mean_delta': delta.mean(0).tolist(),
        'six_se_plus_absolute_gate': gate.tolist(), 'pass': bool(np.all(np.abs(delta.mean(0)) <= gate)),
        'mean_images': metrics(b.mean(0), a.mean(0)),
        'empirical_replicate_variance_ratio': variance[1]/variance[0],
        'mean_512spp_gpu_seconds': costs,
        'empirical_variance_cost_ratio': variance[1]*costs[1]/(variance[0]*costs[0]),
        'caveat': 'Eight paired pseudo-randomizations; neither mean image is a convergence golden.'}
    preview = metrics(image(root/'final/hdr-interactive-0', 7), image(root/'baseline/hdr-interactive-0', 7))
    preview['pass'] = all(preview[key] <= value for key, value in plan['preview_gate'].items())
    result['fidelity']['interactive'] = preview
    for renderer in ('interactive', 'quality'):
        for scene in ('cloud', 'plume', 'support', 'overlap', 'receiver-matte', 'receiver-native'):
            before = blocks(root/'baseline'/f'control-{renderer}-{scene}').mean(0)
            after = blocks(root/'final'/f'control-{renderer}-{scene}').mean(0)
            result['fidelity'][f'{renderer}-{scene}'] = metrics(after, before)
    (root/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'timing': result['timing'], 'fidelity': result['fidelity']}, indent=2))
    if not preview['pass'] or not result['fidelity']['quality']['pass']:
        raise RuntimeError('predeclared archive fidelity gate failed')


if __name__ == '__main__':
    main()
