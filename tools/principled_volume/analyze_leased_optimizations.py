#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Analyze the clock-leased, exact-graph/preview-point optimization campaign."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_archive_performance import image, profile, timing
from analyze_quality_baseline import blocks, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root/'qualification-plan.json').read_text())
    baseline = Path(plan['baseline'])
    report = {'timing': {}, 'profiles': {}, 'fidelity': {}, 'clocks': {}, 'leases': {}}
    for version in ('baseline', 'final'):
        report['timing'][version] = {}
        for pattern in ('perf-*', 'motion-*', 'viewer-*', 'canonical-perf-*'):
            for folder in sorted((root/version).glob(pattern)):
                report['timing'][version][folder.name] = timing(folder)
        for renderer in ('interactive', 'quality'):
            report['profiles'][f'{version}-{renderer}'] = profile(root/version/f'profile-{renderer}')
    count_exact = count_preview = 0
    for renderer in ('interactive', 'quality'):
        seeds = plan['seeds'] if renderer == 'quality' else [0]
        for index, _ in enumerate(seeds):
            folder = f'hdr-{renderer}-{index}'
            for frame in range(2, 18 if renderer == 'quality' else 8):
                a, b = baseline/folder/f'frame-{frame}.rgba', root/'final'/folder/f'frame-{frame}.rgba'
                if renderer == 'quality':
                    if a.read_bytes() != b.read_bytes():
                        raise RuntimeError(f'Quality exact-image gate: {folder} frame {frame}')
                    count_exact += 1
                else:
                    result = metrics(image(b.parent, frame), image(a.parent, frame))
                    for key, gate in plan['interactive_gates'].items():
                        if result[key] is None or result[key] > gate:
                            raise RuntimeError(f'Interactive gate: {folder} frame {frame} {key}')
                    report['fidelity'][f'{folder}-{frame}'] = result
                    count_preview += 1
        for scene in ('cloud', 'plume', 'support', 'overlap', 'receiver-matte', 'receiver-native'):
            folder = f'control-{renderer}-{scene}'
            a, b = baseline/folder, root/'final'/folder
            for frame in range(2, 18):
                if renderer == 'quality':
                    if (a/f'frame-{frame}.rgba').read_bytes() != (b/f'frame-{frame}.rgba').read_bytes():
                        raise RuntimeError(f'Quality exact-image gate: {folder} frame {frame}')
                    count_exact += 1
                else:
                    before, after = image(a, frame), image(b, frame)
                    result = metrics(after, before)
                    if result['normalized_rms'] is None:
                        if not np.array_equal(before, after):
                            raise RuntimeError(f'Black-image gate: {folder}')
                    elif any(result[key] > gate for key, gate in plan['interactive_gates'].items()):
                        raise RuntimeError(f'Interactive gate: {folder} frame {frame}')
                    count_preview += 1
            report['fidelity'][folder] = metrics(blocks(b).mean(0), blocks(a).mean(0))
    report['quality_exact_frames'] = count_exact
    report['interactive_qualified_frames'] = count_preview
    for phase in ('perf', 'motion', 'hdr', 'controls', 'profile'):
        rows = list(csv.DictReader((root/f'{phase}-telemetry.csv').open()))
        active = [row for row in rows if int(row[' utilization.gpu [%]'].strip().split()[0]) > 0]
        report['clocks'][phase] = {key.strip(): sorted({row[key].strip() for row in active})
                                  for key in rows[0] if 'clock' in key}
    for phase in ('perf', 'motion', 'hdr', 'controls', 'profile', 'canonical-perf'):
        report['leases'][phase] = (root/f'{phase}-lease-success.txt').read_text().strip()
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'timing': report['timing'], 'quality_exact_frames': count_exact,
                      'interactive_qualified_frames': count_preview, 'clocks': report['clocks']}, indent=2))


if __name__ == '__main__':
    main()
