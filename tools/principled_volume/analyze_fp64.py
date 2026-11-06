#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Separate static OptiX warnings from executed SASS and sampled PC attribution."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re


def warnings(path):
    locations = Counter()
    total = 0
    for line in path.read_text().splitlines():
        if 'Warning: Program is doing double precision' not in line:
            continue
        total += 1
        match = re.search(r' (/.*?):(\d+):(\d+)', line)
        locations[f'{match[1]}:{match[2]}' if match else '(unresolved/truncated)'] += 1
    files = Counter()
    for name, count in locations.items():
        files[name.rsplit(':', 1)[0] if name.startswith('/') else name] += count
    return dict(occurrences=total, files=dict(files.most_common()),
                locations=dict(locations.most_common()))


def source_locations(path):
    """First approximate CUDA location per address; never double-count inlining."""
    result = {}
    filename, line = '(unmapped)', '?'
    if not path.exists():
        return result
    for row in csv.reader(path.open()):
        if not row:
            continue
        if row[0] == 'File Path':
            filename, line = row[1], '?'
        elif row[0].isdigit():
            line = row[0]
        elif len(row) > 3 and row[2].startswith('0x'):
            result.setdefault(row[2], (filename, line))
    return result


def sass(path, locations):
    functions, files, lines = (defaultdict(Counter) for _ in range(3))
    function, header = '(unknown)', []
    seen = set()
    for row in csv.reader(path.open()):
        if not row:
            continue
        if row[0] == 'Kernel Name':
            function = row[1]  # Nsight labels device functions this way too.
        elif row[0] == 'Address':
            header = row
        elif row[0].startswith('0x') and row[0] not in seen:
            seen.add(row[0])
            record = dict(zip(header, row))
            tokens = record['Source'].split()
            op = (tokens[1] if tokens[0].startswith('@') else tokens[0]).split('.')[0]
            executed = int(record['Predicated-On Thread Instructions Executed'])
            samples = int(record['# Samples'])
            value = Counter(pc_samples=samples, thread_instructions=executed)
            if op in ('DADD', 'DMUL', 'DFMA'):
                value[op] = executed
                value['fp64_arithmetic_instructions'] = executed
                value['fp64_pc_samples'] = samples
            if record.get('Address Space') == 'Local':
                value[f"local_{record['Access Operation'].lower()}_bytes"] = (
                    executed * int(record['Access Size']) // 8)
                value['local_pc_samples'] = samples
            filename, line = locations.get(row[0], ('(unmapped)', '?'))
            functions[function].update(value)
            files[filename].update(value)
            lines[f'{filename}:{line}'].update(value)
    total = sum(functions.values(), Counter())
    def ordered(values):
        return dict(sorted(values.items(), key=lambda item: item[1]['pc_samples'], reverse=True))
    return dict(totals=dict(total), functions=ordered(functions), files=ordered(files),
                lines=ordered(lines), unique_sass_addresses=len(seen),
                mapped_addresses=len(seen & locations.keys()))


def analyze(root):
    report = {'caveats': [
        'Warnings count compiled diagnostics, not runtime cost.',
        'PC samples are not elapsed-time partitions.',
        'Instruction counts use per-address predicated-on thread counters; aggregate OptiX opcode metrics may report zero.',
        'Local bytes are logical accesses, not DRAM traffic or proof of register-pressure spills.',
        'CUDA line attribution is approximate; the first mapping per address is retained.',
        'Profiler replay durations are diagnostic; use unprofiled paired runs to accept changes.'
    ], 'compile_logs': {}, 'profiles': {}}
    for path in sorted(root.glob('compile-*/run.log')):
        report['compile_logs'][path.parent.name] = warnings(path)
    for path in sorted(root.glob('profile-*/details.csv')):
        folder = path.parent
        details = {r['Metric Name']: {'value': r['Metric Value'], 'unit': r['Metric Unit']}
                   for r in csv.DictReader(path.open())}
        result = sass(folder/'source.csv', source_locations(folder/'correlated.csv'))
        result['metrics'] = details
        report['profiles'][folder.name] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, profile in result['profiles'].items():
        metrics = profile['metrics']
        print(name, metrics['Duration'],
              'FP64 arithmetic instructions', profile['totals'].get('fp64_arithmetic_instructions', 0),
              'source mappings', profile['mapped_addresses'], '/', profile['unique_sass_addresses'])


if __name__ == '__main__':
    main()
