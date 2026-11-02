#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Extract a viewer-state's embedded scene through Vela's canonical text tools.

Never modifies the input. Large archives need substantial temporary disk space.
Requires vsrPrint and vsrConvert from a compatible Vela installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('refusing to replace existing extraction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inside = done = envelope = False
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        text = Path(tmp)/'scene.txt'
        process = subprocess.Popen(['vsrPrint', str(args.input)], stdout=subprocess.PIPE, text=True)
        try:
            with text.open('w') as out:
                out.write('vsr-text 1\n')
                for line in process.stdout:
                    if not inside and not done:
                        envelope |= line.strip() == 'fileType = "application-state"'
                        if line == '  scene {\n' and envelope:
                            inside = True
                            continue
                    if inside and line == '  }\n':
                        inside, done = False, True
                        continue
                    if inside:
                        if not line.startswith('    '):
                            raise RuntimeError('unexpected canonical scene indentation')
                        out.write(line[4:])
            if process.wait() != 0 or not done:
                raise RuntimeError('failed to extract embedded scene')
            subprocess.run(['vsrConvert', '--binary', str(text), str(args.output)], check=True)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    args.output.with_suffix('.manifest.json').write_text(json.dumps({
        'input': str(args.input.resolve()), 'input_sha256': digest(args.input),
        'extracted_scene_sha256': digest(args.output),
        'method': 'canonical Vela text extraction of archives.scene; original untouched',
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
