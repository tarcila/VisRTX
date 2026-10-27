# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent Decimal90 log oracle; no production polynomial or GPU imports."""
import argparse
from decimal import Decimal, localcontext
from pathlib import Path
import random
import struct


def generate():
    bits = {1, 2, 0x7fffff, 0x800000, 0x3f7fffff, 0x3f800000}
    # Every power-of-two boundary and its representable neighbors.
    for k in range(150):
        b = struct.unpack('<I', struct.pack('<f', 2.0 ** -k))[0]
        bits.update(x for x in (b - 1, b, b + 1) if 0 < x <= 0x3f800000)
    rng = random.Random(45821)
    bits.update(rng.randrange(1, 0x3f800001) for _ in range(256))
    lines = ['# float32 bits; -ln(C), Decimal90 rounded to binary64\n']
    with localcontext() as ctx:
        ctx.prec = 90
        for b in sorted(bits):
            value = struct.unpack('<f', struct.pack('<I', b))[0]
            exact = Decimal.from_float(value)
            expected = float(-exact.ln())
            lines.append(f'{b:08x} {expected:.17g}\n')
    return ''.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    target = Path(__file__).with_name('log-oracle.txt')
    result = generate()
    if args.write:
        target.write_text(result)
    elif target.read_text() != result:
        raise SystemExit('log oracle differs; regenerate explicitly and review')
    else:
        print('Decimal90 log oracle matches')
