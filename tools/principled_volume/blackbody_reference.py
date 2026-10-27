#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent direct Planck/CIE quadrature oracle (no production LUT import).
Generated numeric output is CC-BY-SA-4.0; see the data attribution beside it.
Run --write to deliberately regenerate, otherwise verify the checked-in oracle.
"""
import argparse
import bisect
import csv
import hashlib
import math
import random
from pathlib import Path
import struct

DATA = Path(__file__).resolve().parents[2] / 'devices/rtx/device/gpu/blackbody/data'
M = ((3.2404542,-1.5371385,-0.4985314),(-.9692660,1.8760108,.0415560),(.0556434,-.2040259,1.0572252))


def read(name, digest):
    p=DATA/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    return [tuple(map(float,r)) for r in csv.reader(p.read_text().splitlines()) if r]


def lerp(rows, x, channel):
    i=max(0,min(len(rows)-2,bisect.bisect_right([r[0] for r in rows],x)-1))
    f=(x-rows[i][0])/(rows[i+1][0]-rows[i][0])
    return (1-f)*rows[i][channel]+f*rows[i+1][channel]


def generate():
    observer=read('CIE_xyz_1931_2deg.csv','fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1')
    daylight=read('CIE_std_illum_D65.csv','e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb')
    wavelengths=[r for r in observer if 360<=r[0]<=830]
    d=[lerp(daylight,r[0],1) for r in wavelengths]
    k=sum((d[i]+d[i+1])*.5 for i in range(len(d)-1))/sum((d[i]*wavelengths[i][2]+d[i+1]*wavelengths[i+1][2])*.5 for i in range(len(d)-1))
    det=M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0])
    yw=((M[1][2]*M[2][0]-M[1][0]*M[2][2])/det,(M[0][0]*M[2][2]-M[0][2]*M[2][0])/det,(M[0][2]*M[1][0]-M[0][0]*M[1][2])/det)
    rows=[]
    for i in range(1881):
        w=360+i*.25;lam=w*1e-9
        a=6.62607015e-34*299792458/(lam*1.380649e-23)
        b=2*6.62607015e-34*299792458**2/lam**5*1e-9
        weight=.25*(.5 if i in (0,1880) else 1)*k
        rows.append((a,b,[weight*lerp(observer,w,c) for c in (1,2,3)]))
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    temperatures={f32(math.exp(math.log(300)+i*math.log(100000/300)/256)) for i in range(257)}
    # Probe every knot and quarter-cell, plus a reproducible independent sweep.
    temperatures.update(f32(math.exp(math.log(300)+i*math.log(100000/300)/508)) for i in range(509))
    rng=random.Random(80319)
    temperatures.update(f32(math.exp(rng.uniform(math.log(300),math.log(100000)))) for _ in range(512))
    temperatures.update(map(f32,(0,1e-45,1,25,30,50,100,200,250,299,300,301,1000,2000,3000,6500,100000,100001,1e6,1e12,1e33,3.4028234663852886e38)))
    for t in (300.,100000.):
        bits=struct.unpack('<I',struct.pack('<f',t))[0]
        temperatures.update(struct.unpack('<f',struct.pack('<I',bits+d))[0] for d in (-1,1))
    output=['# SPDX-License-Identifier: CC-BY-SA-4.0\n','# Direct 0.25nm quadrature; T R G B Y; K='+repr(k)+'\n']
    for t in sorted(temperatures):
        xyz=[0.,0.,0.]
        if t>0:
            for a,b,weights in rows:
                x=a/t;value=b*math.exp(-x)/(-math.expm1(-x))
                for c in range(3):xyz[c]+=weights[c]*value
        rgb=[sum(M[c][j]*xyz[j] for j in range(3)) for c in range(3)]
        if min(rgb)<0:
            rgb=[max(0,x) for x in rgb];y=sum(x*w for x,w in zip(rgb,yw))
            # Form the dimensionless scale first: x*Y underflows for cold
            # tails even when the projected RGB itself is representable.
            scale=xyz[1]/y if y>0 else 0
            rgb=[x*scale for x in rgb]
        output.append(' '.join(format(x,'.17g') for x in (t,*rgb,xyz[1]))+'\n')
    return ''.join(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    expected=generate();path=DATA/'reference.txt'
    if args.write:path.write_text(expected)
    elif path.read_text()!=expected:raise SystemExit('blackbody oracle differs')
    else:print('direct spectral blackbody oracle matches')
