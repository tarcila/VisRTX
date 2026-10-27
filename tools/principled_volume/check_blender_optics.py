"""Run via Blender: --python check_blender_optics.py -- writer output_directory.

Tests the actual reference shader against homogeneous analytic expectations.
No VisRTX device is loaded. A narrow orthographic beam traverses exactly 2 m.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from blender_reference import build, render_metadata
from generate import presets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('writer',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--compute-device',choices=['CPU','CUDA','OPTIX'],default='CPU')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    writer, output = args.writer.resolve(), args.output.resolve()
    if (output/'results.json').exists():
        raise ValueError('Choose a fresh checks directory')
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((HERE/'diagnostics.json').read_text())['cases']
    results = []
    for case in cases:
        directory = output/case['name']
        directory.mkdir(exist_ok=True)
        p = dict(presets('cloud')['isotropic'])
        p.update(case['parameters'])
        p.pop('density.field')
        for name, data, channels in [('density',np.ones((16,16,16),dtype=np.float32),1),
                ('reference_j',np.broadcast_to(p.get('emission',[0]*3),(16,16,16,3)).astype(np.float32),3)]:
            data.tofile(directory/(name+'.raw'))
            subprocess.run([str(writer),str(directory/(name+'.raw')),str(directory/(name+'.nvdb')),
                            name,str(channels),'16','16','16',str(2/16),'-1','-1','0','cell'],check=True)
        subprocess.run([str(writer),'--merge-vdb-cell',str(directory/'reference.vdb'),
                        str(directory/'density.nvdb'),str(directory/'reference_j.nvdb')],check=True)
        assert len(set(case['background'])) == 1
        cfg = dict(scene='analytic',preset=case['name'],parameters=p,bounds=[[-1,-1,0],[1,1,2]],
                   meshes=[],environment=dict(file='',scale=case['background'][0]),
                   camera=dict(position=[0,-4,1],target=[0,0,1],up=[0,0,1],fovy_degrees=1),
                   render=dict(samples=128,width=32,height=32,volume_bounces=12,
                               view_transform='AgX',exposure=0,compute_device=args.compute_device,
                               volume_precision='FULL',volume_clipping=0.,volume_interpolation='LINEAR'),status='Blender analytic check',
                   reference=dict(vdb=str(directory/'reference.vdb'),has_emission=True,limitation='Homogeneous input'))
        scene = build(cfg,directory/'check.blend')
        scene.camera.data.type = 'ORTHO'
        scene.camera.data.ortho_scale = .1
        scene.cycles.volume_step_rate = .1
        if 'expected_alpha' in case:
            scene.render.film_transparent = True
        bpy.ops.wm.save_as_mainfile(filepath=str(directory/'check.blend'))
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(directory/'check.exr'),check_existing=False)
        pixels = np.array(image.pixels[:]).reshape(-1,4)
        mean = pixels[:,:3].mean(axis=0)
        expected = np.array(case['expected_radiance'])
        rgb_passed = bool(np.allclose(mean,expected,rtol=.02,atol=.002))
        result = dict(case=case['name'],expected=expected.tolist(),measured=mean.tolist(),
                      rgb_passed=rgb_passed,passed=rgb_passed,render=render_metadata(scene))
        if 'expected_alpha' in case:
            alpha = float(pixels[:,3].mean())
            alpha_passed = bool(abs(alpha-case['expected_alpha']) <= .002)
            result.update(expected_alpha=case['expected_alpha'],measured_alpha=alpha,
                          alpha_passed=alpha_passed,passed=rgb_passed and alpha_passed)
        results.append(result)
        (output/'results.partial.json').write_text(json.dumps(results,indent=2)+'\n')
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    assert all(r['passed'] for r in results), 'Blender coefficient mapping failed'


if __name__ == '__main__':
    main()
