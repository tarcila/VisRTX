#!/usr/bin/env python3
"""Build one shared scene description, VSR archive, and Blender coefficient asset."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from reference_optics import ROOT, blackbody, coefficients_per_density, thermal_lookup
from generate import cell_coordinates

HERE = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def lua(value):
    if isinstance(value, dict):
        return '{'+','.join('['+json.dumps(k, ensure_ascii=False)+']='+lua(v) for k, v in value.items())+'}'
    if isinstance(value, list):
        return '{'+','.join(map(lua, value))+'}'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    # Optical numbers MUST be Lua floats, even 0 and 1 (VSR preserves types).
    if isinstance(value, (int, float)):
        return repr(float(value))
    raise TypeError(type(value))


def quad(name, vertices, color):
    return dict(name=name, vertices=vertices, triangles=[[0, 1, 2], [0, 2, 3]], color=color)


def meshes(kind):
    if kind == 'cloud':
        return []
    if kind == 'emission':
        result = []
        for x in range(12):
            for z in range(10):
                a, b = -1.5+x*.25, z*.25-.25
                result.append(quad(f'checker_{x}_{z}',
                    [[a, .65, b], [a+.25, .65, b], [a+.25, .65, b+.25], [a, .65, b+.25]],
                    [.12 if (x+z)%2 else .8]*3))
        return result
    return [quad('floor', [[-2,-2,0], [2,-2,0], [2,2,0], [-2,2,0]], [.65]*3),
            quad('receiver', [[-.55,-.35,.05], [-.55,.35,.05],
                              [-.55,.35,1.25], [-.55,-.35,1.25]], [.7]*3)]


def sample(data, info, positions):
    """Domain membership first, then cell-center interpolation with clamped edges.

    Missing data_centering denotes legacy endpoint data for read-only historical
    comparisons. New scene preparation explicitly requires schema-2 cell data.
    """
    positions = tuple(np.asarray(p) for p in positions)
    centering = info.get('data_centering', 'node')
    if centering not in ('node', 'cell'):
        raise ValueError('Unknown field centering: '+str(centering))
    offset = .5 if centering == 'cell' else 0.
    dims = np.asarray(info['dimensions'])
    origin = np.asarray(info['origin'])
    domain = info.get('domain', [origin, origin + info['voxel_size'] *
                               (dims if centering == 'cell' else dims-1)])
    coords = [(p-o)/info['voxel_size'] - offset for p, o in zip(positions, origin)]
    shape = np.broadcast_shapes(*(p.shape for p in positions))
    valid = np.ones(shape, dtype=bool)
    lower, fraction = [], []
    for axis, (c, n) in enumerate(zip(coords, dims)):
        valid &= (positions[axis] >= domain[0][axis]) & (positions[axis] <= domain[1][axis])
        c = np.clip(c, 0, n-1)
        i = np.minimum(np.floor(c).astype(int), n-2)
        lower.append(i)
        fraction.append(c-i)
    result = np.zeros(shape+((3,) if data.ndim == 4 else ()))
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                bits = (x, y, z)
                weight = np.ones(shape)
                for bit, f in zip(bits, fraction):
                    weight *= f if bit else 1-f
                if data.ndim == 4:
                    weight = weight[..., None]
                result += weight * data[lower[0]+x, lower[1]+y, lower[2]+z]
    return result * (valid[..., None] if data.ndim == 4 else valid)


def validate_cell_manifest(manifest):
    """Do not silently relabel historical endpoint assets as full-cell data."""
    if manifest.get('schema_version') != 2:
        raise ValueError('Scene preparation requires schema-2 cell-centered assets; '
                         'regenerate into a fresh dataset directory')
    for name, info in manifest['fields'].items():
        origin = np.asarray(info['origin'])
        dims = np.asarray(info['dimensions'])
        h = info['voxel_size']
        sample_origin = np.asarray(info.get('sample_origin', []))
        if (sample_origin.shape != (3,) or info.get('data_centering') != 'cell' or info.get('filter') != 'linear'
                or dims.shape != (3,) or np.any(dims < 2) or not h > 0
                or not np.allclose(info['domain'][0], origin, rtol=0, atol=1e-12)
                or not np.allclose(info['domain'][1], origin+dims*h, rtol=0, atol=1e-12)
                or not np.allclose(sample_origin, origin+.5*h,
                                   rtol=0, atol=1e-12)):
            raise ValueError('Inconsistent full-cell sampling metadata: '+name)


def bake_reference(manifest, dataset, p, out, writer):
    validate_cell_manifest(manifest)
    density_info = manifest['fields']['density']
    dims = density_info['dimensions']
    origin, spacing = density_info['origin'], density_info['voxel_size']
    _, absorption = coefficients_per_density(p)
    emitting = p.get('blackbodyIntensity', 0) > 0 and np.any(absorption > 0)
    emitting |= 'emission.field' in p and np.any(np.array(p['emission']) > 0)
    merge = [str(dataset/density_info['file'])]
    if emitting:
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            tmp = Path(tmp)
            arrays = {}
            needed = {'density'}
            if p.get('blackbodyIntensity', 0) > 0:
                needed.add(p['temperature.field'])
            if 'emission.field' in p:
                needed.add(p['emission.field'])
            for name in needed:
                info = manifest['fields'][name]
                raw = tmp/(name+'.raw')
                subprocess.run([str(writer), '--dump', str(dataset/info['file']), str(raw)], check=True)
                shape = tuple(info['dimensions'])+((3,) if info['channels'] == 3 else ())
                arrays[name] = np.memmap(raw, dtype=np.float32, mode='r', shape=shape)
            table = None
            if p.get('blackbodyIntensity', 0) > 0:
                max_t = float(arrays[p['temperature.field']].max())*p['temperature']
                table = blackbody(np.arange(math.ceil(max_t)+2))
            jraw = tmp/'j.raw'
            j = np.memmap(jraw, dtype=np.float32, mode='w+', shape=tuple(dims)+(3,))
            y = cell_coordinates(origin[1], spacing, np.arange(dims[1]))[None,:,None]
            z = cell_coordinates(origin[2], spacing, np.arange(dims[2]))[None,None,:]
            for i in range(0, dims[0], 8):
                x = cell_coordinates(origin[0], spacing, np.arange(i, min(i+8, dims[0])))[:,None,None]
                positions = (x,y,z)
                shape = np.broadcast_shapes(x.shape, y.shape, z.shape)+(3,)
                value = np.zeros(shape)
                if table is not None:
                    name = p['temperature.field']
                    temperature = sample(arrays[name], manifest['fields'][name], positions)*p['temperature']
                    rho = np.asarray(arrays['density'][i:i+8])*p['density']
                    value += rho[...,None]*absorption*thermal_lookup(temperature, table)*p['blackbodyIntensity']*np.array(p['blackbodyTint'])
                if 'emission.field' in p:
                    name = p['emission.field']
                    value += sample(arrays[name], manifest['fields'][name], positions)*p['emission']
                if not np.isfinite(value).all() or (value < 0).any():
                    raise ValueError('Invalid derived emission')
                j[i:i+8] = value
            j.flush()
            subprocess.run([str(writer), str(jraw), str(out/'reference_j.nvdb'), 'reference_j',
                            '3', *map(str,dims), str(spacing), *map(str,origin), 'cell'], check=True)
            merge.append(str(out/'reference_j.nvdb'))
            del j
            arrays.clear()
    subprocess.run([str(writer), '--merge-vdb-cell', str(out/'reference.vdb'), *merge], check=True)
    return bool(emitting)


def prepare(args):
    dataset, out, writer = args.dataset.resolve(), args.output.resolve(), args.writer.resolve()
    m = json.loads((dataset/'manifest.json').read_text())
    validate_cell_manifest(m)
    expected_asset = 'cloud' if args.scene == 'cloud' else 'plume'
    if m['asset'] != expected_asset:
        raise ValueError(f'{args.scene} requires {expected_asset} data')
    if (out/'scene.json').exists():
        raise ValueError('Choose a fresh output directory (scene.json exists)')
    preset = args.preset or {'cloud':'forward', 'plume':'thermal', 'emission':'vacuum'}[args.scene]
    if args.scene == 'emission':
        if preset not in ('vacuum', 'smoke'):
            raise ValueError('Emission presets: vacuum, smoke')
        p = dict(m['presets']['explicit'])
        p['density'] = 0. if preset == 'vacuum' else 1.
    else:
        if preset not in m['presets']:
            raise ValueError('Unknown preset '+preset)
        p = dict(m['presets'][preset])
    out.mkdir(parents=True, exist_ok=True)
    bound_fields = {v for k,v in p.items() if k.endswith('.field')}
    field_info = {}
    for name in sorted(bound_fields):
        info = dict(m['fields'][name])
        path = dataset/info['file']
        if sha256(path) != info['sha256']:
            raise ValueError('Dataset hash mismatch: '+str(path))
        info['file'] = str(path)
        field_info[name] = info
    angle = math.radians(args.environment_rotation)
    environment = dict(file='', scale=0., direction=[math.cos(angle),math.sin(angle),0.], up=[0.,0.,1.])
    if args.scene == 'cloud':
        environment.update(file=str(args.hdri.resolve()), scale=1.)
        environment['sha256'] = sha256(args.hdri)
    elif args.scene == 'emission':
        environment['scale'] = .5
    s = dict(schema_version=2, scene=args.scene, preset=preset,
             provenance=dict(prepare_scene_sha256=sha256(__file__),
                 scene_builder_sha256=sha256(HERE/'build_scene.lua'),
                 reference_optics_sha256=sha256(HERE/'reference_optics.py'),
                 writer_sha256=sha256(writer),
                 vsr_executable=str(Path(shutil.which(args.vsr) or args.vsr).resolve()),
                 vsr_sha256=sha256(shutil.which(args.vsr) or args.vsr)),
             source_manifest=str(dataset/'manifest.json'), source_manifest_sha256=sha256(dataset/'manifest.json'),
             fields=field_info, parameters=p, bounds=m['bounds'], meshes=meshes(args.scene),
             environment=environment, camera=dict(position=[0.,-4.8,1.6], target=[0.,0.,1.],
             up=[0.,0.,1.], fovy_degrees=32.),
             render=dict(width=640,height=640,samples=args.samples,exposure=0. if args.scene != 'plume' else -3.,
                         renderer='quality', view_transform='AgX', volume_bounces=12,
                         max_bounces=16, volume_step_rate=.5, seed=42,
                         volume_precision='FULL', volume_clipping=0., volume_interpolation='LINEAR',
                         compute_device=getattr(args,'compute_device','CPU')),
             status='VSR authoring/archive test only; VisRTX principled transport not yet supported',
             reference=dict(vdb=str(out/'reference.vdb'),
                 emission_model='Calibrated Planck/CIE 360-830 nm, D65 unit-radiance normalization, 1 K LUT',
                 sampling='Full-cell centers; secondary OpenVDB is translated by half a cell and padded with a replicated one-voxel border; shader clips to explicit volume bounds.',
                 limitation='j is baked at density cell centers: interpolation of derived emission differs from evaluating nonlinear optics after field interpolation. Different-domain edges are resampled. Blender precision/clipping and convergence remain separate qualifications.',
                 cie_sha256={name:sha256(ROOT/name) for name in ('CIE_xyz_1931_2deg.csv','CIE_std_illum_D65.csv')}))
    s['reference']['has_emission'] = bake_reference(m, dataset, p, out, writer)
    s['reference']['vdb_sha256'] = sha256(out/'reference.vdb')
    (out/'scene.json').write_text(json.dumps(s, indent=2)+'\n')
    (out/'scene.lua').write_text('return '+lua(s)+'\n')
    subprocess.run([args.vsr, str(HERE/'build_scene.lua'), str(out/'scene.lua'), str(out/'scene.vsr')], check=True)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--scene', choices=['cloud','plume','emission'], required=True)
    p.add_argument('--preset')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hdri', type=Path, default=Path('/data/Envmaps/sunflowers_puresky_4k.hdr'))
    p.add_argument('--environment-rotation', type=float, default=-126.1669921875,
                   help='Degrees about +Z; default backlights from the sunflower HDRI sun')
    p.add_argument('--writer', type=Path, default=HERE/'build-vdb/write_nvdb')
    p.add_argument('--vsr', default='vsrLua')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--compute-device', choices=['CPU','CUDA','OPTIX'], default='CPU',
                   help='Recorded Blender reference backend; never silently falls back')
    args = p.parse_args()
    if args.samples < 1 or not math.isfinite(args.environment_rotation):
        p.error('samples must be positive and environment rotation finite')
    prepare(args)


if __name__ == '__main__':
    main()
