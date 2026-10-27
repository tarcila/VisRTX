#!/usr/bin/env python3
"""Deterministic, continuous procedural fields; no simulation cache required."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np

DEFAULTS = dict(seed=42, resolution=128, density_scale=5.0,
                detail_strength=0.16, temperature_min=800.0,
                temperature_max=2400.0, emission_scale=3.0)
BOUNDS = [[-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]]


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def fields(asset, x, y, z, cfg):
    """Coordinates in meters. Frequencies and random phases never depend on N."""
    rng = np.random.default_rng(cfg['seed'])
    phases = rng.uniform(0, 2 * np.pi, 12)
    noise = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape))
    for i, frequency in enumerate((5., 11., 23., 47.)):
        noise += (0.5 ** i) * (np.sin(frequency * x + phases[3*i])
                   * np.sin(frequency * y + phases[3*i+1])
                   * np.sin(frequency * z + phases[3*i+2]))
    detail = cfg['detail_strength'] * noise
    edge = (1 - smooth(.85, .98, np.abs(x))) * (1 - smooth(.85, .98, np.abs(y)))
    edge = edge * smooth(.02, .12, z) * (1 - smooth(1.8, 1.98, z))
    if asset == 'cloud':
        shape = np.zeros_like(noise)
        for cx, cy, cz, radius in [(-.35, 0, .85, .42), (0, .1, 1.1, .53),
                                   (.4, 0, 1., .36), (-.1, -.3, .65, .36),
                                   (.2, .25, .65, .4)]:
            r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
            shape = np.maximum(shape, 1 - smooth(radius-.12, radius+.08, r+detail))
        cavity = 1 - .95 * np.exp(-((x-.14)**2+(y+.38)**2+(z-.95)**2)/.035)
        veil = .16 * np.exp(-((z-.5-.12*np.sin(6*x))/.09)**2 - (y/.3)**2)
        veil *= 1 - smooth(.65, .9, np.abs(x))
        density = cfg['density_scale'] * np.maximum(shape*cavity, veil) * edge
        return {'density': density}
    cx = .16 * np.sin(5*z) + .08 * np.sin(11*z)
    cy = .12 * np.cos(6*z)
    radius = .12 + .17 * z
    r = np.sqrt((x-cx)**2+(y-cy)**2)
    body = 1 - smooth(radius*.55, radius*1.2, r+detail*.7)
    soot = (.25 + .75*smooth(.3, 1.3, z)) * (.75+.25*np.sin(14*z+noise))
    density = cfg['density_scale'] * body * soot * edge
    hot = np.exp(-(r/(radius*.65))**2) * (1-smooth(.25, 1.65, z))
    temperature = cfg['temperature_min'] + (cfg['temperature_max']-cfg['temperature_min'])*hot
    temperature = np.where((body*edge) > .001, temperature, 0)
    # A separate ribbon, deliberately in the density-free right margin.
    ribbon = np.exp(-((x-.76-.035*np.sin(9*z))/.035)**2-(y/.12)**2)
    ribbon *= smooth(.25, .4, z)*(1-smooth(1.5, 1.7, z))
    emission = cfg['emission_scale'] * ribbon[..., None] * np.array([.1, .55, 1.])
    return {'density': density, 'temperature': temperature, 'emission': emission}


def presets(asset):
    base = {'density': 1., 'density.field': 'density', 'density.outside': 0.,
            'transmissionColor': [float(np.exp(-1))]*3, 'transmissionDistance': 1.,
            'scatteringColor': [1., 1., 1.], 'anisotropy': 0.,
            'secondaryAnisotropy': 0., 'secondaryWeight': 0.,
            'emission': [0., 0., 0.], 'blackbodyIntensity': 0.}
    if asset == 'cloud':
        variants = {'isotropic': {}, 'forward': {'anisotropy': .65},
                    'two_lobe': {'anisotropy': .7, 'secondaryAnisotropy': -.3,
                                 'secondaryWeight': .25},
                    'colored_absorption': {'transmissionColor': [.7, .4, .2],
                                           'scatteringColor': [.7]*3}}
    else:
        base.update({'scatteringColor': [.2]*3, 'temperature': 1.,
                     'temperature.field': 'temperature', 'temperature.outside': 0.,
                     'blackbodyTint': [1.]*3})
        variants = {'thermal': {'blackbodyIntensity': 1.},
                    'explicit': {'emission': [1.]*3, 'emission.field': 'emission'},
                    'combined': {'blackbodyIntensity': 1., 'emission': [1.]*3,
                                 'emission.field': 'emission'},
                    'different_domains': {'blackbodyIntensity': 1.,
                        'temperature.field': 'temperature_small',
                        'emission': [1.]*3, 'emission.field': 'emission_offset'},
                    'pure_scattering_no_thermal': {'blackbodyIntensity': 1.,
                                                   'scatteringColor': [1.]*3}}
    return {name: dict(base, **overrides) for name, overrides in variants.items()}


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError('Unknown configuration keys: '+str(set(cfg)-set(DEFAULTS)))
    if not isinstance(cfg['resolution'], int) or cfg['resolution'] < 8:
        raise ValueError('resolution must be an integer >= 8')
    if not isinstance(cfg['seed'], int) or cfg['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in set(DEFAULTS)-{'resolution', 'seed'}:
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(key+' must be finite and nonnegative')
    if cfg['temperature_max'] < cfg['temperature_min']:
        raise ValueError('temperature_max must be >= temperature_min')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cell_coordinates(origin, spacing, indices):
    """Samples are at full-cell centers; origin is the lower domain boundary."""
    return origin + spacing * (np.asarray(indices) + .5)


def generate(asset, cfg, output, writer):
    validate(cfg)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Output already contains a manifest; choose a new directory')
    n = cfg['resolution']
    manifest = {'schema_version': 2, 'asset': asset, 'configuration': cfg,
                'provenance': {'generator_sha256': file_hash(__file__),
                               'writer_sha256': file_hash(writer),
                               'python_version': platform.python_version(),
                               'numpy_version': np.__version__},
                'bounds': BOUNDS, 'meters_per_scene_unit': 1,
                'sampling': 'cell-centered; N full cells; origin is lower boundary; '
                            'samples at origin+(index+0.5)*voxel_size; '
                            'clamp to edge inside domain; zero sparse background',
                'fields': {}, 'presets': presets(asset),
                'view': {'camera': [3.4, -5., 2.7], 'target': [0., 0., 1.],
                         'light_position': [-2., 2., 4.], 'light_power_watts': 1000.,
                         'light_size_m': 2., 'world_strength': .05,
                         'exposure_stops': 0. if asset == 'cloud' else -6.},
                'reference_warning': 'Blender appearance only; shader is not an optical oracle.'}
    names = ['density'] if asset == 'cloud' else ['density', 'temperature', 'emission',
                                                'temperature_small', 'emission_offset']
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        for name in names:
            source = name.split('_')[0]
            lower, upper = np.array(BOUNDS[0]), np.array(BOUNDS[1])
            field_n = n
            if name == 'temperature_small':
                lower, upper = np.array([-.5, -.5, 0.]), np.array([.5, .5, 1.])
                field_n = max(8, (n+1)//2)
            if name == 'emission_offset':
                lower, upper = np.array([.5, -.5, .5]), np.array([1.5, .5, 1.5])
                field_n = max(8, (n+1)//2)
            origin = lower
            spacing = float((upper[0]-lower[0])/field_n)
            dims = np.full(3, field_n, dtype=int)
            channels = 3 if source == 'emission' else 1
            shape = tuple(dims) + ((3,) if channels == 3 else ())
            raw = Path(tmp)/'field.raw'
            data = np.memmap(raw, dtype=np.float32, mode='w+', shape=shape)
            y = cell_coordinates(lower[1], spacing, np.arange(field_n))[None, :, None]
            z = cell_coordinates(lower[2], spacing, np.arange(field_n))[None, None, :]
            for i in range(0, dims[0], 8):
                x = cell_coordinates(origin[0], spacing, np.arange(i, min(i+8, dims[0])))[:, None, None]
                data[i:i+8] = fields(asset, x, y, z, cfg)[source]
            data.flush()
            if not np.isfinite(data).all() or (data < 0).any():
                raise ValueError('Invalid field '+name)
            info = {'file': name+'.nvdb', 'grid_name': name, 'channels': channels,
                    'storage': 'float32' if channels == 1 else 'Vec3f',
                    'units': {'density': 'dimensionless', 'temperature': 'kelvin',
                              'emission': 'linear sRGB radiance per meter'}[source],
                    'dimensions': dims.tolist(), 'origin': origin.tolist(),
                    'sample_origin': (origin + .5*spacing).tolist(),
                    'data_centering': 'cell', 'filter': 'linear',
                    'voxel_size': spacing, 'background': 0 if channels == 1 else [0]*3,
                    'domain': [origin.tolist(), upper.tolist()],
                    'minimum': float(data.min()), 'maximum': float(data.max())}
            del data
            subprocess.run([str(writer), str(raw), str(output/info['file']), name,
                            str(channels), *map(str, dims), str(spacing),
                            *map(str, origin), 'cell'], check=True)
            with (output/info['file']).open('rb') as f:
                info['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
            manifest['fields'][name] = info
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', choices=['cloud', 'plume'], required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--resolution', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--writer', type=Path, default=Path(__file__).parent/'build/write_nvdb')
    args = p.parse_args()
    cfg = copy.deepcopy(DEFAULTS)
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    for key in ('resolution', 'seed'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    generate(args.asset, cfg, args.output.resolve(), args.writer.resolve())


if __name__ == '__main__':
    main()
