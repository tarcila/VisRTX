import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from generate import DEFAULTS, BOUNDS, cell_coordinates, fields, generate, presets, validate


class GeneratorTests(unittest.TestCase):
    def test_resolution_independent_samples(self):
        for asset in ('cloud', 'plume'):
            # Odd refinement factors share cell centers; doubling does not.
            low = cell_coordinates(0., 2/16, np.arange(16))
            high = cell_coordinates(0., 2/48, np.arange(48))
            np.testing.assert_allclose(low, high[1::3], rtol=0, atol=1e-14)
            def evaluate(axis, n):
                x, y, z = np.meshgrid(axis-1, axis-1, axis, indexing='ij')
                return fields(asset, x, y, z, dict(DEFAULTS, resolution=n))
            a, b = evaluate(low, 16), evaluate(high, 48)
            for name in a:
                np.testing.assert_allclose(a[name], b[name][1::3, 1::3, 1::3],
                                           rtol=1e-12, atol=1e-12)
                self.assertTrue(np.isfinite(a[name]).all())
                self.assertTrue((a[name] >= 0).all())

    def test_emission_in_vacuum_and_temperature_range(self):
        f = fields('plume', np.array([.76]), np.array([0.]), np.array([.8]), DEFAULTS)
        self.assertEqual(f['density'][0], 0)
        self.assertGreater(f['emission'].max(), 0)
        axis = np.linspace(0, 2, 48)
        x, y, z = np.meshgrid(axis-1, axis-1, axis, indexing='ij')
        t = fields('plume', x, y, z, DEFAULTS)['temperature']
        self.assertGreater(t.max(), 2000)
        self.assertTrue(((t == 0) | ((t >= 800) & (t <= 2400))).all())

    def test_seed_and_density_control(self):
        x = np.linspace(-.5, .5, 100)
        a = fields('cloud', x, x*0, x+1, DEFAULTS)['density']
        b = fields('cloud', x, x*0, x+1, dict(DEFAULTS, seed=43))['density']
        c = fields('cloud', x, x*0, x+1, dict(DEFAULTS, density_scale=10))['density']
        self.assertFalse(np.array_equal(a, b))
        np.testing.assert_allclose(c, 2*a)

    def test_integrated_density_converges(self):
        for asset in ('cloud', 'plume'):
            masses = []
            for n in (48, 96):
                axis = (np.arange(n)+.5)*2/n
                x, y, z = np.meshgrid(axis-1, axis-1, axis, indexing='ij')
                masses.append(fields(asset, x, y, z, DEFAULTS)['density'].mean()*8)
            self.assertLess(abs(masses[1]/masses[0]-1), .03)

    def test_validation(self):
        for key, value in [('resolution', 1), ('seed', -1),
                           ('density_scale', float('nan')), ('temperature_max', 100)]:
            with self.assertRaises(ValueError):
                validate(dict(DEFAULTS, **{key: value}))
        with self.assertRaises(ValueError):
            validate(dict(DEFAULTS, typo=1))

    def test_analytic_diagnostic_values(self):
        cases = json.loads((Path(__file__).parent/'diagnostics.json').read_text())['cases']
        for case in cases[:3]:
            p = case['parameters']
            sigma = -p['density'] * np.log(p.get('transmissionColor', [np.exp(-1)]*3))
            sigma /= p.get('transmissionDistance', 1)
            transmittance = np.exp(-2*sigma)
            integrated = np.full(3, 2.)
            np.divide(1-transmittance, sigma, out=integrated, where=sigma != 0)
            radiance = transmittance * case['background'] + integrated * p.get('emission', [0]*3)
            np.testing.assert_allclose(radiance, case['expected_radiance'])

    def test_bindings(self):
        for p in presets('plume').values():
            self.assertEqual(p['temperature'], 1)
            if 'emission.field' in p:
                self.assertEqual(p['emission'], [1]*3)
        self.assertEqual(presets('plume')['pure_scattering_no_thermal']['scatteringColor'], [1]*3)

    def test_export_roundtrip_and_manifest(self):
        writer = Path(__file__).parent/'build/write_nvdb'
        if not writer.exists():
            self.skipTest('Build the writer to run export integration checks')
        with tempfile.TemporaryDirectory() as tmp:
            for asset in ('cloud', 'plume'):
                hashes = []
                for run in (0, 1):
                    directory = Path(tmp)/f'{asset}{run}'
                    m = generate(asset, dict(DEFAULTS, resolution=16), directory, writer.resolve())
                    self.assertEqual(m['bounds'], BOUNDS)
                    self.assertEqual(m['schema_version'], 2)
                    self.assertEqual(json.loads((directory/'manifest.json').read_text()), m)
                    hashes.append([v['sha256'] for v in m['fields'].values()])
                    for v in m['fields'].values():
                        self.assertTrue((directory/v['file']).exists())
                        self.assertEqual(v['data_centering'], 'cell')
                        self.assertEqual(v['filter'], 'linear')
                        np.testing.assert_allclose(np.array(v['origin']) +
                            v['voxel_size']*np.array(v['dimensions']), v['domain'][1])
                        np.testing.assert_allclose(v['sample_origin'],
                            np.array(v['origin'])+.5*v['voxel_size'])
                        # Check actual serialized metadata, not just the manifest.
                        stored = json.loads(subprocess.check_output(
                            [str(writer.resolve()), '--info', str(directory/v['file'])]))
                        np.testing.assert_allclose(stored['world_bounds'], v['domain'], atol=1e-12)
                        np.testing.assert_allclose(stored['origin'], v['origin'], atol=1e-12)
                        np.testing.assert_array_equal(stored['index_bounds'][1],
                            np.array(v['dimensions'])-1)
                        coords = [cell_coordinates(o, v['voxel_size'], np.arange(n))
                                  for o, n in zip(v['origin'], v['dimensions'])]
                        xyz = np.meshgrid(*coords, indexing='ij')
                        source = v['grid_name'].split('_')[0]
                        expected = fields(asset, *xyz, m['configuration'])[source].astype(np.float32)
                        readback = Path(tmp)/'readback.raw'
                        subprocess.run([str(writer.resolve()), '--dump',
                                        str(directory/v['file']), str(readback)], check=True)
                        actual = np.fromfile(readback, dtype=np.float32).reshape(expected.shape)
                        np.testing.assert_array_equal(actual, expected)
                self.assertEqual(*hashes)


if __name__ == '__main__':
    unittest.main()
