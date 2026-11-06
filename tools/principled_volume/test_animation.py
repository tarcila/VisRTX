import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from animate import MOTION_DEFAULTS, generate_sequence, validate_animation
from generate import DEFAULTS, fields
from package_animation import package, sequence_config

HERE = Path(__file__).parent
WRITER = HERE/'build/write_nvdb'


class AnimationTests(unittest.TestCase):
    def samples(self, phase, n=24, motion=None):
        a = (np.arange(n)+.5)*2/n
        xyz = np.meshgrid(a-1, a-1, a, indexing='ij')
        return fields('plume', *xyz, DEFAULTS, phase=phase,
                      motion=MOTION_DEFAULTS if motion is None else motion)

    def test_static_regression(self):
        # Float32 sample hashes from the pre-animation generator, seed 42, 16^3.
        a = (np.arange(16)+.5)/8
        xyz = np.meshgrid(a-1, a-1, a, indexing='ij')
        expected = {
            'cloud': ['5b42d0123a5fd4ffeb2efbf4149e12a1f72e4557311cae56078ffaaf113a5b63'],
            'plume': ['fb3628a9e0504d29224ee17f7bead115c2119f81e210b3941c0da95caf154295',
                      '61cdf9d6f03d5a8fd545f0ad342c399430880bdb1c4d1d29102b767f408a6118',
                      '98f15a74d0065002db4f221a13acb454f7f2ebcd9d99d248b0e9e359e1c395de']}
        for asset, hashes in expected.items():
            actual = fields(asset, *xyz, DEFAULTS)
            self.assertEqual([hashlib.sha256(v.astype('<f4').tobytes()).hexdigest()
                              for v in actual.values()], hashes)

    def test_periodicity_coherence_and_validity(self):
        first = self.samples(0.)
        for phase in (0., .13, .5, .999):
            a, b = self.samples(phase), self.samples(phase+1)
            for name in a:
                self.assertTrue(np.isfinite(a[name]).all())
                self.assertTrue((a[name] >= 0).all())
                np.testing.assert_allclose(a[name], b[name], rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(a[name], self.samples(phase)[name])
            np.testing.assert_array_equal(a['emission'], first['emission'])
        for name in ('density', 'temperature'):
            self.assertFalse(np.array_equal(first[name], self.samples(.25)[name]))
            # Small changes through the seam, not an unrelated final random seed.
            near = np.mean(np.abs(self.samples(1-1e-6)[name]-self.samples(1e-6)[name]))
            far = np.mean(np.abs(self.samples(.25)[name]-first[name]))
            self.assertLess(near, far*.01)
        # Adjacent production frames, including the seam, change less on average
        # than quarter-cycle-separated frames.
        sequence = [self.samples(i/48, n=16)['density'] for i in range(48)]
        close = [np.mean(np.abs(sequence[i]-sequence[(i+1)%48])) for i in range(48)]
        distant = [np.mean(np.abs(sequence[i]-sequence[(i+12)%48])) for i in range(48)]
        self.assertLess(np.mean(close), np.mean(distant))
        self.assertLess(close[-1], 2*np.mean(close))

    def test_validation(self):
        for frames, duration, motion in [
            (1, 4., MOTION_DEFAULTS), (3, 0., MOTION_DEFAULTS),
            (3, float('nan'), MOTION_DEFAULTS),
            (3, 4., dict(MOTION_DEFAULTS, pulse_strength=1.)),
            (3, 4., dict(MOTION_DEFAULTS, rise_cycles=.5)),
            (3, 4., dict(MOTION_DEFAULTS, sway_strength=-1.))]:
            with self.assertRaises(ValueError):
                validate_animation(frames, duration, motion)

    @unittest.skipUnless(WRITER.exists(), 'build write_nvdb for export tests')
    def test_sequence_roundtrip_safety_and_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'sequence'
            cfg = dict(DEFAULTS, resolution=16)
            m = generate_sequence(cfg, directory, WRITER, frames=3, diagnostics=True)
            self.assertEqual(len(m['frames']), 3)
            self.assertEqual(len(list(directory.glob('frames/*'))), 3)
            transforms = {}
            for frame in m['frames']:
                self.assertLess(frame['time_seconds'], m['duration_seconds'])
                for name, info in frame['fields'].items():
                    path = directory/info['file']
                    stored = json.loads(subprocess.check_output([str(WRITER.resolve()), '--info', str(path)]))
                    np.testing.assert_allclose(stored['world_bounds'], info['domain'])
                    if name in transforms:
                        self.assertEqual(transforms[name], (info['origin'], info['voxel_size']))
                    transforms[name] = (info['origin'], info['voxel_size'])
                    raw = Path(tmp)/'dump.raw'
                    subprocess.run([str(WRITER.resolve()), '--dump', str(path), str(raw)], check=True)
                    axes = [o+(np.arange(n)+.5)*info['voxel_size']
                            for o, n in zip(info['origin'], info['dimensions'])]
                    expected = fields('plume', *np.meshgrid(*axes, indexing='ij'), cfg,
                        phase=frame['phase'], motion=MOTION_DEFAULTS)[name.split('_')[0]]
                    actual = np.fromfile(raw, dtype=np.float32).reshape(expected.shape)
                    np.testing.assert_array_equal(actual, expected.astype(np.float32))
            with self.assertRaises(ValueError):
                generate_sequence(cfg, directory, WRITER, frames=3)
            original = (directory/'manifest.json').read_bytes()
            with patch('animate.generate', side_effect=RuntimeError('injected writer failure')):
                with self.assertRaises(RuntimeError):
                    generate_sequence(cfg, directory, WRITER, frames=3, overwrite=True)
            self.assertEqual((directory/'manifest.json').read_bytes(), original)
            again = generate_sequence(cfg, directory, WRITER, frames=3,
                                      diagnostics=True, overwrite=True)
            self.assertEqual(m, again)
            if shutil.which('vsrLua'):
                for preset in ('thermal', 'combined', 'different_domains'):
                    package(directory, preset=preset, overwrite=True)
                    self.assertIn('PASS', (directory/'animation.validation.txt').read_text())
                with self.assertRaises(ValueError):
                    package(directory)
            else:
                print('Vela archive integration not run: vsrLua unavailable')
            # Hash checking catches corrupted frames before Vela imports them.
            (directory/m['frames'][0]['fields']['density']['file']).write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                sequence_config(directory, 'thermal')

    @unittest.skipUnless(WRITER.exists(), 'build write_nvdb for export tests')
    def test_plume_only_and_foreign_directory_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                generate_sequence(dict(DEFAULTS, resolution=8), root, WRITER, overwrite=True)
            m = generate_sequence(dict(DEFAULTS, resolution=8), root/'plume', WRITER, frames=2)
            self.assertEqual(set(m['frames'][0]['fields']), {'density', 'temperature'})
            with self.assertRaises(ValueError):
                sequence_config(root/'plume', 'combined')


if __name__ == '__main__':
    unittest.main()
