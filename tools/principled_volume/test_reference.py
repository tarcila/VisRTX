import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from prepare_scene import lua, prepare, sample, validate_cell_manifest
from generate import DEFAULTS, generate
from reference_optics import blackbody, observer, XYZ_TO_RGB, RGB_Y, coefficients_per_density, thermal_lookup


class ReferenceTests(unittest.TestCase):
    def test_d65_normalization(self):
        from reference_optics import ROOT
        wl, cmf = observer()
        d = np.loadtxt(ROOT/'CIE_std_illum_D65.csv', delimiter=',')
        spectrum = np.interp(wl, d[:,0], d[:,1])
        weights = np.ones(len(wl)); weights[[0,-1]] = .5
        spectrum /= np.sum(spectrum*weights)
        rgb = (spectrum @ cmf) @ XYZ_TO_RGB.T
        np.testing.assert_allclose(rgb, [1,1,1], atol=.001)

    def test_thermal_monotonicity_and_lookup(self):
        t = np.array([0.,800.,1200.,1800.,2400.])
        rgb = blackbody(t)
        np.testing.assert_array_equal(rgb[0], [0,0,0])
        self.assertTrue(np.isfinite(rgb).all() and (rgb >= 0).all())
        self.assertTrue((np.diff(rgb @ RGB_Y) > 0).all())
        # Temperature changes brightness, not merely chromaticity.
        self.assertGreater((rgb[-1] @ RGB_Y)/(rgb[1] @ RGB_Y), 1e6)
        probe = np.array([800.3,1200.7,1800.2,2399.8])
        table = blackbody(np.arange(2402))
        np.testing.assert_allclose(thermal_lookup(probe, table), blackbody(probe), rtol=3e-4)

    def test_absorption_coupling(self):
        p = dict(transmissionColor=[np.exp(-1)]*3, transmissionDistance=1,
                 scatteringColor=[1]*3)
        scattering, absorption = coefficients_per_density(p)
        np.testing.assert_allclose(scattering, 1)
        np.testing.assert_array_equal(absorption*blackbody(2400), [0]*3)

    def test_domain_membership_and_vector_sampling(self):
        info = dict(origin=[0,0,0], voxel_size=1, dimensions=[2,2,2])
        xyz = (np.array([-1.,0.,.5,1.,2.]), np.array([.5]), np.array([.5]))
        scalar = np.arange(8).reshape(2,2,2)
        result = sample(scalar, info, xyz)
        np.testing.assert_allclose(result, [0,1.5,3.5,5.5,0])
        vector = np.repeat(scalar[...,None],3,axis=3)
        np.testing.assert_allclose(sample(vector,info,xyz), np.repeat(result[:,None],3,axis=1))

    def test_cell_domains_and_clamped_half_cells(self):
        info = dict(origin=[0,0,0], voxel_size=1, dimensions=[2,2,2],
                    data_centering='cell', domain=[[0,0,0],[2,2,2]])
        scalar = np.arange(8).reshape(2,2,2)+10
        xyz = (np.array([-1e-6,0.,.25,.5,1.,1.5,1.75,2.,2.+1e-6]),
               np.array([.5]), np.array([.5]))
        expected = np.array([0,10,10,10,12,14,14,14,0])
        np.testing.assert_allclose(sample(scalar,info,xyz), expected)
        vector = scalar[...,None]*np.array([1,2,3])
        np.testing.assert_allclose(sample(vector,info,xyz), expected[:,None]*[1,2,3])
        # Independent subdomain: outside is selected before edge clamping.
        info.update(origin=[1,0,0], domain=[[1,0,0],[3,2,2]])
        np.testing.assert_allclose(sample(scalar,info,(np.array([.99,1.,3.,3.01]),
                                                     np.array([.5]),np.array([.5]))),
                                   [0,10,14,0])

    def test_cell_manifest_rejects_legacy_or_inconsistent_extents(self):
        field = dict(origin=[0,0,0], sample_origin=[.5,.5,.5],
                     voxel_size=1, dimensions=[2,2,2], data_centering='cell',
                     filter='linear', domain=[[0,0,0],[2,2,2]])
        validate_cell_manifest(dict(schema_version=2, fields={'density':field}))
        with self.assertRaisesRegex(ValueError, 'schema-2'):
            validate_cell_manifest(dict(schema_version=1, fields={'density':field}))
        with self.assertRaisesRegex(ValueError, 'sampling metadata'):
            validate_cell_manifest(dict(schema_version=2, fields={
                'density':dict(field, domain=[[0,0,0],[1,1,1]])}))

    def test_cell_openvdb_export_centers_and_border(self):
        writer = Path(os.environ.get('PRINCIPLED_VOLUME_VDB_WRITER',
                      Path(__file__).parent/'build-vdb/write_nvdb')).resolve()
        if not writer.exists():
            self.skipTest('Build OpenVDB writer for secondary-export checks')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Nonzero edge values expose accidental fading into background.
            for channels in (1,3):
                data = (np.arange(8,dtype=np.float32).reshape(2,2,2)+10)
                if channels == 3:
                    data = (data[...,None]*np.array([1,2,3],dtype=np.float32))
                data.tofile(tmp/'in.raw')
                subprocess.run([str(writer),str(tmp/'in.raw'),str(tmp/'grid.nvdb'),
                    'test',str(channels),'2','2','2','1','0','0','0','cell'],check=True)
                subprocess.run([str(writer),'--merge-vdb-cell',str(tmp/'merged.vdb'),
                                str(tmp/'grid.nvdb')],check=True)
                # Test both direct and merged exports after actual file readback.
                for vdb in (tmp/'grid.nvdb.vdb',tmp/'merged.vdb'):
                    probes = [((x,.5,.5),value) for x,value in
                              ((0.,10),(.25,10),(.5,10),(1.,12),(1.5,14),(1.75,14),(2.,14))]
                    probes += [((0.,0.,0.),10),((2.,2.,2.),17),((1.,1.,1.),13.5)]
                    for position, value in probes:
                        measured = json.loads(subprocess.check_output([str(writer),
                            '--sample-vdb',str(vdb),'test',*map(str,position)]))
                        expected = [value] if channels == 1 else np.array([1,2,3])*value
                        np.testing.assert_allclose(measured,expected,rtol=0,atol=1e-6)
                # Halo extends beyond the logical domain. The reference shader,
                # not OpenVDB's raw sampler, owns the explicit volume clip.

    def test_cell_scene_archive_and_reference_bake(self):
        writer = Path(os.environ.get('PRINCIPLED_VOLUME_VDB_WRITER',
                      Path(__file__).parent/'build-vdb/write_nvdb')).resolve()
        primary = (Path(__file__).parent/'build/write_nvdb').resolve()
        vsr = shutil.which('vsrLua')
        if not writer.exists() or not primary.exists() or not vsr:
            self.skipTest('Build both writers and provide vsrLua for scene integration')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            generate('plume',dict(DEFAULTS,resolution=16),tmp/'dataset',primary)
            args = SimpleNamespace(dataset=tmp/'dataset',output=tmp/'scene',writer=writer,
                scene='plume',preset='different_domains',environment_rotation=0.,
                hdri=Path('unused'),samples=1,vsr=vsr)
            scene = prepare(args)
            self.assertEqual(scene['schema_version'],2)
            self.assertEqual(scene['render']['volume_precision'],'FULL')
            self.assertEqual(scene['render']['volume_clipping'],0.)
            self.assertEqual(scene['render']['volume_interpolation'],'LINEAR')
            self.assertEqual(scene['render']['compute_device'],'CPU')
            self.assertIn('reference_optics_sha256',scene['provenance'])
            self.assertIn('vdb_sha256',scene['reference'])
            self.assertTrue(scene['reference']['has_emission'])
            self.assertTrue((tmp/'scene/reference.vdb').exists())
            report = (tmp/'scene/scene.vsr.validation.txt').read_text()
            self.assertIn('cell centering/filter/ROI',report)
            self.assertIn('NOT TESTED: VisRTX rendering',report)
            stored = json.loads(subprocess.check_output([str(writer),'--info',
                                      str(tmp/'scene/reference_j.nvdb')]))
            np.testing.assert_allclose(stored['world_bounds'],scene['bounds'],atol=1e-12)

    def test_lua_optical_scalar_types(self):
        self.assertEqual(lua({'density':1,'emission':[0,1,2]}),
                         '{["density"]=1.0,["emission"]={0.0,1.0,2.0}}')

    def test_writer_dump_roundtrip(self):
        writer = Path(__file__).parent/'build/write_nvdb'
        if not writer.exists():
            self.skipTest('Build writer first')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for channels in (1,3):
                data = np.arange(8*8*8*channels, dtype=np.float32)
                data.tofile(tmp/'in.raw')
                subprocess.run([str(writer),'%s/in.raw'%tmp,'%s/grid.nvdb'%tmp,'test',
                    str(channels),'8','8','8','.1','-1','0','0'],check=True)
                subprocess.run([str(writer),'--dump',str(tmp/'grid.nvdb'),str(tmp/'out.raw')],check=True)
                np.testing.assert_array_equal(data,np.fromfile(tmp/'out.raw',dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
