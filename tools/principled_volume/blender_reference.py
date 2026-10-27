"""Matched scene reference: blender -b --python blender_reference.py -- scene.json output.blend.

Reference j is calibrated but baked before interpolation; see scene.json caveats.
No Blender Blackbody node or Principled Volume defaults are used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector



def math_node(tree, op, a, b=None):
    node = tree.nodes.new('ShaderNodeMath')
    node.operation = op
    for i, value in enumerate((a, b)):
        if value is None:
            continue
        if isinstance(value, (int, float)):
            node.inputs[i].default_value = value
        else:
            tree.links.new(value, node.inputs[i])
    return node.outputs[0]


def vector_node(tree, op, a, b=None):
    node = tree.nodes.new('ShaderNodeVectorMath')
    node.operation = op
    for i, value in enumerate((a, b)):
        if value is None:
            continue
        if isinstance(value, (tuple, list, Vector)):
            node.inputs[i].default_value = value
        else:
            tree.links.new(value, node.inputs[i])
    return node.outputs['Value' if op == 'DOT_PRODUCT' else 'Vector']


def world_material(env):
    world = bpy.data.worlds.new('Matched environment')
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new('ShaderNodeOutputWorld')
    background = tree.nodes.new('ShaderNodeBackground')
    background.inputs['Color'].default_value = (1,1,1,1)
    background.inputs['Strength'].default_value = env['scale']
    tree.links.new(background.outputs[0], output.inputs['Surface'])
    if env['file']:
        # Match VisRTX HDRI.cpp / gpu_util.h, including bottom-up image rows.
        # This explicit UV mapping avoids assuming Blender's native longitude convention.
        up = -Vector(env['up']).normalized()
        forward = -up.cross(Vector(env['direction'])).normalized()
        right = forward.cross(up).normalized()
        geometry = tree.nodes.new('ShaderNodeNewGeometry')
        ray = vector_node(tree, 'MULTIPLY', geometry.outputs['Incoming'], (-1,-1,-1))
        x = vector_node(tree, 'DOT_PRODUCT', ray, right)
        y = vector_node(tree, 'DOT_PRODUCT', ray, forward)
        z = vector_node(tree, 'DOT_PRODUCT', ray, up)
        phi = math_node(tree, 'ARCTAN2', y, x)
        u = math_node(tree, 'FRACT', math_node(tree, 'ADD', math_node(tree, 'MULTIPLY', phi, 1/(2*math.pi)), 1))
        v = math_node(tree, 'MULTIPLY', math_node(tree, 'ARCCOSINE', z), 1/math.pi)
        uv = tree.nodes.new('ShaderNodeCombineXYZ')
        tree.links.new(u, uv.inputs['X']); tree.links.new(v, uv.inputs['Y'])
        texture = tree.nodes.new('ShaderNodeTexImage')
        texture.image = bpy.data.images.load(env['file'], check_existing=True)
        texture.image.colorspace_settings.name = 'Non-Color'
        texture.interpolation = 'Linear'
        tree.links.new(uv.outputs[0], texture.inputs['Vector'])
        tree.links.new(texture.outputs['Color'], background.inputs['Color'])
    return world


def volume_material(cfg):
    p = cfg['parameters']
    mat = bpy.data.materials.new('Explicit sigma_s, sigma_a, calibrated j')
    mat.use_nodes = True
    mat.cycles.volume_interpolation = cfg['render'].get('volume_interpolation', 'LINEAR')
    tree = mat.node_tree
    tree.nodes.clear()
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    attr = tree.nodes.new('ShaderNodeAttribute')
    attr.attribute_name = 'density'
    geometry = tree.nodes.new('ShaderNodeNewGeometry')
    xyz = tree.nodes.new('ShaderNodeSeparateXYZ')
    tree.links.new(geometry.outputs['Position'], xyz.inputs[0])
    mask = 1.
    for axis in range(3):
        inside = math_node(tree, 'MULTIPLY',
            math_node(tree, 'GREATER_THAN', xyz.outputs[axis], cfg['bounds'][0][axis]),
            math_node(tree, 'LESS_THAN', xyz.outputs[axis], cfg['bounds'][1][axis]))
        mask = math_node(tree, 'MULTIPLY', mask, inside)
    density = math_node(tree, 'MULTIPLY', attr.outputs['Fac'], p['density'])
    density = math_node(tree, 'MULTIPLY', density, mask)
    extinction = -np.log(p['transmissionColor'])/p['transmissionDistance']
    albedo = np.asarray(p['scatteringColor'])
    scattering, absorption = extinction*albedo, extinction*(1-albedo)
    closures = []
    secondary = p.get('secondaryWeight', 0)
    for coefficient, g in ((scattering*(1-secondary), p['anisotropy']),
                           (scattering*secondary, p.get('secondaryAnisotropy', 0))):
        magnitude = float(coefficient.max())
        if magnitude == 0:
            continue
        scatter = tree.nodes.new('ShaderNodeVolumeScatter')
        scatter.inputs['Color'].default_value = (*map(float, coefficient/magnitude), 1)
        scatter.inputs['Anisotropy'].default_value = g
        tree.links.new(math_node(tree, 'MULTIPLY', density, magnitude), scatter.inputs['Density'])
        closures.append(scatter.outputs[0])
    magnitude = float(absorption.max())
    if magnitude > 0:
        absorb = tree.nodes.new('ShaderNodeVolumeAbsorption')
        # Cycles absorption coefficient = density * (1 - Color).
        absorb.inputs['Color'].default_value = (*map(float, 1-absorption/magnitude), 1)
        tree.links.new(math_node(tree, 'MULTIPLY', density, magnitude), absorb.inputs['Density'])
        closures.append(absorb.outputs[0])
    if cfg['reference']['has_emission']:
        j = tree.nodes.new('ShaderNodeAttribute')
        j.attribute_name = 'reference_j'
        emit = tree.nodes.new('ShaderNodeEmission')
        tree.links.new(mask, emit.inputs['Strength'])
        tree.links.new(j.outputs['Color'], emit.inputs['Color'])
        closures.append(emit.outputs[0])
    assert closures
    closure = closures[0]
    for other in closures[1:]:
        add = tree.nodes.new('ShaderNodeAddShader')
        tree.links.new(closure, add.inputs[0]); tree.links.new(other, add.inputs[1])
        closure = add.outputs[0]
    tree.links.new(closure, output.inputs['Volume'])
    return mat


def configure_compute(scene, backend):
    """Choose a recorded backend explicitly; never silently fall back to CPU."""
    if backend == 'CPU':
        scene.cycles.device = 'CPU'
        return ['CPU']
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = backend
    prefs.get_devices()
    selected = [d for d in prefs.devices if d.type == backend]
    if not selected:
        raise RuntimeError('Requested Cycles backend is unavailable: '+backend)
    for device in prefs.devices:
        device.use = device.type == backend
    scene.cycles.device = 'GPU'
    return [d.name for d in selected]


def render_metadata(scene):
    volume = next(o for o in scene.objects if o.type == 'VOLUME')
    return dict(blender_version=bpy.app.version_string,
        blender_executable=bpy.app.binary_path,
        reference_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        blender_build_hash=bpy.app.build_hash.decode(errors='replace'),
        compute_backend=scene['testbed_compute_backend'],
        compute_devices=json.loads(scene['testbed_compute_devices_json']),
        samples=scene.cycles.samples, seed=scene.cycles.seed,
        volume_precision=volume.data.render.precision,
        volume_clipping=volume.data.render.clipping,
        volume_interpolation=volume.data.materials[0].cycles.volume_interpolation,
        volume_step_rate=scene.cycles.volume_step_rate,
        volume_bounces=scene.cycles.volume_bounces, max_bounces=scene.cycles.max_bounces,
        denoising=scene.cycles.use_denoising, adaptive_sampling=scene.cycles.use_adaptive_sampling,
        clamp_direct=scene.cycles.sample_clamp_direct,
        clamp_indirect=scene.cycles.sample_clamp_indirect,
        film_transparent=scene.render.film_transparent)


def build(cfg, output, samples=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples or cfg['render']['samples']
    scene.cycles.seed = cfg['render'].get('seed', 42)
    backend = cfg['render'].get('compute_device', 'CPU')
    scene['testbed_compute_backend'] = backend
    scene['testbed_compute_devices_json'] = json.dumps(configure_compute(scene, backend))
    scene.cycles.sample_clamp_direct = 0
    scene.cycles.sample_clamp_indirect = 0
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.volume_bounces = cfg['render']['volume_bounces']
    scene.cycles.max_bounces = cfg['render'].get('max_bounces', 16)
    scene.cycles.volume_step_rate = cfg['render'].get('volume_step_rate', .5)
    scene.render.resolution_x = cfg['render']['width']
    scene.render.resolution_y = cfg['render']['height']
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = cfg['render']['view_transform']
    scene.view_settings.exposure = cfg['render']['exposure']
    scene.world = world_material(cfg['environment'])
    bpy.ops.object.volume_import(filepath=cfg['reference']['vdb'])
    volume = bpy.context.object
    volume.name = cfg['scene']+'/'+cfg['preset']
    volume.data.materials.append(volume_material(cfg))
    volume.data.display.density = 1
    volume.data.render.precision = cfg['render'].get('volume_precision', 'FULL')
    volume.data.render.clipping = cfg['render'].get('volume_clipping', 0.)
    for mesh in cfg['meshes']:
        data = bpy.data.meshes.new(mesh['name'])
        data.from_pydata(mesh['vertices'], [], mesh['triangles'])
        data.update()
        obj = bpy.data.objects.new(mesh['name'], data)
        scene.collection.objects.link(obj)
        material = bpy.data.materials.new(mesh['name'])
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        diffuse = tree.nodes.new('ShaderNodeBsdfDiffuse')
        diffuse.inputs['Color'].default_value = (*mesh['color'],1)
        diffuse.inputs['Roughness'].default_value = 0
        out = tree.nodes.new('ShaderNodeOutputMaterial')
        tree.links.new(diffuse.outputs[0], out.inputs['Surface'])
        data.materials.append(material)
    camera = cfg['camera']
    bpy.ops.object.camera_add(location=camera['position'])
    scene.camera = bpy.context.object
    # Camera roll follows the shared up vector, not a hard-coded axis.
    direction = (Vector(camera['target'])-scene.camera.location).normalized()
    right = direction.cross(Vector(camera['up'])).normalized()
    up = right.cross(direction)
    from mathutils import Matrix
    scene.camera.rotation_euler = Matrix((right, up, -direction)).transposed().to_euler()
    scene.camera.data.sensor_fit = 'VERTICAL'
    scene.camera.data.sensor_height = 24
    scene.camera.data.lens = 12/math.tan(math.radians(camera['fovy_degrees'])/2)
    scene.camera.data.clip_start = .01
    scene.camera.data.clip_end = 100
    scene['testbed_status'] = cfg['status']
    scene['reference_limitations'] = cfg['reference']['limitation']
    scene['scene_description'] = json.dumps(cfg)
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.filepath = str(output.with_suffix('.exr'))
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    return scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--samples', type=int)
    p.add_argument('--no-render', action='store_true')
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    cfg = json.loads(args.config.read_text())
    output = args.output.resolve()
    scene = build(cfg, output, args.samples)
    if args.no_render:
        return
    bpy.ops.render.render(write_still=True)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    bpy.data.images['Render Result'].save_render(str(output.with_suffix('.png')), scene=scene)
    image = bpy.data.images.load(str(output.with_suffix('.exr')), check_existing=False)
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(-1,4)
    rgb = pixels[:,:3]
    assert np.isfinite(rgb).all(), 'Nonfinite reference image'
    stats = dict(render_metadata(scene),
                 minimum=float(rgb.min()), maximum=float(rgb.max()), mean_rgb=rgb.mean(axis=0).tolist(),
                 nonzero_fraction=float(np.mean(np.max(rgb,axis=1)>1e-8)),
                 warning='Reference render only; no VisRTX comparison has been performed.')
    output.with_suffix('.stats.json').write_text(json.dumps(stats,indent=2)+'\n')


if __name__ == '__main__':
    main()
