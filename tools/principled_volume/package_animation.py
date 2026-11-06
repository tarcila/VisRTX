#!/usr/bin/env python3
"""Package generated plume NVDBs into matching scene/animation VSR archives."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from animate import validate_animation
from generate import BOUNDS, file_hash, presets, validate

HERE = Path(__file__).resolve().parent


def lua(value):
    """Encode generated data, including paths, without executable Lua fragments."""
    if isinstance(value, dict):
        return '{'+','.join('['+lua(k)+']='+lua(v) for k, v in value.items())+'}'
    if isinstance(value, list):
        return '{'+','.join(map(lua, value))+'}'
    if isinstance(value, str):
        # Decimal escapes work for all UTF-8 bytes and cannot inject Lua code.
        return '"'+''.join('\\%03d' % b for b in value.encode('utf-8'))+'"'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(float(value))
    raise TypeError(type(value))


def sequence_config(directory, preset):
    directory = Path(directory).resolve()
    m = json.loads((directory/'manifest.json').read_text())
    if m.get('kind') != 'plume_sequence' or m.get('schema_version') != 1:
        raise ValueError('expected a plume sequence manifest')
    validate(m['configuration'])
    validate_animation(m['frame_count'], m['duration_seconds'], m['motion'])
    if len(m['frames']) != m['frame_count'] or m['bounds'] != BOUNDS:
        raise ValueError('inconsistent sequence metadata')
    parameters = presets('plume')[preset]
    names = sorted({v for k, v in parameters.items() if k.endswith('.field')})
    baseline = {}
    for i, frame in enumerate(m['frames']):
        if frame['index'] != i or frame['phase'] != i/m['frame_count']:
            raise ValueError('invalid frame ordering or phase')
        for name in names:
            if name not in frame['fields']:
                raise ValueError(f'{preset} requires {name}; generate with --diagnostics')
            info = frame['fields'][name]
            path = (directory/info['file']).resolve()
            if not path.is_relative_to(directory) or file_hash(path) != info['sha256']:
                raise ValueError('field path or hash mismatch: '+str(path))
            if info['data_centering'] != 'cell' or info['filter'] != 'linear':
                raise ValueError('expected cell-centered linear fields')
            transform = {key: info[key] for key in ('dimensions', 'origin', 'voxel_size',
                         'domain', 'channels', 'grid_name', 'data_centering', 'filter')}
            if i and transform != baseline[name]:
                raise ValueError('field transform changes across frames: '+name)
            baseline[name] = transform
            info['file'] = str(path)
    return dict(frames=m['frames'], names=names, parameters=parameters,
                bounds=m['bounds'], duration=m['duration_seconds'], preset=preset)


def package(directory, *, preset='thermal', executable='vsrLua', overwrite=False):
    directory = Path(directory).resolve()
    outputs = ('scene.vsr', 'animations.vsr', 'animation.validation.txt')
    if not overwrite and any((directory/name).exists() for name in outputs):
        raise ValueError('archives already exist; use --overwrite')
    cfg = sequence_config(directory, preset)
    with tempfile.TemporaryDirectory(prefix='.package-', dir=directory) as tmp:
        tmp = Path(tmp)
        config = tmp/'config.lua'
        config.write_text('return '+lua(cfg)+'\n')
        subprocess.run([executable, str(HERE/'build_animation.lua'), str(config),
                        str(tmp)], check=True)
        # A separate process verifies that nothing depends on live builder objects.
        subprocess.run([executable, str(HERE/'build_animation.lua'), str(config),
                        str(tmp), 'validate'], check=True)
        for name in outputs:
            (tmp/name).replace(directory/name)
    return directory/'scene.vsr', directory/'animations.vsr'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--preset', choices=list(presets('plume')), default='thermal')
    p.add_argument('--vsr-lua', default='vsrLua')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    for path in package(args.sequence, preset=args.preset, executable=args.vsr_lua,
                        overwrite=args.overwrite):
        print(path)


if __name__ == '__main__':
    main()
