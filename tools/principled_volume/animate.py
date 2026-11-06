"""Sequence orchestration; generation itself has no Vela dependency."""
import json
import tempfile
from pathlib import Path

import numpy as np
from generate import BOUNDS, file_hash, generate, validate

MOTION_DEFAULTS = dict(sway_strength=.08, rise_cycles=1, pulse_strength=.2)


def validate_animation(frames, duration, motion):
    if type(frames) is not int or frames < 2:
        raise ValueError('frames must be an integer >= 2')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and positive')
    if set(motion) != set(MOTION_DEFAULTS):
        raise ValueError('unknown or missing motion controls')
    if type(motion['rise_cycles']) is not int or motion['rise_cycles'] < 0:
        raise ValueError('rise_cycles must be a nonnegative integer')
    for key in ('sway_strength', 'pulse_strength'):
        if not np.isfinite(motion[key]) or motion[key] < 0:
            raise ValueError(key+' must be finite and nonnegative')
    if motion['pulse_strength'] >= 1:
        raise ValueError('pulse_strength must be < 1 (continuously active source)')


def generate_sequence(cfg, output, writer, *, frames=48, duration=4.,
                      motion=None, diagnostics=False, overwrite=False):
    motion = dict(MOTION_DEFAULTS if motion is None else motion)
    validate(cfg)
    validate_animation(frames, duration, motion)
    output, writer = Path(output).absolute(), Path(writer).resolve()
    if output.is_symlink():
        raise ValueError('sequence output must not be a symlink')
    if output.exists():
        if not overwrite:
            raise ValueError('sequence output exists; use --overwrite explicitly')
        marker = output/'manifest.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('kind') != 'plume_sequence':
            raise ValueError('overwrite only accepts a completed plume sequence directory')
    # Fail on a missing writer before touching any existing sequence.
    writer_hash = file_hash(writer)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.plume-', dir=output.parent) as tmp:
        staged = Path(tmp)/'sequence'
        staged.mkdir()
        manifest = dict(schema_version=1, kind='plume_sequence', asset='plume',
                        configuration=cfg, motion=motion, diagnostics=diagnostics,
                        duration_seconds=duration, frame_count=frames,
                        frames_per_second=frames/duration, bounds=BOUNDS,
                        sampling='cell-centered', loop=True,
                        closure='first field reused at normalized time 1; no duplicate NVDB',
                        provenance=dict(orchestrator_sha256=file_hash(__file__),
                                        writer_sha256=writer_hash), frames=[])
        for i in range(frames):
            relative = Path('frames')/f'{i:04d}'
            sample = generate('plume', cfg, staged/relative, writer,
                              phase=i/frames, motion=motion, diagnostics=diagnostics)
            manifest['frames'].append(dict(index=i, time_seconds=i*duration/frames,
                phase=i/frames, manifest=str(relative/'manifest.json'),
                fields={name: dict(info, file=str(relative/info['file']))
                        for name, info in sample['fields'].items()}))
        (staged/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        # Build completely before replacing a previous owned directory. Failed
        # generation leaves it intact; no partial sequence is published.
        backup = Path(tmp)/'previous'
        if output.exists():
            output.rename(backup)
        try:
            staged.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
    return manifest
