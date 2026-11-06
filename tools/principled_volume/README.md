# Procedural principled-volume datasets

Two reproducible assets for the proposed [`principledVolume`](../../principledVolume.md)
contract, not a fluid simulation or an implementation of its renderer:

- **Cloud:** billows, an eroded cavity, and a thin trailing veil. Density only;
  isotropic, forward HG, two-lobe HG, and colored-absorption presets.
- **Plume:** curling hot core and cooler soot. Distinct density and kelvin fields;
  optional blue RGB-emission ribbon in density-free space. Thermal, explicit,
  combined, pure-scattering/no-thermal, and different-domain presets.

## Build and generate

Requires CMake, C++17, Python 3.11+ and NumPy. The CPU writer uses this repository's
NanoVDB headers; no CUDA, ANARI build, or OpenVDB is required for primary output.

```sh
cmake -S tools/principled_volume -B tools/principled_volume/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/principled_volume/build --parallel
python3 tools/principled_volume/generate.py --asset cloud --resolution 128 --output _out/principled_volume/cloud-cell-128
python3 tools/principled_volume/generate.py --asset plume --resolution 256 --output _out/principled_volume/plume-cell-256
# Arbitrary higher resolution, same physical asset:
python3 tools/principled_volume/generate.py --asset plume --resolution 512 --seed 42 --output _out/principled_volume/plume-cell-512
# Optional configuration; CLI resolution/seed override its values:
python3 tools/principled_volume/generate.py --asset cloud --config tools/principled_volume/config.example.json --output _out/principled_volume/custom-cloud
```

Use a fresh output directory. A completed manifest prevents accidental overwrite.
A failed export can leave partial files; remove them or choose a new directory.
The manifest is written last, after verification. Generated data belongs under
ignored `_out/`, not in source control.

`--resolution N` is the longest-axis **cell count**. Both baseline domains are
2-meter cubes `[-1,-1,0]` to `[1,1,2]`, so their shape is N³ and spacing is `2/N`.
The NanoVDB transform's index zero is the lower **cell boundary**; values are
sampled at `origin + (index + 0.5) * spacing`. This matches VisRTX's cell-centered
lookup and its edge clamping over full voxels, including the outer half-cells.
Diagnostic subdomains are fixed 1-meter cubes with approximately half as many
cells and their own spacing. All coordinates are object-space meters. World-space
scale changes optical depth; voxel-count changes do not compensate density.

New manifests use **schema 2** and explicitly record `data_centering: cell`,
`filter: linear`, boundary `origin`, first-center `sample_origin`, spacing, and the
full-cell domain. Historical schema-1 endpoint datasets remain unchanged and must
not be relabeled. Rebuild the writer and regenerate into fresh directories before
preparing new matched scenes; preparation rejects legacy manifests. The old
unflagged writer CLI remains available for historical numerical fixtures.

Random phases, noise frequencies, shapes, and physical domains do not depend on
resolution. The same continuous fields are evaluated at the chosen cell centers.
Odd refinement factors share centers; a factor of two does not. Fine features
remain under-resolved at low N; this is point sampling, not prefiltered downsampling.
The RGB ribbon is particularly thin. The generator evaluates eight x slices at a
time into a temporary memory-mapped file. The writer holds raw values and sparse
construction/serialized/read-back grids in memory: peak usage grows with N³ and
is **not** out-of-core. 512+ should be budgeted accordingly.

Configuration controls are in `config.example.json`: density scale, geometric
noise amplitude (`detail_strength`, meters), temperature range (kelvin), RGB
emission strength, seed, and resolution. Defaults produce approximately meter-sized
features. These are synthetic test fields, not quantitatively validated soot data.

## Looping plume animation

Animation is opt-in; existing static invocations and seeded field values are
unchanged. Generate NVDBs without Vela, then package them separately:

```sh
python3 tools/principled_volume/generate.py --asset plume --animate \
  --output _out/principled_volume/plume-animation-64
python3 tools/principled_volume/package_animation.py \
  _out/principled_volume/plume-animation-64
# Optional representative renders; uses ANARI_LIBRARY and LD_LIBRARY_PATH:
vsrLua tools/principled_volume/render_animation.lua \
  _out/principled_volume/plume-animation-64 32 512
```

Defaults: **48 frames, 64³ cells, four seconds**, seed 42, thermal plume only.
`--resolution`, `--seed`, and `--config` retain their existing precedence; the
animation resolution default is applied before config-file overrides.

| Animation option | Default | Meaning |
| --- | --- | --- |
| `--frames` | 48 | Unique sampled frames, at least 2 |
| `--duration` | 4 | Positive loop duration in seconds |
| `--sway-strength` | 0.08 | Dimensionless sway multiplier in the fixed meter-sized domain |
| `--rise-cycles` | 1 | Nonnegative integer phase cycles per loop; higher octaves use multiples |
| `--pulse-strength` | 0.2 | Relative source modulation, in [0, 1) |
| `--diagnostics` | off | Also export emission, cropped temperature, and offset emission |
| `--overwrite` | off | Replace an existing completed sequence directory |

This remains an analytic visualization asset, **not a fluid simulation**. Smooth
periodic phases move noise upward, sway the centerline, and propagate source
pulses upward. Density and temperature share the evolving core geometry;
temperature still cools with height. No velocity, combustion, or conservation-law
solver is implied. Diagnostic subdomains sample those same continuous functions;
the detached blue emission ribbon stays stationary. Point-sampling and discrete
playback can still alias thin features, particularly at low resolution.

Output layout:

```text
<sequence>/
  manifest.json                 # sequence controls, frame times, paths and hashes
  frames/0000/
    manifest.json               # schema-2 cell-centered field metadata
    density.nvdb
    temperature.nvdb
  frames/0001/...
  scene.vsr                     # produced by package_animation.py
  animations.vsr                # matching object-index bindings and clock
  animation.validation.txt      # fresh-process archive/playback verification
```

Frames cover `[0, duration)`. The VSR timeline adds the first field objects again
at normalized time 1, without duplicate NVDBs. Vela's current clock includes this
closure as an extra endpoint frame (`N+1` clock entries at `N/duration` FPS).
**Its built-in frame-loop playback therefore holds the closure for one extra
frame interval** before wrapping. For exact-duration playback/export, drive
`animationMgr:setAnimationTime((seconds % duration)/duration)` or sample only the
N unique phases. Object references switch discretely; Vela currently selects the
nearest adjacent object even with `"step"`, not an interpolated voxel field.

Packaging requires a Vela `vsrLua` with `addObjectParameterBinding` and
`spatialField` object-key support. It checks input hashes and stable transforms,
imports only fields used by the selected preset, and validates **every frame**
and the loop closure after loading both archives in a fresh process. Use
`--preset combined` or `--preset different_domains` after `--diagnostics` to
exercise optional fields. `--vsr-lua /path/to/vsrLua` selects a particular build.

Archives are a matched pair: load the scene before its animation archive into
an otherwise fresh scene/manager. For example, in a Vela Lua context:

```lua
vsr.io.loadSceneArchive(scene, "/absolute/path/to/sequence/scene.vsr")
vsr.io.loadAnimationManagerArchive(animationMgr, "/absolute/path/to/sequence/animations.vsr")
animationMgr:setAnimationTime(0.25) -- normalized phase, not seconds
```

The scene embeds the NVDB arrays; **all selected frames are memory-resident**.
Exported NVDBs remain independently usable, but VSR playback does not stream them.
Storage and memory grow with frame count and resolution. Budget before increasing
to 128³/256³. The camera carries a -6-stop exposure metadata hint; the supplied
render smoke check uses raw renderer exposure, a black background, and no denoising.
It writes quarter-cycle PNGs plus closure for visual inspection, not Blender
reference comparisons or quantitative transport qualification.

Sequence generation stages all frames before publishing. Existing paths are
refused by default; `--overwrite` accepts only a completed plume-sequence
manifest and replaces the **entire owned directory**, including old archives and
previews. Failed generation leaves an existing sequence intact. There is no
resume support. Packaging has its own `--overwrite` to rebuild archives without
regenerating fields. Do not run concurrent writers against the same directory.

## Output and binding

Each `.nvdb` contains exactly one named, uncompressed float32 or Vec3f grid.
The manifest includes SHA-256 hashes, grid types, transforms, domains, backgrounds,
ranges, units, resolved configuration, material presets, and a camera/light setup.
Grid class is generic rather than a level set. No quantization is used.

Bind each referenced manifest field as an ANARI `nanovdb` spatial field. Field
references in presets are manifest keys, not automatic channel discovery. Apply
all preset parameters and the manifest's explicit volume `bounds`.
These instructions describe consumer behavior; this tool does not add a loader.

- Density is dimensionless; default transmission `exp(-1)` and distance 1 give
  extinction numerically equal to density per meter.
- Temperature is already kelvin: multiplier 1. Zero denotes no thermal source.
- RGB emission is linear sRGB/D65 radiance per meter: multiplier `(1,1,1)` enables it.
- Thermal presets use albedo 0.2, intensity 1, and white tint; no per-temperature
  brightness normalization. Pure-scattering preset deliberately removes thermal emission.
- All sparse backgrounds are zero. Eight active corner cells pin the intended
  field domain independently of nonzero support. Their values are sampled at
  centers, including when zero. NanoVDB world bounds now agree with the explicit
  manifest domain (`origin + dimensions * spacing`); consumers must not shrink
  that domain to nonzero support.
- `temperature_small` samples the **same** continuous temperature field on
  `[-.5,-.5,0]` to `[.5,.5,1]`; outside it, the binding uses zero.
- `emission_offset` samples the same ribbon on `[.5,-.5,.5]` to `[1.5,.5,1.5]`.
  Part of this field lies outside volume bounds and must be clipped by the volume.

Presets are complete relative to documented API defaults. They are not direct
Blender material settings. In particular, the default white scattering albedo
would suppress the plume's absorption-coupled thermal source.

## Verification and diagnostic cases

```sh
python3 -m unittest discover -s tools/principled_volume -v
```

The C++ writer reads each NanoVDB back and checks every sample, value type, name,
transform, index bounds, and full-cell world bounds. `--info INPUT.nvdb` prints
metadata for independent checks. Python checks finite/nonnegative values. Tests cover
repeatable file hashes in the same toolchain, cross-resolution shared samples,
integrated density convergence, seed/scale controls, temperature ranges, emission
in vacuum, preset multipliers, configuration validation, and export manifests.
Cross-platform binary hashes are not promised.

`diagnostics.json` contains four small **constant-medium** configurations and
analytic expected radiances; no voxel files are necessary. They exercise colored
Beer–Lambert attenuation, independent emission in vacuum, an absorbing emitter,
and absence of thermal emission in a purely scattering medium. These are future
renderer tests, not claims that the current renderer passed them.

## VSR scenes and matched Blender references

See **[SCENES.md](SCENES.md)** for the shared scene builder, VSR archive validation,
Blender renders, diagnostic variants, and comparison limitations. Fresh representative
cell-convention assets and their qualification report are under
`_out/principled_volume/cell-v2/`; the historical testbed remains unchanged.

The current cell-centered workflow requires newly generated schema-2 data.
Existing historical testbed images and archives have not been regenerated by this
change. The reference workflow supersedes the original appearance-only previews
under the dataset directories. It uses explicit optical coefficients and calibrated
thermal emission rather than the original artistic brightness curve. The new
`blender_reference.py` takes **scene.json and output.blend**, not the old three-argument
manifest/VDB/output interface. Existing dataset files are unchanged.

Blender needs a secondary OpenVDB file. Build the optional writer using an installed
OpenVDB development package:

```sh
cmake -S tools/principled_volume -B tools/principled_volume/build-vdb \
  -DEXPORT_OPENVDB=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MODULE_PATH=/path/to/openvdb/lib/cmake/OpenVDB
cmake --build tools/principled_volume/build-vdb --parallel
```

The default reference writer is `tools/principled_volume/build-vdb/write_nvdb`;
use `--writer` to select another build. For tests, set
`PRINCIPLED_VOLUME_VDB_WRITER=/absolute/path/to/write_nvdb` to enable secondary
export and scene-archive integration checks.

New cell-mode OpenVDB export shifts each grid transform by half a voxel and adds
one replicated edge-value halo, so its integer-center interpolation matches
VisRTX's clamped reconstruction **inside the logical domain**. The reference shader
still clips to the explicit volume bounds. This does not remove nonlinear thermal
baking differences or the resampling of independent-field domain edges.

OpenVDB's CMake module and dependencies must be discoverable through your package
manager/CMake paths. The primary NanoVDB-only generator still needs no OpenVDB.

No changes to the VisRTX renderer or Vela source are made by these tools.
