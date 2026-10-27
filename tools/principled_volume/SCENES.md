# Principled-volume renderer testbed

These are **future VisRTX tests**, not a workaround for missing renderer support.
VSR constructs and reloads archives without creating an ANARI device. Blender
provides checked reference inputs and images; no VisRTX renders are claimed.

## Fresh cell-convention qualification

The refreshed output tree is `_out/principled_volume/cell-v2/`:

- `datasets/`: cloud and plume at 128 and 256 full cells per axis.
- `testbed/`: fresh `cloud-128`, `plume-128`, and `plume-different_domains-256`
  archives and Blender references. Other variants have not yet been rebuilt.
- `checks/`: fresh homogeneous RGB checks, plus a transparent-background
  vacuum-emission alpha check.
- `asset_audit.json` and `qualification.md`: input hashes, serialized grid/domain
  checks, archive payload identities, saved settings, output fingerprints, results,
  and the limits of this qualification.

New references explicitly select FULL volume precision, zero volume clipping,
LINEAR interpolation, zero direct/indirect sample clamping, fixed seed, and no
adaptive sampling or denoising. These remove the historical HALF/clipping defaults
as input approximations; they do not establish convergence or renderer parity.
The regenerated three images remain finite-sample baselines (256/256/128 spp).

Dataset manifests record generator/writer fingerprints and Python/NumPy versions.
Scene descriptions record preparation/builder/optical-reference/writer/Vela
fingerprints and the baked VDB hash. Render statistics record the actual Blender
executable/version, reference-script fingerprint, selected compute backend/devices,
precision, clipping, filtering, step/bounce settings, and clamping controls.

## Historical generated cases

Existing outputs under `_out/principled_volume/testbed/` (ignored by VCS) were
created from schema-1 endpoint data. They remain unchanged for provenance and
inspection. The generator now uses VisRTX's full-cell convention; create fresh
schema-2 datasets and scene directories for new comparisons rather than relabeling
these archives/images. The historical set is:

| Directory | Purpose |
|---|---|
| `cloud-128`, `cloud-256` | Forward-scattering cloud under `sunflowers_puresky_4k.hdr`. |
| `plume-128`, `plume-256` | Calibrated thermal plume illuminating a matte floor and receiver card. No external light. |
| `emission-128`, `emission-256` | Blue RGB ribbon over a checkerboard, with volume density zero. Background remains visible through emission. |
| `cloud-isotropic-256` | Compare with forward scattering at identical scene settings. |
| `cloud-two_lobe-256` | Weighted sum of forward/backward HG lobes. |
| `cloud-colored_absorption-256` | Colored transmission and scattering/absorption partition. |
| `plume-combined-256` | Independent RGB and absorption-coupled thermal sources together. |
| `plume-pure_scattering_no_thermal-256` | Hot but completely black in the dark studio: no absorption, hence no thermal source. |
| `plume-different_domains-256` | Smaller temperature domain and offset RGB-emission domain, clipped by explicit volume bounds. |
| `emission-smoke-256` | Same checkerboard/ribbon, with smoke enabled. The central smoke attenuates the background; the spatially separate ribbon remains primarily in vacuum. |
| `checks/` | Four homogeneous Blender analytic checks and measured/expected values. |

Each scene directory contains:

- `scene.json`: canonical resolved scene description, source hashes, camera,
  geometry, environment orientation, volume bindings, rendering settings, caveats.
- `scene.lua`: the same data serialized as a **trusted Lua module**.
- `scene.vsr`: archive containing original scalar/RGB fields, surfaces, camera,
  and environment data. The VSR scene retains `principledVolume`, not TF1D.
- `scene.vsr.validation.txt`: archive save/reload results; explicitly not a render test.
- `reference.vdb`: secondary Blender asset; density plus derived RGB emission
  where applicable. `reference_j.nvdb` is the intermediate calibrated emission grid.
- `reference.blend`, `reference.exr`, `reference.png`, `reference.stats.json`:
  Blender scene, 32-bit linear HDR result, AgX display image, and image statistics.

The main six renders use 256 samples; diagnostic variants use 128. They are
**noisy Monte Carlo baselines**, not converged golden images. Denoising and adaptive
sampling are disabled. Increase samples for close pixel comparisons.
The old previews directly inside `cloud-128/` and `plume-128/`, and intermediate
prototypes under `scenes/`, are superseded by this `testbed/` set.

## Rebuild a scene

Requirements: generated datasets, Python/NumPy, OpenVDB-enabled writer (see
[README.md](README.md)), and Vela with `vsr.box3.new` plus Vec3f NanoVDB support.
Tested using Vela's 2026-10-27 package and Blender 5.2.2 LTS.
Rebuild both writers after the sampling change. The default OpenVDB-enabled
executable is `build-vdb/write_nvdb`; supply `--writer` if yours has another path.
The older local `build/write_nvdb_vdb` binary does not gain new export modes merely
because its source was edited.

```sh
python3 tools/principled_volume/prepare_scene.py \
  --scene cloud --dataset _out/principled_volume/cloud-cell-256 \
  --hdri /data/Envmaps/sunflowers_puresky_4k.hdr \
  --output _out/principled_volume/my-cloud --samples 256 --compute-device OPTIX

blender -b --python tools/principled_volume/blender_reference.py -- \
  _out/principled_volume/my-cloud/scene.json \
  _out/principled_volume/my-cloud/reference.blend
```

First generate the example's `cloud-cell-256` dataset with the updated generator.
For plume, choose `--scene plume --dataset .../plume-cell-256`; default preset is
`thermal`. For the checkerboard, choose `--scene emission --dataset .../plume-cell-256`;
default preset is `vacuum`, with `smoke` as the alternative. Cloud/plume accept
preset names from their source manifests. Change resolution by selecting another
dataset directory; the scene composition remains fixed.

Use a fresh output directory. Preparation verifies source field hashes, bakes the
reference asset, writes both scene descriptions, and invokes the Lua builder. A
failure may leave partial output; `scene.vsr.validation.txt` is the success marker
for archive validation. To rebuild only the VSR archive after a Lua change:

```sh
vsrLua tools/principled_volume/build_scene.lua \
  _out/principled_volume/my-cloud/scene.lua \
  _out/principled_volume/my-cloud/scene.vsr
```

Preparation's `--compute-device CPU|CUDA|OPTIX` records the Blender backend
(default CPU); GPU requests fail if no matching device is available instead of
silently falling back. The reference builder applies the recorded selection.
Blender options after `--`: `--samples N` overrides sampling; `--no-render` saves
only the `.blend`. The `.blend` retains paths to the VDB and HDRI. Move those assets
with care or regenerate using new source paths. The JSON/Lua modules record local
absolute paths; they are rebuild descriptions, not portable packages by themselves.
VSR archives embed their scene data and do not require the original field files
for reload, as exercised by their normal archive deserializer.

## Shared scene conventions

- Z-up, one unit = one meter. Identity volume transform; bounds from the dataset.
- New schema-2 data use N full cells over each axis, spacing L/N, boundary-origin
  NanoVDB transforms, and samples at origin+(index+0.5)*spacing. Each field has
  explicit `cell` centering, linear filtering, and a manifest-domain ROI in VSR.
  Archive checks verify these parameter values and enabled states after reload.
- New secondary OpenVDB grids translate their integer centers by half a cell and
  replicate a one-voxel halo to match clamping in the outer half-cells. The shader
  clips the padded grid to the volume domain. Padding does not make a raw VDB
  sampler return zero outside the logical domain by itself.
- The same triangle vertices/indices and diffuse colors are used in both tools.
- A 32-degree **vertical** perspective FOV in both; ANARI receives radians.
- Main cloud uses the native HDRI radiance at scale one, without another sun.
  Default rotation is -126.1669921875 degrees about +Z. This places the brightest
  sunflower HDRI texel behind the volume for the fixed camera (peak approximately
  UV 0.60046, 0.73901 in bottom-up coordinates). `--environment-rotation` overrides
  it; other HDRIs need a suitable rotation of their own.
- Blender evaluates the same explicit lat-long UV equations as VisRTX
  `light/HDRI.cpp` and `gpu/gpu_util.h`, rather than assuming native Blender
  environment longitude conventions. Both interpret HDR pixels as linear RGB.
- Plume studio has black environment and no lamps. Receiver/floor brightness is
  therefore caused by volumetric emission. The RGB diagnostic uses a uniform
  radiance-0.5 environment for readable matte checker tiles.
- Display exposure is 0 stops for cloud/checker and -3 for plume. EXR stores
  scene-linear radiance without AgX/exposure baking; PNG uses AgX and exposure.
- Renderer preference `quality` and exposure are recorded as camera metadata and
  in JSON; this does **not** force a viewer's renderer or postprocessing choices.
  Select the archived camera and suitable settings when backend support arrives.

## Optical correspondence

`reference_optics.py` independently implements the documented contract, not a
call into VisRTX's device code. Original density, temperature and emission fields
remain bound separately in VSR. The Blender reference uses:

1. Density multiplied by `sigma_s / density` in Volume Scatter closures.
2. Density multiplied by `sigma_a / density` in Volume Absorption. Cycles uses
   `sigma_a = Density * (1 - Color)`, so the shader performs that conversion
   explicitly, including RGB extinction. The homogeneous tests verify it.
3. Two additive HG scatter closures weighted by `(1-secondaryWeight)` and
   `secondaryWeight`; no averaging of anisotropy.
4. A separate Emission closure containing calibrated `j`, without a density
   multiplier. This preserves emission in vacuum. Explicit volume bounds clip
   both extinction and emission, including VDB voxel-padding regions.

Thermal RGB is integrated from Planck spectral radiance, using the repository's
CIE 1931 2-degree and D65 CSV tables over 360–830 nm with 1 nm trapezoidal integration.
D65 with integrated spectral radiance one maps approximately to RGB white.
Negative out-of-gamut RGB components are clipped with Y-preserving rescaling;
HDR components are not capped. A 1 K LUT interpolates the resulting function.
Source table hashes are recorded. Respect the CIE dataset attribution/licensing
metadata shipped alongside those CSV files if redistributing the inputs.
There is no artistic blackbody gain or per-temperature brightness normalization.

### Important limitations

- **Nonlinear interpolation:** Blender's derived emission grid is now baked at
  density cell centers (historical assets used vertices). Interpolating this `j`
  differs from first interpolating temperature
  and then evaluating `sigma_a * B(T)`. Coefficients match at the sample locations,
  not at every point. This is particularly relevant at thermal edges and differing
  field domains; those diagnostic boundaries are resampled for Blender.
- Scattering/absorption are linear in the baseline density and therefore do not
  have this thermal nonlinearity. The current scene builder is for these generated
  manifests, not an arbitrary-field principled-volume implementation.
- HG mixture construction is explicit, but no numerical phase-function or
  multiple-scattering cross-renderer comparison has been performed.
- HDRI orientation is matched by source equations and shared inputs, not by an
  end-to-end VisRTX render. HDRI file radiance has no independently measured SI
  calibration, so physical light-source comparisons need additional calibration.
- Blender step size, bounce limits, filtering, and Monte Carlo estimators differ
  from a future VisRTX renderer. Historical inspection confirmed LINEAR filtering,
  HALF volume precision, and clipping 0.001 in the selected historical scenes.
  The fresh cell-v2 references explicitly switch to FULL precision and zero
  clipping, with actual values recorded in their statistics. This does not measure
  historical quantization error or establish step/bounce/noise convergence.
- Checkerboard images do not test zero-alpha emission: the opaque background makes
  framebuffer alpha opaque. The fresh homogeneous vacuum-emission check now uses
  a transparent background and separately asserts zero alpha while retaining
  nonzero RGB; historical results do not gain that evidence retroactively.

## Verification

```sh
python3 -m unittest discover -s tools/principled_volume -v
blender -b --python tools/principled_volume/check_blender_optics.py -- \
  tools/principled_volume/build-vdb/write_nvdb _out/principled_volume/cell-analytic-checks --compute-device OPTIX
```

The Python tests cover generation, serialized full-cell metadata, center values,
CIE normalization, thermal LUT accuracy/brightness, absorption coupling, cell-domain
membership and edge clamping, legacy-reference sampling, and NanoVDB round-trip.
Optional OpenVDB tests reread Float/Vec3f exports and check centers and outer
half-cells; an optional Vela test prepares/reloads a small different-domain scene.
Set `PRINCIPLED_VOLUME_VDB_WRITER` when the optional writer is built elsewhere.
Every newly prepared archive verifies typed bounds, optical float types, field
references/byte counts, cell centering/filter/ROI, object counts, camera FOV, and
nonempty scene graph after reload. The scalar/RGB file contents
are separately verified by the dataset writer; archive verification is not a full
byte-for-byte comparison of every restored array.

When explicitly run, the Blender checks render the **actual reference shader** through a two-meter
homogeneous slab and compare mean linear RGB to analytic expectations with 2%
relative / 0.002 absolute tolerance. The values below are historical schema-1
results, not a rerun after the cell-convention update:

| Case | Expected RGB | Measured RGB (rounded) |
|---|---|---|
| Colored absorption | 0.25, 0.0625, 0.015625 | 0.250035, 0.062600, 0.015587 |
| Emission in vacuum | 2, 4, 8 | 1.999643, 3.999287, 7.998573 |
| Absorbing emitter | 1.729329, 2.593994, 3.458659 | 1.729122, 2.593682, 3.458243 |
| Pure scattering, no illumination/emission | 0, 0, 0 | 0, 0, 0 |

The fresh check results are recorded separately under `cell-v2/checks/results.json`.
All four RGB checks and the vacuum-alpha check passed in that run. They are
bounded shader-mapping checks, not convergence or VisRTX acceptance tests.

The historical generated hot/pure-scattering plume also rendered exactly black. Thermal
calibration is verified numerically on CPU; the black control only checks that
no source was supplied and no illumination leaked into the scene.

These checks provide a useful baseline for implementing VisRTX. They do not
constitute approval of unsupported VisRTX transport or a claim of renderer parity.
