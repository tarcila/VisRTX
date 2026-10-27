# Principled volume: technical findings and resolution proposals

Status: investigation complete for the items below; **only explicitly accepted
items are decisions; other proposals remain pending**. No renderer behavior has
been implemented. This supplements
[the agreed model](principled-volume.md), rather than superseding decisions silently.
Repository paths below are relative to `devices/rtx/`.

## 1. Depth semantics: align quality with interactive first

Primary source: [ANARI frame specification source](https://github.com/KhronosGroup/ANARI-Docs/blob/main/chapters/object_types/frame.txt),
`frame_channels` table. The retrieved source specifies:

- `channel.depth`: Euclidean distance to the camera; minimum across pixel samples.
- `channel.normal`: average world-space normal of the first hit.
- `channel.albedo`: average material albedo at the first hit.
- IDs identify geometry primitives, surface/volume IDs, and instance IDs.

Our accepted deterministic center-ray selection requirement conflicts with replacing
standard depth and average albedo with one representative center-ray value. This
is not a numerical implementation detail. Do not ship that replacement silently.
The exact volume-hit interpretation still needs documenting, but it does not erase
the specified multi-sample reduction.

Repository evidence:

- Both interactive and quality center the first sample's pixel position, but
  retain a Halton lens sample when depth of field is enabled.
- Interactive `device/gpu/renderer/raygen_helpers.h` writes depth/IDs only under
  `isVeryFirstRay`: its current metadata is from the first center-pixel sample.
- Quality `device/renderer/Quality_ptx.cu` writes first-bounce depth/IDs for each
  pixel sample, not just the first pixel sample.
- `device/gpu/gpu_util.h::setPixelIds()` updates IDs when `accumDepth()` wins.
- `device/frame/Frame.cu::mapAlbedoBuffer()` resolves the accumulated albedo buffer.
- `device/gpu/cameraCreateRay.h::makePrimaryRay(centerPixel=true)` centers the
  pixel but still obtains its lens sample from Halton. It is not a complete stable
  center-ray implementation for depth of field.

### Accepted resolution — first standalone implementation commit

Quality must match interactive's first center-pixel sample for depth and primitive/
object/instance IDs. Later jittered samples must not replace them. Do this in a
standalone commit at the beginning of implementation, before principled-volume
changes. Document the deliberate difference from ANARI's minimum-depth reduction.
The proposed separate picking extension/channels and deferring alignment were not
selected. Do not add a picking mode or new channels for this prerequisite.

Keep color accumulation and current normal/albedo averaging unchanged. Preserve
current camera/lens sampling and current volume-hit interpretation in this narrowly
scoped commit. Audit all metadata writers, not just surface hits: volume hits,
misses, and light-proxy depth handling must follow the same sample-selection rule.
Test stability across later jittered samples and frame calls, multiple samples per
call, and proper refresh after accumulation reset. The full volume-selection and
neutral-guide design remains subsequent work.

The deterministic selection algorithm retains all accepted rules: first active
support, surface precedence at coincident depth, stable identity ties, and emission-
only selection. A representative ray is constructed without sampling the radiance
RNG: perspective uses pixel center and lens center, orthographic uses pixel center,
and ray-buffer cameras use the supplied per-pixel ray and limits. For perspective,
construct the lens-center ray directly rather than guessing the RNG coordinates
that a disk sampler maps to its center. These are implementation proposals.

Picking depth must follow an explicitly stated distance origin, especially for
orthographic and ray-buffer cameras. Do not confuse the ray parameter with the
standard channel's Euclidean distance to the camera. A geometric first-support
boundary may itself have zero density; evaluate material attributes just inside
the selected support with a documented tolerance rather than divide zero by zero.

### Numerical limitation

A finite ray-march lattice cannot guarantee finding the first nonzero feature of
an arbitrary procedural function. Scalar sample ranges obtained by probing a finite
grid are not a proof of support or extinction bounds. Built-in nearest/trilinear
fields permit cell-aware conservative bounds; custom fields need certified range/
support capabilities, or an explicitly approximate selection contract. This is part
of custom opt-in, not a reason to claim arbitrary custom fields work automatically.

Selection tolerances must be renderer-independent and tested against thin cells,
large world coordinates, nonuniform instances, touching domains, and discontinuities.
Do not use an opacity/brightness threshold to hide a numerical support problem.

## 2. Typed fields: no NanoVDB dependency upgrade indicated

The bundled `external/nanovdb/include/nanovdb/NanoVDB.h` already declares
`GridType::Vec3f`, vector tree/grid types, and major ABI version 32. Use the bundled
header as the compatibility baseline, not upstream master's additional formats.
`cmake/Findnanovdb.cmake` points to this vendored include directory.

Current `NvdbRegularField.cpp` accepts an ANARI_UINT8 byte array and a single grid,
but chooses callable families only for Float/Fp4/Fp8/Fp16/FpN. Its `isValid()` checks
only the array's element type, not that the grid was accepted and assigned a valid
callable. New typed validation must carry explicit successful-finalization state.
This is a relevant integration requirement, not a separate unrelated cleanup task.

`StructuredRegularField.cpp` and `StructuredRectilinearField.cpp` currently accept
scalar fixed-point/float arrays. Inline sampling returns float. The NanoVDB templates
also explicitly return float, so instantiating them with Vec3f alone is insufficient.

### Accepted first-version format matrix

| Field subtype | Scalar | RGB |
| --- | --- | --- |
| structuredRegular | Existing supported scalar formats | ANARI_FLOAT32_VEC3 |
| structuredRectilinear | Existing supported scalar formats | ANARI_FLOAT32_VEC3 |
| nanovdb | Existing Float/Fp4/Fp8/Fp16/FpN | GridType::Vec3f in the existing UINT8 byte buffer |
| nanovdbRectilinear | Existing Float/Fp4/Fp8/Fp16/FpN | GridType::Vec3f in the existing UINT8 byte buffer |
| custom | Existing scalar clients unchanged | Explicit typed-capability opt-in |

No implicit scalar-to-RGB broadcast, RGB-to-scalar reduction, RGBA opacity channel,
sRGB decoding, Vec3d, or packed integer color formats in the initial proposal.
RGB field samples are linear numerical data; the consuming property determines
whether they represent albedo, transmission, or emission. Keep one-grid-per-object
semantics and use separate objects for independent simulation grids.

For dense RGB textures, propose internal float4 storage and tex3D<float4> sampling,
exposing only xyz. This follows the existing CUDA image-texture expansion pattern
in `device/utility/CudaImageTexture.cpp`; no public fourth component is introduced.
Sparse RGB interpolation uses Vec3f-aware samplers and typed result conversion.
Nearest/linear filtering retains current field coordinate conventions.

Verification: a C++17 syntax-only compile against the bundled headers instantiated
`Vec3fGrid::getAccessor()` and `math::createSampler<0>` / `<1>`, with static assertions
that both filtered results are Vec3f. It passed. This verifies host template
availability, not CUDA/OptiX integration, vector acceleration bounds, or runtime
rendering.

Propose `VISRTX_SPATIAL_FIELD_RGB` as a separate capability used by the principled
extension. Query metadata must list the exact array formats. Scalar consumers
(TF1D, isosurfaces) must reject RGB fields instead of taking red or luminance.
A read-only field property such as `valueType` (ANARI_DATA_TYPE) can expose runtime
output arity; this is a proposal, not an existing property.

### Custom-field compatibility

`SpatialFieldRegistry.h` registers a factory; it has no typed capability descriptor.
`CustomFieldSamplerInline.h` uses existing float-returning callable slots. Preserve
those slots, the scalar registration entry point, and scalar consumers. Add a
versioned opt-in descriptor/registration route for typed samples and conservative
range/support queries. Do not add pure virtual requirements to existing subclasses
or reinterpret their opaque payload. Audit binary layout and build-time plugin
assumptions before claiming binary ABI compatibility; preserving a source signature
alone does not prove it. Unsupported custom principled bindings fail explicitly.

### Conservative ranges

Cache property-independent component ranges with the field; derive medium-specific
optical bounds in the volume. For nonnegative density, transmission C in (0,1],
and distance D>0, a conservative per-channel extinction upper bound is:

```
sigma_t_max = d_max * (-log(C_min)) / D_min
```

Bounds must include all contributing interpolation neighbors, sparse tile/background
values, outside values, and field/ROI boundaries. Different field layouts require
covering each field's footprint for the queried spatial region. Independently
bounding inputs can be loose but remains conservative. Finite sampled estimates
cannot replace certified bounds. Overlapping-medium bounds add. Independent emission
requires a separate occupancy/range track even where sigma_t_max is zero.

## 3. Error handling: completed is not successful

Primary source: [ANARI frame operations](https://github.com/KhronosGroup/ANARI-Docs/blob/main/chapters/rendering_frames.txt).
`anariFrameReady(ANARI_WAIT)` returns true on completion, not success. A channel
that cannot be mapped returns NULL, zero dimensions, and ANARI_UNKNOWN. The spec
requires valid scene objects before rendering; a graceful invalid-input policy is
an additional vendor guarantee, not a reason to redefine readiness.

Repository evidence:

- `Frame::isValid()` checks renderer/camera/world, not every volume binding.
- `Group::partitionValidVolumes()` warns and skips invalid volumes. A world may
  therefore render without a bad volume unless validation runs before partitioning.
- `Frame::renderFrame()` reports an error and returns for invalid frame objects.
- Samples, accumulation, denoising, and output conversion currently proceed after
  launch without a medium-data error gate.
- `Frame::frameReady()` and `ready()` report event completion. No failed-result
  property exists in the inspected `getProperty()` branches.

### Accepted policy: diagnose, omit invalid primitives, continue

The proposed frame-wide failure policy, `renderStatus` property, and failed-mapping
contract were rejected. Diagnose invalid/incomplete primitives, omit them, and
render remaining valid content. Keep readiness, mapping, and completion semantics
unchanged. Invalid whole-frame configuration remains subject to existing frame
handling; this decision concerns invalid primitives.

Validation is lightweight and confined to commit/finalize: required inputs,
parameter/object types, supported field formats and capabilities, structural
metadata, bounds, and constant/outside-value constraints. Reuse group partitioning
for known-invalid volumes and identify offending bindings in diagnostics.

Do not scan field contents for NaNs or range violations. Do not add per-sample
finite/range checks, GPU invalid-sample records, or mid-render recovery. Those
previous proposals were withdrawn after clarification of the intended scope.
Field authors/providers guarantee valid samples and effective products. Custom
providers also guarantee conservative bounds; their presence/metadata can be
checked upfront, but finite sample probing cannot certify arbitrary functions.
Invalid field contents are outside the supported input contract; detection and
graceful output are not promised.

Constructing conservative ranges/acceleration structures needed by the integrator
is still necessary and is not a separate input-validation pass. Built-in range
construction must account for interpolation neighbors, sparse backgrounds, and
outside values, assuming valid data. Numerically stable formulas for valid inputs
remain an implementation responsibility, distinct from testing for bad samples.

Inactive-input validation details remain to be specified within this lightweight
policy; they must not introduce field-content scans or runtime validation.

## 4. Thermal calibration: physics is known, RGB convention must be explicit

Evidence:

- [ANARI lights](https://github.com/KhronosGroup/ANARI-Docs/blob/main/chapters/object_types/lights.txt)
  specifies radiance in W/sr/m^2, intensity in W/sr, and irradiance in W/m^2.
- `device/gpu/sampleLight.h` uses inverse-square point-light intensity and radiance
  terminology; this is not a hidden photometric/candela pipeline.
- `PhysicallyBasedShader_ptx.cu::evalSensitivity()` uses an XYZ-to-linear-sRGB/D65
  matrix. Texture inputs explicitly distinguish sRGB decoding from raw linear data.
- [PBRT Blackbody function](https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/spectrum.h)
  evaluates Planck radiance. Its separate `BlackbodySpectrum` wrapper normalizes
  its peak, so that wrapper is not an absolute-radiance reference for this design.
- [Cycles principled closure](https://github.com/blender/blender/blob/main/intern/cycles/kernel/svm/closure.h)
  currently uses `sigma*T^4/pi` with an additional 1e-6 factor and a color fit.
  Its explicit emission is independent of density. Do not copy that brightness
  convention while claiming our absorption-coupled model has identical semantics.

Proposed reference calculation, using wavelengths in meters:

```
B_lambda(T) = 2*h*c^2 / (lambda^5 * expm1(h*c/(lambda*k*T)))
h = 6.62607015e-34 J s
c = 299792458 m/s
k = 1.380649e-23 J/K
```

B_lambda is W/(sr m^2 m). Numerical integration with wavelengths in nanometers
must include the 1e-9 differential conversion. T=0 returns zero. Use stable large-
exponent evaluation; do not allow overflow in an intermediate exponent to yield
NaN emission. Do not clamp temperature to a short color-fit range silently.

Integrate Planck radiance against a pinned standard observer and convert XYZ to
linear sRGB/D65 using one documented fixed radiometric normalization. A promising
explicit normalization is to map a D65 reference spectrum, normalized to one unit
of integrated visible-band radiance, to RGB white one; the same constant and matrix
must apply at every temperature. This is a proposed RGB encoding convention, not
something established by the existing light code. Pin the observer, illuminant,
wavelength bounds, quadrature, and negative-RGB gamut policy before freezing it.
Do not use per-temperature peak/luminance normalization or insert the photometric
683 lm/W factor into a radiometric RGB convention without a declared conversion.

Absolute light comparisons assume world length units correspond to meters. The
accepted inverse-scene-unit optical model remains useful in arbitrary authoring
units, but SI comparisons with fixed point-light intensities require consistent
scene/light scaling. No automatic geometry unit conversion is proposed.

### Numerical check performed during this investigation

A double-precision Python reference integrated Planck radiance on 40,000 logarithmic
wavelength intervals from 1 nm to 1 m and compared against Stefan–Boltzmann/pi,
with sigma computed from the same SI constants. Relative errors were below 1e-7
(at the tested temperatures, below 6e-14). A separate 10,000-interval integration
covered 360–830 nm:

| Temperature K | Total radiance W/(sr m^2) | 360–830 nm radiance W/(sr m^2) | Visible-band fraction |
| --- | --- | --- | --- |
| 1000 | 18049.3624 | 0.512180314 | 0.0000283766 |
| 2000 | 288789.798 | 7152.37492 | 0.0247667 |
| 3000 | 1461998.35 | 231669.565 | 0.158461 |
| 6500 | 32219239.9 | 17778386.1 | 0.551794 |
| 10000 | 180493624 | 83753510.5 | 0.464025 |

The reference is reproducible with Python's standard library:

```python
import math
h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
sigma = 2 * math.pi**5 * k**4 / (15 * h**3 * c**2)
def B(lam, T):
    x = h*c/(lam*k*T)
    return 0.0 if x > 700 else 2*h*c*c/(lam**5 * math.expm1(x))
def integrate(T, a, b, n, logarithmic=False):
    lo, hi = (math.log(a), math.log(b)) if logarithmic else (a, b)
    step, total = (hi-lo)/n, 0.0
    for i in range(n+1):
        x = lo + i*step
        lam = math.exp(x) if logarithmic else x
        weight = 0.5 if i in (0, n) else 1.0
        total += weight * B(lam, T) * (lam if logarithmic else 1.0)
    return total * step
for T in (1000., 2000., 3000., 6500., 10000.):
    full = integrate(T, 1e-9, 1., 40000, True)
    reference = sigma*T**4/math.pi
    visible = integrate(T, 360e-9, 830e-9, 10000)
    assert abs(full-reference)/reference < 1e-7
    assert 0 < visible < full
    print(T, full, visible, visible/full)
```

This verifies units and scalar radiance integration, not RGB calibration or GPU
transport. It demonstrates why total T^4 radiation times an arbitrary normalized
visible color is not automatically a calibrated visible-light model: at 1000 K
almost all energy lies outside that visible interval.

## 5. Denoising and guides

`Frame::mapAlbedoBuffer()` and `mapNormalBuffer()` currently resolve the same
accumulation buffers used by the denoiser. Changing public AOV definitions therefore
requires separating their storage/evaluation from guides when semantics differ.
`Denoiser.cu` passes float3 albedo/normal buffers to OptiX. This inspection alone does
not establish that a zero-normal sentinel is accepted or beneficial for every
configured denoiser mode.

Propose public no-normal encoding (0,0,0), preserving zero through normalization.
Validate OptiX's guide requirements before selecting an internal neutral encoding.
If no valid mixed surface/volume guide policy is available, use color-only denoising
for affected frames rather than invent a physical normal. This fallback is a
proposal needing quality/performance tests, not a verified best choice.

## Decision summary

Decision state:

1. Align quality depth/IDs with interactive's first center-pixel sample in the
   first standalone implementation commit. Preserve normal/albedo averaging and
   document the difference from ANARI's minimum-depth reduction. No separate
   picking extension is selected.
2. The concrete field-format matrix and versioned custom capability route are
   accepted, including no implicit broadcast/channel reduction/sRGB decoding.
3. The failed-frame proposal and runtime field-validation/recovery proposals are
   rejected. Use lightweight commit/finalize validation, diagnose and omit invalid
   primitives, and render valid content. Field authors/providers guarantee data
   validity; no NaN scans or per-sample validity checks are required.

Still requiring targeted evidence/measurement:

- Direct retrieval of the official CIE tables was attempted at
  `https://files.cie.co.at/CIE_xyz_1931_2deg.csv` and
  `https://files.cie.co.at/CIE_std_illum_D65.csv`; both failed DNS resolution in
  this environment. Access was subsequently granted; both URLs then returned
  HTML 404 responses rather than CSV data. Resolve the dataset locations before
  using their contents; fetch-tool success does not establish valid table data.
- Pin observer/D65 tables, choose fixed RGB normalization and gamut handling, and
  compare thermal emission to existing light/material radiance numerically.
- Verify OptiX neutral-guide requirements and mixed-scene denoising quality.
- Establish a bounded-support selection algorithm and tolerances, including custom
  field opt-in guarantees. Arbitrary sampled functions cannot promise exact support.

No extension JSON, callable signatures, or frame behavior was changed. Do not treat
these proposals as implementation authorization or a completed conformance claim.
