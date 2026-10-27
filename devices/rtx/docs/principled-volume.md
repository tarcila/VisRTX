# Principled volume: agreed model and proposed interface

Status: **optical/behavioral decisions accepted; interface draft, not implemented**.
This document records the design discussion. It does not advertise device support.
See the [survey](principled-volume-survey.md) for external precedents and the
[decision record](adr/0011-explicit-volume-optical-models.md) for compatibility intent.
The [technical findings](principled-volume-technical-findings.md) propose concrete
field formats, validation handling, and thermal calibration. They also identify
an ANARI depth conformance difference. The accepted local decision is to align
quality depth/IDs with interactive's first center-pixel sample, as the first
standalone implementation commit; no new picking channels are introduced.

## Purpose and scope

`principledVolume` is a bounded participating medium for VFX: smoke, fog, clouds,
and fire. Its optical meaning is independent of the renderer. A new RTX-device
vendor extension will expose it, with semantics another ANARI device could adopt.
`VISRTX_VOLUME_PRINCIPLED` is a proposed identifier, not a registered capability.

The subtype supplies artist-facing optical controls, not raw absorption and
scattering coefficients as a competing parameterization. It has no transfer-function
arrays, `valueRange`, implicit named-grid lookup, or built-in ramps. Remapping is
an upstream authoring operation. This does not prohibit future remapping facilities.

## Accepted optical model

At position x, let d be dimensionless density, C the RGB transmission color, D the
reference distance in world-space scene units, and a the RGB scattering albedo:

```
sigma_t = d * (-log(C)) / D
sigma_s = a * sigma_t
sigma_a = (1 - a) * sigma_t
j = emission + blackbodyIntensity * blackbodyTint * sigma_a * B(temperature)
```

RGB operations are componentwise. B(T) is blackbody radiance converted into the
device's linear RGB convention; its absolute calibration is not yet frozen.
Emission is a source coefficient (radiance per scene-unit path length), not a
source color multiplied implicitly by density. It may be nonzero when extinction
is zero. Thermal emission vanishes when absorption is zero. Temperature is kelvin;
zero produces no thermal emission. Blackbody intensity one and white tint are the
intended calibrated thermal model; other settings are artistic adjustments.

For homogeneous density one, unscattered transmission through distance D is C.
Density two squares this transmission. Albedo partitions extinction without changing
unscattered transmission. A red transmission color preserves red light; it does
not mean preferentially absorbing red light. No separate absorption-color control
is exposed.

The phase function is a convex mixture of two normalized Henyey–Greenstein lobes.
Zero anisotropy is isotropic, positive is forward, negative backward. Secondary
weight zero selects the primary lobe. A precise incident/outgoing direction
convention must be documented with the implementation to avoid a sign reversal.

Distance is measured along the world-space ray. Scaling an instance changes optical
thickness without automatically compensating density. Resampling the same physical
field at a different voxel resolution must not change the medium.

Overlapping media, including principled/TF1D mixtures and TF1D-only overlaps, add
sigma_a, sigma_s, and j. Neither subtype takes precedence. Scattering kernels add
with their scattering coefficients: for RGB coefficients, each channel's effective
phase mixture can differ. Do not average anisotropy parameters or choose a medium
by traversal order. Sampling probabilities and RGB throughput compensation remain
an estimator design decision.

## Accepted domain and binding contract

- A finite object-space `bounds` box is required. Missing bounds invalidate the
  object; there is no implicit infinite medium or field-derived domain.
- The box is the medium domain. It clips all contributions, but does not stretch
  or extend field data. Constant fog fills the box.
- Every optical input can be constant or field-driven. Different inputs may use
  different spatial layouts and extents in the volume's object coordinate system.
- A property equals its constant times its optional field sample. Without a field,
  the constant is the value. Bindings are explicit object references, not channel
  names searched in a file.
- Each binding has an outside value, applied before the constant multiplier.
  Defaults are zero for density, explicit RGB emission, and temperature; one for
  the remaining controls. Inside-domain sparse background values remain the
  field's responsibility and are not replaced with the outside value.
- No field contributes outside the volume domain. Conversely, emission outside the
  density field can contribute inside that domain.

The binding's sampling domain is the intersection of field data bounds and field
ROI. Outside that intersection, use the binding's outside value. A field ROI
restricts only that input, not other inputs or the volume domain. An unrestricted
ROI does not make finite field data infinite. Procedural fields use their declared
`bounds()` intersected with ROI; they do not need a separate data-extent mechanism.

RGB properties bind one typed RGB-valued spatial field, not three separate scalar
fields. Extend typed field sampling while preserving existing scalar consumers and
custom-field compatibility; do not silently change scalar callable signatures.
Initial scalar and RGB support covers `structuredRegular`, `structuredRectilinear`,
`nanovdb`, and `nanovdbRectilinear`. Custom fields explicitly opt into new typed
sampling/integration capabilities, including required conservative bounds.
Unsupported combinations must diagnose the limitation, not discard channels.
Accepted initial RGB formats are ANARI_FLOAT32_VEC3 structured arrays and NanoVDB
GridType::Vec3f grids; existing scalar formats remain supported. No implicit
scalar broadcast, RGB channel reduction, or sRGB decoding is introduced. Custom
capabilities use a versioned opt-in route; exact capability packaging remains open.

Bindings use each field's existing object-space coordinate mapping; there is no
additional per-binding transform in the first version. Different origins, spacings,
resolutions, and rectilinear axes remain supported. Independent rotation/shear
requires a field representation supporting that mapping or upstream resampling.
The volume instance transform moves all its inputs together.

Test binding-domain membership before filtering. Inside, use the field's existing
interpolation and edge behavior; outside, return the binding's outside value without
blending it into interior samples. ROI is a hard restriction, not a feather.
Volume bounds require finite coordinates and strictly positive extent on every
axis. An empty field-bounds/ROI intersection returns the outside value everywhere;
it does not by itself invalidate the volume.

## Accepted parameter spelling; proposed ANARI type encoding

The parameter names below and the `P.field` / `P.outside` spelling are accepted.
ANARI type encoding remains proposed. No query JSON should advertise these inputs
until the implementation supports the complete contract. Propose ANARI_FLOAT32 for
scalar controls and ANARI_FLOAT32_VEC3 for RGB controls.

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `bounds` | ANARI_FLOAT32_BOX3 | required | Finite medium domain in object space |
| `density` | float | 1 | Dimensionless multiplier |
| `scatteringColor` | vec3 | (1,1,1) | Scattering albedo |
| `transmissionColor` | vec3 | (exp(-1),exp(-1),exp(-1)) | Unscattered transmission at reference distance and density 1 |
| `transmissionDistance` | float | 1 | Reference world-space distance |
| `anisotropy` | float | 0 | Primary HG lobe |
| `secondaryAnisotropy` | float | 0 | Secondary HG lobe |
| `secondaryWeight` | float | 0 | Secondary phase-lobe weight |
| `emission` | vec3 | (0,0,0) | Independent RGB emission coefficient |
| `temperature` | float | 1 | Kelvin constant or scale for a temperature field |
| `blackbodyIntensity` | float | 0 | Thermal emission multiplier |
| `blackbodyTint` | vec3 | (1,1,1) | Artistic thermal tint |

For each optical property P (not `bounds`), use these accepted names:

| Parameter | Type | Meaning |
| --- | --- | --- |
| `P.field` | ANARI_SPATIAL_FIELD | Optional scalar or RGB field matching P |
| `P.outside` | Same numeric type as P | Sample used outside that field's domain |

The accepted temperature default of one makes a kelvin field pass through unchanged.
Without a field it means 1 K; blackbody emission remains off by default. For uniform
fire, users set an actual kelvin temperature. Emission and blackbody-intensity
fields similarly require nonzero constant multipliers to enable them; merely
binding a field does not replace their zero defaults.

Use one density multiplier (`density`), not both `density` and `densityScale`.
Base volume `id` and `visible` should continue to apply. No new public binding
object is needed for this spelling. This proposal does not overload an optical
constant parameter with an object type.

## Accepted validation and defaults

The input contract requires finite values. Effective values after multiplication
must satisfy the following ranges; these are not a requirement to scan field data
or check each sample at runtime:

- Density, RGB emission, temperature, blackbody intensity/tint: nonnegative.
- Scattering albedo and secondary weight: [0,1].
- Transmission color: (0,1]; exact zero would imply infinite extinction.
- Transmission distance: strictly positive.
- Anisotropies: strictly between -1 and 1.

Invalid constants invalidate the object with diagnostics, rather than silently
changing the material. Invalid/incomplete primitives are diagnosed and omitted;
rendering proceeds with the remaining valid scene. This supersedes the earlier
proposal to fail the entire frame for an invalid volume. No new frame-result status
property or failed-mapping contract is introduced for this feature.
Validation is lightweight and performed at commit/finalize time: required inputs,
parameter/object types, supported field formats/capabilities, structural metadata,
volume bounds, and constant/outside-value constraints. Do not scan field contents
for NaNs or add per-sample finite/range checks, GPU error records, or mid-render
invalid-data recovery. Field authors/providers are responsible for valid samples,
valid effective products, and conservative bounds supplied by custom fields.
Violations within field data are outside the supported input contract; detection
or graceful rendering is not promised. Acceleration/range construction required
by the algorithm is separate from a validation scan. A temperature-field scale
is not itself a physical temperature sample.

Default appearance is neutral scattering fog, with both emission sources off.
No field is required. Empty/inverted volume boxes are invalid. Validation of
inactive parameters still needs an explicit interface rule.

## Accepted renderer contract

| Subtype | Required behavior |
| --- | --- |
| `quality` | Multiple-scattering transport; both emission sources illuminate surfaces and other volumes |
| `interactive`, `default` | Single-scattering preview with the same optical coefficients, phase, attenuation, and visible emission; may omit illumination originating from emissive volumes |
| Other subtypes | Explicit unsupported-combination failure, not omission or TF-style substitution |

Scattering depth and sampling controls belong to renderers. Single scattering is
not expected to reproduce multiply scattered cloud illumination. Explicit sampling
of volume emitters is an implementation choice, but practical convergence must be
measured, not inferred from theoretical transport support.

## Accepted scalar-opacity reduction

For a volume segment with RGB unscattered transmittance T, scalar opacity is
`1 - luminance(T)`. RGB transport continues to use T; do not replace colored
attenuation with its scalar reduction. Emission does not increase opacity, so an
emission-only region can produce nonzero RGB with zero alpha.

For overlapping media, combine their RGB transmittance before reducing it to
scalar opacity. Do not combine independently reduced per-medium opacities.
Carry an RGB uncovered fraction along the entire camera ray, initially (1,1,1).
Multiply it by each volume segment's RGB transmittance and by (1 - surface opacity)
at surfaces, then apply background coverage. Reduce only at the end:
`alpha = 1 - luminance(uncoveredFraction)`. This coverage track is separate from
radiance throughput and colored surface transmission. An opaque background yields
alpha one; emission does not change coverage. Retaining RGB across successive
segments makes alpha invariant to splitting a medium into adjacent volumes.

Ordinary RGBA cannot encode exact colored attenuation over an arbitrary later
background; this coverage convention is a documented approximation. A separate
RGB transmittance framebuffer channel is outside the first version and would need
its own compositing-output contract. Do not promise exact colored-background
recompositing from initial RGBA output.

## Accepted AOV requirements

**Accepted first implementation commit:** align quality depth and primitive/object/
instance IDs with interactive: the first center-pixel sample is the reference;
later radiance samples must not replace these values. Implement and test this as a
standalone commit before principled-volume work. Do not introduce separate picking
channels or defer alignment. This deliberately differs from ANARI's documented
minimum-depth-across-samples reduction and must be documented as such.

This prerequisite changes depth/ID sample selection only: retain color accumulation
and existing normal/albedo averaging, camera/lens sampling, and current volume-hit
interpretation. It does not implement the full future medium-selection model below.
In particular, current first-center-pixel sampling still uses a sampled lens
position with depth of field. A dedicated fixed-lens selection ray remains part
of the later volume-selection design, not this initial alignment commit.

Volume depth and object/instance IDs use a renderer-independent camera-ray
selection rule, separate from stochastic scattering events. Select the first
optically active region along the camera ray: nonzero extinction or nonzero
emission, not merely entry into a bounding box. Emission-only regions are
selectable. This rule is independent of illumination, exposure, and random
scattering; a faint foreground wisp can win over a visually dominant region.
Depth/IDs are selection metadata, not the location of dominant visible radiance.
Selection competes by distance with existing eligible surface hits; the nearest
candidate wins. A faint active volume in front still wins over a surface behind.
Within the selection tolerance, coincident surface/volume candidates prefer the
surface. Existing surface-picking eligibility is unchanged.

Use a dedicated deterministic camera ray per pixel, not a jittered radiance sample.
Ordinary cameras use pixel centers and a fixed lens sample; other camera types
need an explicit mapping. Selection is a sharp representative view, not an average
of antialiasing, depth of field, or motion blur. Additional radiance accumulation
must not change metadata. Compute selection only when requested; reuse is allowed
while relevant scene and camera state are unchanged.

If active volume contributors begin at the same selected depth, choose one by a
stable ordering of object and instance identity, with an internal tie-breaker for
duplicate user IDs. Never use brightness, density, or BVH traversal order to choose
IDs. Transport still includes every contributor. Exact identity ordering, numerical
support detection, coincidence tolerance, and camera-specific ray construction
remain implementation/interface details to specify.

A principled medium has no intrinsic surface normal. Do not present a density
gradient or the opposite camera direction as its physical normal. Surface normals
remain unchanged. Define a neutral/no-normal representation for volume AOVs and
validate its denoiser-guide treatment; a diagnostic gradient is not a physical
surface orientation. Phase functions remain independent of those gradients.
At the selected volume point, the public albedo AOV is the combined medium's
scattering albedo: componentwise `sum(sigma_s) / sum(sigma_t)`, with zero where
summed extinction is zero. Include all overlapping contributors, regardless of
which contributor supplies the selected ID. Pure absorption and emission-only
regions have zero scattering albedo; TF1D adds absorption but no scattering.
Lighting, exposure, and emission intensity do not alter this quantity.
If the denoiser needs a different guide, keep that internal representation separate
from the public AOV. The precise neutral-normal and guide-buffer encoding remain open.

## Accepted change to transferFunction1D

Both interactive and quality rendering will use emission–absorption:

```
sigma_s = 0
sigma_a = sigma_t = -log(1 - alpha) / unitDistance
j = sigma_t * color
```

Alpha includes the color alpha times opacity. The formula describes the optical
model; alpha exactly one requires the existing limiting/saturation behavior to be
specified and tested, not evaluated as an unguarded logarithm.

In quality rendering this model applies on all transport rays and may illuminate
other objects. Interactive rendering does not gain an obligation to compute that
indirect illumination. There is no legacy scattering mode. This is an intentional
appearance change requiring release notes and migration documentation, not a
claim that existing scenes preserve their images. Additive overlap also replaces
traversal-order compositing for TF1D-only overlaps; document this additional
compatibility impact alongside mixed TF1D/principled overlap behavior.

For migrating the former quality interpretation, conceptually bake the old scalar
extinction into a density field and the old RGB TF into an albedo field. With neutral
C=exp(-1), reference D, set d=sigma_t*D, a=old TF RGB, isotropic phase, and emission
off. This is only valid for admissible albedo values; exact opacity saturation and
interpolation need care. Classifying voxels and then interpolating generally differs
from classifying an interpolated scalar, so baking is not an exact universal
replacement. No native TF inputs will be added solely to make migration lossless.

## Repository evidence and implementation seams

Paths in this section are relative to `devices/rtx/`.

| Existing code | Finding and consequence |
| --- | --- |
| `device/volume/Volume.cpp` | Factory knows TF1D/scivis; BLAS AABB comes from volume GPU bounds. Add a new subtype without changing the meaning of TF1D parameters. |
| `device/volume/TransferFunction1D.cpp` | One field, one TF texture, field-owned opacity bounds. Not a suitable universal representation of optical properties. |
| `device/gpu/gpu_objects.h` | Volume payload contains only TF1D; spatial-field payload has no output arity. RGB transport and typed property bindings need distinct representation. |
| `device/spatial_field/StructuredRegularField.cpp`, `StructuredRectilinearField.cpp` | Accepted data types are scalar. Regular textures clamp addresses; explicit domain checking must precede texture lookup for outside-value semantics. |
| `device/spatial_field/*SamplerInline.h` | `sampleValue()` returns float; RGB fields are not currently supported by this interface. |
| `device/spatial_field/CustomFieldSamplerInline.h` | Custom sampler callable returns float too; changing it affects external custom-field compatibility. |
| `device/spatial_field/space_skipping/UniformGrid.h` | Field stores scalar value ranges and TF-derived opacity bounds. Principled acceleration must account for all bound properties and be owned/keyed by the medium, not shared as one mutable field classification. |
| `device/gpu/volumeIntegration.h` | Dispatch assumes `data.tf1d.field`; loops advance past a chosen segment. Aggregate overlapping-medium transport needs a different interval/evaluation strategy. |
| `device/gpu/volumeIntegrationDetail.h` | Scalar extinction, TF classification, scalar-opacity marching. Density-only skipping loses independently emissive regions. |
| `device/renderer/Quality_ptx.cu` | TF RGB is used as albedo with isotropic scattering; this is the behavior being replaced. |
| `device/gpu/renderer/raygen_helpers.h` | Interactive path composites marched TF color. Principled direct-lit scattering needs its own transport evaluation. |
| `device/CMakeLists.txt`, `device/visrtx_device.json.in` | Query generation uses extension JSON definitions. Add metadata only alongside support; regenerate through the existing build. |
| `device/material/shaders/PhysicallyBasedShader_ptx.cu` | Thin-film conversion uses XYZ-to-sRGB/D65 coefficients. Evidence for RGB primaries, not proof of global radiometric calibration. |
| `device/light/Point.cpp`, `device/gpu/gpu_objects.h` | Light payloads distinguish intensity and irradiance; inspected code does not establish the absolute conversion needed for thermal radiance. |

Proposed internal seam: evaluate one medium at a point into extinction, scattering,
emission, and phase data. TF1D and principled evaluation are two concrete adapters;
field sampling supplies values, not the meaning of those values. Renderer integrators
consume this optical result rather than reinterpret user parameters. Interval
aggregation combines active media before transport. Keep the existing separate
surface/volume TLAS architecture ([ADR 0004](adr/0004-separate-surface-and-volume-tlas.md)).

Independent fields require conservative extinction bounds across all relevant
inputs, including transmissionDistance, and emission-aware traversal. A finite
sample grid alone cannot prove bounds for arbitrary procedural/custom fields.
No scalar majorant or RGB distance-sampling estimator is selected by this draft.

## Interface decisions still requiring confirmation

1. **Capability packaging:** FLOAT32_VEC3 structured RGB data, NanoVDB Vec3f,
   existing scalar formats, and a versioned custom capability route are accepted.
   A separate vector-field capability extension remains a recommendation. Define
   its metadata and the custom descriptor interface while preserving existing
   scalar callable signatures.
2. **Field-domain implementation:** declared field bounds intersected with ROI,
   membership testing before filtering, existing interior interpolation/edge
   behavior, and outside-value handling of empty intersections are accepted.
   Specify exact boundary inclusion and numerical tolerance consistently across
   backends; per-binding transforms are out of scope for the first version.
3. **Validation:** lightweight commit/finalize checks diagnose invalid/incomplete
   primitives and omit them while rendering valid content. No field-content scans,
   per-sample NaN/range checks, or runtime invalid-data recovery are required.
   Field sample validity is the author's/provider's responsibility. Frame-result
   properties and failed-mapping behavior were rejected.
4. **Thermal calibration:** determine spectral integration, observer/color matrix,
   gamut treatment, and absolute relation to existing light/emission units. Do not
   normalize blackbody RGB independently at each temperature. RGB absorption times
   RGB blackbody radiance is an RGB approximation, not spectral Kirchhoff transport.
5. **AOVs:** quality depth/IDs must match interactive's first center-pixel sample
   in the first standalone implementation commit, with the ANARI depth-reduction
   difference documented. No separate picking interface is planned. Whole-ray RGB
   coverage and final alpha reduction are accepted; no
   separate RGB transmittance output is included in the first version. Depth/IDs
   require renderer-independent camera-ray selection, including emission-only
   regions, not stochastic collision metadata. First optically active support is
   the accepted criterion. Nearest eligible surface/volume selection, surface
   preference at coincident depth, deterministic per-pixel rays, and stable identity
   ordering for coincident volumes are accepted. Specify numerical detection,
   tolerance, exact identity order, and camera-specific rays. Principled media have no
   intrinsic surface normal. Public albedo is the combined scattering-to-extinction
   ratio at the selected volume point, zero in zero-extinction channels. Neutral
   normal encoding and denoiser-guide handling still need explicit rules.
6. **Renderer fog composition:** additive overlap across both volume subtypes is
   accepted. Existing renderer fog is an appearance effect, not another physical
   medium; its composition with surfaces and volumes must be tested.
7. **Type encoding and discovery:** confirm ANARI parameter types, extension
   names/dependencies, capability reporting, and unsupported-frame behavior.
   Parameter spelling and the temperature default are accepted.

## Implementation ordering

First, make a standalone quality-depth/ID alignment commit, before changing volume
subtypes or transport. Gate all relevant depth/ID writes consistently, including
misses and existing volume hits; do not gate radiance or normal/albedo accumulation.
Test a pixel where the center ray and a later jittered ray select different objects
or depths, accumulation across multiple calls, multiple samples per call, and
reset after camera/scene changes. Cover background/miss, surface, and volume paths.
This commit preserves existing camera/lens and volume-hit behavior; the complete
future deterministic volume-selection model is subsequent work.

Then implement the agreed volume/field interface and transport in dependency order.
The detailed plan remains to be written after the outstanding technical decisions.

## Acceptance checks for eventual implementation

- Homogeneous unlit slab: transmission exp(-sigma_t*L); source contribution
  j*(1-exp(-sigma_t*L))/sigma_t, with j*L limit when sigma_t=0, per RGB channel.
- Reference distance/color, density scaling, nonuniform instance scaling, and
  voxel-resolution invariance.
- Bounds required; independent density/emission extents; explicit outside values;
  sparse inside-domain backgrounds; field ROI; empty and invalid domains.
- All constants and field-driven controls, mismatched field resolutions, shared
  fields in differently parameterized volumes, and recommit invalidation.
- Zero-density emission and purely scattering zero-thermal-emission limits.
- HG normalization, direction convention, both lobe endpoints, and blend weights.
- Additive overlapping media independent of object insertion/traversal order;
  heterogeneous phases and colored extinction.
- Quality illumination of a nearby surface by RGB and thermal volume emission;
  interactive's declared omissions tested separately.
- Invalid constants, binding types/formats, structural metadata, and volume bounds:
  diagnose and omit invalid primitives while rendering valid scene content.
  Exercise near-unit anisotropy and guarded TF opacity saturation with valid inputs.
  Do not require detection or recovery for NaNs/range violations in field samples.
- Extension discovery and unsupported renderer combinations.
- TF1D camera emission–absorption in both renderers, quality indirect emission,
  and migration examples with interpolation/saturation limitations documented.
- Blackbody against a calibrated reference at several temperatures, without
  per-temperature brightness normalization.
- Scalar opacity equals one minus luminance of combined RGB transmittance;
  emission-only RGB can be nonzero with zero opacity. Check colored overlap
  against reducing after aggregation rather than per-medium alpha compositing.
  Splitting a colored medium into adjacent segments must preserve final alpha.
  Test surface coverage and transparent/opaque backgrounds independently of
  radiance throughput; emission must not change coverage.
- Hard field-domain boundaries, interior edge filtering, empty field ROI, and
  invalid zero-width/inverted volume bounds.
- Renderer-independent depth/ID selection, including emission-only regions,
  without dependence on random scattering events or radiance sample count.
  Nearest eligible surface/volume wins, with surface preference at coincident
  depth. Coincident volumes use stable identity ordering, independent of BVH
  traversal and brightness. Test camera mappings and selection-cache invalidation.
  No fabricated physical volume normal. Public volume albedo equals the combined
  scattering-to-extinction ratio at the selected point, independently of selected
  contributor ID, lighting, exposure, and emission intensity; zero-extinction
  channels return zero. Exact numerical expectations depend on the remaining
  AOV decisions.
- AOV/denoiser and mixed fog/surface/volume regressions after those contracts settle.

Documentation-only work requires link/consistency checks. Implementation will require
an RTX-only configure/build and relevant CTest API/transport tests; no GPU behavior
is claimed verified by this document.
