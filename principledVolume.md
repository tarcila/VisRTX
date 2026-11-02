# `principledVolume` API

## Status and purpose

This document describes the agreed input contract and intended ANARI encoding for
`principledVolume`: a bounded participating medium for VFX smoke, fog, clouds, and
fire. It is a design/API reference, **not an announcement of complete device support**.

The factory now creates `principledVolume` objects with committed typed bindings,
required bounds, stored-field reference snapshots and component-wise optical bounds.
Public creation/commit/update, GPU payload evaluation and shared aggregate
traversal are tested. **Experimental Interactive/default transport is integrated**,
including additive emission–absorption, direct single scattering and calibrated
thermal emission, with bounded analytic controls. **Experimental Quality transport
is now active too**, including RGB stochastic multiple scattering, source on
continuation segments and independent coverage/background handling; see the
[Quality activation record](devices/rtx/docs/quality-volume-activation.md).
Fast/Debug/Test still diagnose and omit principled contributions without invalidating
the objects. [Initial direct volume-source sampling and ownership](devices/rtx/docs/quality-volume-source.md)
now support native reflection, Matte and medium vertices.
[Source importance](devices/rtx/docs/quality-source-importance.md),
[dynamic-scene checks](devices/rtx/docs/quality-volume-dynamic.md),
[randomized convergence/error-cost work](devices/rtx/docs/quality-volume-convergence.md)
and [source-built typed custom opt-in](devices/rtx/docs/typed-custom-fields.md) are
implemented with bounded qualification. Broader material/closure and deployment
qualification remain pending; the
[thermal provider checks](devices/rtx/docs/blackbody-qualification.md) cover sampled
CPU/CUDA accuracy and installed attribution, not complete renderer qualification; subtype/extension discovery is deliberately not advertised.
See the [preview implementation record](devices/rtx/docs/interactive-volume-preview.md)
and [readiness/compatibility record](devices/rtx/docs/interactive-volume-readiness.md)
for the tested scope and remaining work, including the
[preview precision limits](devices/rtx/docs/preview-lattice-precision.md) and the
[FP32 execution/performance qualification](devices/rtx/docs/interactive-volume-fp32.md).

The volume will be exposed by an RTX-device vendor extension. The proposed
extension identifier is `VISRTX_VOLUME_PRINCIPLED`; that identifier and its discovery
metadata are not yet finalized or advertised. The agreed subtype name is exactly
`principledVolume`.

Unlike `transferFunction1D`, this subtype describes optical properties directly.
It does not accept transfer-function arrays, a scalar `value` classification input,
or `valueRange`. Ramps and other remapping belong upstream of the volume.

## Creation

The creation call is:

```cpp
ANARIVolume volume = anariNewVolume(device, "principledVolume");
```

Set parameters with `anariSetParameter()` and commit with
`anariCommitParameters()`, following normal ANARI object and array lifetimes.
Field inputs are explicit `ANARI_SPATIAL_FIELD` object references, not names of
channels to discover in a file.

## Inputs and defaults

The table lists the intended ANARI types. All RGB quantities are **linear RGB**
using the device's linear sRGB/D65 convention, not sRGB-encoded display values.

| Parameter | ANARI type | Default | Meaning |
| --- | --- | --- | --- |
| `bounds` | `ANARI_FLOAT32_BOX3` | **Required** | Finite object-space domain of the medium. |
| `density` | `ANARI_FLOAT32` | `1` | Dimensionless multiplier of extinction. |
| `scatteringColor` | `ANARI_FLOAT32_VEC3` | `(1, 1, 1)` | Scattering albedo: fraction of extinction attributable to scattering, per RGB channel. |
| `transmissionColor` | `ANARI_FLOAT32_VEC3` | `(exp(-1), exp(-1), exp(-1))`, approximately `(0.367879, 0.367879, 0.367879)` | RGB unscattered transmission through the reference distance at density one. |
| `transmissionDistance` | `ANARI_FLOAT32` | `1` | Reference distance in world-space scene units. |
| `anisotropy` | `ANARI_FLOAT32` | `0` | Primary Henyey–Greenstein phase-function anisotropy. |
| `secondaryAnisotropy` | `ANARI_FLOAT32` | `0` | Secondary Henyey–Greenstein anisotropy. |
| `secondaryWeight` | `ANARI_FLOAT32` | `0` | Weight of the secondary phase-function lobe. |
| `emission` | `ANARI_FLOAT32_VEC3` | `(0, 0, 0)` | Independent RGB emission coefficient, in radiance units per world-space scene-unit path length. |
| `temperature` | `ANARI_FLOAT32` | `1` | Temperature in kelvin when unbound, or multiplier converting a temperature field to kelvin. |
| `blackbodyIntensity` | `ANARI_FLOAT32` | `0` | Multiplier of absorption-coupled thermal emission. Zero disables thermal emission. |
| `blackbodyTint` | `ANARI_FLOAT32_VEC3` | `(1, 1, 1)` | Artistic multiplier of thermal RGB emission. |

Common volume parameters are available: `id` (`ANARI_UINT32`,
default `0xffffffff`), `visible` (`ANARI_BOOL`, default true), and the optional
standard object `name` (`ANARI_STRING`). These are not field-driven optical inputs.

The default material is neutral, isotropically scattering fog with emission off.
It requires illumination to appear bright; it does not glow by default.

## Constant and field-driven properties

Every optical input in the table, **except `bounds`**, supports the same binding
pattern:

| Parameter | ANARI type | Meaning |
| --- | --- | --- |
| `P` | Scalar or RGB type listed above | Constant value, or multiplier when a field is bound. |
| `P.field` | `ANARI_SPATIAL_FIELD` | Optional field driving that property. |
| `P.outside` | Same numeric type as `P` | Field sample to use outside the field's sampling domain, before multiplication by `P`. |

At an object-space position `x` inside the volume:

```text
No P.field:                      effectiveP(x) = P
Inside the bound field's domain: effectiveP(x) = P * sample(P.field, x)
Outside that field's domain:     effectiveP(x) = P * P.outside
```

RGB multiplication is componentwise. Binding a field does **not** replace the
constant or change its default. Consequently:

- `density.field` passes through unchanged with default `density = 1`.
- `temperature.field` containing kelvin passes through with `temperature = 1`.
- To use an absolute RGB transmission field, set `transmissionColor = (1,1,1)`.
  Otherwise the default transmission color multiplies it.
- To drive anisotropy directly from a field, set its constant multiplier to `1`.
  The default multiplier of `0` would otherwise keep anisotropy at zero.
- Binding `emission.field` does not enable emission by itself: set `emission` to a
  nonzero multiplier, commonly `(1,1,1)`.
- Binding `blackbodyIntensity.field` likewise does not override the default zero
  intensity. Set `blackbodyIntensity` explicitly to enable it.

The valid ranges below apply to the **effective values**. A bound parameter can
serve as a conversion factor; for example, an anisotropy multiplier of one is
useful even though the resulting anisotropy must remain strictly inside `(-1,1)`.

### Outside values

| Properties | Default `P.outside` |
| --- | --- |
| `density`, `temperature` | `0` |
| `emission` | `(0,0,0)` |
| Other scalar optical properties | `1` |
| Other RGB optical properties | `(1,1,1)` |

These defaults are binding behavior, not instructions to extend edge voxels.
A sparse field's background values **inside** its domain remain part of the field
and are distinct from `P.outside`.

### Field types and coordinates

Scalar properties require scalar fields; RGB properties require RGB fields.
There is no implicit scalar broadcast, RGB channel reduction, or sRGB decoding.
The initial stored-field format contract is:

| Field subtype | Scalar input | RGB input |
| --- | --- | --- |
| `structuredRegular` | Existing supported scalar formats | `ANARI_FLOAT32_VEC3` array |
| `structuredRectilinear` | Existing supported scalar formats | `ANARI_FLOAT32_VEC3` array |
| `nanovdb` | Existing supported Float/Fp4/Fp8/Fp16/FpN grids | NanoVDB `Vec3f` grid |
| `nanovdbRectilinear` | Existing supported Float/Fp4/Fp8/Fp16/FpN grids | NanoVDB `Vec3f` grid |

NanoVDB data uses the existing `ANARI_UINT8` byte-array container, with one grid per
field object. The current backend requires host-visible NanoVDB metadata.
Custom fields require explicit opt-in to the new typed capabilities; legacy
scalar callable signatures are not implicitly reinterpreted.

Inputs may use different origins, resolutions, spacings, and extents. Sample them
in the volume's object coordinate system using each field's existing mapping.
There is no additional per-binding transform in the first interface. Independently
rotated/sheared inputs need an appropriate field representation or upstream
resampling. The volume's instance transform moves all its inputs together.

The implemented stored-field groundwork provides a read-only `valueType` property
of type `ANARI_DATA_TYPE`: `ANARI_FLOAT32`, `ANARI_FLOAT32_VEC3`, or `ANARI_UNKNOWN`
for an invalid field. `ANARI_WAIT` flushes pending commits before this query.
Scalar storage such as normalized fixed-point data still samples as float.

## Volume domain versus field domains

`bounds` is required and has finite coordinates with strictly positive extent on
all three axes. There is no default infinite medium and no automatic union of
input-field bounds.

- Outside `bounds`, every contribution from this volume is zero.
- Inside `bounds`, unbound constants apply everywhere.
- A field's sampling domain is its declared bounds intersected with its `roi`.
  Procedural fields use their declared bounds in the same way.
- Outside that intersection, the binding supplies `P.outside`.
- An empty field-domain intersection means the outside value applies everywhere;
  it does not by itself invalidate the volume.
- Membership is tested before interpolation. Inside the field domain, retain the
  field's interpolation and edge behavior; do not blend in `P.outside` to feather
  the boundary.

Volume bounds thus act as a clip **and** define the extent of constant media.
They do not stretch, resample, or extend field data. An emission field may occupy
regions where density is zero, provided those regions remain inside the volume.

## Optical meaning

Let `d`, `C`, `D`, and `a` be the effective density, transmission color,
transmission distance, and scattering color at a position. Componentwise:

```text
extinction  sigma_t = d * (-log(C)) / D
scattering  sigma_s = a * sigma_t
absorption  sigma_a = (1 - a) * sigma_t
```

For a homogeneous medium at density one, an unscattered beam traveling distance
`D` retains fraction `C` of its RGB radiance. A red transmission color preserves
red light; it does not mean preferentially absorbing red light.

Increasing density to two squares that transmittance. Changing scattering color
partitions extinction between scattering and absorption without changing
unscattered transmission. Black scattering color gives pure absorption; white
gives pure scattering wherever extinction is nonzero.

Path length is measured in world-space scene units. Scaling a volume instance
changes optical thickness; there is no automatic density compensation. Changing
voxel resolution alone must not change the optical model. SI comparisons with
light intensities require consistently scaled scene/light units; the API does
not infer a meters-per-scene-unit conversion.

### Phase function

```text
phase = (1 - secondaryWeight) * HG(anisotropy)
        + secondaryWeight * HG(secondaryAnisotropy)
```

Both HG lobes are normalized. Zero anisotropy is isotropic; positive values favor
forward scattering along the incoming propagation direction, and negative values
favor backward scattering. Secondary weight zero selects only the primary lobe;
weight one selects only the secondary lobe. A volume gradient is not a surface
normal and does not orient this phase function.

### Emission and temperature

The two emission sources add:

```text
j = emission
    + blackbodyIntensity * blackbodyTint * sigma_a * B(temperature)
```

`emission` is independent of density/extinction. It can produce visible radiance
without absorbing or scattering background light. In contrast, thermal emission
is coupled to absorption: a purely scattering medium emits no blackbody radiation,
even at a nonzero temperature.

`B(T)` uses Planck spectral radiance integrated against the CIE 1931 2-degree
observer over 360–830 nm and converted to linear sRGB/D65. The fixed normalization
maps a D65 spectrum with integrated radiance 1 W/(sr m²) over that band to
approximately RGB white `(1,1,1)`. The same normalization applies at every
temperature; colors are not independently brightness-normalized.

For negative RGB components resulting from out-of-gamut conversion, clamp those
components to zero and rescale to preserve the original integrated CIE Y. Leave
in-gamut RGB unchanged and do not clamp HDR values to one. This is an explicit RGB
approximation, not a claim of spectral transport equivalence.

Temperature zero produces no thermal emission. `blackbodyIntensity = 1` and white
tint select the calibrated thermal convention; intensity/tint changes are artistic
adjustments. High temperatures can produce very large radiances. The experimental
lookup interval is an optimization, **not a supported-temperature clamp**.

## Example: field-driven fire

The following illustrates the API. It assumes that the density and kelvin-temperature
fields already exist and have been committed. Creation and commit can succeed with
supported stored fields, but this is not yet an end-to-end rendering example.

```cpp
ANARIVolume volume = anariNewVolume(device, "principledVolume");

// ANARI_FLOAT32_BOX3: lower corner followed by upper corner.
const float bounds[6] = {-1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
const float density = 1.f;
const float temperatureScale = 1.f; // Field values are already kelvin.
const float blackbodyIntensity = 1.f;
const float scatteringColor[3] = {0.2f, 0.2f, 0.2f};

anariSetParameter(device, volume, "bounds", ANARI_FLOAT32_BOX3, bounds);
anariSetParameter(device, volume, "density", ANARI_FLOAT32, &density);
anariSetParameter(device, volume, "density.field",
    ANARI_SPATIAL_FIELD, &densityField);
anariSetParameter(device, volume, "scatteringColor",
    ANARI_FLOAT32_VEC3, scatteringColor);
anariSetParameter(device, volume, "temperature", ANARI_FLOAT32,
    &temperatureScale);
anariSetParameter(device, volume, "temperature.field",
    ANARI_SPATIAL_FIELD, &temperatureField);
anariSetParameter(device, volume, "blackbodyIntensity", ANARI_FLOAT32,
    &blackbodyIntensity);
anariCommitParameters(device, volume);
```

The scattering color below one leaves absorption available for thermal emission.
Keeping the default white scattering color would make this example non-emissive
thermally. The default zero outside values for density and temperature prevent
those bound fields from implicitly filling uncovered parts of the volume box.

For independent emission driven by an RGB field instead, bind `emission.field`
and set `emission` to `(1,1,1)`. No nonzero density or temperature is required for
that source. Explicit and thermal emission may both be enabled.

## Overlap and renderer support

Overlapping media add absorption, scattering, and emission coefficients. Their
scattering kernels combine according to each medium's scattering contribution;
do not average anisotropy parameters or select one material by traversal order.
This rule includes overlaps with `transferFunction1D` under its intended
emission–absorption model.

| Renderer | Intended behavior |
| --- | --- |
| `quality` | Multiple-scattering transport. RGB and thermal emission can illuminate surfaces and other volumes. |
| `interactive` / `default` | Single-scattering preview using the same optical properties. May omit illumination originating from emissive volumes. |
| Other renderer subtypes | Unsupported initially; must diagnose unsupported use rather than reinterpret the inputs. |

The table describes the target renderer support. Interactive/default provide
experimental preview transport; Quality now provides experimental stored-provider
stochastic transport, but full source-lighting/material qualification remains
pending. Fast/Debug/Test omit `principledVolume` with a preparation diagnostic.
Object validity and intrinsic
bounds are independent of renderer eligibility. The extension will not be
advertised until the promised rendering support is integrated and qualified.

Renderer selection will change the transport approximation, not parameter meaning.
Scattering-depth and sampling controls belong to the renderer. Single scattering
will not generally reproduce the appearance of multiply scattered clouds.

RGB emission does not increase opacity. An emission-only region can therefore
produce nonzero RGB with zero alpha. Scalar framebuffer alpha cannot encode exact
colored attenuation for later compositing over an arbitrary background. A separate
RGB transmittance output is outside the first version. Picking improvements are
also deferred; this document does not promise a new picking interface.

## Valid input contract and validation scope

Effective optical values must be finite and satisfy:

| Quantity | Valid range |
| --- | --- |
| Density, emission, temperature, blackbody intensity/tint | Nonnegative |
| Scattering color, secondary weight | `[0,1]` |
| Transmission color | `(0,1]` per channel |
| Transmission distance | Strictly positive |
| Both anisotropies | Strictly between `-1` and `1` |

Exact zero transmission would imply infinite extinction and is outside this input
contract. Finite input values alone do not guarantee that arbitrarily large derived
radiance/products fit in float32.

Validation is lightweight and performed at commit/finalize time: required inputs,
types, array shapes, supported formats, and structural metadata. This is **not** a
promise to scan field contents, check every sample for NaNs, or add runtime
invalid-data recovery. Applications and field providers are responsible for valid
samples, valid effective products, and any required conservative custom-field
bounds. No new frame-result status or failed-mapping API is introduced here.

## Relationship to `transferFunction1D`

`transferFunction1D` remains the scalar-classification volume. Its intended optical
model is emission–absorption in both interactive and quality rendering:

```text
sigma_t = sigma_a = -log(1 - alpha) / unitDistance
sigma_s = 0
j = sigma_t * transferFunctionColor
```

The current numerical saturation policy handles opacity one. The experimental
Quality path now removes its interpretation of TF RGB as scattering albedo; there
is no legacy scattering switch. Applications wanting lit scattering should author
a `principledVolume` instead. Migrating a TF into density/albedo fields is not always
lossless because classification and spatial interpolation do not generally commute.

The stored-field groundwork already rejects RGB fields in TF1D and isosurfaces.
The full TF1D transport change described above remains part of subsequent renderer
integration, not an already completed behavior change.
