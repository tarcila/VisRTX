# Experimental MDL absorption slice

MDL-enabled builds accept a directly requested `mdl` Volume, but do **not** yet
advertise the subtype or `VISRTX_VOLUME_MDL`. This is an implementation slice,
not the complete extension. It currently requires:

- A same-device `mdl` Material (`material`, `ANARI_MATERIAL`).
- A finite object-space box with strictly positive finite sides (`extent`,
  `ANARI_FLOAT32_BOX3`), or a valid same-device scalar Spatial Field (`value`,
  `ANARI_SPATIAL_FIELD`). Explicit extent takes precedence over field bounds.
  With no field, no voxel array is needed. Invalid supplied inputs never fall
  back to another domain source.
- An explicit finite nonnegative scalar bound (`extinctionMajorant`,
  `ANARI_FLOAT32`), covering every RGB extinction channel per world distance.
- The `interactive` Renderer (including its `default` alias) or `quality`.

Optional `stepSize` is a finite positive `ANARI_FLOAT32` object-space distance.
Without an explicit step, use the present field's positive finite suggested
step, otherwise the shortest extent side / 128. The field-step choice is
intentional even with an extent override: a tiny crop may use one clipped
segment and a large expansion may be expensive. Extent changes never stretch
field coordinates, resample data or mutate another Volume's field/domain.
An explicit step overrides either automatic choice; unusable automatic scales
are diagnosed when marching actually needs them. The extent-only automatic
scale is derived lazily; quality validates explicit inputs but never uses a
marching step. Marching divides the base
step by the length of the transformed world ray direction, then applies the
Renderer's primary/shadow sampling rate. Each segment is clipped to the ray's
Interval and evaluated at its midpoint. Positions are object-space; optical
lengths are world-space. Normals are zero. Pure absorption uses an FP64 optical
depth and exponential transmission; alpha is mean-transmittance coverage, not
a scalar substitute for chromatic radiance attenuation. Screen backgrounds
(including their premultiplication convention) are resolved in transport once
for these Worlds; the Frame compositor does not apply them again.

The separate volume Target Code has its own initializer, state-usage checks,
Argument Block layout and string/resource indices. It does not initialize
surface geometry normals or fabricate a SurfaceHit. Surface-only attributes,
scene-data bindings and texture parameters do not supply volume inputs.

A present `value` requires a nonempty string `fieldName`. Each Volume owns this
one binding, independent of its shared Material and any surface `sceneData.*`
bindings. A varying float lookup of that name returns the raw interpolated
scalar, or its MDL-authored default outside the field's geometric support.
`data_isvalid` is a uniform binding-presence query, not a support mask. Executed
unbound, wrong-type and uniform lookups are errors; guarded/unexecuted lookups
and surface-only expressions are not volume errors. Every actual medium
evaluation samples and validates a present supported field once, even if MDL
ignores it, then caches the value for all lookups. NaN/Inf is an error, never a
support sentinel. Absent/unsupported fields and zero-majorant-skipped positions
are not probed. A retained string `fieldName` alone is legal metadata;
unset/null `value` is absent and makes every volume `data_isvalid` false.
Setting/unsetting extent or value updates the traversal domain, binding and
automatic scale through the ordinary commit/dependency lifecycle. Restoring a
field requires a valid name again.

Interactive supports RGB single scattering, absorption and vacuum. Each
midpoint samples one light from the shared full-sphere volume-NEE family with
its complete selection-times-direction PDF. The isotropic source uses a stable
exponential segment integral, including the zero-extinction limit. Shadows
include opaque-surface visibility and medium attenuation from the midpoint to
the light, including the evaluating Volume and other nonoverlapping domains.
Primary and shadow rates refine independently, respecting finite endpoints and
nonuniform Instance step conversion. This is direct single scattering, not
multiple scattering or indirect lighting. Quality still supports absorption
and vacuum only in this slice.

The supported
isotropic models are default `vdf()` and elemental `df::anisotropic_vdf` with
an evaluated directional bias of exactly zero. Compiler temporaries and folded
expressions are resolved before classification. Other unresolved VDF graphs
(including `fog_vdf`, mixtures and conditionals) are explicitly unsupported;
the SDK is never asked to generate a VDF evaluator. Nonzero/nonfinite bias or
volume emission fails the render, including after parameter/time edits.
Surface emission is independent of volume emission.

Volume expressions may use object-space position, actual Instance transforms,
Frame `mdl.animationTime`, the Volume's 32-bit ID, zero normal/geometry normal,
and one meter per scene unit. Explicit-coordinate texture lookups use the
Material's resources with separate volume-bundle mappings and editable Argument
Blocks. Entry-surface texture coordinates/tangents, tangent space, motion,
direction, rounded-corner normals and unknown state bits are rejected per
volume Callable/initializer, independently of unrelated surface dependencies.

Quality scattering and unsupported Renderers remain explicit errors, not
approximations. Multiple-volume preflight now checks every transport-active
pair involving MDL, including off-camera domains and each transform of a shared
Instance/Volume. It uses the resolved extent (or field-derived domain), and the
TF1D field/ROI intersection, without density-based exclusions. `visible=false`
retains its existing device meaning: removal from all traversal, not just camera
visibility. Surfaces are not medium-domain pairs.

World-AABB separation is a broad phase only. The narrow phase tests transformed
box face normals and edge cross-products, including affine shear. Arithmetic
roundoff is enclosed using TwoSum/FMA residuals and directed adjacent values;
exact arithmetic is not expanded by an epsilon. Ambiguous double predicates
are retried in long double. A certified nonnegative separating gap permits
separation/contact; positive-volume overlap fails preflight with both Volume
registry indices and Instance/transform placements. Remaining uncertainty fails
with `indeterminate overlap`, never silent medium omission. Results are
recomputed after each render's commit flush, so extent, field-domain and
Instance edits cannot reuse stale classifications. TF1D-only pairs keep their
existing behavior.

Interactive retains the product of MDL RGB transmission and TF1D scalar
transmission for mixed absorption. Generalized mixed compositing and
whole-primary-track coverage remain pending work; this experimental slice is
not the complete advertised extension.

Returned FP32 absorption/scattering coefficients must be finite/nonnegative.
The evaluator promotes both and the scalar bound to FP64, retains the error-free
TwoSum residual, and rejects `t > M` or `t == M && residual > 0` before checking
this slice's transport restrictions. Thus both FP32- and FP64-downward-rounded
sums cannot hide an observed violation. Equality is valid.

Quality selects one RGB channel uniformly per camera sample, locks it through
all surface/TF1D/light/background contributions, and compensates RGB radiance
by three exactly once. Alpha and AOVs do not receive that factor. Worlds without
MDL retain their existing full-RGB sampling path. Rate-M world-distance
candidates use a stable open-variate exponential flight and explicit progress
checks; nulls consume no real-event depth. Event classification uses one exact
scaled 24-bit variate and ordered smaller-event/total-extinction thresholds,
not independently rounded category probabilities. Absorption ends radiance;
scattering remains an explicit unsupported-medium error in this slice. Light
transmittance uses ratio tracking with the same validated FP64 extinction and
factor `1-t/M`, without a contribution threshold or transmittance cutoff. M=0
skips evaluation under the caller's vacuum guarantee. The isolated-medium
coverage outcome is unweighted; generalized whole-track surface/medium coverage
is a later slice. Nonadvancing steps report numerical
failure. All evaluated errors use the Frame-owned error record and the shared
failure/publication/recovery policy in `../frame/README.md`.

**The device does not prove your extinction bound. An underestimated bound can
bias an image without triggering an error, including when a zero bound skips
all evaluations. The application must supply a conservative bound for every
color channel, position, resource/parameter state and animation time used by
the render. Runtime checks only reject violations that are actually evaluated.**
