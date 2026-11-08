# Experimental MDL absorption slice

MDL-enabled builds accept a directly requested `mdl` Volume, but do **not** yet
advertise the subtype or `VISRTX_VOLUME_MDL`. This is an implementation slice,
not the complete extension. It currently requires:

- A same-device `mdl` Material (`material`, `ANARI_MATERIAL`).
- A finite object-space box with strictly positive finite sides (`extent`,
  `ANARI_FLOAT32_BOX3`). No Spatial Field or voxel array is needed.
- An explicit finite nonnegative scalar bound (`extinctionMajorant`,
  `ANARI_FLOAT32`), covering every RGB extinction channel per world distance.
- The `interactive` Renderer (including its `default` alias) or `quality`.

Optional `stepSize` is a finite positive `ANARI_FLOAT32` object-space distance.
An automatic value, derived only when marching needs it, is the shortest
extent side / 128. Quality validates explicit inputs but never uses a marching
step. Marching divides that
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

Currently supported transport is default-`vdf()` pure absorption and vacuum.
Nonzero scattering/emission, other VDF expressions, unsupported volume state,
executed texture operations, executed unbound scene-data lookups, field
bindings, multiple-volume Worlds involving MDL, and other Renderers are
explicit errors, not approximations. A retained string `fieldName` alone is
legal metadata; unset/null `value` is absent, and volume `data_isvalid` is false
for every name. Later tickets implement these deliberately gated capabilities.

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
