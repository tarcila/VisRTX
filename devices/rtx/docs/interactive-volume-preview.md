# Experimental Interactive volume preview

Interactive and its `default` alias now consume the aggregate reference-volume
scene. This is an implementation checkpoint, **not complete feature qualification**
and not extension advertisement. The bounded activation and compatibility pass
are now reconciled in [the readiness record](interactive-volume-readiness.md).
The input contract remains [`principledVolume.md`](../../../principledVolume.md).

## Transport and output

`gpu/renderer/volumePreview.h` integrates additive RGB extinction and source on
anchored, metadata-guided cells, split at occurrence and field-domain endpoints.
It includes direct single scattering, both phase lobes, and calibrated thermal
emission. TF1D records in this path use emission–absorption semantics. Variable
medium sampling uses a separate PCG stream. There is no opacity cutoff or hidden
maximum-step truncation.

`gpu/renderer/interactiveVolumeRaygen.h` tracks uncovered RGB separately from
straight-through backdrop throughput. Thus zero-alpha source radiance survives,
and colored extinction/transmission affects the backdrop without conflating it
with coverage. Fog ownership persists through covered transmitting surfaces;
coverage holes can acquire fog at a subsequent surface. Public albedo/normal use
first-event coverage weights and arithmetic sample averages. Legacy approximate
TF picking stays separate from optical integration; surface IDs retain per-hit
depth arbitration. Principled picking improvements remain deferred.

The shared renderer helpers are bounded by OptiX callable programs to avoid
pathological compiler expansion: scalar/RGB stored-reference sampling and medium
attenuation use direct calls; camera integration uses a continuation call because
it issues shadow traces. Pipeline stack calculation permits two nested direct
calls and one continuation call for reference-volume renderers. Legacy renderers
reserve the SBT indices but do not link preview transport through those slots.

## Sampling and denoising policy

Interactive/default discovery reports:

| Parameter | Default | Minimum | Maximum |
|---|---:|---:|---|
| `volumeSamplingRate` | 1 | 0.001 | none |
| `volumeSamplingRateShadows` | 1 | 0.0001 | none |
| `denoise` | false | — | — |
| `denoiseStart` | 0 | — | — |
| `denoiseMode` | `color` | — | — |

Rates must be finite; they scale field-resolution/domain-fallback cell sizes.
The 1/1 defaults are **experimental qualification starting points**, not measured
latency optima. Removing the old metadata maximum of 10 reflects the existing
implementation, not a universal precision/refinement guarantee. Extremely fine
unrepresentable lattices currently integrate remaining represented spans. The
[finite-precision assessment](preview-lattice-precision.md) records tested ties,
coarsening, a corrected inverse-direction overflow, and the unrepaired possibility
of intersection endpoints collapsing before stepping.

When the prepared scene contains eligible reference media and denoising is enabled,
the effective mode is color-only. Requests for `colorAlbedo` or
`colorAlbedoNormal` produce a warning; public AOVs remain independently available.
A Frame deduplicates that warning by scene generation and renderer finalization
stamp, so unchanged render calls stay quiet, while renderer replacement or a
committed mode/enable revision is diagnosed again. Explicit color mode, disabled
denoising, and scenes without eligible reference media do not warn.

## Bounded evidence

- `TestPrincipledInteractive`: public analytic controls for vacuum source,
  colored Beer/background/source, overlaps, directional two-lobe scattering,
  event-weighted AOVs, glass/fog, color-only denoising, cold/thermal source,
  scalar/RGB dense fields and ROI clipping/recommit. Additional analytic controls
  cover a spatially varying HDR image backdrop through colored extinction, with
  alpha premultiplication and denoising on/off, plus observed image-array updates;
  TF1D alone and additive TF1D/principled overlaps in both occurrence orders; and
  off-camera absorbing slabs on surface-light shadow connections, including
  duplicate occurrences.
- `TestInteractiveVolumePolicy`: Interactive/default discovery values and absent
  rate ceiling; same-frame warning deduplication, mode/enable transitions,
  renderer replacement, scene revision, removal and restoration of media.
- The preceding integration checkpoint passed 35 targeted API/reference tests,
  including CPU/CUDA/fast-math reference optics and aggregate tests, native
  material sampling, MDL pipeline rebuild, frame channels, and fog regressions.
- Compute Sanitizer memcheck of the public transport controls, including both
  field callables and ROI update, reported zero errors and zero leaked allocations.

- Dedicated CPU/CUDA/fast-math blackbody tests pass 1,297 spectral-oracle
  temperature controls, including cold absorption coupling and both LUT seams.
  The oracle reproducibility and eight-file data/attribution component install
  checks also pass; see [thermal qualification](blackbody-qualification.md).

The public transport test also reuses frames across eight progressive launches
with regular and checkerboard sampling, then explicitly resets and repeats the
sequence. It checks analytic RGB/alpha/AOV values and public sample counts.
Automatic mode is restored before source/scattering updates and removal/restoration
of media to test discarded history. The inherited initial allocation/version-latch
reset is handled exactly as in `TestSampleAddressing`, not redesigned here.

The expanded integration selection passed eight API regressions; memcheck of the
expanded public transport test reported zero errors and zero leaked allocations.
No production renderer changes were needed for these additional controls.

These controls are not image convergence goldens, exhaustive provider coverage,
or a performance/default qualification campaign.

## Stored-provider OptiX coverage

`PrincipledInteractiveProviders.h`, exercised by `TestPrincipledInteractive`, adds
92 analytic public renders through the actual reference-field callables:

- 28 dense controls: regular and rectilinear, nearest and linear, for FLOAT32,
  FLOAT64, signed/unsigned normalized 8/16-bit scalar storage and FLOAT32_VEC3.
  Rectilinear coordinates are nonuniform; these constant fields test transport
  plumbing without conflating it with heterogeneous quadrature error.
- 24 constant NanoVDB controls: regular/rectilinear and nearest/linear for Float,
  Vec3f, Fp4, Fp8, Fp16 and FpN.
- 16 packed-ramp controls: regular/rectilinear Fp4/Fp8/Fp16 and all five FpN widths
  (1, 2, 4, 8, 16 bits). The test asserts the generated width, uses NanoVDB’s own
  host decoder for the expected value, and places the entire ray segment inside
  voxel (5,5,5), with nonzero decoded extinction even at one bit. The resulting
  homogeneous Beer/source integral is independent of the production decoder.
- 24 sparse ROI change/restoration controls retain the same observing volume while
  the field snapshot/domain changes. Scalar cases retain explicit emission outside
  the density ROI; RGB source cases use zero outside emission.

All twelve selected field/binding/aggregate/public regressions passed. Memcheck of
the expanded public test reported zero errors and zero leaked allocations. No
production sampler, callable ABI, or renderer changes were needed. Existing CUDA
raw-sampler tests remain complementary coverage for varying interpolation, inactive
values, and coordinate mappings; the public controls do not exhaust all such cases.

## Limits and work beyond this activation

- The bounded lattice assessment and provider matrix are not exhaustive
  heterogeneous-scene or transformed-coordinate guarantees; the documented
  extreme-coordinate limitations remain.
- Quality multiple scattering and volume-source lighting, approved BSDF work,
  and custom-provider opt-in. Quality/Fast/Debug/Test still diagnose and omit
  principled contributions; object validity is independent of that eligibility.
- Final advertised extension/subtype metadata after promised paths are qualified.

CIE inputs and derived numeric tables under `gpu/blackbody/data/` remain separately
attributed CC BY-SA 4.0 data, not BSD-only implementation code.
