# FP32 Interactive volume preview

**Follow-up:** [precise invalidation, column building and uncached optimization](interactive-volume-cache-followup.md)
now supersede the initial cache implementation/timings below. The selected plume
measures about 14.4 ms cached, 16.9 ms with direction changes, and 28.5 ms without
the directional cache. The earlier sections retain the original qualification
history, not the latest invalidation policy or builder.

The user-approved execution policy is **FP32, fast math and FMA for ordinary
preview rendering**, with an unchanged controlled reference evaluator and
explicit wide exceptions. Bitwise equality to the reference is not the preview
contract. Physical parameter meanings, additive overlaps, emission in vacuum,
thermal calibration, output ownership and provider scope remain unchanged.

This is experimental Interactive/default transport, not Quality transport,
custom-provider opt-in, extension advertisement, or universal performance/error
qualification.

## Interfaces and arithmetic

- `gpu/volumePreviewMath.h`: FP32 optical coefficients, TF emission–absorption,
  segment integration, and thermal evaluation. `expm1f` preserves thin-segment
  accuracy; no new optical clamps or opacity cutoff.
- `gpu/previewVolumeTraversal.h` and `renderer/previewFloatLattice.h`: FP32
  world-distance traversal and anchored cells. Single occurrences reuse ray
  transforms/intersections/step lengths; arbitrary overlaps retain the uncapped
  merged-domain/lattice traversal. Directions are not renormalized in object space.
- `spatial_field/StoredFieldPreviewSampler.h`: FP32 mapping and interpolation
  for all existing stored adapters. NanoVDB accessors cache nodes across corner
  fetches, and across samples within density-integral requests. Inactive values,
  tile/background values, centering, ROI and rectilinear LUT conventions remain.
- `renderer/volumePreviewFloat.h`: FP32 quadrature, direct scattering and source
  accumulation. A single occurrence reuses camera coefficients for its phase
  kernel; overlaps still evaluate each occurrence's phase contribution.
- The ordinary point sampler is a noinline device function rather than an OptiX
  callable transition per point. Reference sampling and density-integral requests
  keep their direct-callable seam. Camera integration remains a continuation
  callable because it traces surface shadows; stack-depth limits are unchanged.

The FP32 evaluator exposes **no Bounds/majorant interface**. This renderer uses
quadrature, not majorant-based tracking. Existing reference certificates remain
valid only for the reference graph; they are not relabeled as FP32 certificates.
A future FP32 tracker must construct bounds for its actual sampling/optics graph.
Existing range metadata can select optimization eligibility without certifying
FP32 null-collision probabilities or skipping.

Exactly zero absorption suppresses only absorption-coupled thermal work. Explicit
emission is still sampled in vacuum and outside the density field's domain.
Similarly, zero scattering avoids irrelevant phase work. These are exact optical
identities, not density-only source skipping, validation scans or hidden thresholds.

## Explicit wide exceptions

Preparation uses the already available field/range/revision metadata to choose
ordinary FP32 execution. Noneligible scenes retain the wide preview; they are
not omitted or made intrinsically invalid. Internal eligibility thresholds are
conservative optimization policy, not new public scene/rate caps.

Wide execution covers exceptional operand magnitudes/metadata-known subnormal operands, very
small transmission/distance ranges, large thermal gains/coupling or temperatures,
unavailable FP32 range metadata, unsuitable field mappings, ill-conditioned or
extreme instance transforms, and poorly resolved ray coordinates. Very high
sampling rates retain the wide lattice. No per-sample finite/validity tests are
introduced. Ordinary near-parallel directions and tiny off-diagonal transform
terms do not alone require wide execution.

FP64 remains in:

- CPU preparation, reference certificates and constant-log/cold-table construction;
- the preserved wide fallback and exceptional spectral thermal evaluation;
- the small packed sparse leaf decoder, retained for storage-decoding compatibility
  (not claimed to be mathematically indispensable to every FP32 application);
- shared per-ray setup and HDR/surface helpers outside the medium sample loop.

OptiX warnings describe compiled code, including unused fallback branches. They
are not an execution profile. In the final ordinary-plume 128x128 capture, the
camera continuation executed **zero DADD/DMUL/DFMA instructions**, and the FP32
field sampler executed none. Remaining double arithmetic was in raygen/per-ray
and shared surface work; FP64 pipe utilization was **0.443% of elapsed peak**.
This does not claim that packed or exceptional scenes execute no FP64.

## Cold thermal lookup

A volume-owned 64-knot log-XYZ/Hermite lookup over 100–300 K is constructed on the
CPU from the existing attributed spectral coefficients. FP32 interpolation and
log-domain absorption coupling avoid the 1,881-sample spectral integration in
ordinary cold camera samples. The existing 300–100000 K LUT remains unchanged.
No new numeric dataset is embedded; runtime-derived values retain the existing
CC BY-SA 4.0 adaptation policy and installed attribution.

The lookup is published only for the prepared FP32 coupling envelope. Under this
envelope, contributions below 100 K are below the FP32 normal range even after
coupling. The test verifies an absolute-matrix spectral upper bound with 2^30
coupling headroom (preparation's limit is 2^20). Returning zero there is the
selected FP32/FTZ result, not clamping temperature or choosing a visibility cutoff.
Larger coupling retains the wide path and its tiny-radiance preservation.

4102 CPU/CUDA fast-math controls cover ordinary optics, thin/thick segments,
50–300 K cold coupling and adjacent 100/300 K floats. Cold error remains within
the existing 1e-3 fallback budget; the 300 K seam retains the 5e-5 jump budget.
The independent 1297-temperature thermal oracle also checks the FP32 warm LUT
against the unchanged 2e-4 LUT budget. GPU controls caught a normalization error
with a compound GLM vector scale; explicit component multiplication corrected it
without relaxing any tolerance.

## Directional volume-shadow cache

For a single ordinary principled occurrence with a nonconstant, linearly filtered
stored density field, constant transmission color/distance, and one directional
light, a Frame can build a 257^3 nodal density-integral cache. Density's domain
must cover the volume domain. Small fields/frames bypass it to avoid setup cost.
Other light sets, overlaps, TF1D, nearest fields, internal density-ROI jumps and
noneligible precision cases keep the marcher. No provider is removed.

The cache approximates the **volume attenuation only**. Phase, radiance/PDF,
explicit/thermal source, and surface shadow visibility remain evaluated normally.
Points outside its domain use the marcher. It does not cache surface occlusion,
change light ownership, or supply an optical majorant.

Keys include aggregate generation, light-set revision and shadow sampling rate.
Field/volume/instance changes feed the existing aggregate-generation mechanism.
Unchanged camera frames reuse it. Disabling eligibility clears publication;
restoration rebuilds it. The builder and subsequent launch use the same CUDA
stream, and Frame destruction waits before releasing storage. Optional allocation
failure retains the marcher. No new asynchronous-lifetime or frame-failure API
was introduced.

**Shadow rates above 1 bypass this fixed spatial cache**, avoiding an interpolation
error floor on higher-quality refinement. There is no new public rate ceiling.
Cache memory is 257^3 floats: 67,898,372 bytes (about 64.75 MiB) per eligible Frame.
At the measured plume setting, its GPU build costs about 87.8 ms once per key.
Animated density/light changes pay that rebuild cost; the 60 FPS result below is
for warmed, unchanged scene data, not animated-volume throughput.

## Frozen acceptance and measured result

Before measurement, acceptance was fixed at:

- coefficients: 2e-5 relative plus 2e-6 absolute near zero;
- analytic segments: 2e-4 relative plus 2e-5 absolute;
- existing thermal oracle/seam tolerances unchanged;
- existing public integration/provider thresholds unchanged;
- representative linear HDR: relative L1 RGB <=0.5%, normalized RMS <=1%,
  maximum absolute alpha difference <=0.002.

Performance target selected by the user: **16.7 ms warmed GPU frames at 640x640,
one sample per launch, primary/shadow rates 1/1**. No lower resolution,
checkerboarding, denoising or sample-count reduction was used to meet it.

RTX 5880 Ada, CUDA 13.4, driver 615.78.08, Release build. Host-managed clock lock
was left untouched; telemetry reports 2400 MHz graphics and 9501 MHz memory under
load (10001 MHz at idle). Two 12-launch Systems captures discarded their first
two launches:

| Run | Median launch ms | Retained range ms | Cache builds |
|---|---:|---:|---:|
| 1 | 16.232 | 16.189–16.258 | 1, 87.822 ms |
| 2 | 16.293 | 16.240–16.368 | 1, 87.802 ms |

A final verification capture reports 16.317 ms median launch time and
**16.329 ms median / 16.399 ms maximum including the following background
composition kernel** (ten retained frames). GUI presentation/interop is not
included.

This meets the selected warmed-launch target for the historical plume-128 archive
and its fixed archive camera/default directional light. It is about 296x faster
than the cached-log FP64 baseline (4.82 s), or about 540x versus the original
8.81 s baseline. It is not a promise of 60 FPS for arbitrary views, light sets,
overlap counts, providers or GUI presentation pipelines. Sampling defaults remain
experimental rather than deployment-optimal.

Linear HDR comparisons use a separate public-ANARI runner with the same fields,
bounds, camera and geometry, an explicitly fixed directional light `(1,-1,-1)`,
and identical settings/sample addresses between the wide and optimized libraries.
This lighting is deliberately explicit rather than inferred from Vela UI angles.
Five corresponding frames were compared per scene; numbers below are maxima:

| Scene | Relative L1 RGB | Normalized RMS RGB | Max alpha error |
|---|---:|---:|---:|
| Thermal plume, 640x640 | 1.267e-6 | 1.554e-6 | 2.236e-6 |
| Cloud, no thermal, albedo 0.9, 128x128 | 1.429e-5 | 4.142e-5 | 4.301e-6 |

Ratios are fractions, not percentages. Plume reference radiance exceeds 2500;
this is not an 8-bit-only comparison. The cloud control prevents bright thermal
emission from hiding scattering/shadow error. Tone-mapped side-by-side checks
showed no discernible differences. These are bounded controls, not a universal
error bound or a full stochastic convergence campaign.

## Verification and evidence

- Full Release build passed.
- One uninterrupted broad CTest run passed **141/141**, excluding only the
  dedicated `TestMdlCompileBenchmark` performance harness (most recent recorded
  run: 279.50 s). The passing-name set matches the CTest inventory.
- New controls cover FP32 optics, cold/seam arithmetic, prepared lattices,
  precision fallback/recovery (including amplified subnormal bindings and large\n  thermal gain with tiny absorption), and shadow-cache analytic lookup, world-unit\n  instance scaling, reuse and invalidation.
- Memcheck/full leak checking passed for `VolumePreviewFloatCUDA`,
  `PreviewShadowCache`, `PrincipledVolumeState`, and `TestPrincipledInteractive`:
  zero errors and zero leaked allocations. A fifth memcheck run exercises the
  public HDR plume runner, including the full 257^3 NanoVDB shadow-cache build,
  also with zero errors/leaks.
- Existing reference/bound/oracle tests remain intact. Qualification does not
  assert that those reference bounds enclose the different FP32 graph.

Local captures, HDR buffers, comparisons and scripts:
`_out/principled_volume/performance/fp32/`, especially `final-profile/summary.json`
and `final-quality/results.json`. Build/test logs: `/tmp/visrtx-fp32-*`.
The original survey and constant-log optimization remain recorded in
[interactive-volume-performance.md](interactive-volume-performance.md).

## Final-code cache-off and animated-light ablation

A subsequent controlled ablation disabled only Frame's shadow-cache construction
and publication. All 15 embedded OptiX shader blobs match between the frozen
on/off libraries. The temporary source change was reverted, the normal Release
build restored, and the public principled/shadow-cache regressions passed.

Same 640x640 plume/camera, one sample, rates 1/1; two reversed-order repetitions
of 12 launches per condition, first two discarded. Medians of render-related GPU
timeline spans through composition (not GUI/file-writing time):

| Workload | Cache on | Cache off |
|---|---:|---:|
| Static directional light | 16.34 ms (~61 FPS) | 176.02 ms (~5.7 FPS) |
| Direction changes every frame | 107.1–107.6 ms (~9.3 FPS) | 171.2–173.0 ms (~5.8 FPS) |
| Intensity changes every frame | 106.4–106.5 ms (~9.4 FPS) | 176.0–176.3 ms (~5.7 FPS) |

Animated cache-on runs rebuild on every launch; construction costs about 89 ms.
Intensity-only changes should not require density-integral rebuilding, but the
current broad light-set revision key does so. In this workload caching remains
faster even with rebuilding, although neither animated configuration meets 30 FPS.
The final uncached path does **not** meet the user's 33.3 ms target and needs about
5.3x further improvement. The older ~210 ms uncached checkpoint was not a final-code
ablation and should not be used as the direct cached/uncached comparison.

Host clock settings were preserved; telemetry was mostly 2400 MHz graphics, with
brief 2370/2385 MHz readings. Cache-on/off PPM differences are at most one code
value; this is not an additional HDR animation qualification. Full measurements,
reproduction scripts, hashes and restoration evidence are under
`_out/principled_volume/performance/cache-ablation/RESULTS.md`. No public toggle or
automatic cache policy was introduced by this measurement.
