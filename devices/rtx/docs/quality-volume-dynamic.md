# Quality dynamic-scene checkpoint

## Correctness controls

`TestQualityVolumeUpdates` compares a repeatedly edited scene/frame against a
freshly constructed scene at the same state. Two cycles cover camera motion,
light direction/intensity, instance translation/non-unit scale, density and
Kelvin-temperature array edits, RGB emission array edits, matte material edits,
source disable/recovery and restoration. Array edits do not recommit the volume.
Every edited frame resets to16 samples; subsequent unchanged frames accumulate.
All24 edited/fresh image pairs match exactly. This is a deterministic lifecycle
control, not a stochastic convergence test. Stored array controls use dense
structured fields; the motion benchmark uses the canonical NanoVDB plume/cloud.

The default test shares one device across independent scenes. The
`--separate-devices` variant also passes uninstrumented and remains a separate
CTest. Single-device memcheck with explicit OptiX tracking reports zero errors
and leaks.

### Multi-context sanitizer finding — unresolved tool-stack coverage

With two live devices, Compute Sanitizer reports an internal hardware-exception
handling failure followed by an illegal-address error on the first reference
render. This reproduces without VisRTX: a minimal program creates two OptiX
contexts on one CUDA primary context, each with its own pipeline, stream, SBT,
parameter/output buffers, and a one-thread raygen that writes123. It passes
uninstrumented; one context passes memcheck; two contexts fail under memcheck.
Thus the failure is not specific to the volume update implementation, but this
does not establish complete multi-device memory safety. Keep that qualification
open; do not report the failing sanitizer mode as passed. Reproduction and logs
are retained with the dynamic evidence (RTX5880 Ada, driver615.78.08).

## Motion and rebuild measurements

`quality_baseline` now accepts an optional motion mode after its existing
arguments: `static`, `camera`, `light`, `instance`, `density`, `temperature`, or
`material`. Motion uses a fixed0.1*sin(0.2*frame) sequence. Density/temperature
benchmark edits change binding multipliers, not NanoVDB voxel bytes. Actual array
invalidation is covered separately above. Wall timing includes commits and
preparation; GPU launch duration alone is insufficient for animated work.

Canonical scenes,640x640, one sample, depth32, rates1,60 launches, first two
excluded. All changing states report one accumulated sample; static states
continue accumulating. GPU-wide used memory has zero first-to-last warm delta
in all twelve initial60-launch workloads. This is a bounded trend check, not a
long-running memory-growth proof.

| Scene/edit | Median GPU ms | Median render+prepare wall ms |
|---|---:|---:|
| Cloud camera |30.20|30.22|
| Cloud light |30.20|30.33|
| Cloud instance |30.21|30.35|
| Cloud density multiplier |30.95|31.19|
| Plume camera |29.26|29.28|
| Plume light |29.45|29.66|
| Plume instance |29.50|29.71|
| Plume material |29.83|29.85|
| Plume density, before guide fix |29.84|42.33|
| Plume temperature, before guide fix |30.20|43.57|
| Plume density, after guide fix |29.61|31.26|
| Plume temperature, after guide fix |30.52|32.14|

## Implemented optimization

Source-guide pilots previously called the full cold spectral evaluator even
though Quality had a qualified bounded cold-source lookup. Guide construction
now binds the same `QualityBlackbody` evaluator to the volume's prepared table.
This is heuristic source importance, not an extinction certificate. Full-support
mixing, missing-table spectral fallback, guide invalidation and physical source
evaluation remain intact. No source support or provider capability is removed.

Nsight CUDA captures show one guide build across eight camera launches and one
across eight instance launches; temperature edits rebuild on each of eight
launches. The optimized pilot kernel takes approximately0.20 ms per build in
these captures (previous thermal-plume construction was about10.18 ms).
The static plume512-spp HDR average remains bit-identical to the frozen round4
image. Density/temperature render+prepare maxima remain about36.35/36.49 ms:
median30-FPS behavior is not an every-frame guarantee for animated sources.
External clock configuration was not changed; profile telemetry is retained.

Evidence: `_out/principled_volume/quality-dynamic/`. Full regression log:
`/tmp/visrtx-quality-dynamic-qualified-tests.log`. The next ordered stage is
randomized convergence/time-to-quality qualification; custom typed-provider
implementation follows it. Broader provider/animation coverage remains open.
