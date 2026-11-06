# Principled volume approximation experiments

Baseline: `02e66fce` (ray/achromatic reuse and point-texture endpoints), with
`2658e96a` animation tooling. Experiments are sequential: retain and commit a
qualified improvement before starting the next; revert unsuccessful candidates.
Software interpolation and reference Quality transport remain the defaults.

## Acceptance policy

Compare against the committed baseline, not merely the slower disabled mode of a
new candidate. Check both enabled and disabled modes. A useful initial performance
signal is at least 5% less time on representative workloads; investigate default
path regressions above 3%. These are screening criteria, not universal guarantees.

Initial Interactive HDR gates remain relative L1 <= 0.005, normalized RMS <= 0.01,
and maximum alpha error <= 0.002. Black references require exact equality. These
bounded scene checks do not establish universal sampling error or motion quality.
Quality experiments must also assess time-to-error when the estimator changes.
Any deliberately looser approximation needs its own explicit error budget, not a
retroactive relaxation of a failed test.

Use bounded GPU-wide clock leases, reversed execution order, and telemetry.
Clock-policy configuration is not exposed here; do not infer its maximum duration.

## 1. Hardware-filtered Interactive optics — rejected

Implemented an opt-in `volumeHardwareFiltering` renderer parameter for the
practical FP32 principled camera integrator. Eligible scalar fields reused the
existing linear texture with canonical coordinate mapping. RGB, custom,
rectilinear, nearest-filtered, and texture-ineligible fields retained software
sampling. Quality, TF1D classification and shadow integration were unchanged.
The default remained point-texture endpoints plus software interpolation.

Two candidates were tested:

1. Runtime sampling-policy propagation through the camera coefficient evaluator.
2. Compile-time camera-integration specializations selected once per segment,
   avoiding the runtime policy in the inlined common NanoVDB sampling path.

The second improved the enabled path but still failed the benefit/regression
screen. All production and measurement-driver changes were reverted. No dormant
parameter, shader branch, new sampling policy or metadata remains.

### Clock-leased timing results

RTX 5880 Ada, 2400 MHz active graphics clocks; 640x640, one sample, rates 1/1,
no denoising/checkerboarding/firefly reduction. Two reversed-order repetitions;
discard two warmups from 24 subway launches and 12 canonical launches.
Ranges below are the two medians, not confidence intervals.

| Scene | Baseline ms (round 2) | Candidate disabled ms | Candidate enabled ms |
| --- | ---: | ---: | ---: |
| Subway | 23.76–23.96 | 23.99–24.20 | 23.27–23.56 |
| Cloud | 28.80–29.04 | 29.41–29.44 | 28.04–28.08 |
| Plume | 17.02–17.11 | 17.28–17.28 | 16.43–16.45 |
| Separate support | 6.73–6.74 | 7.23–7.29 | 6.89–6.91 |
| Overlap | 0.81–0.87 | 0.87–0.88 | 0.81–0.83 |

The first candidate measured 23.42–23.56 ms subway, 28.76–28.77 ms cloud and
16.86–16.97 ms plume when enabled, also insufficient. The second candidate's
roughly 2–4% gains on representative scenes do not justify the added policy and
approximately 8% disabled-path regression on separate support. This does not
prove hardware interpolation cannot help; it rejects these opt-in integrations.
Earlier unconditional experimental filtering had better results, but is not an
acceptable silent replacement of the reference preview policy.

### Fidelity and checks

The first candidate's 42 retained 128-square frames across subway, cloud, plume,
separate support, overlap and native/Matte receivers passed the unchanged gates:
maximum relative L1 0.000160, NRMS 0.000222, alpha error 0.000109. All 42 disabled
frames matched the baseline exactly. Release builds passed; stored-reference,
public Interactive and public Quality tests passed before benchmarking. The
second candidate was rejected on performance; no broad correctness/sanitizer or
motion qualification is claimed for it.

Both timing leases and the first candidate HDR lease exited successfully after
release/reset. Evidence: `_out/principled_volume/hardware-optics/` contains the
runner, gates, raw captures/timings/telemetry, lease logs, second-candidate patch
and frozen library. The first candidate binary was not separately frozen.

## 2. Coarser camera integration — retain opt-in rate 0.75

The existing `volumeSamplingRate` already controls the anchored camera lattice.
Before implementing an adaptive scheme, sweep camera rates 1, 0.75, 0.5 and 0.25
with **shadow rate fixed at 1**. Compare images against both camera rates 1 and 4;
rate 4 is a denser numerical comparison, not a converged physical reference.
No renderer code, interpolation policy, default, or sample-count semantics change.

The retained result is an explicitly selected setting for these workloads:

```cpp
const float cameraRate = 0.75f, shadowRate = 1.f;
anariSetParameter(device, renderer, "volumeSamplingRate", ANARI_FLOAT32, &cameraRate);
anariSetParameter(device, renderer, "volumeSamplingRateShadows", ANARI_FLOAT32, &shadowRate);
anariCommitParameters(device, renderer);
```

### Measured benefit

Same GPU/clock/image/work controls as experiment 1, two reversed rate orders,
24 subway launches or 12 canonical launches, two warmups discarded.

| Scene | Camera rate 1 ms | Camera rate 0.75 ms | Approximate time reduction |
| --- | ---: | ---: | ---: |
| Subway | 23.65–23.92 | 17.65–17.75 | 26% |
| Cloud | 28.99–29.11 | 20.68–20.84 | 29% |
| Plume | 17.03–17.13 | 11.01–11.07 | 35% |
| Separate support | 6.72–6.73 | 5.07–5.11 | 24% |
| Overlap | 0.81–0.85 | 0.76–0.86 | No reliable gain |

These are observed end-to-end GPU sample durations, not a claim that every scene
scales linearly with rate. Rate 0.5 reaches about 11.8 ms subway and 6.3 ms plume,
but fails the unchanged relative-L1 gate for emitting scenes against rate 4.
Rate 0.25 also fails. Neither is recommended as a general replacement.

### Error and motion controls

Rate 0.75 passes the original gates in:

- Seven static scenes at 128-square, six retained cumulative frames each.
- Eight camera/density/temperature-change cases at 128-square, six frames each:
  subway and plume in all three modes, cloud in camera/density modes.
- Four 640-square final-frame checks: subway, cloud, plume and separate support.

Each is compared with matching rate-1 and rate-4 captures. Worst error across
these comparisons is relative L1 **0.001763**, NRMS **0.002319**, maximum alpha
error **0.000371**. Forty-two restored rate-1 frames also match the pre-experiment
baseline byte-for-byte. The static receiver controls are black in Interactive;
exact equality there is not evidence of volume-source illumination quality.

Rate 0.75 sometimes matches rate 4 more closely than rate 1 does. This is a
sampling-pattern effect in these fixtures, not monotonic convergence or a reason
to call the coarser setting more accurate universally. Thin/hot features can still
alias; global image metrics do not bound every pixel. The motion checks change
camera or binding multipliers, not time-varying voxel contents, and do not prove
absence of temporal flicker. No automatic motion policy or adaptive error bound
is introduced. A future adaptive implementation needs richer thin-feature and
voxel-animation controls rather than assuming these results certify it.

### Retained tooling and validation

The baseline runner gains an optional independent shadow rate, defaulting to its
previous shared rate. The archive runner gains explicit camera/shadow rate CLI
arguments, defaulting to 1/1. Both reject nonfinite/below-minimum rates.
Lua values are explicitly converted to floating-point before ANARI submission.
An initial subway rate-4 capture accidentally submitted an integer parameter and
therefore retained the renderer's default rate; it is preserved under
`hdr-invalid-subway-reference` and excluded from acceptance. All reported HDR
checks use the corrected floating-point submission.

The measurement driver builds successfully; four invalid-rate CLI controls reject
before device loading. Four restored renderer/stored-field CTests pass. Clangd
cannot validate this standalone driver with the repository's main compile
configuration (missing NanoVDB includes and cascading GCC-header errors); the
standalone CMake build and actual executions are the verification here.

All four accepted measurement leases exit successfully; active graphics clocks
remain 2400 MHz. Post-run telemetry returns to 210/405 MHz, 0% utilization, 2 MiB.
Evidence, including runner, captures, metrics and telemetry:
`_out/principled_volume/camera-rates/`. The 0.75 setting and measurement tooling are
retained; adaptive integration remains a distinct unimplemented extension.

## 3. Reduced-frequency lighting — rejected in this implementation

Keep every optical/emission sample, but refresh lighting every two camera cells.
Cache the scattered-source/scattering ratio, multiply by the new scattering
coefficient on the intervening cell, and reset at span boundaries and zero-medium
cells. Newly nonzero scattering channels force a refresh. Independent emission
and transmittance integration do not use this cache.

A forced single-volume prototype showed substantial gains, but was not a
production-ready policy. A subsequent opt-in implementation restricted reuse to
one principled occurrence, constant phase bindings, one directional light and no
ambient term. Overlaps and other configurations retained ordinary lighting.
Wide cached factors avoided division overflow for very small scattering. Both
variants retained software optical interpolation and camera/shadow rates 1/1.

### Results

Same 640-square, 2400 MHz, reversed-order, full-sample protocol as experiments 1/2.
The final opt-in candidate used a boolean switch and isolated the cached loop
from the inlined default loop.

| Scene | Baseline ms | Opt-in disabled ms | Opt-in enabled ms |
| --- | ---: | ---: | ---: |
| Subway | 23.84–23.96 | 23.98–24.10 | 23.86–23.93 |
| Cloud | 28.77–28.90 | 28.80–28.91 | 27.36–27.39 |
| Plume | 17.08–17.14 | 17.19–17.22 | 15.26–15.29 |
| Separate support | 6.65–6.66 | 7.37–7.39 | 7.35–7.39 |
| Overlap | 0.77–0.82 | 0.79–0.81 | 0.87–0.88 |

The approximately 5% cloud and 11% plume savings do not justify an approximately
11% separate-support regression even with the feature disabled. An intermediate
integer-policy/noinline-default implementation also regressed separate support.
The original forced prototype reached 18.60–18.63 ms subway, 22.22–22.38 ms cloud,
and 14.98–15.09 ms plume, but used a broader, less-qualified approximation and
also regressed separate support (7.67–7.82 ms versus 6.73 ms).

Instrumentation explains the constrained candidate's lack of subway benefit:
the actual GPU registry reports `enabled=1`, one principled occurrence, one light,
zero ambient, and light type **GEOMETRY**, not DIRECTIONAL. The archived authored
light name alone does not establish what the renderer samples. Reusing a sampled
geometry-light connection at a displaced point requires a different proposal/
visibility treatment; the forced prototype's small image error does not qualify
that algorithm. This runtime observation differs from earlier directional-light
attribution in the historical notes; do not assume those scopes are equivalent.
The inspected extracted scene SHA-256 is
`07183001103a05400764171d6fb41e179f053f02f3559a162a9241e59bf1dbee`.

All three candidates pass the unchanged static HDR gates on 42 retained frames
across seven scenes. The final candidate has worst relative L1 0.000375 and NRMS
0.001074 (cloud), with zero alpha error; its disabled frames match baseline
exactly. It was rejected before motion, sharp-occluder, and broad sanitizer
qualification. These are not claims of general radiance-cache correctness.

All production code, metadata and temporary driver switches are reverted;
experiment 2's independent camera/shadow rates remain. Restored Release builds
and four focused renderer/stored-field CTests pass. Six measurement leases
complete successfully at 2400 MHz active graphics clocks. Evidence:
`_out/principled_volume/lighting-stride/`, including raw timing/HDR, telemetry,
registry diagnostics and the frozen final candidate library/patch. The rebuilt
frozen final candidate matches its measured library SHA-256 exactly.

## 4. Reduced-resolution volume rendering — not retained

Built a throwaway split-pass renderer probe, not whole-image downsampling:

- Full-resolution surface shading retains medium shadow queries.
- A separate volume pass stops at the first opaque surface and exports volume
  radiance V, RGB transmittance T, and the geometric endpoint depth.
- Reconstruction composites `V + T * surfaceRGB`, with scalar coverage derived
  from RGB T and surface coverage. It does not substitute alpha for colored T.
- Four-tap bilinear and depth-aware (2% relative endpoint-depth tolerance)
  reconstruction are compared. Missing compatible taps are counted; an
  oracle-repair diagnostic is explicitly not part of the measured work.

The probe deliberately rejects transparent first layers. Fog, layered transparent
surfaces, full AOV semantics and an integrated multi-launch Frame lifecycle are
not implemented. Temporary renderer hooks and AOV repurposing are not public API.
Cloud, plume, separate-support and a new colored-extinction/thin-opaque-strip
control are evaluated. At equal resolution the split composition agrees with
reference to NRMS below 2.6e-8, establishing the decomposition for these fixtures.

### First variant: reduce both V and T

640-square surfaces are combined with 480-square or 320-square volume buffers.
All four scenes fail the original alpha gate at both reductions. Even cloud has
maximum alpha error 0.00338 at 480-square and 0.00651 at 320-square. Separate
support reaches 0.0687/0.1082. Depth guidance helps surface discontinuities but
cannot detect density/ROI discontinuities in an otherwise surface-free ray.
The 320-square plume also fails radiance gates (L1 0.0117, NRMS 0.0144).

### Second variant: full-resolution T, reduced-resolution V

Fuse full-resolution camera extinction with surface shading using the existing
attenuation query, while retaining reduced-resolution volume radiance. No
full-resolution reference buffers are used by this reconstruction. The 480-square
radiance variant passes the static gates on all four scenes. Maximum alpha error
is <=0.000323; plume has L1 0.004949 and NRMS 0.006449, close to the L1 limit.
320-square still fails the plume and thin-strip radiance gates.

Exploratory GPU pass timings (two reversed-order medians, milliseconds):

| Scene | Reference 640 | Surface + T 640 | Volume V 480 | Sum of two passes |
| --- | ---: | ---: | ---: | ---: |
| Cloud | 29.11–29.33 | 0.927–0.931 | 21.86–21.86 | about 22.79 |
| Plume | 15.33–15.51 | 1.143–1.145 | 12.97–12.98 | about 14.12 |
| Separate support | 6.89–7.00 | 0.873–0.920 | 4.54–4.56 | about 5.45 |
| Thin strips | 5.35–5.36 | 1.325–1.345 | 3.08–3.10 | about 4.43 |

The sum excludes reconstruction, buffer handoff and integrated-frame overhead;
it is **not** a measured end-to-end speedup. Reconstruction here is offline CPU
code. One centered primary sample is reset each launch so V/T/surface
factorization can be checked without covariance from averaging jittered samples;
these timings are not directly interchangeable with earlier accumulating runs.

### Motion gate and verdict

At 480-square, six retained 640-square comparisons per motion case give:

| Case | Maximum relative L1 | Maximum NRMS | Gate |
| --- | ---: | ---: | --- |
| Cloud camera | 0.0000533 | 0.000193 | Pass |
| Plume camera | **0.005312** | 0.007841 | **Fail** |
| Plume density multiplier | 0.004980 | 0.006488 | Pass |
| Plume temperature multiplier | 0.004754 | 0.006168 | Pass |

Do not relax the 0.005 L1 gate after observing the camera failure. Full-resolution
T plus reduced-resolution V remains an interesting design, particularly for
smooth clouds, but these implementations do not support a generally qualified
preview mode. They are not retained, and no production pipeline/parameter change
is committed. More robust reconstruction or selective full-resolution radiance
repair would be a new experiment, not a hidden assumption in these results.

Three bounded clock-lease commands completed successfully. Telemetry shows
2400 MHz during load and several 210/405 MHz transition observations in the two
static campaigns; raw telemetry is preserved rather than describing every
nonzero-utilization observation as clock-locked. Motion telemetry records 2400 MHz
for active samples. The exploratory timings are not used to accept the failed
candidate. Restored Release builds and four focused renderer/stored-field CTests
pass after removing the probe.

Evidence: `_out/principled_volume/resolution-probe/`, including both variants,
all motion captures, reconstruction code, prototype patch, frozen library and
probe executable. No complete sanitizer, transparent-layer or deployment
qualification is claimed.

## Remaining sequence

5. Mixed-precision Quality sampling: pending.
6. Baked optical representation: pending.
7. Improved Quality proposals/control variates: pending.
8. Approximate multiple-scattering tail: pending.
