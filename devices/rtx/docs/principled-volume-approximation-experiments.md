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

## 5. Mixed-precision Quality sampling — retained, opt-in

Quality now accepts `volumeFastSampling` (`ANARI_BOOL`, default false). It uses
bounded FP32 coordinate mapping/interpolation and canonical point-texture
endpoints for eligible dense scalar and NanoVDB Float fields. Unsupported fields
and scenes retain reference sampling. RGB, packed NanoVDB, rectilinear and custom
fields are not accelerated by this initial path.

Path state, optical arithmetic, thermal coupling, proposals and global bounds
remain reference precision. The new interpolant stays in eligible certified raw
hulls, so unchanged optical bounds remain conservative. This is an approximation
of the sampled field, **not** a claim of unbiased convergence to the canonical
interpolant. See [the numerical contract](quality-volume-fast-sampling.md),
including the ordinary-hull/fast-math argument and exact fallback conditions.
The gain combines FP32 interpolation, point-array access and avoidance of the
per-sample field callable; it is not an isolated FP64-instruction ablation.

### Clock-leased performance

RTX 5880 Ada, active graphics 2400 MHz; 640x640, depth 32, one full sample,
no denoising/checkerboarding/firefly reduction. Two reversed-order repetitions,
two warmups discarded; 24 subway launches or 12 canonical launches.

| Scene | Baseline ms | New default ms | Fast sampling ms |
| --- | ---: | ---: | ---: |
| Subway | 83.87–83.87 | 83.70–84.18 | **69.13–69.50** |
| Cloud | 22.30–22.33 | 22.13–22.25 | **16.84–16.92** |
| Plume | 24.43–24.83 | 24.31–24.46 | **20.52–20.76** |
| Separate support | 8.96–8.99 | 8.96–8.98 | **7.98–8.04** |
| Overlap | 9.33–9.42 | 9.36–9.39 | 9.36–9.36 |
| Matte receiver | 4.36–4.36 | 4.27–4.40 | 4.31–4.40 |
| Native receiver | 6.51–6.55 | 6.45–6.47 | 6.45–6.51 |

Representative reductions are about 17% subway, 24% cloud, 16% plume and 11%
separate support. Default-path differences stay within noise; no universal
improvement is claimed. Subway GPU-wide memory increases about 6 MiB
(2596.19 -> 2602.19 MiB) for both new modes, including module/stack/context memory.
No new voxel array is allocated by this option; eligible point arrays already
exist. These are sample timings, not converged-image FPS or a 30 FPS subway claim.

### Statistical and dynamic checks

- 112 retained default-mode cumulative HDR frames match the baseline byte-for-byte.
- Eight predeclared paired randomizations per scene, seven scenes, 512 retained
  spp at 128-square, pass the paired RGB-mean gate: six estimated replicate
  standard errors plus 5e-5 absolute. Largest paired NRMS is about 0.000201
  (plume); changed collision decisions can diverge individual paths.
- Canonical fixtures also use the existing independent eight 4096-spp reference
  replicates, after checking asset hashes. All mean-consistency checks pass at
  32/64/128/256/512 spp. Finite reference noise is explicitly subtracted/reported,
  not treated as a converged golden.
- At 512 spp, noise-subtracted MSE times measured GPU time ratios are 0.741 cloud,
  0.796 plume, 0.858 separate support, and approximately 1 for overlap/receivers.
  Subway's replicate-variance times time ratio is 0.743; it has no independent
  high-spp reference in this campaign. These are bounded error-cost diagnostics,
  not an asymptotic bias bound.
- Ten dynamic cases (cloud/plume camera, light, instance and density changes;
  plume temperature/material changes) contribute 60 retained paired frames.
  They pass the frozen L1 0.005 / NRMS 0.01 / alpha 0.002 gates. Worst L1 is
  0.0000441, NRMS 0.005291 (a low-spp cloud-density comparison), alpha 2.99e-7.
  These edits change parameters; actual density-array replacement is separately
  exercised by the public API test.

Four measurement leases exit successfully. Active graphics telemetry remains
2400 MHz. Memory clocks contain 9501/10001 MHz; the dynamic image-only campaign
also contains an 810 MHz transition observation. Dynamic timing is not used for
the performance table or error-cost ratios.

### Verification and artifacts

Release build and **169/169** selected CTests pass in 668.08 s (MDL compile
benchmark excluded as before). New tests exercise raw hulls on CPU/CUDA fast-math,
actual sampled fields, renderer toggling/default restoration, array updates,
nearest filtering and scene-envelope fallback. Five memcheck/full-leak workloads
pass with zero device errors/leaks: stored reference, public Quality, public
Interactive, and actual fast-mode plume/subway captures. Clangd reports no
diagnostics in the checked host renderer/API-test files. Python tooling runs
23 tests with two optional OpenVDB-dependent checks skipped.

The final source-built library differs in host binary hash after header/comment
rebuilds, but all 15 embedded PTX resources match the measured library exactly.
Only Quality PTX differs from the pre-experiment baseline. The final build and
public API toggling tests were rerun; the measured and final binaries are retained
separately rather than assigning timings to an unrecorded hash.

Evidence: `_out/principled_volume/quality-fast-fields/` contains frozen plans,
paired captures, timings/telemetry, independent-reference error-cost analysis,
verification logs and binaries. Measurement drivers expose `reference|fast` as
an optional final argument; existing invocations retain reference sampling.
No install, push or principled-volume extension advertisement is performed.

## 6. Baked source coefficients — rejected at tested resolutions

Tested the useful nonlinear part of coefficient baking on the thermal plume:
precompute its RGB emission coefficient, including absorption coupling, at
128^3 or 256^3 cell centers. A CUDA probe uses the actual committed volume state,
canonical stored-field sampler and Quality blackbody evaluator. The resulting
float3 grid is supplied through ordinary `emission.field`; thermal emission is
then disabled. Density/extinction, albedo, bounds and tracking are unchanged.
With constant optical multipliers, extinction is already a cheap linear function
of density, so baking that part would not address this fixture's expensive source.
This tests source-coefficient baking, not a new generic packed nine-channel
transport cache.

Both camera and shadow quadrature steps in Interactive are held fixed at the
original 128-grid physical step (rates 128/bakeResolution). Quality uses unchanged
tracking rates. An earlier pass compensated camera but not shadow rate; it is
preserved under `initial-*-shadow-rate1` and is not the matched-work timing result.

### Results

640-square, one full sample, two reversed-order runs, 12 launches with two
warmups discarded, active graphics 2400 MHz:

| Mode | Original ms | 128^3 bake ms | 256^3 bake ms |
| --- | ---: | ---: | ---: |
| Interactive | 16.84–16.97 | 25.30–25.60 | 28.72–28.83 |
| Quality reference sampling | 23.21–23.85 | 16.77–16.79 | 16.87–16.90 |
| Quality fast sampling | 19.76–20.08 | 13.30–13.35 | 13.24–13.37 |

Quality benefits from eliminating repeated thermal evaluation. Interactive gets
slower: an independent RGB emission field must be sampled even where density is
zero, whereas the original thermal source is known to vanish there. A specialized
packed representation could have different costs; that performance is not
measured here.

The deterministic Interactive HDR comparison fails the unchanged gates:

| Bake | Relative L1 | NRMS | Alpha error |
| --- | ---: | ---: | ---: |
| 128^3 | 0.04947 | 0.04766 | 0 |
| 256^3 | 0.01093 | 0.01405 | 0 |

Thus even 256^3 exceeds both the 0.005 L1 and 0.01 NRMS limits. Interpolating
precomputed `density * B(temperature)` does not reproduce evaluating that product
after interpolating density and temperature. Merely packing the same node values
more efficiently cannot remove this bias. The single-seed Quality screening
comparisons also differ (about 0.0218 L1 at 256^3), but include changed source
importance sampling and stochastic error; they are not an independent converged
bias estimate or a completed Quality statistical qualification.

The node arrays alone cost 24/192 MiB. Observed GPU-wide usage increases about
50/442 MiB, including retained authoring inputs and device-owned representations.
CUDA bake kernels take about 3.8–4.0/13.5–13.9 ms, but complete bake/publication
wall time is about 82–84/485–502 ms. Preparation is not hidden in steady-state
sample timings and would matter for animated fields.

No production renderer or API changes are retained. The temporary driver hook is
removed; the probe source, frozen executable/library, patch, captures, timings
and bake costs are under `_out/principled_volume/baked-optics/`. Both matched-work
leases completed successfully; timing telemetry records 2400 MHz under load,
while the HDR campaign includes idle-clock transition observations. Higher
resolution/adaptive/sparse baking remains a separate possibility, not a result
established by this rejected uniform-grid candidate.

## 7. Volume-source proposals — two-candidate resampling retained

### Rejected near-field mixture

First increased the angular/distance proposal weight from 10% to 50% for
receivers inside the source bounding sphere, retaining full support and evaluating
the entire mixture PDF. Analytic CPU/CUDA/fast-math source controls pass, as do
paired mean checks, but the intended near-emitter receiver gets worse:
variance-times-GPU-time ratios are 1.022 subway, 0.889 plume, **1.833 near receiver**,
and approximately 1 for the far receiver. A sphere-based near classification and
more angular samples are not a generally better guide. This policy was removed.
Evidence is preserved under `_out/principled_volume/source-guiding/`.

### Retained two-candidate estimator

`volumeSourceResampling` (`ANARI_BOOL`, default false) draws two candidates from
the existing complete source proposal, selects one using the maximum absolute
RGB source weight, and evaluates at most one visibility query. Selection mixes
75% score-proportional probability with 25% uniform probability. Dividing the
selected weight by twice its selection probability preserves the average of the
two estimators in real arithmetic. Invalid/zero draws are not retried. Source
ownership, optical properties, bounds and phase functions are unchanged; no
history or neighboring-pixel reuse is introduced.

See [the estimator contract](quality-volume-source-resampling.md). Unlike the
fast-field option, this changes the estimator, not its physical target. Both
options can be enabled together.

### Timings and error-cost

Final 640-square, one-sample, depth-32 timings at 2400 MHz, two reversed-order
runs, 24 subway launches or 12 other launches, two warmups excluded:

| Scene | Baseline ms | New default ms | Resampling ms |
| --- | ---: | ---: | ---: |
| Subway | 84.18–84.28 | 84.20–85.20 | 87.51–88.05 |
| Plume | 24.66–24.80 | 24.50–24.55 | 24.99–25.21 |
| Near emitter receiver | 4.92–4.93 | 4.80–4.84 | 5.25–5.28 |
| Far Matte receiver | 4.26–4.31 | 4.26–4.35 | 4.61–4.70 |
| Non-emitting cloud | 22.13–22.29 | 22.22–22.23 | 22.44–22.50 |

Samples generally cost more. The benefit is lower variance. Eight predeclared
randomizations, 512 retained spp at 128-square, give these final error-cost
ratios (lower is better):

| Scene | Variance ratio | Variance x GPU-time ratio | Finite-reference MSE x GPU-time ratio |
| --- | ---: | ---: | ---: |
| Subway | 0.926 | 0.951 | Not available |
| Plume | 0.904 | 0.923 | 0.922 |
| Near emitter receiver | 0.597 | 0.731 | 0.722 |
| Far Matte receiver | 0.589 | 0.669 | 0.669 |

The near receiver is a new synthetic control: a nonuniform RGB emitter lies
just outside the primary-camera footprint, lighting a Matte plane. It isolates
source sampling instead of letting directly visible emission dominate the metric.
Eight independent 4096-spp baseline references were rendered for it. Plume and
far-receiver references use the existing independent reference campaign.
Reference-noise estimates are retained, not treated as converged goldens.
Both estimators pass mean-consistency checks at 32/64/128/256/512 spp. Paired mean
checks use six estimated replicate standard errors plus 5e-5 absolute per RGB
channel, fixed before rendering. No seed/tolerance tuning is used.

A separate fast-field-plus-resampling campaign passes paired means on subway and
plume; its variance-time ratios are recorded separately and do not replace the
reference-field table. These are bounded convergence diagnostics, not confidence
intervals or universal guarantees for highly directional/occluded scenes.

### Verification

- 48 retained default-mode frames match baseline exactly; 512 paired coverage
  frames are unchanged when resampling is enabled.
- Original and resampled source samplers pass 18 analytic measure fixtures on
  CPU, CUDA and CUDA fast-math. Mean gates are unchanged. The new per-sample
  bound includes the derived factor-four maximum selection correction; the
  original sampler's per-sample bound cannot be reused unchanged.
- Public tests cover metadata, toggling/default restoration, Matte/native
  receivers, duplicates, holes, covered transmission, blockers, absorbing
  media, source-intensity edits, disappearance and restoration.
- Release build and 169/169 selected CTests pass in 652.87 s, excluding the MDL
  compile benchmark as before. Five memcheck/full-leak workloads report zero
  device errors/leaks: source CUDA fast-math, public source lighting, and actual
  resampled plume, near receiver and subway captures (with fast fields enabled
  for the latter three). The attempted LSP diagnostic call did not return a
  usable report; build/test results, not LSP, provide verification for this pass.

Final timing/default/replicate and combined-mode leases complete successfully with
2400 MHz active graphics telemetry. The independent near-reference campaign also
contains idle-clock transition observations; it supplies reference images, not
timing claims. Source dispatch was separated into two operation codes to avoid
carrying selection state through the default path; final results use that version.

Evidence: `_out/principled_volume/source-resampling/`, including intermediate and
final captures, fixed plans, the independent near reference, corrected error-cost
analysis, binaries and verification logs. Driver arguments now include optional
`standard|resample`; existing invocations preserve standard sampling. No install,
push or principled-volume extension advertisement is performed.

## 8. Transport/diffusion tail — promising, not qualified or retained

Tested a depth-dependent transport approximation after 2, 4 or 8 real medium
scatters. For one principled occurrence with constant nonnegative HG parameters,
compute the phase first moment `g = (1-w)*g1 + w*g2`. Later transport uses
`S' = (1-g)*S` and isotropic scattering, preserving absorption and emission.
This is a moment-matching approximation, not a truncation of all remaining light.
Overlaps, field-driven phase and negative anisotropy fall back to the original
transport. Isotropic media have no reduction.

The prototype scales conservative scattering bounds upward consistently with the
sampled S'. Continuation and connection/shadow rays use the same reduced medium;
the event that activates the tail retains its already sampled phase. Subsequent
vertices use isotropic phase. Surface/source connections carry the tail state,
so this does not silently combine old transmittance with new continuation MIS.
Independent camera coverage remains on the original optical model.

### Controls and speed potential

A 65,536-path homogeneous CPU control checks absorption/scattering/escape
probabilities, unchanged source integration and shadow transmission against the
reduced-medium analytic solution. A public OptiX white-HDRI equilibrium control
checks energy balance with the tail enabled. Those controls and seven selected
tracking/Quality tests pass. The disabled prototype matches 48 retained baseline
frames exactly.

Successful short clock-leased 640-square, one-sample campaigns (depth limit 128,
two reversed orders, 12 launches, two warmups excluded) measure:

| Scene | Baseline ms | Tail after 2 ms | After 4 ms | After 8 ms |
| --- | ---: | ---: | ---: | ---: |
| Cloud (albedo .9, g .65) | 22.10–22.46 | 14.58–14.59 | 17.00–17.02 | 20.15–20.36 |
| Same cloud, density x8 | 167.41–167.83 | 76.82–77.02 | 90.16–90.40 | 110.99–113.71 |
| Isotropic plume | 24.36–24.49 | 24.32–24.70 | 24.44–24.77 | 24.59–24.65 |

Default-mode medians stay close to baseline. The isotropic plume has no useful
gain, as expected. These timings establish potential, not acceptance of a changed
physical model.

### Image evidence and interruption

The long eight-seed, 512-spp, 128-square campaign exceeded the 1200-second tool
command limit. The lease client ended, but its Python child continued running.
The agent explicitly terminated that job and its monitor/render child, verified
those PIDs were gone, and discarded **all timing data from that campaign**.
No unseen process was terminated. Complete finite image windows with checked
sample counts remain usable for correctness analysis; partial plume outputs are
not qualified results.

The completed cloud and density-x8 sets show:

- Activation after two scatters fails the frozen paired RGB-mean gate (six
  replicate standard errors plus 5e-5 absolute). Relative mean shifts are about
  -0.91% and -0.35%, respectively.
- Activation after four or eight passes those global-mean checks. This alone does
  not bound spatial bias. Eight-scatter ensemble differences are L1/NRMS
  0.00186/0.00484 for the original cloud and 0.00264/0.00702 for density x8.
- Coverage remains byte-identical across completed paired images.

A separate, successfully completed bounded lease tests a more forward-scattering
control (albedo .95, g .85, density x8), eight seeds, 128 spp at 64-square, tail
after eight. Its global mean also passes, but ensemble L1/NRMS differences are
0.0287/0.0888. Most RMS difference is sampling noise: subtracting estimated paired
noise gives a spatial-bias NRMS estimate of about 0.0210, with substantial
uncertainty (squared-bias estimate 0.000440, jackknife SE 0.000395). This is **not**
a demonstrated 2.1% bias, but it also does not establish fidelity within 1%.
Global-mean cancellation must not be mistaken for image agreement.

The follow-up runner has its own wall-time budget, per-child timeout and SIGTERM
cleanup, rather than relying solely on the outer tool timeout. The short timing,
default-image and forward-control leases completed normally at 2400 MHz active
graphics clocks. After cleanup/verification, the GPU returned to 2 MiB, 0%
utilization and automatic idle 210/405 MHz clocks.

### Decision

Do not ship this approximation on the current evidence. Late activation is
promising for some dense clouds, but robust spatial/temporal qualification,
boundary handling and stronger forward scattering need a focused follow-up.
No error gate is loosened, and no production tail parameter/path is retained.
The probe, frozen library/runner, patch, analytic checks, complete image analyses
and interrupted-run evidence are under `_out/principled_volume/transport-tail/`.
Restored Release builds and 11 focused tests pass after removing the probe.

## Campaign checkpoint

The initial candidate sequence has been exercised, not exhaustively explored.
Retained: the measured Interactive camera-rate 0.75 setting (default unchanged),
opt-in Quality `volumeFastSampling`, and opt-in `volumeSourceResampling`.
Hardware-filtered preview optics, lighting reuse, split-resolution rendering and
uniform source-coefficient baking were not retained. The transport tail remains
a promising but unqualified follow-up, not a completed rendering feature.
