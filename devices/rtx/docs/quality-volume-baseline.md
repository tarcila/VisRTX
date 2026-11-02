# Quality principled-volume fidelity/performance baseline

> Sampling integration status: recorded images, convergence and timings predate
> the Halton-camera / compact Sobol-transport split. They are historical evidence,
> not qualification of the current sampling policy. Seed zero preserves the new
> Halton baseline, not the former all-Sobol images. See the [sampling contract](../apps/tests/unit/sobol/README.md)
> for the seed policy and outstanding rendering requalification. Optical/unit
> checks that do not consume camera or transport samples remain applicable.

Status: bounded baseline completed; results below. This is a bounded baseline,
not another surface-material qualification campaign. No transport optimization or
rendering-model change is part of this measurement slice.

## Fixed scope

Use the current Release library explicitly through `LD_LIBRARY_PATH`; record its
hash and actual loader mapping. Preserve the user's host clock lock. No clock
setter/reset is permitted in this series. Record GPU clocks, memory and other
compute processes; a missing broker client does not invalidate the external lock.

Scenes:

1. Canonical cell-v2 cloud-128 density, albedo .9, g=.65, directional light.
2. Canonical cell-v2 plume-128 density/temperature, albedo .2, thermal intensity 1,
   directional light, matte floor/card.
3. Spatially separate support: density clipped to x<=-.25, RGB emission_offset
   field at x>=.5, albedo zero, no thermal source. Test source outside density.
4. Two overlapping homogeneous colored absorbing/emitting boxes with a shifted
   domain; independent analytic segment solution.
5. Off-camera homogeneous vacuum emitter illuminating a receiver, no other light:
   matte and native diffuse PBR versions. Independent volume-integral reference;
   this deliberately exposes the current direct-source sampling gap.

The public-ANARI runner uses fixed cameras, medium bounds and light parameters.
These are controlled cases using qualified input assets, NOT camera/light matches
to the archived Blender EXRs. Those images are not used as convergence goldens.

- Performance: **640x640, one sample per launch, 12 launches**, first two excluded
  from warmed summaries. Rates 1/1; no denoising, firefly filtering or checkerboard.
  Quality maxRayDepth=32. Public CUDA-event duration plus selected Nsight Systems
  captures; CPU setup/wait/map/file time is kept separate.
- Convergence diagnostics: **128x128, 32 samples/launch, 18 launches**. Analyze
  disjoint 32-spp blocks after startup, prefix estimates and a disjoint held-out
  tail. These are disjoint sample blocks, not independently scrambled QMC
  replicates or proof of an unbiased/converged reference. Report uncertainty.
- Depth sensitivity: cloud/plume at depths 12 and 64, compared to depth32 at
  matched work. Initial results may justify a later depth/convergence campaign,
  not an unbounded rerender loop.
- Analytic receiver reference: Gauss volume integration at orders12/20, with
  pixel-footprint averaging, requiring relative agreement <=1e-6 before use.
- Separate-support comparison: preview at increased rates supplies only an
  emission-absorption quadrature cross-check; its own refinement difference is
  reported. It is not a multiple-scattering Quality reference.
- Per-process safety timeout: 240 seconds. A timeout is incomplete coverage, not
  permission to reduce resolution/work and call the performance target met.

Report linear HDR RGB relative L1/normalized RMS, alpha differences, spatial noise,
GPU milliseconds/sample, setup cost and memory. For cloud/plume, held-out comparisons
are noise/convergence diagnostics, not a physical-fidelity certificate. The 128x128
image campaign is not counted as satisfying any 640x640 frame-rate goal.

The next implementation priority must follow the measured correctness/variance/
work bottlenecks. Broad MDL/MaterialX qualification stays paused.

## Measured performance

RTX 5880 Ada, driver615.78.08, CUDA13.4, OptiX9.1. The frozen library SHA256 is
`667caee31706b33cbefd108b7f97fc0f3e376cf1f261cc1906c501a33aac8b2c`.
All 24 planned render processes completed; no workload was reduced after a
failure. Two reversed-order performance runs agreed. Graphics clocks were
2400 MHz throughout recorded telemetry; memory was9501 MHz under load and
10001 MHz at idle. No other compute process was present at phase boundaries.
The host lock was not modified. Shader caches were already warm: first-frame
numbers are resource/setup observations, not isolated cold OptiX compilation.

Median public CUDA-event duration, 640x640, one sample/launch, depth32:

| Scene | Run medians | Approximate sample-launch rate |
|---|---:|---:|
| Cloud | 55.06–55.42 ms | 18 samples/s |
| Thermal plume | 224.32–225.11 ms | 4.4 samples/s |
| Separate emission/density support | 19.18–19.24 ms | 52 samples/s |
| Homogeneous overlap | 12.90–12.98 ms | 77 samples/s |
| Native diffuse receiver | 2.65–2.81 ms | 355–377 samples/s |
| Matte receiver | 1.84–1.86 ms | **Not a useful speed success: lighting is missing** |

One-sample launch rates are not converged-image frame rates. The public duration
covers render-related GPU work including resolve/composition, not file writes or
GUI overhead. Nsight Systems separately confirms the launch-dominated workload.

Steady GPU-wide used memory was about **1.28–1.34 GiB**. This includes CUDA/OptiX
modules, contexts, framebuffer and scene allocations; it is not an isolated volume
cache size. Profiled runs carried about28 MiB extra compared with corresponding
unprofiled repeats. The raw measurements and allocator/driver overhead must not
be conflated with field storage alone.

## Fidelity and convergence observations

The image campaign uses128x128 for bounded diagnostics, not the performance goal.
Actual accumulation counts were checked: the initial allocation/reset repeated the
first32-spp count, then the sixteen analyzed batches progressed monotonically.
All captured diagnostic and final performance images are finite.

| Scene | 32-spp prefix vs held-out256-spp tail NRMS | 256 vs256 NRMS | Estimated noise in the256-spp tail |
|---|---:|---:|---:|
| Cloud | 20.29% | 9.61% | 6.84% |
| Plume | 19.94% | 9.85% | 6.85% |
| Separate support | 6.27% | 2.79% | 1.98% |
| Overlap | 11.62% | 5.45% | 3.87% |
| Native receiver | 147.0% | 68.9% | 49.1% |

These differences fall as work increases, but the cloud/plume held-out images are
still too noisy to certify representative image fidelity. The tail noise figures
are estimates from disjoint blocks, not independent randomized-QMC confidence
intervals. No noisy image was promoted to a convergence golden.

### Independent controls

- **Overlap:** the512-spp image has2.75% pixel NRMS against the analytic segmented
  solution, but RGB means closely match: measured `(0.481520,0.440696,0.290033)`
  versus analytic `(0.481524,0.440669,0.290027)`. The remaining pixel difference is
  consistent with sampling noise; this is not an all-scene bias certificate.
- **Emission outside density support:**363 bright pixels retain zero alpha. The
  preview rate2→4 cross-check changes RGB NRMS by only1.04e-6 (fraction); Quality
  at512spp differs from that reference by1.30% NRMS. Quality/preview image means
  agree closely. This pure emission–absorption comparison does not test multiple
  scattering and is not a generic preview/Quality parity requirement.
- **Native receiver:** the independently integrated expected mean is0.01002715;
  measured mean is0.01002325 (−0.039%, estimated block-mean SE0.00003696). However,
  pixel NRMS is **36.22% at512spp**. Continuation finds the source with the correct
  bounded mean but very poor practical variance.
- **Matte receiver:** measured RGB is exactly zero against that same strictly
  positive reference: **100% error**. Existing DirectOnly matte has no continuation
  and direct volume-source sampling is absent. This is a real functionality gap,
  not Monte Carlo noise or an MDL issue.
- Receiver oracle orders12/20 agree to1.52e-15 NRMS. Pixel footprints are averaged;
  the emitter remains outside the complete primary-camera footprint.

### Depth

At matched512-spp work, cloud depth12 versus32 differs by0.109% relative RGB L1
and0.575% NRMS. Cloud depth32 and64 are identical for these samples. Plume depths
12/32/64 are identical for these samples. Raising depth is not the dominant
improvement indicated by this corpus; equality is not a universal depth guarantee.

## Profiling evidence and prioritized gaps

1. **Finish direct volume-source sampling and ownership first.** It repairs the
   matte failure and addresses the native receiver's severe noise. Use this
   analytic off-camera-emitter fixture as a fixed acceptance case. Broader MDL
   auditing is not the next work item; retain continuation ownership wherever
   replacement has not been justified.
2. **Reduce reference volume work without changing its physical meaning.** Nsight
   PC samples concentrate in the Quality volume callable:74.3% for cloud and94.4%
   for plume; the stored-field scalar sampler accounts for22.6% and5.0% respectively.
   The FP64 pipeline reaches65.4%/51.4% of its own peak, while DRAM throughput is
   only6.54%/0.49%; achieved occupancy is28.2%/30.7%. This points to volume arithmetic,
   sampling and control work, not surface-material overhead or a bandwidth-only
   bottleneck. PC-sample shares are not exclusive elapsed milliseconds.
3. **Optimize time to a stated error, not just sample cost.** Reusing per-ray and
   per-scattering-vertex reference calculations and improving conservative bounds
   are candidates. Thermal evaluation needs finer attribution before declaring it
   the sole plume bottleneck. Reducing null/source proposals can trade speed for
   variance; evaluate both. A new FP32 tracking path would require matched bounds,
   unlike merely changing source-evaluation arithmetic.

SASS contains substantial local-memory traffic. The profiler's static spilling
request counter is zero, so the traffic is not labeled wholesale as compiler
spilling; callable/explicit local state also contributes.

## Reproduction and evidence

Tracked tools: `tools/principled_volume/quality_baseline.cpp`,
`run_quality_baseline.sh`, and `analyze_quality_baseline.py`. The data-tool CMake
option `BUILD_QUALITY_BASELINE=ON` builds the runner. The frozen binary/source,
24 run directories, telemetry, loader mappings, raw HDR images, Nsight reports,
`results.json`, and `convergence-montage.png` are under
`_out/principled_volume/quality-baseline/`.

Validation: runner configure/build, shell/Python syntax checks, actual sample-count
and finite-image checks, independent oracle refinement, and an overlap-driver
memcheck/full-leak workload all pass. No renderer behavior changed, and the full
renderer regression suite was not rerun for this measurement-only slice. No
installation, commit or push was performed.
