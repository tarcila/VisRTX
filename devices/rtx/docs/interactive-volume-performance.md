# Interactive volume performance

> Sampling integration status: recorded images, convergence and timings predate
> the Halton-camera / compact Sobol-transport split. They are historical evidence,
> not qualification of the current sampling policy. Seed zero preserves the new
> Halton baseline, not the former all-Sobol images. See the [sampling contract](../apps/tests/unit/sobol/README.md)
> for the seed policy and outstanding rendering requalification. Optical/unit
> checks that do not consume camera or transport samples remain applicable.

**Later result:** the user-approved [FP32 preview iteration](interactive-volume-fp32.md)
meets the selected 16.7 ms warmed 640x640 plume target at rates 1/1, with bounded
HDR comparisons. The sections below retain the original FP64 survey/cache history.

## Constant transmission-log cache (2026-11-02)

The first measured optimization caches the reference `-log(transmissionColor)`
for unbound constant RGB transmission colors in the committed volume payload.
Camera optics, scattering kernels, and shadow attenuation consume this cache.
Field-bound transmission colors retain the existing sample-time evaluation.

Preparation computes the same reference logarithm under scoped round-to-nearest,
after successful optical metadata validation. Every successful finalization
rebuilds the cache; rejected states clear it together with the payload. Synthetic
payloads default to the uncached path. Density multiplication, distance division,
partition arithmetic, thermal coupling, bounds, field sampling, and quadrature
are unchanged. In particular, the cache does **not** pre-divide by distance or
replace the controlled logarithm with a different approximation.

### Evidence motivating this change

An Nsight Compute source capture of the historical plume-128 scene at 128x128
attributed 43.49% of sampled warp PCs to `negativeLogBits<false>()`, 40.67% to the
attenuation callable excluding callees, and 10.26% to stored-field sampling.
These are sampled-PC shares, not exclusive time or guaranteed speedup fractions.
At 640x640 the original renderer reached 84.65% FP64-pipeline utilization over
elapsed cycles, with only 0.284% DRAM throughput. Removing thermal emission alone
had a small effect; removing scattering and its nested shadows removed most of
the runtime. The constant logarithm was therefore the first bounded target.

### Before/after measurement

RTX 5880 Ada, driver 615.78.08, Release builds, Nsight Systems 2026.3.2. The
uncached Release library from `b90989432b96` was preserved before rebuilding and
loaded explicitly for the A/B runs. Per-process mappings verify library choice.

Workload: original `_out/principled_volume/scenes/plume-128/scene.vsr`, archive
camera, Vela default light (archive has none), Interactive, one sample per launch,
rates 1/1, no checkerboarding, denoising, firefly filter, or ambient illumination.
Headless Lua rendering avoids CUDA-GL interop. This is the historical schema-1
asset, not the later cell-v2 qualification asset. Images were not regenerated or
re-authored for this measurement.

The user confirmed a host-managed clock lock. No clock-control operation was
performed in this optimization slice. Confirmation telemetry reports 2400 MHz
graphics throughout; memory reports 9501 MHz under sustained load and 10001 MHz
at idle. Use contemporaneous A/B timings rather than the earlier survey's lower
absolute timings. The first A/B series had a transition in uncached 640x640
latency; it is retained in the logs, not used as the final comparison.

Each confirmation process performs five launches. Discard the first two; report
the median and range of the remaining three GPU launch durations, not image-save
wall time. Run order was cached then uncached at each resolution.

| Resolution | Uncached median (range), ms | Cached median (range), ms | Speedup |
|---|---:|---:|---:|
| 128x128 | 689.80 (689.02–689.94) | 400.99 (400.70–401.81) | 1.72x |
| 640x640 | 8809.98 (8808.16–8818.21) | 4817.90 (4812.26–4819.02) | 1.83x |

All five corresponding PPM frames at both resolutions are byte-identical, in
both initial and confirmation series. PPM equality alone is not an HDR-error
proof; the coefficient-level tests below additionally require bit identity.
Rates and the numerical contract were not relaxed. Runtime remains poor:
approximately 4.82 seconds per 640x640 launch is not interactive performance.
No compile-time improvement is claimed from this change.

### Verification

- Full Release build passed.
- The 701-case reference matrix now compares host-precomputed logs against
  uncached optical and thermal coefficients bit-for-bit on CPU, CUDA, and CUDA
  fast-math. Includes subnormal colors/densities, extreme distances, transparent
  channels, zero density, and albedo endpoints. Existing bounds/oracles remain.
- Committed-state tests check host/device cache publication, changed constants,
  caller rounding-mode restoration, constant/field transitions, observer-only
  field updates, invalid-constant rejection and recovery. Each sampled cached
  result is compared bit-for-bit with uncached GPU binding evaluation.
- 37 selected CTests passed: optical/reference/binding/stored-provider tests,
  aggregate/world/lattice tests, blackbody/install, public principled rendering
  and policy, sample addressing, denoiser reuse, and fog compatibility.
- Compute Sanitizer memcheck with full leak checking passed for
  `rtx_test_VolumeOpticsReferenceCUDAFastMath`, `rtx_test_PrincipledVolumeState`,
  and `TestPrincipledInteractive`: zero errors and zero leaked allocations.

Local evidence is under `_out/principled_volume/performance/`:
`benchmark.lua`, `summary.json`, `log-cache-confirm-*.nsys-rep`, matching logs,
images, and `log-cache-confirm-telemetry.csv`. The earlier baseline report is
`RESULTS.md`. Build/test logs use `/tmp/visrtx-log-cache-*`.

### Remaining work

This slice does not cache ray/span-invariant traversal or lattice calculations,
reuse camera optical state for phase kernels, alter stored-field sampling, or
change shadow integration. The cached-path profile below supersedes the original
source-PC distribution when selecting the next target.

## Cached-path profile (2026-11-03)

Re-profiled the cached Release library without further renderer edits, using the
same headless runner, historical archive, rates, camera, and light. Captures
select the third `optixLaunch` after two warmups, with `--clock-control none`.
The host-managed lock was left untouched. Recorded graphics clocks remained
2400 MHz; memory reported 9501 MHz under load and 10001 MHz at idle. The installed
Vela runner is now the 2026-10-30 package. Non-replayed frames 1, 2, and 4 from the
line-info run match the earlier cached 128x128 PPM files byte-for-byte.

### Updated function attribution

128x128 Release, sampled warp PCs (not exclusive milliseconds):

| Function | Sample share |
|---|---:|
| Attenuation callable, excluding callees | 72.05% |
| Stored scalar-field sampler | 17.91% |
| Camera continuation states combined | 9.80% |
| Scalar sampler callable wrapper | 0.23% |

The reference transmission-log routine is no longer a sampled hotspot.
FP64 arithmetic remains prominent: DFMA, DADD, and DMUL instructions together
account for approximately 49.2% of sampled PCs. Do not interpret PC samples at
an instruction as its isolated latency; dependencies and stalls are included.

### Full-resolution hardware confirmation

640x640 Release, four hardware-counter replay passes:

- Kernel duration: **4.820780 s**, consistent with the earlier Systems result.
- FP64 pipeline: **83.20%** peak sustained over elapsed cycles.
- Achieved occupancy: **32.68%**.
- Local requests: **5.986 billion loads**, **4.144 billion stores**.
- DRAM throughput: **0.474%** peak sustained.
- Executed warp instructions: **484.754 billion**.

Thus caching logs reduced elapsed time but did not make the remaining program
memory-bandwidth bound. Local-memory traffic remains large and is not exclusively
attributable to compiler spills; stack/callable state also contributes.

### Inlined source attribution

Rebuilt the same changes in RelWithDebInfo to resolve source locations, then
captured a separate 128x128 profile. Across unique sampled PCs, source attribution
places 32.17% in `aggregateVolume.h` and 10.18% in `volumePreviewLattice.h`.
Hot locations within the attenuation callable include:

| Mapped source region | Share of all sampled PCs |
|---|---:|
| Slab-intersection direction branch (`aggregateVolume.h:79`) | 19.01% |
| Lattice step division (`volumePreviewLattice.h:40`) | 8.41% |
| Ray-origin transform arithmetic (`aggregateVolume.h:60`) | 6.10% |
| Transform-origin initialization (`aggregateVolume.h:57`) | 2.64% |
| Per-cell extinction exponential (`volumePreview.h:114`) | 7.07% |

These optimized line mappings include associated/inlined instructions; the
19.01% entry does not mean a literal comparison alone consumes that time. Source
file shares and function shares are different partitions and must not be added.
The combined source export repeats 16 instruction addresses; the aggregation
checks matching sample counts and counts each PC once.

### Next bounded target

The evidence supports caching ray/span invariants in the attenuation march:
transformed rays, clipped/full-domain intersections, lattice anchors, and step
lengths. `nextCell()` currently repeats these calculations for each cell, and
`extinctionAt()` repeats occurrence intersection and ray transformation again.
Keep exact arithmetic order, endpoint/tie rules, ROI cuts, and additive overlap
semantics. Avoid replacing the current uncapped overlap support with a fixed
hit limit. Cache storage and callable lifetime must be designed without simply
moving the same cost into a larger per-thread stack.

This pass identifies the target; it does not implement that cache or claim its
speedup. Field sampling is the second major component, but is not the first
remaining target. Thermal approximations and blanket FP32 conversion are not
justified by this profile.

Evidence: `_out/principled_volume/performance/cached-profile/`, containing Release
128/640 captures, the RelWithDebInfo line-info capture, binary fingerprints,
telemetry, complete SASS/source exports, and aggregation scripts/results. Both
build configurations now contain the transmission-log cache; `_out/lib` was not
reinstalled. No additional renderer changes or clock-control operations occurred.
