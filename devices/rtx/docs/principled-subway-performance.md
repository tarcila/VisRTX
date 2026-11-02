# Principled subway performance pass

## Checkpoint and submission split

User confirmed the installed viewer looked correct and authorized a checkpoint,
then requested a performance pass for both Interactive and Quality using
`test-principled-volume.vsr`. Existing changes were split into:

- `fc94143e` — `fix(rtx): qualify native and MDL surface sampling`.
- `ca33ce67` — `feat(rtx): qualify and accelerate principled volumes`.

The surface commit builds independently in `/tmp/visrtx-surface-checkpoint` with
12 focused material/MDL/fog/draw-order tests and the complete145-test selected
surface-checkpoint suite passing (291.24 s, compile benchmark excluded).
Isolating it exposed an early
Dead-event exit that skipped backdrop coverage in legacy Quality; the surface
commit now records accepted coverage before that exit without consuming dead
sample fields. Combined Quality/source/fog controls also pass. The volume commit
depends on the surface-event contract, but surface repairs contain no Quality
volume activation. Further surface work should remain separately reviewable.
Unrelated root files and the separately added animation tooling were not included.

## Target and measurement contract

Original archive SHA256:
`96486702da001a82cbb70075435f207ba41efb794327447d68c14d904986fd53`.

This is a995,778,744-byte **viewer application-state archive**, not a standalone
scene. Its `archives.scene` was extracted through Vela's canonical text tools;
the original is unchanged. It contains one principled volume, two spatial fields,
64 surfaces/geometries,18 materials,333 arrays and a disabled directional light.
The subway has emissive/native textured surfaces and transmitting glass. Density
multiplier is0.02; albedo0.2; anisotropy0.5; thermal intensity1. Camera0 is retained:
position[93.8433,110.571205,225.15753], direction[-.115681164,-.061351676,-.9913899],
FOV32 degrees. Layer transforms and enable states are preserved.

Two settings profiles are distinguished:

1. **Controlled work:**640x640, one sample/launch, depth32, rates1/1, checkerboard
   and denoiser off, firefly filtering off, ambient off, black transparent plate.
   This is the optimization/fidelity comparison, not an exact reproduction of
   saved viewer defaults. No reduced-work setting counts as a performance gain.
2. **Recorded viewer settings:** Interactive uses ambient0.25/one AO sample,
   tonemap firefly filter, gray plate, no checkerboard/denoiser. Quality uses
   depth12, color denoising, checkerboarding, no firefly filter and black opaque
   plate. The archive's active viewport selects Quality. These timings are listed
   separately and are not full-frame sample rates.

`profile_archive.lua` loads the extracted scene with the selected profile.
`archive_capture.cpp` interposes public ANARI calls to record GPU duration,
render/prepare wall time, sample counts and GPU-wide memory. Timing mode leaves
color format unchanged. HDR mode requests float4, saves linear samples, then
supplies an sRGB8 display copy to Vela; conversion/image I/O is outside recorded
render timing. Lua signed integer seed parameters are marshalled to the public
uint32 randomSeed type. Optional camera motion changes its public position before
render. No private renderer toggles or estimator instrumentation are used.

24 launches/run, first two excluded (viewer checkerboard readout additionally
reported after startup). Motion is sinusoidal camera X amplitude1 scene unit or
10% density/temperature multiplier edits. Raw field-array replacement remains
covered by the separate dynamic/API suite, not this scene-motion benchmark.
RTX5880 Ada, driver615.78.08; external clocks preserved. Nsight Compute uses
`--clock-control none`. Frozen before/after libraries, telemetry, raw data, scripts
and Nsight captures live under `_out/principled_volume/subway/`.

## Findings and retained changes

Baseline GPU medians are about33 ms Interactive and130 ms Quality. Disabling the
volume only as a diagnostic reduces final time to2.95 ms/11.72 ms respectively;
disabling thermal/explicit source but retaining optics gives30.29 ms/78.41 ms.
These ablations change the image and are not optimization results.

Baseline PC sampling attributes about31% to Interactive's general float field
sampler and substantial additional work to camera/shadow quadrature. Quality
attributes63% to the volume callable and21.5% to the scalar reference sampler.
PC shares are not elapsed-time partitions. Quality FP64 activity is55.8% of its
own peak, achieved occupancy32%, DRAM13%; this is not a surface-only bottleneck.

Retained **volume-only** changes:

- `aggregateVolumeTracking.h`: when a true surface/medium vertex has transferred
  source ownership to explicit sampling, continuation flight uses only its matching
  optical bound. It no longer generates positive-source-floor events with no
  source contribution. Camera, transmission/delta and unqualified closures retain
  their full source floor. This is ownership-based, never density-derived source
  omission. A new CPU/CUDA/fast-math control verifies one random optical-depth
  draw and zero candidates for unowned vacuum while owned vacuum still emits.
- `StoredFieldReferenceSampler.h`: reuse NanoVDB hierarchy traversal across an
  interpolation footprint via its accessor. The exact endpoint decoder, mapping
  and controlled interpolation graph stay unchanged, including packed formats,
  inactive values, tiles and background. No new majorant arithmetic is needed.
- `ReferenceSampling.h`: inline the ordinary float NanoVDB preview path at its
  caller rather than repeatedly crossing the all-format noinline sampler. Other
  formats/providers retain their fallback. No quadrature cells or samples are
  removed and no hardware filtering replaces optical sampling.

Rejected candidates are recorded, not kept dormant: a leaf-only accessor variant
failed public/stored-field fidelity and lost plume contributions; persistent
camera density/temperature accessors were correct but slower (~35 ms); replacing
the custom float decoder call with native getValue was neutral. The apparent
3-ms result from the failing leaf variant is **not** a performance success.

## Matched timings

|640-square controlled mode|Before GPU ms|After GPU ms|After render/prepare wall ms|
|---|---:|---:|---:|
|Interactive static|32.82|31.49|31.53|
|Interactive density edit|31.89|30.45|30.92|
|Interactive temperature edit|32.00|30.53|31.03|
|Interactive camera motion|32.48|30.70|see raw timing|
|Quality static|129.83|89.36|89.40|
|Quality density edit|129.62|87.85|92.74|
|Quality temperature edit|128.93|86.82|91.92|
|Quality camera motion|129.29|86.03|see raw timing|

Reversed-order static repetitions: Interactive33.46->31.14 ms;
Quality130.28->86.59 ms. Thus this pass saves roughly4–7% Interactive and31–34%
Quality on the target. **Quality is not at30 FPS on this larger scene.**

Recorded viewer profile: Interactive39.87->37.55 ms; Quality52.22->42.43 ms.
Quality here uses checkerboarding plus denoising, so these are partial-update
latencies, not640-square full-sample throughput. The controlled fixed-work table
above is the primary performance comparison.

Additional canonical640-square controls:

|Case|Before ms|After ms|
|---|---:|---:|
|Interactive cloud|32.81|30.91|
|Interactive plume|28.29|26.24|
|Quality cloud|25.42|25.42|
|Quality plume|25.62|25.40|

Some long qualification runs recorded elevated GPU-wide memory as other activity
changed; do not attribute those totals to this library. A clean paired subprocess
check measured steady Quality usage2531.75 MiB before and2533.75 MiB after (about
2 MiB delta). Keep all contaminated/global readings in the evidence.

Post-change Nsight: Interactive capture duration34.03->31.87 ms, occupancy
17.99->18.24%; inlined field work is now attributed to its camera caller. Quality
capture133.31->92.28 ms, FP64 activity54.7%, occupancy31.59%, DRAM25.26%.
The remaining Quality bottleneck is still volume transport/reference sampling,
with memory traffic becoming more important. Further work should examine tighter
spatial optical bounds or source/transport variance—not silently change surface
models or lower sample work.

## Fidelity and verification

- Frozen before-run gates: Interactive HDR relative L1<=.005, NRMS<=.01,
  alpha<=.002. Eight fixed SHA256-derived paired Quality randomizations,
  128-square/512 retained spp; RGB image-mean difference must fit6 paired standard
  errors+5e-5. No seeds or tolerances changed after inspection.
- Interactive target differences: relative L1~1.58e-12, NRMS~2.08e-12, alpha0.
- Quality paired mean deltas[-.0033292,-.0009173,-.00005349] pass gates
  [.0060657,.0015373,.00014650]. Eight-mean image NRMS0.43%, alpha unchanged.
  Neither low-spp mean image is a convergence golden.
- Empirical replicate RGB variance ratio0.926; variance times128-square GPU cost
  ratio0.837 (~1.19x time-to-error diagnostic improvement). The low-resolution
  cost ratio is not the640-square throughput gain; do not conflate them.
- All six canonical Quality512-spp means are bit-identical. Interactive cloud
  NRMS1.5e-8, others unchanged; black receiver references have zero denominator,
  reported as undefined relative error rather than an invented score.
-169/169 selected tests passed in631.11 s (compile benchmark excluded as before).
- Seven explicit OptiX memcheck/leak workloads pass: source CUDA fast-math,
  stored-field reference, public Interactive, public Quality, custom fields, and
  the actual subway in both renderers at32-square. Zero errors/leaks.

No extension advertisement, optical clamps, work caps, source-support reduction,
install or push was performed. The existing multi-context sanitizer limitation
remains separate. This is a measured bounded performance pass, not complete
shipping/material qualification.
