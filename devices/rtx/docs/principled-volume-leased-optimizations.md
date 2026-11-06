# Clock-leased volume optimization rounds

Status: bounded correctness, performance and memory qualification completed for
both Interactive and Quality. Changes are volume-only and remain separate from
the earlier surface checkpoint. No archive modification, install, push or
extension advertisement was performed.

## Clock protocol and baseline

The user authorized the repaired `gpu-profile-clocks` broker to set GPU-wide
clocks and restore automatic management afterward. Each timing/HDR/profile
campaign ran inside a bounded lease; successful client exits acknowledged release
and reset. Nsight Compute used `--clock-control none` inside the lease. Active
graphics telemetry stayed2400 MHz; memory telemetry contained9501 and10001 MHz.
The GPU returned to idle210/405 MHz and2 MiB after the campaign. Host policy
configuration/max duration was not exposed inside the sandbox; those values are
not inferred from the client. Failed/expired leases would invalidate measurements.

Baseline is the frozen `12ffb09b` library under
`_out/principled_volume/subway/final/library/`. Do not compare absolute numbers
against earlier unlocked/contended captures as though the clock policy matched.
Target remains the same unmodified subway application-state archive and extracted
scene, camera0, layers and materials. Primary comparisons use640x640, one full
sample, depth32, rates1/1, no checkerboarding, denoising, ambient term or firefly
filter.24 launches/run; discard two warmups; reverse baseline/candidate order in
the second static repetition. Saved-viewer settings are reported separately.

Evidence: `_out/principled_volume/leased-volume-qualified/` contains the frozen
candidate, plan, captures, telemetry, lease acknowledgements, HDR, profiles and
`analysis.json`. Exploratory/rejected results remain in
`_out/principled_volume/leased-volume/`. Reanalyze with
`tools/principled_volume/analyze_leased_optimizations.py ROOT`.

## Retained implementation

### Quality: reuse invariant ray state

After span enumeration establishes a nonempty interval in a single-occurrence
scene, candidate traversal retains the object ray and global proposal rate for
that flight. Radiance, shadow and coverage queries reuse them instead of repeating
matrix transforms/slab tests per candidate. The rate retains the same rounded
bound-reduction graph. Domain membership at sampled points is unchanged.

Multiple occurrences still use the complete uncapped path. Each per-candidate
occurrence transform is computed once for intersection and position. Reference
span enumeration also reuses its transform and avoids clipped field slabs when
metadata proves that the field domain covers the entire Medium Domain. Smaller,
disjoint and empty field domains retain their cuts. No source-support inference,
new majorant or work cap is introduced. The analogous preview traversal change
was measured slower and reverted.

### Quality: reuse achromatic optical results

`evaluateOptical` reuses the first executed absorption/scattering result when
transmission-color bits, albedo bits and any supplied cached logarithms agree
across RGB. Independent RGB emission is always retained. All original scalar
operations and their order remain; no division is replaced by a reciprocal or
reassociated factor. Chromatic inputs take the previous per-channel path.

### Interactive: point textures with software filtering

The existing optional immutable scalar texture array gains a second texture
object using **point** filtering. Reads return the stored/decoded scalar endpoints;
the original FP32 software trilinear interpolation still computes the weights and
combines them. This is not hardware linear interpolation for optics. The original
linear texture remains shadow-only, and **neither view is used by Quality's
reference sampler or to construct tracking majorants**.

Eligible scalar fields are prepared, including temperature rather than density
alone. Preparation still uses canonical endpoint decoding, includes inactive/tile/
background values, and retains the existing dimension/allocation bounds and
fallback on unsupported fields or allocation failure. The two texture objects
share one CUDA array per field; there is no second voxel-data copy for the point
object. Additional scalar fields such as temperature may need their own array.
RGB/custom/unsupported fields keep their existing samplers. Field generations
continue to own/invalidate the representations; there is no user cache toggle.
The internal stored-view contract is bumped from3 to4 for the layout change.

Point sampling has its own noinline device function. Initial attempts to merge
point/tree paths into the camera sampler regressed the separate-support control
and increased memory use. Isolating the point function fixed that without a
scene-specific restriction. A trial ROI-based eligibility heuristic was also
removed: independent field domains retain the normal optimization/fallback rules.

## Matched results

### Subway: full-frame work

|Mode|Before GPU median ms|After GPU median ms|After render/prepare wall ms|
|---|---:|---:|---:|
|Interactive static, order1|34.37|23.76|23.79|
|Interactive static, order2|34.65|23.94|23.96|
|Interactive camera motion|34.14|24.04|24.07|
|Interactive density edits|33.75|23.45|23.93|
|Interactive temperature edits|34.00|23.70|24.17|
|Quality static, order1|96.38|83.89|83.92|
|Quality static, order2|96.58|83.97|84.00|
|Quality camera motion|95.61|82.27|82.30|
|Quality density edits|96.10|82.40|87.75|
|Quality temperature edits|95.45|81.99|87.64|

About31% less Interactive time and13% less Quality time on the subway. Interactive
static maxima25.43/25.04 ms; Quality91.70/93.46 ms. Quality on this larger scene
remains well short of30 full samples/second. Camera motion edits X by one scene
unit; density/temperature modes change multipliers, not raw voxel arrays.

Saved viewer settings: Interactive41.46->31.29 ms, Quality47.12->43.45 ms.
Quality here includes checkerboarding, denoising and depth12. These are partial-
update latencies, not the full-frame throughput used above.

### Other canonical640-square cases

|Renderer/case|Before ms|After ms|
|---|---:|---:|
|Interactive cloud|34.87|29.01|
|Interactive plume|29.33|16.98|
|Interactive separate support|8.20|6.79|
|Interactive overlap|0.76|0.79|
|Quality cloud|28.44|22.19|
|Quality plume|28.39|23.91|
|Quality separate support|11.06|8.96|
|Quality overlap|9.28|9.31|

The tiny overlap case is effectively neutral/slightly slower; no universal win
is claimed. No resolution/sample-count reduction contributes to these gains.

### Memory and attribution

Steady subway GPU-wide usage in the isolated paired runs: Interactive3933.75 ->
3953.75 MiB (+20 MiB); Quality2533.75 ->2595.75 MiB (+62 MiB). These totals include
CUDA/OptiX modules/stacks and context state, not just texture storage. They are not
per-grid allocation figures. The earlier merged point path had much larger
observed overhead and was not retained.

Leased Nsight captures: Interactive34.97->23.40 ms, occupancy17.61->19.80%, DRAM
4.41->7.04%. Camera-integration PC share fell43.8->26.7%; point sampling is8.4%.
Quality98.24->87.27 ms, FP64 active56.36->47.96%, occupancy31.55->31.53%, DRAM
23.69->28.58%. Volume-callable PC share fell58.7->52.5%; reference scalar sampling
remains22.7%. Replay timings/PC shares are diagnostic, not elapsed-time partitions
or replacements for the unprofiled paired timings.

## Fidelity and verification

-169/169 selected tests passed in592.00 s after rebuilding all affected consumers;
  compile benchmark excluded as in earlier campaigns. A subsequent header-comment
  rebuild changed the shared-library hash but retained all15 embedded PTX blobs
  byte-identically;11 focused tests passed again. The manifest preserves both the
  measured binary and this source-build binary rather than conflating their hashes.
-224 retained Quality HDR frames (eight subway randomizations plus six canonical
  scenes) are **byte-identical** to the baseline. These changes do not trade noise
  for throughput; fixed-work error is unchanged for those retained samples.
-102 Interactive retained frames pass the unchanged HDR gates: relative L1<=.005,
  NRMS<=.01, alpha<=.002. Subway difference is around2.1e-12 NRMS; canonical cloud
  around1.5e-8; other controls unchanged. Black references require exact equality
  rather than inventing a relative error with zero denominator.
- Point-texture controls compare4096 locations per fixture against the ordinary
  software sampler: five scalar NanoVDB formats, negative indices, inactive/
  background values, cell/vertex centering, dense data and an array revision.
  All samples are equal; reference sampling with/without the optional texture
  handle is also equal. These cover49,152 sampled locations across12 fixtures.
- CPU/CUDA/fast-math optical controls compare the reused graph byte-for-byte with
  the frozen independent-channel graph. Tracking controls compare cached and
  forced-generic flights/shadows, including final RNG states and overlaps.
- Ten explicit OptiX memcheck/leak workloads pass: stored fields, reference optics
  fast-math, aggregate tracking fast-math, production Quality probe, public
  Interactive, public Quality, dynamic updates, custom providers, and the actual
  subway in both renderers at32-square. Zero errors/leaks.

### Build-artifact qualification issue

An intermediate run found an illegal access in a test consumer after payload
layout changes. Ninja reported **zero header dependencies** for five CUDA test
objects, leaving stale binaries despite library rebuilds. All five objects were
explicitly rebuilt, their dependency lists were restored, and the final complete
suite/sanitizers were rerun. Intermediate passes from stale objects are not counted
as verification of this pass. The origin of the missing dependency records was
not established; no tool limitation or driver fault is asserted. Logs of failures,
dependency inspection and final verification are retained.

## Pre-commit review and revalidation

The retained ray reuse, achromatic reuse and point-texture implementation passed
source review. The later miss pre-test was removed after a compiled counterexample
showed direction underflow causing a false miss; see the
[optimization review](principled-volume-optimization-review.md). It is not part of
this checkpoint.

Fresh validation after that removal:

- Release build and 169/169 selected CTests pass (805.26 s), excluding the MDL
  compile benchmark as in the earlier campaign.
- Four memcheck/full-leak workloads pass with zero errors/leaks: stored reference
  fields, aggregate tracking CUDA fast-math, public Interactive and public Quality.
- Re-analysis of the original leased campaign passes its 224 exact Quality frames
  and 102 Interactive error-gated frames; these are historical captures, not new
  renders.
- Fresh subway captures reproduce six Interactive and sixteen Quality retained
  frames byte-for-byte against this round's frozen library.
- Reversed-order leased 640-square, one-sample subway runs measure 23.67–23.84 ms
  Interactive and 83.59–84.40 ms Quality for the reviewed build. Frozen-library
  controls measure 23.76–23.80 ms and 83.09–84.14 ms, respectively. These support
  checkpoint equivalence, not an additional speedup.
- Active graphics clocks remain 2400 MHz; memory telemetry contains 9501/10001 MHz.
  The lease client exits successfully after release/reset; subsequent telemetry
  shows idle 210/405 MHz, 0% utilization and 2 MiB. Host policy configuration is
  not exposed in this environment, so its maximum lease duration is not inferred.

Evidence and fresh verification logs live in
`_out/principled_volume/baseline-review/`. Source review also retains the animated
plume tooling as a separate change for future dynamic-scene experiments. Its
Python suite runs 23 tests with two optional OpenVDB-dependent checks skipped;
NVDB generation and fresh-process Vela animation archive validation pass. Duplicate
root-level CIE downloads are removed; the CPU reference tooling now uses the
identical versioned datasets under `device/gpu/blackbody/data/`.

## Rejected work and next limits

Lazy source-binding evaluation and diagonal inverse-map specialization were
numerically correct in their focused controls but neutral in measured speed, so
both were reverted. The slower preview span change and ROI-specific point-fetch
heuristic were also removed. No dormant experimental path is left behind.

Further large Quality gains likely need spatial optical bounds or a better
source-proposal strategy, with matching certificates/statistical qualification.
This round deliberately preserves the controlled reference arithmetic and full
source support rather than borrowing approximate preview bounds or weakening
fidelity gates. Existing extreme-coordinate, deployment and multi-context
sanitizer limitations remain separate; no surface qualification scope is removed.
