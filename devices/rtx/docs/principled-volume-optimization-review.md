# Volume optimization review: remaining candidates and measured outcomes

Follows [the clock-leased rounds](principled-volume-leased-optimizations.md).
All timings below are leased2400 MHz,640x640, one full sample, depth32,
no checkerboarding/denoising, on the extracted subway scene unless stated.
Baseline for this review is the leased-qualified candidate
(`_out/principled_volume/leased-volume-qualified/final/library/`): about
23.9 ms Interactive and83.6 ms Quality.

## Profile summary before this review

Quality: volume callable52.5% and reference scalar sampling22.7% of sampled PCs,
FP64 activity48%, DRAM29%. Interactive: camera segment26.7%, texture density
integral13.8%, attenuation callable12%, point sampling8.4%. PC shares are not
elapsed-time partitions; ablations below quantify actual time.

Ablations (diagnostics, not optimizations): making every Interactive volume
attenuation query return one changed23.3 ->21.0 ms, so shadow/attenuation work
is at most about2.3 ms; the volume-off ablation leaves2.95 ms. Camera quadrature
therefore dominates Interactive. Light-count caps (maxSampledLights0/1/2/4) had
no effect: the scene instances exactly one directional light, and the light-aligned
shadow cache is prepared (complete, eligible, one occurrence, rate1, origin limit
2150). Shadow rate4 bypasses the cache and costs285 ms, confirming the cache's
role at rate1.

## Candidates measured and rejected

1. **Certified local optical bounds (Quality).** Implemented a32^3 per-volume
   density range grid built on the GPU with the canonical endpoint decoder,
   footprint expansion including binary32 position margins, the same
   `buildBounds` construction per cell, cells intersected with the global
   certificate, and a piecewise-constant majorant walk during tracking (DDA
   with carried residual optical depth). All tracking/world/public controls
   passed. Performance: subway Quality84 ->105 ms with the walk,94 ms with a
   candidate-count cost gate; canonical cloud22 ->37 ms ungated. These media
   already have low global majorants, so null-event rejection is not the
   bottleneck; cell traversal costs more than it saves. Fully reverted.
2. **Coverage/radiance first-hit reuse (Quality).** Identical images, but paired
   runs measured83.5 ->92 ms, consistent with extra live state in raygen.
   Reverted.
3. **Preview span enumeration reuse / full-domain cut skipping.** Slower in the
   earlier leased pass; reverted.
4. **Lazy source-binding evaluation, diagonal inverse-map specialization.**
   Numerically identical in controls, neutral in time; reverted.
5. **Hardware-linear texture sampling for preview optics.** Within the frozen
   preview gates (subway NRMS1.4e-4, plume2.9e-5) and faster on subway/plume/
   cloud (23.3 ->21.9,17.0 ->15.3,29.0 ->26.3 ms) but slower on separate support
   (6.8 ->7.4 ms), and it changes preview sampling semantics beyond the currently
   documented shadow-only use of the linear texture. Not retained; recorded as a
   possible future policy decision with its measured error budget.
6. **Surface-point directional shadow lookup in the light-aligned cache and the
   zero-radiance bounce skip (Interactive).** Exact, but within noise (<=1%) once
   measured in paired leases; they also touch the Interactive surface shading
   file that the user wants separately reviewable. Reverted.

## Miss pre-test: rejected during pre-commit review

The round initially retained `gpu/volumeMissTest.h`, a binary32 pre-test used by
Quality medium shadows and both Interactive attenuation wrappers. Paired leases
measured Quality83.5–83.6 ->81.2–82.5 ms; Interactive was unchanged within noise.
The retained scene replays were bit-identical, but did not establish conservatism.

Pre-commit review reproduced a false miss: with inverse scale
`1 / double(1e38f)`, object-space X translation -1, domain
`[(0,-1,-1),(1,1,1e20)]`, and a world ray from zero in direction `(1e-10f,0,1)`
over `[0,infinity)`, reference traversal intersects at approximately
`[1e48,2e48]`. Narrowing the transformed X direction to float produces zero;
the pre-test then rejects the ray as parallel and outside. Domain widening does
not protect against direction underflow. The shortcut and all three call sites
were removed rather than accepting the unproven claim of conservative rejection.
No changes from this round remain in production.

Current round verification:169/169 selected tests (581.43 s); OptiX memcheck for
public Quality, Interactive and source lighting zero errors/leaks. Evidence:
`_out/principled_volume/leased-volume-round3/` and the rejected-candidate runs in
`_out/principled_volume/leased-volume/`.

## Assessment

Exact volume-side micro-optimizations are exhausted for these targets: remaining
gains were within a few percent or negative. Larger speedups now require policy
decisions rather than engineering alone:

- Interactive: hardware-filtered preview sampling (measured above), or fewer
  quadrature samples per ray, both of which trade accuracy within or beyond the
  current preview gates.
- Quality: fewer field evaluations per candidate (FP32 reference sampling with a
  matched certification), or estimator changes such as different source/light
  connection strategies, each needing new statistical qualification.

Neither is adopted here. With the miss pre-test removed, the applicable retained
checkpoint is the preceding leased round: about84 ms per full Quality sample
and23.8 ms Interactive on subway. The approximately82 ms result above belongs
only to the rejected pre-test candidate.
