# Subway follow-up: per-ray reuse candidate

Historical checkpoint: correctness verified; performance was blocked by changing
GPU conditions. The subsequent [clock-leased rounds](principled-volume-leased-optimizations.md)
qualify the retained Quality reuse, revert the slower preview traversal candidate,
and add qualified preview point sampling. The remainder records the earlier
blocked experiment, not the current qualification status. Do not treat preliminary
measurements as a qualified speedup or silently replace the previous frozen
baseline (`12ffb09b`). No installed library, clock policy or other process was
changed by this pass.

## Candidate

- `aggregateVolumeTracking.h`: after `nextSpan` establishes a nonempty active
  interval in a one-occurrence scene, retain its object-space ray and matching
  global proposal rate for the flight. Source, optical-only, shadow and coverage
  callbacks reuse that ray instead of repeating transform/slab arithmetic at
  every candidate. Domain membership at sampled positions remains checked by
  the same coefficient evaluators. Global bounds, candidate locations, residual
  optical depths, RNG draws and source ownership are unchanged.
- Multiple occurrences retain the complete uncapped traversal. Their candidate
  evaluation computes each transform once and reuses it for intersection and
  sample position instead of computing it twice.
- Reference and FP32 preview span enumeration similarly reuse the local ray.
  A field domain containing the complete Medium Domain cannot add an interior
  endpoint, so it avoids a redundant clipped slab calculation. Independent,
  smaller, empty and disjoint field domains still use the original cuts.
- No spatial majorant approximation, density-derived source skipping, sample
  reduction or surface-shader change is introduced.

## Verification

A CPU/CUDA/fast-math control compares actual single-occurrence flights and shadows
against forced generic traversal: an empty second occurrence changes the dispatch
without changing the active set. Events, flight distance, RGB source/throughput,
shadow transmission and final random state must match exactly. Existing transformed,
overlapping, thermal, field-domain and phase controls retain their analytic gates.

-169/169 selected tests pass in806.08 s (compile benchmark excluded as before).
- Eight previous subway randomizations replayed at128-square/512 retained spp:
 128 cumulative HDR frames match byte-for-byte. Six Interactive cumulative target
  frames also match exactly.
- Six canonical scenes in each renderer contribute another192 retained cumulative
  frames, all byte-identical:326 retained frames total.
- Explicit OptiX memcheck/leak checks for aggregate tracking CUDA fast-math,
  public Quality and public Interactive pass with zero errors/leaks.

## Measurements and blocker

Preliminary unprofiled640-square, one-sample Quality runs were76.7–77.7 ms versus
about87–89 ms at the previous checkpoint. Interactive stayed near31 ms. These
suggest that per-candidate transform/intersection work is worth removing, but
are **provisional**, not accepted performance results.

The subsequent long matched campaign was contaminated: clocks varied from2055
to2715 MHz while active, with idle210 MHz samples, instead of the previous stable
2400 MHz telemetry. GPU-wide memory rose by roughly3 GiB and baseline camera/
source-update cases sometimes exceeded170–210 ms. The final idle observation,
with no visible benchmark process, still showed about2968 MiB and10% GPU activity;
process details were not exposed in this environment. Do not identify or terminate
an unseen process, infer a memory leak from GPU-wide totals, or change the external
clock policy. All measurements are retained, not cherry-picked.

Evidence: `_out/principled_volume/subway-next/` (`plan.json`, `timing.json`, raw
frames, telemetry, frozen candidate library, qualification script/logs). The plan
uses exact HDR equality because this is an arithmetic-reuse optimization, not a
new estimator. Timing and memory readouts must be repeated with a quiet GPU and
an agreed stable clock configuration before keeping/reverting individual preview
changes on performance grounds or committing this candidate.

Next: repeat short interleaved baseline/candidate captures for both renderers,
then post-change Nsight attribution if the gain persists. Only then proceed to
larger spatial-bound work, which requires matched reference range certification
and should not borrow the approximate preview cache as a majorant.
