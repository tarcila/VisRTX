# Quality volume optimization qualification

Status: bounded stored-provider/native-volume campaign qualified. This is not
whole-subtype shipping qualification or a converged-image frame-rate claim.
Evidence: `_out/principled_volume/quality-opt/`, especially `comparison.json` and
`round4/`. Reproduce analysis with
`python3 tools/principled_volume/analyze_quality_optimizations.py _out/principled_volume/quality-opt`.

## Work preserved

Performance uses640x640, one sample per launch, depth32, rates1, twelve launches
with the first two excluded. No resolution reduction, checkerboarding, hidden
work limit, optical clamp or source-support omission. Cloud/plume repeat order
was reversed. NVIDIA graphics clock telemetry stayed2400 MHz; memory telemetry
contained9501 and10001 MHz. External clock configuration was not changed.

| Round | Change | Cloud median ms | Plume median ms |
|---|---|---:|---:|
| Importance baseline | Full-support source guide |55.52|230.43|
|1|Coefficient-only candidates; skip unowned source evaluation|57.16|220.95|
|2|Source-only100–300 K lookup|56.35|165.91|
|2b|Proven binary64 spectral-underflow shortcut|—|125.70|
|2c|Source-only25–300 K lookup|—|47.06|
|3|Constant-albedo partition bounds|36.57|30.53|
|4|Single-occurrence phase reuse|29.90|29.93|
|4 repeat|Reversed cloud/plume order|30.20|29.41|

Final cloud maxima were30.95/31.21 ms; plume maxima31.75/31.66 ms.
A final host-source formatting rebuild retained all15 embedded PTX blobs byte\nidentically. Its follow-up medians were30.03 ms cloud and29.51 ms plume, with\nmaxima31.30/33.18 ms. The manifest distinguishes its library hash from the frozen\nround4 binary; focused regressions also passed after formatting.\nThese are GPU sample-launch timings, not application wall-clock or converged-frame
FPS. Other final640-square medians: support12.06 ms, overlap9.15 ms,
native receiver6.07 ms, Matte receiver4.25 ms.

## Algorithm and numerical contracts

1. Transport candidates need coefficients, not an unused phase kernel/bounds
   reduction. Unowned source terms need no temperature/emission field evaluation.
   The A/S reference graph stays unchanged. Optical-only sampling checks the same
   Medium Domain as full coefficient evaluation.
2. `QualityBlackbody` binds a volume-owned optional256-knot cold table. It uses
   log-XYZ Hermite interpolation and binary64 reconstruction/gamut restoration,
   preserving tiny radiance before large coupling gains. Preparation uses stable
   log-sum spectral integration. Missing tables and temperatures outside25–300 K
   retain the calibrated evaluator, except the proven underflow identity below.
   Eligibility currently follows the prepared preview metadata envelope; other
   inputs retain the spectral fallback, not reduced support.
   For0<T<=20 K every exp(-x) in the original spectral graph is binary64 zero:
   the attributed spectrum has minimum inverse-temperature coefficient>16000 K,
   hence x>800. Returning zero there reproduces the executed graph; it is not a
   visibility threshold or an input-temperature clamp.20–25 K remains spectral.
   These source approximations never supply or alter extinction majorants.
3. For unbound albedo a, partition the certified E upper bound with upward
   products E*(1-a) and E*a, using the same rounded subtraction as sampling.
   Monotonicity bounds each executed partition separately. Summation remains
   upward-rounded. Field-driven albedo retains E/E bounds and requires no new
   range-provider capability. No field-content scan is introduced.
4. A real scattering event in a one-occurrence scene reuses its computed S and
   samples phase bindings once. Direction/light queries use that complete phase
   function and preserve the medium-selection RNG draw. Multi-occurrence scenes
   retain the uncapped sorted-mixture path and its PDFs.

Cold numeric values are separately attributed CC BY-SA4.0 adaptations, not BSD
software data. Each prepared cold table is6144 bytes, in addition to the existing
preview table. Final GPU-wide memory was about1418 MiB cloud/1412 MiB plume;
this includes CUDA/OptiX context/module allocations, not just tables or grids.

## Fidelity, statistical diagnostics and verification

- Source-only rounds and phase reuse produced bit-identical matched512-spp HDR
  averages in the captured cloud/plume controls. Bounds tightening changes the
  random experiment; pixel equality is not an applicable acceptance criterion.
-2062 cold controls per CPU/CUDA/fast-math mode include endpoints, seams, tiny
  source, amplified source and the underflow identity. Fast-math maximum relative
  L1 error was2.858e-5 against the controlled spectral evaluator, below the
  unchanged1e-3 cold budget. Seam residual budget remains5e-5.
- Constant-albedo bounds have ordinary and162 metadata/point combinations,
  including subnormal inputs, maximal finite float scales and albedo endpoints,
  checked against executed A/S/extinction on CPU/CUDA/fast-math.
-163/163 selected tests passed in648.52 s (compile benchmark excluded as in the
  baseline). The subsequent test-only extreme-bound expansion passed all three
  tracking targets. Logs: `/tmp/visrtx-quality-30fps-qualified-tests.log` and
  `/tmp/visrtx-quality-bounds-extremes-tests.log`.
- Five memcheck/leak workloads passed: cold CUDA fast-math, tracking CUDA
  fast-math, committed World reference, public Quality and source lighting.
  Logs: `/tmp/visrtx-quality-final-*-memcheck.log`.
- Analytic native/Matte receiver512-spp NRMS stayed1.478%; mean0.010026835 versus
  independent quadrature0.010027149. Receiver samples are unchanged by these
  optimizations. Overlap mean RGB is[.481351,.440567,.289985] versus analytic
  [.481524,.440669,.290027]; its pixel NRMS2.53% remains stochastic noise.

Tighter proposals increase some fixed-sample noise. Cloud split256-vs256 NRMS
changed9.61%->10.57%; plume10.64%->12.58%. Squared split-error times128-square
batch-time ratios versus the importance baseline were0.689 cloud and0.193 plume
(about1.45x and5.18x diagnostic improvements). Support was1.104, overlap0.617,
and receivers approximately1.0. Do not claim every case improves time-to-error.
These are disjoint QMC blocks, not independent randomized replicates, confidence
intervals or convergence goldens. No statistical seed/gate was retuned.

## Still open

The practical sample-speed target is met for these two representative static
scenes, not universally. Heterogeneous convergence, dynamic-scene performance,
custom typed providers, broad MDL/MaterialX source-query qualification and
extreme-coordinate/visibility limits remain open. Existing MDL/transmission/delta
continuation ownership is preserved. Extension advertisement remains off.
