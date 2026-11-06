# Autonomous Quality rendering work

The user authorized continuing without turn-by-turn approval: finish missing
Quality rendering features, then perform measured optimization rounds targeting
**>=30 FPS (<=33.3 ms GPU frames) at640x640, one sample per launch**, without
checkerboarding or resolution reductions. Convergence/time-to-error remains a
separate acceptance measure. No automatic install/commit/push/advertisement.

## Current rendering checkpoint

- Stored-provider principled optics, overlaps, TF migration, multiple scattering,
  source ownership, hole-only source visibility, native/Matte direct source and
  source-importance grid are implemented with bounded tests.
- MDL continuations retain source ownership unless replacement is justified;
  broad MDL/MaterialX auditing stays off the critical path.
- Custom-provider typed opt-in is now implemented as an explicit source-build
  contract with bounded fixtures. Broader heterogeneous/extreme-coordinate and
  deployment qualification remain incomplete. Extension advertisement stays off.
- The64^3 source CDF uses90% spatial importance +10% full-support angular sampling.
  Initial matched results: plume split-image NRMS29.6% ->10.6%; analytic receiver
 512-spp NRMS5.42% ->1.48%.160 selected tests and four sanitizer workloads pass.

## Completed optimization checkpoint

The [qualified rounds](quality-volume-optimization.md) reach29.9–30.2 ms cloud
and29.4–29.9 ms plume at640-square/one sample; all measured warm launches in
two runs per scene were below33.3 ms. Cold-source lookup, constant-albedo bounds
and single-medium phase reuse preserve the documented numerical contracts.
163 selected tests and five sanitizer workloads pass. Additional162 extreme
bound combinations pass CPU/CUDA/fast-math. Source-only and phase-reuse matched
HDR controls are bit-identical; bound tightening changes the sample experiment.
Diagnostic error-cost improves about1.45x cloud/5.18x plume, but support worsens
about10%; no universal efficiency or converged-image claim.

The ordered follow-up now has bounded [dynamic qualification](quality-volume-dynamic.md)
and [randomized convergence/error-cost qualification](quality-volume-convergence.md).
Dynamic source-guide rebuilds were reduced from about12–13 ms extra work to about
1–2 ms. A subsequent exact-graph probability reuse improves cloud/plume error-cost
another3–4%, with all768 candidate accumulation images byte-identical and165 tests
passing. The multi-OptiX-context sanitizer issue is independently reproduced and
remains an explicit qualification gap. [Typed custom opt-in](typed-custom-fields.md)
is now implemented and its public/lifecycle controls pass;169 selected tests pass.
Late performance sessions contain stalls also reproduced with the prior frozen
library, so an every-frame latency claim remains unqualified. Do not interpret
the static sample-speed target as completion of all
promised rendering/provider/qualification work.

## Latest clock-leased follow-up

[Leased volume optimization rounds](principled-volume-leased-optimizations.md)
retain Quality ray/achromatic reuse and Interactive point-texture endpoint
sampling with the original software interpolation. Subway fixed-work medians
improve from34.4–34.7 to23.8–23.9 ms Interactive and96.4–96.6 to83.9–84.0 ms
Quality at2400 MHz. Quality HDR remains byte-identical; Interactive retains the
frozen gates.169 tests and ten sanitizer workloads pass after stale CUDA test
dependency records were repaired. The Quality subway target remains below30 FPS;
no surface-shader, source-support or majorant approximation is included.

## Execution order and controls

1. Close source-guide evidence and lifecycle/zero-pilot/transform tests.
2. Remove redundant work while retaining the certified coefficient graph:
   unnecessary phase/bound reductions at candidates, unowned source evaluation,
   repeated single-occurrence/vertex sampling. Freeze each measurement build.
3. Optimize thermal source evaluation within existing thermal/HDR error gates;
   source approximations must not change extinction or its certificates.
4. Tighten or accelerate proposal/attenuation work with matching arithmetic bounds.
   Never reuse preview majorants or infer source absence from density.
5. Add rendering controls wherever a feature gap is demonstrated, and repeat
   representative fidelity/variance/depth/memory/performance measurements.

Stop/escalate on unresolved estimator discrepancy, unsupported required provider
contract, or exhausted useful optimization options—not by weakening the gates,
clipping source, adding hidden work caps or declaring one-sample speed converged
image quality. Preserve before/after binaries, raw HDR and logs under
`_out/principled_volume/`; record each completed round and outstanding work here.
