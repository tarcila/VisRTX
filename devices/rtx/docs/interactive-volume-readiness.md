# Interactive volume activation: readiness and compatibility

## Conclusion

The bounded **experimental Interactive/default activation is implemented and its
selected functional qualification is complete**. It is suitable for further
preview evaluation within the documented input/provider and precision scope.
This is not a shipping-readiness claim for the whole principled-volume feature,
a measured choice of production sampling defaults, or authorization to advertise
the proposed extension. No further renderer activation is implied.

The implementation and evidence are described in:

- [API contract](../../../principledVolume.md)
- [Preview transport, output policy and provider matrix](interactive-volume-preview.md)
- [Thermal qualification and installed attribution](blackbody-qualification.md)
- [Lattice assessment and remaining precision limits](preview-lattice-precision.md)
- [Aggregate preparation and eligibility](aggregate-volume-reference.md)

## Current compatibility boundary

| Area | Interactive / default | Other renderers / qualification boundary |
|---|---|---|
| `principledVolume` | Stored-provider aggregate transport, additive overlap, explicit/thermal source, direct single scattering | Quality/Fast/Debug/Test diagnose and omit it; intrinsic object validity is not renderer support |
| TF1D | Reference classification, `A=extinction`, `S=0`, `j=extinction*color`; additive with principled occurrences | Quality/Fast/Debug/Test retain their legacy TF integration in this incremental state |
| Providers | Dense regular/rectilinear scalar/RGB and supported NanoVDB regular/rectilinear formats | Custom typed opt-in remains unfinished, not retired from intended scope |
| Unavailable typed reference | Warn and omit from this renderer's eligible set; no silent legacy-optics fallback | Strict aggregate preparation still returns incomplete, not a prefix; legacy intrinsic validity remains independent |
| Coverage/backdrop | Separate uncovered RGB and tinted straight-through backdrop; zero-alpha source survives | No assertion that all legacy renderers already implement this transport policy |
| Public AOV resolve | Arithmetic sample averages with actual counts and checkerboard source selection; medium first-event weights | Shared Frame resolve changes also affect legacy renderers and are included in compatibility tests |
| Fog | First covered-surface ownership persists through tinted transmission; holes can acquire a later owner | Fast retains its per-layer cueing expectations; other fog regressions pass |
| Denoising with eligible media | Effective color-only; requested guides diagnosed by scene/renderer revision; public AOVs remain available | No volume-guide qualification or denoiser redesign is claimed |
| Picking | Existing approximate TF path and per-hit surface depth arbitration retained | Principled depth/ID improvements remain deferred |
| Sampling rates | Finite 1/1 experimental defaults, documented minima, no old maximum of 10 | No convergence/latency optimum, hidden step budget, or universal refinement guarantee |
| Discovery | Interactive/default policy metadata matches implemented rates and denoising defaults | Principled subtype/extension remains unadvertised |

These TF and output differences are deliberate consequences of the approved
incremental preview integration, not a newly exposed legacy-model selector. In
particular, applications using a legacy custom scalar field for TF1D must not
assume it is supported by Interactive's new typed optical path. The diagnostic
must be observed; the planned custom-provider adapter is still required.

## Compatibility pass

A full Release build succeeded. The broad CTest selection excluded only
`TestMdlCompileBenchmark`, to avoid starting an unrequested performance campaign.
It covered **138 distinct tests**, including native and MDL/MaterialX material
paths, device initialization, geometry, lights, frame channels, denoising,
accumulation, fog, typed fields, optical references, aggregates, thermal data and
installation, and the new preview controls.

The first shell command reached its 600-second overall limit after 105 completed
passes and during `SamplerContractsCUDANDEBUG`. That interruption was not counted
as a pass. No test processes remained. A separate tail run, with the same
180-second per-test timeout, completed the interrupted test and remaining tests.
The union of passing names was checked against CTest's JSON inventory excluding
the benchmark: **138 expected, 138 passed, zero missing, zero unexpected**. This
is combined coverage, not a claim that one uninterrupted suite run completed.

Logs for this local qualification:

- `/tmp/visrtx-readiness-build.log`
- `/tmp/visrtx-readiness-tests.log`
- `/tmp/visrtx-readiness-tail.log`

Readiness reconciliation also added a strict/permissive cache regression to
`test_WorldVolumeReference.cu`: unavailable TF reference metadata produces a
warned eligible subset only when explicitly requested; unchanged requests reuse
it without repeated warnings; a subsequent strict request cannot reuse that
subset; object validity remains intact and corrected metadata recovers normally.
That updated test and three public activation/policy tests passed afterward.
Its memcheck/full leak check reported **zero errors and zero leaked allocations**.
Previous expanded public-transport, lattice and thermal sanitizer results remain
recorded in their respective implementation notes.

## Explicit exclusions and remaining work

- Quality multiple scattering, volume-emitter lighting/source ownership, approved
  native BSDF repairs and actual MDL closure/generator qualification remain separate
  work. Passing existing material compatibility tests does not complete those tasks.
- Custom-provider typed opt-in remains pending. Existing custom scalar signatures
  were not silently reinterpreted or removed.
- No rendered plume/cloud convergence campaign, throughput study, final deployment
  rate selection, or full-render independent-replicate qualification was performed.
- Extreme coordinate failures remain possible: intersection endpoints can collapse
  before stepping; unresolvable lattice boundaries coarsen represented spans;
  float sample positions have their own limits. The documented tests characterize
  these limits rather than declaring arbitrary finite geometry accurate.
- The existing supported wait/update/prepare lifecycle is retained. No device-wide
  asynchronous lease proof, callback/interop redesign, new frame-status API,
  content-validation scan, or multiword cursor was added as an implicit prerequisite.
- No extension advertisement, commit, merge, or next transport slice follows
  automatically from closing this bounded activation checkpoint.
