# Quality volume transport — implementation progress

The user authorized starting Quality after the Interactive performance follow-up.
**Initial activation checkpoint:** [experimental Quality rendering is active](quality-volume-activation.md),
including the new path-loop branch, RGB medium shadows, independent coverage/AOVs
and screen-plate handling, and TF1D emission–absorption. Public analytic frames,
149 selected CTests and three sanitizer workloads pass. This is not complete
feature qualification; source importance sampling, BSDF/closure work and custom
providers remain pending. A subsequent [MDL event-consumer repair](mdl-sample-events.md)
adds explicit dead ordering and preserves live SDK event bits; 152 selected tests
pass. The subsequent [native reflection repair](native-reflection-qualification.md)
adds the shared kernel/PDF, full-support proposal and explicit native component
tags; 155 selected tests pass. Real-target MDL closure/source-query qualification
and practical emitter sampling remain pending. The first
[real generated-MDL manifest](mdl-generated-target-qualification.md) now passes:
six closures, eighteen fixture/state pairs, including actual stale-ABSORB reproduction. It is
not the complete closure/source-query matrix. Preview and its cache policy are unchanged.

**Refocused Quality baseline:** the [fidelity/performance baseline](quality-volume-baseline.md)
now measures representative volume workloads: about55 ms cloud /225 ms plume at
640x640 and one sample, with substantial remaining noise. The analytic off-camera
emitter exposes missing matte illumination and high-variance native continuation.
The subsequent [direct-source implementation](quality-volume-source.md) repairs
matte lighting and reduces the native receiver NRMS from36.2% to5.4% at512spp,
with bounded ownership/visibility controls. The [source guide](quality-source-importance.md)
then reduced receiver NRMS to1.48% and plume split-image NRMS to10.64%.
[Four hot-path optimization rounds](quality-volume-optimization.md) now yield
29.9–30.2 ms cloud and29.4–29.9 ms plume at640-square/one sample, with163 tests
and five sanitizer workloads passing. Fixed-sample noise and error-cost
diagnostics are recorded separately; these are not converged-frame rates.
Subsequent [dynamic qualification](quality-volume-dynamic.md) and
[randomized convergence/error-cost work](quality-volume-convergence.md) passed
their bounded controls. [Typed custom opt-in](typed-custom-fields.md) now supports
scalar/RGB source-built providers, optional CUDA sampling and immutable leased
resources;169 selected tests pass. Broader MDL/MaterialX, extreme-coordinate and
deployment qualification remain open; extension advertisement stays off.

The earlier checkpoints below are historical implementation records; their
pre-activation statements describe those checkpoints, not current dispatch.

## Implemented

`gpu/volumeTracking.h` provides host/device primitives using
`optical_reference::Coefficients` and controlled binary64 arithmetic:

- RGB absorption/scattering/null proposals and compensated continuation weights;
- source scoring at **all** candidates, before the selected event, including
  absorption and null outcomes;
- RGB ratio-tracking shadow factors;
- exponential optical depth carried across piecewise-rate intervals and zero-rate
  gaps (no extra Beer factor on escape);
- a positive, separately supplied source proposal on an active domain, even when
  its extinction bound is zero;
- stable inverse HG sampling without a small-g snap, two-lobe sampling with its
  complete PDF, and accumulation of a full RGB phase kernel/marginal PDF across
  explicitly supplied medium-selection probabilities;
- directional correction separated from the scattering-event correction, avoiding
  an extra albedo/scattering multiplication.

Event masses are scaled by the proposal rate before summation. The three event
intervals are ordered small-first and weights use the represented CDF interval
widths. This avoids cancellation of a small branch appended to a near-one CDF,
without probability/optical clamps. The direction proposal must use the same
medium-selection probabilities that the full-mixture evaluator receives.

These primitives trust their inputs: finite positive rate enclosing the actual
reference extinction, valid optics, open-unit random variates and representable
arithmetic. They add no content scans, per-sample validity filtering, frame-failure
API, epsilon termination, or hidden null/scattering count limits.

## Executed controls

Three CTest executables compile the same code for CPU, CUDA, and CUDA fast math.
Each runs five fixed-seed 131072-path experiments:

1. Vacuum emission: nonzero source with unit transmission and no physical events.
2. Chromatic absorption plus independent emission: analytic Beer/source solution.
3. Piecewise coefficients/rates, including an emissive zero-extinction channel:
   analytic composition across both spans.
4. Repeated isotropic scattering in a homogeneous unit slab: all-direction unit
   boundary radiance and `j=A` have exact equilibrium radiance one in every
   channel. Paths with three or more scattering events are required to occur.
5. Two overlapping scattering components with distinct RGB weights and two HG
   lobes each: weighted normalization, analytic angular first moments, complete
   lobe PDF and azimuth symmetry. The medium proposal is deliberately not assumed
   equal to any individual channel's scattering fraction.

The statistical gate was fixed before running: six estimated standard errors
plus 5e-5 absolute. It is separate from deterministic arithmetic gates and does
not replace optical/image acceptance thresholds. Host/device deterministic controls
exercise expectation identities at rate scales 1e-150, 1 and 1e150, a 1e-100
scattering branch, residual flight depth and vacuum source support. Host HG checks
also use an independent long-double inverse-CDF expression, near-extreme g and
small nonzero g. Finite successful tests are not a universal finite-RNG proof.

Verification:

- Full Release build passed.
- Broad suite excluding only `TestMdlCompileBenchmark`: **144/144 passed**,
  280.87 s. A subsequent azimuth-control addition passed all three targeted tests.
- CUDA-fast-math memcheck/full leak check: **zero errors, zero leaked allocations**.
- Logs: `/tmp/visrtx-quality-core-build.log`,
  `/tmp/visrtx-quality-core-full-tests.log`,
  `/tmp/visrtx-quality-core-memcheck.log`.

## Aggregate / phase / PCG checkpoint

`gpu/aggregateVolumeTracking.h` now connects candidate generation to strict
World reference publications, matching per-span bounds, actual reference field
sampling and the supplied calibrated thermal evaluator. It carries optical depth
across merged domains, field-domain cuts and gaps, returning absorption,
scattering or boundary escape. RGB shadow connections use a source-free evaluator;
no density-based source-support skipping or preview acceleration is consumed.
The connector requires the strict World view, not Interactive's permissive
eligible subset (whose `complete` flag alone does not establish strict coverage).

`gpu/aggregateVolumePhase.h` selects scattering occurrences by scaled RGB mean
scattering mass, samples their two lobes, rotates to a double-precision world
direction and replays all components for the complete RGB kernel/PDF. The CDF
orders positive components small-first and the PDF uses its represented interval
widths. This initial implementation uses constant storage and O(N^2) replay,
without an overlap cap. It is a correctness baseline, not a final importance-
distribution data structure or performance claim.

A PCG adapter produces open-unit binary64 variates from two integer outputs
(52 random bits plus half-bin centering). The existing float RNG convention and
integer sequence are unchanged. The integer initialization/advance helpers are
now callable on both host and device. Renderer per-pixel/ordinal stream assignment
still belongs to the pending ray-generation integration.

Additional evidence:

- Fifteen fixed-seed **32768-path** cases on CPU, CUDA and CUDA fast math: additive
  and touching overlaps, nonuniform instance scale, vacuum source, TF/principled
  mixtures, duplicate occurrences, density ROI with source outside it, permutation,
  gaps/misses, calibrated thermal coupling, first-scattering weights, two-lobe
  chromatic phase mixtures, and thirteen overlapping scattering occurrences.
- Phase normalization, weighted angular moments, component-selection frequencies
  and strict world-direction norm/dot-product controls (including rotated/polar
  bases). Statistical gates remain six estimated SE plus 5e-5 absolute; geometric
  rotation controls use 2e-14 absolute.
- `test_WorldVolumeReference.cu` additionally tracks publicly committed World/field
  snapshots on CUDA, before/after TF and density edits without unnecessary volume
  recommits. Cases include thirteen transformed/authored occurrences, mixed media,
  shared-field TF classifications and an empty World. These consume the private
  prepared-view seam; they are not public Quality OptiX image tests.
- Full Release build and **147/147 selected CTests passed**, excluding only the
  MDL compile benchmark (291.48 s). The initial full run exposed a new test-harness
  readback-count bug; copying the complete result array fixed it without changing
  the transport implementation or tolerances.
- Aggregate CUDA-fast-math and expanded World tests both pass memcheck/full leak
  checking: zero errors/leaked allocations.
- Logs: `/tmp/visrtx-quality-aggregate-qualified-{build,tests}.log` and
  `/tmp/visrtx-quality-rtx_test_*-memcheck.log`.

## OptiX callable / pipeline checkpoint

The production Quality module now defines `__direct_callable__qualityVolume`.
`gpu/renderer/qualityVolume.h` specifies its request ABI for flight, shadow,
direction sampling and phase evaluation; `qualityVolumeProgram.h` implements the
worker using the previously qualified aggregate core. A dedicated SBT slot and
`usesVolumeTrackingCallables()` separate linking these programs from activating
reference-volume Frame output. Quality links real scalar/RGB field callables;
other pipelines retain explicit placeholders for unused slots.

The call chain is raygen -> Quality transport DC -> reference field DC, with
maximum direct depth two and no continuation call in this worker. Aggregate and
thermal code expansion stays outside the raygen's surface/light graph. The common
pipeline compile-options helper is shared inline so the probe cannot silently
compile with a different payload/attribute/launch-parameter ABI.

`rtx::QualityVolumeOptix` borrows the **actual device Quality and field modules**,
links an isolated probe raygen with matching SBT indices/stack calculation, and
runs the public-commit World fixtures through OptiX. It executes flight/shadow
tracking, and direction plus phase operations on a scattering scene. The fixtures
also exercise nested scalar and RGB field calls, edits, transformed overlaps,
mixed TF/principled records and empty scenes. Probe module/PG/pipeline ownership
ends before the ANARI device; no testing entry point was added to the public API.

This is an OptiX transport probe, **not a rendered Quality-frame/image test**.
The legacy public Quality raygen still diagnoses/omits principled media. Merely
flipping its eligibility would retain incompatible legacy throughput, volume-light
MIS, opacity/AOV and screen-background behavior, so that activation was not folded
into the callable plumbing.

Verification:

- Full Release build and **148/148 selected tests passed**, excluding only the MDL
  compile benchmark (286.97 s).
- OptiX probe and expanded native World tests: memcheck/full leak checking both
  report zero errors and zero leaked allocations.
- Probe compilation exposed an optional-MDL `DeviceGlobalState` layout mismatch;
  the test now inherits the device target's compile definitions and optional MDL
  dependency. No numerical tolerance was changed.
- Logs: `/tmp/visrtx-quality-optix-qualified-{build,tests}.log`,
  `/tmp/visrtx-quality-optix-rtx_test_*-memcheck.log`.

## Work after experimental activation

The [activation record](quality-volume-activation.md) supersedes the preceding
integration checklist. The path loop, medium shadows, output policy, RNG stream
assignment and TF migration now have bounded public-frame controls.

Remaining work includes:

- Broader native material qualification, real-target MDL closure/source-query
  qualification. Explicit source ownership, practical volume-emitter importance
  sampling, bounded native kernel/proposal and MDL consumer repairs are implemented.
- Broader external-provider and convergence controls beyond the qualified typed
  custom fixtures. The source-built typed opt-in mechanism is implemented.
- Broader heterogeneous/phase-field, transform and OptiX float-ray conversion
  qualification. Existing extreme-coordinate limits remain; local-length flight
  does not repair collapsed world coordinates or authorize epsilon jumps.
- Quality performance, tighter acceleration and deployment sampling choices.
  No preview-derived majorants or quality frame-time guarantees are implied.

The fast preview evaluator, density textures and directional shadow cache remain
quadrature accelerations, **not** tracking-majorant providers. None is consumed by
this core. The existing controlled evaluator and matching bounds are the initial
integration baseline; using another arithmetic graph requires matching bounds.

The feature remains unadvertised despite experimental Quality activation. No
preview behavior change, commit, install or push is included.
