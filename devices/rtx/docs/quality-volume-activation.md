# Experimental Quality volume activation

Quality now renders stored-provider `principledVolume` and migrated TF1D media
through the reference aggregate transport. This is an **experimental activation**,
not complete feature/shipping qualification. The extension remains unadvertised.
Interactive/default and their eligible-scene cache policy are unchanged.

## Transport

The Quality raygen now consumes the qualified OptiX transport callable for camera,
phase-continuation and surface-continuation segments. It uses the complete strict
reference World publication, matching optical bounds, world-distance intervals,
additive occurrence optics, independent emission and calibrated absorption-coupled
thermal emission. Every candidate scores source before absorption/scattering/null
selection. Null/domain transitions do not consume scattering depth.

Scattering vertices sample the active-medium/two-lobe mixture and evaluate its
full RGB kernel/PDF. Direct lighting uses that kernel rather than the legacy
isotropic albedo approximation. HDRI NEE uses the unfolded sphere mixture density;
environment, geometry-light and light-proxy continuation deposits use compatible
MIS weights. Surface-light connections now use RGB aggregate ratio-tracked medium
transmittance, separately from surface visibility.

Medium flight/phase, shadow connections and coverage have separately seeded PCG
streams keyed by pixel and sample ordinal. RGB throughput and medium transmittance
remain double through the new source/connection calculations until radiance is
written into the float framebuffer. Existing surface material evaluations and
surface-visibility payloads remain float; this is not a universal all-input
numerical guarantee for the whole surface renderer.

The reference branch does not use the old nonzero throughput kill, pre-shadow
contribution threshold, volume-scatter position epsilon, preview density texture
or directional cache. Roulette compensates surviving paths. The existing explicit
`maxRayDepth` still bounds radiance paths; it is a user-visible approximation,
not a claim of infinite-order convergence at every setting. The initial source
proposal is one inverse world unit, raised to the reference extinction bound as
needed; it changes variance, never physical extinction. No performance/default
optimum is claimed.

## Independent output query

Coverage/AOVs are queried along the **whole original camera ray**, independently
of radiance absorption, scattering, refraction, roulette and depth limits.

- Ratio-tracking RGB survival is multiplied by scalar surface coverage holes.
- Each candidate's RGB survival loss contributes to the luminance-weighted
  first-optical-event albedo law. This marginalizes the channel-mixture event
  selection; it does not condition the public average on a surviving scatter.
- Medium normals are zero. Surface normals are unit local query results weighted
  by first-event coverage and averaged arithmetically, not renormalized afterward.
- Quality's private RGB screen-plate weight is uncovered RGB. A fully covered
  transmitting surface therefore covers the screen plate; the screen image does
  not become illumination or a refracted environment.
- Visible HDRIs/proxies cover the residual background; hidden HDRIs still provide
  illumination. Independent source survives zero alpha.
- Approximate TF picking is retained separately, including ID-only requests;
  principled picking improvements remain deferred.

Frames with reference media use diagnosed color-only denoising. Quality metadata
now reports the actual existing `color` default. Public AOVs remain independently
available. The common Frame compositor still owns background composition after
resolve/denoise.

## TF1D compatibility change

Quality now uses `A=extinction`, `S=0`, `j=extinction*TF-color`, matching Interactive.
TF color is **not scattering albedo** in these paths. This deliberately changes
Quality TF appearance and removes its old dependence on incident lighting. There
is no public legacy-scattering switch. Applications needing lit scattering should
use principled optical inputs; scalar remapping remains upstream, not new TF arrays
on `principledVolume`.

Quality asks for a strict prepared optical scene. Unavailable typed/reference
metadata is diagnosed during preparation rather than silently rendered with old
optics or omitted from an allegedly complete aggregate. Custom-provider typed
opt-in is still unfinished and is not claimed supported by this activation.
Fast/Debug/Test remain unsupported for principled media and retain legacy TF paths.

## Verification

`TestPrincipledQuality` exercises actual public ANARI render/map calls with
131072 camera samples per control (32x32, 128 samples/pixel), linear float HDR,
filtering/denoising disabled. Its fixed statistical criterion is six estimated
standard errors across pixels plus 2e-4 absolute float-resolve allowance; it is not
an image-dependent tolerance adjustment or a replacement for deterministic optics
and thermal gates. Controls include:

- independent source in vacuum and zero-alpha output;
- colored Beer/source/screen-background composition;
- duplicate occurrences and TF/principled source overlap;
- RGB medium shadows on a surface-light connection;
- surface holes versus covered glass screen-plate ownership;
- two-lobe multiple-scattering equilibrium under a constant HDRI;
- hidden-HDRI illumination without camera background ownership;
- thermal emission against the independent CIE oracle and exact pure-scattering
  thermal zero;
- coverage/AOV independence from the radiance depth limit.

The broader fog interaction test now uses a principled slab for its actual
scattering case, rather than mistaking migrated TF emission for scattering. Its
TF absorption case retains the approximate picking checks. Existing surface,
MDL/MaterialX, fog, backdrop, frame, sampler, provider and Interactive regressions
are included in the broad pass.

- Full Release build passed.
- **149/149 selected CTests passed**, excluding only `TestMdlCompileBenchmark`:
  646.84 s. The first broad run had stale unsupported-renderer expectations,
  missing TF picking metadata and a fog-test timeout; expectations/picking were
  corrected without loosening numeric gates, and the final run used a 600-second
  default timeout for the slower reference path.
- `TestPrincipledQuality`, `rtx_test_QualityVolumeOptix` and
  `TestPrincipledInteractive` pass Compute Sanitizer memcheck/full-leak checks:
  zero errors and zero leaked allocations.
- Logs: `/tmp/visrtx-quality-render-qualified-{build,tests}.log` and
  `/tmp/visrtx-quality-render-*-memcheck.log`.

A subsequent [representative Quality baseline](quality-volume-baseline.md) measures
fidelity, noise, GPU time and memory. It identifies direct volume-source sampling
and volume-transport arithmetic as the next priorities, not a broad surface audit.

## Remaining work

This closes neither the complete Quality feature nor its performance qualification:

- [Direct volume-source sampling and ownership](quality-volume-source.md) now
  provide bounded native/Matte/medium support. The full-domain proposal repairs
  missing matte lighting and reduces receiver variance, but sparse thermal plumes
  still need source-importance sampling and broader performance qualification.
- The [MDL absorbed-event consumer repair](mdl-sample-events.md) is now implemented.
  The [native reflection consistency repair](native-reflection-qualification.md)
  also has bounded qualification. Broader material and real-target MDL closure
  qualification remain pending. Retained transmission and ray-offset approximations are
  retained; passing compatibility tests does not certify those designs.
- Custom providers, broader heterogeneous/transformed/phase-field convergence,
  extreme-coordinate/float-ray conversion limits, and deployment sampling choices
  need their own qualification.
- The current reference path is substantially more expensive than preview. There
  is no Quality frame-time claim; small-first phase distribution replay is still
  O(N^2) with constant storage and no overlap cap.

No automatic extension advertisement, commit, install or push accompanies this
checkpoint. See [implementation history](quality-volume-progress.md) for the
preceding core, aggregate and OptiX-boundary controls.
