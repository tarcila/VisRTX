# Quality direct volume sources — implementation checkpoint

Historical checkpoint: initial full-domain estimator implemented and bounded controls pass.
The later [importance grid](quality-source-importance.md) and
[hot-path optimizations](quality-volume-optimization.md) supersede the performance
and next-step statements below. Broad MDL/MaterialX work
remains paused. This step adds direct volume-source sampling, not a surface-model
rewrite or an optical-majorant shortcut.

## Estimator and ownership

- Select a possibly emitting authored occurrence uniformly, without a float CDF
  or occurrence cap. Metadata excludes only structurally zero source; density
  occupancy is not source support. Duplicate occurrences stay distinct.
- Sample a world direction from a subtended-sphere cone mixed with 1/8 uniform
  sphere. Inside the sphere, or for numerically unresolved narrow cones, use the
  full sphere. Intersect the actual transformed Medium Domain, then draw uniform
  world distance on that chord. Misses and zero-source points are not redrawn.
- Evaluate only the selected occurrence's source. The score is
  `j_i * chord_length / (occurrence_probability * angular_pdf)` in dω dr measure;
  no extra r² or determinant compensation belongs in it. The full-sphere term
  preserves support independently of cone tightness. This avoids the local
  inverse-square variance singularity of uniform-volume sampling near a receiver.
- All active media attenuate the connection. A separate Quality ray type uses
  **surface holes only**, not accepted transmission tint. Ordinary-light shadows
  retain their policy.
- Camera, native transmission/delta and unqualified MDL continuations retain
  all-candidate source scoring. Native reflected continuations and real medium
  scattering continuations replace their direct source term with dedicated NEE.
  Matte gains direct source lighting without acquiring an indirect BSDF bounce.
- Incoming source is scored before a real event. Nulls, field/domain boundaries
  and coverage holes preserve ownership; an accepted event replaces it. The
  outcome of the NEE draw never restores ownership. Geometry/environment emission
  and ordinary-light MIS are not governed by this flag.
- If preparation cannot publish a complete usable source proposal, the entire
  optical scene remains continuation-owned; no source/volume subset is omitted.

This is a full-domain angular-distance baseline. It does not yet build the later
source-importance grid for sparse thermal plumes. Sampling, shadows and path
continuation use separate PCG streams.

## Fixed controls

CPU/CUDA/fast-math source tests: 65536 fixed draws for a separated box, an enclosing
box, duplicated occurrences and anisotropically transformed boxes, including a
non-unit determinant. Far references
come from independent order16/24 volume integration; the enclosing-box reference
uses an analytic rectangular-potential difference. Statistical gate: six estimated
standard errors plus1e-5 absolute. Near-source scores must remain bounded without
a distance clamp. Exact controls check ownership, metadata preparation and fallback.

Public ANARI image controls: 32x32 at512spp, source outside the camera footprint,
matte/native diffuse receivers, duplicate emitters, a fractional transmitting pane,
a covered transmitting pane and opaque source visibility. Independent Gauss-volume
reference with pixel-footprint integration. Mean residual gate: six estimated
standard errors plus2e-5 absolute. The pane is outside the primary-camera footprint
so it isolates source connections: .5 holes + .5*.8 accepted transmission gives
.9 for native continuation, not1.3 from double counting; Matte DirectOnly receives
only the .5 direct-hole term.

The existing receiver benchmark, frozen pre-source library and analytic reference
remain the fixed performance/variance comparison.

## Results

The public controls pass for matte/native direct source, duplicated labeled
emitters, the .9 hole/covered-transmission partition, a fully covered transmitting
pane restoring continuation ownership, matte's .5 direct-hole term, and an opaque
source connection. An added non-emitting absorbing slab attenuates the source
correctly, demonstrating that connection transmittance does not use only the
selected emitting occurrence. World publication tests cover source-zero recommit
and restoration without stale eligibility. No floating error gate was loosened.

At the frozen receiver setup (128x128,512spp):

| Receiver | Before | Source NEE |
|---|---:|---:|
| Matte mean | 0 | 0.01002044 |
| Native mean | 0.01002325 | 0.01002044 |
| Analytic mean | 0.01002715 | 0.01002715 |
| Native pixel NRMS | 36.22% | 5.42% |

The source-NEE mean error is about−0.067%. Both receiver materials agree because
the same Lambertian query is estimated, while native continuation no longer
counts that direct source a second time. At128x128, a32-spp batch costs20.27 ms
for native versus5.08 ms previously: roughly4x more sample work, but about44.6x
less squared pixel error, an empirical **11.2x variance-cost improvement** for
this bounded case. This is not a universal convergence-rate claim.

Warmed640x640, one-sample median (unprofiled second run):

| Scene | Before | Source NEE |
|---|---:|---:|
| Cloud | 55.06 ms | 55.93 ms |
| Plume | 225.11 ms | 231.51 ms |
| Separate support | 19.24 ms | 19.58 ms |
| Overlap | 12.90 ms | 13.12 ms |
| Matte receiver | 1.86 ms, missing light | 3.95 ms |
| Native receiver | 2.65 ms | 6.46 ms |

Cloud, separate-support and overlap diagnostic blocks are bit-identical to the
pre-source baseline. Plume now illuminates its previously dark matte floor/card,
so its changed mean is expected, not a before/after fidelity reference.

**Sparse thermal-source variance remains the next gap.** The new plume's two
held-out256-spp estimates differ by29.6% NRMS; full-domain angular sampling still
wastes draws in dark regions. A source-importance representation with a full-support
mixture is the next measured improvement, not a broad MDL audit. The displayed
plume image is not a converged reference or a claim that Quality is finished.

## Verification and evidence

- Full Release build; **160/160 selected CTests passed**, excluding only the MDL
  compile benchmark (722.35 s). Additional determinant/absorber controls and final
  formatting were built and passed targeted tests afterward.
- CPU/CUDA/fast-math proposal controls pass. Public source, public Quality and
  source-CUDA-fast-math memcheck/full-leak workloads report zero errors/leaks.
- Logs: `/tmp/visrtx-volume-source-*`.
- Frozen source-enabled library, matched render data, timings, telemetry,
  `comparison.json` and `comparison.png`:
  `_out/principled_volume/quality-source/`. Pre-source baseline remains untouched.

## Limits and ownership boundary

This changes Quality only; Interactive remains its declared preview approximation.
The source representation is O(number of occurrences), with no field-content
validation scan, source majorant, density-only support mask, or per-ray overlap
cap. Source snapshots and bounds follow the existing supported preparation/update
lifecycle. Source proposal geometry and the OptiX float visibility rays retain the
existing extreme-coordinate limitations; no universal numerical certificate is
claimed for arbitrary finite transforms.

Native reflection and medium phase source terms are replaced; native transmission,
singular events and unqualified MDL remain continuation-owned. Matte receives direct
source queries but remains DirectOnly: an accepted intervening transmission vertex
is an indirect path that Matte itself does not generate. This is why the pane
control expects .5 for Matte and .9 for native, rather than borrowing ordinary
lights' tinted-shadow approximation.

No native/MDL material definitions, ordinary-light MIS or source-independent
coverage/AOV rules are changed by this step. No extension advertisement, public
source toggle, commit, installation or push is included.
