# Interactive volume cache and uncached follow-up

This implements the approved steps 1–3 after the FP32 activation. Directional
volume-shadow caching remains enabled whenever eligible; no automatic animation
policy, public toggle, or additional light/provider scope was introduced.
The uncached target is 33.3 ms at 640x640, one pixel sample and rates 1/1.

## 1. Precise invalidation

The broad aggregate generation/light stamp now triggers inspection, not an
unconditional rebuild. A separate density-integral content key includes:

- stable density field identity, revision and sampling contract (not allocation
  pointer or registry index);
- density multiplier/outside value, medium/density domains;
- occurrence transform, normalized world-space light direction;
- quadrature step/rate and the canonical set of internal field-domain cuts.

Full-domain and duplicate cuts are removed from the key. Constant transmission
color/distance refresh only the published attenuation coefficients. Camera motion,
light color/intensity, and unrelated surface/material changes do not rebuild the
integrals. Unrelated emission/temperature *values* do not rebuild them either.
Changing a field domain or the minimum step can still affect the existing
quadrature partition; those changes remain dependencies. Editing a field shared
with density is of course a density revision.

Eligibility loss clears publication; restoration rebuilds. This uses the existing
supported prepare/render/update lifecycle, without new per-sample validity checks
or an asynchronous ownership redesign.

Tests cover radiometric changes, coefficient-only updates, unrelated source
changes, in-place density edits with a new revision, domain cuts, rate changes,
scaled instances, and restoration. A public-ANARI transmission-color animation
produces one cache build for five renders, while the images follow the new
coefficients and match the wide-reference quality limits.

## 2. Light-aligned column construction

The default builder now uses a light-aligned nodal grid. Each GPU thread handles
one column, integrating the shared suffix once and writing its nodal results.
Full cells retain the anchored quadrature; each node's first partial cell is
sampled at its own midpoint. Internal domain cuts are retained. This is not a
recursive interpolation sweep: interpolation is applied only when the renderer
looks up the completed cache.

The original per-node marcher remains an internal comparison builder. Both pass
analytic controls, including non-axis directions, basis changes, world-distance
scaling and partial-domain cuts. Cache coordinate bounds/transform now describe
the light-aligned cache, not necessarily the volume's object axes.

For the measured plume, rebuilding drops from approximately **89 ms to 1.94 ms**.
The 257^3 float allocation is unchanged (about 64.75 MiB per eligible Frame).
Cache eligibility and the higher-shadow-rate bypass remain unchanged. The cache
still contains volume density integrals only, never surface visibility or radiance.

## 3. Uncached integration

Several independent changes reduce pointwise marching cost:

- Precompute regular field-ray coordinates rather than transforming every point.
- Reuse constant extinction-per-density factors and exact isotropic/single-lobe
  phase cases; field-bound optical properties retain their general path.
- Reuse segment weights for radiance/coverage and identical extinction channels.
- Avoid identity arithmetic for an already evaluated zero-optics, zero-source
  cell. Independent emission is evaluated before this decision; no future sample
  or source-support interval is discarded.
- Retain conservative zero-*constant-tile* skipping in software density integrals.
  It does not infer zero from active masks or active-only min/max statistics, and
  it is not applied as density-only skipping to camera emission.

### Light-independent density texture

Eligible regular scalar density snapshots can lazily acquire a CUDA 3D texture.
Sparse samples are expanded using the canonical decoder, including inactive,
tile and background values. Reference sampling and its certificates do not use
this view. Camera density/temperature sampling remains software-filtered; the
texture accelerates **shadow density integration**, not preintegrated lighting.

This is why cache-off measurements can benefit without a directional shadow
cache: every ray still integrates its own density samples. Light changes never
rebuild the field texture. Field revisions replace the owning immutable snapshot
and derived representation. Allocation failure, rectilinear/nearest fields,
oversized fields and unsupported cases retain software sampling.

The policy limits each texture to dimensions <=512 and <=2^24 float samples
(64 MiB), and avoids tiny fields. The plume's additional density storage is 8 MiB
per field snapshot, shared by consumers rather than allocated per Frame. This is
an internal memory/optimization policy, not a provider-support limit. The stored
view ABI contract is now version 2; canonical reference arithmetic is unchanged.

Hardware linear filtering is not bit-identical to the software interpolant, so
it is qualified by the unchanged image/error gates, not by reference certificates.
Shadow rates above 1 bypass the hardware-filtered integration path. The fast loop
uses the same mathematical anchored sample locations/counts (with FP32
reassociation permitted), handles first/last partial cells,
and factors uniform full-cell lengths. Queries are already split at domain cuts.
A regular device call avoids an extra OptiX callable transition for texture-backed
integrals; other formats keep the existing callable path.

Direct-leaf shortcuts, speculative primary zero-tile skipping and grouped texture
read scheduling were measured but rejected: they did not improve the relevant
frame time. Their traces are retained; their code is not enabled or left dormant.

## Controlled measurements

RTX 5880 Ada, CUDA 13.4, driver 615.78.08. Host clock settings were preserved.
Telemetry was predominantly 2400 MHz graphics, with one 2325 MHz reading; memory
was 9501 MHz under load and 10001 MHz at idle. Reversed-order runs agreed.
Historical plume-128 archive and fixed camera/default directional light, 640x640,
one sample per launch, rates 1/1, no checkerboarding/denoising/filter reduction.
Two reversed-order repetitions, twelve launches each; discard the first two.

These are median render-related GPU timeline spans through background composition,
including cache construction when it occurs, but excluding GUI/file-writing time.
Ranges are the two run medians, not confidence intervals.

| Workload | Cache on | Cache off |
|---|---:|---:|
| Static directional light | 14.39–14.43 ms (~69 FPS) | 28.51–28.52 ms (~35 FPS) |
| Direction changes each frame | 16.82–16.90 ms (~59 FPS) | 28.65–28.89 ms (~35 FPS) |
| Intensity changes each frame | 14.76–14.78 ms (~68 FPS) | 28.81–28.95 ms (~35 FPS) |

Static and intensity runs build once; direction runs build twelve times. Off runs
build no directional shadow cache. The normal source/build was restored after
constructing the controlled off binaries. All other optimizations remain enabled.
There is still no public cache-off parameter.

Compared with the prior final-code ablation, uncached static rendering improves
from 176 ms to 28.5 ms (~6.2x). Direction animation improves from ~107 ms to ~16.9 ms.
The selected uncached 30 FPS target is met. This is not a universal throughput
claim for other views, resolutions, dense/rectilinear/large fields, overlaps,
multiple lights or animated volume uploads.

## Quality and verification

The original gates were not loosened: linear-HDR relative L1 <=0.5%, normalized
RMS <=1%, maximum alpha error <=0.002, plus unchanged optical/thermal and public
integration tolerances. Five frames per case/mode were compared with the preserved
wide implementation. Static plume uses 640x640; cloud and animation cases use
128x128. The independent HDR runner uses its explicitly fixed light, not inferred
Vela UI angles.

Worst observed ratios (fractions):

| Case | Cache on L1 / NRMS | Cache off L1 / NRMS |
|---|---:|---:|
| Thermal plume | 1.316e-6 / 1.555e-6 | 1.254e-6 / 1.554e-6 |
| Scattering-only cloud | 3.488e-5 / 9.694e-5 | 1.570e-6 / 7.213e-6 |
| Direction animation | 3.488e-5 / 9.694e-5 | 2.587e-6 / 1.201e-5 |
| Transmission-color animation | 4.896e-5 / 1.465e-4 | 1.727e-6 / 8.331e-6 |
| Transmission-distance animation | 3.488e-5 / 9.694e-5 | 1.570e-6 / 7.213e-6 |

Maximum alpha error across these cases is 4.492e-6. These are bounded comparisons,
not a universal interpolation-error or stochastic-convergence proof.

- Full Release build passed.
- Broad CTest selection: **141/141 passed**, excluding only the MDL compile
  benchmark (latest run: 279.83 s).
- Expanded controls exercise both builders, non-axis/scaled rays, precise keys,
  texture/software ROI spans, field-update publication and prepared coefficients.
- Seven memcheck/full-leak workloads passed: shadow cache, stored-field reference,
  principled state, FP32 arithmetic, public principled integration, and HDR plume
  with cache on and off. Zero errors/leaked allocations.

Evidence: `_out/principled_volume/performance/cache-next/final/` contains frozen
on/off libraries, static/animated traces, telemetry, HDR data, scripts and
`quality/results.json`. Earlier candidates are retained in sibling directories.
Build/test logs use `/tmp/visrtx-cache-next-*`. No install, commit or push was made
by the agent. Whole-feature shipping, Quality transport, custom providers and
extension advertisement remain separate work.
