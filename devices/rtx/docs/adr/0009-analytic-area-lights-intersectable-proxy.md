# Analytic area lights get an intersectable proxy

Analytic **area** lights (`quad` → `LightType::RECT`, `ring` → `RING`) get a real
intersectable **proxy** in a dedicated light-proxy BLAS, with an analytic
custom-primitive intersection program and an integrator-side, MIS-weighted hit
deposit. This mirrors the existing Geometry-Light hit/NEE pairing (ADR 0005) but
keeps the light a *light*, with its own radiance and pdf math, rather than
lowering it to a synthesized emissive surface.

The proxy is a traceable **representation of an existing light**, never a second
light: one pick-CDF entry, shared by NEE and the hit deposit.

Before this, `RECT`/`RING` were NEE-only — added to the light-pick CDF by
`World::appendLight` with `surfaceInstanceIndex = -1`, but placed in no BLAS. No
intersectable geometry existed, so a BSDF-continuation or escape ray could not
hit them: the integrator deposits ray radiance only on a surface-emission hit or
an environment miss. Off a near-mirror BSDF (roughness ≈ 0) NEE contributes ≈ 0,
so the light was entirely absent from the reflected image.

## Rejected alternatives

- **Author a synthesized emissive surface (a Geometry Light).** No engine change;
  reuses `synthesizeGeometryLights` and `geometryLightHitPdf` wholesale. Rejected
  as the fix because it changes what the user's `quad` light *is* — materialized
  geometry rather than an analytic light — cannot represent `visible=false`
  (ordinary geometry is always intersectable), forces a hand unit conversion, and
  does not generalize. Retained as the **equivalence oracle** for testing: the
  analytic proxy and an authored emissive quad must converge to the same image.
- **A custom proxy hung off a synthesized `Surface`.** Reuses the surface-instance
  array, but reintroduces the surface/material coupling this decision exists to
  avoid and fights the `surfaceInstanceCursor` invariant.
- **An in-integrator O(N) analytic test per escaped ray.** No acceleration; a
  per-ray loop over all lights. A performance trap.
- **A triangulated proxy.** Recreates a second area representation that can
  disagree with `sampleRectLight`'s `area = |cross(edge1, edge2)|`. The analytic
  solver is instead *tested* against a two-triangle oracle, which buys the
  confidence without shipping the divergence.

## Design

Shared leaf functions live in `gpu/lightGeometry.h`, which is **host-compilable
and CUDA-free** (glm only). `sampleLight.h` pulls CUB and CUDA atomics, so math
left there is untestable on the host; keeping the leaves out of it is what makes
the pdf identity assertable in a unit test rather than merely hoped for.

- `rectEmissionCosTheta` / ring equivalent — the single `side` predicate driving
  both the NEE cosθ sign and the hit-side cull. One function, so front/back/both
  cannot be interpreted two ways.
- `rectRadiance` / `ringRadiance` — `color * intensity`, plus the ring's
  smoothstep cone falloff.
- `rectSolidAnglePdf` / ring equivalent — `(1/area) · dist² / cosθ`.
- The analytic ray/rect (plane + edge bounds) and ray/ring (plane + annulus)
  solvers, reporting a parametric uv that feeds the pdf leaf with no
  reconstruction step.

`GeometryType::LIGHT_PROXY` discriminates the proxy in the custom
`__intersection__` dispatch. The proxy record carries only a `lightIndex` into
`registry.lights` — no surface, no material, no geometry — so it cannot drift
from the light it represents.

The hit-side density is reconstructed as
`sharedSolidAnglePdf(...) × instancePickProbability(...)`, reading the same
`lightPickDelta` and `totalLightPower` the sampler reads, exactly as
`geometryLightHitPdf` does for Geometry Lights.

Proxies are excluded from shadow traversal by an OptiX visibility-mask bit, so a
light neither shadows the scene nor self-occludes. This is a mask exclusion, not
an epsilon trim: the proxy is not real occluding geometry. (Contrast
`GEOMETRY_LIGHT_SHADOW_EPSILON`, which is correct for Geometry Lights precisely
because that emitter *is* opaque.) The same mask mechanism implements `visible`
by clearing only the primary-ray bit.

Quad lands first, proving every seam; ring follows as a second `case` and a
second pdf leaf.

## Consequences

- **The hit-side density and the NEE density are one function.** This is the
  single most important invariant. Any divergence biases the MIS balance
  heuristic, silently and in a way that is very hard to see in an image. It is
  asserted directly in a host unit test, not inferred from renders.
- **One pick-CDF entry per light.** `appendLight` is unchanged; rect/ring
  additionally emit a proxy AABB tagged with the existing `lightIndex`. The NEE
  path is untouched. Double-counting is therefore structurally impossible rather
  than something tests must catch.
- Once the proxy BLAS exists, *every* renderer's rays can hit it. Renderers that
  do not deposit proxy hits (Interactive, Debug, Fast) must **pass through**
  safely. Only Quality deposits.
- The proxy enters a hit path built entirely around the assumption that a hit has
  a material, a geometry, and a surface instance — `populateSurfaceHit`
  dereferences all three, and Interactive's bounce deposit dereferences
  `hit.material->emissionIsSampleable`. A proxy has none of them. A null material
  reaching a shading call is a device crash, not a wrong pixel. This is the
  riskiest part of the change, and argues for a dedicated closest-hit path over
  null checks threaded through the shared one.
- Proxy hits must not populate the object/primitive/instance ID channels, or
  picking and ID passes stop identifying real scene objects.
- ADR 0004 rejected visibility masks for the surface/volume split on hit-semantics
  grounds. That reasoning is untouched here: this mask bit discriminates ray
  *classes* against one traversable, not two kinds of traversal.
- `visible` becomes meaningful for area lights, but is **implemented without being
  advertised**, deliberately. The parameter is honored on `quad` (and `ring` once
  it lands); a query for it still reports nothing.

  The extension that would advertise it moved: SDK 0.15 had `khr_area_lights`
  (bundling `visible` with `angularDiameter`, `radius`, `radiance`), and the 0.16
  this device builds against replaces it with `khr_light_primary_visibility`,
  which declares `visible` on **all six** light subtypes — `directional`,
  `point`, `spot`, `hdri`, `quad`, `ring`.

  Advertising it would therefore claim `visible` on `directional`, `point` and
  `spot`, which have no proxy, no extent to show, and no implementation. Claiming
  a parameter the device silently ignores is a worse defect than not claiming one
  it honors: an application can test for an extension, but it cannot test for an
  extension that lies. The extension goes in when those three are implemented, in
  one change, not before.
- As ADR 0005 put it for Geometry Lights: "Every renderer that deposits path-hit
  emission must generalize its environment-only MIS weighting, or next-event
  estimation plus the hit deposit double-counts the emitter." Light Proxies are
  the third such type, and that sentence applies verbatim.
