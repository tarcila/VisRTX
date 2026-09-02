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
- **Sharing one density function is not the same as that density being correct**,
  and this ADR should not be read as claiming otherwise. The rect/ring samplers
  use OBJECT-space area with world-space distances and no transform Jacobian, so
  a light under an instance scale `s` reports a density `s²` times the true one
  (measured: `s=2` → 4×, `s=4` → 16×). Sharing it makes the two MIS weights sum
  to 1 for the *wrong* integral; a diffuse surface reached only by NEE has no
  competing technique to rebalance against and is biased by `1/s²` outright.
  This is pre-existing sampler behavior, reproduced deliberately so the hit side
  cannot drift from it, and out of scope here — fixing it means changing the
  sampler, with its own equivalence test against a *scaled* emissive-surface
  oracle. Unscaled and rigidly-transformed lights are exact.
- **One pick-CDF entry per light.** `appendLight` is unchanged; rect/ring
  additionally emit a proxy AABB tagged with the existing `lightIndex`. The NEE
  path is untouched. Double-counting is therefore structurally impossible rather
  than something tests must catch.
- Once the proxy BLAS exists, *every* renderer's rays can hit it. Renderers that
  do not deposit proxy hits (Interactive, Debug, Fast) must **pass through**
  safely. Only Quality deposits.
- **A proxy is opaque to continuation rays but invisible to shadow rays**, and
  that asymmetry is deliberate rather than an oversight. It reproduces what an
  authored emissive surface does *in the cases that matter*, without the
  self-occlusion an emissive surface actually suffers (the ~15% loss
  `GEOMETRY_LIGHT_SHADOW_EPSILON` exists to patch). The two ray classes are
  answering different questions: a shadow ray asks "is the path from this point
  to the light I already picked unobstructed", where the light's own surface is
  never an obstruction; a continuation ray asks "what does this direction see",
  where the light is the answer and the path terminates. Verified against the
  emissive-surface oracle rather than argued: `TestVisibleAreaLight` renders both
  with the emitter between the camera and the lit floor and requires agreement
  (measured relErr 0.0014).
- **`rectFrame` derives the world normal from the transformed EDGES**, as
  `cross(M·e1, M·e2)`, not by forward-transforming the object normal. The two
  differ by `det(M)`, so under a **mirroring** instance transform (negative
  determinant) the naive form points into the opposite hemisphere and the side
  predicate calls the front face the back one. Because shadow rays never test
  proxies, nothing downstream would catch it. Regression-tested over randomized
  affine transforms with mirroring cases explicitly counted.
- **`ringWorldAxis` normalizes AFTER transforming.** cosθ is a dot against a unit
  direction, so a non-unit axis rescales it and silently shifts the cone
  thresholds under any scaled instance.
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
- `visible` becomes meaningful for area lights. It is honored on `quad` and
  `ring`, and the device does **not** yet advertise
  `khr_light_primary_visibility` — but the reason is narrower than it first
  appears, and worth stating precisely.

  The extension moved between SDK versions: 0.15 had `khr_area_lights` (bundling
  `visible` with `angularDiameter`, `radius`, `radiance`); the 0.16 this device
  builds against dissolves that into the base light extensions and adds
  `khr_light_primary_visibility`, declaring `visible` on all six subtypes.

  Per the extension, `visible` is "whether the light can be directly seen", and
  is **only meaningful for area lights**. So leaving it inert on a light with no
  extent is conformant, not a silent lie — there is nothing to see, and the
  parameter is defined to have no other effect. That covers `spot` and
  `directional`, which are true delta lights in this device.

  It does **not** cover `point`. VisRTX maps `point` to `LightType::SPHERE`
  whenever `radius > 0`, and `radius` **defaults to 1** — so an ANARI `point`
  light is an area light with real extent by default, sampled by
  `sampleSphereLight`. That is a genuine gap in this work's coverage, not a
  spec-sanctioned omission, and it is what blocks advertising the extension.

  Sphere lights need the same treatment (a proxy, an analytic ray/sphere solver
  reusing `intersectPrimitives.h`, and a hit-side density mirroring
  `sampleSphereLight`). The extension goes in with that change.
- As ADR 0005 put it for Geometry Lights: "Every renderer that deposits path-hit
  emission must generalize its environment-only MIS weighting, or next-event
  estimation plus the hit deposit double-counts the emitter." Light Proxies are
  the third such type, and that sentence applies verbatim.
