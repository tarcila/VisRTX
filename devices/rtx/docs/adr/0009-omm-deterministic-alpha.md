# OMM-based deterministic alpha acceleration (no stochastic transparency)

Alpha-cutout/blend vegetation is slow in the rtx device: shadow/AO rays run a
per-hit any-hit → direct-callable chain, and primary rays re-trace through
every non-opaque hit in the raygen Transparency Loop. We accelerate this with
OptiX Opacity Micromaps — conservatively baked, 4-state, on all CUDA-material
surfaces (2-state-like for mask, 4-state for blend) — rather than stochastic
transparency, because Interactive must stay deterministic and produce
pixel-identical images on every path that evaluates alpha exactly (the
primary Transparency Loop and the shadow/AO any-hit chain). Unknown Opacity
States fall back to today's exact paths, so on those paths OMM is purely an
accelerator, never a semantics change. Paths that never evaluated alpha
diverge by design — see the determinism-scope consequence below.

## Considered options

- **Stochastic transparency** (Bernoulli per hit on alpha): best raw perf for
  blend, unbiased under accumulation — rejected: introduces noise, breaks the
  Interactive determinism requirement.
- **True 2-state OMM for mask**: cutoff-straddling microtriangles forced to
  opaque/transparent — rejected: silhouettes would depend on subdivision
  level and bake-vs-shading filtering mismatch. Conservative 4-state costs
  ~nothing (unknowns only on edge microtris) and stays pixel-exact.
- **Importer-only fix** (reclassify blend→mask in TSD): kept, but as an
  opt-in generic pass — it rewrites author intent, so the device must be fast
  on honest blends without it.

## Consequences

- **Material commits now invalidate acceleration structures.** Editing an
  opacity-relevant material parameter (alphaCutoff, opacity/baseColor
  samplers, alphaMode, transmission) rebakes OMMs and rebuilds the owning
  groups' triangle GASes (per-geometry-kind gating on surface input stamps)
  — never a full-world rebuild. A refit via
  `ALLOW_OPACITY_MICROMAP_UPDATE` is a possible future optimization.
- **Cut planes disable the fast path**: cut-plane culling lives in any-hit,
  so active cut planes force `ENFORCE_ANYHIT` per ray. Clipped scenes revert
  to today's perf by design. (No per-instance OMM disable is needed while
  states are transparent-only: TRANSPARENT microtris are invisible with or
  without clipping.)
- Bake evaluates the real GPU sampler objects (CUDA kernel, min/max over
  microtri footprint + filter margin) — conservative by construction, no CPU
  reimplementation of sampler semantics to drift.
- **The pixel-identical guarantee is scoped to alpha-evaluating paths.**
  Once provably transparent microtriangles stop being hit at all, three
  behaviors change on paths that never evaluated alpha: (1) rays traced with
  `DISABLE_ANYHIT` (Interactive/Debug ambient-bounce rays) previously treated
  cutout holes as opaque and now pass through them — more correct, but
  different; (2) pick/ID/depth/normal channels written from a fully
  transparent first hit now report the surface visible through the hole
  instead of the invisible one; (3) removed alpha=0 Transparency-Loop
  iterations no longer consume RNG draws, so accumulation noise under blend
  is a different — equally unbiased — realization. TestOpacityMicromap pins
  bitwise equality inside the guaranteed domain and keeps `ambientRadiance`
  at 0 to stay out of the divergent one.
- MDL materials are out of scope for v1 (arbitrary compiled opacity
  expressions); they keep the existing path. The separate MDL
  `isFullyOpaque` gap remains worth fixing independently.
- `OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT` is set per-Surface for Fully Opaque
  materials, and `REQUIRE_SINGLE_ANYHIT_CALL` on blend surfaces — the latter
  also fixes a latent double-attenuation bug (shadow any-hit multiplies
  transmittance per invocation, which OptiX may repeat per primitive).
- **v1 emits only TRANSPARENT and UNKNOWN_OPAQUE states** (no hard OPAQUE):
  every surviving hit runs the existing any-hit/loop paths, so the micromap
  is purely subtractive. Hard-OPAQUE states are a measured follow-up; the
  shadow-ray machinery for them already exists — shadow traces keep
  closest-hit enabled and commit blocked hits exactly, because a hit
  accepted *without* any-hit (DISABLE_ANYHIT geometry, future OMM-opaque)
  used to leave the payload reading "unoccluded" (mesh emitters stopped
  self-occluding — caught by TestEmissiveCylinderConeLight).
