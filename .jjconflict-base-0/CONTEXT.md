# CONTEXT — Ubiquitous Language

Glossary of domain terms for VisRTX. Terms here are canonical: code, docs, and
discussion use them exactly as defined.

## Alpha / opacity acceleration

- **Opacity Function** — the material-defined mapping from a surface point to
  alpha: baseColor.alpha × opacity input, post-processed by the material's
  alpha mode (opaque / blend / mask+cutoff). The single source of truth for
  what "transparent" means at a point; every acceleration structure derives
  from it, never redefines it.
- **Opacity State** — classification of a region of a triangle under the
  Opacity Function: *transparent* (alpha identically 0), *opaque* (alpha
  identically 1), or *unknown* (varies / partial). Unknown regions must be
  resolved by exact per-hit evaluation.
- **OMM (Opacity Micromap)** — OptiX acceleration structure attachment that
  stores per-microtriangle Opacity States so traversal can skip transparent
  regions and accept opaque ones without invoking any-hit / re-trace.
- **OMM Bake** — evaluation of the Opacity Function over each triangle's
  domain at a fixed subdivision level, producing Opacity States. A bake is
  *conservative* when a microtriangle is marked transparent/opaque only if the
  Opacity Function provably holds over its whole footprint; otherwise it is
  marked unknown.
- **Fully Opaque material** — material whose Opacity Function is constant 1
  everywhere (`isFullyOpaque`); eligible to skip any-hit entirely.
- **Transparency Loop** — the raygen-side loop that re-traces primary rays
  through non-opaque hits, compositing alpha deterministically. The exact
  fallback path for unknown Opacity States.
- **Alpha Classification** (TSD/importer track) — assigning an asset's
  intended alpha mode (opaque / mask / blend) at import time, correcting
  exporter mislabeling (e.g. foliage marked blend that is really mask).
