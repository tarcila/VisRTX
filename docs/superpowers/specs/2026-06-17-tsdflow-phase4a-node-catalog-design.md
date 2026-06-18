# tsdFlow Phase 4a — Headless Node Catalog Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan
**Builds on:** Phases 1–3 (tsd_graph engine, async evaluator, GraphRenderBridge), all committed & bookmarked.

## Phase 4 decomposition (context)

Phase 4 ("catalog + interactive UI + CUDA") is multiple independent subsystems and
is delivered as sub-phases, each independently testable/mergeable:

- **4a — Headless node catalog** (this doc): real source/processor/display nodes,
  rendered headlessly via the bridge. TDD + VisRTX. Depends on Phases 1–3 only.
- **4b — CUDA TransferRegistry**: real host↔CUDA transfers, NanoVDB native-CUDA.
  Engine-side, headless (GPU). Independent.
- **4c — Interactive app shell + viewports**: the `tsdFlow` ImGui/SDL3 app, N
  viewports on the bridge, device picker, camera manipulator.
- **4d — Node editor canvas + GraphInspector + transfer-function editor**.
- **4e — Undo/redo + copy/paste** (coordinates with Phase 5 serialization).

4a is first: it makes the engine do real scivis work, is fully TDD-testable, and
de-risks the catalog before any UI consumes it.

## Summary

A concrete node catalog — a procedural volume source, scalar-range + transfer-
function processors, a bounding-box surface op, and volume/surface display sinks —
that drives the "load → process → display" pipeline end-to-end, rendered headlessly
through the Phase 3 `GraphRenderBridge` with VisRTX. Every node is core-only host-
data manipulation (no `tsd_io`/`tsd_scene` in 4a); file importers are a later slice.

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| Scope | Procedural vertical slice (no file I/O): 6 nodes proving source→process→display + bridge end-to-end with deterministic tests. File importers deferred. |
| Catalog location | New library `tsd_graph_nodes`, depending on `tsd_graph` only (4a). Home for future importer/algorithm nodes (which will add `tsd_io`/`tsd_scene` deps). Keeps `tsd_graph` engine pure. |
| Registration | Nodes self-register via `TSD_GRAPH_REGISTER_NODE`, plus an explicit `registerBuiltinNodes()` the app/tests call (avoids the static-lib whole-archive stripping caveat from Phase 1). |
| Data currency | Core-only descriptors carried as `Value` payloads: `field`, `range` (float2), `transferFunction`, `surface`; reuse Phase 3 `Renderable` as the display currency the bridge consumes. |
| Rendering | No new rendering code — display nodes emit `renderable`; the Phase 3 bridge turns it into ANARI. |
| Residency | Host-only in 4a (CUDA is 4b). |

## Architecture

### Library

`tsd/src/tsd/graph_nodes/` → static lib `tsd_graph_nodes`, `project_link_libraries(PUBLIC tsd_graph)`.
Added to `tsd/src/tsd/CMakeLists.txt`. Tests link it; the future app links it.

### Data-currency descriptors (core-only)

Defined in `tsd_graph_nodes` (or a shared header), carried as `std::shared_ptr<T>`
`Value` payloads, host residency:

```cpp
// PortType "field"
struct Field {
  tsd::core::Token subtype;                 // "structuredRegular"
  tsd::core::math::uint3 dims;
  tsd::core::math::float3 origin, spacing;
  tsd::core::AnyArray data;                 // dims.x*dims.y*dims.z scalars
};

// PortType "range" -> a float2 carried directly in Any (no struct)

// PortType "transferFunction"
struct TransferFunctionData {
  tsd::core::AnyArray colormap;             // float4 (RGBA), `samples` entries
  tsd::core::math::float2 valueRange;
};

// PortType "surface"
struct SurfaceData {
  tsd::core::Token geomSubtype;             // "triangle"
  tsd::graph::RenderableParams prim;        // geometry params (vertex.position, index)
  tsd::graph::RenderableParams appearance;  // material params (color)
};
```

`Renderable` (Phase 3) remains the display currency; `DisplayVolume`/`DisplaySurface`
produce it.

### Registration

```cpp
namespace tsd::graph_nodes {
void registerBuiltinNodes(); // idempotent; registers all six into GlobalNodeRegistry
}
```
Each node also uses `TSD_GRAPH_REGISTER_NODE` for static-init registration; the
explicit call guarantees availability when the lib is statically linked.

## Node set

Ports: `name:type`. All inputs accept host residency.

| Node | Category | Inputs | Output | Params | Behavior |
|------|----------|--------|--------|--------|----------|
| `GenerateNoiseVolume` | source | — | `out:field` | `dims:uint3` (32³), `seed:int` | Seeded value-noise into a `float` AnyArray; `field` subtype `structuredRegular`, origin `{-1,-1,-1}`, spacing `2/dims`. Deterministic → cacheable. |
| `ScalarRange` | processor | `field` | `out:range` | — | min/max over `field.data` → `float2`; sets `contentTag`. |
| `TransferFunction` | processor | `range` (req) | `out:transferFunction` | `preset:token` (`coolToWarm`/`grayscale`), `samples:int` (256) | Builds float4 colormap from preset; `valueRange` = input range. |
| `DisplayVolume` | sink | `field` + `transferFunction` | `out:renderable` | — | `Renderable{Volume}` (structuredRegular + transferFunction1D) from field + colormap + valueRange. |
| `BoundingBox` | processor | `field` | `out:surface` | `color:float3` | Triangle box surface spanning the field bounds. |
| `DisplaySurface` | sink | `surface` | `out:renderable` | — | `Renderable{Surface}` from the surface descriptor. |

### End-to-end demo graphs

```
Volume:  GenerateNoiseVolume ──┬───────────────────────────────► DisplayVolume
                               └─► ScalarRange ─► TransferFunction ─┘
Surface: GenerateNoiseVolume ──► BoundingBox ─► DisplaySurface
```

The volume path exercises fan-out (source → `ScalarRange` and → `DisplayVolume`) and
a multi-input sink (`DisplayVolume`). Both render via the Phase 3 bridge.

None of the nodes wrap `tsd_io`/`tsd_scene`: each is a few lines over `AnyArray`
(noise fill, min/max, colormap gradient, 8-corner box). `GenerateNoiseVolume` does
NOT reuse the scene-coupled `tsd_io::generate_noiseVolume`.

## Error handling (fail loud at node boundaries)

| Case | Behavior |
|------|----------|
| Required input unconnected/invalid | Node detects the invalid input `Value` in `evaluate()`, sets `Error` + message; evaluator short-circuits downstream; bridge renders that display's layer empty; other branches unaffected. |
| Bad params (`dims` component 0, `samples < 2`) | Node `Error` with a message — no silent empty buffer. |
| Unknown `preset` | `TransferFunction` falls back to `grayscale` + logs a warning. |
| Malformed field (`data` size ≠ ∏dims) | `ScalarRange`/`DisplayVolume` `Error` with a message. |

## Testing (headless + VisRTX)

1. **Per-node unit tests** (no ANARI): determinism + correct outputs for each of the
   six nodes; `registerBuiltinNodes()` makes all six resolvable in `GlobalNodeRegistry`.
2. **Wiring tests** (no ANARI): the demo volume graph pulls a correct `Renderable`;
   fan-out + multi-input resolve; `seed` change recomputes downstream; no-edit →
   version short-circuit.
3. **VisRTX render smoke (via the Phase 3 bridge)**: register `DisplayVolume` and
   `DisplaySurface` as displays, `update()`, render → volume yields non-background
   pixels; box surface yields objectId hits. Reuses Phase 3 render-count helpers.
4. **Full-suite gate** (all prior tests stay green).

## Out of scope for 4a

- File importers (ImportRaw/VTI/VTU) and `tsd_io`/`tsd_scene` deps — a later slice.
- CUDA residency (4b).
- Any UI: app shell, node editor, inspector, TF editor (4c/4d).
- Undo/redo, persistence (4e/Phase 5).
- Resample/Crop and a richer catalog (incremental after the slice lands).

## Phasing note

First slice of Phase 4. Depends only on the committed Phases 1–3. The interactive
app (4c/4d) builds on the catalog this delivers.
