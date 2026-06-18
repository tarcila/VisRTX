# tsdFlow Phase 3 — Render Bridge & Viewport Masks Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan
**Builds on:** Phase 1 (headless engine) + Phase 2 (async evaluator), both committed & bookmarked.

## Summary

Introduce a `GraphRenderBridge` that turns evaluated `renderable` outputs of the
`tsd_graph` engine into rendered ANARI worlds — one per viewport — selected by a
per-display-object **viewport mask**. The bridge reuses TSD's existing
`RenderIndex` for all TSD→ANARI translation; the mask maps onto `RenderIndex`
layer filtering. Validated headlessly with the **VisRTX** reference device
(confirmed loading + rendering in this GPU sandbox: RTX 5880 Ada, `visrtx` is the
default `ANARI_LIBRARY` and is on the runtime path).

## Architecture decision (refines Phase 1 Approach C)

Phase 1 Approach C envisioned the bridge feeding each viewport's `RenderIndex`
with bridge-built `anari::Instance`s and explicitly avoided layers/scene. Phase 3
**revises this**: the bridge maintains an internal, fully-derived `tsd::Scene`
(the "render-scene") and lets `RenderIndex` own TSD→ANARI exposure. Rationale:

- `RenderIndex` already translates `tsd::Surface` (geometry+material) and
  `tsd::Volume` (spatial field + transfer function) into ANARI via
  `AnariHandleCache`. Reusing it makes supporting **both surfaces and volumes**
  cheap and correct, instead of hand-writing ANARI translation in the bridge.
- The render-scene is **not** the user-editable scene. It is bridge-internal,
  single-writer (only the bridge mutates it, from graph output), and rebuilt on
  change. So Approach C's "two mutation sources on one editable scene" concern
  does not apply.
- `RenderIndexAllLayers::setIncludedLayers(const std::vector<const Layer*>&)`
  already exists — the viewport mask maps directly onto it.

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| Renderable kinds | **Both surface and volume** (a tagged core-only descriptor). |
| Translation | Bridge maps `renderable` → `tsd::Surface`/`tsd::Volume` in a derived render-scene; `RenderIndex` does TSD→ANARI. No hand-written ANARI object code in the bridge. |
| Layer model | **One `Layer` per display node** in the render-scene. The display's objects live in that one layer. |
| Mask mechanism | Per viewport `i`: a `RenderIndexAllLayers` with `setIncludedLayers({ layer(d) : d.enabled && (mask(d) >> i) & 1 })`. One data copy; multiple viewports include the same layer. |
| Mask ownership | Bridge-side: `setDisplay(NodeId, uint64_t viewportMask, bool enabled)`. `tsd_graph` stays viewport-free. |
| `renderable` type | Core-only `PortType` + `Renderable` descriptor in `tsd_graph` (kind + subtype token + a small param/array set). |
| Bridge location | New files under `tsd/src/tsd/rendering/bridge/`; `tsd_rendering` newly links `tsd_graph`. |
| Update model | `bridge.update()`: pull each enabled display node, rebuild its layer's objects from the `renderable`, recompute each viewport's included layers, refresh each viewport's `RenderIndex`. Replace-per-display (not incremental within a display) — matches available APIs. |
| Test strategy | Pure mask→included-layers unit tests (no ANARI) + `visrtx` headless render smokes asserting on the **objectId AOV** (non-zero where geometry/volume is hit, zero on background) — robust vs lighting/exposure. Surface and volume smokes. |
| Reference device | `visrtx` (GPU/OptiX). First render incurs OptiX module compilation (seconds) — render tests use a generous ctest timeout. |
| Scope | Headless only. No interactive ImGui viewport, no camera manipulator, no UI — those are Phase 4. A "viewport" here = an offscreen `RenderIndex` + ANARI frame. |

## Components

### `renderable` descriptor (`tsd/src/tsd/graph/Renderable.hpp`, core-only)

A backend-agnostic description the bridge can turn into TSD scene objects:

```cpp
namespace tsd::graph {

// name -> scalar value and name -> array, for driving tsd Object::setParameter.
struct RenderableParams
{
  std::vector<std::pair<tsd::core::Token, tsd::core::Any>> scalars;
  std::vector<std::pair<tsd::core::Token, tsd::core::AnyArray>> arrays;
};

struct Renderable
{
  enum class Kind { Surface, Volume };
  Kind kind{Kind::Surface};
  // Surface: geometry subtype (e.g. "sphere","triangle"); Volume: spatial-field
  // subtype (e.g. "structuredRegular").
  tsd::core::Token primSubtype;
  RenderableParams prim;        // geometry params, or spatial-field params
  RenderableParams appearance;  // material params (surface) or TF params (volume)
};

} // namespace tsd::graph
```

`renderable` registers as a `PortType`; a `Renderable` travels as a `Value`
payload (`std::shared_ptr<Renderable>`), host residency. Phase 3 ships at least
one minimal display/emitter node per kind (or test-only emitters) producing a
`Renderable`; the full catalog is Phase 4.

### `GraphRenderBridge` (`tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp/.cpp`)

```cpp
class GraphRenderBridge
{
 public:
  GraphRenderBridge(Graph &graph, Evaluator &eval,
      tsd::core::Token deviceName, anari::Device device, int numViewports);
  ~GraphRenderBridge();

  // Register/replace/remove a display node's viewport mask + enabled flag.
  void setDisplay(NodeId node, uint64_t viewportMask, bool enabled);
  void removeDisplay(NodeId node);

  // Pure: which display layers are included by viewport i (testable w/o ANARI).
  std::vector<const Layer *> layersForViewport(int i) const;

  // Pull enabled displays, rebuild their layers, refresh each viewport index.
  void update();

  anari::World world(int viewport) const; // for rendering / frame setup

 private:
  // Per display: NodeId -> { mask, enabled, Layer* in m_renderScene, lastVersion }
  // Per viewport: RenderIndexAllLayers* (over m_renderScene)
  Scene m_renderScene;
  ...
};
```

`update()`:
1. For each registered display `d` that is `enabled`: `eval.pull(d.node)` (blocking
   `pull` is fine here; Phase 2's async is available but the bridge uses the
   synchronous path in Phase 3), read its `renderable` output, and rebuild `d`'s
   layer in `m_renderScene` (clear + create `tsd::Surface`/`tsd::Volume` from the
   descriptor). Skip rebuild if the producing node's output version is unchanged.
2. For each viewport `i`: `index[i].setIncludedLayers(layersForViewport(i))` then
   refresh (`populate()` / delegate update) so its ANARI world reflects the
   current included layers.

### Renderable → TSD object translation

- **Surface:** `createObject<Geometry>(primSubtype)`, apply `prim` params;
  `createObject<Material>(matte)`, apply `appearance` params;
  `createSurface(name, geom, mat)`; insert into the display's layer.
- **Volume:** `createObject<SpatialField>(primSubtype)`, apply `prim` params;
  `createObject<Volume>(transferFunction1D)`, set `value=field` and apply
  `appearance` params (`color`, `opacity`, `valueRange`); insert into the layer.

Array payloads in `RenderableParams.arrays` become `Scene` arrays
(`scene.createArray(type, count)` + fill) bound to the object parameters.

## Error handling

| Case | Behavior |
|------|----------|
| Display node pull fails / in Error | Its layer is left empty (or cleared); other displays/viewports unaffected; bridge logs and continues. |
| Renderable of unknown kind/subtype | That display's layer is left empty; logged; no crash. |
| Mask references viewport index >= numViewports | Bits beyond `numViewports` are ignored. |
| Device/world creation fails | Constructor reports failure; bridge unusable (surfaced to caller). |

## Testing (headless)

1. **Mask → included layers (no ANARI)** (`test_bridge_Mask.cpp`): register displays
   with masks/enabled; assert `layersForViewport(i)` returns exactly the expected
   layers; toggling enabled/mask updates membership; `removeDisplay` drops it. (May
   need a null/helide device to construct the bridge; if so, this is a `helide`
   test too — but the assertions are on membership, not pixels.)
2. **visrtx surface render smoke** (`test_bridge_RenderSurface.cpp`): a graph whose
   display node emits a sphere surface, masked to viewport 0 only (of 2). After
   `update()`, render both viewports with `visrtx` requesting `channel.objectId`;
   assert viewport 0 has some non-zero objectId pixels (geometry hit) and viewport
   1 is all-zero (background).
3. **visrtx volume render smoke** (`test_bridge_RenderVolume.cpp`): a display node
   emits a small `structuredRegular` field + transfer function; render the masked
   viewport and assert it produced a hit (non-background) — proves the volume path
   through `RenderIndex`. (For volumes, assert on a non-background color/alpha or a
   non-empty depth, since objectId may be surface-specific; the implementer picks
   the channel the device populates for volumes.)
4. **Multi-viewport sharing** (`test_bridge_MultiViewport.cpp`): one display masked
   to viewports 0 and 1 → both render it (objectId non-zero in both); a second
   display masked to only viewport 1 → present in 1, absent in 0.

The reference device is loaded via `anari::loadLibrary("visrtx", ...)` (available
in this environment). A render test still guards the device handle: if device
creation returns null, it fails with a clear message rather than dereferencing
null. A camera framing the content and a default light are set up by the test
helper so the content is actually in-frame.

## Out of scope for Phase 3

- Interactive ImGui viewport, camera manipulator, picking, AOVs, ImagePipeline UI
  (Phase 4).
- The real node catalog (importers, processors, `DisplayVolume`/`DisplaySurface`
  with rich params) and CUDA residency at the device boundary (Phase 4).
- Async bridge updates / per-frame scheduling (Phase 3 uses synchronous
  `eval.pull`; async hookup is a later refinement).
- Lua, persistence (Phase 5).

## As-built deviations (post-implementation, 2026-06-17)

- **Shared `RenderIndexAllLayers` fix.** `setIncludedLayers({})` previously fell back
  to "all active layers" (so an unmasked/empty viewport wrongly showed everything).
  Fixed: `setIncludedLayers` always sets `m_customIncludedLayers=true`; the
  "sync-all" branch is guarded by `!m_customIncludedLayers`; `updateWorld` skips
  cached instances for non-included layers. Verified safe for all existing callers
  (tsdRender/tsdOffline/scivisStudio never call it; MultiDeviceViewport passes a
  non-empty set). Full 43-test pre-existing suite still green.
- **VisRTX writes objectId for volumes too** (the spec assumed objectId was
  surface-only). The multi-viewport test therefore differentiates by placing the
  surface (sphere) at x=2.5, outside the volume's bbox, and asserting
  `c1.objectId > c0.objectId` (additive, non-overlapping coverage) — not
  `c0.objectId == 0`.
- **Renderer needs `ambientRadiance=1`** for matte surfaces to be non-black in
  `channel.color`; objectId background sentinel is `~0u` (not 0).
- **Object reclamation on rebuild.** `Layer::clear()` clears only the layer tree, not
  the pooled objects — so rebuilding a display's layer leaked Geometry/Material/
  arrays. Fixed: the bridge removes the display's object-nodes via
  `Scene::removeNode(child, /*deleteReferencedObjects=*/true)` + `removeUnusedObjects()`
  before repopulating. Regression test `test_bridge_Rebuild.cpp` asserts the
  render-scene object count is bounded across 20 version-bumping rebuilds.
- **`update()`** no longer calls an explicit `populate()` after `setIncludedLayers`
  (which already rebuilds the world); `removeDisplay` refreshes all viewport indices
  immediately so no index retains a freed `Layer*`.

## Phasing note

Phase 3 of the 5-phase delivery. Depends on the committed Phase 1 + Phase 2
engine. First ANARI dependency in the tsdFlow line. Phase 4 (catalog + UI + CUDA)
builds the interactive viewport on top of this bridge.
