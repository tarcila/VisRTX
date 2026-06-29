# tsdFlow: Per-Viewport Lights — Design

**Date:** 2026-06-29
**Status:** Approved (pending user spec review)
**Phase:** Intermediary step 3 of 3 (per-viewport menus → lights)

## Problem

tsdFlow has no real lighting model. The render bridge currently injects a single hard-coded directional light (added as a stopgap so the introspected `ambientRadiance=0` default did not leave scenes black). There is no way to author lights from the graph, and no per-viewport control.

Two distinct needs, deliberately handled by two mechanisms:

1. **Plain lights** are ordinary scene objects — the *same* set for every viewport — that participate in the existing per-display `viewportMask` filtering (so a light can be limited to a subset of viewports exactly like a surface or volume).
2. **A headlight** is the one genuinely per-viewport light: a camera-attached directional light, owned by each viewport, that exists so a viewport is never unintentionally black.

## Decisions (settled in brainstorm)

| Topic | Decision |
|-------|----------|
| Plain-light authoring | A single **`DisplayLight`** graph node (subtype + light params + `viewportMask`), mirroring `DisplaySurface`/`DisplayVolume`. No separate "light source" node. |
| Default light | **Replace** the hard-coded bridge light with a `DisplayLight` node seeded in the demo graph (mask = all viewports). The default light becomes a first-class, editable, mask-able graph object. |
| Headlight default | **Auto**: enabled for a viewport iff **no `DisplayLight` is masked to that viewport**; manual `On`/`Off` override the auto decision. |
| Headlight type | **Directional**, aimed along the camera view direction; **intensity (irradiance) and color editable** per viewport. |
| Light subtypes (v1) | Directional first for both paths; point/HDRI reachable later through the same `subtype` seam. No light gizmo in v1. |

## Architecture

Two independent units. Plain lights ride the existing graph→bridge→masked-layer path; the headlight is bridge-owned per-viewport state injected through the one per-index hook (`setExternalInstances`).

```
Plain light:  DisplayLight node → Renderable(Kind::Light) → bridge buildLight()
              → tsd::Light in the display layer → masked into viewport indices
              (normal layersForViewport path; same as surfaces/volumes)

Headlight:    GraphViewport camera + Lights menu → bridge setViewportHeadlight(i, …)
              → per-viewport anari::Light/Group/Instance on index i via
              setExternalInstances (auto-resolved against that viewport's lights)
```

## Component A — Plain lights through the graph

### A1. `Renderable` gains a light kind
`tsd/src/tsd/graph/Renderable.hpp`: add `Kind::Light` to the enum. For a light, `primSubtype` is the ANARI light subtype (e.g. `directional`) and `appearance` carries the light parameters (`color`, `irradiance`, `direction`, …). `prim` is unused for lights.

### A2. `DisplayLight` node
New `tsd/src/tsd/graph_nodes/DisplayLight.{hpp,cpp}` + `registerDisplayLight(reg)` wired into `registerBuiltinNodes` (`BuiltinNodes.cpp`). Mirrors `DisplaySurface` but is a pure producer — a light has no upstream data:

- **Ports:** no inputs; one output `out : renderable`. Category `sink`.
- **Params:** `viewportMask` (seeded `kDefaultViewportMask`), `subtype` (`directional`), and the light params surfaced for editing — `color` (float3), `irradiance` (float), `direction` (float2 az/el, matching the existing default light convention). Not an `ITransformableNode` in v1 (directional lights have no position).
- **`evaluate`:** builds a `Renderable{ kind=Light, primSubtype=subtype, appearance={color,irradiance,direction,…} }` and sets it on `out`.

### A3. Bridge `buildLight()`
`GraphRenderBridge`: a third branch alongside `buildSurface`/`buildVolume`. `rebuildLayer` dispatches on `r->kind`; `buildLight(layer, r, name)` does `createObject<Light>(r.primSubtype)`, `applyParams(*light, r.appearance)`, `setName(name)`, and `insertChildObjectNode(layer->root(), light)`. The light therefore lives in the display's layer and is routed by the existing `viewportMask`/`layersForViewport` filtering — shared across viewports, mask-limited per display. Unknown subtype → skip with a `logWarning`, consistent with the other build paths.

### A4. Remove the stopgap default light
Delete `addDefaultLight()`, the `m_lightsLayer` member, and `indexLayers()` (revert the three call sites to `layersForViewport(i)`). Plain lights now ride the normal included-layers path, so the special "gather all lights" workaround is unnecessary — and avoids the `alwaysGatherAllLights` overrun documented in the bridge. Seed the demo graph (`DemoGraph.cpp`) with one `DisplayLight` (directional, mask = all viewports) so default scenes stay lit.

## Component B — Per-viewport headlight

### B1. Bridge-owned per-viewport headlight handles
`GraphRenderBridge` holds, per viewport, a headlight record: an `anari::Light` (directional), an `anari::Group` containing it, and an `anari::Instance` (identity transform) referencing the group — all created on that viewport's device. Created lazily and **recreated in `setViewportDevice`** (device switch) so handles always match `m_viewportDevices[i]`. Released in the dtor and on recreate.

### B2. Bridge API
```cpp
struct HeadlightState {
  enum class Mode { Auto, On, Off } mode{Mode::Auto};
  tsd::math::float3 direction{0,0,-1}; // camera forward (light travel dir)
  tsd::math::float3 color{1,1,1};
  float irradiance{1.f};
};
void setViewportHeadlight(int viewport, const HeadlightState &s);
```
Each call: update the light's `direction`/`color`/`irradiance` params and commit the light (cheap — no world rebuild, the instance already references it). Then resolve effective on/off:
- `Auto` → on iff **no enabled `DisplayLight` is masked to this viewport** (the bridge already tracks displays + masks + kind);
- `On`/`Off` → forced.
On an on→off or off→on transition, call `m_indices[viewport]->setExternalInstances(&inst, 1)` / `setExternalInstances(nullptr, 0)`. `updateWorld` concatenates external instances with the layer instances, so the headlight coexists with masked scene objects and survives the per-frame `setIncludedLayers` rebuild.

### B3. `GraphViewport` Lights menu + per-frame feed
- New `ui_menu_Lights()` added to the menu bar (`Device | Renderer | Lights`). Controls: headlight **Auto/On/Off** radio, **color** edit, **intensity** slider. State stored on the viewport.
- Each frame (after the camera/manipulator update), `GraphViewport` computes the view forward `normalize(at - eye)` and calls `m_bridge->setViewportHeadlight(m_viewportIndex, {mode, forward, color, irradiance})`. This also keeps the headlight aimed as the camera orbits.
- On device switch, the headlight handles are rebuilt by the bridge (B1); the viewport's stored UI state is re-pushed on the next frame.

## Data flow & boundaries

- **Plain lights:** entirely graph + bridge; no viewport involvement. Identical lifecycle to surfaces/volumes (rebuild on output-version change, cleared by `clearLayerObjects`).
- **Headlight:** entirely viewport (UI + camera) + bridge (handles + injection); never touches the graph or render scene. The bridge is the only component holding both the per-viewport device and the headlight handles, so it owns them.
- No change to the evaluator, graph engine, or persistence.

## Error handling

| Case | Strategy |
|------|----------|
| Unknown light subtype in `buildLight` | Skip the light + `logWarning`; rest of the layer builds (mirrors surface/volume paths). |
| Headlight on a null/old device | Guard every headlight ANARI call on a valid device; recreate handles in `setViewportDevice`. |
| `DisplayLight` with no params | Light created with introspected ANARI defaults; renders with defaults. |
| Auto headlight + lights toggled at runtime | Re-resolved every frame from current masks; transitions flip the external instance. |

## Testing

### Bridge unit tests (`tsd/tests/`, helide/visrtx as available)
- **Plain light masking:** a `DisplayLight` masked to viewport 0 only contributes to index 0's world; viewport 1's world is unaffected (mirrors the existing `[bridge-mask]` surface test).
- **Headlight auto:** with no `DisplayLight` masked to viewport i, `setViewportHeadlight(i, Auto)` makes `world(i)` non-null and adds exactly one external instance; with a `DisplayLight` masked in, Auto resolves off (no external instance). `On`/`Off` force regardless.
- **Headlight update is cheap:** repeated `setViewportHeadlight` with changing direction does not change `renderSceneObjectCount()` and leaves `world(i)` valid (the render scene is untouched; only an external ANARI light is updated).
- **Device switch:** after `setViewportDevice`, a subsequent `setViewportHeadlight` still yields a valid `world(i)` (handles rebuilt on the new device).

### Manual smoke (the app)
- A fresh scene is lit by the seeded default `DisplayLight`; deleting it falls back to each viewport's Auto headlight (scene stays lit, not black).
- Masking the `DisplayLight` to a subset of viewports lights only those; the others light via their Auto headlight.
- Lights menu: switching Auto/On/Off, editing color/intensity updates the image live; orbiting re-aims the headlight.
- Device switch on a viewport preserves its headlight and renders on the new device.

## Out of scope

- Point/spot/HDRI light authoring UI (the `subtype` seam supports them; only directional is wired in v1).
- A light manipulator/gizmo (directional lights have no position).
- Per-viewport *plain* lights (plain lights are shared + mask-filtered by design; only the headlight is per-viewport).
- Persisting headlight state across sessions (Phase 5 persistence).
- Shadow/AO renderer settings (already covered by the per-viewport Renderer menu).
