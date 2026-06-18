# tsdFlow Phase 4c — Interactive App Shell & Viewports Design

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan
**Builds on:** Phases 1–3 + 4a (engine, async evaluator, GraphRenderBridge, headless node catalog), all committed & bookmarked.

## Summary

The first interactive `tsdFlow` ImGui/SDL3 app: it builds the Phase 4a demo graph
programmatically, drives it through the committed `GraphRenderBridge`, and shows the
result in two camera-controlled viewport windows (volume in one, bounding-box
surface in the other — showcasing the per-viewport mask). No node editor yet (that
is Phase 4d); the graph is fixed. A headless Catch2 smoke test, backed by a shared
demo-graph builder the app also uses, guards the render path in CI.

## Phase 4 ordering note

4c is taken before 4b (CUDA TransferRegistry): 4c renders the host-only 4a catalog,
which needs no CUDA residency, so it does not depend on 4b. 4b remains independent
and additive, slotting in later.

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| App | New `tsd/apps/interactive/tsdFlow/`, `class Application : public tsd::ui::imgui::Application`, under `TSD_BUILD_INTERACTIVE_APPS`. |
| Graph source | Built programmatically via a shared core-only builder (the 4a demo). No node editor (4d). |
| Viewport | New reusable `GraphViewport : tsd::ui::imgui::Window` (standalone — NOT `BaseViewport`, which is coupled to TSD *scene* cameras/renderers). It owns its own `Manipulator`, `ImagePipeline` (`AnariSceneRenderPass` + `CopyToSDLTexturePass`), and `anari::Camera`/`Renderer` on the bridge's device; handles mouse→manipulator input itself; renders `bridge.world(index)` via `AnariSceneRenderPass::setWorld` (which accepts an external world) and blits the SDL texture. |
| Viewports | Two (vp0=volume display, vp1=surface display) to showcase the viewport mask. |
| Device | One device chosen at startup (default `visrtx`, CLI/`TSD_ANARI_LIBRARIES` overridable). No in-app device switching (deferred). |
| Smoke test | Separate headless Catch2 test in `tsd/tests`, calling the SAME shared demo builder + bridge the app uses, rendering with VisRTX and asserting pixels. |
| Interactivity | Camera orbit/pan/zoom per viewport + a single "Regenerate" button (bump seed → markDirty → bridge.update()) to exercise the live re-render path. |
| Engine/bridge changes | None. 4c is app + one new window type + one shared builder. |

## Architecture

### Shared demo builder — `tsd/src/tsd/graph_nodes/DemoGraph.hpp/.cpp` (core-only)

```cpp
namespace tsd::graph_nodes {
struct DemoDisplays {
  tsd::graph::NodeId volumeDisplay;
  tsd::graph::NodeId surfaceDisplay;
};
// Builds the 4a demo graph into `g` using node types from `reg` (the caller has
// already registerBuiltinNodes'd it); returns the two display node ids. No ANARI,
// no GUI.
DemoDisplays buildVolumeSurfaceDemo(
    tsd::graph::Graph &g, tsd::graph::NodeRegistry &reg);
}
```
Wiring built: `GenerateNoiseVolume → ScalarRange → TransferFunction → DisplayVolume`
(source also fanned out to `DisplayVolume.field`) and `GenerateNoiseVolume →
BoundingBox → DisplaySurface`. Returns `{DisplayVolume id, DisplaySurface id}`. The
app and the headless test both call this — single source of the graph wiring.

### App — `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`

`class Application : public tsd::ui::imgui::Application` owns: a `tsd::graph::Graph`,
`tsd::graph::Evaluator`, a `tsd::graph::NodeRegistry` (`registerBuiltinNodes`), the
ANARI device (from `appContext()->anari.loadDevice(defaultLib)`), and a
`tsd::rendering::GraphRenderBridge`.

`setupWindows()`:
1. `registerBuiltinNodes(m_registry)`.
2. `auto d = buildVolumeSurfaceDemo(m_graph, m_registry);`
3. Load the device (default `visrtx`); if null → log fatal + exit (the bridge ctor
   rejects a null device).
4. Construct `GraphRenderBridge(m_graph, m_eval, Token(lib), device, /*numViewports=*/2)`.
5. `bridge.setDisplay(d.volumeDisplay, 0b01, true); bridge.setDisplay(d.surfaceDisplay, 0b10, true); bridge.update();`
6. Create two `GraphViewport`s (index 0 and 1), a `Log` window, and a small
   "Regenerate" control; return the window array.

### `GraphViewport` — `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp/.cpp`

A **standalone** reusable window (lives with the other viewports; 4d builds on it).
It does NOT inherit `BaseViewport` — BaseViewport's camera/renderer path is bound to
TSD *scene* objects (`CameraAppRef`/`RendererAppRef`, `updateCameraObject`), but the
bridge supplies a raw `anari::World` with no scene camera. So `GraphViewport` owns
its own ANARI camera/renderer/manipulator/pipeline directly.

```cpp
struct GraphViewport : public tsd::ui::imgui::Window
{
  GraphViewport(Application *app, tsd::rendering::GraphRenderBridge *bridge,
      int viewportIndex, anari::Device device, const char *name);
  ~GraphViewport() override; // release camera/renderer; pipeline owns its passes
  void buildUI() override;
 private:
  tsd::rendering::GraphRenderBridge *m_bridge;
  int m_viewportIndex;
  anari::Device m_device;
  anari::Camera m_camera{nullptr};       // perspective, on m_device
  anari::Renderer m_renderer{nullptr};   // "default", ambientRadiance=1
  tsd::rendering::Manipulator m_manip;   // owned; mouse-driven
  tsd::rendering::UpdateToken m_manipToken{0};
  tsd::rendering::ImagePipeline m_pipeline;
  tsd::rendering::AnariSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};
};
```

- **ctor:** create `m_camera = anari::newObject<Camera>(device,"perspective")` and
  `m_renderer = anari::newObject<Renderer>(device,"default")` (with
  `ambientRadiance=1`), commit both; configure `m_manip` to frame the demo bounds
  (center origin, distance ~3); build `m_pipeline` with
  `m_anariPass = emplace_back<AnariSceneRenderPass>(device)` (set camera + renderer)
  then `m_outputPass = emplace_back<CopyToSDLTexturePass>(app->sdlRenderer())`.
- **`buildUI()`** (standalone render loop, modeled on `Viewport::buildUI` minus the
  scene/RenderIndex parts):
  1. measure `ImGui::GetContentRegionAvail()`; on size change
     `m_pipeline.setDimensions(w,h)` + update camera aspect.
  2. handle mouse drag over the image → `m_manip.rotate/zoom/pan` (small input
     block, lifted from BaseViewport's input pattern).
  3. if `m_manip.hasChanged(m_manipToken)`:
     `updateCameraParametersPerspective(m_device, m_camera, m_manip)` then
     `anari::commitParameters(m_device, m_camera)`.
  4. `m_anariPass->setWorld(m_bridge->world(m_viewportIndex))` (re-fetched each frame
     so it reflects the latest `bridge.update()`).
  5. `m_pipeline.render()`.
  6. `ImGui::Image((ImTextureID)m_outputPass->getTexture(), avail, ImVec2(0,1), ImVec2(1,0))`.

Exact signatures (`AnariSceneRenderPass::setWorld/setCamera/setRenderer`,
`CopyToSDLTexturePass::getTexture`, `updateCameraParametersPerspective`,
`ImagePipeline::setDimensions/render`, `Manipulator`) are verified against the
headers in the plan.

## Smoke test — `tsd/tests/test_tsdflow_Smoke.cpp`

Catch2, VisRTX, `set_tests_properties(... TIMEOUT 300)`:
`registerBuiltinNodes(reg)` → `buildVolumeSurfaceDemo(g, reg)` → `Evaluator e(g)` →
`GraphRenderBridge bridge(g, e, Token("visrtx"), dev, 2)` →
`setDisplay(d.volumeDisplay, 0b01)`, `setDisplay(d.surfaceDisplay, 0b10)` →
`update()` → assert `world(0)` has non-background color pixels (volume) and
`world(1)` has objectId hits (box), reusing the `renderCounts(device, world)` helper
pattern from `tsd/tests/test_bridge_RenderVolume.cpp`. This guards the shared builder
+ bridge wiring the app depends on, with no SDL window.

## Error handling

| Case | Behavior |
|------|----------|
| Device load returns null | Log a fatal message and exit cleanly before constructing the bridge. |
| A display node in `Error` | Bridge renders that viewport's layer empty; the viewport shows background; `Log` surfaces warnings. No crash. |
| SDL/ImGui init failure | Handled by the `Application` base. |

## Manual test plan (GUI — no automated pixel assertions)

Launch `tsdFlow`: two docked viewports — vp0 a colored volume blob, vp1 a green
bounding box; orbit/pan/zoom works independently per viewport; the `Log` window shows
startup; window resize and clean close work; the "Regenerate" button bumps the noise
seed and the viewports visibly update (exercising `bridge.update()` live).

## Out of scope for 4c

- Node editor / interactive graph construction (4d).
- GraphInspector / parameter UI / transfer-function editor (4d).
- In-app device switching (deferred; default device only).
- CUDA residency (4b).
- Undo/redo, persistence (4e/Phase 5).
- Picking, AOV visualization, multi-device compositing (later).

## Phasing note

First interactive slice of Phase 4. Depends on committed Phases 1–3 + 4a. The node
editor (4d) adds interactive graph construction + inspector on top of this shell and
the reusable `GraphViewport`.
