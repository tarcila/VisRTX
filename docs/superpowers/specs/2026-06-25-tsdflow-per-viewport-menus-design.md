# tsdFlow: Per-Viewport Device + Renderer Menus — Design

**Date:** 2026-06-25
**Status:** Approved (pending user spec review)
**Phase:** Intermediary step 2 of 3 (per-viewport menus → then lights)

## Problem

Each `GraphViewport` renders with a hard-coded ANARI device (the bridge's shared device) and a hard-coded `"default"` renderer with no UI. Give every viewport a menu bar with the controls the main viewport has: a **Renderer** menu (subtype + introspected parameters) and a **Device** menu (choose the ANARI library/device for that viewport).

`GraphViewport` is deliberately not a `BaseViewport` (it renders the bridge's per-viewport worlds and owns raw `anari::Camera`/`Renderer`/`Device` handles). So the controls are built fresh against ANARI introspection + the app's device manager, reusing the `buildUI_object` widget for renderer params.

## Decisions (settled in brainstorm)

- **Renderer settings:** subtype radios from `tsd::scene::getANARIObjectSubtypes(device, ANARI_RENDERER)` + full introspected params via `buildUI_object` on a `tsd::scene::Object` built by `parseANARIObjectInfo`.
- **Device selection:** genuine **per-viewport device**. Each viewport may render on a different ANARI device; the bridge realizes its render scene through a per-viewport `RenderIndex` on that viewport's device.

## Component A — Viewport menu bar + Renderer settings

**File:** `tsd/src/tsd/ui/imgui/windows/GraphViewport.{hpp,cpp}`. No bridge change.

### Menu bar

Override `int windowFlags() const` to return the base flags `| ImGuiWindowFlags_MenuBar` (today `GraphViewport` uses the default no-menubar flags). In `buildUI`, wrap the menus in `if (ImGui::BeginMenuBar()) { ui_menu_Device(); ui_menu_Renderer(); ImGui::EndMenuBar(); }`, drawn before the image blit / gizmo / nav so the menu bar reserves its row.

### Renderer state + menu

New members: `tsd::core::Token m_rendererSubtype{Token("default")};` and `tsd::scene::Object m_rendererObj;` (the editable renderer mirror). Build `m_rendererObj` once at construction (and whenever the subtype or device changes):

```cpp
m_rendererObj = tsd::scene::parseANARIObjectInfo(
    m_device, ANARI_RENDERER, m_rendererSubtype.c_str());
```

`ui_menu_Renderer()` (mirrors `BaseViewport::ui_menubar_Renderer`):
- Subtype radios over `getANARIObjectSubtypes(m_device, ANARI_RENDERER)`; selecting a new subtype sets `m_rendererSubtype`, rebuilds `m_rendererObj`, and reifies (below).
- `tsd::ui::buildUI_object(m_rendererObj, <scene>, /*useTable=*/true)` to edit params. The `<scene>` argument is `m_bridge->renderScene()` (a real scene for any object-reference resolution; the renderer object itself lives standalone — its params are read/written in place).

### Reify the edited renderer onto the ANARI handle

`m_rendererObj` is the source of truth; the live `anari::Renderer m_renderer` (set on `m_anariPass`) must follow edits. Attach a tiny update-delegate to `m_rendererObj` that sets `bool m_rendererDirty`. Each frame, if `m_rendererDirty` (or after a subtype/device rebuild), reify:

```cpp
if (m_renderer) anari::release(m_device, m_renderer);
m_renderer = (anari::Renderer)m_rendererObj.makeANARIObject(m_device); // creates + commits
m_anariPass->setRenderer(m_renderer);
m_rendererDirty = false;
```

The hard-coded `ambientRadiance` seed in the current ctor is dropped in favor of the introspected defaults from `parseANARIObjectInfo`.

## Component B — Per-viewport device

### Bridge (`tsd/src/tsd/rendering/bridge/GraphRenderBridge.{hpp,cpp}`)

The bridge already holds one `RenderIndexAllLayers` per viewport (`m_indices[i]`), all built on the single ctor device. Generalize to a per-viewport device:

- Track per-viewport device + name: `std::vector<anari::Device> m_viewportDevices;` and `std::vector<tsd::core::Token> m_viewportDeviceNames;`, both initialized to the ctor device/name for all viewports. `m_indices[i]` is built on `m_viewportDevices[i]` (initially identical to today).
- New API:
  ```cpp
  void setViewportDevice(int i, tsd::core::Token deviceName, anari::Device d);
  ```
  It rebuilds that viewport's index on the new device:
  ```cpp
  m_viewportDeviceNames[i] = deviceName;
  m_viewportDevices[i] = d;
  m_indices[i] = std::make_unique<RenderIndexAllLayers>(m_renderScene, deviceName, d);
  m_indices[i]->setIncludedLayers(layersForViewport(i));
  ```
  (The render scene is device-agnostic; each index maintains its own per-device ANARI handle cache. Realizing one scene across several devices is the existing multi-device pattern.)
- `world(int i)` is unchanged (`m_indices[i]->world()`), now on `m_viewportDevices[i]`.
- `update()` continues to refresh every index; each syncs on its own device.
- A no-op guard: `setViewportDevice` with the same device the viewport already has returns early.

### GraphViewport device switch

New ctor parameter `tsd::core::Token deviceName` (the initial device's name), stored as `m_deviceName`. New `ui_menu_Device()`:
- Radios over `appContext()->anari.libraryList()`; the active one is `m_deviceName`.
- On selecting a different library `name`:
  ```cpp
  anari::Device d = appContext()->anari.loadDevice(name); // cached; may be null on failure
  if (d && d != m_device) switchDevice(Token(name), d);
  ```

`switchDevice(Token name, anari::Device d)`:
1. Release this viewport's own handles on the old device: `anari::release(m_device, m_camera)`, `anari::release(m_device, m_renderer)`.
2. `m_device = d; m_deviceName = name;`
3. Rebuild the pipeline on the new device: `m_pipeline.clear();` then re-`emplace_back<AnariSceneRenderPass>(m_device)` (store as `m_anariPass`) and `emplace_back<CopyToSDLTexturePass>(m_app->sdlRenderer())` (store as `m_outputPass`); re-apply `setRunAsync(false)` and `m_pipeline.setDimensions(...)`.
4. Recreate the camera (`anari::newObject<Camera>(d,"perspective")`) and the renderer (rebuild `m_rendererObj` on `d` via `parseANARIObjectInfo`, then reify); `m_anariPass->setCamera/setRenderer`; force a camera-param update.
5. `m_bridge->setViewportDevice(m_viewportIndex, m_deviceName, m_device);`
6. The existing per-frame `m_anariPass->setWorld(m_bridge->world(m_viewportIndex))` now feeds the new device's world.

ANARI's single-device invariant (world + camera + renderer + frame all on one device) holds: after the switch everything for viewport `i` — including `bridge->world(i)` — is on `m_device`.

Devices are owned/cached by `appContext()->anari`; `GraphViewport` never releases the device itself, only its own camera/renderer.

### App wiring (`tsd/apps/interactive/tsdFlow/tsdFlow.cpp`)

Pass the initial device name to each `GraphViewport` (the bridge already receives a device-name Token; reuse it). No other app change — the View menu / window set is unchanged.

## Data flow & boundaries

- Component A is contained in `GraphViewport` (menu bar + renderer mirror + reify). Independent of B.
- Component B spans the bridge (per-viewport device + `setViewportDevice`) and `GraphViewport` (Device menu + `switchDevice`), tied by the existing `world(i)` contract.
- No graph/evaluator/node changes.

## Testing

GUI + device-dependent rendering. Consistent with prior tsdFlow GUI specs, verification is build-green + full-suite-green + a manual smoke checklist. One automatable seam if a second ANARI device is loadable in CI: a bridge test that calls `setViewportDevice(i, name2, device2)` and asserts `world(i)` becomes non-null and `renderSceneObjectCount()` is unchanged. If no second device is loadable in the test environment, that assertion is skipped and the device path is verified manually (note the skip explicitly rather than asserting nothing).

**Manual smoke checklist:**
- Each viewport shows a menu bar with **Device** and **Renderer**.
- Renderer menu: switching subtype changes the image; editing a param (e.g. a sample count / background) updates live; defaults come from introspection (no hard-coded `ambientRadiance`).
- Device menu lists the configured libraries; switching a single viewport to another loadable device re-renders that viewport on the new device while other viewports keep their devices; switching back works (cached).
- No crash on device switch; the gizmo/nav still work after a switch; the layer-debug panel and bridge are unaffected.

## Out of Scope

- Camera / transform-manipulator menus (only Device + Renderer requested; nav + the 4h gizmo already exist).
- Persisting per-viewport device/renderer choices across sessions (Phase 5).
- A shared "all viewports" device control (each viewport is independent).
- Multi-device compositing within a single viewport (`MultiDeviceSceneRenderPass`) — each viewport uses exactly one device.
