# tsdFlow Per-Viewport Device + Renderer Menus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every `GraphViewport` a menu bar with a **Renderer** menu (subtype + introspected params) and a **Device** menu (per-viewport ANARI device), matching the controls the main `Viewport` has — built fresh against ANARI introspection since `GraphViewport` is not a `BaseViewport`.

**Architecture:** Two independent tasks. (A) Menu bar + introspected renderer mirror, contained entirely in `GraphViewport` — no bridge change. (B) Genuine per-viewport device: the bridge tracks a device per viewport and rebuilds that viewport's `RenderIndexAllLayers` on `setViewportDevice`; `GraphViewport::switchDevice` rebuilds its pipeline/camera/renderer on the new device. Task A lands first and is verifiable alone; Task B builds on A's renderer mirror.

**Tech Stack:** C++17, Dear ImGui, ANARI (`anari_cpp`), `tsd::scene::Object`/`parseANARIObjectInfo`/`getANARIObjectSubtypes`/`updateAllANARIParameters`/`updateANARIParameter`, `tsd::scene::EmptyUpdateDelegate`, `tsd::ui::buildUI_object`, `GraphRenderBridge`, `RenderIndexAllLayers`, `ImagePipeline`/`AnariSceneRenderPass`/`CopyToSDLTexturePass`, `ANARIDeviceManager`.

## Global Constraints

- **jj, not git.** Commit with explicit file paths; never bare `jj commit`, never raw `git`.
- **No `Co-Authored-By` lines.**
- **Never run clang-format on any `CMakeLists.txt`.**
- **ANARI single-device invariant:** for any viewport `i`, its world, camera, renderer, and frame must all live on one device. After `switchDevice`, everything for viewport `i` — including `bridge->world(i)` — is on the new `m_device`.
- **Device ownership:** devices are owned/cached by `appContext()->anari` (`ANARIDeviceManager`). `GraphViewport` releases its **own** camera/renderer handles, and balances the transient `+1` ref that each `loadDevice` call returns (release it after the new pass/index have taken their own retains — see Task 2 Step 7). It never holds a long-term owning ref to the device.
- **Object mirror is source of truth:** `m_rendererObj` (a `tsd::scene::Object` from `parseANARIObjectInfo`) holds the edited renderer; the live `anari::Renderer` follows. Two reify paths, mirroring `MultiDeviceViewport`:
  - **Param edit** → a `RendererUpdateDelegate` attached via `m_rendererObj.setUpdateDelegate(&m_rud)`; its `signalParameterUpdated` pushes the one changed param onto the live renderer and commits. (A delegate is *required*: `Object::parameterChanged` only fires the callback when a delegate is set — there is no version bump on edits otherwise, so any "reify on version change" scheme is dead.)
  - **Full rebuild** (initial build, subtype change, device switch) → `reifyRenderer()` creates the handle with `anari::newObject<anari::Renderer>(device, subtype)`, then `m_rendererObj.updateAllANARIParameters(device, r)` + `anari::commitParameters(device, r)`, then `m_anariPass->setRenderer(r)`. (Note: `Object::makeANARIObject` on a base `Object` returns null — it must NOT be used to build the renderer handle.)
- **clang-format:** run `clang-format -i` on every touched `.cpp`/`.hpp` before each commit (Google style, per AGENTS.md). **Never** clang-format any `CMakeLists.txt`.
- Configure: `cmake _out/_cmake`. Build: `cmake --build _out/_cmake --parallel`. Test: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` (`-C RelWithDebInfo` REQUIRED; suite is 68, must stay green).
- **Testing:** GUI + device-dependent rendering; no pure-logic seam for Component A (consistent with prior tsdFlow GUI work). Component B has **one** automatable bridge seam (below) if a second ANARI device is loadable in CI; otherwise it is skipped explicitly and verified via the manual smoke checklist. Verification overall = build-green + suite-green + the manual smoke walk.

---

### Task 1: Component A — viewport menu bar + introspected Renderer menu

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp`

**Interfaces:**
- Consumes (existing): `tsd::scene::parseANARIObjectInfo(anari::Device, anari::DataType, const char*) -> Object`; `tsd::scene::getANARIObjectSubtypes(anari::Device, anari::DataType) -> std::vector<std::string>`; `tsd::scene::Object::updateAllANARIParameters(anari::Device, anari::Object)`; `tsd::scene::Object::updateANARIParameter(anari::Device, anari::Object, const Parameter&, const char*)`; `tsd::scene::Object::setUpdateDelegate(BaseUpdateDelegate*)`; `tsd::scene::EmptyUpdateDelegate` (subclass it — `BaseUpdateDelegate::signalParameterUpdated` and siblings are pure virtual; `EmptyUpdateDelegate` stubs them); `anari::newObject<anari::Renderer>(device, subtype)` + `anari::commitParameters`; `tsd::ui::buildUI_object(Object&, Scene&, bool useTable)`; `AnariSceneRenderPass::setRenderer(anari::Renderer)`; `GraphRenderBridge::renderScene() -> Scene&`; `Window::windowFlags()` (base: `protected virtual int`, returns `0`).
- Reference implementation to mirror: `tsd/src/tsd/ui/imgui/windows/MultiDeviceViewport.cpp` — `loadANARIRendererParameters` (build + `setUpdateDelegate`), `updateAllRendererParameters` (full reify), and `RendererUpdateDelegate::signalParameterUpdated` (per-param push).
- Produces: a menu bar on `GraphViewport` (Renderer menu now; Device menu added in Task 2) and an introspected, editable renderer with no hard-coded `ambientRadiance`.

**Context:** `GraphViewport` currently builds `m_renderer` directly in its ctor as `"default"` with a hard-coded `ambientRadiance=1`, uses default (no-menubar) window flags, and blits the image at the top of `buildUI`. We add a menu bar row, an introspected `Object` renderer mirror, and a per-frame reify so menu edits drive the live handle.

- [ ] **Step 1: Header — includes, menu-bar flag, renderer-mirror members + method declarations**

In `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`:

Add includes (after the existing `#include "tsd/ui/imgui/windows/Window.h"`):
```cpp
#include "tsd/core/Token.hpp"
#include "tsd/scene/Object.hpp"
#include "tsd/scene/UpdateDelegate.hpp" // EmptyUpdateDelegate
```

In the `private:` section, add method declarations (after `bool drawGizmo(...);`):
```cpp
  void ui_menu_Renderer();
  void rebuildRendererObject(); // (re)introspect m_rendererObj on m_device + attach delegate
  void reifyRenderer();         // full rebuild: new handle + push all params + commit

  // Pushes a single edited renderer param onto the live anari::Renderer.
  // Mirrors MultiDeviceViewport::RendererUpdateDelegate. Subclass
  // EmptyUpdateDelegate (NOT BaseUpdateDelegate, whose methods are pure
  // virtual) so only signalParameterUpdated must be overridden.
  struct RendererUpdateDelegate : public tsd::scene::EmptyUpdateDelegate
  {
    anari::Device *device{nullptr};
    anari::Renderer *renderer{nullptr};
    void signalParameterUpdated(
        const tsd::scene::Object *o, const tsd::scene::Parameter *p) override;
  };
```
Also add the `windowFlags` override in the `private:` section (matching the base's `protected virtual` visibility — keep it non-public):
```cpp
  int windowFlags() const override;
```

Add data members (after `anari::Renderer m_renderer{nullptr};`):
```cpp
  tsd::core::Token m_rendererSubtype{tsd::core::Token("default")};
  tsd::scene::Object m_rendererObj;     // editable renderer mirror
  RendererUpdateDelegate m_rud;         // reifies edits onto m_renderer
```

- [ ] **Step 2: Implementation — includes, windowFlags, renderer build/reify helpers**

In `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp`, add includes (after `#include "tsd/ui/imgui/Application.h"`):
```cpp
#include "tsd/ui/imgui/tsd_ui_imgui.h"
```

Add the window-flags override (anywhere in the file, e.g. just below the ctor/dtor):
```cpp
int GraphViewport::windowFlags() const
{
  return Window::windowFlags() | ImGuiWindowFlags_MenuBar;
}
```

Add the renderer helpers + the delegate definition. `rebuildRendererObject` introspects the mirror and wires the delegate to the live device/renderer pointers; `reifyRenderer` builds the live handle the correct way (NOT `makeANARIObject`); the delegate pushes single-param edits:
```cpp
void GraphViewport::rebuildRendererObject()
{
  m_rendererObj =
      tsd::scene::parseANARIObjectInfo(m_device, ANARI_RENDERER,
          m_rendererSubtype.c_str());
  // Point the delegate at this viewport's live device/renderer members (their
  // addresses are stable; values are refreshed on each reify / device switch).
  m_rud.device = &m_device;
  m_rud.renderer = &m_renderer;
  m_rendererObj.setUpdateDelegate(&m_rud);
}

void GraphViewport::reifyRenderer()
{
  if (m_renderer)
    anari::release(m_device, m_renderer);
  m_renderer =
      anari::newObject<anari::Renderer>(m_device, m_rendererSubtype.c_str());
  m_rendererObj.updateAllANARIParameters(m_device, m_renderer);
  anari::commitParameters(m_device, m_renderer);
  if (m_anariPass)
    m_anariPass->setRenderer(m_renderer);
}

void GraphViewport::RendererUpdateDelegate::signalParameterUpdated(
    const tsd::scene::Object *o, const tsd::scene::Parameter *p)
{
  if (!device || !renderer || !*renderer)
    return;
  o->updateANARIParameter(*device, *renderer, *p, p->name().c_str());
  anari::commitParameters(*device, *renderer);
}
```

- [ ] **Step 3: Ctor — replace the hard-coded renderer with the introspected mirror**

In the ctor, replace:
```cpp
  m_renderer = anari::newObject<anari::Renderer>(m_device, "default");
  anari::setParameter(m_device, m_renderer, "ambientRadiance", 1.f);
  anari::commitParameters(m_device, m_renderer);
```
with:
```cpp
  rebuildRendererObject(); // introspected defaults; no hard-coded ambientRadiance
```

Then, **after** `m_anariPass` is created and `setRunAsync(false)` is called (so `setRenderer` has a pass to target), reify once:
```cpp
  reifyRenderer();
```
Remove the now-redundant `m_anariPass->setRenderer(m_renderer);` line (reify already calls it). Keep `m_anariPass->setCamera(m_camera);`.

> Note: the ctor currently calls `m_anariPass->setRenderer(m_renderer)` before `setRunAsync`. After this change `reifyRenderer()` is the single place that sets the renderer on the pass; place the `reifyRenderer()` call right after `m_anariPass->setRunAsync(false);`.

- [ ] **Step 4: buildUI — draw the menu bar and reify on edit**

At the **top** of `buildUI()` (before the content-region/size logic), draw the menu bar so it reserves its row:
```cpp
  if (ImGui::BeginMenuBar()) {
    ui_menu_Renderer();
    ImGui::EndMenuBar();
  }
```

No per-frame reify is needed: param edits are pushed immediately by the delegate (`signalParameterUpdated`), and subtype/device changes call `reifyRenderer()` at the edit site (Step 5 / Task 2 Step 8). Drawing the menu bar before the existing `if (m_size <= 0) return;` early-return is fine — the bar is closed with `EndMenuBar` in the same `if`, and there is nothing else to do on a zero-size frame.

- [ ] **Step 5: Implement `ui_menu_Renderer()`**

Add (mirrors `BaseViewport::ui_menubar_Renderer`, but over the single `m_rendererObj`; subtype radios introspected each open):
```cpp
void GraphViewport::ui_menu_Renderer()
{
  if (!ImGui::BeginMenu("Renderer"))
    return;

  const auto subtypes =
      tsd::scene::getANARIObjectSubtypes(m_device, ANARI_RENDERER);
  if (subtypes.size() > 1) {
    ImGui::Text("Subtype:");
    for (size_t i = 0; i < subtypes.size(); ++i) {
      ImGui::PushID(int(i));
      const bool selected = m_rendererSubtype == tsd::core::Token(subtypes[i].c_str());
      if (ImGui::RadioButton(subtypes[i].c_str(), selected) && !selected) {
        m_rendererSubtype = tsd::core::Token(subtypes[i].c_str());
        rebuildRendererObject();
        reifyRenderer();
      }
      ImGui::PopID();
    }
    ImGui::Separator();
  }

  ImGui::Text("Parameters:");
  // The scene arg only resolves object-reference params; the standalone
  // renderer object has none, but buildUI_object requires a scene handle.
  tsd::ui::buildUI_object(m_rendererObj, m_bridge->renderScene(), /*useTable=*/true);

  ImGui::EndMenu();
}
```

- [ ] **Step 6: Configure, build, run the suite**

Run: `cmake _out/_cmake && cmake --build _out/_cmake --parallel`
Then: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: builds clean; 68 pass. If `ImGuiWindowFlags_MenuBar` is unresolved, ensure `imgui.h` is included (it is, via `GraphViewport.cpp`’s existing `#include "imgui.h"`).

- [ ] **Step 7: Manual smoke (Component A)**

- Each viewport shows a menu bar with a **Renderer** menu.
- Switching subtype (if the device exposes more than one) changes the image; editing a param (e.g. sample count / background / ambientRadiance) updates the render live.
- Renderer defaults come from introspection — no hard-coded `ambientRadiance` seed.
- Gizmo and camera-nav still work (the menu bar reserves its row; the image blit/gizmo/nav are unchanged below it).

- [ ] **Step 8: Format + commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp
jj commit tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp -m "feat(tsdflow): GraphViewport menu bar + introspected per-viewport renderer settings"
```

---

### Task 2: Component B — per-viewport ANARI device

**Files:**
- Modify: `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`
- Modify: `tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp`
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp`
- Modify: `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`
- (Optional, if a second device is loadable in CI) Modify: the bridge test source under `tsd/tests/`.

**Interfaces:**
- Bridge produces: `void setViewportDevice(int i, tsd::core::Token deviceName, anari::Device d);`
- Bridge consumes (existing): `RenderIndexAllLayers(Scene&, Token, anari::Device)`; `setIncludedLayers(layersForViewport(i))`; `world(int)` unchanged.
- `GraphViewport` produces: ctor gains `tsd::core::Token deviceName`; `ui_menu_Device()`; `switchDevice(tsd::core::Token, anari::Device)`.
- `GraphViewport` consumes (existing): `appContext()->anari.libraryList()`, `appContext()->anari.loadDevice(const std::string&)`; `ImagePipeline::{clear, emplace_back, setDimensions}`; `AnariSceneRenderPass(anari::Device)`; `CopyToSDLTexturePass(SDL_Renderer*)`; `m_app->sdlRenderer()`.

**Context:** The bridge holds one `RenderIndexAllLayers` per viewport, all built on the single ctor device (`m_device`). Generalize to a per-viewport device so a viewport can render on a different ANARI library. The render scene is device-agnostic; each index keeps its own per-device handle cache (the existing multi-device pattern). `world(i)` already returns `m_indices[i]->world()`, so it needs no change once `m_indices[i]` is rebuilt on the new device.

- [ ] **Step 1: Bridge header — per-viewport device tracking + new API**

In `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`, add the public method (after `void update();`):
```cpp
  // Rebuild viewport i's RenderIndex on a new device. No-op if d is already
  // this viewport's device. The render scene is device-agnostic; the index
  // maintains its own per-device ANARI handle cache.
  void setViewportDevice(int i, tsd::core::Token deviceName, anari::Device d);
```

Add data members (after `anari::Device m_device;`):
```cpp
  // Authoritative per-viewport device + name. m_device/m_deviceName above are
  // only the ctor defaults (used to seed these and for device-agnostic layer
  // creation); per-viewport device is m_viewportDevices[i].
  std::vector<anari::Device> m_viewportDevices;
  std::vector<tsd::core::Token> m_viewportDeviceNames;
```

- [ ] **Step 2: Bridge ctor — initialize per-viewport device vectors; build indices per device**

In `tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp`, replace the index-construction loop:
```cpp
  for (int i = 0; i < numViewports; ++i) {
    m_indices.push_back(std::make_unique<RenderIndexAllLayers>(
        m_renderScene, m_deviceName, m_device));
  }
```
with:
```cpp
  m_viewportDevices.assign(numViewports, m_device);
  m_viewportDeviceNames.assign(numViewports, m_deviceName);
  for (int i = 0; i < numViewports; ++i) {
    m_indices.push_back(std::make_unique<RenderIndexAllLayers>(
        m_renderScene, m_viewportDeviceNames[i], m_viewportDevices[i]));
  }
```

- [ ] **Step 3: Bridge — implement `setViewportDevice`**

Add (near `update()`):
```cpp
void GraphRenderBridge::setViewportDevice(
    int i, Token deviceName, anari::Device d)
{
  if (i < 0 || i >= int(m_indices.size()) || !d)
    return;
  if (m_viewportDevices[i] == d) // no-op guard: same device
    return;

  m_viewportDeviceNames[i] = deviceName;
  m_viewportDevices[i] = d;
  m_indices[i] =
      std::make_unique<RenderIndexAllLayers>(m_renderScene, deviceName, d);
  m_indices[i]->setIncludedLayers(layersForViewport(i));
}
```

`world(i)` is unchanged (`m_indices[i]->world()`, now on the per-viewport device). `update()` continues to refresh every index; each syncs on its own device.

- [ ] **Step 4: Bridge — build + suite (B half-way checkpoint)**

Run: `cmake --build _out/_cmake --parallel && ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: builds clean; 68 pass (the existing single-device path is byte-for-byte unchanged: vectors are all the ctor device, indices built identically).

- [ ] **Step 5: GraphViewport header — device-switch members + declarations**

In `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`:

Add a ctor parameter `tsd::core::Token deviceName` (place it right after `anari::Device device,`):
```cpp
  GraphViewport(Application *app,
      tsd::rendering::GraphRenderBridge *bridge,
      int viewportIndex,
      anari::Device device,
      tsd::core::Token deviceName,
      tsd::graph::Graph *graph,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Viewport");
```

Add private method declarations (next to `ui_menu_Renderer();`):
```cpp
  void ui_menu_Device();
  void switchDevice(tsd::core::Token name, anari::Device d);
```

Add a data member (after `anari::Device m_device{nullptr};`):
```cpp
  tsd::core::Token m_deviceName;
```

- [ ] **Step 6: GraphViewport ctor — store the device name**

In the ctor initializer list, add `m_deviceName(deviceName),` (e.g. right after `m_device(device),`). Update the ctor signature to match Step 5.

- [ ] **Step 7: GraphViewport — implement `ui_menu_Device()` and add it to the menu bar**

Add to the menu bar in `buildUI()` (before `ui_menu_Renderer();` so Device is leftmost, matching the main Viewport order Device→Renderer):
```cpp
  if (ImGui::BeginMenuBar()) {
    ui_menu_Device();
    ui_menu_Renderer();
    ImGui::EndMenuBar();
  }
```
(Replace the Task-1 single-menu `BeginMenuBar` block with this.)

Implement:
```cpp
void GraphViewport::ui_menu_Device()
{
  if (!ImGui::BeginMenu("Device"))
    return;

  for (const auto &name : appContext()->anari.libraryList()) {
    const bool selected = m_deviceName == tsd::core::Token(name.c_str());
    if (ImGui::RadioButton(name.c_str(), selected) && !selected) {
      // loadDevice ALWAYS retains (both create and cache-hit paths), so this
      // call returns a +1 handle. switchDevice's pass + bridge index each take
      // their own retain; release this transient menu ref afterward (mirrors
      // ANARIDeviceManager::loadDeviceExtensions). The manager keeps the device
      // cached/alive regardless.
      anari::Device d = appContext()->anari.loadDevice(name); // may be null
      if (d && d != m_device) {
        switchDevice(tsd::core::Token(name.c_str()), d);
        anari::release(d, d);
      } else if (d) {
        anari::release(d, d); // already our device: drop the transient ref
      }
    }
  }

  ImGui::EndMenu();
}
```

> Why release is safe: after `switchDevice`, the viewport's `AnariSceneRenderPass` (built on `d`) and the bridge's rebuilt `RenderIndexAllLayers` (built on `d`) each hold their own reference to `d`; `m_device` is a non-owning copy. The single extra ref that `loadDevice` took for this menu interaction must be dropped or it leaks one device ref per switch. `GraphViewport` still never *owns* the device — it only balances the transient `loadDevice` ref it just created.

- [ ] **Step 8: GraphViewport — implement `switchDevice`**

```cpp
void GraphViewport::switchDevice(tsd::core::Token name, anari::Device d)
{
  // 1) Release this viewport's own handles on the old device.
  if (m_camera)
    anari::release(m_device, m_camera);
  if (m_renderer)
    anari::release(m_device, m_renderer);
  m_camera = nullptr;
  m_renderer = nullptr;

  // 2) Adopt the new device.
  m_device = d;
  m_deviceName = name;

  // 3) Rebuild the pipeline on the new device.
  m_pipeline.clear();
  m_anariPass =
      m_pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(m_device);
  m_anariPass->setRunAsync(false);
  m_outputPass = m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());
  if (m_size.x > 0 && m_size.y > 0)
    m_pipeline.setDimensions(uint32_t(m_size.x), uint32_t(m_size.y));

  // 4) Recreate camera + renderer on the new device.
  m_camera = anari::newObject<anari::Camera>(m_device, "perspective");
  anari::commitParameters(m_device, m_camera);
  m_anariPass->setCamera(m_camera);
  rebuildRendererObject();
  reifyRenderer();

  // Force a camera-param refresh next frame.
  m_manipToken = 0;

  // 5) Tell the bridge to rebuild this viewport's world on the new device.
  m_bridge->setViewportDevice(m_viewportIndex, m_deviceName, m_device);

  // 6) The per-frame m_anariPass->setWorld(m_bridge->world(m_viewportIndex))
  //    in buildUI() now feeds the new device's world. Reset size so the next
  //    frame re-applies aspect on the new camera.
  m_size = tsd::math::int2(0, 0);
}
```

> Rationale for `m_size = {0,0}`: the size block in `buildUI()` only reapplies `aspect`/dimensions when the size changes; zeroing it forces a re-apply against the freshly created camera and pipeline next frame. `m_manipToken = 0` forces `m_manip.hasChanged()` true so the camera params are pushed.

- [ ] **Step 9: App wiring — pass the device name to each GraphViewport**

In `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`, the initial library name is the local `lib` (a `std::string`) already used for the bridge. Pass it to each viewport ctor:
```cpp
      auto *vp = new ui::GraphViewport(this,
          m_bridge.get(),
          i,
          m_device,
          Token(lib.c_str()),
          &m_graph,
          &m_selected,
          &m_graphDirty,
          nm);
```
(`Token` is already in scope in this file — it is used for the bridge ctor on the lines above.) No other app change; the View menu / window set is unchanged.

- [ ] **Step 10: Build + suite**

Run: `cmake --build _out/_cmake --parallel && ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: builds clean; 68 pass.

- [ ] **Step 11: (Optional) Bridge unit seam for the device path**

The bridge tests have **no** `ctx`/`ANARIDeviceManager` — they create devices directly via `anari::loadLibrary(...)` + `anari::newDevice(...)` (see `tsd/tests/test_bridge_Rebuild.cpp`, which hardcodes `"visrtx"`). So gate on a directly-loadable second library, not `ctx->anari`:
- In a bridge test, attempt a second device: `auto lib2 = anari::loadLibrary("helide", nullptr, nullptr);` (or another known-present library distinct from the fixture's `visrtx`); if null, **skip** (see below).
- With `dev2 = anari::newDevice(lib2, "default")`: after a normal `update()` yielding `renderSceneObjectCount() == baseline`, call `bridge.setViewportDevice(0, tsd::core::Token("helide"), dev2)`.
- Assert `bridge.world(0) != nullptr` and `bridge.renderSceneObjectCount() == baseline` (render scene is device-agnostic; only the index was rebuilt). Release `dev2`/`lib2` at test end.

If no second library loads, **do not** add a vacuous assertion — `WARN(...)`/comment that the device-switch path is verified manually and skip. State the skip explicitly rather than asserting nothing.

- [ ] **Step 12: Manual smoke (Component B)**

- Each viewport's **Device** menu lists the configured libraries (`appContext()->anari.libraryList()`); the active one is checked.
- Switching one viewport to another loadable device re-renders **that** viewport on the new device while other viewports keep their devices.
- Switching back works (the device is cached by the manager).
- No crash on switch; the gizmo/nav still work afterward; the Renderer menu still edits on the new device (renderer was rebuilt via `rebuildRendererObject`/`reifyRenderer`).
- The layer-debug panel and the bridge are unaffected (render scene unchanged; only per-viewport indices differ).

- [ ] **Step 13: Format + commit**

```bash
clang-format -i tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/apps/interactive/tsdFlow/tsdFlow.cpp
jj commit tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/apps/interactive/tsdFlow/tsdFlow.cpp -m "feat(tsdflow): per-viewport ANARI device selection (bridge + GraphViewport Device menu)"
```
(If Step 11 added a test, clang-format + include its path in the commit.)

---

## Self-Review

**Spec coverage:**
- Menu bar via `windowFlags() | ImGuiWindowFlags_MenuBar`, drawn before image/gizmo/nav → Task 1 Steps 1, 4. ✓
- Renderer: subtype radios from `getANARIObjectSubtypes`; params via `buildUI_object` on a `parseANARIObjectInfo` `Object`; drop hard-coded `ambientRadiance` → Task 1 Steps 2, 3, 5. ✓
- Reify the edited renderer onto the live handle → Task 1 Steps 2, 5 (update-delegate for per-param edits + `reifyRenderer` using `newObject`+`updateAllANARIParameters`+`commit` for full rebuilds; mirrors `MultiDeviceViewport`). The earlier `versionChanged`-without-delegate idea was dropped — meta-review confirmed `Object::parameterChanged` only fires/bumps when a delegate is attached, and base `Object::makeANARIObject` returns null. ✓
- Per-viewport device in the bridge (`setViewportDevice`, per-viewport device/name vectors, index rebuilt on the device, no-op guard, `world(i)` unchanged) → Task 2 Steps 1–3. ✓
- `GraphViewport` Device menu + `switchDevice` (release own handles, adopt device, rebuild pipeline/camera/renderer, call `setViewportDevice`, world feeds new device) → Task 2 Steps 5–8. ✓
- Device-ref lifetime: `loadDevice` always retains; the menu releases its transient ref after the pass/index take theirs → Task 2 Step 7 + Global Constraints. ✓ (meta-review MAJOR fix)
- App passes initial device name to each viewport; no other app change → Task 2 Step 9. ✓
- Testing = build + suite + smoke, plus the single optional bridge seam with explicit skip semantics → Task 1 Step 6–7; Task 2 Steps 10–12. ✓

**Behavior-preservation:** the bridge's existing single-device path is unchanged — vectors initialize to the ctor device/name and indices build identically (Task 2 Step 2); `world()`/`update()` untouched. Task 1 changes only `GraphViewport`'s own renderer construction (introspected defaults replace the `"default"`+`ambientRadiance` seed) and add a menu bar row.

**Placeholder scan:** none — every step carries complete code or an exact mechanical edit. Step 11 is explicitly conditional with a stated skip rule (no vacuous assertion).

**Type consistency:** ctor signature change (Task 2 Step 5) threads through the initializer list (Step 6) and the single call site in `tsdFlow.cpp` (Step 9); `m_rendererSubtype`/`m_deviceName` are `tsd::core::Token` compared via `==`; `getANARIObjectSubtypes`/`libraryList` yield `std::string`, wrapped as `Token(s.c_str())` for comparison and stored; the live renderer is built with `anari::newObject<anari::Renderer>` (NOT `makeANARIObject`, which returns null on a base `Object`) and parameterized via `updateAllANARIParameters`+`commitParameters`; the `RendererUpdateDelegate` overrides `signalParameterUpdated(const Object*, const Parameter*)` and uses `updateANARIParameter`; `setViewportDevice` signature matches its definition and the `switchDevice` call site; `RenderIndexAllLayers(Scene&, Token, anari::Device)` matches the ctor and rebuild call.

**Ordering note:** Task 1 leaves a single-menu `BeginMenuBar` block; Task 2 Step 7 replaces it with the two-menu (Device, Renderer) block. If both tasks land together, write the two-menu block directly and skip the interim single-menu form.
