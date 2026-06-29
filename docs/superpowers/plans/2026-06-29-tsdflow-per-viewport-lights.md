# tsdFlow Per-Viewport Lights — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lighting to tsdFlow in two forms: **plain lights** authored as `DisplayLight` graph nodes (ordinary scene objects, shared across viewports, routed by the existing per-display `viewportMask`), and a per-viewport **headlight** (camera-attached directional light owned by the bridge, auto-on when no plain light reaches a viewport). Replaces the stopgap hard-coded default light.

**Architecture:** Two tasks. (1) Plain lights end-to-end — `Renderable::Kind::Light`, a `DisplayLight` node, bridge `buildLight()`, demo-graph default light, and removal of the temporary `defaultLights`/`indexLayers` workaround. (2) Per-viewport headlight — bridge-owned per-viewport ANARI light injected via `setExternalInstances`, auto-resolved against each viewport's masked lights, plus a `GraphViewport` "Lights" menu feeding the camera direction each frame.

**Tech Stack:** C++17, Dear ImGui, ANARI (`anari_cpp`), `tsd::scene::Light`/`tokens::light`, `tsd::graph` nodes + `Renderable`, `GraphRenderBridge`, `RenderIndexAllLayers::setExternalInstances`, `tsd::ui::buildUI_object`-style ImGui edits.

## Global Constraints

- **jj, not git.** Commit with explicit file paths; never bare `jj commit`, never raw `git`.
- **No `Co-Authored-By` lines.**
- **Never run clang-format on any `CMakeLists.txt`.** Run `clang-format -i` on every touched `.cpp`/`.hpp` before each commit (Google style, per AGENTS.md).
- Configure: `cmake _out/_cmake`. Build: `cmake --build _out/_cmake --parallel`. Test: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` (`-C RelWithDebInfo` REQUIRED). Suite is **69** now → **70** after Task 1 (adds `[bridge-light]`) → **71** after Task 2 (adds `[bridge-headlight]`). Existing tests whose counts change (see Task 1 Step 8) are updated, not worked around.
- **Plain lights ride the normal path.** A `DisplayLight`'s light object goes into its display layer and is included per viewport by `layersForViewport(i)` → the index's `objectMask_all()` path. Do **not** reintroduce `alwaysGatherAllLights` (it overruns on viewport-excluded surface layers — see the bridge history).
- **Headlight is never in the render scene.** It is per-viewport, per-device ANARI state owned by the bridge and injected via `setExternalInstances`; the shared render scene is untouched.
- **Directional only in v1.** Both paths support the `directional` subtype; the `subtype` seam leaves point/HDRI for later.

---

### Task 1: Plain lights through the graph

**Files:**
- Modify: `tsd/src/tsd/graph/Renderable.hpp`
- Add: `tsd/src/tsd/graph_nodes/DisplayLight.cpp`
- Modify: `tsd/src/tsd/graph_nodes/BuiltinNodes.hpp`, `tsd/src/tsd/graph_nodes/BuiltinNodes.cpp`
- Modify: `tsd/src/tsd/graph_nodes/DisplayMask.cpp` (recognize `DisplayLight` as a display)
- Modify: `tsd/src/tsd/graph_nodes/DemoGraph.cpp` (seed the default light; add `DisplayMask.hpp` include)
- Modify: `tsd/src/tsd/graph_nodes/CMakeLists.txt` (add `DisplayLight.cpp`)
- Modify: `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`, `tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp`
- Modify (existing tests, counts shift): `tsd/tests/test_nodes_DisplayMask.cpp`, `tsd/tests/test_nodes_MultiViewport.cpp`
- Add: `tsd/tests/test_bridge_Light.cpp`; Modify: `tsd/tests/CMakeLists.txt`

**Interfaces:**
- Consumes (existing): `tsd::graph_nodes::{portRenderable, kDefaultViewportMask, kMaxViewports}`; `Node`/`NodeTypeInfo`/`ParameterList`/`EvalContext`; `tsd::scene::Scene::createObject<Light>`, `tsd::scene::tokens::light::directional`; `Scene::insertChildObjectNode`; the bridge's `displayName`/`applyParams`/`clearLayerObjects`/`layersForViewport`.
- Produces: `Renderable::Kind::Light`; node type `"DisplayLight"`; bridge `buildLight(layer, r, name)`.

**Context:** Display nodes emit a `renderable` `Value` consumed by the bridge, which builds a TSD object into the display's layer (mask-filtered per viewport). Lights become a third renderable kind on this exact path. The last commit added a stopgap hard-coded light via a `defaultLights` layer + `indexLayers()`; this task removes that and makes the default light a real `DisplayLight`.

- [ ] **Step 1: `Renderable::Kind::Light`**

In `tsd/src/tsd/graph/Renderable.hpp`, extend the enum and document the light mapping:
```cpp
  enum class Kind
  {
    Surface,
    Volume,
    Light
  };
```
Add a comment that for `Light`, `primSubtype` is the ANARI light subtype and `appearance` carries the light params (`color`, `irradiance`, `direction`); `prim` is unused.

- [ ] **Step 2: `DisplayLight` node**

Create `tsd/src/tsd/graph_nodes/DisplayLight.cpp` (mirrors `DisplaySurface.cpp` but is a pure producer — no input port; not transformable in v1):
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;
using float3 = tsd::core::math::float3;

struct DisplayLight : Node
{
  ParameterList params;
  DisplayLight()
  {
    params.set(Token("viewportMask"), kDefaultViewportMask);
    params.set(Token("color"), float3(1.f, 1.f, 1.f));
    params.set(Token("irradiance"), 1.f);
    params.set(Token("direction"), float2(0.f, 240.f)); // azimuth/elevation deg
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("DisplayLight");
    i.category = Token("sink");
    i.outputs.push_back({Token("out"), PortType{portRenderable()}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Light;
    r->primSubtype = Token("directional");
    r->appearance.scalars.push_back(
        {Token("color"), params.getOr<float3>(Token("color"), float3(1.f))});
    r->appearance.scalars.push_back(
        {Token("irradiance"), params.getOr<float>(Token("irradiance"), 1.f)});
    r->appearance.scalars.push_back({Token("direction"),
        params.getOr<float2>(Token("direction"), float2(0.f, 240.f))});

    Value out;
    out.type = PortType{portRenderable()};
    out.residency = hostResidency();
    out.payload = r;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerDisplayLight(NodeRegistry &reg)
{
  reg.registerType(
      Token("DisplayLight"), [] { return std::make_unique<DisplayLight>(); });
}

} // namespace tsd::graph_nodes
```
(Confirm `ParameterList::getOr` and `params.set` accept `float2`/`float3` — they back the same `tsd::core::Any` used by `DisplaySurface`/`viewportMask`. If `getOr<float2>` is unavailable, read via `params.get(...)` as the surrounding nodes do.)

- [ ] **Step 3: Register `DisplayLight`**

In `BuiltinNodes.hpp` add `void registerDisplayLight(tsd::graph::NodeRegistry &reg);`. In `BuiltinNodes.cpp`, call `registerDisplayLight(reg);` inside `registerBuiltinNodes(NodeRegistry&)` (alongside the other displays). Add `DisplayLight.cpp` to `tsd/src/tsd/graph_nodes/CMakeLists.txt` next to `DisplaySurface.cpp` (do **not** clang-format the CMake file).

- [ ] **Step 4: Recognize `DisplayLight` as a display**

In `DisplayMask.cpp` `collectDisplayMasks`, extend the type filter:
```cpp
    if (info.name != Token("DisplayVolume")
        && info.name != Token("DisplaySurface")
        && info.name != Token("DisplayLight"))
      continue;
```
So tsdFlow's `syncDisplays()` calls `setDisplay` for `DisplayLight` nodes and they realize through the bridge like any other display.

- [ ] **Step 5: Bridge `buildLight()` + kind dispatch**

In `GraphRenderBridge.hpp`, declare:
```cpp
  void buildLight(tsd::scene::Layer *layer,
      const tsd::graph::Renderable &r,
      const std::string &name);
```
In `GraphRenderBridge.cpp`, add the `#include "tsd/scene/objects/Light.hpp"` (already present from the stopgap) and define:
```cpp
void GraphRenderBridge::buildLight(
    Layer *layer, const Renderable &r, const std::string &name)
{
  // Spec: unknown light subtype → skip + warn (rest of the layer still builds).
  if (r.primSubtype != tokens::light::directional) {
    tsd::core::logWarning(
        "[GraphRenderBridge] unsupported light subtype '%s'; skipping",
        r.primSubtype.c_str());
    return;
  }
  auto light = m_renderScene.createObject<Light>(r.primSubtype);
  light->setName(name.c_str());
  applyParams(*light, r.appearance);
  m_renderScene.insertChildObjectNode(layer->root(), light);
}
```
(`tokens::light::directional` and `tsd::core::logWarning` are already available in the bridge TU — `Light.hpp` and `Logging.hpp` are included. Confirm the log helper name; the file uses `tsd::core::log*` helpers.)
In `rebuildLayer`, replace the surface/volume `if/else` with a switch on `r->kind`:
```cpp
  auto r = std::static_pointer_cast<Renderable>(out->payload);
  const std::string name = displayName(node);
  switch (r->kind) {
  case Renderable::Kind::Surface: buildSurface(d.layer, *r, name); break;
  case Renderable::Kind::Volume:  buildVolume(d.layer, *r, name);  break;
  case Renderable::Kind::Light:   buildLight(d.layer, *r, name);   break;
  }
  d.isLight = (r->kind == Renderable::Kind::Light); // Task 2 uses this
```
Add `bool isLight{false};` to the bridge's `Display` struct (used by Task 2's headlight auto-resolution; harmless now). Note `d.isLight` is **sticky**: it is set on each successful (re)build and the version-unchanged early-return earlier in `rebuildLayer` leaves it intact — correct, since a realized light display stays a light.

- [ ] **Step 6: Remove the stopgap default light**

In `GraphRenderBridge.{hpp,cpp}`, delete `addDefaultLight()`, the `m_lightsLayer` member, and `indexLayers()`. Revert the three `setIncludedLayers(indexLayers(i))` call sites (`setViewportDevice`, `removeDisplay`, `update`) to `setIncludedLayers(layersForViewport(i))`. Remove the `addDefaultLight();` call from the ctor. (Plain lights now ride the normal included-layers path; no special light gathering.)

- [ ] **Step 7: Seed the demo graph with a default light**

In `DemoGraph.cpp` `buildVolumeSurfaceDemo`, add a `DisplayLight` masked to **all** viewports so default scenes are lit:
```cpp
  const NodeId dl = add("DisplayLight");
  g.node(dl)->impl->parameters().set(
      Token("viewportMask"), int((1 << kMaxViewports) - 1));
```
`add(...)` returns a `NodeId`; set the mask on the created node. **Add `#include "tsd/graph_nodes/DisplayMask.hpp"` to `DemoGraph.cpp`** — it is not currently included and `kMaxViewports` lives there (`Token` is already in scope). No connection needed (DisplayLight has no inputs). Returning it in `DemoDisplays` is optional.

- [ ] **Step 8: Update existing tests for the seeded light (REQUIRED — else suite goes red)**

The seeded all-viewports `DisplayLight` is recognized by `collectDisplayMasks` (Step 4), so it is a fourth display realized into every viewport. Two demo-driven suites assert exact counts and must be updated (this is a real behavior change, not a workaround):
- `tsd/tests/test_nodes_DisplayMask.cpp`: `masks.size()` `3 → 4` (line ~48, viskores build) and `2 → 3` (line ~50, non-viskores). The `find(masks, d.volumeDisplay/surfaceDisplay)` checks still hold; optionally assert the new light display's mask is all-ones.
- `tsd/tests/test_nodes_MultiViewport.cpp`: every `layersForViewport(i)` for a masked viewport gains **+1** (the light is enabled and masked to all viewports). Concretely: vp0 `3 → 4` / `2 → 3` (lines ~65/67), vp1 `1 → 2` (line ~69), vp2 `empty() → size()==1` (line ~70); in the second WHEN, vp0 `2 → 3` / `1 → 2` (lines ~83/85), vp1 `empty() → 1` (line ~87), vp2 `empty() → 1` (line ~88). Adjust each `REQUIRE` to the new value; replace the `.empty()` checks with `.size() == 1`.

Read both files and apply the actual numbers (the demo's viskores `#ifdef` gates which branch is live). Add both files to the Step 10 commit.

- [ ] **Step 9: Build + suite**

`cmake _out/_cmake && cmake --build _out/_cmake --parallel`
`ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: builds clean; **70** pass (69 existing with the two count-updates above, +1 for the new `[bridge-light]` test in Step 10's sibling). Also re-confirm `[tsdflow-smoke]` and `[edit-render]` still pass: they `setDisplay` only the volume/surface (not the seeded light), so vp0 renders with no light after the stopgap removal — their assertions are `color > 0` (volume TF emission, light-independent) and `objectId > 0`; verify both still hold, and if the volume now reads black, add a `setDisplay` for a light in those tests or assert on `objectId` only.

- [ ] **Step 10: Bridge unit test — plain-light masking**

Add `tsd/tests/test_bridge_Light.cpp` (new file) with an `EmitLight` test node mirroring `EmitSphere` (`test_bridge_Rebuild.cpp`) but emitting `Renderable{kind=Light, primSubtype="directional", appearance={color,irradiance,direction}}`. SCENARIO `[bridge-light]`: wire it into a display masked to viewport 0 only; after `update()`, assert `bridge.layersForViewport(0).size() == 1` and `bridge.layersForViewport(1).empty()` and `bridge.world(0) != nullptr` (mirrors `test_bridge_Mask.cpp`).

Register it with **all three** lines in `tsd/tests/CMakeLists.txt` (mirror `BridgeRebuild`):
1. add `test_bridge_Light.cpp` to the `project_add_executable(...)` source list;
2. `add_test(NAME tsd::rendering::BridgeLight COMMAND ${PROJECT_NAME} "[bridge-light]")`;
3. `set_tests_properties(tsd::rendering::BridgeLight PROPERTIES TIMEOUT 300)`.
(Do **not** clang-format `CMakeLists.txt`.)

- [ ] **Step 11: Format + commit**

```bash
clang-format -i <touched .cpp/.hpp>   # NOT CMakeLists.txt
jj commit tsd/src/tsd/graph/Renderable.hpp tsd/src/tsd/graph_nodes/DisplayLight.cpp tsd/src/tsd/graph_nodes/BuiltinNodes.hpp tsd/src/tsd/graph_nodes/BuiltinNodes.cpp tsd/src/tsd/graph_nodes/DisplayMask.cpp tsd/src/tsd/graph_nodes/DemoGraph.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/tests/test_nodes_DisplayMask.cpp tsd/tests/test_nodes_MultiViewport.cpp tsd/tests/test_bridge_Light.cpp tsd/tests/CMakeLists.txt -m "feat(tsdflow): DisplayLight node — plain lights through the graph, mask-filtered"
```
(Drop `test_tsdflow_Smoke.cpp`/`test_nodes_EditRender.cpp` from the commit only if Step 9 confirmed they need no edit.)

---

### Task 2: Per-viewport headlight

**Files:**
- Modify: `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp`, `tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp`
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`, `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp`
- Add: `tsd/tests/test_bridge_Headlight.cpp`; Modify: `tsd/tests/CMakeLists.txt`

**Interfaces:**
- Bridge produces: `struct HeadlightState { enum class Mode {Auto,On,Off} mode; tsd::math::float3 direction, color; float irradiance; };` and `void setViewportHeadlight(int viewport, const HeadlightState &s);`
- Bridge consumes (existing): `m_viewportDevices[i]`, `m_indices[i]->setExternalInstances(const anari::Instance*, size_t)`; the `Display::isLight`/`mask` tracked in Task 1; ANARI `newObject<Light/Group/Instance>` + `setParameterArray1D` + `commitParameters` (idiom mirrors `RenderToAnariObjectsVisitor::createInstanceFromTop`).
- `GraphViewport` produces: a `Lights` menu (Auto/On/Off + color + intensity) and a per-frame `setViewportHeadlight` call with the camera forward.

**Context:** Each viewport's `RenderIndex` exposes one external-instance slot (`setExternalInstances`), concatenated with the layer instances in `updateWorld`. The headlight is the bridge's per-viewport, per-device directional light placed in a one-light group/instance and injected there, aimed by the viewport's camera each frame. Auto resolution uses `Display::isLight` + masks (Task 1).

- [ ] **Step 1: Bridge headlight state + storage (header)**

In `GraphRenderBridge.hpp`, add (public) the state struct + API:
```cpp
  struct HeadlightState
  {
    enum class Mode { Auto, On, Off } mode{Mode::Auto};
    tsd::math::float3 direction{0.f, 0.f, -1.f}; // light travel dir (world)
    tsd::math::float3 color{1.f, 1.f, 1.f};
    float irradiance{1.f};
  };
  void setViewportHeadlight(int viewport, const HeadlightState &s);
  // Whether viewport i's headlight is currently injected (for tests + UI).
  bool headlightActive(int viewport) const;
```
Add private per-viewport storage + helpers:
```cpp
  struct Headlight
  {
    anari::Light light{nullptr};
    anari::Instance instance{nullptr};
    bool active{false}; // currently injected via setExternalInstances
  };
  std::vector<Headlight> m_headlights; // sized to numViewports
  bool viewportHasPlainLight(int i) const;
  void releaseHeadlight(int i);
```
Size `m_headlights` in the ctor: `m_headlights.resize(numViewports);`.

- [ ] **Step 2: `viewportHasPlainLight` + `releaseHeadlight`**

```cpp
bool GraphRenderBridge::viewportHasPlainLight(int i) const
{
  const uint64_t bit = uint64_t(1) << i;
  for (const auto &kv : m_displays) {
    const Display &d = kv.second;
    if (d.enabled && d.realized && d.isLight && (d.mask & bit))
      return true;
  }
  return false;
}

void GraphRenderBridge::releaseHeadlight(int i)
{
  auto &h = m_headlights[i];
  if (h.active && i < int(m_indices.size()))
    m_indices[i]->setExternalInstances(nullptr, 0);
  if (h.instance)
    anari::release(m_viewportDevices[i], h.instance);
  if (h.light)
    anari::release(m_viewportDevices[i], h.light);
  h = Headlight{};
}
```

- [ ] **Step 3: `setViewportHeadlight`**

```cpp
void GraphRenderBridge::setViewportHeadlight(int i, const HeadlightState &s)
{
  if (i < 0 || i >= int(m_indices.size()))
    return;
  auto &h = m_headlights[i];

  // Resolve on/off FIRST so an off headlight does no per-frame ANARI work.
  const bool on = s.mode == HeadlightState::Mode::On
      || (s.mode == HeadlightState::Mode::Auto && !viewportHasPlainLight(i));

  if (!on) {
    if (h.active) {
      m_indices[i]->setExternalInstances(nullptr, 0);
      h.active = false;
    }
    return; // do not create/commit handles for an off headlight
  }

  auto d = m_viewportDevices[i];

  // Lazily build the light + one-light group + instance on this viewport's
  // device (idiom mirrors RenderToAnariObjectsVisitor::createInstanceFromTop).
  if (!h.light) {
    h.light = anari::newObject<anari::Light>(d, "directional");
    auto group = anari::newObject<anari::Group>(d);
    anari::setParameterArray1D(d, group, "light", &h.light, 1);
    anari::commitParameters(d, group);
    h.instance = anari::newObject<anari::Instance>(d, "transform");
    anari::setParameter(d, h.instance, "group", group);
    anari::commitParameters(d, h.instance);
    anari::release(d, group); // instance holds the ref
  }

  // Aim + tint (cheap; no world rebuild — the instance already references it).
  anari::setParameter(d, h.light, "direction", s.direction);
  anari::setParameter(d, h.light, "color", s.color);
  anari::setParameter(d, h.light, "irradiance", s.irradiance);
  anari::commitParameters(d, h.light);

  if (!h.active) {
    m_indices[i]->setExternalInstances(&h.instance, 1);
    h.active = true;
  }
}

bool GraphRenderBridge::headlightActive(int i) const
{
  return i >= 0 && i < int(m_headlights.size()) && m_headlights[i].active;
}
```

- [ ] **Step 4: Device-switch + dtor lifetime**

In `setViewportDevice`, call `releaseHeadlight(i);` as the **first statement** — before the no-op guard and before `m_viewportDevices[i]`/`m_viewportDeviceNames[i]` are reassigned. This is critical: `releaseHeadlight` releases the handles on `m_viewportDevices[i]`, which must still be the **old** device. (The next per-frame `setViewportHeadlight` rebuilds handles on the new device; `GraphViewport` pushes state every frame.) Note the no-op-guard early-return (`m_viewportDevices[i] == d`) then skips needlessly tearing down the headlight when the device is unchanged — so put `releaseHeadlight(i)` *after* the no-op guard but *before* the device reassignment:
```cpp
void GraphRenderBridge::setViewportDevice(int i, Token deviceName, anari::Device d)
{
  if (i < 0 || i >= int(m_indices.size()) || !d)
    return;
  if (m_viewportDevices[i] == d) // no-op guard: same device
    return;
  releaseHeadlight(i);           // release on the OLD device, before reassign
  m_viewportDeviceNames[i] = deviceName;
  m_viewportDevices[i] = d;
  m_indices[i] = std::make_unique<RenderIndexAllLayers>(m_renderScene, deviceName, d);
  m_indices[i]->setIncludedLayers(layersForViewport(i));
}
```
Change the bridge dtor from `= default` to release all headlights:
```cpp
GraphRenderBridge::~GraphRenderBridge()
{
  for (int i = 0; i < int(m_headlights.size()); ++i)
    releaseHeadlight(i);
}
```
(`releaseHeadlight` calls `setExternalInstances` on the index; ensure `m_indices[i]` still exists at dtor time — it does, both are bridge members; if ordering is a concern, guard on `i < m_indices.size()` as written.)

- [ ] **Step 5: `GraphViewport` headlight state + Lights menu (header)**

In `GraphViewport.hpp`, add:
```cpp
  void ui_menu_Lights();
  tsd::rendering::GraphRenderBridge::HeadlightState m_headlight;
```
(`m_headlight.mode` defaults to `Auto`.) Declare `ui_menu_Lights` in the private section.

- [ ] **Step 6: Lights menu + per-frame feed (cpp)**

Add `ui_menu_Device(); ui_menu_Renderer();` → also `ui_menu_Lights();` in the menu bar block:
```cpp
  if (ImGui::BeginMenuBar()) {
    ui_menu_Device();
    ui_menu_Renderer();
    ui_menu_Lights();
    ImGui::EndMenuBar();
  }
```
Implement the menu (radio for mode + color/intensity edits):
```cpp
void GraphViewport::ui_menu_Lights()
{
  if (!ImGui::BeginMenu("Lights"))
    return;
  using Mode = tsd::rendering::GraphRenderBridge::HeadlightState::Mode;
  ImGui::TextUnformatted("Headlight:");
  int mode = int(m_headlight.mode);
  ImGui::RadioButton("Auto", &mode, int(Mode::Auto));
  ImGui::RadioButton("On", &mode, int(Mode::On));
  ImGui::RadioButton("Off", &mode, int(Mode::Off));
  m_headlight.mode = Mode(mode);
  ImGui::ColorEdit3("Color", &m_headlight.color.x);
  ImGui::DragFloat("Intensity", &m_headlight.irradiance, 0.05f, 0.f, 100.f);
  ImGui::EndMenu();
}
```
Feed the headlight **every frame, unconditionally** — NOT inside the `if (m_manip.hasChanged(...))` block (else the light only re-aims when the camera moves). Place it after the existing `if (m_size.x <= 0 || m_size.y <= 0) return;` early-return and after the manipulator-update `if` block, just before `m_anariPass->setWorld(...)`/`m_pipeline.render()`:
```cpp
  m_headlight.direction = linalg::normalize(m_manip.at() - m_manip.eye());
  m_bridge->setViewportHeadlight(m_viewportIndex, m_headlight);
```
`m_manip.at()/eye()` return `anari::math::float3`; `HeadlightState::direction` is `tsd::math::float3`, and `tsd::core::math` does `using namespace anari::math`, so they are the **same type** — assign directly (no component-wise wrapper). `linalg::normalize`/`length` are already used in `drawGizmo`.

- [ ] **Step 7: Build + suite**

`cmake --build _out/_cmake --parallel`
`ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: builds clean; suite stays green.

Add `tsd/tests/test_bridge_Headlight.cpp`, SCENARIO `[bridge-headlight]`, on a visrtx bridge. Assert on the new `headlightActive(i)` accessor (the resolution is otherwise unobservable through the public API — `world(i)` is non-null and `renderSceneObjectCount()` is unchanged whether or not the headlight is injected, so those alone prove nothing):
- No light display: `setViewportHeadlight(0, {Auto})` → `REQUIRE(bridge.headlightActive(0))` (Auto resolves on); `world(0) != nullptr`; `renderSceneObjectCount()` unchanged (headlight is not a render-scene object).
- Realize a `DisplayLight` (reuse `EmitLight` from `[bridge-light]`) masked to viewport 0, `update()`, then `setViewportHeadlight(0, {Auto})` → `REQUIRE(!bridge.headlightActive(0))` (plain light covers it).
- `setViewportHeadlight(0, {On})` → `REQUIRE(bridge.headlightActive(0))`; `{Off}` → `REQUIRE(!bridge.headlightActive(0))`. Repeated `On` calls with changing `direction` leave `renderSceneObjectCount()` unchanged.
- After `setViewportDevice(0, Token("helide"), dev2)` (skip if helide unloadable, per `test_bridge_Rebuild.cpp`'s pattern), `setViewportHeadlight(0, {On})` → `world(0) != nullptr` and `headlightActive(0)` (handles rebuilt on the new device).

Register with all three CMakeLists lines (source-list entry, `add_test(... "[bridge-headlight]")`, `set_tests_properties(... TIMEOUT 300)`) mirroring `BridgeRebuild`; name the test `tsd::rendering::BridgeHeadlight`.

- [ ] **Step 9: Format + commit**

```bash
clang-format -i <touched .cpp/.hpp>   # NOT CMakeLists.txt
jj commit tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp tsd/tests/test_bridge_Headlight.cpp tsd/tests/CMakeLists.txt -m "feat(tsdflow): per-viewport camera headlight (auto when no light reaches a viewport)"
```

- [ ] **Step 10: Manual smoke**

- Fresh demo scene is lit by the seeded `DisplayLight`; deleting it → each viewport's Auto headlight keeps it lit (not black).
- Masking the `DisplayLight` to a subset of viewports lights only those; the rest light via Auto headlight.
- Lights menu: Auto/On/Off switches, color/intensity edits update live; orbiting re-aims the headlight.
- Device switch on a viewport preserves its headlight and renders on the new device; gizmo/nav unaffected.

---

## Self-Review

**Spec coverage:**
- `Renderable::Kind::Light` + `DisplayLight` node + registration + display recognition → Task 1 Steps 1–4. ✓
- Bridge `buildLight()` on the masked-layer path; kind dispatch → Task 1 Step 5. ✓
- Remove the stopgap default light (`defaultLights`/`indexLayers`); seed demo `DisplayLight` (all viewports) → Task 1 Steps 6–7. ✓
- Update existing demo-driven count tests broken by the seeded light (`test_nodes_DisplayMask`, `test_nodes_MultiViewport`) → Task 1 Step 8. ✓ (meta-review BLOCKER)
- Per-viewport headlight: bridge-owned handles, `setViewportHeadlight`, Auto resolution vs masked lights, `setExternalInstances` injection, device-switch recreate (release-first), dtor release → Task 2 Steps 1–4. ✓
- `GraphViewport` Lights menu + **unconditional** per-frame camera feed → Task 2 Steps 5–6. ✓ (meta-review BLOCKER: feed outside the `hasChanged` block)
- Tests: plain-light masking (`[bridge-light]`) + headlight via `headlightActive()` accessor (`[bridge-headlight]`); both registered with source-list + `add_test` + `TIMEOUT 300`; manual smoke → Task 1 Step 10, Task 2 Step 8. ✓ (meta-review: honest, observable assertions)
- Out of scope (point/HDRI UI, gizmo, persistence) honored — only `directional` wired, no gizmo, no persistence. ✓

**Meta-review fixes applied:** (1) seeded-light test-count updates made an explicit required step; (2) headlight test asserts on a new `headlightActive(i)` accessor instead of unobservable counts; (3) per-frame feed specified unconditional; (4) `releaseHeadlight(i)` pinned before device reassignment in `setViewportDevice`; (5) off-headlight skips per-frame create/commit; (6) `buildLight` warns + skips unknown subtype (spec promise); (7) CMake test registration spelled out (3 lines each); (8) `DemoGraph.cpp` gets a firm `DisplayMask.hpp` include; (9) `float3` assigned directly (same type via `tsd::core::math`'s `using namespace anari::math`).

**Placeholder scan:** none — each step carries complete code or an exact edit. Remaining `(Confirm …)` notes are API-name sanity checks (`logWarning` helper name) with the surrounding pattern stated; not TODOs.

**Internal consistency:** `Display::isLight` introduced in Task 1 Step 5 (sticky), consumed by `viewportHasPlainLight` in Task 2 Step 2; `headlightActive()` declared Task 2 Step 1, defined Step 3, consumed by the Step 8 test; `HeadlightState` declared Task 2 Step 1, used by `GraphViewport` Steps 5–6; the menu-bar block in Task 2 Step 6 extends the committed Device+Renderer block — keep all three menus. Suite: 69 → 70 (Task 1, after the two count-updates + `[bridge-light]`) → 71 (Task 2, `[bridge-headlight]`).

**Type consistency:** `DisplayLight` params are `float2` (direction az/el), `float3` (color), `float` (irradiance), matching `tsd::scene::Light`'s `directional` params, applied by the existing `applyParams`. The **headlight** is a raw ANARI `directional` light whose `direction` is `float3` (world travel direction) — set directly, no az/el conversion. `setExternalInstances(const anari::Instance*, size_t)` matches `RenderIndex.hpp`. Bridge ctor sizes both `m_viewportDevices` and `m_headlights` to `numViewports`.

**Ordering note:** Task 1 leaves the bridge with no default light layer; the demo `DisplayLight` keeps scenes lit between tasks. Task 2's headlight then covers viewports a plain light doesn't reach. Both tasks are independently buildable and suite-green.
