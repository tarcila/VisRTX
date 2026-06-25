# tsdFlow Phase 4i — Viewport Input Routing & Compact Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the tsdFlow viewport so the ImGuizmo transform gizmo captures drags and camera nav only responds to genuine in-viewport drags, and make GraphEditor nodes compact-by-default with double-click expansion.

**Architecture:** Two fully independent UI changes, one file each. Component A restructures `GraphViewport::buildUI` input order (blit → gizmo → conditional nav button) and rewrites `handleNavigation` to drive off the InvisibleButton's own ImGui item state. Component B adds a collapsed-by-default node rendering path in `GraphEditor` with title-bar proxy pins so links stay visible, a double-click toggle, and proxy→sole-port link resolution.

**Tech Stack:** C++17, Dear ImGui, vendored imnodes v0.5, vendored ImGuizmo, ANARI. Build via CMake (`_out/_cmake`), tests via CTest.

## Global Constraints

- **Version control is jj, not git.** Use `jj` commands only; raw `git` will fail. Commit with **explicit file paths** (`jj commit <path>... -m "..."`) — never a bare `jj commit` (it would sweep unrelated working-copy files, including `.envrc`).
- **Never commit `.envrc`.**
- **No `Co-Authored-By` lines** in commit messages.
- **Never run clang-format on any `CMakeLists.txt`.**
- **VisRTX render tests need a long timeout**; they already carry `TIMEOUT 300` in their CTest definitions — just run the suite normally.
- **Build:** `cmake --build _out/_cmake --parallel`
- **Test:** `ctest --test-dir _out/_cmake --output-on-failure` (suite is currently 63 tests; it must stay green with no regressions).
- **Testing approach for this plan:** both components are pure ImGui/imnodes/ImGuizmo interaction code with no logic seam the unit harness can drive. There are no new unit tests; per-task verification is (1) the build stays green, (2) the full CTest suite stays green with no regressions, and (3) a reasoned walk-through of the manual smoke checklist in the task (the implementer confirms the code satisfies each checklist item by reading the control flow — actual on-screen GUI testing is the human's follow-up). Do **not** fabricate a unit test that asserts nothing.

---

### Task 1: Component A — GraphViewport input routing

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp` (swap nav state members)
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp` (`buildUI`, `handleNavigation`)

**Interfaces:**
- Consumes (existing, unchanged): `GraphViewport::drawGizmo(const ImVec2&, const ImVec2&)` returns `bool` (`ImGuizmo::IsUsing() || ImGuizmo::IsOver()`); `tsd::rendering::Manipulator m_manip` with `.rotate(float2)`, `.pan(float2)`, `.zoom(float)`, `.startNewRotation()`; file constant `kWheelZoomScale`; `using float2 = anari::math::float2;`; `m_size` is `tsd::math::int2`. `GraphViewport.hpp` already `#include`s `"ImGuizmo.h"`.
- Produces: none (self-contained window behavior change).

**Context:** `GraphViewport` is a standalone ImGui window rendering one of the bridge's per-viewport ANARI worlds. The bug: the full-rect `InvisibleButton` is currently created *before* the gizmo runs, so it grabs ImGui's `ActiveId` and the gizmo never receives the drag (#2); and `handleNavigation` keys off global `IsMouseDown` + sticky flags, so it rotates for title-bar / out-of-rect drags (#1).

- [ ] **Step 1: Swap the navigation state members in the header**

In `tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp`, replace this block:

```cpp
  // Mouse-navigation state (mirrors BaseViewport's normalized-delta model).
  tsd::math::float2 m_prevMouse{-1.f};
  bool m_manipulating{false};
  bool m_rotating{false};
```

with:

```cpp
  // Camera-navigation state: a single rising-edge latch so startNewRotation()
  // re-arms whenever orbit resumes within one held drag (e.g. orbit → Shift to
  // dolly → release back to orbit).
  bool m_orbiting{false};
```

- [ ] **Step 2: Reorder the input handling in `buildUI`**

In `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp`, replace the current tail of `buildUI` — this block:

```cpp
  // Reserve the viewport area with an invisible button BEFORE blitting: while
  // held it owns ImGui's ActiveId, so drags are consumed as navigation rather
  // than moving the dock/window. The rendered texture is then drawn into the
  // same rect via the window draw list.
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImVec2 imgSize(float(m_size.x), float(m_size.y));
  ImGui::InvisibleButton("##viewport",
      imgSize,
      ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight
          | ImGuiButtonFlags_MouseButtonMiddle);
  ImGui::GetWindowDrawList()->AddImage((ImTextureID)m_outputPass->getTexture(),
      pos,
      ImVec2(pos.x + imgSize.x, pos.y + imgSize.y),
      ImVec2(0, 1),
      ImVec2(1, 0));

  const bool gizmoActive = drawGizmo(pos, imgSize);
  if (!gizmoActive)
    handleNavigation();
}
```

with:

```cpp
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImVec2 imgSize(float(m_size.x), float(m_size.y));
  // Blit the rendered texture first. AddImage is draw-list only: it does not
  // advance the cursor or claim input, so the cursor stays at pos for the
  // InvisibleButton below.
  ImGui::GetWindowDrawList()->AddImage((ImTextureID)m_outputPass->getTexture(),
      pos,
      ImVec2(pos.x + imgSize.x, pos.y + imgSize.y),
      ImVec2(0, 1),
      ImVec2(1, 0));

  // Run the gizmo first; ImGuizmo hit-tests and captures the mouse internally.
  const bool gizmoActive = drawGizmo(pos, imgSize);

  // Camera drag-nav only when the gizmo is not hot. The InvisibleButton owns
  // ImGui's ActiveId only when a press lands inside it, so window-decoration
  // drags and out-of-rect presses never manipulate; and when the gizmo is hot
  // no button is submitted to steal its press.
  if (!gizmoActive) {
    ImGui::InvisibleButton("##viewport",
        imgSize,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight
            | ImGuiButtonFlags_MouseButtonMiddle);
    handleNavigation();
  } else {
    m_orbiting = false; // gizmo owns the frame; drop any orbit latch
  }

  // Wheel zoom works anywhere over the viewport (ImGuizmo ignores the wheel),
  // suppressed only during an active gizmo drag.
  ImGuiIO &io = ImGui::GetIO();
  if (ImGui::IsWindowHovered() && !ImGuizmo::IsUsing() && io.MouseWheel != 0.f)
    m_manip.zoom(io.MouseWheel * kWheelZoomScale);
}
```

- [ ] **Step 3: Rewrite `handleNavigation`**

Replace the entire current `handleNavigation` body:

```cpp
void GraphViewport::handleNavigation()
{
  const bool hovered = ImGui::IsItemHovered();
  ImGuiIO &io = ImGui::GetIO();

  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Right)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftShift));
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftAlt));
  const bool orbit =
      ImGui::IsMouseDown(ImGuiMouseButton_Left) && !dolly && !pan;

  const bool anyMovement = dolly || pan || orbit;
  if (!anyMovement) {
    m_manipulating = false;
    m_prevMouse = float2(-1.f);
  } else if (hovered && !m_manipulating) {
    m_manipulating = true;
  }
  if (m_rotating && !orbit)
    m_rotating = false;

  if (m_manipulating) {
    const float2 mouse(io.MousePos.x, io.MousePos.y);
    if (m_prevMouse != float2(-1.f)) {
      const float2 delta = (mouse - m_prevMouse) * 2.f / float2(m_size);
      if (delta != float2(0.f)) {
        if (orbit) {
          if (!m_rotating) {
            m_manip.startNewRotation();
            m_rotating = true;
          }
          m_manip.rotate(delta);
        } else if (dolly)
          m_manip.zoom(delta.y);
        else if (pan)
          m_manip.pan(delta);
      }
    }
    m_prevMouse = mouse;
  }

  if (hovered && io.MouseWheel != 0.f)
    m_manip.zoom(io.MouseWheel * kWheelZoomScale);
}
```

with this — note wheel zoom is gone (now handled in `buildUI`), and manipulation is gated on the button's own `IsItemActive()`:

```cpp
void GraphViewport::handleNavigation()
{
  // Manipulate only while the viewport's InvisibleButton is held — i.e. the
  // press landed inside the viewport rect. Title-bar drags and presses that
  // started elsewhere never set this item active, so they never manipulate.
  if (!ImGui::IsItemActive()) {
    m_orbiting = false;
    return;
  }

  ImGuiIO &io = ImGui::GetIO();
  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Right)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftShift));
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftAlt));
  const bool orbit =
      ImGui::IsMouseDown(ImGuiMouseButton_Left) && !dolly && !pan;

  // Re-baseline rotation on the rising edge of orbit (including resume after a
  // dolly/pan interlude in the same held drag), or the view jumps.
  if (orbit && !m_orbiting)
    m_manip.startNewRotation();
  m_orbiting = orbit;

  const float2 delta =
      float2(io.MouseDelta.x, io.MouseDelta.y) * 2.f / float2(m_size);
  if (delta == float2(0.f))
    return;
  if (orbit)
    m_manip.rotate(delta);
  else if (dolly)
    m_manip.zoom(delta.y);
  else if (pan)
    m_manip.pan(delta);
}
```

- [ ] **Step 4: Build**

Run: `cmake --build _out/_cmake --parallel`
Expected: builds clean. If it fails on `m_prevMouse`/`m_manipulating`/`m_rotating` still referenced, you missed a use — they must no longer appear anywhere in `GraphViewport.cpp`.

- [ ] **Step 5: Run the full test suite**

Run: `ctest --test-dir _out/_cmake --output-on-failure`
Expected: all tests pass (63), no regressions. (No test exercises this GUI code directly; this confirms nothing else broke.)

- [ ] **Step 6: Walk the manual smoke checklist against the code**

Confirm by reading the control flow that each holds (these are the human's on-screen acceptance criteria; verify the code path supports each):
- Pressing on the gizmo: `drawGizmo` returns true (`IsOver`/`IsUsing`) → the `if (!gizmoActive)` branch is skipped → no InvisibleButton → ImGuizmo keeps the drag, camera does not orbit.
- Dragging the window title bar: the press never lands on the InvisibleButton → `IsItemActive()` is false → `handleNavigation` early-returns → no rotation.
- Press inside the viewport then drag outside: the button stays active (ImGui holds active id until release) so the drag continues, but a press that *starts* outside never activates it. (This matches intended "drag started in viewport keeps control" semantics.)
- Orbit (left), pan (middle or left+Alt), dolly (right or left+Shift), wheel zoom all still function when the press starts on the image and the gizmo is not under the cursor; wheel also works while merely hovering the gizmo (handled in `buildUI`, gated only by `!ImGuizmo::IsUsing()`).

- [ ] **Step 7: Commit**

```bash
jj commit tsd/src/tsd/ui/imgui/windows/GraphViewport.hpp tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp -m "fix(tsdflow): route viewport input so the gizmo captures drags and nav is scoped to in-viewport presses"
```

---

### Task 2: Component B — GraphEditor compact nodes with proxy pins

**Files:**
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp` (add `m_expanded`, `isCollapsed`, `resolvePort`)
- Modify: `tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp` (`drawNode`, link loop, double-click toggle, `handleDeletion`, `handleCreation`)

**Interfaces:**
- Consumes (existing, unchanged): `GraphEditor::pinId(NodeId, Token, bool)` returns `int` (stable imnodes pin id, index+1, appends to `m_pins` on first sight); `struct PinKey { NodeId node; Token port; bool isInput; }`; `nodeImId(NodeId)`; `m_graph->node(id)->impl->typeInfo()` returns a `NodeTypeInfo` **by value** with `std::vector<PortSpec> inputs, outputs;` where `PortSpec` has a `.name` of type `tsd::core::Token`; `m_model->canConnect/connect/classify`; `m_positioned`; file constant `kConversionColor`; `using namespace tsd::graph;` and `using tsd::core::Token;` are already in the .cpp.
- Produces: none (self-contained window behavior change).

**Context:** `GraphEditor` is the imnodes canvas. Today every node renders title + all per-port pins. This task makes nodes collapsed-by-default (title + one proxy pin per side), expandable on double-click. Proxy pins use sentinel port tokens `Token("##in")`/`Token("##out")` (no real port is named that), routed through the existing `pinId` registry so links to/from a collapsed node stay drawn on its stubs.

**Note on proxy tokens:** construct `Token("##in")`/`Token("##out")` inline at each use (matching the codebase pattern of inline `Token(...)` in BoundingBox/TransferFunction/DisplayMask) — do **not** introduce namespace-scope `const Token` constants (static-init-order risk against the intern pool).

- [ ] **Step 1: Add state + helper declarations to the header**

In `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp`, add two private method declarations next to the existing `drawNode` declaration:

```cpp
  void drawNode(tsd::graph::NodeId);
  bool isCollapsed(tsd::graph::NodeId) const;
  // Map a (possibly proxy) pin to a concrete port; false (logged) if the
  // collapsed node's direction has zero or multiple ports.
  bool resolvePort(const PinKey &pin, tsd::core::Token &outPort) const;
```

and add the expanded-set member next to `m_positioned`:

```cpp
  std::set<tsd::graph::NodeId> m_positioned; // nodes already given a position
  std::set<tsd::graph::NodeId> m_expanded;   // collapsed unless present here
```

(`<set>` is already included.)

- [ ] **Step 2: Add `isCollapsed` and rewrite `drawNode`**

In `tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp`, replace the entire current `drawNode`:

```cpp
void GraphEditor::drawNode(NodeId id)
{
  const auto *gn = m_graph->node(id);
  if (!gn || !gn->impl)
    return;
  const auto info = gn->impl->typeInfo();

  ImNodes::BeginNode(nodeImId(id));

  ImNodes::BeginNodeTitleBar();
  ImGui::TextUnformatted(info.name.c_str());
  ImNodes::EndNodeTitleBar();

  for (const auto &in : info.inputs) {
    ImNodes::BeginInputAttribute(
        pinId(id, in.name, true), ImNodesPinShape_CircleFilled);
    ImGui::TextUnformatted(in.name.c_str());
    ImNodes::EndInputAttribute();
  }
  for (const auto &out : info.outputs) {
    ImNodes::BeginOutputAttribute(
        pinId(id, out.name, false), ImNodesPinShape_TriangleFilled);
    ImGui::TextUnformatted(out.name.c_str());
    ImNodes::EndOutputAttribute();
  }

  ImNodes::EndNode();
}
```

with `isCollapsed` plus a `drawNode` that branches on it. The collapsed branch deliberately does **not** call `BeginNodeTitleBar` (imnodes forces the title bar first and `EndNodeTitleBar` resets the cursor below it, which would split the proxy pins onto separate rows); it renders the name flanked by two empty proxy attributes on one `SameLine` row so the pins anchor to the node's left/right edges:

```cpp
bool GraphEditor::isCollapsed(NodeId id) const
{
  return m_expanded.find(id) == m_expanded.end();
}

void GraphEditor::drawNode(NodeId id)
{
  const auto *gn = m_graph->node(id);
  if (!gn || !gn->impl)
    return;
  const auto info = gn->impl->typeInfo();

  ImNodes::BeginNode(nodeImId(id));

  if (isCollapsed(id)) {
    const bool hasIn = !info.inputs.empty();
    const bool hasOut = !info.outputs.empty();
    if (hasIn) {
      ImNodes::BeginInputAttribute(
          pinId(id, Token("##in"), true), ImNodesPinShape_CircleFilled);
      ImNodes::EndInputAttribute();
      ImGui::SameLine();
    }
    ImGui::TextUnformatted(info.name.c_str());
    if (hasOut) {
      ImGui::SameLine();
      ImNodes::BeginOutputAttribute(
          pinId(id, Token("##out"), false), ImNodesPinShape_TriangleFilled);
      ImNodes::EndOutputAttribute();
    }
  } else {
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(info.name.c_str());
    ImNodes::EndNodeTitleBar();

    for (const auto &in : info.inputs) {
      ImNodes::BeginInputAttribute(
          pinId(id, in.name, true), ImNodesPinShape_CircleFilled);
      ImGui::TextUnformatted(in.name.c_str());
      ImNodes::EndInputAttribute();
    }
    for (const auto &out : info.outputs) {
      ImNodes::BeginOutputAttribute(
          pinId(id, out.name, false), ImNodesPinShape_TriangleFilled);
      ImGui::TextUnformatted(out.name.c_str());
      ImNodes::EndOutputAttribute();
    }
  }

  ImNodes::EndNode();
}
```

- [ ] **Step 3: Substitute proxy pin ids in the link loop**

In `GraphEditor::buildUI`, replace the current link-loop body:

```cpp
  for (const auto &c : m_graph->connections()) {
    const int lid = linkCounter++;
    m_linkId[lid] = c.id;
    const bool conv =
        m_model->classify(c) == tsd::graph_nodes::LinkKind::Conversion;
    if (conv)
      ImNodes::PushColorStyle(ImNodesCol_Link, kConversionColor);
    ImNodes::Link(lid,
        pinId(c.fromNode, c.fromPort, false),
        pinId(c.toNode, c.toPort, true));
    if (conv)
      ImNodes::PopColorStyle();
  }
```

with (only the endpoint-id computation changes; the conversion-color styling is preserved verbatim):

```cpp
  for (const auto &c : m_graph->connections()) {
    const int lid = linkCounter++;
    m_linkId[lid] = c.id;
    const bool conv =
        m_model->classify(c) == tsd::graph_nodes::LinkKind::Conversion;
    if (conv)
      ImNodes::PushColorStyle(ImNodesCol_Link, kConversionColor);
    const int fromPin = isCollapsed(c.fromNode)
        ? pinId(c.fromNode, Token("##out"), false)
        : pinId(c.fromNode, c.fromPort, false);
    const int toPin = isCollapsed(c.toNode)
        ? pinId(c.toNode, Token("##in"), true)
        : pinId(c.toNode, c.toPort, true);
    ImNodes::Link(lid, fromPin, toPin);
    if (conv)
      ImNodes::PopColorStyle();
  }
```

- [ ] **Step 4: Add the double-click toggle after `EndNodeEditor`**

In `GraphEditor::buildUI`, the current post-editor section reads:

```cpp
  contextMenu();
  ImNodes::MiniMap();
  ImNodes::EndNodeEditor();

  // After EndNodeEditor: creation, deletion, selection.
  handleCreation();
  handleDeletion();
```

Insert the toggle between `EndNodeEditor()` and `handleCreation()`:

```cpp
  contextMenu();
  ImNodes::MiniMap();
  ImNodes::EndNodeEditor();

  // Double-click a node to toggle compact/expanded. IsNodeHovered returns the
  // topmost node under the cursor, so this targets the right node when stacked.
  int hoveredNode = 0;
  if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)
      && ImNodes::IsNodeHovered(&hoveredNode)) {
    const NodeId id = NodeId(hoveredNode);
    if (m_expanded.count(id))
      m_expanded.erase(id);
    else
      m_expanded.insert(id);
  }

  // After EndNodeEditor: creation, deletion, selection.
  handleCreation();
  handleDeletion();
```

- [ ] **Step 5: Clear expanded state on node deletion**

In `GraphEditor::handleDeletion`, the node-removal block currently reads:

```cpp
      m_model->removeNode(id);
      m_positioned.erase(id);
      *m_graphDirty = true;
```

Add the `m_expanded.erase(id)` line:

```cpp
      m_model->removeNode(id);
      m_positioned.erase(id);
      m_expanded.erase(id);
      *m_graphDirty = true;
```

- [ ] **Step 6: Add `resolvePort` and use it in `handleCreation`**

Add `resolvePort` (place it just above `handleCreation`):

```cpp
bool GraphEditor::resolvePort(const PinKey &pin, Token &outPort) const
{
  const Token proxy = pin.isInput ? Token("##in") : Token("##out");
  if (pin.port != proxy) { // already a concrete port
    outPort = pin.port;
    return true;
  }
  const auto *gn = m_graph->node(pin.node);
  if (!gn || !gn->impl)
    return false;
  const auto info = gn->impl->typeInfo(); // by value — copy the Token out
  const auto &ports = pin.isInput ? info.inputs : info.outputs;
  if (ports.size() != 1) {
    tsd::core::logWarning("[GraphEditor] expand node to choose a port");
    return false;
  }
  outPort = ports.front().name;
  return true;
}
```

Then in `handleCreation`, replace the connect tail. Current:

```cpp
  PinKey *outPin = a->isInput ? b : a;
  PinKey *inPin = a->isInput ? a : b;
  if (outPin->isInput || !inPin->isInput)
    return; // not an out->in pairing

  auto chk =
      m_model->canConnect(outPin->node, outPin->port, inPin->node, inPin->port);
  if (!chk.ok()) {
    tsd::core::logWarning(
        "[GraphEditor] link rejected: %s", chk.detail.c_str());
    return;
  }
  m_model->connect(outPin->node, outPin->port, inPin->node, inPin->port);
  *m_graphDirty = true;
```

Replace with (resolve proxy endpoints to concrete ports first):

```cpp
  PinKey *outPin = a->isInput ? b : a;
  PinKey *inPin = a->isInput ? a : b;
  if (outPin->isInput || !inPin->isInput)
    return; // not an out->in pairing

  Token outPort, inPort;
  if (!resolvePort(*outPin, outPort) || !resolvePort(*inPin, inPort))
    return; // ambiguous proxy on a multi-port collapsed node — expand first

  auto chk = m_model->canConnect(outPin->node, outPort, inPin->node, inPort);
  if (!chk.ok()) {
    tsd::core::logWarning(
        "[GraphEditor] link rejected: %s", chk.detail.c_str());
    return;
  }
  m_model->connect(outPin->node, outPort, inPin->node, inPort);
  *m_graphDirty = true;
```

- [ ] **Step 7: Build**

Run: `cmake --build _out/_cmake --parallel`
Expected: builds clean.

- [ ] **Step 8: Run the full test suite**

Run: `ctest --test-dir _out/_cmake --output-on-failure`
Expected: all tests pass (63), no regressions.

- [ ] **Step 9: Walk the manual smoke checklist against the code**

Confirm the control flow supports each (human's on-screen acceptance criteria):
- All nodes start compact: `m_expanded` is empty initially → `isCollapsed` true for every id → the collapsed branch renders title + stub pins only.
- Double-click toggles: a stationary double-click over a node flips `m_expanded`; imnodes only drags a node when the mouse moves while held, so a clean double-click does not nudge it. The first click also selects the node (intended — drives the Inspector).
- Links to/from a collapsed node stay visible: the link loop substitutes the node's `##in`/`##out` proxy pin id, which the collapsed branch submitted this frame.
- Wire a collapsed single-port node: dragging from its stub creates a `PinKey` with the proxy token; `resolvePort` maps it to the sole port and `connect` proceeds.
- Multi-port collapsed node (e.g. `DisplayVolume`, inputs `field`+`tf`): `resolvePort` sees `ports.size() != 1`, logs the hint, and rejects — you must expand to choose a port.
- Deleting a node clears its expanded flag (Step 5), so a recycled `NodeId` is not stale-expanded.

- [ ] **Step 10: Commit**

```bash
jj commit tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp -m "feat(tsdflow): compact-by-default nodes with title-bar proxy pins and double-click expand"
```

---

## Self-Review

**Spec coverage:**
- Component A reorder (blit → gizmo → conditional button) → Task 1 Step 2. ✓
- `handleNavigation` rewrite off `IsItemActive` + rising-edge `m_orbiting` latch → Task 1 Steps 1, 3. ✓
- Wheel zoom in `buildUI`, `IsWindowHovered` + `!IsUsing` → Task 1 Step 2. ✓
- Remove `m_prevMouse`/`m_manipulating`/`m_rotating`, add `m_orbiting` → Task 1 Step 1. ✓
- Component B collapsed-by-default via `m_expanded` + `isCollapsed` → Task 2 Steps 1, 2. ✓
- Collapsed layout drops title bar, one-row proxy pins → Task 2 Step 2. ✓
- Link-loop proxy substitution, conversion color preserved → Task 2 Step 3. ✓
- Double-click toggle via `IsNodeHovered` → Task 2 Step 4. ✓
- Delete cleanup `m_expanded.erase` → Task 2 Step 5. ✓
- Proxy→sole-port resolution, multi-port reject, `typeInfo()` by-value → Task 2 Step 6. ✓
- Inline `Token("##in")`/`Token("##out")` (no static-init-order risk) → Task 2 note + Steps 2, 3, 6. ✓

**Placeholder scan:** none — every code step carries complete code.

**Type consistency:** `isCollapsed(NodeId) const` and `resolvePort(const PinKey&, Token&) const` declared (Step 1) and defined (Steps 2, 6) with matching signatures; `m_orbiting` declared (Task 1 Step 1) and used (Steps 2, 3); proxy tokens `"##in"`/`"##out"` consistent across drawNode, link loop, and resolvePort; `pinId`/`PinKey`/`typeInfo()` usages match the existing declarations.
