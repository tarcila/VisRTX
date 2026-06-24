# tsdFlow: Viewport Input Routing & Compact Nodes — Design

**Date:** 2026-06-24
**Status:** Approved (pending user spec review)
**Phase:** 4i (interaction polish, post-4h)

## Problem

Three interaction defects/refinements in the tsdFlow app, surfaced after Phase 4h:

1. **Stray camera capture.** `GraphViewport` orbits/pans the scene for drags that did not start on the viewport image — dragging the window title bar (to move/dock the window) rotates the scene, and a drag that leaves the viewport keeps rotating.
2. **Dead manipulator.** The ImGuizmo transform gizmo (shipped in 4h) highlights on hover but does not drag: pressing on it orbits the camera instead of moving the object.
3. **Cluttered node graph.** Every `GraphEditor` node renders title + all per-port pins always. Nodes should be **compact** (title only) by default and expand to show pins on double-click.

## Root Cause (1 & 2 — same bug)

`GraphViewport::buildUI` reserves the viewport with `ImGui::InvisibleButton("##viewport", …)` **before** blitting the image and **before** `drawGizmo` runs `ImGuizmo::Manipulate`. That button covers the whole viewport and grabs ImGui's `ActiveId` on any press inside the rect:

- **#2:** the button swallows the press before ImGuizmo (called later) can capture it. The highlight works because it is hover-only (`ImGuizmo::IsOver`); the drag is stolen → camera orbits.
- **#1:** `handleNavigation` decides orbit/pan/dolly from **global** `ImGui::IsMouseDown(...)` plus a sticky `m_manipulating` flag cleared only on button-release. Nothing ties the motion to "the press landed on the viewport," so title-bar drags and drags that leave the rect keep manipulating.

## Component A — Viewport Input Routing (fixes #1 + #2)

**File:** `tsd/src/tsd/ui/imgui/windows/GraphViewport.cpp` (+ `.hpp` state removal).

Reorder `buildUI` so the gizmo and camera nav cannot both claim the mouse, and scope navigation to genuine in-viewport drags.

### New per-frame order

After the existing size/camera/render block produces the texture:

1. Compute `pos = ImGui::GetCursorScreenPos()` and `imgSize`.
2. Blit the texture with `ImGui::GetWindowDrawList()->AddImage(...)` at `pos` (draw-list only — does not advance the cursor or claim input). The cursor remains at `pos`.
3. Call `drawGizmo(pos, imgSize)` — it runs `ImGuizmo::Manipulate`, which hit-tests and captures input internally. Returns `ImGuizmo::IsUsing() || ImGuizmo::IsOver()`.
4. **If `!gizmoActive`:** create `ImGui::InvisibleButton("##viewport", imgSize, <L|R|M>)` at `pos` (cursor is still there), then call `handleNavigation()` (drag nav only). **Else:** skip both — the gizmo owns this frame's input, and no button exists to steal the press.
5. Handle wheel zoom (see "Wheel zoom" below) — always, independent of step 4, so it works while hovering the gizmo.

`drawGizmo` is unchanged (it already builds the view/proj manually, calls `ImGuizmo::BeginFrame/SetDrawlist/SetRect/Manipulate`, writes `itf->transform()` on edit, sets `*m_graphDirty`).

### `handleNavigation` rewrite

Drop the global-mouse + sticky-flag machinery; drive everything off the button's own ImGui item state (the button exists only when this runs, since step 4 gates on `!gizmoActive`):

- **Activation edge:** on `ImGui::IsItemActivated()` (press landed on the button this frame), reset per-gesture state and, for an orbit gesture, call `m_manip.startNewRotation()`.
- **Active drag:** while `ImGui::IsItemActive()` (held after a press on the button — title-bar drags and out-of-rect presses never qualify), read the frame delta from `ImGui::GetIO().MouseDelta`, normalize by `2.f / float2(m_size)` (matching the prior model), and dispatch:
  - dolly: right button down, or left+`LeftShift`;
  - pan: middle button down, or left+`LeftAlt`;
  - orbit: left button down and not dolly/pan.
  Button-modifier reads use `ImGui::IsMouseDown(...)`/`ImGui::IsKeyDown(...)` **only to classify the gesture while the item is active** — they no longer gate whether manipulation happens.

**`startNewRotation()` latch (rising-edge, re-arms on resume):** keep a single `bool m_orbiting`. Fire `m_manip.startNewRotation()` on the rising edge of orbit-active (`orbit && !m_orbiting`), set `m_orbiting = orbit` each frame the item is active, and clear it when the item is not active. This re-arms a fresh rotation baseline whenever orbit *resumes* within one held gesture (e.g. orbit → hold Shift to dolly → release Shift back to orbit) — matching the current `m_rotating` behavior exactly, just keyed to item-active instead of global mouse. (The spec deliberately does **not** want a once-per-hold latch; resuming orbit after a dolly/pan interlude must re-baseline or the view jumps.)

### Wheel zoom (handled in `buildUI`, not `handleNavigation`)

Wheel zoom must keep working even when the cursor merely *hovers* the gizmo (ImGuizmo ignores the wheel). Because the InvisibleButton is not submitted while the gizmo is hot, `IsItemHovered()` cannot gate it. Handle the wheel in `buildUI`, independent of the button:

- while `ImGui::IsWindowHovered()` and `!ImGuizmo::IsUsing()` and `io.MouseWheel != 0`, call `m_manip.zoom(io.MouseWheel * kWheelZoomScale)`.

The `!ImGuizmo::IsUsing()` guard suppresses zoom only during an active gizmo drag, not on mere hover.

### State removed from `GraphViewport.hpp`

Delete the now-unused navigation state: `m_prevMouse`, `m_manipulating`, `m_rotating`. Add a single `bool m_orbiting{false}` for the rising-edge latch above.

### Alternative considered (rejected)

Keep the button always-present and feed ImGuizmo first via `ImGui::PushID`/a separate draw layer. Rejected: conditionally skipping the button when the gizmo is hot is simpler, has no ID-stack juggling, and is the documented pattern for ImGuizmo inside a custom ImGui viewport.

## Component B — Compact Nodes with Proxy Pins (#3)

**File:** `tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp` (+ `.hpp` for state).

### Collapsed state

Add `std::set<tsd::graph::NodeId> m_expanded;` to `GraphEditor`. **A node is collapsed unless it appears in `m_expanded`.** This makes all nodes — the demo graph and every newly-created node — compact by default with no per-node initialization. Define a helper `bool isCollapsed(NodeId id) const { return !m_expanded.count(id); }`.

### Proxy pin tokens

Collapsed nodes route every connection through a single title-bar input stub and a single output stub. Reserve two sentinel port tokens:

- `kProxyIn = Token("##in")`
- `kProxyOut = Token("##out")`

These flow through the existing `pinId(node, port, isInput)` registry unchanged (they are just two more `PinKey`s), so each collapsed node gets one stable proxy-in id and one stable proxy-out id.

### `drawNode` — collapsed branch

**No `BeginNodeTitleBar` in the collapsed branch.** imnodes requires the title bar to be submitted *first* and `EndNodeTitleBar` resets the cursor to the node content origin *below* the title (`imnodes.cpp` `EndNodeTitleBar`/`GetNodeContentOrigin`). A `SameLine` before or after the title bar is therefore defeated, and proxy-in/proxy-out would land on separate rows above/below the title at different vertical heights. Collapsed nodes drop the colored title bar and render the name as a plain item flanked by the two empty proxy attributes on **one row**:

```
BeginNode(nodeImId(id))
if collapsed:
    hasIn  = !info.inputs.empty()
    hasOut = !info.outputs.empty()
    if hasIn:  BeginInputAttribute(pinId(id, kProxyIn, true), CircleFilled); EndInputAttribute(); SameLine()
    ImGui::TextUnformatted(info.name.c_str())
    if hasOut: SameLine(); BeginOutputAttribute(pinId(id, kProxyOut, false), TriangleFilled); EndOutputAttribute()
else:
    <current full rendering: BeginNodeTitleBar + title + per-input attrs + per-output attrs>
EndNode()
```

imnodes anchors an input attribute's pin X to the node's left edge and an output attribute's pin X to the right edge regardless of inline content (pin X uses `node_rect.Min.x`/`Max.x`). With proxy-in, the title text, and proxy-out on one `SameLine` row, the two pins sit on opposite edges of the compact chip. A node with no inputs (e.g. `GenerateNoiseVolume`) draws no proxy-in; a sink node with no outputs (e.g. `DisplayVolume`/`DisplaySurface`) draws no proxy-out.

Cosmetic note: the empty proxy attribute groups have ~0 height, so each pin's Y is the row top rather than the text vertical center. Both pins share that Y, so they stay aligned with each other — acceptable for a compact chip. (Optional polish, not required: pad each proxy attribute with a zero-width `Dummy` of the text height to center the pins.)

### Link loop — substitute proxy ids for collapsed endpoints

In the `buildUI` link loop, when an endpoint node is collapsed, target its proxy pin instead of the per-port pin:

```
fromPin = isCollapsed(c.fromNode) ? pinId(c.fromNode, kProxyOut, false)
                                  : pinId(c.fromNode, c.fromPort, false)
toPin   = isCollapsed(c.toNode)   ? pinId(c.toNode,   kProxyIn,  true)
                                  : pinId(c.toNode,   c.toPort,  true)
ImNodes::Link(lid, fromPin, toPin)
```

Only the endpoint-id computation changes — the rest of the existing link loop is preserved verbatim, including the `m_model->classify(c)` conversion check and its `ImNodes::PushColorStyle`/`PopColorStyle(ImNodesCol_Link)` styling. Existing links to/from a collapsed node stay drawn, merged onto its proxy stubs. Multiple links converging on one proxy pin render fine (imnodes draws each `Link` independently).

### Toggle on double-click

After `ImNodes::EndNodeEditor()` (where the other post-frame handlers run), detect a double-click on a hovered node and flip its collapsed state:

```
int hoveredNode = 0;
if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)
        && ImNodes::IsNodeHovered(&hoveredNode)) {
    const NodeId id = NodeId(hoveredNode);
    if (m_expanded.count(id)) m_expanded.erase(id); else m_expanded.insert(id);
}
```

(`ImNodes::IsNodeHovered(int*)` exists in the vendored imnodes v0.5; it returns the topmost node under the cursor, so the toggle targets the right node when nodes overlap.)

**Interaction precedence (intended, not a bug):** the first click of the double-click also selects the node via imnodes — this is *desired*, since selection drives the Inspector and the gizmo's target. imnodes only begins a node drag when the mouse *moves* while held, so a stationary double-click toggles collapse without nudging the node. No suppression of imnodes' own click handling is needed.

### Cleanup on delete

In `handleDeletion`, alongside the existing `m_positioned.erase(id)`, add `m_expanded.erase(id)`. Because collapsed is the default (`!m_expanded.count(id)`), the only stale state a recycled `NodeId` could inherit is a *previously-expanded* flag; this erase prevents that.

### Link creation from a proxy pin

`handleCreation` resolves attr ids to `PinKey`s. A proxy `PinKey` has port `kProxyIn`/`kProxyOut`, which is not a real port. Resolve it to the node's **sole** port for that direction via `typeInfo()`:

- If the dragged endpoint's `PinKey.port` is a proxy token, look up the node's `typeInfo().inputs` (for `kProxyIn`) or `.outputs` (for `kProxyOut`). Note `typeInfo()` returns a `NodeTypeInfo` **by value** — copy the resolved port Token out; do not retain a pointer into the temporary.
- If exactly one port exists on that side, substitute its real port Token and proceed with the existing `canConnect`/`connect` path.
- If more than one exists, reject the link and `logWarning("[GraphEditor] expand node to choose a port")`.

Multi-port nodes are a **real case**, not theoretical: `DisplayVolume` has two inputs (`field`, `tf`). Connecting such a node while collapsed is ambiguous, so it requires expansion first — consistent with "expand to edit connections." Single-port nodes (the common case) wire directly from the proxy. `expand first is fine` (confirmed).

## Data Flow & Boundaries

- Component A is self-contained in `GraphViewport` (one window's input handling). No bridge, graph, or node changes.
- Component B is self-contained in `GraphEditor` (one window's rendering + imnodes id mapping). No graph-engine, model, or node changes; `GraphEditModel::canConnect`/`connect` are reused as-is.
- The two components share no code and can be implemented and reviewed independently.

## Testing

Both components are ImGui/imnodes interaction code with no pure-logic seam the existing unit harness exercises. Verification is the **build staying green + the full test suite staying green with no regressions** (currently 63 tests) plus a **manual smoke checklist**:

**Component A**
- Drag the gizmo arrows → the object translates; the camera does not orbit.
- Drag the window title bar → the scene does not rotate.
- Press inside the viewport and drag outside it → rotation stops at the edge / never starts from outside.
- Orbit / pan (middle or left+Alt) / dolly (right or left+Shift) / wheel-zoom still work when the press starts on the image and the gizmo is not under the cursor.

**Component B**
- All nodes start compact (title + stub pins only).
- Double-click a node → expands to full pins; double-click again → collapses.
- Links to/from a collapsed node remain visible, merged onto the title-bar stubs.
- Drag a link from a collapsed single-port node's stub → connection is created (resolves to the sole port).
- Newly-created nodes (context menu) appear collapsed; deleting a node clears its expanded state.

No new unit test is written, because none would assert behavior the harness can drive — stating this explicitly rather than adding a hollow test.

## Out of Scope

- Persisting collapsed/expanded state across sessions (Phase 5 persistence).
- Gizmo operation/mode key toggles (translate/rotate/scale, world/local) — deferred from 4h, still deferred.
- Animating the collapse/expand transition.
- Proxy wiring that disambiguates multi-port nodes without expanding.
