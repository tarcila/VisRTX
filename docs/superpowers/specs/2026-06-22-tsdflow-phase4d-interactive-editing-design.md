# tsdFlow Phase 4d — Interactive Editing Design

**Status:** approved (brainstorm), pending implementation plan
**Date:** 2026-06-22
**Depends on:** Phase 4a (node catalog), Phase 4c (app shell + viewports)
**Precedes:** Phase 4b (CUDA residency), Phase 4e (undo/redo), Phase 5 (Lua + persistence)

## Goal

Turn `tsdFlow` from a fixed-demo viewer into a user-editable node graph: a visual
node-graph editor, a selection-driven parameter inspector, and a node-bound
interactive transfer-function editor. The user builds and rewires processing
pipelines on a canvas and tunes them live, with results re-rendered through the
Phase 4c `GraphRenderBridge`.

## Decisions (from brainstorm)

| # | Decision |
|---|----------|
| Q1 | Node editor mirrors the in-repo viskores `NodeEditor` pattern on vendored `tsd_imnodes` (no new dependency). |
| Q2 | Full editing: add nodes (catalog menu), delete nodes, create/delete connections by pin drag, edit parameters. |
| Q3 | Validate **and** visualize: reject incompatible/cyclic links; show implicit **conversion** feedback on accepted links. Transfer/residency feedback deferred to 4b. |
| Q4 | Reuse the existing `TransferFunctionEditor` curve/palette internals, rebound to the `TransferFunction` node; full interactive curve editing; file load/save deferred to Phase 5. |
| Q5 | Separate, selection-driven **Inspector** panel (canvas nodes show title + pins only). |
| Q6 | Extract a UI-free `GraphEditModel` logic layer, unit-tested; windows are thin views. |
| Q7 | Inspector uses plain (unbounded) widgets dispatched on `Any::type()`; bounded sliders deferred (engine has no min/max metadata). |
| Q8 | TF editor ships **full** interactive color **and** opacity point editing — interactive color editing is new work (the existing editor does opacity only). |

## Architecture

Two new UI windows (and one embedded panel) over a tested, UI-free editor-logic
core, wired into the existing `tsdFlow` app and `tsd_graph`.

```
tsd_graph  (Graph, NodeRegistry, ConversionRegistry, TransferRegistry, Evaluator)
   ▲
   │  pure ops + queries (no ImGui)
GraphEditModel ──────────────── tsd_graph_nodes (new: GraphEditModel.hpp/.cpp)   [Catch2-tested]
   ▲                ▲
   │ thin views     │
GraphEditor      Inspector ── embeds ──▶ TFCurveEditor   (tsd_ui_imgui)          [build + manual]
 (imnodes canvas) (selection→params)     (node-bound TF panel)
   │
tsdFlow app: GraphEditor + Inspector + 2 GraphViewports + Log                    [manual checklist]
```

`GraphEditor` and `Inspector` are dockable `Window`s; `TFCurveEditor` is a
reusable **panel** (not a standalone window) that the `Inspector` renders inline
when a `TransferFunction` node is selected.

- **`GraphEditModel`** (core-only, in `tsd_graph_nodes`): wraps `Graph&` +
  `NodeRegistry&` + `ConversionRegistry*`. Exposes edit ops, non-mutating
  validation/classification queries, the node catalog, and pure TF sampling.
  After any mutating op it calls `graph.markDirty`. Headless core (Q6).
- **`GraphEditor`** window (`tsd_ui_imgui`): imnodes canvas mirroring the viskores
  `NodeEditor` — draws nodes/pins/links from `Graph`, routes drags + context menu
  through `GraphEditModel`, colors links per classification (Q1/Q2/Q3).
- **`Inspector`** window: renders the selected node's `ParameterList` with the
  viskores `editor_NodeParameter` widget set; embeds `TFCurveEditor` for a
  `TransferFunction` node (Q5).
- **`TFCurveEditor`**: reused `TransferFunctionEditor` curve/palette internals,
  rebound to a node's typed transfer-function state (Q4).
- **`tsdFlow`** gains these windows; the 4c viewports, Log, and "Regenerate" stay.
  Selection is single app-level state (`NodeId`), shared by canvas ↔ inspector.

## Component: `GraphEditModel` (tested core)

Located in `tsd_graph_nodes` (core-only, no ImGui). Every mutating op marks the
graph dirty; the app then triggers `bridge.update()` for live re-render (the
async evaluator from Phase 2 already exists).

```cpp
namespace tsd::graph_nodes {

enum class LinkKind { Direct, Conversion, Incompatible, Cycle };

struct ConnectCheck {
  LinkKind kind{LinkKind::Incompatible};
  std::string detail;            // "float→double" (conversion) or reject reason
  bool ok() const { return kind == LinkKind::Direct || kind == LinkKind::Conversion; }
};

class GraphEditModel {
 public:
  GraphEditModel(graph::Graph &, graph::NodeRegistry &, const graph::ConversionRegistry *);

  // Mutating ops — each marks dirty. Thin wrappers over Graph.
  graph::NodeId     addNode(core::Token type);   // registry.create → graph.addNode
  void              removeNode(graph::NodeId);
  graph::LinkResult connect(graph::NodeId, core::Token, graph::NodeId, core::Token);
  void              disconnect(graph::ConnectionId);

  // Non-mutating — drives pin-drag coloring + existing-link coloring + tooltips.
  ConnectCheck canConnect(graph::NodeId, core::Token, graph::NodeId, core::Token) const;
  LinkKind     classify(const graph::Connection &) const;

  const std::vector<core::Token> &nodeCatalog() const;   // from NodeRegistry::types() (new API)

  // Pure TF sampling — control points → sampled RGBA colormap (unit-tested).
  // ColorPoint is {position, r, g, b} (`.x` = position, NOT alpha — the
  // ColorMapUtil.hpp doc comment is misleading); OpacityPoint is {position,
  // opacity}. Each output float4 = (RGB from the color curve, opacity from the
  // opacity curve), composed from core::detail::interpolateColor (returns float3)
  // + core::detail::interpolateOpacity (returns float). Those helpers are in the
  // `detail` namespace today — qualify as `core::detail::…` or promote them.
  static std::vector<math::float4> sampleColormap(
      const std::vector<core::ColorPoint> &,
      const std::vector<core::OpacityPoint> &, int samples);
};
} // namespace tsd::graph_nodes
```

### Sub-decision (a): three additive engine APIs

The editor needs three small, additive (non-breaking) engine APIs that don't
exist yet. (4c's "no engine changes" constraint does not apply to 4d.)

1. **`Graph::canConnect(NodeId, Token, NodeId, Token) const → LinkResult`** —
   validation single-sourcing. `connect()` already runs the type-compat + cycle
   checks (`wouldCreateCycle`, `findOutputSpec`/`findInputSpec`, all already
   `const`) and returns `LinkResult{ok,id,reason}`, but only at mutate time. The
   editor needs the *same* logic as a non-mutating pre-flight (to color a link
   while dragging). Extract a public `canConnect` from the head of `connect()`
   (everything before the first mutation); `connect()` then calls it and proceeds
   only if `ok` (allocating the `ConnectionId` itself — `canConnect` returns
   `INVALID_CONNECTION` for `id`). This avoids the duplication Q6 exists to prevent.

2. **`Graph` node enumeration** — `Graph` exposes `node(NodeId)`, `connections()`,
   `inputConnection(...)` but **no way to iterate nodes** (`m_nodes` is private).
   The `GraphEditor`'s "draw every node" loop has nothing to walk. Add a public
   accessor (e.g. `std::vector<NodeId> nodeIds() const`, or a const range over
   `m_nodes`).

3. **`NodeRegistry::types() const → std::vector<Token>`** — `NodeRegistry` exposes
   only `registerType`/`create`/`isRegistered` (`m_entries` private). The add-node
   menu (`nodeCatalog()`) needs the list of registered type names.

`GraphEditModel::canConnect` wraps (1) and synthesizes the tooltip **detail**
string itself as `"<from.name>→<to.name>"` from the two `PortType.name` tokens —
`ConversionEntry` has **no name field**, so there is no stored conversion name to
read.

### Sub-decision (b): conversion feedback now, transfer feedback deferred

Link classification from the **static graph**:
- exact `PortType` match → `Direct`
- `ConversionRegistry::find(from,to)` hit → `Conversion` (+ name for tooltip)
- miss → `Incompatible` (link rejected)
- back-edge → `Cycle` (link rejected)

`Incompatible`/`Cycle` never exist as committed links — they are rejected at
creation. **Residency/transfer** feedback ("host→device") is *not* knowable
pre-eval: residency is chosen by nodes at evaluate time, and 4d is host-only (no
transfers occur until 4b adds CUDA). So 4d ships **conversion** link feedback;
**transfer** link feedback is deferred to 4b, where it becomes real.

### Unit tests (`tsd_graph_nodes`)

- `canConnect`/`classify` matrix: direct, conversion (+ synthesized
  `"from→to"` detail), incompatible reject, cycle reject.
- `addNode`/`removeNode`/`disconnect` effects on the `Graph`.
- `nodeCatalog()` non-empty / contains the builtin types (exercises
  `NodeRegistry::types()`).
- `sampleColormap`: known control points → exact RGBA at endpoints/midpoints,
  opacity ramp, monotonic value mapping.

## Component: UI windows (`tsd_ui_imgui`, thin views)

### `GraphEditor` (imnodes canvas, follows the viskores `NodeEditor.cpp` pattern)

- **ID mapping (the editor owns it).** imnodes addresses everything by `int`;
  tsd identifies nodes by `NodeId` (`uint64_t`), pins by `(NodeId, port Token,
  direction)`, and links by `ConnectionId` (`uint64_t`) — there are no integer
  port ids in the model. The editor owns the bidirectional maps:
  `int ⟷ NodeId`, `int ⟷ (NodeId, port Token, in/out)`, and
  `int linkId ⟷ ConnectionId`, (re)built from `Graph::nodeIds()` /
  `connections()` each time the graph changes. This mapping layer is real
  GraphEditor work, not a one-liner.
- Each frame draw a node per `Graph::nodeIds()` (title bar + input pins left /
  output pins right, enumerated from the node's `NodeTypeInfo`). Pin shape is
  chosen from `PortType.name` (a Token — `PortType` is a struct, **not** an enum,
  so this is a token→shape lookup, unlike viskores's enum switch).
- **Add node:** right-click canvas → context menu listing `model.nodeCatalog()`
  → `model.addNode(type)` → `ImNodes::SetNodeScreenSpacePos` at the cursor.
- **Connect:** on `ImNodes::IsLinkCreated(&startPin,&endPin)`, decode both pins via
  the map, `model.canConnect(...)`; if `ok()`, `model.connect(...)`, else flash the
  reject reason to the Log.
- **Delete (nodes and links):** Delete key over the current imnodes selection —
  `NumSelectedNodes`/`GetSelectedNodes` → `model.removeNode(id)` (clearing app
  selection if it pointed there) and `NumSelectedLinks`/`GetSelectedLinks` →
  `model.disconnect(connId)`. This is the path the viskores editor actually uses;
  4d does **not** rely on the optional `IsLinkDestroyed` drag-detach callback.
- **Link coloring:** existing links push `ImNodes::PushColorStyle` by
  `model.classify()` — `Direct` neutral, `Conversion` amber + hover tooltip
  showing the synthesized `"from→to"` detail.
- **Selection:** writes the app's single selected `NodeId` (imnodes
  `NumSelectedNodes`/`GetSelectedNodes`).
- **Node positions:** in-memory `map<NodeId, ImVec2>` owned by the editor (imnodes
  keeps position in its own session ini); **not** persisted to disk in 4d
  (Phase 5).

### `Inspector` (selection-driven)

- Reads the app's selected `NodeId`; nothing selected → "No selection".
- Renders the node's `ParameterList` with a **new** widget renderer dispatching on
  `Any::type()` (the viskores `editor_NodeParameter` set is not reusable — it keys
  off viskores's typed `ParameterType` enum + `hasMinMax`/`scheduleParameterUpdate`,
  none of which `tsd::graph::ParameterList` — a `{Token, Any}` list — has). Widgets:
  `ANARI_BOOL`→checkbox, `ANARI_FLOAT32`→`InputFloat`, `ANARI_INT32`→`InputInt`,
  `ANARI_STRING`→input/combo. On edit: write back via `ParameterList::set` +
  `graph.markDirty(id)`.
- **No bounded sliders in 4d:** the engine has no min/max metadata on parameters,
  so all numeric params use plain inputs. Bounded-slider metadata is deferred (a
  later parameter-descriptor change).
- If the selected node is a `TransferFunction`, embed the `TFCurveEditor` panel in
  place of the generic parameter list.

## Component: `TransferFunction` node model + `TFCurveEditor`

`Any` cannot hold `ColorPoint`/`OpacityPoint` vectors (the `Any` type limits hit
in 4a), so control points cannot live in the generic `ParameterList`.

**Node model change:** `TransferFunction` gains typed state outside
`ParameterList`, exposed via a narrow interface the inspector downcasts to. The
state **reuses the existing `tsd::core::TransferFunction`** struct
(`{colorPoints, opacityPoints, range}`, ColorMapUtil.hpp) rather than redefining
it; only the sample count is added:

```cpp
// new public header: tsd/graph_nodes/TransferFunctionNode.hpp
struct ITransferFunctionNode {
  virtual ~ITransferFunctionNode() = default;
  virtual core::TransferFunction &tfState() = 0;   // colorPoints/opacityPoints/range
  virtual int  &samples() = 0;
};
```

The `TransferFunction` node (currently a struct in an **anonymous** namespace in
`TransferFunction.cpp`) must be moved into a **named** namespace so it implements
`ITransferFunctionNode` and is reachable across TUs; the inspector reaches it via
`dynamic_cast<ITransferFunctionNode*>(graph.node(id)->impl.get())` (safe — `Node`
has a virtual dtor; avoid `-fvisibility=hidden` traps by keeping it in the named
`tsd::graph_nodes` namespace, not anonymous).

- `evaluate()` samples the colormap from `tfState()` via
  `GraphEditModel::sampleColormap`, replacing the hardcoded cool-to-warm/grayscale
  formula. The existing `getPreset(params)` / `params.getOr<int>("samples", …)`
  reads in `evaluate()` are **removed** — `samples` and the curve now come from the
  typed state, not the `ParameterList` (don't leave both paths live).
- **Re-eval trigger:** TF state lives *outside* `ParameterList`, so
  `ParameterList::hash()` does **not** see curve edits. Re-eval is driven the
  normal way — the editor calls `graph.markDirty(id)` after a curve edit, which
  clears the node's cache; the evaluator recomputes on the empty cache. (Today the
  node never *declares* `preset`/`samples` in its `ParameterList` — it only reads
  them via `getOr` defaults — so there are no param entries to migrate, just the
  eval-time reads to redirect.)
- A **"Load preset"** action *populates* `colorPoints`/`opacityPoints` from a
  `ColorMapUtil` preset (the `tsd::core::colormap::*` globals — `viridis`,
  `cool_to_warm`, … are `std::vector<float3>` RGB, so positions are distributed
  evenly and alpha/opacity is seeded with a ramp). `core::makeDefaultTransferFunction()`
  can seed the initial node state. The `range` field is a `math::box1` (the TF
  panel's value axis maps over it), not a scalar pair.

**`TFCurveEditor`** (a panel, **not** a `Window`): the existing
`TransferFunctionEditor` is a `Window` tightly coupled to `scene::Volume`/`Array`/
`SDL_Texture` and interactively edits **opacity only** (color comes from presets).
4d's editor therefore reuses what genuinely transfers — palette-texture build,
the opacity-point widget, the preset dropdown — but **adds interactive color-point
editing as new work** (full color+opacity curve editing, per the Q4 decision). It
edits a node's `core::TransferFunction` (via `ITransferFunctionNode`), drops the
scene coupling, and calls `markDirty` on change. File load/save returns in Phase 5.

This keeps the generic inspector for all nodes and a specialized panel only where
the data is genuinely non-scalar.

## Data flow & live re-evaluation

Reuses the existing async evaluator and 4c bridge — no new evaluation machinery.

```
user edits (canvas drag / inspector widget / TF curve)
   → GraphEditModel op  OR  param write
   → graph.markDirty(nodeId)          [dirties node + transitive consumers]
   → app: bridge.update()             [pulls dirty display nodes, async]
   → GraphViewport renders bridge.world(i)   [next frame, as in 4c]
```

- **Selection** is single app-level state (`NodeId m_selected`): `GraphEditor`
  writes it, `Inspector`/`TFCurveEditor` read it. One source of truth.
- **Dirty coalescing:** many edits per frame collapse to one `bridge.update()` at
  frame end (an app dirty flag checked in the UI loop), not one per keystroke; the
  async evaluator's cancel-epoch (Phase 2) handles a newer edit superseding an
  in-flight eval.
- **Display masks** (display node → viewport) keep the 4c wiring; the editor does
  not change masks in 4d.
- **Display node add/remove — explicit bridge wiring required.** The bridge does
  **not** auto-prune: `GraphRenderBridge::update()` iterates its registered
  displays and `pull`s each `NodeId`, so a node erased via `removeNode` would leave
  a stale layer pulling a dead node. The app must therefore wire display-node
  removal to `bridge.removeDisplay(id)` before/with `removeNode`, and a newly added
  display node must be registered with an explicit `bridge.setDisplay(id, mask,
  true)` (4d defaults new display nodes to viewport 0's mask `0b01`). Per-display→
  viewport assignment UI is later work.

## Testing

**Catch2 (headless, `tsd/tests`):** (repo convention `test_nodes_*` / `test_graph_*`)
- `test_nodes_GraphEditModel.cpp` — validation/classification matrix; add/remove/
  disconnect effects; `nodeCatalog()`/`NodeRegistry::types()`.
- `test_nodes_TfnSampling.cpp` — `sampleColormap` interpolation (endpoints,
  midpoints, opacity ramp, monotonic mapping).
- A render regression (extend the 4c smoke or add one): after a programmatic
  `GraphEditModel` edit (swap TF preset / rewire), the bridge still renders.
  VisRTX render test → `set_tests_properties(... PROPERTIES TIMEOUT 300)`.

**Build-verified + manual checklist (GUI):** the three new windows and the
`tsdFlow` layout that hosts them. Manual: add/delete nodes, drag/reject links,
conversion-link color + tooltip, inspector edits re-render, TF curve edit
re-renders the volume.

## Scope

**In scope (4d):** full node add/delete/connect/disconnect on canvas; type + cycle
validation with conversion feedback; selection-driven inspector with plain
(unbounded) scalar param widgets; node-bound interactive TF curve editor (color +
opacity); live re-eval; and three additive engine APIs — `Graph::canConnect`,
`Graph` node enumeration, and `NodeRegistry::types()`.

**Out of scope (deferred):** transfer/residency link feedback (4b);
per-display→viewport assignment UI; node-position and graph persistence, TF
load/save (Phase 5); undo/redo (4e); multi-select editing, copy/paste, node
search/grouping.

## File plan (indicative)

| File | New/Mod | Responsibility |
|------|---------|----------------|
| `tsd/src/tsd/graph/Graph.hpp/.cpp` | Mod | extract public `canConnect` from `connect`; add node-enumeration accessor (`nodeIds()`) |
| `tsd/src/tsd/graph/NodeRegistry.hpp/.cpp` | Mod | add `types() const → std::vector<Token>` |
| `tsd/src/tsd/graph_nodes/GraphEditModel.hpp/.cpp` | New | tested editor-logic core |
| `tsd/src/tsd/graph_nodes/TransferFunctionNode.hpp` | New | public `ITransferFunctionNode` interface |
| `tsd/src/tsd/graph_nodes/TransferFunction.cpp` | Mod | move node out of anon namespace; implement `ITransferFunctionNode` (reuse `core::TransferFunction` + samples); sample from control points; remove eval-time `getPreset`/`getOr("samples")` reads |
| `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp/.cpp` | New | imnodes canvas + int⟷id mapping |
| `tsd/src/tsd/ui/imgui/windows/Inspector.hpp/.cpp` | New | selection-driven params (dispatch on `Any::type()`) |
| `tsd/src/tsd/ui/imgui/windows/TFCurveEditor.hpp/.cpp` | New | node-bound TF curve panel (not a `Window`) |
| `tsd/apps/interactive/tsdFlow/tsdFlow.cpp` | Mod | host the windows + selection state + dirty-coalesced `bridge.update()`; wire display-node add/remove to `bridge.setDisplay`/`removeDisplay` |
| `tsd/tests/test_nodes_GraphEditModel.cpp`, `test_nodes_TfnSampling.cpp` | New | unit tests |
| relevant `CMakeLists.txt` (graph, graph_nodes, ui/imgui, tests) | Mod | wiring (`tsd_imnodes` link for `tsd_ui_imgui` if not already) |
