# tsdFlow Phase 4d — Interactive Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `tsdFlow` from a fixed-demo viewer into a user-editable node graph — a visual node-graph editor (imnodes), a selection-driven parameter inspector, and a node-bound interactive transfer-function curve editor — all re-rendering live through the Phase 4c `GraphRenderBridge`.

**Architecture:** A UI-free, unit-tested `GraphEditModel` (in `tsd_graph_nodes`) wraps the engine and exposes edit ops + validation + TF sampling. Three thin ImGui views in `tsd_ui_imgui` sit on top: a `GraphEditor` imnodes canvas, a selection-driven `Inspector`, and an embedded `TFCurveEditor` panel. The `TransferFunction` node grows typed state (reusing `tsd::core::TransferFunction`) exposed via an `ITransferFunctionNode` interface. Three small additive engine APIs (`Graph::canConnect`, `Graph::nodeIds`, `NodeRegistry::types`) back the editor.

**Tech Stack:** C++17, `tsd_graph` + `tsd_graph_nodes` (engine/catalog), `tsd_rendering` `GraphRenderBridge` (Phase 3), `tsd_ui_imgui` (Application/Window), vendored `tsd_ext_imnodes` (Nelarius/imnodes v0.5), ANARI + VisRTX, ImGui/SDL3, Catch2, jj.

## Global Constraints

- Version control is **jj**, not git. Commit ONLY a task's files with explicit paths: `jj commit <paths> -m "..."`. **NEVER** a bare `jj commit` — an unrelated `.envrc` in the working copy must stay uncommitted.
- Build tree is `_out/_cmake` (Ninja Multi-Config, RelWithDebInfo). Do NOT create a new `build/` dir.
  - Configure (only if CMake files changed): `cmake --build _out/_cmake --config RelWithDebInfo --target <t>` re-runs CMake automatically; if a fresh configure is needed: `cmake -S . -B _out/_cmake` (preset already cached).
  - Build tests: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - Build the app: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow`
  - Run a test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- `clang-format -i` ONLY `.cpp`/`.hpp` files — **NEVER** clang-format `CMakeLists.txt` (it mangles CMake; edit by hand).
- File header on every new file: `// Copyright 2026 NVIDIA Corporation` then `// SPDX-License-Identifier: Apache-2.0`. Headers use `#pragma once`.
- Namespaces: engine `tsd::graph`; catalog `tsd::graph_nodes`; UI `tsd::ui::imgui`; core `tsd::core`.
- VisRTX is the render device (GPU sandbox; the first render does an OptiX warmup, so any test that renders gets `set_tests_properties(<name> PROPERTIES TIMEOUT 300)`).
- The `tsdFlow` app is under the `TSD_BUILD_INTERACTIVE_APPS` CMake guard.
- 4d is host-residency only; no transfers occur. No changes to the Evaluator or the bridge.

## Verified API reference (cite these exactly — already confirmed against source)

```cpp
// tsd/graph/Graph.hpp
using NodeId = uint64_t;  using ConnectionId = uint64_t;
constexpr NodeId INVALID_NODE = 0;  constexpr ConnectionId INVALID_CONNECTION = 0;
struct Connection { ConnectionId id; NodeId fromNode; tsd::core::Token fromPort; NodeId toNode; tsd::core::Token toPort; };
struct LinkResult { bool ok; ConnectionId id; std::string reason; };
struct GraphNode { NodeId id; std::unique_ptr<Node> impl; /* ... */ };
class Graph {
  explicit Graph(const ConversionRegistry *conversions = nullptr);
  NodeId addNode(std::unique_ptr<Node> node);
  void removeNode(NodeId id);                                  // already dirties consumers
  LinkResult connect(NodeId from, Token fromPort, NodeId to, Token toPort);
  void disconnect(ConnectionId id);
  GraphNode *node(NodeId id);  const GraphNode *node(NodeId id) const;
  const std::vector<Connection> &connections() const;
  const Connection *inputConnection(NodeId to, Token toPort) const;
  void setConversionRegistry(const ConversionRegistry *r);
  void markDirty(NodeId id);                                   // clears cache, recurses to consumers
 private:                                                      // all const, reusable by canConnect:
  bool wouldCreateCycle(NodeId from, NodeId to) const;
  bool findOutputSpec(const GraphNode&, Token, PortSpec&) const;
  bool findInputSpec(const GraphNode&, Token, PortSpec&) const;
  std::map<NodeId, GraphNode> m_nodes;  std::vector<Connection> m_connections;
  ConnectionId m_nextConnId{1};  const ConversionRegistry *m_conversions{...};
};

// tsd/graph/NodeRegistry.hpp  — m_entries is private std::vector<Entry{Token name; NodeFactory factory;}>
struct NodeRegistry {
  void registerType(Token name, NodeFactory factory);
  std::unique_ptr<Node> create(Token name) const;
  bool isRegistered(Token name) const;
};

// tsd/graph/Port.hpp
struct PortSpec { tsd::core::Token name; PortType type; bool required{true}; std::vector<Token> acceptedBackends; };
struct NodeTypeInfo { Token name; Token category; std::vector<PortSpec> inputs; std::vector<PortSpec> outputs; bool isCacheable{true}; };
// tsd/graph/PortType.hpp
struct PortType { tsd::core::Token name; };   // struct, NOT enum; operator== compares .name
// tsd/graph/Node.hpp
class Node { virtual ~Node()=default; virtual NodeTypeInfo typeInfo() const=0; virtual ParameterList &parameters()=0; virtual void evaluate(EvalContext&)=0; };
// tsd/graph/Parameter.hpp
struct Parameter { tsd::core::Token name; tsd::core::Any value; };
struct ParameterList { template<class T> void set(Token,T); template<class T> T get(Token) const;
  template<class T> T getOr(Token, const T&) const; bool has(Token) const; const std::vector<Parameter>& items() const; uint64_t hash() const; };

// tsd/graph/ConversionRegistry.hpp
struct ConversionEntry { PortType from; PortType to; /* fn, estimateElements — NO name field */ };
struct ConversionRegistry { const ConversionEntry *find(PortType from, PortType to) const; };

// tsd/core/Any.hpp
anari::DataType Any::type() const;  bool Any::is(anari::DataType) const;  std::string Any::getString() const;
template<class T> T Any::get() const;            // e.g. a.get<float>()

// tsd/core/ColorMapUtil.hpp  (namespace tsd::core)
using ColorPoint = float4;     // {.x=position, .y=R, .z=G, .w=B}   (header's "RGBA" comment is WRONG)
using OpacityPoint = float2;   // {.x=position, .y=opacity}
struct TransferFunction { std::vector<ColorPoint> colorPoints; std::vector<OpacityPoint> opacityPoints; math::box1 range; };
TransferFunction makeDefaultTransferFunction();
namespace detail { math::float3 interpolateColor(const std::vector<ColorPoint>&, float x);   // returns RGB
                   float interpolateOpacity(const std::vector<OpacityPoint>&, float x); }
namespace colormap { extern std::vector<float3> viridis, cool_to_warm, grayscale, jet, inferno, black_body, ice_fire; }

// tsd/graph_nodes/Descriptors.hpp
struct TransferFunctionData { tsd::core::AnyArray colormap; tsd::core::math::float2 valueRange{0,1}; };
Token portRange();  Token portTF();   // "range" / "transferFunction"

// tsd/rendering/bridge/GraphRenderBridge.hpp
class GraphRenderBridge {
  GraphRenderBridge(graph::Graph&, graph::Evaluator&, core::Token lib, anari::Device, int numViewports);
  void setDisplay(graph::NodeId, uint64_t viewportMask, bool enabled);
  void removeDisplay(graph::NodeId);
  void update();
  anari::World world(int viewport) const;
};
// tsd/graph/Evaluator.hpp
class Evaluator { explicit Evaluator(Graph&, /*...*/); bool pull(NodeId); };

// imnodes (v0.5) — namespace ImNodes
void BeginNodeEditor(); void EndNodeEditor(); void MiniMap();
void BeginNode(int id); void EndNode();  void BeginNodeTitleBar(); void EndNodeTitleBar();
void BeginInputAttribute(int id, ImNodesPinShape shape=ImNodesPinShape_CircleFilled);  void EndInputAttribute();
void BeginOutputAttribute(int id, ImNodesPinShape shape=ImNodesPinShape_CircleFilled);  void EndOutputAttribute();
void Link(int id, int start_attr_id, int end_attr_id);
void SetNodeScreenSpacePos(int node_id, const ImVec2&);
void PushColorStyle(ImNodesCol item, unsigned int color);  void PopColorStyle();
bool IsLinkCreated(int* started_at_attr_id, int* ended_at_attr_id, bool* snap=nullptr);
int NumSelectedNodes(); int NumSelectedLinks(); void GetSelectedNodes(int*); void GetSelectedLinks(int*);
bool IsEditorHovered();
enum ImNodesPinShape_ { ImNodesPinShape_Circle, _CircleFilled, _Triangle, _TriangleFilled, _Quad, _QuadFilled };
// imnodes context: ImNodes::CreateContext()/DestroyContext() once per app (see Application or create in editor ctor).
```

---

## Task 1: Additive engine APIs (`Graph::canConnect`, `Graph::nodeIds`, `NodeRegistry::types`)

**Files:**
- Modify: `tsd/src/tsd/graph/Graph.hpp`, `tsd/src/tsd/graph/Graph.cpp`
- Modify: `tsd/src/tsd/graph/NodeRegistry.hpp`, `tsd/src/tsd/graph/NodeRegistry.cpp`
- Test: `tsd/tests/test_graph_EditApis.cpp` (new), `tsd/tests/CMakeLists.txt` (modify)

**Interfaces:**
- Produces:
  - `LinkResult Graph::canConnect(NodeId from, Token fromPort, NodeId to, Token toPort) const;` — non-mutating; same checks as `connect`, returns `{ok, INVALID_CONNECTION, reason}`.
  - `std::vector<NodeId> Graph::nodeIds() const;` — all node ids, ascending.
  - `std::vector<tsd::core::Token> NodeRegistry::types() const;` — all registered type names, registration order.

- [ ] **Step 1: Write the failing test** — create `tsd/tests/test_graph_EditApis.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"

using tsd::core::Token;
using namespace tsd::graph;

namespace {
NodeId add(Graph &g, NodeRegistry &r, const char *t) { return g.addNode(r.create(Token(t))); }
} // namespace

SCENARIO("Graph::nodeIds enumerates all nodes", "[graph-editapis]")
{
  NodeRegistry reg; tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  const NodeId a = add(g, reg, "GenerateNoiseVolume");
  const NodeId b = add(g, reg, "ScalarRange");
  auto ids = g.nodeIds();
  REQUIRE(ids.size() == 2);
  REQUIRE(std::find(ids.begin(), ids.end(), a) != ids.end());
  REQUIRE(std::find(ids.begin(), ids.end(), b) != ids.end());
}

SCENARIO("NodeRegistry::types lists registered type names", "[graph-editapis]")
{
  NodeRegistry reg; tsd::graph_nodes::registerBuiltinNodes(reg);
  auto types = reg.types();
  REQUIRE(types.size() >= 6);
  REQUIRE(std::find(types.begin(), types.end(), Token("TransferFunction")) != types.end());
}

SCENARIO("Graph::canConnect mirrors connect's validation without mutating", "[graph-editapis]")
{
  NodeRegistry reg; tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  const NodeId src = add(g, reg, "GenerateNoiseVolume");   // out: "out" (field)
  const NodeId sr  = add(g, reg, "ScalarRange");           // in: "in" (field), out: "out" (range)
  const NodeId tf  = add(g, reg, "TransferFunction");      // in: "in" (range)

  GIVEN("a valid pair") {
    auto chk = g.canConnect(src, Token("out"), sr, Token("in"));
    THEN("canConnect says ok and no connection was created") {
      REQUIRE(chk.ok);
      REQUIRE(g.connections().empty());            // non-mutating
    }
  }
  GIVEN("an unknown port") {
    auto chk = g.canConnect(src, Token("nope"), sr, Token("in"));
    THEN("it reports the same rejection connect would") { REQUIRE_FALSE(chk.ok); }
  }
  GIVEN("a type-incompatible pair (field out -> range in)") {
    auto chk = g.canConnect(src, Token("out"), tf, Token("in"));
    THEN("it is rejected") { REQUIRE_FALSE(chk.ok); }
  }
  GIVEN("a cycle (sr.out -> tf.in committed, then tf back to sr)") {
    REQUIRE(g.connect(src, Token("out"), sr, Token("in")).ok);
    REQUIRE(g.connect(sr, Token("out"), tf, Token("in")).ok);
    // tf has no output feeding sr's input type, but verify cycle path on a self-feed attempt:
    auto chk = g.canConnect(sr, Token("out"), sr, Token("in"));
    THEN("a self/cycle link is rejected") { REQUIRE_FALSE(chk.ok); }
  }
}
```

Add `#include <algorithm>` at the top alongside the includes.

- [ ] **Step 2: Register the test in CMake** — edit `tsd/tests/CMakeLists.txt` by hand. Add `test_graph_EditApis.cpp` to the `project_add_executable(...)` source list (near the other `test_graph_*`/`test_nodes_*` entries, e.g. after `test_nodes_Scaffold.cpp`), and add this `add_test` line near the other `tsd::graph::*` registrations:

```cmake
add_test(NAME tsd::graph::EditApis COMMAND ${PROJECT_NAME} "[graph-editapis]")
```

- [ ] **Step 3: Build and confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -20`
Expected: compile error — `'canConnect' is not a member of 'tsd::graph::Graph'` (and `nodeIds`, `types`).

- [ ] **Step 4: Declare the new methods.** In `tsd/src/tsd/graph/Graph.hpp`, in the `public:` section after the `connect` declaration, add:

```cpp
  // Non-mutating pre-flight of connect(): identical validation, no mutation.
  // Returns {ok, INVALID_CONNECTION, reason} — id is never allocated here.
  LinkResult canConnect(NodeId from,
      tsd::core::Token fromPort,
      NodeId to,
      tsd::core::Token toPort) const;

  // All node ids currently in the graph, ascending.
  std::vector<NodeId> nodeIds() const;
```

Ensure `#include <vector>` is present in `Graph.hpp` (it is, for `m_connections`).

In `tsd/src/tsd/graph/NodeRegistry.hpp`, after `bool isRegistered(...) const;` add:

```cpp
  // All registered type names, in registration order.
  std::vector<tsd::core::Token> types() const;
```

- [ ] **Step 5: Implement `canConnect` by extracting `connect`'s validation.** In `tsd/src/tsd/graph/Graph.cpp`, replace the existing `Graph::connect` with a `canConnect` + a thin `connect`:

```cpp
LinkResult Graph::canConnect(
    NodeId from, tsd::core::Token fromPort, NodeId to, tsd::core::Token toPort) const
{
  const auto *fromN = node(from);
  const auto *toN = node(to);
  if (!fromN || !toN)
    return {false, INVALID_CONNECTION, "unknown node"};

  PortSpec outSpec, inSpec;
  if (!findOutputSpec(*fromN, fromPort, outSpec))
    return {false, INVALID_CONNECTION, "no such output port"};
  if (!findInputSpec(*toN, toPort, inSpec))
    return {false, INVALID_CONNECTION, "no such input port"};

  if (wouldCreateCycle(from, to))
    return {false, INVALID_CONNECTION, "connection would create a cycle"};

  if (outSpec.type != inSpec.type) {
    const bool convertible = m_conversions
        && m_conversions->find(outSpec.type, inSpec.type) != nullptr;
    if (!convertible)
      return {false,
          INVALID_CONNECTION,
          "incompatible port types and no registered conversion"};
  }

  return {true, INVALID_CONNECTION, ""};
}

LinkResult Graph::connect(
    NodeId from, tsd::core::Token fromPort, NodeId to, tsd::core::Token toPort)
{
  const LinkResult check = canConnect(from, fromPort, to, toPort);
  if (!check.ok)
    return check;

  const ConnectionId id = m_nextConnId++;
  m_connections.push_back(Connection{id, from, fromPort, to, toPort});

  // New incoming data invalidates the consumer's cached output.
  auto *toN = node(to);
  toN->state = EvalState::Dirty;
  toN->cache.clear();
  return {true, id, ""};
}
```

> Note: `findOutputSpec`/`findInputSpec`/`wouldCreateCycle` are already `const`, and `node()` has a const overload, so `canConnect` compiles as `const`.

- [ ] **Step 6: Implement `nodeIds`.** Add to `tsd/src/tsd/graph/Graph.cpp`:

```cpp
std::vector<NodeId> Graph::nodeIds() const
{
  std::vector<NodeId> ids;
  ids.reserve(m_nodes.size());
  for (const auto &kv : m_nodes)   // std::map → ascending key order
    ids.push_back(kv.first);
  return ids;
}
```

- [ ] **Step 7: Implement `NodeRegistry::types`.** Add to `tsd/src/tsd/graph/NodeRegistry.cpp`:

```cpp
std::vector<tsd::core::Token> NodeRegistry::types() const
{
  std::vector<tsd::core::Token> out;
  out.reserve(m_entries.size());
  for (const auto &e : m_entries)
    out.push_back(e.name);
  return out;
}
```

- [ ] **Step 8: Build and run the test**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph::EditApis' --output-on-failure`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Graph.hpp tsd/src/tsd/graph/Graph.cpp tsd/src/tsd/graph/NodeRegistry.hpp tsd/src/tsd/graph/NodeRegistry.cpp tsd/tests/test_graph_EditApis.cpp
jj commit tsd/src/tsd/graph/Graph.hpp tsd/src/tsd/graph/Graph.cpp tsd/src/tsd/graph/NodeRegistry.hpp tsd/src/tsd/graph/NodeRegistry.cpp tsd/tests/test_graph_EditApis.cpp tsd/tests/CMakeLists.txt -m "feat(graph): add canConnect/nodeIds/NodeRegistry::types editor APIs"
```

---

## Task 2: `GraphEditModel` — edit ops, validation, classification, catalog

**Files:**
- Create: `tsd/src/tsd/graph_nodes/GraphEditModel.hpp`, `tsd/src/tsd/graph_nodes/GraphEditModel.cpp`
- Modify: `tsd/src/tsd/graph_nodes/CMakeLists.txt`
- Test: `tsd/tests/test_nodes_GraphEditModel.cpp` (new), `tsd/tests/CMakeLists.txt` (modify)

**Interfaces:**
- Consumes: `Graph::canConnect`/`nodeIds` (Task 1), `NodeRegistry::types` (Task 1), `ConversionRegistry::find`.
- Produces:
  - `enum class tsd::graph_nodes::LinkKind { Direct, Conversion, Incompatible, Cycle };`
  - `struct ConnectCheck { LinkKind kind; std::string detail; bool ok() const; };`
  - `class GraphEditModel` with: ctor `(graph::Graph&, graph::NodeRegistry&, const graph::ConversionRegistry*)`; `graph::NodeId addNode(core::Token)`; `void removeNode(graph::NodeId)`; `graph::LinkResult connect(graph::NodeId, core::Token, graph::NodeId, core::Token)`; `void disconnect(graph::ConnectionId)`; `ConnectCheck canConnect(graph::NodeId, core::Token, graph::NodeId, core::Token) const`; `LinkKind classify(const graph::Connection&) const`; `const std::vector<core::Token>& nodeCatalog() const`.
  - (TF sampling `static std::vector<math::float4> sampleColormap(...)` is added in Task 3.)

- [ ] **Step 1: Write the failing test** — create `tsd/tests/test_nodes_GraphEditModel.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::GraphEditModel;
using tsd::graph_nodes::LinkKind;

SCENARIO("GraphEditModel adds, connects, classifies, and removes", "[edit-model]")
{
  NodeRegistry reg; tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  GraphEditModel model(g, reg, nullptr);

  WHEN("the catalog is queried") {
    THEN("it lists builtin types") {
      const auto &cat = model.nodeCatalog();
      REQUIRE(std::find(cat.begin(), cat.end(), Token("TransferFunction")) != cat.end());
    }
  }

  WHEN("nodes are added and connected") {
    const NodeId src = model.addNode(Token("GenerateNoiseVolume"));
    const NodeId sr  = model.addNode(Token("ScalarRange"));
    REQUIRE(src != INVALID_NODE);
    REQUIRE(g.nodeIds().size() == 2);

    auto chk = model.canConnect(src, Token("out"), sr, Token("in"));
    THEN("a valid link is Direct") {
      REQUIRE(chk.ok());
      REQUIRE(chk.kind == LinkKind::Direct);
    }

    auto res = model.connect(src, Token("out"), sr, Token("in"));
    REQUIRE(res.ok);
    THEN("the committed link classifies as Direct") {
      REQUIRE(g.connections().size() == 1);
      REQUIRE(model.classify(g.connections().front()) == LinkKind::Direct);
    }

    AND_WHEN("the connection is removed") {
      model.disconnect(res.id);
      THEN("the graph has no connections") { REQUIRE(g.connections().empty()); }
    }
    AND_WHEN("a node is removed") {
      model.removeNode(sr);
      THEN("it is gone") { REQUIRE(g.nodeIds().size() == 1); }
    }
  }

  WHEN("an incompatible link is checked") {
    const NodeId src = model.addNode(Token("GenerateNoiseVolume")); // field out
    const NodeId tf  = model.addNode(Token("TransferFunction"));    // range in
    auto chk = model.canConnect(src, Token("out"), tf, Token("in"));
    THEN("it is Incompatible and not ok") {
      REQUIRE_FALSE(chk.ok());
      REQUIRE(chk.kind == LinkKind::Incompatible);
    }
  }

  WHEN("a cycle is checked") {
    const NodeId a = model.addNode(Token("GenerateNoiseVolume"));
    const NodeId b = model.addNode(Token("ScalarRange"));
    REQUIRE(model.connect(a, Token("out"), b, Token("in")).ok);
    auto chk = model.canConnect(b, Token("out"), b, Token("in"));
    THEN("it is rejected") { REQUIRE_FALSE(chk.ok()); }
  }
}
```

Add `#include <algorithm>` at the top.

- [ ] **Step 2: Register the test** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_GraphEditModel.cpp` to the executable source list, and add:

```cmake
add_test(NAME tsd::nodes::GraphEditModel COMMAND ${PROJECT_NAME} "[edit-model]")
```

- [ ] **Step 3: Build and confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -20`
Expected: `GraphEditModel.hpp: No such file or directory`.

- [ ] **Step 4: Create `tsd/src/tsd/graph_nodes/GraphEditModel.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/graph/ConversionRegistry.hpp"
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
// std
#include <string>
#include <vector>

namespace tsd::graph_nodes {

enum class LinkKind { Direct, Conversion, Incompatible, Cycle };

struct ConnectCheck
{
  LinkKind kind{LinkKind::Incompatible};
  std::string detail; // "from->to" for Conversion, else the reject reason
  bool ok() const
  {
    return kind == LinkKind::Direct || kind == LinkKind::Conversion;
  }
};

// UI-free editor logic over a Graph + NodeRegistry (+ optional ConversionRegistry).
// Every mutating op marks the graph dirty so the bridge re-renders on update().
class GraphEditModel
{
 public:
  GraphEditModel(tsd::graph::Graph &graph,
      tsd::graph::NodeRegistry &registry,
      const tsd::graph::ConversionRegistry *conversions);

  // Mutating ops.
  tsd::graph::NodeId addNode(tsd::core::Token type);
  void removeNode(tsd::graph::NodeId id);
  tsd::graph::LinkResult connect(tsd::graph::NodeId from,
      tsd::core::Token fromPort,
      tsd::graph::NodeId to,
      tsd::core::Token toPort);
  void disconnect(tsd::graph::ConnectionId id);

  // Non-mutating queries.
  ConnectCheck canConnect(tsd::graph::NodeId from,
      tsd::core::Token fromPort,
      tsd::graph::NodeId to,
      tsd::core::Token toPort) const;
  LinkKind classify(const tsd::graph::Connection &c) const;

  const std::vector<tsd::core::Token> &nodeCatalog() const;

  // Pure TF sampling (implemented in Task 3): control points -> RGBA colormap.
  // ColorPoint is {position, R, G, B}; OpacityPoint is {position, opacity}.
  static std::vector<tsd::core::math::float4> sampleColormap(
      const std::vector<tsd::core::ColorPoint> &colorPoints,
      const std::vector<tsd::core::OpacityPoint> &opacityPoints,
      int samples);

 private:
  // Resolve a node's declared PortType for a named output/input port.
  bool outputType(tsd::graph::NodeId, tsd::core::Token, tsd::graph::PortType &) const;
  bool inputType(tsd::graph::NodeId, tsd::core::Token, tsd::graph::PortType &) const;

  tsd::graph::Graph &m_graph;
  tsd::graph::NodeRegistry &m_registry;
  const tsd::graph::ConversionRegistry *m_conversions{nullptr};
  std::vector<tsd::core::Token> m_catalog; // cached NodeRegistry::types()
};

} // namespace tsd::graph_nodes
```

- [ ] **Step 5: Create `tsd/src/tsd/graph_nodes/GraphEditModel.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/GraphEditModel.hpp"

namespace tsd::graph_nodes {

using namespace tsd::graph;
using tsd::core::Token;

GraphEditModel::GraphEditModel(
    Graph &graph, NodeRegistry &registry, const ConversionRegistry *conversions)
    : m_graph(graph), m_registry(registry), m_conversions(conversions)
{
  m_catalog = m_registry.types();
}

NodeId GraphEditModel::addNode(Token type)
{
  auto node = m_registry.create(type);
  if (!node)
    return INVALID_NODE;
  const NodeId id = m_graph.addNode(std::move(node));
  m_graph.markDirty(id);
  return id;
}

void GraphEditModel::removeNode(NodeId id)
{
  m_graph.removeNode(id); // already dirties downstream consumers
}

LinkResult GraphEditModel::connect(
    NodeId from, Token fromPort, NodeId to, Token toPort)
{
  const LinkResult r = m_graph.connect(from, fromPort, to, toPort);
  if (r.ok)
    m_graph.markDirty(to);
  return r;
}

void GraphEditModel::disconnect(ConnectionId id)
{
  // Capture the consumer before removing so we can dirty it.
  NodeId consumer = INVALID_NODE;
  for (const auto &c : m_graph.connections())
    if (c.id == id) {
      consumer = c.toNode;
      break;
    }
  m_graph.disconnect(id);
  if (consumer != INVALID_NODE)
    m_graph.markDirty(consumer);
}

bool GraphEditModel::outputType(NodeId id, Token port, PortType &out) const
{
  const auto *gn = m_graph.node(id);
  if (!gn || !gn->impl)
    return false;
  for (const auto &p : gn->impl->typeInfo().outputs)
    if (p.name == port) {
      out = p.type;
      return true;
    }
  return false;
}

bool GraphEditModel::inputType(NodeId id, Token port, PortType &out) const
{
  const auto *gn = m_graph.node(id);
  if (!gn || !gn->impl)
    return false;
  for (const auto &p : gn->impl->typeInfo().inputs)
    if (p.name == port) {
      out = p.type;
      return true;
    }
  return false;
}

ConnectCheck GraphEditModel::canConnect(
    NodeId from, Token fromPort, NodeId to, Token toPort) const
{
  const LinkResult r = m_graph.canConnect(from, fromPort, to, toPort);
  if (!r.ok) {
    const bool cycle = r.reason.find("cycle") != std::string::npos;
    return {cycle ? LinkKind::Cycle : LinkKind::Incompatible, r.reason};
  }

  PortType o, i;
  if (outputType(from, fromPort, o) && inputType(to, toPort, i) && o != i) {
    std::string detail =
        std::string(o.name.c_str()) + "->" + std::string(i.name.c_str());
    return {LinkKind::Conversion, std::move(detail)};
  }
  return {LinkKind::Direct, ""};
}

LinkKind GraphEditModel::classify(const Connection &c) const
{
  PortType o, i;
  if (outputType(c.fromNode, c.fromPort, o) && inputType(c.toNode, c.toPort, i)
      && o != i)
    return LinkKind::Conversion;
  return LinkKind::Direct; // committed links are never Incompatible/Cycle
}

const std::vector<Token> &GraphEditModel::nodeCatalog() const
{
  return m_catalog;
}

} // namespace tsd::graph_nodes
```

> `Token::c_str()` is assumed for the detail string. If `Token` has no `c_str()`, use `o.name.value()` cast or whatever string accessor `tsd::core::Token` exposes (check `tsd/core/Token.hpp`) — adjust the two `.c_str()` calls accordingly and report the change.

- [ ] **Step 6: Add to CMake** — edit `tsd/src/tsd/graph_nodes/CMakeLists.txt` by hand, adding `GraphEditModel.cpp` to `project_sources(PRIVATE ...)` (after `DemoGraph.cpp`).

- [ ] **Step 7: Build and run**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::GraphEditModel' --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/GraphEditModel.hpp tsd/src/tsd/graph_nodes/GraphEditModel.cpp tsd/tests/test_nodes_GraphEditModel.cpp
jj commit tsd/src/tsd/graph_nodes/GraphEditModel.hpp tsd/src/tsd/graph_nodes/GraphEditModel.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/tests/test_nodes_GraphEditModel.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): GraphEditModel — UI-free edit ops + link validation/classification"
```

---

## Task 3: `GraphEditModel::sampleColormap` (pure TF sampling)

**Files:**
- Modify: `tsd/src/tsd/graph_nodes/GraphEditModel.cpp` (the declaration already exists from Task 2)
- Test: `tsd/tests/test_nodes_TfnSampling.cpp` (new), `tsd/tests/CMakeLists.txt` (modify)

**Interfaces:**
- Consumes: `tsd::core::detail::interpolateColor` (returns RGB `float3`), `tsd::core::detail::interpolateOpacity` (returns `float`).
- Produces: `static std::vector<math::float4> GraphEditModel::sampleColormap(const std::vector<ColorPoint>&, const std::vector<OpacityPoint>&, int samples)` — `samples` RGBA entries; entry `i` = (interpolated RGB at `t=i/(samples-1)`, interpolated opacity at the same `t`).

- [ ] **Step 1: Write the failing test** — create `tsd/tests/test_nodes_TfnSampling.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph_nodes/GraphEditModel.hpp"

using tsd::core::ColorPoint;
using tsd::core::OpacityPoint;
using tsd::graph_nodes::GraphEditModel;
using float4 = tsd::core::math::float4;

SCENARIO("sampleColormap interpolates color and opacity over [0,1]", "[tfn-sampling]")
{
  // ColorPoint is {position, R, G, B}; OpacityPoint is {position, opacity}.
  std::vector<ColorPoint> colors = {{0.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 0.f}}; // black->red
  std::vector<OpacityPoint> opac = {{0.f, 0.f}, {1.f, 1.f}};                     // 0 -> 1 ramp

  auto cm = GraphEditModel::sampleColormap(colors, opac, 3);

  THEN("there are `samples` entries") { REQUIRE(cm.size() == 3); }
  THEN("endpoints match the control points") {
    REQUIRE(cm.front().x == Approx(0.f)); // R at t=0
    REQUIRE(cm.front().w == Approx(0.f)); // A at t=0
    REQUIRE(cm.back().x == Approx(1.f));  // R at t=1
    REQUIRE(cm.back().w == Approx(1.f));  // A at t=1
  }
  THEN("the midpoint is halfway") {
    REQUIRE(cm[1].x == Approx(0.5f).margin(0.01)); // R at t=0.5
    REQUIRE(cm[1].w == Approx(0.5f).margin(0.01)); // A at t=0.5
    REQUIRE(cm[1].y == Approx(0.f));               // G stays 0
  }
}
```

- [ ] **Step 2: Register the test** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_TfnSampling.cpp` to the source list and:

```cmake
add_test(NAME tsd::nodes::TfnSampling COMMAND ${PROJECT_NAME} "[tfn-sampling]")
```

- [ ] **Step 3: Build and confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -20`
Expected: link error — `undefined reference to GraphEditModel::sampleColormap`.

- [ ] **Step 4: Implement `sampleColormap`.** Add to `tsd/src/tsd/graph_nodes/GraphEditModel.cpp` (inside the namespace; the helpers are in `tsd::core::detail`):

```cpp
std::vector<tsd::core::math::float4> GraphEditModel::sampleColormap(
    const std::vector<tsd::core::ColorPoint> &colorPoints,
    const std::vector<tsd::core::OpacityPoint> &opacityPoints,
    int samples)
{
  using tsd::core::math::float4;
  std::vector<float4> out;
  if (samples < 2)
    return out;
  out.reserve(size_t(samples));
  const float denom = float(samples - 1);
  for (int i = 0; i < samples; ++i) {
    const float t = float(i) / denom;
    const auto rgb = tsd::core::detail::interpolateColor(colorPoints, t);
    const float a = tsd::core::detail::interpolateOpacity(opacityPoints, t);
    out.push_back(float4(rgb.x, rgb.y, rgb.z, a));
  }
  return out;
}
```

- [ ] **Step 5: Build and run**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::TfnSampling' --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/GraphEditModel.cpp tsd/tests/test_nodes_TfnSampling.cpp
jj commit tsd/src/tsd/graph_nodes/GraphEditModel.cpp tsd/tests/test_nodes_TfnSampling.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): GraphEditModel::sampleColormap control-point sampling"
```

---

## Task 4: `TransferFunction` node — typed state via `ITransferFunctionNode`

**Files:**
- Create: `tsd/src/tsd/graph_nodes/TransferFunctionNode.hpp`
- Modify: `tsd/src/tsd/graph_nodes/TransferFunction.cpp`
- Test: `tsd/tests/test_nodes_TransferFunction.cpp` (modify — add a state-driven case), `tsd/tests/CMakeLists.txt` (already registers `[graph-...]`/`[nodes-...]`; the existing TF test already runs)

**Interfaces:**
- Consumes: `GraphEditModel::sampleColormap` (Task 3), `tsd::core::TransferFunction`, `tsd::core::makeDefaultTransferFunction`.
- Produces:
  - `struct tsd::graph_nodes::ITransferFunctionNode { virtual ~ITransferFunctionNode() = default; virtual tsd::core::TransferFunction &tfState() = 0; virtual int &samples() = 0; };`
  - The `TransferFunction` node (now in the named `tsd::graph_nodes` namespace) implements it; its `evaluate()` samples from `tfState()`/`samples()`.

- [ ] **Step 1: Write the failing test** — append to `tsd/tests/test_nodes_TransferFunction.cpp` (keep its existing `#include`s; add `#include "tsd/graph_nodes/TransferFunctionNode.hpp"` and `#include "tsd/graph_nodes/GraphEditModel.hpp"` if not present):

```cpp
SCENARIO("TransferFunction node samples its editable control points", "[nodes-tf-state]")
{
  using namespace tsd::graph;
  using tsd::core::Token;
  NodeRegistry reg; tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  const NodeId src = g.addNode(reg.create(Token("GenerateNoiseVolume")));
  const NodeId sr  = g.addNode(reg.create(Token("ScalarRange")));
  const NodeId tf  = g.addNode(reg.create(Token("TransferFunction")));
  REQUIRE(g.connect(src, Token("out"), sr, Token("in")).ok);
  REQUIRE(g.connect(sr, Token("out"), tf, Token("in")).ok);

  // The TF node exposes editable state.
  auto *node = g.node(tf)->impl.get();
  auto *itf = dynamic_cast<tsd::graph_nodes::ITransferFunctionNode *>(node);
  REQUIRE(itf != nullptr);

  // Set a known black->white ramp, opaque.
  itf->tfState().colorPoints = {{0.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}};
  itf->tfState().opacityPoints = {{0.f, 1.f}, {1.f, 1.f}};
  itf->samples() = 4;
  g.markDirty(tf);

  Evaluator e(g);
  REQUIRE(e.pull(tf));
  // The evaluated output is a TransferFunctionData with a 4-entry colormap.
  // (Reuse whatever accessor the existing TF tests use to read the output payload;
  //  assert colormap size == 4 and the last entry is white & opaque.)
}
```

> The final assertions depend on how the existing `test_nodes_TransferFunction.cpp` reaches the node's output `TransferFunctionData` (it already does this for the preset path — mirror that accessor). If the existing test pulls via an `EvalContext`/cache accessor, reuse it; assert `colormap` has 4 entries and `get<float4>(3) ≈ {1,1,1,1}`.

- [ ] **Step 2: Build and confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -20`
Expected: `TransferFunctionNode.hpp: No such file` or `ITransferFunctionNode is not a member`.

- [ ] **Step 3: Create `tsd/src/tsd/graph_nodes/TransferFunctionNode.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"

namespace tsd::graph_nodes {

// Implemented by the TransferFunction node so UI can edit its control points
// directly (these don't fit in the Any-based ParameterList).
struct ITransferFunctionNode
{
  virtual ~ITransferFunctionNode() = default;
  virtual tsd::core::TransferFunction &tfState() = 0; // colorPoints/opacityPoints/range
  virtual int &samples() = 0;
};

} // namespace tsd::graph_nodes
```

- [ ] **Step 4: Rewrite `tsd/src/tsd/graph_nodes/TransferFunction.cpp`** to move the node into the named namespace, implement the interface, and sample from state. Replace the file body with:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
#include "tsd/graph_nodes/TransferFunctionNode.hpp"

namespace tsd::graph_nodes {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

// Named (not anonymous) so the inspector can dynamic_cast to ITransferFunctionNode.
struct TransferFunctionNode : Node, ITransferFunctionNode
{
  ParameterList params;
  tsd::core::TransferFunction state{tsd::core::makeDefaultTransferFunction()};
  int sampleCount{256};

  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("TransferFunction");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portRange()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portTF()}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  tsd::core::TransferFunction &tfState() override { return state; }
  int &samples() override { return sampleCount; }

  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency());
    auto range = std::static_pointer_cast<float2>(in.payload);
    if (!range) {
      ctx.fail("TransferFunction: missing range input");
      return;
    }
    if (sampleCount < 2) {
      ctx.fail("TransferFunction: samples must be >= 2");
      return;
    }

    auto sampled =
        GraphEditModel::sampleColormap(state.colorPoints, state.opacityPoints, sampleCount);

    auto d = std::make_shared<TransferFunctionData>();
    d->valueRange = *range;
    d->colormap = tsd::core::AnyArray(ANARI_FLOAT32_VEC4, size_t(sampleCount));
    for (int i = 0; i < sampleCount; ++i)
      d->colormap.get<float4>(size_t(i)) = sampled[size_t(i)];

    Value out;
    out.type = PortType{portTF()};
    out.residency = hostResidency();
    out.payload = d;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace tsd::graph_nodes

namespace tsd::graph_nodes {

void registerTransferFunction(NodeRegistry &reg)
{
  reg.registerType(Token("TransferFunction"),
      [] { return std::make_unique<TransferFunctionNode>(); });
}

} // namespace tsd::graph_nodes
```

> This removes the old `getPreset()` / `params.getOr<int>("samples", …)` reads — `samples` and the curve now come exclusively from typed state. `hostResidency()`, `EvalContext::input`/`setOutput`/`fail`, `Value`, and `AnyArray::get<float4>` are the same APIs the old file used; keep their includes (`Evaluator.hpp` provides `EvalContext`). If `makeDefaultTransferFunction()` produces an unset `range`, that's fine — `evaluate` overwrites `valueRange` from the input each time.

- [ ] **Step 5: Build and run the TF tests**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'TransferFunction|tf-state' --output-on-failure`
Expected: PASS. If a pre-existing TF case asserted the old `preset`/`coolToWarm` formula output, update it to drive `tfState()` instead (the preset path is gone) — note the change in your report.

- [ ] **Step 6: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/TransferFunctionNode.hpp tsd/src/tsd/graph_nodes/TransferFunction.cpp tsd/tests/test_nodes_TransferFunction.cpp
jj commit tsd/src/tsd/graph_nodes/TransferFunctionNode.hpp tsd/src/tsd/graph_nodes/TransferFunction.cpp tsd/tests/test_nodes_TransferFunction.cpp -m "feat(graph_nodes): TransferFunction node editable control points via ITransferFunctionNode"
```

---

## Task 5: Headless edit→render regression test (VisRTX)

**Files:**
- Test: `tsd/tests/test_nodes_EditRender.cpp` (new), `tsd/tests/CMakeLists.txt` (modify)

**Interfaces:**
- Consumes: `GraphEditModel` (Task 2), `ITransferFunctionNode` (Task 4), `buildVolumeSurfaceDemo` (Phase 4c), `GraphRenderBridge` (Phase 3). Reuses the `renderCounts` helper pattern from `test_tsdflow_Smoke.cpp`.
- Produces: nothing — a regression guard that a programmatic edit (rewire + TF curve change) leaves the graph renderable.

- [ ] **Step 1: Write the test** — create `tsd/tests/test_nodes_EditRender.cpp`. Copy the `makeVisRTX()` + `renderCounts()` helpers verbatim from `tsd/tests/test_tsdflow_Smoke.cpp` (same file, same anonymous namespace), then:

```cpp
SCENARIO("a programmatic edit keeps the demo renderable", "[edit-render]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  tsd::graph::NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  tsd::graph::Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);
  tsd::graph::Evaluator e(g);
  tsd::graph_nodes::GraphEditModel model(g, reg, nullptr);

  tsd::rendering::GraphRenderBridge bridge(
      g, e, tsd::core::Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(d.volumeDisplay, 0b01, true);
  bridge.setDisplay(d.surfaceDisplay, 0b10, true);
  bridge.update();

  // Edit: find the TransferFunction node and flip its curve to fully opaque white.
  for (auto id : g.nodeIds()) {
    if (auto *itf = dynamic_cast<tsd::graph_nodes::ITransferFunctionNode *>(
            g.node(id)->impl.get())) {
      itf->tfState().colorPoints = {{0.f, 1.f, 1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}};
      itf->tfState().opacityPoints = {{0.f, 1.f}, {1.f, 1.f}};
      g.markDirty(id);
    }
  }
  bridge.update();

  THEN("the volume viewport still renders color") {
    auto c0 = renderCounts(dev, bridge.world(0));
    REQUIRE(c0.color > 0);
  }

  anari::release(dev, dev);
}
```

- [ ] **Step 2: Register the test (with TIMEOUT)** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_EditRender.cpp` to the source list, then:

```cmake
add_test(NAME tsd::nodes::EditRender COMMAND ${PROJECT_NAME} "[edit-render]")
set_tests_properties(tsd::nodes::EditRender PROPERTIES TIMEOUT 300)
```

- [ ] **Step 3: Build and run**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::EditRender' --output-on-failure`
Expected: PASS (first run includes OptiX warmup — may take tens of seconds).

- [ ] **Step 4: Commit**

```bash
clang-format -i tsd/tests/test_nodes_EditRender.cpp
jj commit tsd/tests/test_nodes_EditRender.cpp tsd/tests/CMakeLists.txt -m "test(graph_nodes): headless edit->render regression via GraphEditModel + bridge"
```

---

## Task 6: `GraphEditor` window (imnodes canvas)

**Files:**
- Create: `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp`, `tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp`
- Modify: `tsd/src/tsd/ui/imgui/CMakeLists.txt`

**Interfaces:**
- Consumes: `GraphEditModel` (Task 2), `Graph::nodeIds`/`connections`/`node`, `Window` base, imnodes.
- Produces: `tsd::ui::imgui::GraphEditor(Application*, tsd::graph::Graph*, tsd::graph_nodes::GraphEditModel*, tsd::graph::NodeId* selected, bool* graphDirty, const char* name="Graph Editor")`. Writes `*selected` on node selection and sets `*graphDirty=true` after any mutation.

No automated test (GUI). Deliverable: compiles + links into `tsd_ui_imgui`.

- [ ] **Step 1: Wire CMake first** — edit `tsd/src/tsd/ui/imgui/CMakeLists.txt` by hand:
  - Add `windows/GraphEditor.cpp` to the `project_sources` list (after `windows/GraphViewport.cpp`).
  - In the `project_link_libraries(PUBLIC ...)` block, add `tsd_graph_nodes` and `tsd_ext_imnodes` (neither is currently linked; `tsd_graph_nodes` brings `GraphEditModel`/`ITransferFunctionNode`, `tsd_ext_imnodes` brings imnodes).

- [ ] **Step 2: Create `tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/windows/Window.h"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
// std
#include <map>
#include <vector>

namespace tsd::ui::imgui {

// imnodes canvas over a Graph + GraphEditModel. Owns the int<->id maps imnodes
// needs (imnodes addresses everything by int; tsd uses NodeId/ConnectionId +
// (NodeId, port Token, direction) for pins).
struct GraphEditor : public Window
{
  GraphEditor(Application *app,
      tsd::graph::Graph *graph,
      tsd::graph_nodes::GraphEditModel *model,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Graph Editor");

  void buildUI() override;

 private:
  struct PinKey
  {
    tsd::graph::NodeId node{0};
    tsd::core::Token port;
    bool isInput{false};
  };

  int pinId(tsd::graph::NodeId, tsd::core::Token port, bool isInput);
  void drawNode(tsd::graph::NodeId);
  void handleCreation();
  void handleDeletion();
  void contextMenu();

  tsd::graph::Graph *m_graph{nullptr};
  tsd::graph_nodes::GraphEditModel *m_model{nullptr};
  tsd::graph::NodeId *m_selected{nullptr};
  bool *m_graphDirty{nullptr};

  std::vector<PinKey> m_pins;                 // index+1 == imnodes pin id
  std::map<int, tsd::graph::ConnectionId> m_linkId;   // imnodes link id -> ConnectionId
  bool m_placedInitial{false};
};

} // namespace tsd::ui::imgui
```

- [ ] **Step 3: Create `tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/GraphEditor.hpp"
#include "tsd/ui/imgui/Application.h"
#include "tsd/core/Logging.hpp"
// imnodes
#include <imnodes.h>
// imgui
#include "imgui.h"

namespace tsd::ui::imgui {

using namespace tsd::graph;
using tsd::core::Token;

namespace {
int nodeImId(NodeId id) { return int(id); }                 // NodeId small in practice
constexpr unsigned int kConversionColor = IM_COL32(204, 148, 81, 255); // amber
} // namespace

GraphEditor::GraphEditor(Application *app,
    Graph *graph,
    tsd::graph_nodes::GraphEditModel *model,
    NodeId *selected,
    bool *graphDirty,
    const char *name)
    : Window(app, name),
      m_graph(graph),
      m_model(model),
      m_selected(selected),
      m_graphDirty(graphDirty)
{}

int GraphEditor::pinId(NodeId node, Token port, bool isInput)
{
  for (size_t i = 0; i < m_pins.size(); ++i) {
    const auto &p = m_pins[i];
    if (p.node == node && p.port == port && p.isInput == isInput)
      return int(i) + 1; // 0 reserved
  }
  m_pins.push_back({node, port, isInput});
  return int(m_pins.size()); // size after push == index+1
}

void GraphEditor::drawNode(NodeId id)
{
  const auto *gn = m_graph->node(id);
  if (!gn || !gn->impl)
    return;
  const auto info = gn->impl->typeInfo();

  ImNodes::BeginNode(nodeImId(id));

  ImNodes::BeginNodeTitleBar();
  ImGui::TextUnformatted(info.name.c_str()); // see Token accessor note below
  ImNodes::EndNodeTitleBar();

  for (const auto &in : info.inputs) {
    ImNodes::BeginInputAttribute(pinId(id, in.name, true), ImNodesPinShape_CircleFilled);
    ImGui::TextUnformatted(in.name.c_str());
    ImNodes::EndInputAttribute();
  }
  for (const auto &out : info.outputs) {
    ImNodes::BeginOutputAttribute(pinId(id, out.name, false), ImNodesPinShape_TriangleFilled);
    ImGui::TextUnformatted(out.name.c_str());
    ImNodes::EndOutputAttribute();
  }

  ImNodes::EndNode();
}

void GraphEditor::handleCreation()
{
  int startAttr = 0, endAttr = 0;
  if (!ImNodes::IsLinkCreated(&startAttr, &endAttr))
    return;
  // Attr ids are pin ids (index+1). Output pin is the "from"; figure direction.
  auto resolve = [&](int attr) -> PinKey * {
    const int idx = attr - 1;
    return (idx >= 0 && idx < int(m_pins.size())) ? &m_pins[size_t(idx)] : nullptr;
  };
  PinKey *a = resolve(startAttr), *b = resolve(endAttr);
  if (!a || !b)
    return;
  PinKey *outPin = a->isInput ? b : a;
  PinKey *inPin = a->isInput ? a : b;
  if (outPin->isInput || !inPin->isInput)
    return; // not an out->in pairing

  auto chk = m_model->canConnect(outPin->node, outPin->port, inPin->node, inPin->port);
  if (!chk.ok()) {
    tsd::core::logWarning("[GraphEditor] link rejected: %s", chk.detail.c_str());
    return;
  }
  m_model->connect(outPin->node, outPin->port, inPin->node, inPin->port);
  *m_graphDirty = true;
}

void GraphEditor::handleDeletion()
{
  if (!ImGui::IsKeyPressed(ImGuiKey_Delete))
    return;

  const int nLinks = ImNodes::NumSelectedLinks();
  if (nLinks > 0) {
    std::vector<int> sel(size_t(nLinks));
    ImNodes::GetSelectedLinks(sel.data());
    for (int lid : sel) {
      auto it = m_linkId.find(lid);
      if (it != m_linkId.end()) {
        m_model->disconnect(it->second);
        *m_graphDirty = true;
      }
    }
  }
  const int nNodes = ImNodes::NumSelectedNodes();
  if (nNodes > 0) {
    std::vector<int> sel(size_t(nNodes));
    ImNodes::GetSelectedNodes(sel.data());
    for (int nid : sel) {
      const NodeId id = NodeId(nid);
      if (*m_selected == id)
        *m_selected = INVALID_NODE;
      m_model->removeNode(id);
      *m_graphDirty = true;
    }
  }
}

void GraphEditor::contextMenu()
{
  if (ImNodes::IsEditorHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    ImGui::OpenPopup("addNode");
  if (ImGui::BeginPopup("addNode")) {
    const ImVec2 clickPos = ImGui::GetMousePosOnOpeningCurrentPopup();
    for (const auto &type : m_model->nodeCatalog()) {
      if (ImGui::MenuItem(type.c_str())) {
        const NodeId id = m_model->addNode(type);
        if (id != INVALID_NODE) {
          ImNodes::SetNodeScreenSpacePos(nodeImId(id), clickPos);
          *m_graphDirty = true;
        }
      }
    }
    ImGui::EndPopup();
  }
}

void GraphEditor::buildUI()
{
  ImNodes::BeginNodeEditor();

  for (const NodeId id : m_graph->nodeIds())
    drawNode(id);

  // Links: assign a stable imnodes link id per ConnectionId and color conversions.
  m_linkId.clear();
  int linkCounter = 1;
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

  contextMenu();
  ImNodes::MiniMap();
  ImNodes::EndNodeEditor();

  // After EndNodeEditor: creation, deletion, selection.
  handleCreation();
  handleDeletion();

  if (ImNodes::NumSelectedNodes() == 1) {
    int sel = 0;
    ImNodes::GetSelectedNodes(&sel);
    *m_selected = NodeId(sel);
  }
}

} // namespace tsd::ui::imgui
```

> **Token accessor:** `info.name.c_str()` / `in.name.c_str()` assume `tsd::core::Token` has `c_str()`. Verify against `tsd/core/Token.hpp`; if it's `value()` or `std::string(token)`, adjust all `.c_str()` calls on Tokens here and in Task 2 consistently, and report the accessor used.
> **imnodes context:** imnodes requires `ImNodes::CreateContext()` once before any editor and `DestroyContext()` at shutdown. Check whether the viskores demo relies on the app creating it; if `tsd_ui_imgui`'s `Application` does **not** already create an imnodes context, create one in the first `GraphEditor` ctor (guard with a static) or in the app (Task 9) before constructing the editor, and destroy it on teardown. Report what you did.

- [ ] **Step 4: Build `tsd_ui_imgui`**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui 2>&1 | tail -20`
Expected: compiles + links. Fix include paths / `Token` accessor / imnodes-context per the notes; report any deviation.

- [ ] **Step 5: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp
jj commit tsd/src/tsd/ui/imgui/windows/GraphEditor.hpp tsd/src/tsd/ui/imgui/windows/GraphEditor.cpp tsd/src/tsd/ui/imgui/CMakeLists.txt -m "feat(ui): GraphEditor imnodes canvas over GraphEditModel"
```

---

## Task 7: `Inspector` window (Any-dispatch parameter widgets)

**Files:**
- Create: `tsd/src/tsd/ui/imgui/windows/Inspector.hpp`, `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`
- Modify: `tsd/src/tsd/ui/imgui/CMakeLists.txt`

**Interfaces:**
- Consumes: `Graph::node`, `Node::parameters()`/`typeInfo()`, `ParameterList::items()`/`set`, `Any::type()`/`get`/`getString`, `tsd::graph::NodeId* selected`, `bool* graphDirty`.
- Produces: `tsd::ui::imgui::Inspector(Application*, tsd::graph::Graph*, tsd::graph::NodeId* selected, bool* graphDirty, const char* name="Inspector")`. (Task 8 extends it to embed the TF panel.)

No automated test (GUI). Deliverable: compiles + links.

- [ ] **Step 1: Wire CMake** — edit `tsd/src/tsd/ui/imgui/CMakeLists.txt` by hand: add `windows/Inspector.cpp` to `project_sources`.

- [ ] **Step 2: Create `tsd/src/tsd/ui/imgui/windows/Inspector.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/windows/Window.h"
#include "tsd/graph/Graph.hpp"

namespace tsd::ui::imgui {

struct Inspector : public Window
{
  Inspector(Application *app,
      tsd::graph::Graph *graph,
      tsd::graph::NodeId *selected,
      bool *graphDirty,
      const char *name = "Inspector");

  void buildUI() override;

 private:
  void drawParameters(tsd::graph::NodeId);

  tsd::graph::Graph *m_graph{nullptr};
  tsd::graph::NodeId *m_selected{nullptr};
  bool *m_graphDirty{nullptr};
};

} // namespace tsd::ui::imgui
```

- [ ] **Step 3: Create `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/Inspector.hpp"
#include "tsd/ui/imgui/Application.h"
// imgui
#include "imgui.h"
// anari
#include <anari/anari.h>
// std
#include <string>

namespace tsd::ui::imgui {

using namespace tsd::graph;
using tsd::core::Token;

Inspector::Inspector(Application *app,
    Graph *graph,
    NodeId *selected,
    bool *graphDirty,
    const char *name)
    : Window(app, name),
      m_graph(graph),
      m_selected(selected),
      m_graphDirty(graphDirty)
{}

void Inspector::drawParameters(NodeId id)
{
  auto *gn = m_graph->node(id);
  if (!gn || !gn->impl)
    return;
  auto &params = gn->impl->parameters();

  // Iterate a snapshot of names+types; write back via set on change.
  for (const auto &p : params.items()) {
    const Token name = p.name;
    const auto t = p.value.type();
    ImGui::PushID(name.c_str());
    if (t == ANARI_BOOL) {
      bool v = p.value.get<bool>();
      if (ImGui::Checkbox(name.c_str(), &v)) { params.set(name, v); *m_graphDirty = true; }
    } else if (t == ANARI_FLOAT32) {
      float v = p.value.get<float>();
      if (ImGui::InputFloat(name.c_str(), &v, 0.f, 0.f, "%.4f",
              ImGuiInputTextFlags_EnterReturnsTrue)) { params.set(name, v); *m_graphDirty = true; }
    } else if (t == ANARI_INT32) {
      int v = p.value.get<int>();
      if (ImGui::InputInt(name.c_str(), &v, 1, 10,
              ImGuiInputTextFlags_EnterReturnsTrue)) { params.set(name, v); *m_graphDirty = true; }
    } else if (t == ANARI_STRING) {
      std::string s = p.value.getString();
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%s", s.c_str());
      if (ImGui::InputText(name.c_str(), buf, sizeof(buf),
              ImGuiInputTextFlags_EnterReturnsTrue)) {
        params.set(name, std::string(buf)); *m_graphDirty = true;
      }
    } else {
      ImGui::Text("%s (unsupported type)", name.c_str());
    }
    ImGui::PopID();
  }
}

void Inspector::buildUI()
{
  if (*m_selected == INVALID_NODE) {
    ImGui::TextDisabled("No selection");
    return;
  }
  auto *gn = m_graph->node(*m_selected);
  if (!gn || !gn->impl) {
    ImGui::TextDisabled("No selection");
    return;
  }
  ImGui::Text("%s", gn->impl->typeInfo().name.c_str());
  ImGui::Separator();
  drawParameters(*m_selected);
}

} // namespace tsd::ui::imgui
```

> Verify `ParameterList::set<std::string>` accepts a `std::string` (the engine maps it to `ANARI_STRING` per the existing `getPreset` path; if `set` needs a `const char*` or a typed overload, match what `tsd/graph/Parameter.hpp` provides). Confirm `Any::get<bool>()`/`get<int>()`/`get<float>()` exist; if `bool`/`int` use a different accessor, adjust. Report deviations. The `Token::c_str()` note from Task 6 applies here too.

- [ ] **Step 4: Build `tsd_ui_imgui`**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui 2>&1 | tail -20`
Expected: compiles + links.

- [ ] **Step 5: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/Inspector.hpp tsd/src/tsd/ui/imgui/windows/Inspector.cpp
jj commit tsd/src/tsd/ui/imgui/windows/Inspector.hpp tsd/src/tsd/ui/imgui/windows/Inspector.cpp tsd/src/tsd/ui/imgui/CMakeLists.txt -m "feat(ui): Inspector — selection-driven Any-dispatch parameter widgets"
```

---

## Task 8: `TFCurveEditor` panel (interactive color + opacity) embedded in `Inspector`

**Files:**
- Create: `tsd/src/tsd/ui/imgui/windows/TFCurveEditor.hpp`, `tsd/src/tsd/ui/imgui/windows/TFCurveEditor.cpp`
- Modify: `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`, `tsd/src/tsd/ui/imgui/windows/Inspector.hpp`, `tsd/src/tsd/ui/imgui/CMakeLists.txt`

**Interfaces:**
- Consumes: `ITransferFunctionNode` (Task 4), `tsd::core::TransferFunction`, `tsd::core::colormap::*` presets, `GraphEditModel::sampleColormap` (for the preview strip), `Graph::markDirty`.
- Produces: `tsd::ui::imgui::TFCurveEditor` — a panel (NOT a `Window`): `TFCurveEditor(Application* app); void draw(tsd::core::TransferFunction& tf, int& samples, bool& changed);`. The Inspector instantiates one and calls `draw(...)` when the selected node is an `ITransferFunctionNode`.

No automated test (GUI). Deliverable: compiles + links; the curve math is already unit-tested via `sampleColormap` (Task 3).

- [ ] **Step 1: Wire CMake** — edit `tsd/src/tsd/ui/imgui/CMakeLists.txt` by hand: add `windows/TFCurveEditor.cpp` to `project_sources`.

- [ ] **Step 2: Create `tsd/src/tsd/ui/imgui/windows/TFCurveEditor.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"
// SDL
#include <SDL3/SDL.h>

namespace tsd::ui::imgui {

class Application;

// Embeddable panel (not a Window). Edits a core::TransferFunction's color and
// opacity control points interactively, with a colormap preview strip.
class TFCurveEditor
{
 public:
  explicit TFCurveEditor(Application *app);
  ~TFCurveEditor();

  // Renders the panel for `tf`/`samples`. Sets changed=true if the user edited.
  void draw(tsd::core::TransferFunction &tf, int &samples, bool &changed);

 private:
  void drawPresetCombo(tsd::core::TransferFunction &tf, bool &changed);
  void drawOpacityCurve(tsd::core::TransferFunction &tf, bool &changed);
  void drawColorStops(tsd::core::TransferFunction &tf, bool &changed);
  void refreshPreview(const tsd::core::TransferFunction &tf, int samples);

  Application *m_app{nullptr};
  SDL_Texture *m_preview{nullptr};
  int m_previewWidth{0};
};

} // namespace tsd::ui::imgui
```

- [ ] **Step 3: Create `tsd/src/tsd/ui/imgui/windows/TFCurveEditor.cpp`.** Implement using ImGui drawing primitives, mirroring the opacity-curve interaction in the existing `TransferFunctionEditor.cpp` (`buildUI_drawEditor`) but reading/writing `tf.opacityPoints` and `tf.colorPoints` directly, and adding interactive **color** stop editing (new work — the old editor's color came from presets only):

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/TFCurveEditor.hpp"
#include "tsd/ui/imgui/Application.h"
#include "tsd/graph_nodes/GraphEditModel.hpp"
// imgui
#include "imgui.h"
// std
#include <algorithm>
#include <vector>

namespace tsd::ui::imgui {

using tsd::core::ColorPoint;
using tsd::core::OpacityPoint;
using float3 = tsd::core::math::float3;
using float4 = tsd::core::math::float4;

namespace {
// Distribute an RGB preset evenly as positioned ColorPoints {pos, r, g, b}.
std::vector<ColorPoint> presetToColorPoints(const std::vector<float3> &rgb)
{
  std::vector<ColorPoint> pts;
  if (rgb.empty())
    return pts;
  const float denom = float(rgb.size() - 1);
  pts.reserve(rgb.size());
  for (size_t i = 0; i < rgb.size(); ++i)
    pts.push_back(ColorPoint(float(i) / denom, rgb[i].x, rgb[i].y, rgb[i].z));
  return pts;
}
} // namespace

TFCurveEditor::TFCurveEditor(Application *app) : m_app(app) {}

TFCurveEditor::~TFCurveEditor()
{
  if (m_preview)
    SDL_DestroyTexture(m_preview);
}

void TFCurveEditor::refreshPreview(const tsd::core::TransferFunction &tf, int samples)
{
  const int w = std::max(2, samples);
  if (!m_preview || m_previewWidth != w) {
    if (m_preview)
      SDL_DestroyTexture(m_preview);
    m_preview = SDL_CreateTexture(m_app->sdlRenderer(),
        SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, w, 1);
    m_previewWidth = w;
  }
  auto cm = tsd::graph_nodes::GraphEditModel::sampleColormap(
      tf.colorPoints, tf.opacityPoints, w);
  std::vector<uint32_t> px(size_t(w));
  for (int i = 0; i < w; ++i) {
    const auto c = cm[size_t(i)];
    auto to8 = [](float f) { return uint32_t(std::clamp(f, 0.f, 1.f) * 255.f); };
    px[size_t(i)] = (to8(c.w) << 24) | (to8(c.z) << 16) | (to8(c.y) << 8) | to8(c.x);
  }
  SDL_UpdateTexture(m_preview, nullptr, px.data(), w * int(sizeof(uint32_t)));
}

void TFCurveEditor::drawPresetCombo(tsd::core::TransferFunction &tf, bool &changed)
{
  struct Preset { const char *name; const std::vector<float3> *rgb; };
  static const Preset presets[] = {
      {"viridis", &tsd::core::colormap::viridis},
      {"cool to warm", &tsd::core::colormap::cool_to_warm},
      {"jet", &tsd::core::colormap::jet},
      {"inferno", &tsd::core::colormap::inferno},
      {"grayscale", &tsd::core::colormap::grayscale},
  };
  if (ImGui::BeginCombo("Load preset", "presets")) {
    for (const auto &p : presets) {
      if (ImGui::Selectable(p.name)) {
        tf.colorPoints = presetToColorPoints(*p.rgb);
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
}

void TFCurveEditor::drawOpacityCurve(tsd::core::TransferFunction &tf, bool &changed)
{
  // Canvas: x = position [0,1], y = opacity [0,1]. Drag existing points;
  // double-click empty space to add; right-click a point to delete.
  // Mirror the math in TransferFunctionEditor.cpp buildUI_drawEditor but operate
  // directly on tf.opacityPoints ({position, opacity}). Set changed=true on edit
  // and keep points sorted by .x.  (See that file for the InvisibleButton +
  // GetMouseDragDelta pattern; reproduce it here against tf.opacityPoints.)
  ImGui::TextUnformatted("Opacity");
  // ... interactive implementation (port from TransferFunctionEditor.cpp) ...
}

void TFCurveEditor::drawColorStops(tsd::core::TransferFunction &tf, bool &changed)
{
  // Interactive color stops (NEW work): list tf.colorPoints; each row shows a
  // position slider + ImGui::ColorEdit3 for {r,g,b} (the .y/.z/.w channels);
  // "Add stop" appends at 0.5, "remove" deletes. Keep sorted by .x; clamp .x to
  // [0,1]. Set changed=true on any edit.
  ImGui::TextUnformatted("Color stops");
  for (size_t i = 0; i < tf.colorPoints.size(); ++i) {
    ImGui::PushID(int(i));
    auto &cp = tf.colorPoints[i];
    float pos = cp.x;
    if (ImGui::SliderFloat("pos", &pos, 0.f, 1.f)) { cp.x = pos; changed = true; }
    float rgb[3] = {cp.y, cp.z, cp.w};
    if (ImGui::ColorEdit3("rgb", rgb)) { cp.y = rgb[0]; cp.z = rgb[1]; cp.w = rgb[2]; changed = true; }
    ImGui::PopID();
  }
  if (ImGui::Button("Add stop")) { tf.colorPoints.push_back(ColorPoint(0.5f, 1.f, 1.f, 1.f)); changed = true; }
  if (changed)
    std::sort(tf.colorPoints.begin(), tf.colorPoints.end(),
        [](const ColorPoint &a, const ColorPoint &b) { return a.x < b.x; });
}

void TFCurveEditor::draw(tsd::core::TransferFunction &tf, int &samples, bool &changed)
{
  changed = false;
  drawPresetCombo(tf, changed);
  drawColorStops(tf, changed);
  drawOpacityCurve(tf, changed);

  refreshPreview(tf, samples);
  if (m_preview)
    ImGui::Image((ImTextureID)m_preview, ImVec2(256, 24));
}

} // namespace tsd::ui::imgui
```

> The opacity-curve body is the one substantial port: copy the interaction loop from `tsd/src/tsd/ui/imgui/windows/TransferFunctionEditor.cpp` `buildUI_drawEditor` (the `InvisibleButton` + `IsItemActive` + `GetMouseDragDelta` + clamp pattern, lines ~110-200), retargeting `m_tfnOpacityPoints[i].x/.y` → `tf.opacityPoints[i].x/.y`. Set `changed=true` on any drag/add/delete. Verify `Application::sdlRenderer()` exists (it's used by `CopyToSDLTexturePass`/`GraphViewport`); the SDL texture format/byte order may need to match what other windows use (`SDL_PIXELFORMAT_RGBA32` is the same one `TransferFunctionEditor` uses for its palette — confirm and match). Report any pixel-format adjustment.

- [ ] **Step 4: Embed in `Inspector`.** Modify `Inspector.hpp`: add `#include "tsd/ui/imgui/windows/TFCurveEditor.hpp"`, `#include "tsd/graph_nodes/TransferFunctionNode.hpp"`, and a member `std::unique_ptr<TFCurveEditor> m_tfEditor;` (include `<memory>`). In `Inspector.cpp` `buildUI()`, after printing the title/separator, branch:

```cpp
  if (auto *itf = dynamic_cast<tsd::graph_nodes::ITransferFunctionNode *>(gn->impl.get())) {
    if (!m_tfEditor)
      m_tfEditor = std::make_unique<TFCurveEditor>(m_app);
    bool changed = false;
    m_tfEditor->draw(itf->tfState(), itf->samples(), changed);
    if (changed) { m_graph->markDirty(*m_selected); *m_graphDirty = true; }
  } else {
    drawParameters(*m_selected);
  }
```

(`m_app` is the `Window` base's protected `Application*`. Construct the editor lazily so the SDL renderer is available.)

- [ ] **Step 5: Build `tsd_ui_imgui`**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui 2>&1 | tail -20`
Expected: compiles + links.

- [ ] **Step 6: Commit**

```bash
clang-format -i tsd/src/tsd/ui/imgui/windows/TFCurveEditor.hpp tsd/src/tsd/ui/imgui/windows/TFCurveEditor.cpp tsd/src/tsd/ui/imgui/windows/Inspector.hpp tsd/src/tsd/ui/imgui/windows/Inspector.cpp
jj commit tsd/src/tsd/ui/imgui/windows/TFCurveEditor.hpp tsd/src/tsd/ui/imgui/windows/TFCurveEditor.cpp tsd/src/tsd/ui/imgui/windows/Inspector.hpp tsd/src/tsd/ui/imgui/windows/Inspector.cpp tsd/src/tsd/ui/imgui/CMakeLists.txt -m "feat(ui): TFCurveEditor panel (color+opacity) embedded in Inspector"
```

---

## Task 9: `tsdFlow` app integration

**Files:**
- Modify: `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`

**Interfaces:**
- Consumes: `GraphEditor` (Task 6), `Inspector` (Task 8), `GraphEditModel` (Task 2), the existing `GraphViewport`/`Log`/bridge wiring (Phase 4c).
- Produces: an app hosting GraphEditor + Inspector + 2 GraphViewports + Log, with shared selection + dirty-coalesced re-render and display-node bridge wiring.

No automated test (GUI). Deliverable: `tsdFlow` builds + the full suite stays green; manual checklist recorded.

- [ ] **Step 1: Add editor state + windows to the app.** In `tsd/apps/interactive/tsdFlow/tsdFlow.cpp`:
  - Add includes: `#include "tsd/ui/imgui/windows/GraphEditor.hpp"`, `#include "tsd/ui/imgui/windows/Inspector.hpp"`, `#include "tsd/graph_nodes/GraphEditModel.hpp"`.
  - Add members: `std::unique_ptr<tsd::graph_nodes::GraphEditModel> m_model;`, `tsd::graph::NodeId m_selected{0};`, `bool m_graphDirty{false};`.
  - In `setupWindows()`, after the graph/registry/bridge are built (the existing 4c code), construct the model and the two new windows:

```cpp
    m_model = std::make_unique<tsd::graph_nodes::GraphEditModel>(
        m_graph, m_registry, /*conversions=*/nullptr);

    windows.emplace_back(new ui::GraphEditor(
        this, &m_graph, m_model.get(), &m_selected, &m_graphDirty, "Graph Editor"));
    windows.emplace_back(new ui::Inspector(
        this, &m_graph, &m_selected, &m_graphDirty, "Inspector"));
```

  (Keep the existing two `GraphViewport`s and `Log`.)

- [ ] **Step 2: Coalesced re-render + display wiring.** Add a per-frame check. If the app has a frame hook (e.g. `uiFrameStart()` used for the menu bar), at the end of it:

```cpp
    if (m_graphDirty) {
      // Keep the bridge's display set in sync with display nodes present in the graph.
      syncDisplays();
      m_bridge->update();
      m_graphDirty = false;
    }
```

Add a helper `syncDisplays()` to the app:

```cpp
  void syncDisplays()
  {
    // Register any DisplayVolume/DisplaySurface node not yet known to the bridge
    // (default new ones to viewport 0's mask); the bridge prunes removed nodes
    // when removeDisplay was called at delete time (see GraphEditor → removeNode).
    for (auto id : m_graph.nodeIds()) {
      auto *gn = m_graph.node(id);
      if (!gn || !gn->impl)
        continue;
      const auto cat = gn->impl->typeInfo().name;
      const bool isDisplay = (cat == tsd::core::Token("DisplayVolume")
          || cat == tsd::core::Token("DisplaySurface"));
      if (isDisplay && m_knownDisplays.insert(id).second)
        m_bridge->setDisplay(id, 0b01, true);
    }
  }
```

Add member `std::set<tsd::graph::NodeId> m_knownDisplays;` (and `#include <set>`). Seed it with the demo's two display ids after the initial `setDisplay` calls so they aren't re-registered.

> **Display removal:** `GraphEditor`'s delete path calls `m_model->removeNode(id)`, which erases the node from the `Graph`. Wire the bridge prune here: in the app, before/at delete you need `m_bridge->removeDisplay(id)`. Simplest robust approach: in `syncDisplays()`, also drop bridge displays whose node id is no longer in `m_graph.nodeIds()` and is in `m_knownDisplays`:

```cpp
    // prune
    std::vector<tsd::graph::NodeId> gone;
    auto ids = m_graph.nodeIds();
    for (auto id : m_knownDisplays)
      if (std::find(ids.begin(), ids.end(), id) == ids.end())
        gone.push_back(id);
    for (auto id : gone) { m_bridge->removeDisplay(id); m_knownDisplays.erase(id); }
```

(Include `<algorithm>`.) This keeps the bridge's display set consistent regardless of which display nodes were added or deleted on the canvas.

- [ ] **Step 3: Update the default layout** to include the new windows. Extend `getDefaultLayout()` so "Graph Editor" docks (e.g. a bottom-left dock), "Inspector" docks right, and the two viewports + Log keep their places. Reuse the docking-INI approach from the current `tsdFlow.cpp` (the fixed dockspace id `0x80F5B4C5` / window id `0x079D3A04`). A working split: left column = Graph Editor (top) + Inspector (bottom); center = Volume | Surface viewports; bottom strip = Log. Add `[Window][Graph Editor]` and `[Window][Inspector]` entries with `DockId`s matching the node tree. (If the precise INI is fiddly, dock them to any existing node — the app still works; refine during manual test.)

- [ ] **Step 4: Build the app**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdFlow 2>&1 | tail -20`
Expected: links. Fix any signature mismatches against the windows' actual ctors (Tasks 6–8); report changes.

- [ ] **Step 5: Full suite gate**

Run:
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests --parallel
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo --output-on-failure
```
Expected: all green — prior suite + the 4 new tests (`tsd::graph::EditApis`, `tsd::nodes::GraphEditModel`, `tsd::nodes::TfnSampling`, `tsd::nodes::EditRender`) and the updated TF test. Report the summary line.

- [ ] **Step 6: Confirm `.envrc` uncommitted**

Run: `jj status`
Expected: working copy shows `A .envrc` (and nothing from this task uncommitted after the commit below). NEVER commit `.envrc`.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/apps/interactive/tsdFlow/tsdFlow.cpp
jj commit tsd/apps/interactive/tsdFlow/tsdFlow.cpp -m "feat(app): tsdFlow hosts GraphEditor + Inspector with live edit re-render"
```

- [ ] **Step 8: Record the manual test checklist** (GUI not CI-tested) in the task report:
  - `tsdFlow` launches; Graph Editor shows the 6 demo nodes wired; Inspector + 2 viewports + Log dock sensibly.
  - Right-click canvas → add a node from the catalog; it appears at the cursor.
  - Drag a valid pin→pin link: it forms. Drag an incompatible/cyclic link: rejected (Log message).
  - A link needing a conversion (if a conversion is registered) renders amber.
  - Select a node → Inspector shows its params; editing a param re-renders the viewports.
  - Select the TransferFunction node → TF curve editor appears; drag opacity points / edit a color stop / load a preset → the volume re-renders.
  - Delete a node and a link via selection + Delete; deleting a Display node removes it from its viewport.

---

## Phase 4d completion checklist

- [ ] Additive engine APIs (`Graph::canConnect`, `Graph::nodeIds`, `NodeRegistry::types`) + tests (Task 1)
- [ ] `GraphEditModel` edit/validation/classification/catalog + tests (Task 2)
- [ ] `GraphEditModel::sampleColormap` + tests (Task 3)
- [ ] `TransferFunction` node editable state via `ITransferFunctionNode` + tests (Task 4)
- [ ] Headless edit→render regression (Task 5)
- [ ] `GraphEditor` imnodes canvas (Task 6)
- [ ] `Inspector` Any-dispatch params (Task 7)
- [ ] `TFCurveEditor` panel embedded in Inspector (Task 8)
- [ ] `tsdFlow` integration: windows, selection, dirty-coalesce, display sync (Task 9)
- [ ] Full suite green; `.envrc` uncommitted; manual checklist recorded

## Out of scope (per spec)

Transfer/residency link feedback (4b); per-display→viewport assignment UI; node-position & graph persistence, TF load/save (Phase 5); undo/redo (4e); bounded-slider param metadata; multi-select editing, copy/paste, node search/grouping.

## Self-review notes

- **Spec coverage:** every spec component maps to a task — engine APIs (T1), GraphEditModel incl. sampleColormap (T2/T3), TransferFunction model (T4), render regression (T5), GraphEditor (T6), Inspector (T7), TFCurveEditor panel (T8), app integration + dirty-coalesce + bridge prune/add wiring (T9). The conversion-feedback (amber link + reject reason) is in T6; the "plain inputs, no sliders" decision is in T7; "full color+opacity" is in T8.
- **Type consistency:** `GraphEditModel` ctor/op signatures, `ConnectCheck`/`LinkKind`, `ITransferFunctionNode::tfState()`/`samples()`, and the window ctors `(Application*, Graph*, [model], NodeId* selected, bool* graphDirty)` are used identically across tasks.
- **Tested seam:** all logic that can break silently (validation, cycle/conversion classification, colormap sampling, node state→eval) is unit-tested headlessly (T1–T5); the three windows are build-verified + manually checked (T9 checklist), consistent with prior phases.
- **Flagged for the implementer (verify against source, adjust minimally, report):** `tsd::core::Token` string accessor (`c_str()` vs `value()`), whether an imnodes context must be created by the app, `ParameterList::set<std::string>` form, `Any::get<bool/int>` accessors, the SDL pixel format for the TF preview, and the exact docking-INI for the new windows.
