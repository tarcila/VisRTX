# tsdFlow Phase 4a — Headless Node Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans, task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A core-only `tsd_graph_nodes` catalog library with six procedural nodes (GenerateNoiseVolume, ScalarRange, TransferFunction, DisplayVolume, BoundingBox, DisplaySurface) that drive the "generate → process → display" pipeline end-to-end, rendered headlessly via the Phase 3 `GraphRenderBridge` with VisRTX.

**Architecture:** A new static lib `tsd_graph_nodes` depending only on `tsd_graph`. Node classes live in anonymous namespaces in per-node `.cpp` files; each exposes a `registerXxx(NodeRegistry&)` free function, and `registerBuiltinNodes()` calls them all (explicit registration — reliable under static linking). Nodes exchange core-only descriptors (`Field`, `range`, `transferFunction`, `surface`) and the Phase 3 `Renderable`; display nodes emit `renderable`, which the Phase 3 bridge already turns into ANARI.

**Tech Stack:** C++17, `tsd_graph` (Phases 1–2) + `Renderable`/`GraphRenderBridge` (Phase 3), `tsd::core` (AnyArray/Token/Any/math), ANARI + VisRTX (render smoke), Catch2, CTest, jj.

**Spec:** `docs/superpowers/specs/2026-06-17-tsdflow-phase4a-node-catalog-design.md`.

## Global Constraints

- Version control is **jj**, not git. A "commit" runs `jj commit <explicit paths> -m "..."` — **never** a bare `jj commit` (an unrelated `.envrc` must stay uncommitted).
- Build tree `_out/_cmake` (Ninja Multi-Config, RelWithDebInfo). No new `build/` dir.
  - Build: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - Test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- File header: `// Copyright 2026 NVIDIA Corporation` / `// SPDX-License-Identifier: Apache-2.0`; `#pragma once` for headers. `clang-format -i` touched files before committing.
- Namespaces: engine types `tsd::graph`; catalog types/nodes `tsd::graph_nodes`.
- Register a test: add its source to `project_add_executable(...)` in `tsd/tests/CMakeLists.txt` AND an `add_test(NAME ... COMMAND ${PROJECT_NAME} "[tag]")` line. Render tests also get `set_tests_properties(<name> PROPERTIES TIMEOUT 300)` (VisRTX/OptiX warmup).
- 4a is host-residency only (no CUDA). No `tsd_io`/`tsd_scene` dependency in `tsd_graph_nodes`.

## File structure

| File | Responsibility |
|------|----------------|
| `tsd/src/tsd/graph_nodes/CMakeLists.txt` | `tsd_graph_nodes` static lib, links `tsd_graph` |
| `tsd/src/tsd/graph_nodes/Descriptors.hpp` | `Field`, `TransferFunctionData`, `SurfaceData` + port-type Token helpers |
| `tsd/src/tsd/graph_nodes/BuiltinNodes.hpp` | `registerBuiltinNodes()` declarations |
| `tsd/src/tsd/graph_nodes/BuiltinNodes.cpp` | `registerBuiltinNodes()` calls each node's registrar |
| `tsd/src/tsd/graph_nodes/GenerateNoiseVolume.cpp` | source node |
| `tsd/src/tsd/graph_nodes/ScalarRange.cpp` | processor |
| `tsd/src/tsd/graph_nodes/TransferFunction.cpp` | processor |
| `tsd/src/tsd/graph_nodes/DisplayVolume.cpp` | sink |
| `tsd/src/tsd/graph_nodes/BoundingBox.cpp` | processor (field→surface) |
| `tsd/src/tsd/graph_nodes/DisplaySurface.cpp` | sink |
| `tsd/src/tsd/graph/Evaluator.hpp/.cpp` | MODIFY: add `EvalContext::fail(std::string)` |
| `tsd/tests/test_nodes_*.cpp` | per-node + wiring + render tests |
| `tsd/src/tsd/CMakeLists.txt`, `tsd/tests/CMakeLists.txt` | wiring |

Node classes are anonymous-namespace (registry-created by Token); tests create them
via `GlobalNodeRegistry().create(Token("..."))` after `registerBuiltinNodes()`,
exercising the same path the future UI menu will use.

---

## Task 0: `EvalContext::fail` + scaffold `tsd_graph_nodes` (descriptors, lib, registry)

**Files:**
- Modify: `tsd/src/tsd/graph/Evaluator.hpp`, `tsd/src/tsd/graph/Evaluator.cpp`
- Create: `tsd/src/tsd/graph_nodes/{CMakeLists.txt,Descriptors.hpp,BuiltinNodes.hpp,BuiltinNodes.cpp}`
- Modify: `tsd/src/tsd/CMakeLists.txt`, `tsd/tests/CMakeLists.txt`
- Test: `tsd/tests/test_nodes_Scaffold.cpp`

**Interfaces:**
- Produces: `tsd::graph::EvalContext::fail(const std::string &msg)` — sets the current node to `Error` with `msg`. `tsd::graph_nodes::registerBuiltinNodes()` and `registerBuiltinNodes(tsd::graph::NodeRegistry&)`. Descriptors `Field`, `TransferFunctionData`, `SurfaceData`.

- [ ] **Step 1: Add `EvalContext::fail` declaration.** In `tsd/src/tsd/graph/Evaluator.hpp`, in `class EvalContext` public section after `cancelled()`:
```cpp
  // Mark this node as failed (Error state) with a message; evaluate() should
  // return immediately after calling this. The evaluator short-circuits
  // downstream consumers.
  void fail(const std::string &msg);
```
(`<string>` is already included.)

- [ ] **Step 2: Define it.** In `tsd/src/tsd/graph/Evaluator.cpp`, next to the other `EvalContext` methods:
```cpp
void EvalContext::fail(const std::string &msg)
{
  m_self.state = EvalState::Error;
  m_self.error = msg;
}
```

- [ ] **Step 3: Write the failing scaffold test** `tsd/tests/test_nodes_Scaffold.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::hostResidency;

namespace {

// A node that fails on evaluate, to exercise EvalContext::fail.
struct AlwaysFails : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("AlwaysFails");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override { ctx.fail("boom"); }
};

} // namespace

SCENARIO("EvalContext::fail marks the node Error", "[nodes-scaffold]")
{
  Graph g;
  auto n = g.addNode(std::make_unique<AlwaysFails>());
  Evaluator e(g);
  THEN("pull fails and the node carries the message")
  {
    REQUIRE_FALSE(e.pull(n));
    REQUIRE(g.node(n)->state == tsd::graph::EvalState::Error);
    REQUIRE(g.node(n)->error == "boom");
  }
}

SCENARIO("tsd_graph_nodes registry is callable", "[nodes-scaffold]")
{
  tsd::graph::NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  THEN("the registry exists (node set filled in later tasks)")
  {
    // No builtins registered yet at scaffold time; just confirm linkage.
    REQUIRE_FALSE(reg.isRegistered(Token("DoesNotExist")));
  }
}
```
Register `add_test(NAME tsd::nodes::Scaffold COMMAND ${PROJECT_NAME} "[nodes-scaffold]")`.

- [ ] **Step 4: Create `tsd/src/tsd/graph_nodes/Descriptors.hpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/AnyArray.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/graph/Renderable.hpp" // RenderableParams, Renderable
// std
#include <memory>

namespace tsd::graph_nodes {

// A structured scalar field flowing on a "field" port.
struct Field
{
  tsd::core::Token subtype{tsd::core::Token("structuredRegular")};
  tsd::core::math::uint3 dims{0u, 0u, 0u};
  tsd::core::math::float3 origin{-1.f, -1.f, -1.f};
  tsd::core::math::float3 spacing{1.f, 1.f, 1.f};
  tsd::core::AnyArray data; // dims.x*dims.y*dims.z scalars (ANARI_FLOAT32)
};

// A 1D transfer function flowing on a "transferFunction" port.
struct TransferFunctionData
{
  tsd::core::AnyArray colormap; // float4 RGBA, `samples` entries
  tsd::core::math::float2 valueRange{0.f, 1.f};
};

// A renderable-able surface flowing on a "surface" port.
struct SurfaceData
{
  tsd::core::Token geomSubtype{tsd::core::Token("triangle")};
  tsd::graph::RenderableParams prim;       // geometry params
  tsd::graph::RenderableParams appearance; // material params
};

// Port-type Token helpers (a Token wrapper is created at the PortSpec site).
inline tsd::core::Token portField() { return tsd::core::Token("field"); }
inline tsd::core::Token portRange() { return tsd::core::Token("range"); }
inline tsd::core::Token portTF() { return tsd::core::Token("transferFunction"); }
inline tsd::core::Token portSurface() { return tsd::core::Token("surface"); }
inline tsd::core::Token portRenderable() { return tsd::core::Token("renderable"); }

} // namespace tsd::graph_nodes
```

- [ ] **Step 5: Create `tsd/src/tsd/graph_nodes/BuiltinNodes.hpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/NodeRegistry.hpp"

namespace tsd::graph_nodes {

// Register all built-in catalog node types into `reg`. Explicit registration
// (reliable under static linking — no reliance on static-init side effects).
void registerBuiltinNodes(tsd::graph::NodeRegistry &reg);

// Convenience: register into the process-global registry.
void registerBuiltinNodes();

// Per-node registrars (defined in each node's .cpp; called by the above).
void registerGenerateNoiseVolume(tsd::graph::NodeRegistry &reg);
void registerScalarRange(tsd::graph::NodeRegistry &reg);
void registerTransferFunction(tsd::graph::NodeRegistry &reg);
void registerDisplayVolume(tsd::graph::NodeRegistry &reg);
void registerBoundingBox(tsd::graph::NodeRegistry &reg);
void registerDisplaySurface(tsd::graph::NodeRegistry &reg);

} // namespace tsd::graph_nodes
```

- [ ] **Step 6: Create `tsd/src/tsd/graph_nodes/BuiltinNodes.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"

namespace tsd::graph_nodes {

void registerBuiltinNodes(tsd::graph::NodeRegistry &reg)
{
  registerGenerateNoiseVolume(reg);
  registerScalarRange(reg);
  registerTransferFunction(reg);
  registerDisplayVolume(reg);
  registerBoundingBox(reg);
  registerDisplaySurface(reg);
}

void registerBuiltinNodes()
{
  registerBuiltinNodes(tsd::graph::GlobalNodeRegistry());
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 7: Create `tsd/src/tsd/graph_nodes/CMakeLists.txt`:**
```cmake
project(tsd_graph_nodes)

project_add_library(STATIC)

project_sources(PRIVATE
  BuiltinNodes.cpp
  GenerateNoiseVolume.cpp
  ScalarRange.cpp
  TransferFunction.cpp
  DisplayVolume.cpp
  BoundingBox.cpp
  DisplaySurface.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_graph)
```

> The six node `.cpp` files don't exist yet; later tasks create them. To keep the
> build green at Task 0, create EMPTY stubs for the five not built here — each a
> file with just the copyright header and `// implemented in a later task`. The
> matching `registerXxx` functions are declared in `BuiltinNodes.hpp`; provide a
> temporary empty definition for each in its stub so `BuiltinNodes.cpp` links:
> e.g. `GenerateNoiseVolume.cpp` initially contains
> `#include "tsd/graph_nodes/BuiltinNodes.hpp"\nnamespace tsd::graph_nodes { void registerGenerateNoiseVolume(tsd::graph::NodeRegistry &) {} }`.
> Each later task replaces its stub with the real node + registrar.

- [ ] **Step 8: Wire CMake.** In `tsd/src/tsd/CMakeLists.txt`, after `add_subdirectory(graph)` add `add_subdirectory(graph_nodes)`. In `tsd/tests/CMakeLists.txt`, add `test_nodes_Scaffold.cpp` to the executable sources, add `tsd_graph_nodes` to `project_link_libraries(PRIVATE ...)`, and add the `add_test` line.

- [ ] **Step 9: Build + run.**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::Scaffold' --output-on-failure
```
Expected: PASS (both scenarios). Also re-run `tsd::graph::Evaluator` to confirm the `fail` addition didn't regress it.

- [ ] **Step 10: Commit.**
```bash
clang-format -i tsd/src/tsd/graph/Evaluator.hpp tsd/src/tsd/graph/Evaluator.cpp tsd/src/tsd/graph_nodes/*.hpp tsd/src/tsd/graph_nodes/*.cpp tsd/tests/test_nodes_Scaffold.cpp
jj commit tsd/src/tsd/graph/Evaluator.hpp tsd/src/tsd/graph/Evaluator.cpp tsd/src/tsd/graph_nodes/ tsd/src/tsd/CMakeLists.txt tsd/tests/test_nodes_Scaffold.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): scaffold catalog lib + descriptors + EvalContext::fail"
```

---

## Task 1: `GenerateNoiseVolume` (source)

**Files:**
- Replace stub: `tsd/src/tsd/graph_nodes/GenerateNoiseVolume.cpp`
- Test: `tsd/tests/test_nodes_GenerateNoiseVolume.cpp`

**Interfaces:**
- Produces: node type token `"GenerateNoiseVolume"`, output port `out:field`, params `dims:uint3`(default 32³), `seed:int`(0). Output `Value.payload` is `std::shared_ptr<Field>`, host residency, type `field`.

- [ ] **Step 1: Write the failing test** `tsd/tests/test_nodes_GenerateNoiseVolume.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::hostResidency;
using tsd::graph_nodes::Field;
using uint3 = tsd::core::math::uint3;

namespace {
tsd::graph::NodeId addBuiltin(Graph &g, const char *type)
{
  static tsd::graph::NodeRegistry reg = [] {
    tsd::graph::NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(type)));
}
} // namespace

SCENARIO("GenerateNoiseVolume emits a deterministic field", "[nodes-noise]")
{
  Graph g;
  auto n = addBuiltin(g, "GenerateNoiseVolume");
  g.node(n)->impl->parameters().set(Token("dims"), uint3(16u, 16u, 16u));
  g.node(n)->impl->parameters().set(Token("seed"), 7);
  Evaluator e(g);

  WHEN("pulled")
  {
    REQUIRE(e.pull(n));
    auto out = e.output(n, Token("out"), hostResidency());
    REQUIRE(out != nullptr);
    auto f = std::static_pointer_cast<Field>(out->payload);
    THEN("the field has the requested dims and matching data size")
    {
      REQUIRE(f->dims.x == 16u);
      REQUIRE(f->data.size() == 16u * 16u * 16u);
      REQUIRE(f->data.elementType() == ANARI_FLOAT32);
    }
  }

  WHEN("pulled again with the same seed (fresh graph)")
  {
    Graph g2;
    auto n2 = addBuiltin(g2, "GenerateNoiseVolume");
    g2.node(n2)->impl->parameters().set(Token("dims"), uint3(16u, 16u, 16u));
    g2.node(n2)->impl->parameters().set(Token("seed"), 7);
    Evaluator e2(g2);
    e.pull(n);
    e2.pull(n2);
    THEN("the data matches element-for-element (determinism)")
    {
      auto a = std::static_pointer_cast<Field>(
          e.output(n, Token("out"), hostResidency())->payload);
      auto b = std::static_pointer_cast<Field>(
          e2.output(n2, Token("out"), hostResidency())->payload);
      bool same = true;
      for (size_t i = 0; i < a->data.size(); ++i)
        same = same && (a->data.get<float>(i) == b->data.get<float>(i));
      REQUIRE(same);
    }
  }

  WHEN("given zero dims")
  {
    auto bad = addBuiltin(g, "GenerateNoiseVolume");
    g.node(bad)->impl->parameters().set(Token("dims"), uint3(0u, 4u, 4u));
    Evaluator e3(g);
    THEN("the pull fails with Error")
    {
      REQUIRE_FALSE(e3.pull(bad));
      REQUIRE(g.node(bad)->state == tsd::graph::EvalState::Error);
    }
  }
}
```
Register `add_test(NAME tsd::nodes::Noise COMMAND ${PROJECT_NAME} "[nodes-noise]")`.

- [ ] **Step 2: Build, confirm FAIL** (registry creates nullptr → `addNode(nullptr)` / pull fails; the stub registrar registers nothing).

- [ ] **Step 3: Replace `tsd/src/tsd/graph_nodes/GenerateNoiseVolume.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph/Evaluator.hpp"
// std
#include <cmath>
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using uint3 = tsd::core::math::uint3;
using float3 = tsd::core::math::float3;

struct GenerateNoiseVolume : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("GenerateNoiseVolume");
    i.category = Token("source");
    i.outputs.push_back({Token("out"), PortType{portField()}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    const uint3 dims = params.getOr<uint3>(Token("dims"), uint3(32u, 32u, 32u));
    const int seed = params.getOr<int>(Token("seed"), 0);
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u) {
      ctx.fail("GenerateNoiseVolume: dims must be > 0 on each axis");
      return;
    }
    auto f = std::make_shared<Field>();
    f->dims = dims;
    f->origin = float3(-1.f, -1.f, -1.f);
    f->spacing =
        float3(2.f / dims.x, 2.f / dims.y, 2.f / dims.z);
    f->data = tsd::core::AnyArray(
        ANARI_FLOAT32, size_t(dims.x) * dims.y * dims.z);

    // Deterministic procedural field: a centred radial blob perturbed by seed.
    const float sx = float(seed) * 0.137f;
    size_t idx = 0;
    for (uint32_t z = 0; z < dims.z; ++z)
      for (uint32_t y = 0; y < dims.y; ++y)
        for (uint32_t x = 0; x < dims.x; ++x, ++idx) {
          const float px = (float(x) / float(dims.x - 1 ? dims.x - 1 : 1)) * 2.f - 1.f;
          const float py = (float(y) / float(dims.y - 1 ? dims.y - 1 : 1)) * 2.f - 1.f;
          const float pz = (float(z) / float(dims.z - 1 ? dims.z - 1 : 1)) * 2.f - 1.f;
          const float r = std::sqrt(px * px + py * py + pz * pz);
          float v = 1.f - r;
          v += 0.1f * std::sin(px * 6.f + sx) * std::sin(py * 6.f + sx);
          f->data.get<float>(idx) = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        }

    Value out;
    out.type = PortType{portField()};
    out.residency = hostResidency();
    out.payload = f;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerGenerateNoiseVolume(NodeRegistry &reg)
{
  reg.registerType(
      Token("GenerateNoiseVolume"), [] { return std::make_unique<GenerateNoiseVolume>(); });
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 4: Build + run** `ctest ... -R 'tsd::nodes::Noise' --output-on-failure` → PASS (dims/size, determinism, zero-dims error).

- [ ] **Step 5: Commit.**
```bash
clang-format -i tsd/src/tsd/graph_nodes/GenerateNoiseVolume.cpp tsd/tests/test_nodes_GenerateNoiseVolume.cpp
jj commit tsd/src/tsd/graph_nodes/GenerateNoiseVolume.cpp tsd/tests/test_nodes_GenerateNoiseVolume.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): GenerateNoiseVolume source node"
```

---

## Task 2: `ScalarRange` (processor)

**Files:**
- Replace stub: `tsd/src/tsd/graph_nodes/ScalarRange.cpp`
- Test: `tsd/tests/test_nodes_ScalarRange.cpp`

**Interfaces:**
- Consumes: a `field` input (`std::shared_ptr<Field>`).
- Produces: token `"ScalarRange"`, input `in:field`, output `out:range`; output payload is `std::shared_ptr<tsd::core::math::float2>` (min,max), type `range`, host residency. Sets `contentTag`.

- [ ] **Step 1: Write the failing test** `tsd/tests/test_nodes_ScalarRange.cpp`. Build a graph: a small in-test source node emitting a `Field` with known values {0,1,2,3} → connect to `ScalarRange` → pull → assert range == {0,3}.
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using float2 = tsd::core::math::float2;

namespace {

struct KnownField : Node // emits a 2x2x1 field with values 0,1,2,3
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("KnownField");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::make_shared<Field>();
    f->dims = tsd::core::math::uint3(2u, 2u, 1u);
    f->data = tsd::core::AnyArray(ANARI_FLOAT32, 4);
    for (int k = 0; k < 4; ++k)
      f->data.get<float>(k) = float(k);
    Value v;
    v.type = PortType{Token("field")};
    v.residency = hostResidency();
    v.payload = f;
    ctx.setOutput(Token("out"), v);
  }
};

NodeId addBuiltin(Graph &g, const char *type)
{
  static NodeRegistry reg = [] {
    NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(type)));
}

} // namespace

SCENARIO("ScalarRange computes field min/max", "[nodes-range]")
{
  Graph g;
  auto src = g.addNode(std::make_unique<KnownField>());
  auto sr = addBuiltin(g, "ScalarRange");
  g.connect(src, Token("out"), sr, Token("in"));
  Evaluator e(g);

  WHEN("pulled")
  {
    REQUIRE(e.pull(sr));
    auto out = e.output(sr, Token("out"), hostResidency());
    REQUIRE(out != nullptr);
    auto r = std::static_pointer_cast<float2>(out->payload);
    THEN("the range is {0,3}")
    {
      REQUIRE(r->x == 0.f);
      REQUIRE(r->y == 3.f);
    }
  }
}
```
Register `add_test(NAME tsd::nodes::Range COMMAND ${PROJECT_NAME} "[nodes-range]")`.

- [ ] **Step 2: Build, confirm FAIL** (ScalarRange registrar is an empty stub → create returns null).

- [ ] **Step 3: Replace `tsd/src/tsd/graph_nodes/ScalarRange.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph/Evaluator.hpp"
// std
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;

struct ScalarRange : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ScalarRange");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portField()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portRange()}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency());
    auto f = std::static_pointer_cast<Field>(in.payload);
    if (!f) {
      ctx.fail("ScalarRange: missing field input");
      return;
    }
    const size_t expected = size_t(f->dims.x) * f->dims.y * f->dims.z;
    if (f->data.size() == 0 || f->data.size() != expected) {
      ctx.fail("ScalarRange: field data size does not match dims");
      return;
    }
    float lo = f->data.get<float>(0), hi = lo;
    for (size_t i = 1; i < f->data.size(); ++i) {
      const float v = f->data.get<float>(i);
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
    auto r = std::make_shared<float2>(lo, hi);
    Value out;
    out.type = PortType{portRange()};
    out.residency = hostResidency();
    out.payload = r;
    // cheap host scalar: a content tag lets downstream skip when range unchanged
    out.contentTag = (uint64_t(std::hash<float>{}(lo)) << 32)
        ^ uint64_t(std::hash<float>{}(hi));
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerScalarRange(NodeRegistry &reg)
{
  reg.registerType(Token("ScalarRange"), [] { return std::make_unique<ScalarRange>(); });
}

} // namespace tsd::graph_nodes
```
(`<functional>` for `std::hash` — add the include.)

- [ ] **Step 4: Build + run** `ctest ... -R 'tsd::nodes::Range' --output-on-failure` → PASS.

- [ ] **Step 5: Commit.**
```bash
clang-format -i tsd/src/tsd/graph_nodes/ScalarRange.cpp tsd/tests/test_nodes_ScalarRange.cpp
jj commit tsd/src/tsd/graph_nodes/ScalarRange.cpp tsd/tests/test_nodes_ScalarRange.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): ScalarRange processor node"
```

---

## Task 3: `TransferFunction` (processor)

**Files:**
- Replace stub: `tsd/src/tsd/graph_nodes/TransferFunction.cpp`
- Test: `tsd/tests/test_nodes_TransferFunction.cpp`

**Interfaces:**
- Consumes: `range` input (`std::shared_ptr<tsd::core::math::float2>`).
- Produces: token `"TransferFunction"`, input `in:range`, output `out:transferFunction`; payload `std::shared_ptr<TransferFunctionData>` (colormap float4 of `samples`, valueRange=input). Params `preset:token`("coolToWarm"/"grayscale"), `samples:int`(256).

- [ ] **Step 1: Write the failing test** — a graph: in-test node emitting `range {0,1}` → `TransferFunction` (preset grayscale, samples 8) → pull → assert colormap is ANARI_FLOAT32_VEC4 size 8, valueRange {0,1}, and grayscale endpoints (entry 0 ≈ black, entry 7 ≈ white). (Write the in-test `EmitRange` node analogous to `KnownField`.)
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include <memory>
#include <string>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::TransferFunctionData;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

namespace {
struct EmitRange : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("EmitRange");
    i.outputs.push_back({Token("out"), PortType{Token("range")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    Value v;
    v.type = PortType{Token("range")};
    v.residency = hostResidency();
    v.payload = std::make_shared<float2>(0.f, 1.f);
    ctx.setOutput(Token("out"), v);
  }
};
NodeId addBuiltin(Graph &g, const char *t)
{
  static NodeRegistry reg = [] { NodeRegistry r; tsd::graph_nodes::registerBuiltinNodes(r); return r; }();
  return g.addNode(reg.create(Token(t)));
}
} // namespace

SCENARIO("TransferFunction builds a colormap from a range", "[nodes-tf]")
{
  Graph g;
  auto rng = g.addNode(std::make_unique<EmitRange>());
  auto tf = addBuiltin(g, "TransferFunction");
  g.node(tf)->impl->parameters().set(Token("preset"), std::string("grayscale"));
  g.node(tf)->impl->parameters().set(Token("samples"), 8);
  g.connect(rng, Token("out"), tf, Token("in"));
  Evaluator e(g);

  REQUIRE(e.pull(tf));
  auto d = std::static_pointer_cast<TransferFunctionData>(
      e.output(tf, Token("out"), hostResidency())->payload);
  THEN("colormap is float4 x8 with grayscale ramp and valueRange {0,1}")
  {
    REQUIRE(d->colormap.elementType() == ANARI_FLOAT32_VEC4);
    REQUIRE(d->colormap.size() == 8);
    REQUIRE(d->valueRange.x == 0.f);
    REQUIRE(d->valueRange.y == 1.f);
    REQUIRE(d->colormap.get<float4>(0).x == Approx(0.f));
    REQUIRE(d->colormap.get<float4>(7).x == Approx(1.f));
  }
}
```
Register `add_test(NAME tsd::nodes::TransferFunction COMMAND ${PROJECT_NAME} "[nodes-tf]")`.

- [ ] **Step 2: Build, confirm FAIL** (stub).

- [ ] **Step 3: Replace `tsd/src/tsd/graph_nodes/TransferFunction.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/core/Logging.hpp"
#include <cmath>
#include <memory>
#include <string>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

struct TransferFunction : Node
{
  ParameterList params;
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
  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency());
    auto range = std::static_pointer_cast<float2>(in.payload);
    if (!range) {
      ctx.fail("TransferFunction: missing range input");
      return;
    }
    int samples = params.getOr<int>(Token("samples"), 256);
    if (samples < 2) {
      ctx.fail("TransferFunction: samples must be >= 2");
      return;
    }
    // NB: `Any` cannot hold a Token (no ANARITypeFor<Token>); store presets as
    // std::string (ANARI_STRING).
    const std::string preset =
        params.getOr<std::string>(Token("preset"), std::string("coolToWarm"));
    bool coolToWarm = (preset == "coolToWarm");
    const bool grayscale = (preset == "grayscale");
    if (!coolToWarm && !grayscale) {
      tsd::core::logWarning(
          "[TransferFunction] unknown preset '%s', using grayscale",
          preset.c_str());
      coolToWarm = false; // fall back to grayscale
    }

    auto d = std::make_shared<TransferFunctionData>();
    d->valueRange = *range;
    d->colormap = tsd::core::AnyArray(ANARI_FLOAT32_VEC4, size_t(samples));
    for (int i = 0; i < samples; ++i) {
      const float t = float(i) / float(samples - 1);
      float4 c;
      if (coolToWarm) // blue -> white -> red, alpha ramps with value
        c = float4(t, 1.f - std::abs(0.5f - t) * 2.f, 1.f - t, t);
      else // grayscale
        c = float4(t, t, t, t);
      d->colormap.get<float4>(size_t(i)) = c;
    }

    Value out;
    out.type = PortType{portTF()};
    out.residency = hostResidency();
    out.payload = d;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerTransferFunction(NodeRegistry &reg)
{
  reg.registerType(
      Token("TransferFunction"), [] { return std::make_unique<TransferFunction>(); });
}

} // namespace tsd::graph_nodes
```
(`<cmath>` for `std::abs`.)

- [ ] **Step 4: Build + run** `ctest ... -R 'tsd::nodes::TransferFunction' --output-on-failure` → PASS.

- [ ] **Step 5: Commit.**
```bash
clang-format -i tsd/src/tsd/graph_nodes/TransferFunction.cpp tsd/tests/test_nodes_TransferFunction.cpp
jj commit tsd/src/tsd/graph_nodes/TransferFunction.cpp tsd/tests/test_nodes_TransferFunction.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): TransferFunction processor node"
```

---

## Task 4: `DisplayVolume` (sink → Renderable)

**Files:**
- Replace stub: `tsd/src/tsd/graph_nodes/DisplayVolume.cpp`
- Test: `tsd/tests/test_nodes_DisplayVolume.cpp`

**Interfaces:**
- Consumes: `field` + `transferFunction`.
- Produces: token `"DisplayVolume"`, inputs `field:field`, `tf:transferFunction`, output `out:renderable`; payload `std::shared_ptr<tsd::graph::Renderable>` with `kind=Volume`, `primSubtype="structuredRegular"`, `prim` carrying `dims`(float3), `origin`, `spacing`, `data`(array); `appearance` carrying `color`(float4 colormap array) and `valueRange`(float2).

- [ ] **Step 1: Write the failing test** — `KnownField` (reuse pattern) + an in-test `EmitTF` emitting a `TransferFunctionData` (4-entry colormap, valueRange {0,1}) → `DisplayVolume(field, tf)` → pull → assert the `Renderable` is `kind==Volume`, `primSubtype=="structuredRegular"`, `prim` has a `dims` scalar == field dims and a `data` array of the right size, `appearance` has a `color` array (size 4) and a `valueRange`.
(Write `KnownField` and `EmitTF` in-test; assert on `r->kind`, `r->primSubtype`, and that the expected param tokens are present in `r->prim.scalars/arrays` and `r->appearance`.)
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using tsd::graph_nodes::TransferFunctionData;
using uint3 = tsd::core::math::uint3;
using float2 = tsd::core::math::float2;
using float4 = tsd::core::math::float4;

namespace {
bool hasArray(const RenderableParams &p, Token n)
{
  for (auto &a : p.arrays) if (a.first == n) return true;
  return false;
}
bool hasScalar(const RenderableParams &p, Token n)
{
  for (auto &s : p.scalars) if (s.first == n) return true;
  return false;
}
struct KnownField : Node {
  ParameterList params;
  NodeTypeInfo typeInfo() const override {
    NodeTypeInfo i; i.name = Token("KnownField");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override {
    auto f = std::make_shared<Field>();
    f->dims = uint3(2u, 2u, 2u);
    f->data = tsd::core::AnyArray(ANARI_FLOAT32, 8);
    for (int k = 0; k < 8; ++k) f->data.get<float>(k) = float(k) / 7.f;
    Value v; v.type = PortType{Token("field")}; v.residency = hostResidency(); v.payload = f;
    ctx.setOutput(Token("out"), v);
  }
};
struct EmitTF : Node {
  ParameterList params;
  NodeTypeInfo typeInfo() const override {
    NodeTypeInfo i; i.name = Token("EmitTF");
    i.outputs.push_back({Token("out"), PortType{Token("transferFunction")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override {
    auto d = std::make_shared<TransferFunctionData>();
    d->valueRange = float2(0.f, 1.f);
    d->colormap = tsd::core::AnyArray(ANARI_FLOAT32_VEC4, 4);
    for (int k = 0; k < 4; ++k) d->colormap.get<float4>(k) = float4(float(k)/3.f);
    Value v; v.type = PortType{Token("transferFunction")}; v.residency = hostResidency(); v.payload = d;
    ctx.setOutput(Token("out"), v);
  }
};
NodeId addBuiltin(Graph &g, const char *t) {
  static NodeRegistry reg = [] { NodeRegistry r; tsd::graph_nodes::registerBuiltinNodes(r); return r; }();
  return g.addNode(reg.create(Token(t)));
}
} // namespace

SCENARIO("DisplayVolume packs field+TF into a Renderable", "[nodes-dvol]")
{
  Graph g;
  auto fld = g.addNode(std::make_unique<KnownField>());
  auto tf = g.addNode(std::make_unique<EmitTF>());
  auto dv = addBuiltin(g, "DisplayVolume");
  g.connect(fld, Token("out"), dv, Token("field"));
  g.connect(tf, Token("out"), dv, Token("tf"));
  Evaluator e(g);

  REQUIRE(e.pull(dv));
  auto r = std::static_pointer_cast<Renderable>(
      e.output(dv, Token("out"), hostResidency())->payload);
  THEN("it is a structuredRegular volume renderable")
  {
    REQUIRE(r->kind == Renderable::Kind::Volume);
    REQUIRE(r->primSubtype == Token("structuredRegular"));
    REQUIRE(hasScalar(r->prim, Token("dims")));
    REQUIRE(hasArray(r->prim, Token("data")));
    REQUIRE(hasScalar(r->prim, Token("origin")));
    REQUIRE(hasArray(r->appearance, Token("color")));
  }
}
```
Register `add_test(NAME tsd::nodes::DisplayVolume COMMAND ${PROJECT_NAME} "[nodes-dvol]")`.

- [ ] **Step 2: Build, confirm FAIL** (stub).

- [ ] **Step 3: Replace `tsd/src/tsd/graph_nodes/DisplayVolume.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph/Evaluator.hpp"
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float3 = tsd::core::math::float3;

struct DisplayVolume : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("DisplayVolume");
    i.category = Token("sink");
    i.inputs.push_back({Token("field"), PortType{portField()}, true, {}});
    i.inputs.push_back({Token("tf"), PortType{portTF()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portRenderable()}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto field = std::static_pointer_cast<Field>(
        ctx.input(Token("field"), hostResidency()).payload);
    auto tf = std::static_pointer_cast<TransferFunctionData>(
        ctx.input(Token("tf"), hostResidency()).payload);
    if (!field || !tf) {
      ctx.fail("DisplayVolume: missing field or transferFunction input");
      return;
    }
    if (field->data.size()
        != size_t(field->dims.x) * field->dims.y * field->dims.z) {
      ctx.fail("DisplayVolume: field data size does not match dims");
      return;
    }
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Volume;
    r->primSubtype = Token("structuredRegular");
    r->prim.scalars.push_back({Token("dims"),
        tsd::core::Any(float3(
            float(field->dims.x), float(field->dims.y), float(field->dims.z)))});
    r->prim.scalars.push_back({Token("origin"), tsd::core::Any(field->origin)});
    r->prim.scalars.push_back({Token("spacing"), tsd::core::Any(field->spacing)});
    r->prim.arrays.push_back({Token("data"), field->data});
    r->appearance.arrays.push_back({Token("color"), tf->colormap});
    r->appearance.scalars.push_back(
        {Token("valueRange"), tsd::core::Any(tf->valueRange)});

    Value out;
    out.type = PortType{portRenderable()};
    out.residency = hostResidency();
    out.payload = r;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerDisplayVolume(NodeRegistry &reg)
{
  reg.registerType(Token("DisplayVolume"), [] { return std::make_unique<DisplayVolume>(); });
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 4: Build + run** `ctest ... -R 'tsd::nodes::DisplayVolume' --output-on-failure` → PASS.

- [ ] **Step 5: Commit.**
```bash
clang-format -i tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/tests/test_nodes_DisplayVolume.cpp
jj commit tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/tests/test_nodes_DisplayVolume.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): DisplayVolume sink node"
```

---

## Task 5: `BoundingBox` (field→surface) + `DisplaySurface` (sink)

**Files:**
- Replace stubs: `tsd/src/tsd/graph_nodes/BoundingBox.cpp`, `tsd/src/tsd/graph_nodes/DisplaySurface.cpp`
- Test: `tsd/tests/test_nodes_Surface.cpp`

**Interfaces:**
- `BoundingBox`: token `"BoundingBox"`, input `in:field`, output `out:surface`; param `color:float3`. Output `std::shared_ptr<SurfaceData>` (geomSubtype "triangle", `prim` with a `vertex.position` float3 array of 36 verts = 12 triangles of the field bbox, `appearance` with `color`).
- `DisplaySurface`: token `"DisplaySurface"`, input `in:surface`, output `out:renderable`; payload `Renderable{kind=Surface}` with `primSubtype=surface.geomSubtype`, `prim=surface.prim`, `appearance=surface.appearance`.

- [ ] **Step 1: Write the failing test** `tsd/tests/test_nodes_Surface.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using tsd::graph_nodes::SurfaceData;
using uint3 = tsd::core::math::uint3;
using float3 = tsd::core::math::float3;

namespace {

bool hasArray(const RenderableParams &p, Token n)
{
  for (auto &a : p.arrays)
    if (a.first == n)
      return true;
  return false;
}

struct KnownField : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("KnownField");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::make_shared<Field>();
    f->dims = uint3(2u, 2u, 2u);
    f->origin = float3(-1.f, -1.f, -1.f);
    f->spacing = float3(1.f, 1.f, 1.f);
    f->data = tsd::core::AnyArray(ANARI_FLOAT32, 8);
    for (int k = 0; k < 8; ++k)
      f->data.get<float>(k) = float(k) / 7.f;
    Value v;
    v.type = PortType{Token("field")};
    v.residency = hostResidency();
    v.payload = f;
    ctx.setOutput(Token("out"), v);
  }
};

NodeId addBuiltin(Graph &g, const char *t)
{
  static NodeRegistry reg = [] {
    NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(t)));
}

} // namespace

SCENARIO("BoundingBox -> DisplaySurface produces a triangle surface renderable",
    "[nodes-surface]")
{
  Graph g;
  auto fld = g.addNode(std::make_unique<KnownField>());
  auto bb = addBuiltin(g, "BoundingBox");
  g.node(bb)->impl->parameters().set(Token("color"), float3(0.2f, 0.8f, 0.2f));
  auto ds = addBuiltin(g, "DisplaySurface");
  g.connect(fld, Token("out"), bb, Token("in"));
  g.connect(bb, Token("out"), ds, Token("in"));
  Evaluator e(g);

  REQUIRE(e.pull(ds));

  WHEN("inspecting the BoundingBox surface output")
  {
    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(bb, Token("out"), hostResidency())->payload);
    THEN("it is a triangle box with 36 vertex positions")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("triangle"));
      REQUIRE(hasArray(s->prim, Token("vertex.position")));
      size_t n = 0;
      for (auto &a : s->prim.arrays)
        if (a.first == Token("vertex.position"))
          n = a.second.size();
      REQUIRE(n == 36);
    }
  }

  WHEN("inspecting the DisplaySurface renderable output")
  {
    auto r = std::static_pointer_cast<Renderable>(
        e.output(ds, Token("out"), hostResidency())->payload);
    THEN("it is a Surface renderable carrying the geometry")
    {
      REQUIRE(r->kind == Renderable::Kind::Surface);
      REQUIRE(r->primSubtype == Token("triangle"));
      REQUIRE(hasArray(r->prim, Token("vertex.position")));
    }
  }
}
```
Register `add_test(NAME tsd::nodes::Surface COMMAND ${PROJECT_NAME} "[nodes-surface]")`.

- [ ] **Step 2: Build, confirm FAIL** (stubs).

- [ ] **Step 3: Replace `tsd/src/tsd/graph_nodes/BoundingBox.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph/Evaluator.hpp"
#include <array>
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float3 = tsd::core::math::float3;

struct BoundingBox : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("BoundingBox");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portField()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portSurface()}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::static_pointer_cast<Field>(
        ctx.input(Token("in"), hostResidency()).payload);
    if (!f) {
      ctx.fail("BoundingBox: missing field input");
      return;
    }
    const float3 lo = f->origin;
    const float3 hi = float3(lo.x + f->spacing.x * f->dims.x,
        lo.y + f->spacing.y * f->dims.y,
        lo.z + f->spacing.z * f->dims.z);
    // 8 corners
    const float3 c[8] = {
        {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z},
        {lo.x, hi.y, lo.z}, {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z},
        {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}};
    // 12 triangles (2 per face), 36 vertices
    const int tri[36] = {0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0};
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 36);
    for (int i = 0; i < 36; ++i)
      pos.get<float3>(size_t(i)) = c[tri[i]];

    auto s = std::make_shared<SurfaceData>();
    s->geomSubtype = Token("triangle");
    s->prim.arrays.push_back({Token("vertex.position"), pos});
    s->appearance.scalars.push_back({Token("color"),
        tsd::core::Any(params.getOr<float3>(Token("color"), float3(0.8f)))});

    Value out;
    out.type = PortType{portSurface()};
    out.residency = hostResidency();
    out.payload = s;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerBoundingBox(NodeRegistry &reg)
{
  reg.registerType(Token("BoundingBox"), [] { return std::make_unique<BoundingBox>(); });
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 4: Replace `tsd/src/tsd/graph_nodes/DisplaySurface.cpp`:**
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include "tsd/graph/Evaluator.hpp"
#include <memory>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;

struct DisplaySurface : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("DisplaySurface");
    i.category = Token("sink");
    i.inputs.push_back({Token("in"), PortType{portSurface()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portRenderable()}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto s = std::static_pointer_cast<SurfaceData>(
        ctx.input(Token("in"), hostResidency()).payload);
    if (!s) {
      ctx.fail("DisplaySurface: missing surface input");
      return;
    }
    auto r = std::make_shared<Renderable>();
    r->kind = Renderable::Kind::Surface;
    r->primSubtype = s->geomSubtype;
    r->prim = s->prim;
    r->appearance = s->appearance;

    Value out;
    out.type = PortType{portRenderable()};
    out.residency = hostResidency();
    out.payload = r;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerDisplaySurface(NodeRegistry &reg)
{
  reg.registerType(Token("DisplaySurface"), [] { return std::make_unique<DisplaySurface>(); });
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 5: Build + run** `ctest ... -R 'tsd::nodes::Surface' --output-on-failure` → PASS.

- [ ] **Step 6: Commit.**
```bash
clang-format -i tsd/src/tsd/graph_nodes/BoundingBox.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/tests/test_nodes_Surface.cpp
jj commit tsd/src/tsd/graph_nodes/BoundingBox.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/tests/test_nodes_Surface.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): BoundingBox + DisplaySurface nodes"
```

---

## Task 6: Registry-complete + end-to-end wiring (no ANARI)

**Files:**
- Test: `tsd/tests/test_nodes_Wiring.cpp`

**Interfaces:**
- Consumes: `registerBuiltinNodes`, all six node tokens.

- [ ] **Step 1: Write the test.** (a) After `registerBuiltinNodes(reg)`, all six tokens `isRegistered`. (b) Build the demo volume graph via the registry — GenerateNoiseVolume → ScalarRange → TransferFunction → DisplayVolume, with the source also connected to DisplayVolume's `field` — pull DisplayVolume and assert a `Renderable{Volume}` whose `prim` `dims` matches the source dims. (c) Fan-out + multi-input resolve (the pull succeeds). (d) Bump source `seed`, `markDirty(src)`, pull again → still succeeds and the volume rebuilds.
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using uint3 = tsd::core::math::uint3;

namespace {
NodeRegistry &builtins()
{
  static NodeRegistry reg = [] { NodeRegistry r; tsd::graph_nodes::registerBuiltinNodes(r); return r; }();
  return reg;
}
NodeId add(Graph &g, const char *t) { return g.addNode(builtins().create(Token(t))); }
} // namespace

SCENARIO("all six builtins register", "[nodes-wiring]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  for (auto *t : {"GenerateNoiseVolume", "ScalarRange", "TransferFunction",
           "DisplayVolume", "BoundingBox", "DisplaySurface"})
    REQUIRE(reg.isRegistered(Token(t)));
}

SCENARIO("the demo volume graph resolves end-to-end", "[nodes-wiring]")
{
  Graph g;
  auto src = add(g, "GenerateNoiseVolume");
  g.node(src)->impl->parameters().set(Token("dims"), uint3(16u, 16u, 16u));
  auto sr = add(g, "ScalarRange");
  auto tf = add(g, "TransferFunction");
  auto dv = add(g, "DisplayVolume");
  g.connect(src, Token("out"), sr, Token("in"));
  g.connect(sr, Token("out"), tf, Token("in"));
  g.connect(src, Token("out"), dv, Token("field")); // fan-out
  g.connect(tf, Token("out"), dv, Token("tf"));      // multi-input
  Evaluator e(g);

  REQUIRE(e.pull(dv));
  auto r = std::static_pointer_cast<Renderable>(
      e.output(dv, Token("out"), hostResidency())->payload);
  REQUIRE(r->kind == Renderable::Kind::Volume);

  WHEN("the source seed changes")
  {
    g.node(src)->impl->parameters().set(Token("seed"), 3);
    g.markDirty(src);
    THEN("the graph re-pulls successfully")
    {
      REQUIRE(e.pull(dv));
    }
  }
}
```
Register `add_test(NAME tsd::nodes::Wiring COMMAND ${PROJECT_NAME} "[nodes-wiring]")`.

- [ ] **Step 2: Build + run** `ctest ... -R 'tsd::nodes::Wiring' --output-on-failure` → PASS (all builtins registered; demo graph resolves; re-pull on seed change).

- [ ] **Step 3: Commit.**
```bash
clang-format -i tsd/tests/test_nodes_Wiring.cpp
jj commit tsd/tests/test_nodes_Wiring.cpp tsd/tests/CMakeLists.txt -m "test(graph_nodes): registry-complete + end-to-end wiring"
```

---

## Task 7: VisRTX render smoke via the Phase 3 bridge

**Files:**
- Test: `tsd/tests/test_nodes_Render.cpp`

**Interfaces:**
- Consumes: `tsd::rendering::GraphRenderBridge` (Phase 3), the six builtins.

- [ ] **Step 1: Write the test** `tsd/tests/test_nodes_Render.cpp` (one merged graph holds both display chains, sharing the source; the bridge binds that one graph; `renderCounts` is lifted verbatim from `tsd/tests/test_bridge_RenderVolume.cpp`):
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/rendering/bridge/GraphRenderBridge.hpp"
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
// std
#include <memory>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::rendering::GraphRenderBridge;
using uint2 = anari::math::uint2;
using float3 = anari::math::float3;

namespace {

anari::Device makeVisRTX()
{
  auto lib = anari::loadLibrary("visrtx", nullptr, nullptr);
  return lib ? anari::newDevice(lib, "default") : nullptr;
}

struct Counts
{
  size_t color{0};
  size_t objectId{0};
};

Counts renderCounts(anari::Device d, anari::World world)
{
  auto cam = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(d, cam, "aspect", 1.f);
  anari::setParameter(d, cam, "position", float3(0.f, 0.f, 3.f));
  anari::setParameter(d, cam, "direction", float3(0.f, 0.f, -1.f));
  anari::setParameter(d, cam, "up", float3(0.f, 1.f, 0.f));
  anari::commitParameters(d, cam);
  auto rnd = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, rnd, "ambientRadiance", 1.f);
  anari::commitParameters(d, rnd);
  auto frame = anari::newObject<anari::Frame>(d);
  uint2 sz{64, 64};
  anari::setParameter(d, frame, "size", sz);
  anari::setParameter(d, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(d, frame, "world", world);
  anari::setParameter(d, frame, "camera", cam);
  anari::setParameter(d, frame, "renderer", rnd);
  anari::commitParameters(d, frame);
  anari::render(d, frame);
  anari::wait(d, frame);
  Counts c;
  auto col = anari::map<uint32_t>(d, frame, "channel.color");
  if (col.data)
    for (uint32_t i = 0; i < col.width * col.height; ++i)
      if ((col.data[i] & 0x00ffffffu) != 0u)
        ++c.color;
  anari::unmap(d, frame, "channel.color");
  auto oid = anari::map<uint32_t>(d, frame, "channel.objectId");
  if (oid.data)
    for (uint32_t i = 0; i < oid.width * oid.height; ++i)
      if (oid.data[i] != ~0u)
        ++c.objectId;
  anari::unmap(d, frame, "channel.objectId");
  anari::release(d, frame);
  anari::release(d, rnd);
  anari::release(d, cam);
  return c;
}

NodeId add(Graph &g, const char *t)
{
  static NodeRegistry reg = [] {
    NodeRegistry r;
    tsd::graph_nodes::registerBuiltinNodes(r);
    return r;
  }();
  return g.addNode(reg.create(Token(t)));
}

} // namespace

SCENARIO("the 4a catalog drives rendered pixels via the bridge", "[nodes-render]")
{
  anari::Device dev = makeVisRTX();
  REQUIRE(dev != nullptr);

  Graph g;
  auto src = add(g, "GenerateNoiseVolume");
  g.node(src)->impl->parameters().set(
      Token("dims"), tsd::core::math::uint3(24u, 24u, 24u));
  auto sr = add(g, "ScalarRange");
  auto tf = add(g, "TransferFunction"); // coolToWarm: alpha ramps with value
  auto dv = add(g, "DisplayVolume");
  auto bb = add(g, "BoundingBox");
  auto ds = add(g, "DisplaySurface");
  g.connect(src, Token("out"), sr, Token("in"));
  g.connect(sr, Token("out"), tf, Token("in"));
  g.connect(src, Token("out"), dv, Token("field")); // fan-out
  g.connect(tf, Token("out"), dv, Token("tf"));      // multi-input
  g.connect(src, Token("out"), bb, Token("in"));
  g.connect(bb, Token("out"), ds, Token("in"));

  Evaluator e(g);
  GraphRenderBridge bridge(g, e, Token("visrtx"), dev, /*numViewports=*/2);
  bridge.setDisplay(dv, /*mask=*/0b01, /*enabled=*/true); // volume -> viewport 0
  bridge.setDisplay(ds, /*mask=*/0b10, /*enabled=*/true); // surface -> viewport 1
  bridge.update();

  WHEN("rendering both viewports")
  {
    Counts c0 = renderCounts(dev, bridge.world(0));
    Counts c1 = renderCounts(dev, bridge.world(1));
    THEN("vp0 shows the volume (color) and vp1 the box surface (objectId)")
    {
      REQUIRE(c0.color > 0);    // the volume rendered in viewport 0
      REQUIRE(c1.objectId > 0); // the box surface rendered in viewport 1
    }
  }

  anari::release(dev, dev);
}
```
Register `add_test(NAME tsd::nodes::Render COMMAND ${PROJECT_NAME} "[nodes-render]")` + `set_tests_properties(tsd::nodes::Render PROPERTIES TIMEOUT 300)`.

> **Volume visibility is deterministic here:** the default `coolToWarm` TF sets
> alpha = value, and `GenerateNoiseVolume`'s radial blob always has high-value
> voxels at the centre → non-zero alpha → non-background color. No runtime tuning.

- [ ] **Step 2: Build + run** `ctest ... -R 'tsd::nodes::Render' --output-on-failure` → PASS.
  - If the volume viewport is empty: confirm `DisplayVolume`'s `Renderable` matches what the bridge's `buildVolume` expects (3D `data` via the `dims` scalar; float4 `color` array; `value`/field). The bridge already handles `structuredRegular` + `dims` (Phase 3). Confirm the colormap alpha is non-zero so the volume is visible (TransferFunction grayscale alpha ramps 0→1; if invisible, the test may set the source/TF so the field's mid-range maps to non-zero alpha — adjust the TF preset or valueRange and report).
  - If the surface viewport is empty: the box spans the field bounds ([-1,1]); camera at z=3 frames it. Reuse the surface render lesson (objectId `~0u` sentinel; `ambientRadiance`). Report any adjustment.

- [ ] **Step 3: Commit.**
```bash
clang-format -i tsd/tests/test_nodes_Render.cpp
jj commit tsd/tests/test_nodes_Render.cpp tsd/tests/CMakeLists.txt -m "test(graph_nodes): visrtx render smoke — demo volume + surface graphs via bridge"
```

---

## Task 8: Full-suite gate

- [ ] **Step 1: Build + run the whole suite.**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo --output-on-failure
```
Expected: all green — the prior 45 + the new `tsd::nodes::*` (Scaffold, Noise, Range, TransferFunction, DisplayVolume, Surface, Wiring, Render). Report the summary line and confirm `.envrc` is still uncommitted (`jj status`).

- [ ] **Step 2:** Verification gate (no new commit unless a fix was needed).

---

## Phase 4a completion checklist

- [ ] New `tsd_graph_nodes` lib links `tsd_graph` only (no tsd_io/scene/UI)
- [ ] `EvalContext::fail` added; nodes fail loud on bad input/params
- [ ] Six nodes implemented + per-node tests (determinism, ranges, descriptors)
- [ ] `registerBuiltinNodes()` explicit registration; all six resolvable by token
- [ ] Demo volume graph (fan-out + multi-input) + surface graph resolve headless
- [ ] VisRTX render smoke: volume renders (color), box surface renders (objectId), masked to separate viewports
- [ ] full suite green

## Out of scope (per spec)

File importers + `tsd_io`/`tsd_scene` deps (later slice); CUDA residency (4b); any
UI — app shell, node editor, inspector, TF editor (4c/4d); undo/redo, persistence
(4e/Phase 5); Resample/Crop and a richer catalog.

## Self-review notes

- Every node is core-only `AnyArray` manipulation; none wrap `tsd_io`/`tsd_scene`.
- Descriptors (`Field`/`TransferFunctionData`/`SurfaceData`) + the Phase 3
  `Renderable` are the catalog vocabulary; the bridge consumes only `renderable`.
- `EvalContext::fail` is the one engine addition (tsd_graph), with its own test.
- Node classes are registry-created (anonymous-namespace) — tests use the same
  `create(Token)` path the future UI menu will use.
- Render test reuses Phase 3's VisRTX helper pattern and objectId/color assertions;
  carries `TIMEOUT 300` for OptiX warmup.
- Risk areas flagged inline: volume visibility (colormap alpha / valueRange — made
  deterministic in Task 7), and the bridge's `structuredRegular` `dims`/`data`
  expectations (already satisfied by `DisplayVolume`'s `Renderable`, matching Phase
  3's `buildVolume`).
- `EvalContext::input()` returns a `Value` **by value**; nodes immediately
  `static_pointer_cast` its `.payload` (copies the `shared_ptr` before the temporary
  dies) — do not bind it to a reference.
- `preset` is stored as `std::string`, not `Token` (`Any` has no `ANARITypeFor<Token>`).
- Registration is **explicit-only** via `registerBuiltinNodes()`; the spec also
  mentioned the `TSD_GRAPH_REGISTER_NODE` static-init macro, intentionally dropped
  here because static-lib objects with only static-init side effects can be stripped
  by the linker (the Phase 1 whole-archive caveat). Explicit registration is robust.
