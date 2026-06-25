# tsdFlow IsosurfaceExtract Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `IsosurfaceExtract` compute node that contours a scalar `Field` into a triangle `SurfaceData` via Viskores, flowing into the existing `DisplaySurface`.

**Architecture:** One new node `.cpp` in `tsd_graph_nodes`, mirroring `BoundingBox` (field→surface). It marshals the host float field into a `viskores::cont::DataSet`, runs `viskores::filter::contour::Contour`, and packs the output points/triangles/normals back into a `SurfaceData`. Viskores is an **optional, auto-detected** dependency: when found, the node compiles, registers, gains a demo-graph branch, and its unit test runs; when absent, all of that is `#ifdef`-omitted and the rest of the build is unchanged.

**Tech Stack:** C++17, Viskores 1.1.0 (`viskores::filter_contour`, `viskores::cont`), Catch2 tests, CMake (`_out/_cmake`).

## Global Constraints

- **Version control is jj, not git.** Commit with **explicit file paths** (`jj commit <path>... -m "..."`); never a bare `jj commit`, never raw `git` (sandboxed).
- **No `Co-Authored-By` lines.**
- **Never run clang-format on `CMakeLists.txt`** (or any CMake file).
- **Viskores is optional and auto-detected.** Gate everything on `TSD_GRAPH_NODES_HAVE_VISKORES` (a `PUBLIC` compile definition set only when `find_package(Viskores 1.1.0)` succeeds). Builds without Viskores must be unaffected.
- **Use the default Serial host device** — do not force CUDA or call device-selection APIs.
- **Node name is exactly `IsosurfaceExtract`**; params are `isovalue` (float, 0.5) and `computeNormals` (bool, true).
- **Configure:** `cmake _out/_cmake` (re-runs with cached args; needed after CMakeLists/source-list changes). **Build:** `cmake --build _out/_cmake --parallel`. **Test:** `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure` (currently 67 tests; this plan adds one — the suite must stay green; VisRTX render tests already carry `TIMEOUT 300`).
- This build has `Viskores_DIR` cached, so the node and its test are active here.

---

### Task 1: IsosurfaceExtract node + unit test

**Files:**
- Create: `tsd/src/tsd/graph_nodes/IsosurfaceExtract.cpp`
- Modify: `tsd/src/tsd/graph_nodes/CMakeLists.txt` (find Viskores; conditional source/link/define)
- Modify: `tsd/src/tsd/graph_nodes/BuiltinNodes.hpp` (guarded declaration)
- Modify: `tsd/src/tsd/graph_nodes/BuiltinNodes.cpp` (guarded registration call)
- Create: `tsd/tests/test_nodes_Isosurface.cpp`
- Modify: `tsd/tests/CMakeLists.txt` (add the test source)

**Interfaces:**
- Consumes (existing): `tsd::graph::Node` base (`typeInfo()`, `parameters()`, `evaluate(EvalContext&)`); `EvalContext::input(Token, residency)` returns a `Value` whose `.payload` is `std::shared_ptr<void>`; `EvalContext::fail(const char*)`; `EvalContext::setOutput(Token, Value)`; `hostResidency()`; `tsd::graph_nodes::Field {Token subtype; uint3 dims; float3 origin; float3 spacing; AnyArray data;}`; `SurfaceData {Token geomSubtype; RenderableParams prim; RenderableParams appearance;}` where `RenderableParams` has `std::vector<std::pair<Token,AnyArray>> arrays` and `std::vector<std::pair<Token,Any>> scalars`; `portField()`, `portSurface()`; `ParameterList::set/getOr`; `NodeRegistry::registerType(Token, factory)`. `AnyArray(ANARITypeEnum, size_t)` with `.get<T>(i)`, `.size()`, `.type()`.
- Produces: `void registerIsosurfaceExtract(tsd::graph::NodeRegistry&)`; a node registered under `Token("IsosurfaceExtract")` with input `in`=field, output `out`=surface, emitting `SurfaceData{geomSubtype="triangle"}` carrying `vertex.position` (ANARI_FLOAT32_VEC3), `primitive.index` (ANARI_UINT32_VEC3), and `vertex.normal` (ANARI_FLOAT32_VEC3, when `computeNormals`).

**Context:** `tsd_graph_nodes` is the node catalog (links only `tsd_graph` today). `BoundingBox.cpp` is the closest template (field→surface, builds `AnyArray`s by hand, registers via a free function called from `BuiltinNodes.cpp`). The Evaluator runs `evaluate` and re-runs it when the `ParameterList` hash changes — so `isovalue`/`computeNormals` belong in `params` (editing re-extracts, which is correct here).

- [ ] **Step 1: Create the node skeleton (compiles, registers, fails at eval)**

Create `tsd/src/tsd/graph_nodes/IsosurfaceExtract.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// viskores
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Field.h>
#include <viskores/cont/Initialize.h>
#include <viskores/filter/contour/Contour.h>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float3 = tsd::core::math::float3;
using uint3 = tsd::core::math::uint3;

void ensureViskoresInit()
{
  static std::once_flag once;
  std::call_once(once, [] { viskores::cont::Initialize(); });
}

struct IsosurfaceExtract : Node
{
  ParameterList params;
  IsosurfaceExtract()
  {
    params.set(Token("isovalue"), 0.5f);
    params.set(Token("computeNormals"), true);
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("IsosurfaceExtract");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portField()}, true, {}});
    i.outputs.push_back({Token("out"), PortType{portSurface()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    ctx.fail("IsosurfaceExtract: not yet implemented");
  }
};

} // namespace

void registerIsosurfaceExtract(NodeRegistry &reg)
{
  reg.registerType(Token("IsosurfaceExtract"),
      [] { return std::make_unique<IsosurfaceExtract>(); });
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 2: Gate the node in CMake on Viskores**

In `tsd/src/tsd/graph_nodes/CMakeLists.txt`, immediately after the existing `project_link_libraries(PUBLIC tsd_graph)` line, add:

```cmake
# IsosurfaceExtract requires Viskores. Auto-detected and optional: when Viskores
# is absent the node source, its link deps, and its TSD_GRAPH_NODES_HAVE_VISKORES
# guard are all omitted, leaving the rest of the library unchanged.
find_package(Viskores 1.1.0 QUIET)
if(Viskores_FOUND)
  target_sources(tsd_graph_nodes PRIVATE IsosurfaceExtract.cpp)
  target_link_libraries(tsd_graph_nodes PUBLIC viskores::filter_contour viskores::cont)
  target_compile_definitions(tsd_graph_nodes PUBLIC TSD_GRAPH_NODES_HAVE_VISKORES)
endif()
```

(Do not reformat the file.)

- [ ] **Step 3: Declare and register the node, guarded**

In `tsd/src/tsd/graph_nodes/BuiltinNodes.hpp`, after the `void registerDisplaySurface(tsd::graph::NodeRegistry &reg);` line, add:

```cpp
#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
void registerIsosurfaceExtract(tsd::graph::NodeRegistry &reg);
#endif
```

In `tsd/src/tsd/graph_nodes/BuiltinNodes.cpp`, inside `registerBuiltinNodes(NodeRegistry &reg)`, after the `registerDisplaySurface(reg);` line, add:

```cpp
#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
  registerIsosurfaceExtract(reg);
#endif
```

- [ ] **Step 4: Configure and build the skeleton**

Run: `cmake _out/_cmake && cmake --build _out/_cmake --parallel`
Expected: configures (finds Viskores 1.1.0) and builds clean. The catalog now contains `IsosurfaceExtract`, whose `evaluate` fails by design.

- [ ] **Step 5: Write the unit test**

Create `tsd/tests/test_nodes_Isosurface.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"

#ifdef TSD_GRAPH_NODES_HAVE_VISKORES

#include <cmath>
#include <memory>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::Field;
using tsd::graph_nodes::SurfaceData;
using uint3 = tsd::core::math::uint3;
using float3 = tsd::core::math::float3;

namespace {

// Values are a sphere distance field v = R - |p|, sampled over [-1,1]^3.
// An isovalue in (0,R) yields a closed surface; outside the range yields none.
struct SphereField : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("SphereField");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    const uint32_t N = 16;
    auto f = std::make_shared<Field>();
    f->dims = uint3(N, N, N);
    f->origin = float3(-1.f, -1.f, -1.f);
    f->spacing = float3(2.f / (N - 1), 2.f / (N - 1), 2.f / (N - 1));
    f->data = tsd::core::AnyArray(ANARI_FLOAT32, size_t(N) * N * N);
    size_t idx = 0;
    for (uint32_t z = 0; z < N; ++z)
      for (uint32_t y = 0; y < N; ++y)
        for (uint32_t x = 0; x < N; ++x, ++idx) {
          const float px = f->origin.x + f->spacing.x * x;
          const float py = f->origin.y + f->spacing.y * y;
          const float pz = f->origin.z + f->spacing.z * z;
          const float r = std::sqrt(px * px + py * py + pz * pz);
          f->data.get<float>(idx) = 0.6f - r; // R = 0.6
        }
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

const tsd::core::AnyArray *findArray(const RenderableParams &p, Token n)
{
  for (const auto &a : p.arrays)
    if (a.first == n)
      return &a.second;
  return nullptr;
}

} // namespace

SCENARIO("IsosurfaceExtract contours a scalar field into a triangle mesh",
    "[nodes-isosurface]")
{
  Graph g;
  auto fld = g.addNode(std::make_unique<SphereField>());
  auto iso = addBuiltin(g, "IsosurfaceExtract");
  REQUIRE(g.node(iso) != nullptr); // registered when Viskores is present
  g.connect(fld, Token("out"), iso, Token("in"));

  GIVEN("an isovalue inside the field range")
  {
    g.node(iso)->impl->parameters().set(Token("isovalue"), 0.2f);
    g.node(iso)->impl->parameters().set(Token("computeNormals"), true);
    Evaluator e(g);
    REQUIRE(e.pull(iso));

    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(iso, Token("out"), hostResidency())->payload);
    THEN("it emits a non-empty triangle surface, valid indices, and normals")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("triangle"));
      const auto *pos = findArray(s->prim, Token("vertex.position"));
      const auto *idx = findArray(s->prim, Token("primitive.index"));
      const auto *nrm = findArray(s->prim, Token("vertex.normal"));
      REQUIRE(pos != nullptr);
      REQUIRE(idx != nullptr);
      REQUIRE(nrm != nullptr);
      REQUIRE(pos->size() > 0);
      REQUIRE(idx->size() > 0);
      REQUIRE(nrm->size() == pos->size());
      const size_t nv = pos->size();
      for (size_t t = 0; t < idx->size(); ++t) {
        const uint3 tri = idx->get<uint3>(t);
        REQUIRE(tri.x < nv);
        REQUIRE(tri.y < nv);
        REQUIRE(tri.z < nv);
      }
      for (size_t i = 0; i < nv; ++i) {
        const float3 p = pos->get<float3>(i);
        REQUIRE(p.x >= -1.001f);
        REQUIRE(p.x <= 1.001f);
        REQUIRE(p.y >= -1.001f);
        REQUIRE(p.y <= 1.001f);
        REQUIRE(p.z >= -1.001f);
        REQUIRE(p.z <= 1.001f);
      }
    }
  }

  GIVEN("an isovalue outside the field range")
  {
    g.node(iso)->impl->parameters().set(Token("isovalue"), 100.f);
    Evaluator e(g);
    REQUIRE(e.pull(iso));
    auto s = std::static_pointer_cast<SurfaceData>(
        e.output(iso, Token("out"), hostResidency())->payload);
    THEN("it emits an empty surface without crashing")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("triangle"));
      const auto *pos = findArray(s->prim, Token("vertex.position"));
      REQUIRE((pos == nullptr || pos->size() == 0));
    }
  }
}

#endif // TSD_GRAPH_NODES_HAVE_VISKORES
```

In `tsd/tests/CMakeLists.txt`, add `test_nodes_Isosurface.cpp` to the `project_add_executable(...)` source list, on the line after `test_nodes_Surface.cpp`. (Do not reformat the file.)

- [ ] **Step 6: Build and run the test — expect RED**

Run: `cmake _out/_cmake && cmake --build _out/_cmake --parallel && ctest --test-dir _out/_cmake -C RelWithDebInfo -R nodes-isosurface --output-on-failure`
Expected: FAIL — `REQUIRE(e.pull(iso))` fails because the skeleton `evaluate` calls `ctx.fail`.

- [ ] **Step 7: Implement the full `evaluate`**

In `tsd/src/tsd/graph_nodes/IsosurfaceExtract.cpp`, replace the skeleton `evaluate` body (the single `ctx.fail(...)` line) with:

```cpp
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::static_pointer_cast<Field>(
        ctx.input(Token("in"), hostResidency()).payload);
    if (!f) {
      ctx.fail("IsosurfaceExtract: missing field input");
      return;
    }
    const uint3 dims = f->dims;
    const size_t n = size_t(dims.x) * dims.y * dims.z;
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u || f->data.size() != n) {
      ctx.fail("IsosurfaceExtract: field needs non-zero dims and matching data");
      return;
    }
    if (f->data.type() != ANARI_FLOAT32) {
      ctx.fail("IsosurfaceExtract: field data must be ANARI_FLOAT32");
      return;
    }

    ensureViskoresInit();

    viskores::cont::DataSetBuilderUniform builder;
    auto ds = builder.Create(
        viskores::Id3(
            viskores::Id(dims.x), viskores::Id(dims.y), viskores::Id(dims.z)),
        viskores::Vec3f(f->origin.x, f->origin.y, f->origin.z),
        viskores::Vec3f(f->spacing.x, f->spacing.y, f->spacing.z));

    viskores::cont::ArrayHandle<viskores::Float32> scalars;
    scalars.Allocate(viskores::Id(n));
    {
      auto portal = scalars.WritePortal();
      for (size_t i = 0; i < n; ++i)
        portal.Set(viskores::Id(i), f->data.get<float>(i));
    }
    ds.AddField(viskores::cont::Field(
        "scalars", viskores::cont::Field::Association::Points, scalars));

    const float isovalue = params.getOr<float>(Token("isovalue"), 0.5f);
    const bool computeNormals =
        params.getOr<bool>(Token("computeNormals"), true);
    viskores::filter::contour::Contour c;
    c.SetActiveField("scalars");
    c.SetIsoValue(viskores::Float64(isovalue));
    c.SetGenerateNormals(computeNormals);
    c.SetMergeDuplicatePoints(true);
    const viskores::cont::DataSet result = c.Execute(ds);

    auto s = std::make_shared<SurfaceData>();
    s->geomSubtype = Token("triangle");

    if (result.GetNumberOfPoints() > 0) {
      viskores::cont::ArrayHandle<viskores::Vec3f> pts;
      viskores::cont::ArrayCopy(result.GetCoordinateSystem().GetData(), pts);
      const viskores::Id nPts = pts.GetNumberOfValues();
      tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, size_t(nPts));
      {
        auto pp = pts.ReadPortal();
        for (viskores::Id i = 0; i < nPts; ++i) {
          const auto v = pp.Get(i);
          pos.get<float3>(size_t(i)) = float3(v[0], v[1], v[2]);
        }
      }
      s->prim.arrays.push_back({Token("vertex.position"), pos});

      viskores::cont::CellSetSingleType<> cells;
      result.GetCellSet().AsCellSet(cells);
      const auto conn = cells.GetConnectivityArray(
          viskores::TopologyElementTagCell{},
          viskores::TopologyElementTagPoint{});
      const viskores::Id nConn = conn.GetNumberOfValues();
      const size_t nTris = size_t(nConn / 3);
      tsd::core::AnyArray idx(ANARI_UINT32_VEC3, nTris);
      {
        auto cp = conn.ReadPortal();
        for (size_t t = 0; t < nTris; ++t)
          idx.get<uint3>(t) =
              uint3(uint32_t(cp.Get(viskores::Id(3 * t + 0))),
                  uint32_t(cp.Get(viskores::Id(3 * t + 1))),
                  uint32_t(cp.Get(viskores::Id(3 * t + 2))));
      }
      s->prim.arrays.push_back({Token("primitive.index"), idx});

      if (computeNormals && result.HasPointField("normals")) {
        viskores::cont::ArrayHandle<viskores::Vec3f> nrm;
        viskores::cont::ArrayCopy(result.GetField("normals").GetData(), nrm);
        const viskores::Id nN = nrm.GetNumberOfValues();
        tsd::core::AnyArray normals(ANARI_FLOAT32_VEC3, size_t(nN));
        auto np = nrm.ReadPortal();
        for (viskores::Id i = 0; i < nN; ++i) {
          const auto v = np.Get(i);
          normals.get<float3>(size_t(i)) = float3(v[0], v[1], v[2]);
        }
        s->prim.arrays.push_back({Token("vertex.normal"), normals});
      }
    }

    s->appearance.scalars.push_back({Token("color"),
        tsd::core::Any(params.getOr<float3>(Token("color"), float3(0.8f)))});

    Value out;
    out.type = PortType{portSurface()};
    out.residency = hostResidency();
    out.payload = s;
    ctx.setOutput(Token("out"), out);
  }
```

- [ ] **Step 8: Build and run the test — expect GREEN**

Run: `cmake --build _out/_cmake --parallel && ctest --test-dir _out/_cmake -C RelWithDebInfo -R nodes-isosurface --output-on-failure`
Expected: PASS — both `GIVEN`s pass (non-empty surface with valid indices + normals; empty surface for the out-of-range isovalue).

- [ ] **Step 9: Run the full suite**

Run: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: all pass (68 = prior 67 + the new isosurface test), no regressions.

- [ ] **Step 10: Commit**

```bash
jj commit tsd/src/tsd/graph_nodes/IsosurfaceExtract.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/src/tsd/graph_nodes/BuiltinNodes.hpp tsd/src/tsd/graph_nodes/BuiltinNodes.cpp tsd/tests/test_nodes_Isosurface.cpp tsd/tests/CMakeLists.txt -m "feat(tsdflow): IsosurfaceExtract node — Viskores contour field->triangle SurfaceData"
```

---

### Task 2: Wire the isosurface branch into the demo graph

**Files:**
- Modify: `tsd/src/tsd/graph_nodes/DemoGraph.cpp`

**Interfaces:**
- Consumes: `buildVolumeSurfaceDemo`'s local `add(const char*)->NodeId` and `link(NodeId, const char*, NodeId, const char*)` lambdas; the existing `src` node (a `GenerateNoiseVolume`); the `TSD_GRAPH_NODES_HAVE_VISKORES` macro (defined `PUBLIC` on `tsd_graph_nodes`, so it is in scope when compiling this file); the registered `IsosurfaceExtract` and `DisplaySurface` node types.
- Produces: no signature change. `DemoDisplays` is unchanged; the extra `DisplaySurface` is picked up by the app's generic display collection.

**Context:** `buildVolumeSurfaceDemo` (`DemoGraph.cpp`) builds the default scene and returns `DemoDisplays{src, dv, ds}`. The app's `collectDisplayMasks`/`collectDisplayTransforms` iterate every node, so a new display node needs no struct change. With the default field values in `[0,1]` and `isovalue` defaulting to `0.5`, the demo isosurface is non-empty.

- [ ] **Step 1: Add the conditional isosurface branch**

In `tsd/src/tsd/graph_nodes/DemoGraph.cpp`, locate the final lines of `buildVolumeSurfaceDemo`:

```cpp
  link(src, "out", bb, "in");
  link(bb, "out", ds, "in");

  return DemoDisplays{src, dv, ds};
```

Insert the guarded branch between the `link(bb, "out", ds, "in");` line and the `return`:

```cpp
  link(src, "out", bb, "in");
  link(bb, "out", ds, "in");

#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
  // Isosurface path: field -> IsosurfaceExtract -> its own DisplaySurface.
  const NodeId iso = add("IsosurfaceExtract");
  const NodeId dsIso = add("DisplaySurface");
  link(src, "out", iso, "in");
  link(iso, "out", dsIso, "in");
#endif

  return DemoDisplays{src, dv, ds};
```

- [ ] **Step 2: Build**

Run: `cmake --build _out/_cmake --parallel`
Expected: builds clean (`tsd_graph_nodes` and `tsdFlow` relink).

- [ ] **Step 3: Run the full suite**

Run: `ctest --test-dir _out/_cmake -C RelWithDebInfo --output-on-failure`
Expected: all pass, no regressions. (The existing `test_tsdflow_Smoke` / demo-graph tests now build a graph that includes the isosurface branch.)

- [ ] **Step 4: Commit**

```bash
jj commit tsd/src/tsd/graph_nodes/DemoGraph.cpp -m "feat(tsdflow): add isosurface branch to the demo graph"
```

---

## Self-Review

**Spec coverage:**
- Viskores `Contour` backend, optional/auto-detected, `TSD_GRAPH_NODES_HAVE_VISKORES` gate → Task 1 Steps 2, 3. ✓
- Node interface (`in`=field, `out`=surface; params `isovalue`=0.5, `computeNormals`=true, on the hash) → Task 1 Steps 1, 7. ✓
- evaluate(): validate → `DataSetBuilderUniform` + `AddField("scalars", Points)` → `Contour` (SetActiveField/SetIsoValue/SetGenerateNormals/SetMergeDuplicatePoints) → extract positions/`primitive.index`/normals → triangle `SurfaceData` + appearance color, host residency → Task 1 Step 7. ✓
- One-time guarded `viskores::cont::Initialize()`, default Serial device → Task 1 Steps 1, 7. ✓
- Edge cases: missing/zero-dim/non-FLOAT32 → `ctx.fail`; empty mesh → valid empty `SurfaceData` (no arrays pushed when `GetNumberOfPoints()==0`) → Task 1 Step 7 + test Step 5. ✓
- Demo graph branch, conditional, `DemoDisplays` unchanged → Task 2. ✓
- Test: sphere field, in-range (non-empty, valid indices < vertex count, in-bounds positions, normals present) + out-of-range (empty, no crash), CMake/macro-gated → Task 1 Steps 5–9. ✓

**Placeholder scan:** none — every code step carries complete code; the only `ctx.fail("...not yet implemented")` is the intentional Step-1 skeleton, replaced in Step 7.

**Type consistency:** `registerIsosurfaceExtract(NodeRegistry&)` declared (Step 3), defined (Step 1), called (Step 3); macro spelled `TSD_GRAPH_NODES_HAVE_VISKORES` in CMake define and all four guards; node string `"IsosurfaceExtract"` consistent across registration, demo `add(...)`, and test `addBuiltin(...)`; array tokens `vertex.position`/`primitive.index`/`vertex.normal` consistent between the node and the test's `findArray` checks; `AnyArray` element types `ANARI_FLOAT32_VEC3`/`ANARI_UINT32_VEC3` match the `.get<float3>`/`.get<uint3>` accessors.
