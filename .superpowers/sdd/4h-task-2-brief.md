## Task 2: `ITransformableNode` + display nodes + `collectDisplayTransforms`

**Files:** Create `tsd/src/tsd/graph_nodes/TransformableNode.hpp`, `tsd/src/tsd/graph_nodes/DisplayTransform.hpp`, `tsd/src/tsd/graph_nodes/DisplayTransform.cpp`; Modify `DisplayVolume.cpp`, `DisplaySurface.cpp`, `tsd/src/tsd/graph_nodes/CMakeLists.txt`; Test `tsd/tests/test_nodes_DisplayTransform.cpp`, `tsd/tests/CMakeLists.txt`.

**Interfaces:**
- Produces: `struct tsd::graph_nodes::ITransformableNode { virtual ~ITransformableNode()=default; virtual tsd::core::math::mat4 &transform()=0; };`
- Produces: `struct DisplayTransform { tsd::graph::NodeId node; tsd::core::math::mat4 xfm; };` and `std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g);`
- `DisplayVolume`/`DisplaySurface` now implement `ITransformableNode` (identity default).

- [ ] **Step 1: Write the failing test** — create `tsd/tests/test_nodes_DisplayTransform.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/DisplayTransform.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::collectDisplayTransforms;
using mat4 = tsd::core::math::mat4;

namespace {
const tsd::graph_nodes::DisplayTransform *find(
    const std::vector<tsd::graph_nodes::DisplayTransform> &v, NodeId id)
{
  for (const auto &dt : v)
    if (dt.node == id)
      return &dt;
  return nullptr;
}
} // namespace

SCENARIO("collectDisplayTransforms reports display transforms", "[display-transform]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);

  WHEN("transforms are default") {
    auto xs = collectDisplayTransforms(g);
    THEN("both display nodes appear at identity") {
      REQUIRE(xs.size() == 2);
      REQUIRE(find(xs, d.volumeDisplay) != nullptr);
      REQUIRE(find(xs, d.surfaceDisplay) != nullptr);
      REQUIRE(find(xs, d.volumeDisplay)->xfm == tsd::core::math::IDENTITY_MAT4);
    }
    THEN("non-display nodes are excluded") {
      REQUIRE(find(xs, d.source) == nullptr);
    }
  }

  WHEN("one display's transform is set via ITransformableNode") {
    mat4 m = tsd::core::math::IDENTITY_MAT4;
    m[3].x = 5.f; // translate +5 in x (column-major: column 3 = translation)
    auto *itf = dynamic_cast<tsd::graph_nodes::ITransformableNode *>(
        g.node(d.volumeDisplay)->impl.get());
    REQUIRE(itf != nullptr);
    itf->transform() = m;
    auto xs = collectDisplayTransforms(g);
    THEN("the helper reports it; the other stays identity") {
      REQUIRE(find(xs, d.volumeDisplay)->xfm == m);
      REQUIRE(find(xs, d.surfaceDisplay)->xfm == tsd::core::math::IDENTITY_MAT4);
    }
  }
}
```
(`mat4 operator==` is linalg's element-wise compare — available. If exact `==` is flaky, compare `m[3].x`; but identity vs a single edited element is exact.)

- [ ] **Step 2: Register the test** — edit `tsd/tests/CMakeLists.txt` by hand: add `test_nodes_DisplayTransform.cpp` to the source list and:
```cmake
add_test(NAME tsd::nodes::DisplayTransform COMMAND ${PROJECT_NAME} "[display-transform]")
```

- [ ] **Step 3: Build + confirm FAIL** — `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests 2>&1 | tail -15` → `TransformableNode.hpp`/`DisplayTransform.hpp` not found.

- [ ] **Step 4: Create `tsd/src/tsd/graph_nodes/TransformableNode.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"

namespace tsd::graph_nodes {

// Implemented by nodes carrying a render-time instance transform. Kept OUT of
// the node's ParameterList (so it never enters ParameterList::hash() and a
// transform edit never triggers re-evaluation / layer rebuild). UI reaches it
// via dynamic_cast, like ITransferFunctionNode.
struct ITransformableNode
{
  virtual ~ITransformableNode() = default;
  virtual tsd::core::math::mat4 &transform() = 0;
};

} // namespace tsd::graph_nodes
```

- [ ] **Step 5: Make the display nodes implement it.** In `tsd/src/tsd/graph_nodes/DisplayVolume.cpp`: add `#include "tsd/graph_nodes/TransformableNode.hpp"` near the includes; change `struct DisplayVolume : Node` to `struct DisplayVolume : Node, ITransformableNode`; add a member and accessor:

```cpp
  tsd::core::math::mat4 m_transform{tsd::core::math::IDENTITY_MAT4};
  tsd::core::math::mat4 &transform() override { return m_transform; }
```
(place the member next to `ParameterList params;`, the accessor among the other overrides). Do the **identical** change in `tsd/src/tsd/graph_nodes/DisplaySurface.cpp` (`struct DisplaySurface : Node, ITransformableNode`, same member + accessor + include). `evaluate()` is unchanged — it must NOT read `m_transform`.

- [ ] **Step 6: Create `tsd/src/tsd/graph_nodes/DisplayTransform.hpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"
#include "tsd/graph/Graph.hpp"
// std
#include <vector>

namespace tsd::graph_nodes {

struct DisplayTransform
{
  tsd::graph::NodeId node{0};
  tsd::core::math::mat4 xfm{tsd::core::math::IDENTITY_MAT4};
};

// Every node implementing ITransformableNode and its transform. Graph& non-const
// (node()->impl is non-const for the dynamic_cast). Logically read-only.
std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g);

} // namespace tsd::graph_nodes
```

- [ ] **Step 7: Create `tsd/src/tsd/graph_nodes/DisplayTransform.cpp`:**

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/DisplayTransform.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"

namespace tsd::graph_nodes {

using tsd::graph::NodeId;

std::vector<DisplayTransform> collectDisplayTransforms(tsd::graph::Graph &g)
{
  std::vector<DisplayTransform> out;
  for (const NodeId id : g.nodeIds()) {
    auto *gn = g.node(id);
    if (!gn || !gn->impl)
      continue;
    if (auto *itf = dynamic_cast<ITransformableNode *>(gn->impl.get()))
      out.push_back({id, itf->transform()});
  }
  return out;
}

} // namespace tsd::graph_nodes
```

- [ ] **Step 8: Add to CMake** — edit `tsd/src/tsd/graph_nodes/CMakeLists.txt` by hand: add `DisplayTransform.cpp` to `project_sources(PRIVATE ...)`.

- [ ] **Step 9: Build + run → PASS**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::nodes::DisplayTransform' --output-on-failure`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/TransformableNode.hpp tsd/src/tsd/graph_nodes/DisplayTransform.hpp tsd/src/tsd/graph_nodes/DisplayTransform.cpp tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/tests/test_nodes_DisplayTransform.cpp
jj commit tsd/src/tsd/graph_nodes/TransformableNode.hpp tsd/src/tsd/graph_nodes/DisplayTransform.hpp tsd/src/tsd/graph_nodes/DisplayTransform.cpp tsd/src/tsd/graph_nodes/DisplayVolume.cpp tsd/src/tsd/graph_nodes/DisplaySurface.cpp tsd/src/tsd/graph_nodes/CMakeLists.txt tsd/tests/test_nodes_DisplayTransform.cpp tsd/tests/CMakeLists.txt -m "feat(graph_nodes): ITransformableNode on display nodes + collectDisplayTransforms"
```

---

