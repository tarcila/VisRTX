# tsdFlow Phase 1 — Engine Core (headless) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the headless `tsd_graph` dataflow engine core — typed ports, backend-agnostic residency, version-stamped values, node/graph model with link validation and dirty propagation, and a synchronous lazy-pull evaluator with implicit transfer/conversion insertion — with no UI, no ANARI device/runtime use, and no CUDA dependency. (Note: `tsd_graph` links only `tsd_core`, but `tsd_core`'s `Any` includes `<anari/anari_cpp.hpp>`, so ANARI *headers* arrive transitively. "No ANARI" means no device/library linkage, not header-free.)

**Architecture:** A new static library `tsd_graph` depending only on `tsd_core`. Values carry a logical `PortType`, a `Residency` (backend + deviceId), a type-erased payload, and a monotonic `version` for O(1) cache invalidation. A `Graph` owns nodes (in an `ObjectPool`) and stable-id `Connection`s, validated at link time. An `Evaluator` pulls outputs depth-first, recomputing only stale nodes, inserting transfers/conversions from pluggable registries and recording every implicit op in an `EvalReport`. A fake `"test"` backend exercises residency logic without CUDA.

**Tech Stack:** C++17, Catch2 (BDD `SCENARIO`/`GIVEN`/`WHEN`/`THEN`), CMake (repo `project_*` macros), CTest, jujutsu (`jj`) for commits.

**Spec:** `docs/superpowers/specs/2026-06-16-tsd-dataflow-pipeline-app-design.md` (Phase 1 in the Phasing section).

---

## Conventions for every task

- **Version control is `jj`, not git.** The repo forbids raw git. A "commit" step runs `jj commit -m "<msg>"` (describes the current change and starts a new one). New files are tracked automatically — no `add` step.
- **File header** (top of every new `.hpp`/`.cpp`):
  ```cpp
  // Copyright 2026 NVIDIA Corporation
  // SPDX-License-Identifier: Apache-2.0
  ```
  Headers use `#pragma once`. Format with `clang-format -i <file>` (Google style) before committing.
- **Namespace:** all new types live in `namespace tsd::graph { ... } // namespace tsd::graph`.
- **Build + test loop:** configure once (Task 0), then from the build dir:
  ```bash
  cmake --build . --target tsdTests --parallel
  ctest -C Release -R 'tsd::graph' --output-on-failure
  ```
  Tests are tag-filtered; every `SCENARIO` below uses a `[graph-*]` tag and the CTest name `tsd::graph::*`.

---

## File structure

Created under `tsd/src/tsd/graph/`:

| File | Responsibility |
|------|----------------|
| `CMakeLists.txt` | Declare `tsd_graph` static lib, link `tsd_core` |
| `Residency.hpp` | `Residency` value type + backend token helpers |
| `PortType.hpp` / `PortType.cpp` | `PortType` + `PortTypeRegistry` singleton |
| `Value.hpp` | `Value` (type, residency, payload, version, contentTag) |
| `TransferRegistry.hpp` / `.cpp` | residency↔residency transfer functions + cost |
| `ConversionRegistry.hpp` / `.cpp` | type↔type conversion functions + cost |
| `Parameter.hpp` | `Parameter` / `ParameterList` over `tsd::core::Any` + hash |
| `Port.hpp` | `PortSpec`, `NodeTypeInfo` |
| `Node.hpp` | `Node` interface, `EvalContext` (declared here, defined in Evaluator) |
| `NodeRegistry.hpp` / `.cpp` | node-type factory registry + self-registration helper |
| `Graph.hpp` / `Graph.cpp` | nodes + connections, link validation, dirty propagation |
| `Evaluator.hpp` / `Evaluator.cpp` | lazy pull, per-output/per-residency cache, EvalReport |
| `TestBackend.hpp` / `.cpp` | fake `"test"` backend: payloads, transfers, sample nodes (test-support, but shipped so CI uses it) |

Tests created under `tsd/tests/` (one file per unit), registered in `tsd/tests/CMakeLists.txt`.

---

## Task 0: Scaffold the `tsd_graph` library and test wiring

**Files:**
- Create: `tsd/src/tsd/graph/CMakeLists.txt`
- Create: `tsd/src/tsd/graph/Version.hpp` (trivial seed so the lib has a header)
- Create: `tsd/src/tsd/graph/Version.cpp` (trivial seed so the lib has a source)
- Modify: `tsd/src/tsd/CMakeLists.txt` (add `add_subdirectory(graph)`)
- Modify: `tsd/tests/CMakeLists.txt` (add test file + link `tsd_graph` + register)
- Test: `tsd/tests/test_graph_Smoke.cpp`

- [ ] **Step 1: Create the seed header**

`tsd/src/tsd/graph/Version.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tsd::graph {

// Returns the tsd_graph engine ABI version. Bumped on breaking changes.
int engineVersion();

} // namespace tsd::graph
```

- [ ] **Step 2: Create the seed source**

`tsd/src/tsd/graph/Version.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Version.hpp"

namespace tsd::graph {

int engineVersion()
{
  return 1;
}

} // namespace tsd::graph
```

- [ ] **Step 3: Create the library CMakeLists**

`tsd/src/tsd/graph/CMakeLists.txt`:
```cmake
project(tsd_graph)

project_add_library(STATIC)

project_sources(PRIVATE
  Version.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_core)
```

- [ ] **Step 4: Register the subdirectory**

In `tsd/src/tsd/CMakeLists.txt`, in the "Always present components" block, add after `add_subdirectory(core)`:
```cmake
add_subdirectory(graph)
```

- [ ] **Step 5: Write the smoke test**

`tsd/tests/test_graph_Smoke.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Version.hpp"

SCENARIO("tsd::graph links and runs", "[graph-smoke]")
{
  GIVEN("the engine version")
  {
    THEN("it is at least 1")
    {
      REQUIRE(tsd::graph::engineVersion() >= 1);
    }
  }
}
```

- [ ] **Step 6: Register the test**

In `tsd/tests/CMakeLists.txt`, add `test_graph_Smoke.cpp` to the `project_add_executable(...)` source list, add `tsd_graph` to the `project_link_libraries(PRIVATE ...)` list, and add next to the other `add_test` lines:
```cmake
add_test(NAME tsd::graph::Smoke COMMAND ${PROJECT_NAME} "[graph-smoke]")
```

- [ ] **Step 7: Configure and build**

From the repo root:
```bash
cmake -S . -B build -DVISRTX_BUILD_TSD=ON -DTSD_BUILD_APPS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tsdTests --parallel
```
Expected: builds `tsd_graph` and `tsdTests` with no errors.

- [ ] **Step 8: Run the smoke test**

```bash
cd build && ctest -C Release -R 'tsd::graph::Smoke' --output-on-failure
```
Expected: PASS, 1 test.

- [ ] **Step 9: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Version.hpp tsd/src/tsd/graph/Version.cpp tsd/tests/test_graph_Smoke.cpp
jj commit -m "feat(graph): scaffold tsd_graph library and test wiring"
```

---

## Task 1: `Residency` value type

**Files:**
- Create: `tsd/src/tsd/graph/Residency.hpp`
- Test: `tsd/tests/test_graph_Residency.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_Residency.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Residency.hpp"

using tsd::graph::Residency;
using tsd::graph::hostResidency;

SCENARIO("tsd::graph::Residency equality", "[graph-residency]")
{
  GIVEN("the host residency")
  {
    auto h = hostResidency();
    THEN("its backend is \"host\" and deviceId is -1")
    {
      REQUIRE(h.backend == tsd::core::Token("host"));
      REQUIRE(h.deviceId == -1);
    }
    THEN("two host residencies compare equal")
    {
      REQUIRE(h == hostResidency());
    }
  }

  GIVEN("two cuda residencies on different devices")
  {
    Residency d0{tsd::core::Token("cuda"), 0};
    Residency d1{tsd::core::Token("cuda"), 1};
    THEN("they are not equal")
    {
      REQUIRE(d0 != d1);
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — `tsd/graph/Residency.hpp` not found.

- [ ] **Step 3: Write minimal implementation**

`tsd/src/tsd/graph/Residency.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Token.hpp"
// std
#include <functional>

namespace tsd::graph {

// Backend-agnostic memory residency: which backend owns a value and on which
// device. deviceId is -1 for host-resident data.
struct Residency
{
  tsd::core::Token backend;
  int deviceId{-1};
};

inline bool operator==(const Residency &a, const Residency &b)
{
  return a.backend == b.backend && a.deviceId == b.deviceId;
}

inline bool operator!=(const Residency &a, const Residency &b)
{
  return !(a == b);
}

inline Residency hostResidency()
{
  return Residency{tsd::core::Token("host"), -1};
}

// Strict-weak ordering for use as a std::map key. Orders by backend (interned
// pointer) then deviceId, so a value on CUDA device 0 is distinct from device 1.
struct ResidencyLess
{
  bool operator()(const Residency &a, const Residency &b) const
  {
    if (a.backend.value() != b.backend.value())
      return std::less<const void *>()(a.backend.value(), b.backend.value());
    return a.deviceId < b.deviceId;
  }
};

} // namespace tsd::graph
```

- [ ] **Step 4: Add the test to CMake and run it**

Add `test_graph_Residency.cpp` to the executable source list in `tsd/tests/CMakeLists.txt`, then add:
```cmake
add_test(NAME tsd::graph::Residency COMMAND ${PROJECT_NAME} "[graph-residency]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::Residency' --output-on-failure
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Residency.hpp tsd/tests/test_graph_Residency.cpp
jj commit -m "feat(graph): add Residency value type"
```

---

## Task 2: `PortType` and `PortTypeRegistry`

**Files:**
- Create: `tsd/src/tsd/graph/PortType.hpp`
- Create: `tsd/src/tsd/graph/PortType.cpp`
- Modify: `tsd/src/tsd/graph/CMakeLists.txt` (add `PortType.cpp`)
- Test: `tsd/tests/test_graph_PortType.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_PortType.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/PortType.hpp"

using tsd::graph::PortType;
using tsd::graph::PortTypeRegistry;

SCENARIO("tsd::graph::PortTypeRegistry registration", "[graph-porttype]")
{
  GIVEN("a fresh registry")
  {
    PortTypeRegistry reg;
    WHEN("a type is registered")
    {
      auto field = reg.registerType("spatialField");
      THEN("it reports as registered")
      {
        REQUIRE(reg.isRegistered(tsd::core::Token("spatialField")));
      }
      THEN("re-registering the same name yields an equal PortType")
      {
        auto field2 = reg.registerType("spatialField");
        REQUIRE(field == field2);
      }
      THEN("a different type is not equal")
      {
        auto range = reg.registerType("range");
        REQUIRE(field != range);
      }
    }
    THEN("an unregistered name reports false")
    {
      REQUIRE_FALSE(reg.isRegistered(tsd::core::Token("nope")));
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — `tsd/graph/PortType.hpp` not found.

- [ ] **Step 3: Write the header**

`tsd/src/tsd/graph/PortType.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Token.hpp"
// std
#include <functional>
#include <unordered_set>

namespace tsd::graph {

// A logical data type carried on a port/wire, identified by an interned Token.
struct PortType
{
  tsd::core::Token name;
};

inline bool operator==(const PortType &a, const PortType &b)
{
  return a.name == b.name;
}

inline bool operator!=(const PortType &a, const PortType &b)
{
  return !(a == b);
}

// Token has == / != but no operator<, so provide a strict-weak ordering for use
// as a std::map key. Token interning makes value() pointer-stable.
struct TokenLess
{
  bool operator()(const tsd::core::Token &a, const tsd::core::Token &b) const
  {
    return std::less<const void *>()(a.value(), b.value());
  }
};

// Tracks the set of known port types. Used at link time to validate that a
// connection references registered types.
struct PortTypeRegistry
{
  PortType registerType(const char *name);
  bool isRegistered(tsd::core::Token name) const;

 private:
  // Token interning makes value() pointer-stable, so we key on the raw pointer.
  std::unordered_set<const void *> m_known;
};

} // namespace tsd::graph
```

- [ ] **Step 4: Write the source**

`tsd/src/tsd/graph/PortType.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/PortType.hpp"

namespace tsd::graph {

PortType PortTypeRegistry::registerType(const char *name)
{
  tsd::core::Token t(name);
  m_known.insert(static_cast<const void *>(t.value()));
  return PortType{t};
}

bool PortTypeRegistry::isRegistered(tsd::core::Token name) const
{
  return m_known.count(static_cast<const void *>(name.value())) > 0;
}

} // namespace tsd::graph
```

- [ ] **Step 5: Add source to lib CMake**

In `tsd/src/tsd/graph/CMakeLists.txt`, add `PortType.cpp` to `project_sources(PRIVATE ...)`.

- [ ] **Step 6: Register test and run**

Add `test_graph_PortType.cpp` to the test executable sources, then:
```cmake
add_test(NAME tsd::graph::PortType COMMAND ${PROJECT_NAME} "[graph-porttype]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::PortType' --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/PortType.hpp tsd/src/tsd/graph/PortType.cpp tsd/tests/test_graph_PortType.cpp
jj commit -m "feat(graph): add PortType and PortTypeRegistry"
```

---

## Task 3: `Value` type with version stamps

**Files:**
- Create: `tsd/src/tsd/graph/Value.hpp`
- Test: `tsd/tests/test_graph_Value.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_Value.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Value.hpp"
// std
#include <memory>
#include <vector>

using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::Value;
using tsd::graph::hostResidency;

SCENARIO("tsd::graph::Value holds a typed, residency-tagged payload",
    "[graph-value]")
{
  GIVEN("a default Value")
  {
    Value v;
    THEN("it is invalid")
    {
      REQUIRE_FALSE(v.valid());
    }
  }

  GIVEN("a Value wrapping a host float buffer")
  {
    auto buf = std::make_shared<std::vector<float>>(std::vector<float>{1, 2, 3});
    Value v;
    v.type = PortType{tsd::core::Token("array")};
    v.residency = hostResidency();
    v.payload = buf;
    v.producerNodeId = 7;
    v.version = 42;

    THEN("it is valid and exposes its payload")
    {
      REQUIRE(v.valid());
      auto out = std::static_pointer_cast<std::vector<float>>(v.payload);
      REQUIRE(out->at(1) == 2.0f);
      REQUIRE(v.version == 42);
      REQUIRE(v.residency == hostResidency());
    }
    THEN("contentTag is unset by default")
    {
      REQUIRE_FALSE(v.contentTag.has_value());
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — `tsd/graph/Value.hpp` not found.

- [ ] **Step 3: Write the implementation**

`tsd/src/tsd/graph/Value.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/PortType.hpp"
#include "tsd/graph/Residency.hpp"
// std
#include <cstdint>
#include <memory>
#include <optional>

namespace tsd::graph {

// A value carried on a wire. Cache invalidation uses `version` (a monotonic
// stamp bumped by the producer on each re-emit) — never a content scan.
// `contentTag` is an optional equality tag populated only by cheap host-side
// producers where "identical output despite recompute" is worth detecting.
//
// The `payload`'s shared_ptr deleter is responsible for freeing with the
// correct backend allocator. A payload is valid only on its own `residency`.
struct Value
{
  PortType type;
  Residency residency;
  std::shared_ptr<void> payload;
  uint64_t producerNodeId{0};
  uint64_t version{0};
  std::optional<uint64_t> contentTag;

  bool valid() const
  {
    return static_cast<bool>(payload);
  }
};

} // namespace tsd::graph
```

- [ ] **Step 4: Register test and run**

Add `test_graph_Value.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::Value COMMAND ${PROJECT_NAME} "[graph-value]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::Value' --output-on-failure
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Value.hpp tsd/tests/test_graph_Value.cpp
jj commit -m "feat(graph): add Value type with version stamps"
```

---

## Task 4: `TransferRegistry`

**Files:**
- Create: `tsd/src/tsd/graph/TransferRegistry.hpp`
- Create: `tsd/src/tsd/graph/TransferRegistry.cpp`
- Modify: `tsd/src/tsd/graph/CMakeLists.txt`
- Test: `tsd/tests/test_graph_TransferRegistry.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_TransferRegistry.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/TransferRegistry.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::TransferRegistry;
using tsd::graph::Value;
using tsd::graph::hostResidency;

SCENARIO("tsd::graph::TransferRegistry lookup and apply", "[graph-transfer]")
{
  PortType arrayT{Token("array")};
  Residency testDev{Token("test"), 0};

  GIVEN("a registry with a host->test transfer for arrays")
  {
    TransferRegistry reg;
    reg.registerTransfer(arrayT, Token("host"), Token("test"),
        [](const Value &src, const Residency &target) {
          Value out = src;            // copy metadata
          out.residency = target;     // mark moved
          return out;
        },
        [](const Value &src) -> size_t {
          auto b = std::static_pointer_cast<std::vector<float>>(src.payload);
          return b->size() * sizeof(float);
        });

    WHEN("looking up host->test for arrays")
    {
      const auto *e = reg.find(arrayT, Token("host"), Token("test"));
      THEN("it is found")
      {
        REQUIRE(e != nullptr);
      }
      THEN("applying it retags residency and estimates cost")
      {
        auto buf = std::make_shared<std::vector<float>>(4);
        Value src;
        src.type = arrayT;
        src.residency = hostResidency();
        src.payload = buf;
        REQUIRE(e->estimateBytes(src) == 4 * sizeof(float));
        auto moved = e->fn(src, testDev);
        REQUIRE(moved.residency == testDev);
      }
    }

    WHEN("looking up an unregistered direction")
    {
      const auto *e = reg.find(arrayT, Token("test"), Token("host"));
      THEN("it is not found")
      {
        REQUIRE(e == nullptr);
      }
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — header not found.

- [ ] **Step 3: Write the header**

`tsd/src/tsd/graph/TransferRegistry.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Value.hpp"
// std
#include <functional>
#include <vector>

namespace tsd::graph {

// A registered residency->residency transfer for a given PortType.
struct TransferEntry
{
  PortType type;
  tsd::core::Token from;
  tsd::core::Token to;
  std::function<Value(const Value &src, const Residency &target)> fn;
  std::function<size_t(const Value &src)> estimateBytes;
};

// Holds transfer functions keyed by (PortType, fromBackend, toBackend).
// The engine core registers nothing; backends self-register their transfers.
struct TransferRegistry
{
  void registerTransfer(PortType type,
      tsd::core::Token from,
      tsd::core::Token to,
      std::function<Value(const Value &, const Residency &)> fn,
      std::function<size_t(const Value &)> estimateBytes);

  const TransferEntry *find(
      PortType type, tsd::core::Token from, tsd::core::Token to) const;

 private:
  std::vector<TransferEntry> m_entries;
};

} // namespace tsd::graph
```

- [ ] **Step 4: Write the source**

`tsd/src/tsd/graph/TransferRegistry.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/TransferRegistry.hpp"

namespace tsd::graph {

void TransferRegistry::registerTransfer(PortType type,
    tsd::core::Token from,
    tsd::core::Token to,
    std::function<Value(const Value &, const Residency &)> fn,
    std::function<size_t(const Value &)> estimateBytes)
{
  m_entries.push_back(
      TransferEntry{type, from, to, std::move(fn), std::move(estimateBytes)});
}

const TransferEntry *TransferRegistry::find(
    PortType type, tsd::core::Token from, tsd::core::Token to) const
{
  for (const auto &e : m_entries) {
    if (e.type == type && e.from == from && e.to == to)
      return &e;
  }
  return nullptr;
}

} // namespace tsd::graph
```

- [ ] **Step 5: Add source to lib CMake**

In `tsd/src/tsd/graph/CMakeLists.txt`, add `TransferRegistry.cpp`.

- [ ] **Step 6: Register test and run**

Add `test_graph_TransferRegistry.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::TransferRegistry COMMAND ${PROJECT_NAME} "[graph-transfer]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::TransferRegistry' --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/TransferRegistry.hpp tsd/src/tsd/graph/TransferRegistry.cpp tsd/tests/test_graph_TransferRegistry.cpp
jj commit -m "feat(graph): add TransferRegistry"
```

---

## Task 5: `ConversionRegistry`

**Files:**
- Create: `tsd/src/tsd/graph/ConversionRegistry.hpp`
- Create: `tsd/src/tsd/graph/ConversionRegistry.cpp`
- Modify: `tsd/src/tsd/graph/CMakeLists.txt`
- Test: `tsd/tests/test_graph_ConversionRegistry.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_ConversionRegistry.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/ConversionRegistry.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::ConversionRegistry;
using tsd::graph::PortType;
using tsd::graph::Value;
using tsd::graph::hostResidency;

SCENARIO("tsd::graph::ConversionRegistry lookup and apply", "[graph-convert]")
{
  PortType i32{Token("i32array")};
  PortType f32{Token("f32array")};

  GIVEN("a registry with an i32array->f32array conversion")
  {
    ConversionRegistry reg;
    reg.registerConversion(i32, f32,
        [](const Value &src) {
          auto in = std::static_pointer_cast<std::vector<int>>(src.payload);
          auto out = std::make_shared<std::vector<float>>();
          for (int x : *in)
            out->push_back(static_cast<float>(x));
          Value v = src;
          v.type = PortType{Token("f32array")};
          v.payload = out;
          return v;
        },
        [](const Value &src) -> size_t {
          return std::static_pointer_cast<std::vector<int>>(src.payload)->size();
        });

    WHEN("converting an i32 buffer")
    {
      auto in = std::make_shared<std::vector<int>>(std::vector<int>{1, 2, 3});
      Value src;
      src.type = i32;
      src.residency = hostResidency();
      src.payload = in;

      const auto *e = reg.find(i32, f32);
      THEN("the conversion exists and produces floats")
      {
        REQUIRE(e != nullptr);
        REQUIRE(e->estimateElements(src) == 3);
        auto out = e->fn(src);
        REQUIRE(out.type == f32);
        auto fb = std::static_pointer_cast<std::vector<float>>(out.payload);
        REQUIRE(fb->at(2) == 3.0f);
      }
    }

    WHEN("looking up a missing conversion")
    {
      THEN("it is not found")
      {
        REQUIRE(reg.find(f32, i32) == nullptr);
      }
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — header not found.

- [ ] **Step 3: Write the header**

`tsd/src/tsd/graph/ConversionRegistry.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Value.hpp"
// std
#include <functional>
#include <vector>

namespace tsd::graph {

// A registered type->type conversion.
struct ConversionEntry
{
  PortType from;
  PortType to;
  std::function<Value(const Value &src)> fn;
  std::function<size_t(const Value &src)> estimateElements;
};

// Holds conversion functions keyed by (fromType, toType).
struct ConversionRegistry
{
  void registerConversion(PortType from,
      PortType to,
      std::function<Value(const Value &)> fn,
      std::function<size_t(const Value &)> estimateElements);

  const ConversionEntry *find(PortType from, PortType to) const;

 private:
  std::vector<ConversionEntry> m_entries;
};

} // namespace tsd::graph
```

- [ ] **Step 4: Write the source**

`tsd/src/tsd/graph/ConversionRegistry.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/ConversionRegistry.hpp"

namespace tsd::graph {

void ConversionRegistry::registerConversion(PortType from,
    PortType to,
    std::function<Value(const Value &)> fn,
    std::function<size_t(const Value &)> estimateElements)
{
  m_entries.push_back(
      ConversionEntry{from, to, std::move(fn), std::move(estimateElements)});
}

const ConversionEntry *ConversionRegistry::find(PortType from, PortType to) const
{
  for (const auto &e : m_entries) {
    if (e.from == from && e.to == to)
      return &e;
  }
  return nullptr;
}

} // namespace tsd::graph
```

- [ ] **Step 5: Add source to lib CMake**

Add `ConversionRegistry.cpp` to `tsd/src/tsd/graph/CMakeLists.txt`.

- [ ] **Step 6: Register test and run**

Add `test_graph_ConversionRegistry.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::ConversionRegistry COMMAND ${PROJECT_NAME} "[graph-convert]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::ConversionRegistry' --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/ConversionRegistry.hpp tsd/src/tsd/graph/ConversionRegistry.cpp tsd/tests/test_graph_ConversionRegistry.cpp
jj commit -m "feat(graph): add ConversionRegistry"
```

---

## Task 6: `Parameter` / `ParameterList` (over `tsd::core::Any`)

**Note:** The spec said "reuse `tsd::Parameter`", but that type lives in `tsd_scene`, which would violate the rule that `tsd_graph` depends only on `tsd_core`. We define a minimal `ParameterList` over `tsd::core::Any` here; Phase 4 adapts it to the existing Parameter UI.

**Files:**
- Create: `tsd/src/tsd/graph/Parameter.hpp`
- Test: `tsd/tests/test_graph_Parameter.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_Parameter.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Parameter.hpp"

using tsd::core::Token;
using tsd::graph::ParameterList;

SCENARIO("tsd::graph::ParameterList stores values and hashes", "[graph-param]")
{
  GIVEN("a parameter list with two values")
  {
    ParameterList p;
    p.set(Token("iso"), 0.5f);
    p.set(Token("count"), 3);

    THEN("values read back by name")
    {
      REQUIRE(p.get<float>(Token("iso")) == 0.5f);
      REQUIRE(p.get<int>(Token("count")) == 3);
    }
    THEN("a missing param falls back to the default")
    {
      REQUIRE(p.getOr<float>(Token("missing"), -1.0f) == -1.0f);
    }
    THEN("hash is stable for identical content")
    {
      ParameterList q;
      q.set(Token("iso"), 0.5f);
      q.set(Token("count"), 3);
      REQUIRE(p.hash() == q.hash());
    }
    THEN("hash changes when a value changes")
    {
      auto before = p.hash();
      p.set(Token("iso"), 0.6f);
      REQUIRE(p.hash() != before);
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — header not found.

- [ ] **Step 3: Write the implementation**

`tsd/src/tsd/graph/Parameter.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/Token.hpp"
// std
#include <cstdint>
#include <string>
#include <vector>

namespace tsd::graph {

struct Parameter
{
  tsd::core::Token name;
  tsd::core::Any value;
};

// An ordered name->Any map owned by a Node. `hash()` feeds the evaluator's
// cache-validity check (a node recomputes when its param hash changes).
struct ParameterList
{
  template <typename T>
  void set(tsd::core::Token name, T v)
  {
    for (auto &p : m_params) {
      if (p.name == name) {
        p.value = tsd::core::Any(v);
        return;
      }
    }
    m_params.push_back(Parameter{name, tsd::core::Any(v)});
  }

  template <typename T>
  T get(tsd::core::Token name) const
  {
    for (const auto &p : m_params) {
      if (p.name == name)
        return p.value.get<T>();
    }
    return T{};
  }

  template <typename T>
  T getOr(tsd::core::Token name, const T &alt) const
  {
    for (const auto &p : m_params) {
      if (p.name == name)
        return p.value.getValueOr<T>(alt);
    }
    return alt;
  }

  bool has(tsd::core::Token name) const
  {
    for (const auto &p : m_params)
      if (p.name == name)
        return true;
    return false;
  }

  const std::vector<Parameter> &items() const
  {
    return m_params;
  }

  // Order-independent content hash: each param contributes its name pointer and
  // the bytes of its Any payload (exact-bit for floats).
  uint64_t hash() const
  {
    uint64_t h = 1469598103934665603ull; // FNV-1a offset basis
    auto mix = [&](const void *data, size_t n) {
      const auto *bytes = static_cast<const uint8_t *>(data);
      for (size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
      }
    };
    for (const auto &p : m_params) {
      const void *namePtr = p.name.value();
      mix(&namePtr, sizeof(namePtr));
      if (!p.value.valid())
        continue;
      auto type = p.value.type();
      mix(&type, sizeof(type));
      if (type == ANARI_STRING) {
        const std::string s = p.value.getString();
        mix(s.data(), s.size());
      } else {
        // Hash exactly the bytes this ANARI type occupies — never the full
        // fixed-size Any storage, which would mix uninitialized padding and
        // (for strings) over-read the heap buffer.
        mix(p.value.data(), anari::sizeOf(type));
      }
    }
    return h;
  }

 private:
  std::vector<Parameter> m_params;
};

} // namespace tsd::graph
```

- [ ] **Step 4: Register test and run**

Add `test_graph_Parameter.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::Parameter COMMAND ${PROJECT_NAME} "[graph-param]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::Parameter' --output-on-failure
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Parameter.hpp tsd/tests/test_graph_Parameter.cpp
jj commit -m "feat(graph): add ParameterList over Any with content hash"
```

---

## Task 7: `PortSpec`, `NodeTypeInfo`, and the `Node` interface

**Files:**
- Create: `tsd/src/tsd/graph/Port.hpp`
- Create: `tsd/src/tsd/graph/Node.hpp`
- Test: `tsd/tests/test_graph_Node.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_Node.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Node.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;

namespace {

// A minimal concrete node used only to exercise the interface.
struct ConstantNode : Node
{
  ParameterList params;

  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo info;
    info.name = Token("Constant");
    info.category = Token("source");
    info.outputs.push_back(
        PortSpec{Token("out"), PortType{Token("scalar")}, true, {}});
    info.isCacheable = true;
    return info;
  }

  ParameterList &parameters() override
  {
    return params;
  }

  void evaluate(EvalContext &) override {}
};

} // namespace

SCENARIO("tsd::graph::Node exposes type info and params", "[graph-node]")
{
  GIVEN("a ConstantNode")
  {
    ConstantNode n;
    THEN("its type info names one output and is cacheable")
    {
      auto info = n.typeInfo();
      REQUIRE(info.name == Token("Constant"));
      REQUIRE(info.outputs.size() == 1);
      REQUIRE(info.outputs[0].name == Token("out"));
      REQUIRE(info.outputs[0].required);
      REQUIRE(info.isCacheable);
    }
    THEN("its parameter list is reachable and mutable")
    {
      n.parameters().set(Token("v"), 2.0f);
      REQUIRE(n.parameters().get<float>(Token("v")) == 2.0f);
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — `tsd/graph/Node.hpp` not found.

- [ ] **Step 3: Write the Port header**

`tsd/src/tsd/graph/Port.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/PortType.hpp"
// std
#include <vector>

namespace tsd::graph {

// A node port. `acceptedBackends` lists residency backends this input can
// consume directly; empty means "any" (host-preferred). For outputs the field
// is unused.
struct PortSpec
{
  tsd::core::Token name;
  PortType type;
  bool required{true};
  std::vector<tsd::core::Token> acceptedBackends;
};

// Static description of a node type. UI is generated from this plus the node's
// ParameterList.
struct NodeTypeInfo
{
  tsd::core::Token name;
  tsd::core::Token category;
  std::vector<PortSpec> inputs;
  std::vector<PortSpec> outputs;
  bool isCacheable{true};
};

} // namespace tsd::graph
```

- [ ] **Step 4: Write the Node header**

`tsd/src/tsd/graph/Node.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Parameter.hpp"
#include "tsd/graph/Port.hpp"

namespace tsd::graph {

// Forward-declared; defined alongside the Evaluator (Task 11) since it bridges
// a node to the evaluator's input cache and transfer machinery.
class EvalContext;

// Interface every node type implements. evaluate() pulls inputs and sets
// outputs through the EvalContext; it never performs transfers itself.
class Node
{
 public:
  virtual ~Node() = default;
  virtual NodeTypeInfo typeInfo() const = 0;
  virtual ParameterList &parameters() = 0;
  virtual void evaluate(EvalContext &ctx) = 0;
};

} // namespace tsd::graph
```

- [ ] **Step 5: Register test and run**

Add `test_graph_Node.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::Node COMMAND ${PROJECT_NAME} "[graph-node]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::Node' --output-on-failure
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Port.hpp tsd/src/tsd/graph/Node.hpp tsd/tests/test_graph_Node.cpp
jj commit -m "feat(graph): add PortSpec, NodeTypeInfo, and Node interface"
```

---

## Task 8: `NodeRegistry` with self-registration

**Files:**
- Create: `tsd/src/tsd/graph/NodeRegistry.hpp`
- Create: `tsd/src/tsd/graph/NodeRegistry.cpp`
- Modify: `tsd/src/tsd/graph/CMakeLists.txt`
- Test: `tsd/tests/test_graph_NodeRegistry.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_NodeRegistry.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/NodeRegistry.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Node;
using tsd::graph::NodeRegistry;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;

namespace {

struct DummyNode : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo info;
    info.name = Token("Dummy");
    return info;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &) override {}
};

} // namespace

// Self-register DummyNode under a distinct name at static-init.
TSD_GRAPH_REGISTER_NODE("AutoDummy", DummyNode)

SCENARIO("tsd::graph::NodeRegistry creates registered types", "[graph-noderegistry]")
{
  GIVEN("a registry with Dummy registered")
  {
    NodeRegistry reg;
    reg.registerType(Token("Dummy"), [] { return std::make_unique<DummyNode>(); });

    WHEN("creating a Dummy")
    {
      auto n = reg.create(Token("Dummy"));
      THEN("a node is returned with the right type name")
      {
        REQUIRE(n != nullptr);
        REQUIRE(n->typeInfo().name == Token("Dummy"));
      }
    }
    WHEN("creating an unknown type")
    {
      THEN("nullptr is returned")
      {
        REQUIRE(reg.create(Token("Nope")) == nullptr);
      }
    }
  }
}

SCENARIO("tsd::graph node types self-register at static init",
    "[graph-noderegistry]")
{
  GIVEN("the process-global registry")
  {
    THEN("a type registered via TSD_GRAPH_REGISTER_NODE is present")
    {
      REQUIRE(
          tsd::graph::GlobalNodeRegistry().isRegistered(Token("AutoDummy")));
      auto n = tsd::graph::GlobalNodeRegistry().create(Token("AutoDummy"));
      REQUIRE(n != nullptr);
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — header not found.

- [ ] **Step 3: Write the header**

`tsd/src/tsd/graph/NodeRegistry.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Node.hpp"
// std
#include <functional>
#include <memory>
#include <vector>

namespace tsd::graph {

using NodeFactory = std::function<std::unique_ptr<Node>()>;

// Maps a node-type Token to a factory. Built-in node types self-register into
// a process-global registry via the GlobalNodeRegistry() accessor.
struct NodeRegistry
{
  void registerType(tsd::core::Token name, NodeFactory factory);
  std::unique_ptr<Node> create(tsd::core::Token name) const;
  bool isRegistered(tsd::core::Token name) const;

 private:
  struct Entry
  {
    tsd::core::Token name;
    NodeFactory factory;
  };
  std::vector<Entry> m_entries;
};

NodeRegistry &GlobalNodeRegistry();

// RAII registrar: a static instance self-registers a node type at static-init.
struct NodeRegistrar
{
  NodeRegistrar(tsd::core::Token name, NodeFactory factory)
  {
    GlobalNodeRegistry().registerType(name, std::move(factory));
  }
};

} // namespace tsd::graph

// Place in a node type's .cpp to self-register it:
//   TSD_GRAPH_REGISTER_NODE("MyNode", MyNodeClass)
#define TSD_GRAPH_REGISTER_NODE(NAME, TYPE)                                    \
  namespace {                                                                  \
  const ::tsd::graph::NodeRegistrar s_registrar_##TYPE(                        \
      ::tsd::core::Token(NAME), [] { return std::make_unique<TYPE>(); });      \
  }
```

> **Static-library caveat (carried to Phase 4):** node types compiled into the
> `tsd_graph` static lib whose only purpose is a static-init registrar can be
> stripped by the linker (the object is never otherwise referenced). When the
> real catalog lands in Phase 4, the app target must force-link the catalog
> objects (`--whole-archive` / `/WHOLEARCHIVE` / CMake `$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`)
> or register via an explicit `registerBuiltinNodes()` call. Phase 1 nodes live
> in test translation units, so the registrar runs reliably there.

- [ ] **Step 4: Write the source**

`tsd/src/tsd/graph/NodeRegistry.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/NodeRegistry.hpp"

namespace tsd::graph {

void NodeRegistry::registerType(tsd::core::Token name, NodeFactory factory)
{
  m_entries.push_back(Entry{name, std::move(factory)});
}

std::unique_ptr<Node> NodeRegistry::create(tsd::core::Token name) const
{
  for (const auto &e : m_entries) {
    if (e.name == name)
      return e.factory();
  }
  return nullptr;
}

bool NodeRegistry::isRegistered(tsd::core::Token name) const
{
  for (const auto &e : m_entries)
    if (e.name == name)
      return true;
  return false;
}

NodeRegistry &GlobalNodeRegistry()
{
  static NodeRegistry registry;
  return registry;
}

} // namespace tsd::graph
```

- [ ] **Step 5: Add source to lib CMake**

Add `NodeRegistry.cpp` to `tsd/src/tsd/graph/CMakeLists.txt`.

- [ ] **Step 6: Register test and run**

Add `test_graph_NodeRegistry.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::NodeRegistry COMMAND ${PROJECT_NAME} "[graph-noderegistry]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::NodeRegistry' --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/NodeRegistry.hpp tsd/src/tsd/graph/NodeRegistry.cpp tsd/tests/test_graph_NodeRegistry.cpp
jj commit -m "feat(graph): add NodeRegistry with global self-registration"
```

---

## Task 9: `Graph` — nodes, stable-id connections, link validation

**Files:**
- Create: `tsd/src/tsd/graph/Graph.hpp`
- Create: `tsd/src/tsd/graph/Graph.cpp`
- Modify: `tsd/src/tsd/graph/CMakeLists.txt`
- Test: `tsd/tests/test_graph_GraphLinks.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_GraphLinks.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;

namespace {

// Source: one "field" output. Sink: one "field" input.
struct SourceNode : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Source");
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &) override {}
};

struct SinkNode : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Sink");
    i.inputs.push_back({Token("in"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &) override {}
};

// A sink expecting an incompatible type with no registered conversion.
struct ColorSink : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ColorSink");
    i.inputs.push_back({Token("in"), PortType{Token("color")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &) override {}
};

// Has BOTH a field input and a field output, so it can form real cycles.
struct PassThrough : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("PassThrough");
    i.inputs.push_back({Token("in"), PortType{Token("field")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &) override {}
};

} // namespace

SCENARIO("tsd::graph::Graph link validation", "[graph-links]")
{
  GIVEN("a graph with a source and a sink")
  {
    Graph g;
    auto src = g.addNode(std::make_unique<SourceNode>());
    auto sink = g.addNode(std::make_unique<SinkNode>());

    WHEN("connecting matching types")
    {
      auto r = g.connect(src, Token("out"), sink, Token("in"));
      THEN("the link succeeds with a stable id")
      {
        REQUIRE(r.ok);
        REQUIRE(r.id != tsd::graph::INVALID_CONNECTION);
        REQUIRE(g.connections().size() == 1);
      }
    }

    WHEN("connecting to a mismatched type with no conversion")
    {
      auto csink = g.addNode(std::make_unique<ColorSink>());
      auto r = g.connect(src, Token("out"), csink, Token("in"));
      THEN("the link is rejected with a reason")
      {
        REQUIRE_FALSE(r.ok);
        REQUIRE_FALSE(r.reason.empty());
        REQUIRE(g.connections().empty());
      }
    }

    WHEN("a connection would create a cycle between two passthrough nodes")
    {
      auto a = g.addNode(std::make_unique<PassThrough>());
      auto b = g.addNode(std::make_unique<PassThrough>());
      auto ab = g.connect(a, Token("out"), b, Token("in"));
      REQUIRE(ab.ok);
      // b -> a closes the loop and must be rejected for the cycle reason
      // (both ports exist, so the cycle check is actually exercised).
      auto ba = g.connect(b, Token("out"), a, Token("in"));
      THEN("it is rejected with the cycle reason")
      {
        REQUIRE_FALSE(ba.ok);
        REQUIRE(ba.reason == "connection would create a cycle");
        REQUIRE(g.connections().size() == 1);
      }
    }

    WHEN("a passthrough node is connected to itself")
    {
      auto p = g.addNode(std::make_unique<PassThrough>());
      auto r = g.connect(p, Token("out"), p, Token("in"));
      THEN("it is rejected with the cycle reason")
      {
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.reason == "connection would create a cycle");
      }
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — header not found.

- [ ] **Step 3: Write the header**

`tsd/src/tsd/graph/Graph.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/ConversionRegistry.hpp"
#include "tsd/graph/Node.hpp"
#include "tsd/graph/Value.hpp"
// std
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tsd::graph {

using NodeId = uint64_t;
using ConnectionId = uint64_t;

constexpr NodeId INVALID_NODE = 0;
constexpr ConnectionId INVALID_CONNECTION = 0;

enum class EvalState
{
  Clean,
  Dirty,
  Computing,
  Error
};

struct Connection
{
  ConnectionId id{INVALID_CONNECTION};
  NodeId fromNode{INVALID_NODE};
  tsd::core::Token fromPort;
  NodeId toNode{INVALID_NODE};
  tsd::core::Token toPort;
};

struct LinkResult
{
  bool ok{false};
  ConnectionId id{INVALID_CONNECTION};
  std::string reason;
};

// One output's cached results, keyed by full Residency (backend + deviceId) so
// a copy on device 0 never satisfies a device-1 consumer. Fan-out to consumers
// on different residencies each gets a correctly-resident copy.
using OutputCache = std::map<Residency, Value, ResidencyLess>;

struct GraphNode
{
  NodeId id{INVALID_NODE};
  std::unique_ptr<Node> impl;
  EvalState state{EvalState::Dirty};
  std::string error;
  uint64_t outputVersion{0}; // bumped on each recompute that changes outputs
  bool hasEvaluated{false};
  uint64_t lastParamHash{0};
  // input port -> producer outputVersion consumed at last evaluate
  std::map<tsd::core::Token, uint64_t, TokenLess> consumedInputVersions;
  // outputName -> (residency -> Value)
  std::map<tsd::core::Token, OutputCache, TokenLess> cache;
};

// Owns nodes and connections. Validates connections at link time (type compat
// via exact match or a registered conversion; cycle rejection). Residency
// mismatch is never a link error — it is resolved during evaluation.
class Graph
{
 public:
  explicit Graph(const ConversionRegistry *conversions = nullptr);

  NodeId addNode(std::unique_ptr<Node> node);
  void removeNode(NodeId id);

  LinkResult connect(
      NodeId from, tsd::core::Token fromPort, NodeId to, tsd::core::Token toPort);
  void disconnect(ConnectionId id);

  GraphNode *node(NodeId id);
  const GraphNode *node(NodeId id) const;
  const std::vector<Connection> &connections() const;

  // Connection feeding a given (node,input). Null if unconnected.
  const Connection *inputConnection(NodeId to, tsd::core::Token toPort) const;

  void setConversionRegistry(const ConversionRegistry *r);

 private:
  bool wouldCreateCycle(NodeId from, NodeId to) const;
  // Search by value (typeInfo() returns a temporary, so never return a pointer
  // into it). Returns true and fills `out` if the port exists.
  bool findOutputSpec(
      const GraphNode &n, tsd::core::Token port, PortSpec &out) const;
  bool findInputSpec(
      const GraphNode &n, tsd::core::Token port, PortSpec &out) const;

  std::map<NodeId, GraphNode> m_nodes;
  std::vector<Connection> m_connections;
  NodeId m_nextNodeId{1};
  ConnectionId m_nextConnId{1};
  const ConversionRegistry *m_conversions{nullptr};
};

} // namespace tsd::graph
```

- [ ] **Step 4: Write the source**

`tsd/src/tsd/graph/Graph.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Graph.hpp"
// std
#include <functional>

namespace tsd::graph {

Graph::Graph(const ConversionRegistry *conversions) : m_conversions(conversions)
{}

void Graph::setConversionRegistry(const ConversionRegistry *r)
{
  m_conversions = r;
}

NodeId Graph::addNode(std::unique_ptr<Node> node)
{
  NodeId id = m_nextNodeId++;
  GraphNode gn;
  gn.id = id;
  gn.impl = std::move(node);
  gn.state = EvalState::Dirty;
  m_nodes.emplace(id, std::move(gn));
  return id;
}

GraphNode *Graph::node(NodeId id)
{
  auto it = m_nodes.find(id);
  return it == m_nodes.end() ? nullptr : &it->second;
}

const GraphNode *Graph::node(NodeId id) const
{
  auto it = m_nodes.find(id);
  return it == m_nodes.end() ? nullptr : &it->second;
}

const std::vector<Connection> &Graph::connections() const
{
  return m_connections;
}

bool Graph::findOutputSpec(
    const GraphNode &n, tsd::core::Token port, PortSpec &out) const
{
  auto info = n.impl->typeInfo();
  for (const auto &p : info.outputs) {
    if (p.name == port) {
      out = p;
      return true;
    }
  }
  return false;
}

bool Graph::findInputSpec(
    const GraphNode &n, tsd::core::Token port, PortSpec &out) const
{
  auto info = n.impl->typeInfo();
  for (const auto &p : info.inputs) {
    if (p.name == port) {
      out = p;
      return true;
    }
  }
  return false;
}

const Connection *Graph::inputConnection(
    NodeId to, tsd::core::Token toPort) const
{
  for (const auto &c : m_connections)
    if (c.toNode == to && c.toPort == toPort)
      return &c;
  return nullptr;
}

bool Graph::wouldCreateCycle(NodeId from, NodeId to) const
{
  // A new edge from->to creates a cycle iff `from` is reachable starting at
  // `to` along existing edges. Self-edge is the trivial case.
  if (from == to)
    return true;
  std::function<bool(NodeId)> reaches = [&](NodeId start) {
    for (const auto &c : m_connections) {
      if (c.fromNode == start) {
        if (c.toNode == from)
          return true;
        if (reaches(c.toNode))
          return true;
      }
    }
    return false;
  };
  return reaches(to);
}

LinkResult Graph::connect(
    NodeId from, tsd::core::Token fromPort, NodeId to, tsd::core::Token toPort)
{
  auto *fromN = node(from);
  auto *toN = node(to);
  if (!fromN || !toN)
    return {false, INVALID_CONNECTION, "unknown node"};

  PortSpec outSpec, inSpec;
  if (!findOutputSpec(*fromN, fromPort, outSpec))
    return {false, INVALID_CONNECTION, "no such output port"};
  if (!findInputSpec(*toN, toPort, inSpec))
    return {false, INVALID_CONNECTION, "no such input port"};

  if (wouldCreateCycle(from, to))
    return {false, INVALID_CONNECTION, "connection would create a cycle"};

  // Type compatibility: exact match, or a registered conversion exists.
  if (outSpec.type != inSpec.type) {
    const bool convertible = m_conversions
        && m_conversions->find(outSpec.type, inSpec.type) != nullptr;
    if (!convertible) {
      return {false,
          INVALID_CONNECTION,
          "incompatible port types and no registered conversion"};
    }
  }

  ConnectionId id = m_nextConnId++;
  m_connections.push_back(Connection{id, from, fromPort, to, toPort});

  // New incoming data invalidates the consumer's cached output. (Full downstream
  // propagation is markDirty, added in Task 10; clearing the immediate target's
  // cache makes the evaluator recompute it on the next pull.)
  toN->state = EvalState::Dirty;
  toN->cache.clear();
  return {true, id, ""};
}

void Graph::disconnect(ConnectionId id)
{
  for (auto it = m_connections.begin(); it != m_connections.end(); ++it) {
    if (it->id == id) {
      if (auto *toN = node(it->toNode)) {
        toN->state = EvalState::Dirty;
        toN->cache.clear();
      }
      m_connections.erase(it);
      return;
    }
  }
}

void Graph::removeNode(NodeId id)
{
  // Drop connections touching this node; mark former consumers dirty.
  for (auto it = m_connections.begin(); it != m_connections.end();) {
    if (it->fromNode == id || it->toNode == id) {
      if (it->fromNode == id) {
        if (auto *toN = node(it->toNode))
          toN->state = EvalState::Dirty;
      }
      it = m_connections.erase(it);
    } else {
      ++it;
    }
  }
  m_nodes.erase(id);
}

} // namespace tsd::graph
```

> **Implementation note:** `findOutputSpec`/`findInputSpec` copy the matched `PortSpec` into an out-param because `typeInfo()` returns a temporary `NodeTypeInfo` — never return a pointer into it.

- [ ] **Step 5: Add source to lib CMake**

Add `Graph.cpp` to `tsd/src/tsd/graph/CMakeLists.txt`.

- [ ] **Step 6: Register test and run**

Add `test_graph_GraphLinks.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::GraphLinks COMMAND ${PROJECT_NAME} "[graph-links]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::GraphLinks' --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Graph.hpp tsd/src/tsd/graph/Graph.cpp tsd/tests/test_graph_GraphLinks.cpp
jj commit -m "feat(graph): add Graph with stable-id connections and link validation"
```

---

## Task 10: Dirty propagation and deletion semantics

**Files:**
- Modify: `tsd/src/tsd/graph/Graph.hpp` (add `markDirty`, downstream propagation, missing-required-input detection)
- Modify: `tsd/src/tsd/graph/Graph.cpp`
- Test: `tsd/tests/test_graph_Dirty.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_Dirty.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::EvalState;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;

namespace {

struct PassThrough : Node
{
  ParameterList params;
  bool isSource;
  explicit PassThrough(bool source) : isSource(source) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("PT");
    if (!isSource)
      i.inputs.push_back({Token("in"), PortType{Token("field")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("field")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &) override {}
};

} // namespace

SCENARIO("tsd::graph::Graph dirty propagation and deletion", "[graph-dirty]")
{
  GIVEN("a chain A -> B -> C, all Clean")
  {
    Graph g;
    auto a = g.addNode(std::make_unique<PassThrough>(true));
    auto b = g.addNode(std::make_unique<PassThrough>(false));
    auto c = g.addNode(std::make_unique<PassThrough>(false));
    g.connect(a, Token("out"), b, Token("in"));
    g.connect(b, Token("out"), c, Token("in"));
    g.node(a)->state = EvalState::Clean;
    g.node(b)->state = EvalState::Clean;
    g.node(c)->state = EvalState::Clean;

    WHEN("A is marked dirty")
    {
      g.markDirty(a);
      THEN("A, B, and C are all dirty")
      {
        REQUIRE(g.node(a)->state == EvalState::Dirty);
        REQUIRE(g.node(b)->state == EvalState::Dirty);
        REQUIRE(g.node(c)->state == EvalState::Dirty);
      }
    }

    WHEN("B is deleted")
    {
      g.node(a)->state = EvalState::Clean;
      g.node(c)->state = EvalState::Clean;
      g.removeNode(b);
      THEN("C lost a required input and is in Error")
      {
        REQUIRE(g.node(c)->state == EvalState::Error);
        REQUIRE_FALSE(g.node(c)->error.empty());
      }
      THEN("A is untouched")
      {
        REQUIRE(g.node(a)->state == EvalState::Clean);
      }
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL — `markDirty` not declared; deletion does not set Error.

- [ ] **Step 3: Declare `markDirty` and a missing-input check in the header**

In `tsd/src/tsd/graph/Graph.hpp`, in the public section of `Graph`, add:
```cpp
  // Mark a node and all transitive downstream consumers Dirty.
  void markDirty(NodeId id);
```
In the private section add:
```cpp
  // After topology changes, set any node missing a required input to Error.
  void revalidateRequiredInputs(NodeId id);
```

- [ ] **Step 4: Implement in the source**

In `tsd/src/tsd/graph/Graph.cpp`, add:
```cpp
void Graph::markDirty(NodeId id)
{
  auto *n = node(id);
  if (!n || n->state == EvalState::Dirty)
    return; // already dirty -> its downstream subtree is already marked
  n->state = EvalState::Dirty;
  n->cache.clear();
  for (const auto &c : m_connections) {
    if (c.fromNode == id)
      markDirty(c.toNode);
  }
}

void Graph::revalidateRequiredInputs(NodeId id)
{
  auto *n = node(id);
  if (!n)
    return;
  auto info = n->impl->typeInfo();
  for (const auto &port : info.inputs) {
    if (port.required && inputConnection(id, port.name) == nullptr) {
      n->state = EvalState::Error;
      n->error = "missing required input: " + port.name.str();
      return;
    }
  }
}
```
Then, in `removeNode`, replace the body's `if (it->fromNode == id) { ... }` dirty-marking with a deferred revalidation: after the connection-erase loop and before `m_nodes.erase(id)`, collect the affected consumer ids, and after erasing the node, call `markDirty` + `revalidateRequiredInputs` on each. Concretely, change `removeNode` to:
```cpp
void Graph::removeNode(NodeId id)
{
  std::vector<NodeId> affectedConsumers;
  for (auto it = m_connections.begin(); it != m_connections.end();) {
    if (it->fromNode == id || it->toNode == id) {
      if (it->fromNode == id)
        affectedConsumers.push_back(it->toNode);
      it = m_connections.erase(it);
    } else {
      ++it;
    }
  }
  m_nodes.erase(id);
  for (NodeId c : affectedConsumers) {
    markDirty(c);
    revalidateRequiredInputs(c);
  }
}
```
Add `#include <vector>` if not already present (it is, via the header).

- [ ] **Step 5: Run the test**

```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::Dirty' --output-on-failure
```
First add the registration in `tsd/tests/CMakeLists.txt` (source list + `add_test(NAME tsd::graph::Dirty COMMAND ${PROJECT_NAME} "[graph-dirty]")`).
Expected: PASS.

- [ ] **Step 6: Re-run the link test to confirm no regression**

```bash
cd build && ctest -C Release -R 'tsd::graph::GraphLinks' --output-on-failure
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Graph.hpp tsd/src/tsd/graph/Graph.cpp tsd/tests/test_graph_Dirty.cpp
jj commit -m "feat(graph): add dirty propagation and deletion revalidation"
```

---

## Task 11: `EvalContext` + `Evaluator` (lazy pull, per-residency cache, version short-circuit)

**Files:**
- Create: `tsd/src/tsd/graph/Evaluator.hpp`
- Create: `tsd/src/tsd/graph/Evaluator.cpp`
- Modify: `tsd/src/tsd/graph/CMakeLists.txt`
- Test: `tsd/tests/test_graph_Evaluator.cpp`

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_Evaluator.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

// Emits param "v" as a host scalar; counts evaluations.
struct ConstSource : Node
{
  ParameterList params;
  int *evalCount;
  explicit ConstSource(int *c) : evalCount(c) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ConstSource");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override;
};

// Reads "in", multiplies by 2, emits "out"; counts evaluations.
struct DoubleNode : Node
{
  ParameterList params;
  int *evalCount;
  explicit DoubleNode(int *c) : evalCount(c) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Double");
    i.inputs.push_back({Token("in"), PortType{Token("scalar")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override;
};

} // namespace

// Out-of-line so EvalContext (from the top include) is fully defined here.
void ConstSource::evaluate(EvalContext &ctx)
{
  (*evalCount)++;
  auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
  Value v;
  v.type = PortType{Token("scalar")};
  v.residency = hostResidency();
  v.payload = out;
  ctx.setOutput(Token("out"), v);
}

void DoubleNode::evaluate(EvalContext &ctx)
{
  (*evalCount)++;
  float in = *std::static_pointer_cast<float>(
      ctx.input(Token("in"), hostResidency()).payload);
  auto out = std::make_shared<float>(in * 2.0f);
  Value v;
  v.type = PortType{Token("scalar")};
  v.residency = hostResidency();
  v.payload = out;
  ctx.setOutput(Token("out"), v);
}

SCENARIO("tsd::graph::Evaluator lazy pull, caching, and version short-circuit",
    "[graph-eval]")
{
  int srcEvals = 0, dblEvals = 0;

  Graph g;
  auto src = g.addNode(std::make_unique<ConstSource>(&srcEvals));
  auto dbl = g.addNode(std::make_unique<DoubleNode>(&dblEvals));
  g.node(src)->impl->parameters().set(Token("v"), 5.0f);
  g.connect(src, Token("out"), dbl, Token("in"));

  Evaluator e(g);

  WHEN("pulling the sink once")
  {
    REQUIRE(e.pull(dbl));
    THEN("the result is 10 and each node evaluated once")
    {
      const Value *out = e.output(dbl, Token("out"), hostResidency());
      REQUIRE(out != nullptr);
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 10.0f);
      REQUIRE(srcEvals == 1);
      REQUIRE(dblEvals == 1);
    }
  }

  WHEN("pulling twice with no edits")
  {
    e.pull(dbl);
    e.pull(dbl);
    THEN("nothing re-evaluates the second time")
    {
      REQUIRE(srcEvals == 1);
      REQUIRE(dblEvals == 1);
    }
  }

  WHEN("a param edit dirties the source, then pulling again")
  {
    e.pull(dbl);
    g.node(src)->impl->parameters().set(Token("v"), 6.0f);
    g.markDirty(src);
    e.pull(dbl);
    THEN("both recompute and the result updates to 12")
    {
      REQUIRE(srcEvals == 2);
      REQUIRE(dblEvals == 2);
      const Value *out = e.output(dbl, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 12.0f);
    }
  }

  WHEN("only the sink's own param changes, then pulling again")
  {
    e.pull(dbl);
    // Changing the sink's parameter hash forces the sink to recompute, but the
    // source's output version is unchanged, so the source must NOT re-evaluate.
    g.node(dbl)->impl->parameters().set(Token("scale"), 1.0f);
    g.markDirty(dbl);
    e.pull(dbl);
    THEN("the sink recomputes but the source is short-circuited")
    {
      REQUIRE(srcEvals == 1);
      REQUIRE(dblEvals == 2);
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — `tsd/graph/Evaluator.hpp` not found.

- [ ] **Step 3: Write the header**

`tsd/src/tsd/graph/Evaluator.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/Graph.hpp"
#include "tsd/graph/TransferRegistry.hpp"
// std
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace tsd::graph {

// Records one implicit op inserted during a pull.
struct EvalReportEntry
{
  enum class Kind
  {
    Transfer,
    Convert,
    Failed
  };
  ConnectionId wire{INVALID_CONNECTION};
  Kind kind{Kind::Transfer};
  tsd::core::Token from;
  tsd::core::Token to;
  size_t estCost{0};
  std::string message;
};

struct EvalReport
{
  std::vector<EvalReportEntry> entries;
  void clear()
  {
    entries.clear();
  }
};

// Synchronous lazy-pull evaluator (Phase 1). Walks inputs depth-first. The
// recompute decision is a pure function of state, not the dirty flag: a node
// recomputes iff it is non-cacheable, has never evaluated, has an empty cache,
// its parameter hash changed, or any input's producer outputVersion differs
// from the version it consumed last time. A producer bumps its outputVersion
// only when it recomputes; downstream consumers therefore skip when an upstream
// value is unchanged (the "version short-circuit"). contentTag-based skip (no
// bump when recompute yields identical content) is a Phase 2 optimization.
class Evaluator
{
 public:
  explicit Evaluator(Graph &g,
      const TransferRegistry *transfers = nullptr,
      const ConversionRegistry *conversions = nullptr);

  // Ensure `id`'s outputs are up to date. Returns false if the node (or an
  // ancestor) is in Error.
  bool pull(NodeId id);

  // Look up a cached output in a desired residency (after a pull).
  const Value *output(
      NodeId id, tsd::core::Token port, const Residency &want) const;

  const EvalReport &lastReport() const
  {
    return m_report;
  }

 private:
  friend class EvalContext;
  bool ensure(NodeId id);
  // Materialize a producer's output as `wantType` in residency `want`, inserting
  // a conversion and/or transfer and recording each in the EvalReport. Transfer
  // results are cached. Returns false on failure (no path / no conversion).
  bool materializeForInput(
      const Connection &c, PortType wantType, const Residency &want, Value &out);

  // Transfer cache key: (producerNodeId, producerVersion, targetBackend,
  // targetDeviceId, targetType). Keying on deviceId prevents a device-0 copy
  // from satisfying a device-1 consumer; keying on producerVersion invalidates
  // stale copies when the producer recomputes.
  using TransferCacheKey =
      std::tuple<uint64_t, uint64_t, const void *, int, const void *>;

  Graph &m_graph;
  const TransferRegistry *m_transfers;
  const ConversionRegistry *m_conversions;
  EvalReport m_report;
  std::map<TransferCacheKey, Value> m_transferCache;
  NodeId m_current{INVALID_NODE}; // node being evaluated (for EvalContext)
};

// Passed to Node::evaluate(). Bridges a node to its (already-evaluated) inputs
// and collects its outputs.
class EvalContext
{
 public:
  EvalContext(Evaluator &e, GraphNode &self) : m_eval(e), m_self(self) {}

  // Returns the input value in the requested residency (transfers inserted by
  // the evaluator beforehand). Returns an invalid Value if unconnected.
  Value input(tsd::core::Token name, const Residency &want);
  bool hasInput(tsd::core::Token name) const;
  Value inputOr(tsd::core::Token name, const Residency &want, Value alt);

  template <typename T>
  T param(tsd::core::Token name) const
  {
    return m_self.impl->parameters().get<T>(name);
  }

  void setOutput(tsd::core::Token name, Value v);

 private:
  Evaluator &m_eval;
  GraphNode &m_self;
};

} // namespace tsd::graph
```

- [ ] **Step 4: Write the source**

`tsd/src/tsd/graph/Evaluator.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/Evaluator.hpp"

namespace tsd::graph {

Evaluator::Evaluator(Graph &g,
    const TransferRegistry *transfers,
    const ConversionRegistry *conversions)
    : m_graph(g), m_transfers(transfers), m_conversions(conversions)
{}

const Value *Evaluator::output(
    NodeId id, tsd::core::Token port, const Residency &want) const
{
  const GraphNode *n = m_graph.node(id);
  if (!n)
    return nullptr;
  auto pit = n->cache.find(port);
  if (pit == n->cache.end())
    return nullptr;
  auto rit = pit->second.find(want);
  if (rit == pit->second.end())
    return nullptr;
  return &rit->second;
}

bool Evaluator::pull(NodeId id)
{
  m_report.clear();
  return ensure(id);
}

bool Evaluator::materializeForInput(
    const Connection &c, PortType wantType, const Residency &want, Value &out)
{
  const GraphNode *producer = m_graph.node(c.fromNode);
  if (!producer)
    return false;

  // Producer's freshly-evaluated output in its native residency.
  auto pit = producer->cache.find(c.fromPort);
  if (pit == producer->cache.end() || pit->second.empty())
    return false;
  Value src = pit->second.begin()->second;

  // 1) Type conversion if needed.
  if (src.type != wantType) {
    const ConversionEntry *ce =
        m_conversions ? m_conversions->find(src.type, wantType) : nullptr;
    if (!ce) {
      m_report.entries.push_back({c.id,
          EvalReportEntry::Kind::Failed,
          src.type.name,
          wantType.name,
          0,
          "no registered conversion"});
      return false;
    }
    size_t cost = ce->estimateElements(src);
    src = ce->fn(src);
    m_report.entries.push_back({c.id,
        EvalReportEntry::Kind::Convert,
        ce->from.name,
        ce->to.name,
        cost,
        ""});
  }

  // 2) Residency transfer to the requested residency (incl. deviceId).
  if (!(src.residency == want)) {
    const TransferCacheKey key{producer->id,
        producer->outputVersion,
        want.backend.value(),
        want.deviceId,
        wantType.name.value()};
    auto cached = m_transferCache.find(key);
    if (cached != m_transferCache.end()) {
      out = cached->second; // cache hit: no new transfer, no report entry
      return true;
    }
    const TransferEntry *te = m_transfers
        ? m_transfers->find(src.type, src.residency.backend, want.backend)
        : nullptr;
    if (!te) {
      m_report.entries.push_back({c.id,
          EvalReportEntry::Kind::Failed,
          src.residency.backend,
          want.backend,
          0,
          "no registered transfer"});
      return false;
    }
    size_t cost = te->estimateBytes(src);
    src = te->fn(src, want); // produce at the full target residency
    m_report.entries.push_back(
        {c.id, EvalReportEntry::Kind::Transfer, te->from, te->to, cost, ""});
    m_transferCache[key] = src;
  }

  out = src;
  return true;
}

bool Evaluator::ensure(NodeId id)
{
  GraphNode *n = m_graph.node(id);
  if (!n)
    return false;
  if (n->state == EvalState::Error)
    return false;

  // Ensure all producers are current first, and detect whether any input's
  // producer version changed since we last consumed it (or is newly connected).
  bool inputsChanged = false;
  for (const auto &c : m_graph.connections()) {
    if (c.toNode != id)
      continue;
    if (!ensure(c.fromNode))
      return false;
    const GraphNode *producer = m_graph.node(c.fromNode);
    uint64_t pv = producer ? producer->outputVersion : 0;
    auto it = n->consumedInputVersions.find(c.toPort);
    if (it == n->consumedInputVersions.end() || it->second != pv)
      inputsChanged = true;
  }

  // Recompute decision is a pure function of state, not the dirty flag.
  const bool cacheable = n->impl->typeInfo().isCacheable;
  const uint64_t paramHash = n->impl->parameters().hash();
  const bool recompute = !cacheable || !n->hasEvaluated || n->cache.empty()
      || paramHash != n->lastParamHash || inputsChanged;

  if (!recompute) {
    n->state = EvalState::Clean;
    return true;
  }

  // Run the node.
  n->state = EvalState::Computing;
  n->cache.clear();
  NodeId prev = m_current;
  m_current = id;
  EvalContext ctx(*this, *n);
  n->impl->evaluate(ctx);
  m_current = prev;

  if (n->state == EvalState::Error)
    return false;

  // Record the versions we consumed, then bump our own output version.
  n->consumedInputVersions.clear();
  for (const auto &c : m_graph.connections()) {
    if (c.toNode != id)
      continue;
    const GraphNode *producer = m_graph.node(c.fromNode);
    n->consumedInputVersions[c.toPort] = producer ? producer->outputVersion : 0;
  }
  n->lastParamHash = paramHash;
  n->hasEvaluated = true;
  n->outputVersion++;

  // Stamp freshly-produced outputs with the new version.
  for (auto &outPort : n->cache)
    for (auto &resVal : outPort.second)
      resVal.second.version = n->outputVersion;

  n->state = EvalState::Clean;
  return true;
}

// ---- EvalContext --------------------------------------------------------

bool EvalContext::hasInput(tsd::core::Token name) const
{
  return m_eval.m_graph.inputConnection(m_self.id, name) != nullptr;
}

Value EvalContext::input(tsd::core::Token name, const Residency &want)
{
  const Connection *c = m_eval.m_graph.inputConnection(m_self.id, name);
  if (!c)
    return Value{};
  // The consumer port's declared type is the type we want the input delivered
  // as; `want` is the residency the node is asking for. PortSpec.acceptedBackends
  // is declarative metadata in Phase 1 (the node drives the target via `want`).
  auto info = m_self.impl->typeInfo();
  PortType wantType;
  for (const auto &p : info.inputs)
    if (p.name == name)
      wantType = p.type;
  Value out;
  if (!m_eval.materializeForInput(*c, wantType, want, out)) {
    m_self.state = EvalState::Error;
    m_self.error = "failed to materialize input: " + name.str();
    return Value{};
  }
  return out;
}

Value EvalContext::inputOr(tsd::core::Token name, const Residency &want, Value alt)
{
  if (!hasInput(name))
    return alt;
  return input(name, want);
}

void EvalContext::setOutput(tsd::core::Token name, Value v)
{
  v.producerNodeId = m_self.id;
  // `version` is finalized by Evaluator::ensure() after evaluate() returns.
  m_self.cache[name][v.residency] = v;
}

} // namespace tsd::graph
```

- [ ] **Step 5: Add source to lib CMake**

Add `Evaluator.cpp` to `tsd/src/tsd/graph/CMakeLists.txt`.

- [ ] **Step 6: Register test and run**

Add `test_graph_Evaluator.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::Evaluator COMMAND ${PROJECT_NAME} "[graph-eval]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::Evaluator' --output-on-failure
```
Expected: PASS — result 10, then no recompute, then 12 after edit.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/Evaluator.hpp tsd/src/tsd/graph/Evaluator.cpp tsd/tests/test_graph_Evaluator.cpp
jj commit -m "feat(graph): add Evaluator with lazy pull and per-residency cache"
```

---

## Task 12: Fake `"test"` backend + implicit transfer end-to-end + EvalReport

**Files:**
- Create: `tsd/src/tsd/graph/TestBackend.hpp`
- Create: `tsd/src/tsd/graph/TestBackend.cpp`
- Modify: `tsd/src/tsd/graph/CMakeLists.txt`
- Test: `tsd/tests/test_graph_TestBackendTransfer.cpp`

The `"test"` backend models a non-host residency without CUDA: its payload is a
`std::vector<float>` tagged as living on backend `"test"`, device N. It registers
host→test and test→host transfers so the evaluator's implicit-transfer path and
multi-device keying can be tested in CI.

- [ ] **Step 1: Write the failing test**

`tsd/tests/test_graph_TestBackendTransfer.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph/TestBackend.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::TransferRegistry;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

// Produces a "test"-resident float buffer.
struct TestResidentSource : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("TestSource");
    i.outputs.push_back({Token("out"), PortType{Token("array")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto buf = std::make_shared<std::vector<float>>(std::vector<float>{1, 2, 3, 4});
    Value v;
    v.type = PortType{Token("array")};
    v.residency = Residency{Token("test"), 0};
    v.payload = buf;
    ctx.setOutput(Token("out"), v);
  }
};

// Requires its input on host only.
struct HostOnlySink : Node
{
  ParameterList params;
  float sum{0};
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("HostSink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("array")}, true, {Token("host")}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto v = ctx.input(Token("in"), hostResidency());
    auto b = std::static_pointer_cast<std::vector<float>>(v.payload);
    sum = 0;
    for (float x : *b)
      sum += x;
  }
};

} // namespace

SCENARIO("tsd::graph implicit test->host transfer is inserted and reported",
    "[graph-testbackend]")
{
  TransferRegistry transfers;
  tsd::graph::registerTestBackendTransfers(transfers);

  Graph g;
  auto src = g.addNode(std::make_unique<TestResidentSource>());
  auto sinkId = g.addNode(std::make_unique<HostOnlySink>());
  auto *sinkNode = static_cast<HostOnlySink *>(g.node(sinkId)->impl.get());
  g.connect(src, Token("out"), sinkId, Token("in"));

  Evaluator e(g, &transfers, nullptr);

  WHEN("pulling the host-only sink")
  {
    REQUIRE(e.pull(sinkId));
    THEN("the data was transferred to host and summed")
    {
      REQUIRE(sinkNode->sum == 10.0f);
    }
    THEN("the EvalReport records one test->host transfer with nonzero cost")
    {
      const auto &r = e.lastReport();
      REQUIRE(r.entries.size() == 1);
      REQUIRE(r.entries[0].kind
          == tsd::graph::EvalReportEntry::Kind::Transfer);
      REQUIRE(r.entries[0].from == Token("test"));
      REQUIRE(r.entries[0].to == Token("host"));
      REQUIRE(r.entries[0].estCost == 4 * sizeof(float));
    }
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target tsdTests --parallel
```
Expected: FAIL to compile — `tsd/graph/TestBackend.hpp` not found.

- [ ] **Step 3: Write the header**

`tsd/src/tsd/graph/TestBackend.hpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/TransferRegistry.hpp"

namespace tsd::graph {

// Registers host<->test transfers for the "array" PortType. The "test" backend
// uses the same std::vector<float> payload as host; transfers only retag
// residency (and, for realism, copy the buffer). Used by CI to exercise the
// residency machinery without CUDA.
void registerTestBackendTransfers(TransferRegistry &reg);

} // namespace tsd::graph
```

- [ ] **Step 4: Write the source**

`tsd/src/tsd/graph/TestBackend.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/TestBackend.hpp"
// std
#include <memory>
#include <vector>

namespace tsd::graph {

namespace {

size_t floatBufferBytes(const Value &v)
{
  auto b = std::static_pointer_cast<std::vector<float>>(v.payload);
  return b ? b->size() * sizeof(float) : 0;
}

Value copyRetag(const Value &src, const Residency &target)
{
  auto in = std::static_pointer_cast<std::vector<float>>(src.payload);
  auto out = std::make_shared<std::vector<float>>(*in); // real copy
  Value v = src;
  v.payload = out;
  v.residency = target;
  return v;
}

} // namespace

void registerTestBackendTransfers(TransferRegistry &reg)
{
  PortType arrayT{tsd::core::Token("array")};
  reg.registerTransfer(arrayT,
      tsd::core::Token("host"),
      tsd::core::Token("test"),
      copyRetag,
      floatBufferBytes);
  reg.registerTransfer(arrayT,
      tsd::core::Token("test"),
      tsd::core::Token("host"),
      copyRetag,
      floatBufferBytes);
}

} // namespace tsd::graph
```

- [ ] **Step 5: Add source to lib CMake**

Add `TestBackend.cpp` to `tsd/src/tsd/graph/CMakeLists.txt`.

- [ ] **Step 6: Register test and run**

Add `test_graph_TestBackendTransfer.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::TestBackend COMMAND ${PROJECT_NAME} "[graph-testbackend]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::TestBackend' --output-on-failure
```
Expected: PASS — sum 10, one transfer entry, cost `4*sizeof(float)`.

- [ ] **Step 7: Commit**

```bash
clang-format -i tsd/src/tsd/graph/TestBackend.hpp tsd/src/tsd/graph/TestBackend.cpp tsd/tests/test_graph_TestBackendTransfer.cpp
jj commit -m "feat(graph): add test backend and verify implicit transfer + EvalReport"
```

---

## Task 12b: Implicit conversion insertion + failed-transfer reporting

Covers the two `materializeForInput` branches not exercised by Task 12: an
implicit type conversion inserted at runtime (reported as `Convert`), and a
residency with no registered transfer (reported as `Failed`, node → `Error`,
pull returns false — not a crash). Note: a type mismatch is only reachable at
runtime when a conversion IS registered (link validation rejects unconvertible
type mismatches), so the `Convert`-failure branch is defensive and not tested
here.

**Files:**
- Test: `tsd/tests/test_graph_ImplicitOps.cpp`

- [ ] **Step 1: Write the test**

`tsd/tests/test_graph_ImplicitOps.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::ConversionRegistry;
using tsd::graph::EvalContext;
using tsd::graph::EvalReportEntry;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::TransferRegistry;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

struct I32Source : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("I32Source");
    i.outputs.push_back({Token("out"), PortType{Token("i32array")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto buf = std::make_shared<std::vector<int>>(std::vector<int>{1, 2, 3});
    Value v;
    v.type = PortType{Token("i32array")};
    v.residency = hostResidency();
    v.payload = buf;
    ctx.setOutput(Token("out"), v);
  }
};

struct F32Sink : Node
{
  ParameterList params;
  float sum{0};
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("F32Sink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("f32array")}, true, {Token("host")}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto v = ctx.input(Token("in"), hostResidency());
    auto b = std::static_pointer_cast<std::vector<float>>(v.payload);
    sum = 0;
    for (float x : *b)
      sum += x;
  }
};

// Requires an input on a backend for which no transfer is registered.
struct VulkanSink : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("VulkanSink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("i32array")}, true, {Token("vulkan")}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    ctx.input(Token("in"), Residency{Token("vulkan"), 0});
  }
};

ConversionRegistry makeI32ToF32()
{
  ConversionRegistry reg;
  reg.registerConversion(PortType{Token("i32array")},
      PortType{Token("f32array")},
      [](const Value &src) {
        auto in = std::static_pointer_cast<std::vector<int>>(src.payload);
        auto out = std::make_shared<std::vector<float>>();
        for (int x : *in)
          out->push_back(static_cast<float>(x));
        Value v = src;
        v.type = PortType{Token("f32array")};
        v.payload = out;
        return v;
      },
      [](const Value &src) -> size_t {
        return std::static_pointer_cast<std::vector<int>>(src.payload)->size();
      });
  return reg;
}

} // namespace

SCENARIO("tsd::graph inserts and reports an implicit conversion",
    "[graph-implicitops]")
{
  ConversionRegistry conversions = makeI32ToF32();
  Graph g(&conversions);
  auto src = g.addNode(std::make_unique<I32Source>());
  auto sinkId = g.addNode(std::make_unique<F32Sink>());
  auto *sink = static_cast<F32Sink *>(g.node(sinkId)->impl.get());

  WHEN("linking i32 source to an f32 sink (conversion registered)")
  {
    auto r = g.connect(src, Token("out"), sinkId, Token("in"));
    REQUIRE(r.ok); // link allowed because a conversion exists

    Evaluator e(g, nullptr, &conversions);
    REQUIRE(e.pull(sinkId));
    THEN("the data is converted to float and summed")
    {
      REQUIRE(sink->sum == 6.0f);
    }
    THEN("the EvalReport records one i32array->f32array conversion")
    {
      const auto &rep = e.lastReport();
      REQUIRE(rep.entries.size() == 1);
      REQUIRE(rep.entries[0].kind == EvalReportEntry::Kind::Convert);
      REQUIRE(rep.entries[0].from == Token("i32array"));
      REQUIRE(rep.entries[0].to == Token("f32array"));
      REQUIRE(rep.entries[0].estCost == 3);
    }
  }
}

SCENARIO("tsd::graph reports a missing transfer path as a failed op, not a crash",
    "[graph-implicitops]")
{
  TransferRegistry transfers; // empty: no host->vulkan transfer
  Graph g;
  auto src = g.addNode(std::make_unique<I32Source>());
  auto sinkId = g.addNode(std::make_unique<VulkanSink>());
  g.connect(src, Token("out"), sinkId, Token("in")); // same type, link ok

  Evaluator e(g, &transfers, nullptr);

  WHEN("pulling a sink whose backend has no registered transfer")
  {
    bool ok = e.pull(sinkId);
    THEN("the pull fails and the node is in Error")
    {
      REQUIRE_FALSE(ok);
      REQUIRE(g.node(sinkId)->state == tsd::graph::EvalState::Error);
    }
    THEN("the EvalReport records one failed host->vulkan op")
    {
      const auto &rep = e.lastReport();
      REQUIRE(rep.entries.size() == 1);
      REQUIRE(rep.entries[0].kind == EvalReportEntry::Kind::Failed);
      REQUIRE(rep.entries[0].from == Token("host"));
      REQUIRE(rep.entries[0].to == Token("vulkan"));
    }
  }
}
```

- [ ] **Step 2: Register test and run**

Add `test_graph_ImplicitOps.cpp` to the test executable sources, then:
```cmake
add_test(NAME tsd::graph::ImplicitOps COMMAND ${PROJECT_NAME} "[graph-implicitops]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::ImplicitOps' --output-on-failure
```
Expected: PASS — conversion summed to 6 with one `Convert` report entry; missing
transfer yields a failed pull, `Error` node, and one `Failed` report entry.

> **Implementation note:** `EvalContext::input()` sets the node's state to `Error`
> when `materializeForInput` fails. Confirm `Evaluator::ensure()` returns false
> when `n->state == EvalState::Error` after `evaluate()` (it does, per Task 11),
> so the failed pull propagates without throwing.

- [ ] **Step 3: Commit**

```bash
clang-format -i tsd/tests/test_graph_ImplicitOps.cpp
jj commit -m "test(graph): cover implicit conversion and failed-transfer reporting"
```

---

## Task 12c: deviceId-keyed transfer cache (multi-device, no CUDA)

Verifies the transfer cache implemented in Task 11: it is keyed on the target
`(backend, deviceId)` so a device-0 copy never satisfies a device-1 consumer,
and a repeat request for the same `(producer, version, target)` is served from
cache without re-running the transfer. Uses the fake `"test"` backend with two
"devices" — no CUDA required.

**Files:**
- Test: `tsd/tests/test_graph_DeviceKeying.cpp`

- [ ] **Step 1: Write the test**

`tsd/tests/test_graph_DeviceKeying.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <memory>
#include <vector>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;
using tsd::graph::Residency;
using tsd::graph::TransferRegistry;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

struct HostArraySource : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("HostArraySource");
    i.outputs.push_back({Token("out"), PortType{Token("array")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto buf = std::make_shared<std::vector<float>>(std::vector<float>{1, 2});
    Value v;
    v.type = PortType{Token("array")};
    v.residency = hostResidency();
    v.payload = buf;
    ctx.setOutput(Token("out"), v);
  }
};

// Requests its input on the "test" backend at a specific device id.
struct TestDeviceSink : Node
{
  ParameterList params;
  int device{0};
  Residency got;
  explicit TestDeviceSink(int d) : device(d) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("TestDeviceSink");
    i.inputs.push_back(
        {Token("in"), PortType{Token("array")}, true, {Token("test")}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto v = ctx.input(Token("in"), Residency{Token("test"), device});
    got = v.residency;
  }
};

} // namespace

SCENARIO("tsd::graph transfer cache is keyed on target deviceId",
    "[graph-devicekeying]")
{
  int transferCount = 0;
  TransferRegistry transfers;
  transfers.registerTransfer(PortType{Token("array")},
      Token("host"),
      Token("test"),
      [&transferCount](const Value &src, const Residency &target) {
        ++transferCount;
        auto in = std::static_pointer_cast<std::vector<float>>(src.payload);
        auto out = std::make_shared<std::vector<float>>(*in);
        Value v = src;
        v.payload = out;
        v.residency = target;
        return v;
      },
      [](const Value &src) -> size_t {
        return std::static_pointer_cast<std::vector<float>>(src.payload)->size()
            * sizeof(float);
      });

  Graph g;
  auto src = g.addNode(std::make_unique<HostArraySource>());
  auto s0Id = g.addNode(std::make_unique<TestDeviceSink>(0));
  auto s1Id = g.addNode(std::make_unique<TestDeviceSink>(1));
  auto *s0 = static_cast<TestDeviceSink *>(g.node(s0Id)->impl.get());
  auto *s1 = static_cast<TestDeviceSink *>(g.node(s1Id)->impl.get());
  g.connect(src, Token("out"), s0Id, Token("in"));
  g.connect(src, Token("out"), s1Id, Token("in"));

  Evaluator e(g, &transfers, nullptr);

  WHEN("two consumers request the same data on device 0 and device 1")
  {
    REQUIRE(e.pull(s0Id));
    REQUIRE(e.pull(s1Id));
    THEN("each gets its own device-resident copy (no cross-device collision)")
    {
      REQUIRE(s0->got == Residency{Token("test"), 0});
      REQUIRE(s1->got == Residency{Token("test"), 1});
    }
    THEN("two distinct transfers ran (device 1 did not reuse device 0's copy)")
    {
      REQUIRE(transferCount == 2);
    }
  }

  WHEN("the same device-0 consumer is recomputed against an unchanged producer")
  {
    REQUIRE(e.pull(s0Id)); // transferCount == 1
    g.markDirty(s0Id);     // force the sink to re-evaluate
    REQUIRE(e.pull(s0Id));
    THEN("the transfer is served from cache, not re-run")
    {
      REQUIRE(transferCount == 1);
    }
  }
}
```

- [ ] **Step 2: Register test and run**

Add `test_graph_DeviceKeying.cpp` to the test executable sources, then:
```cmake
add_test(NAME tsd::graph::DeviceKeying COMMAND ${PROJECT_NAME} "[graph-devicekeying]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::DeviceKeying' --output-on-failure
```
Expected: PASS — device 0 and device 1 each get their own copy (2 transfers); a
recompute against an unchanged producer reuses the cached transfer (still 1).

- [ ] **Step 3: Commit**

```bash
clang-format -i tsd/tests/test_graph_DeviceKeying.cpp
jj commit -m "test(graph): verify deviceId-keyed transfer cache (multi-device)"
```

---

## Task 13: `isCacheable=false` always-recompute behavior

**Files:**
- Test only: `tsd/tests/test_graph_NonCacheable.cpp`

This is a **characterization test** (not test-first): it locks in the
`recompute = ... || !cacheable` branch already implemented in Task 11 — a node
with `isCacheable=false` re-evaluates on every pull even when clean and
unchanged. It passes immediately; its value is regression protection.

- [ ] **Step 1: Write the characterization test**

`tsd/tests/test_graph_NonCacheable.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortSpec;
using tsd::graph::PortType;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

struct Counter : Node
{
  ParameterList params;
  int *count;
  bool cacheable;
  Counter(int *c, bool cache) : count(c), cacheable(cache) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Counter");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    i.isCacheable = cacheable;
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    (*count)++;
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = std::make_shared<float>(1.0f);
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph non-cacheable nodes always recompute", "[graph-noncache]")
{
  GIVEN("a cacheable node")
  {
    int n = 0;
    Graph g;
    auto id = g.addNode(std::make_unique<Counter>(&n, true));
    Evaluator e(g);
    e.pull(id);
    e.pull(id);
    THEN("it evaluates only once across two pulls")
    {
      REQUIRE(n == 1);
    }
  }

  GIVEN("a non-cacheable node")
  {
    int n = 0;
    Graph g;
    auto id = g.addNode(std::make_unique<Counter>(&n, false));
    Evaluator e(g);
    e.pull(id);
    e.pull(id);
    THEN("it evaluates on every pull")
    {
      REQUIRE(n == 2);
    }
  }
}
```

- [ ] **Step 2: Register test and run to verify it passes**

Add `test_graph_NonCacheable.cpp` to test sources, then:
```cmake
add_test(NAME tsd::graph::NonCacheable COMMAND ${PROJECT_NAME} "[graph-noncache]")
```
Run:
```bash
cmake --build build --target tsdTests --parallel
cd build && ctest -C Release -R 'tsd::graph::NonCacheable' --output-on-failure
```
Expected: PASS. (If the non-cacheable case reports `n == 1`, the `!cacheable`
branch in `Evaluator::ensure` is missing — fix Task 11's `recompute` expression.)

- [ ] **Step 3: Run the entire graph test suite**

```bash
cd build && ctest -C Release -R 'tsd::graph' --output-on-failure
```
Expected: all `tsd::graph::*` tests PASS.

- [ ] **Step 4: Commit**

```bash
clang-format -i tsd/tests/test_graph_NonCacheable.cpp
jj commit -m "test(graph): verify non-cacheable nodes always recompute"
```

---

## Phase 1 completion checklist

After Task 13, confirm against the spec's Phase 1 scope:

- [ ] `tsd_graph`'s lib CMake links `tsd_core` only — no scene/io/rendering/UI libs (ANARI headers arrive transitively via `tsd_core`'s `Any`; that is expected. The `tsdTests` binary still links the existing scene/rendering stack — that is the test harness, not `tsd_graph`.)
- [ ] PortType/Residency/Value (version stamps) implemented and tested
- [ ] PortTypeRegistry / TransferRegistry / ConversionRegistry implemented and tested
- [ ] Port/Node/Registry incl. `isCacheable`, per-port `acceptedBackends`, ParameterList + hash
- [ ] Graph: stable-id connections, link validation (type compat via exact match or conversion), cycle rejection, dirty propagation, deletion → missing-required-input Error
- [ ] Evaluator: lazy pull, per-output/per-residency cache, version short-circuit, implicit transfer/conversion insertion, EvalReport
- [ ] Transfer cache keyed on `(producer, version, target backend, target deviceId, type)`; multi-device test green (Task 12c)
- [ ] Implicit conversion insertion and missing-transfer-path failure both tested (Task 12b)
- [ ] Node self-registration via `TSD_GRAPH_REGISTER_NODE` works at static init (Task 8)
- [ ] `"test"` fake backend exercises residency (incl. multi-device) without CUDA
- [ ] `ctest -C Release -R 'tsd::graph' --output-on-failure` is green

**Deferred to later phases (do NOT build here):** async scheduling/cancellation (Phase 2), cache eviction/budget (Phase 2), bridge/viewports + RenderIndex (Phase 3), real node catalog + CUDA `TransferRegistry` + UI (Phase 4), Lua + persistence (Phase 5). The residency- and deviceId-keyed caches are already in place, so Phase 3 only swaps the fake backend for real CUDA transfers.

---

## Notes carried forward to Phase 2+

- **Concurrency:** Phase 1 `ensure()` recurses synchronously. Phase 2 replaces the
  driver with a single worker thread + snapshot isolation; the per-node `evaluate`
  contract is unchanged.
- **Version stamping:** `setOutput` only stores the Value; `ensure()` records
  consumed input versions, bumps the node's `outputVersion`, and stamps every
  cached output with it after `evaluate()` returns. Phase 2 must preserve this
  post-evaluate finalize step when publishing from the worker.
- **contentTag optimization (Phase 2):** today `outputVersion` bumps on every
  recompute. When a node sets a `contentTag` equal to the previous round's, the
  bump can be skipped so downstream truly short-circuits even when an upstream
  recomputes to identical content.
- **Param binding:** `graph::ParameterList` is core-only by design (see Task 6
  note). Phase 4 bridges it to the existing `tsd::Parameter` UI rather than
  replacing it.
- **Cache key width:** the output cache keys on full `Residency` and the
  transfer cache on `(producerNodeId, producerVersion, targetBackend,
  targetDeviceId, targetType)` — implemented and tested in Phase 1 (Task 12c)
  so multi-device consumers work without a redesign. Phase 3 swaps the fake
  `"test"` backend for real CUDA transfers; the keying is unchanged.
- **Transfer-cache eviction (deferred):** the transfer cache never evicts —
  stale entries from old producer versions linger until process exit. Acceptable
  for Phase 1 (cache eviction is an explicit Phase 2+ item); a version-aware
  prune belongs with the broader cache-budget work.
