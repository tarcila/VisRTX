# tsdFlow Phase 2 — Async Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the `tsd_graph` `Evaluator` asynchronous — `pullAsync` runs the `ensure()` DFS on a single background `tsd::core::TaskQueue` worker, with a `PullHandle`/poll API, edit-cancels-in-flight cancellation (atomic cancel + epoch supersession, no snapshot), cooperative `EvalContext::cancelled()`, completion-gated results, and error isolation. The blocking `pull()` is retained so all 16 Phase 1 tests pass unchanged. Still headless (no UI/ANARI/CUDA).

**Architecture:** One worker, one task per pull (the whole DFS). A new `pullAsync` bumps `m_epoch`, sets atomic `m_cancel`, and enqueues a task; the running task checks `m_cancel`/epoch between nodes and bails. FIFO single-worker + cancel-before-enqueue means only one thread ever touches node cache/state, so no snapshot is needed; the main thread reads results only after `isReady`.

**Tech Stack:** C++17, `tsd::core::TaskQueue` (header-only, in `tsd_core`), `std::atomic`, Catch2 BDD, CTest, jj.

**Spec:** `docs/superpowers/specs/2026-06-17-tsdflow-phase2-async-evaluator-design.md`.

---

## Conventions (every task)

- Version control is **jj**, not git. A "commit" runs `jj commit <explicit paths> -m "..."` — **never** a bare `jj commit` (an unrelated `.envrc` must stay uncommitted).
- Build tree is the pre-configured Ninja Multi-Config `_out/_cmake` (RelWithDebInfo). Do **not** create a `build/` dir.
  - Build: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests`
  - Test: `ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '<name>' --output-on-failure`
- File header: `// Copyright 2026 NVIDIA Corporation` / `// SPDX-License-Identifier: Apache-2.0`; `#pragma once` for headers. `clang-format -i` touched files before committing. Namespace `tsd::graph`.
- Register a test: add its source to `project_add_executable(...)` in `tsd/tests/CMakeLists.txt` **and** add an `add_test(NAME ... COMMAND ${PROJECT_NAME} "[tag]")` line.

## File structure

| File | Change |
|------|--------|
| `tsd/src/tsd/graph/Evaluator.hpp` | Modify: add `PullHandle`, async API, atomics, `TaskQueue` member, `EvalContext::cancelled()`; `ensure` gains an `epoch` param |
| `tsd/src/tsd/graph/Evaluator.cpp` | Modify: async impl, cancellation checks in `ensure`, retained blocking `pull()` |
| `tsd/tests/test_graph_AsyncEval.cpp` | New: async completion + poll + callback |
| `tsd/tests/test_graph_AsyncCancel.cpp` | New: cooperative cancellation via `cancel()` |
| `tsd/tests/test_graph_AsyncSupersede.cpp` | New: edit-cancels-in-flight / supersession |
| `tsd/tests/test_graph_AsyncError.cpp` | New: error isolation across async |

`tsd_graph` already links `tsd_core`, which provides `tsd::core::TaskQueue` — no new link deps.

---

## Task 1: Async core in `Evaluator` (+ async-completion test)

**Files:**
- Modify: `tsd/src/tsd/graph/Evaluator.hpp`, `tsd/src/tsd/graph/Evaluator.cpp`
- Test: `tsd/tests/test_graph_AsyncEval.cpp`

- [ ] **Step 1: Write the failing test** `tsd/tests/test_graph_AsyncEval.cpp`:

```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <atomic>
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::PullHandle;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

struct ConstSource : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ConstSource");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator runs pullAsync on a worker", "[graph-async]")
{
  Graph g;
  auto src = g.addNode(std::make_unique<ConstSource>());
  g.node(src)->impl->parameters().set(Token("v"), 5.0f);

  Evaluator e(g);

  WHEN("pulling asynchronously and polling to completion")
  {
    std::atomic<int> cbCount{0};
    std::atomic<bool> cbOk{false};
    PullHandle h =
        e.pullAsync(src, [&](bool ok) { cbCount++; cbOk = ok; });

    // Busy-poll on the main thread until the worker finishes (bounded).
    for (int i = 0; i < 100000 && !e.isReady(h); ++i) {
    }

    THEN("the pull is ready, succeeded, and produced the value")
    {
      REQUIRE(e.isReady(h));
      REQUIRE(e.result(h));
      const Value *out = e.output(src, Token("out"), hostResidency());
      REQUIRE(out != nullptr);
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 5.0f);
    }
    THEN("the completion callback fired exactly once with success")
    {
      e.waitIdle();
      REQUIRE(cbCount.load() == 1);
      REQUIRE(cbOk.load());
    }
  }

  WHEN("using the blocking pull() (Phase 1 compatibility)")
  {
    THEN("it returns true and the value is available immediately after")
    {
      REQUIRE(e.pull(src));
      const Value *out = e.output(src, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 5.0f);
    }
  }
}
```
Register `add_test(NAME tsd::graph::Async COMMAND ${PROJECT_NAME} "[graph-async]")`.

- [ ] **Step 2: Run, confirm FAIL** — `pullAsync`/`isReady`/`result`/`PullHandle` undeclared.

```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
```

- [ ] **Step 3: Edit `tsd/src/tsd/graph/Evaluator.hpp`.**

Add includes near the top (after the existing ones):
```cpp
#include "tsd/core/TaskQueue.hpp"
// std
#include <atomic>
#include <cstdint>
#include <functional>
```
(keep the existing `<map> <string> <tuple> <vector>` includes.)

Add this struct + threading-contract comment just before `class Evaluator`:
```cpp
// Opaque ticket identifying one pullAsync request.
struct PullHandle
{
  uint64_t id{0};
};

// THREADING CONTRACT (Phase 2):
//  - pullAsync/pull/waitIdle/result/lastReport/output are called only from a
//    single owner thread (the thread that drives the evaluator). They are NOT
//    internally synchronized against each other.
//  - cancel() is the only method safe to call from another thread (it touches
//    one atomic).
//  - Results: after waitIdle() (or after isReady(h) is true for the LATEST
//    handle h), the owner thread may read output()/lastReport(). result(h) is
//    meaningful only for the latest handle; an older handle reports false once
//    superseded (single shared completion scalar, by design).
//  - The completion atomics are seq-cst on purpose: a seq-cst load of
//    m_doneEpoch observing >= h.id publishes all of the worker's prior
//    (sequenced-before) writes to node cache/state. Do NOT weaken them to
//    relaxed — that would turn the isReady()-then-output() handshake into a
//    data race.
//  - onComplete fires on the worker thread and MUST NOT re-enter the evaluator
//    (no pullAsync/pull/cancel/waitIdle/graph mutation from within it).
//  - pullAsync may block briefly if the worker queue (capacity 8) is full;
//    under single-owner one-in-flight use this never happens.
```

Replace the `Evaluator` class's public section so it reads:
```cpp
 public:
  explicit Evaluator(Graph &g,
      const TransferRegistry *transfers = nullptr,
      const ConversionRegistry *conversions = nullptr);
  ~Evaluator();

  // Schedule an evaluation of `id` on the worker. Cancels any in-flight pull
  // first. Returns a handle to poll. `onComplete(success)` fires from the
  // worker when the task finishes (success is false if superseded or failed).
  PullHandle pullAsync(NodeId id, std::function<void(bool)> onComplete = {});

  // True once the task for `h` has finished (or been superseded by a newer pull).
  bool isReady(PullHandle h) const;
  // True iff `h` is the latest pull and it completed successfully.
  bool result(PullHandle h) const;

  // Request cancellation of any in-flight pull (cooperative).
  void cancel();
  // Block until the worker is idle (safe to mutate the Graph or destroy).
  void waitIdle();

  // Blocking convenience (Phase 1 API): pullAsync + waitIdle + result.
  bool pull(NodeId id);

  const Value *output(
      NodeId id, tsd::core::Token port, const Residency &want) const;

  const EvalReport &lastReport() const
  {
    return m_report;
  }

  // True while a cancellation is requested; polled by EvalContext::cancelled().
  bool cancelRequested() const
  {
    return m_cancel.load();
  }
```

Change the private `ensure` declaration to take an epoch:
```cpp
  bool ensure(NodeId id, uint64_t epoch);
```

Replace the private data members block with (note `m_worker` is declared **last**
so it destructs **first**, joining the worker before the other members vanish):
```cpp
  Graph &m_graph;
  const TransferRegistry *m_transfers;
  const ConversionRegistry *m_conversions;
  EvalReport m_report;
  std::map<TransferCacheKey, Value> m_transferCache;
  NodeId m_current{INVALID_NODE}; // node being evaluated (for EvalContext)

  std::atomic<uint64_t> m_epoch{0};     // bumped per pullAsync
  std::atomic<uint64_t> m_doneEpoch{0}; // highest epoch whose task has finished
  std::atomic<bool> m_doneOk{false};    // success of the most-recent finished task
  std::atomic<bool> m_cancel{false};    // cooperative cancel flag
  tsd::core::Future m_lastFuture;       // future of the most-recently enqueued task
  tsd::core::TaskQueue m_worker{8};     // MUST be declared last (see above)
```

In `class EvalContext`'s public section, add after `inputOr(...)`:
```cpp
  // True if a cancellation has been requested; long evaluate() loops should
  // poll this and return early (leaving no finalized output).
  bool cancelled() const;
```

- [ ] **Step 4: Edit `tsd/src/tsd/graph/Evaluator.cpp`.**

Add `#include <utility>` under the existing include. Then make these changes.

**(a)** After the constructor definition, add a destructor:
```cpp
Evaluator::~Evaluator()
{
  cancel();
  waitIdle();
}
```

**(b)** Replace `bool Evaluator::pull(NodeId id)` entirely with the async machinery:
```cpp
PullHandle Evaluator::pullAsync(NodeId id, std::function<void(bool)> onComplete)
{
  const uint64_t e = ++m_epoch;
  m_cancel.store(true); // ask any running task to bail; the new task resets it
  m_lastFuture = m_worker.enqueue([this, id, e, onComplete]() {
    // This task starts only after any prior task has fully returned (FIFO
    // single worker), so resetting the cancel flag here is safe.
    m_cancel.store(false);
    m_report.clear();
    const bool ok = ensure(id, e);
    const bool effective = ok && (e == m_epoch.load());
    m_doneOk.store(effective);
    m_doneEpoch.store(e);
    if (onComplete)
      onComplete(effective);
  });
  return PullHandle{e};
}

bool Evaluator::isReady(PullHandle h) const
{
  // Tasks run FIFO, so m_doneEpoch increases monotonically; h is done once a
  // task with an epoch >= h.id has finished.
  return m_doneEpoch.load() >= h.id;
}

bool Evaluator::result(PullHandle h) const
{
  return isReady(h) && h.id == m_epoch.load() && m_doneOk.load();
}

void Evaluator::cancel()
{
  m_cancel.store(true);
}

void Evaluator::waitIdle()
{
  tsd::core::wait(m_lastFuture);
}

bool Evaluator::pull(NodeId id)
{
  PullHandle h = pullAsync(id);
  waitIdle();
  return result(h);
}
```

**(c)** **Replace the ENTIRE `Evaluator::ensure` function** with the version below
(it takes `epoch`, adds two cancellation guards, clears partial output on a
cancellation bail, and passes `epoch` to every recursive call). Do not edit it
surgically — replace the whole function so no recursive `ensure(...)` call site is
missed:
```cpp
bool Evaluator::ensure(NodeId id, uint64_t epoch)
{
  if (m_cancel.load() || epoch != m_epoch.load())
    return false; // cancelled or superseded

  GraphNode *n = m_graph.node(id);
  if (!n)
    return false;
  if (n->state == EvalState::Error)
    return false;

  bool inputsChanged = false;
  for (const auto &c : m_graph.connections()) {
    if (c.toNode != id)
      continue;
    if (!ensure(c.fromNode, epoch))
      return false;
    const GraphNode *producer = m_graph.node(c.fromNode);
    uint64_t pv = producer ? producer->outputVersion : 0;
    auto it = n->consumedInputVersions.find(c.toPort);
    if (it == n->consumedInputVersions.end() || it->second != pv)
      inputsChanged = true;
  }

  const bool cacheable = n->impl->typeInfo().isCacheable;
  const uint64_t paramHash = n->impl->parameters().hash();
  const bool recompute = !cacheable || !n->hasEvaluated || n->cache.empty()
      || paramHash != n->lastParamHash || inputsChanged;

  if (!recompute) {
    n->state = EvalState::Clean;
    return true;
  }

  n->state = EvalState::Computing;
  n->cache.clear();
  NodeId prev = m_current;
  m_current = id;
  EvalContext ctx(*this, *n);
  n->impl->evaluate(ctx);
  m_current = prev;

  if (n->state == EvalState::Error)
    return false;
  // A cancellation observed during evaluate() must not finalize a partial run.
  // Clear the partial output so a later pull recomputes rather than serving a
  // half-written, un-version-stamped cache entry.
  if (m_cancel.load() || epoch != m_epoch.load()) {
    n->cache.clear();
    return false;
  }

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

  for (auto &outPort : n->cache)
    for (auto &resVal : outPort.second)
      resVal.second.version = n->outputVersion;

  n->state = EvalState::Clean;
  return true;
}
```
(The body is identical to Phase 1 except the signature, the two `m_cancel`/epoch
guard checks, and the recursive `ensure(c.fromNode, epoch)` calls.)

**(d)** Add the `EvalContext::cancelled()` definition (next to the other
`EvalContext` methods):
```cpp
bool EvalContext::cancelled() const
{
  return m_eval.cancelRequested();
}
```

- [ ] **Step 5: Build + run**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph::Async' --output-on-failure
```
Expected: PASS (async poll → ready+success+value 5; callback once with true; blocking `pull()` works).

- [ ] **Step 6: Regression — the whole Phase 1 suite still passes**
```bash
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph' --output-on-failure
```
Expected: all green (16 Phase 1 + the new Async test).

- [ ] **Step 7: Commit**
```bash
clang-format -i tsd/src/tsd/graph/Evaluator.hpp tsd/src/tsd/graph/Evaluator.cpp tsd/tests/test_graph_AsyncEval.cpp
jj commit tsd/src/tsd/graph/Evaluator.hpp tsd/src/tsd/graph/Evaluator.cpp tsd/tests/test_graph_AsyncEval.cpp tsd/tests/CMakeLists.txt -m "feat(graph): asynchronous Evaluator (pullAsync + handle/poll, blocking pull retained)"
```

---

## Task 2: Cooperative cancellation

**Files:**
- Test: `tsd/tests/test_graph_AsyncCancel.cpp`

- [ ] **Step 1: Write the test.** A `SlowNode` spins until either the test releases
it or cancellation is requested; a hard iteration cap guarantees the test never
hangs.

`tsd/tests/test_graph_AsyncCancel.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <atomic>
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::EvalState;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::PullHandle;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

// Spins until cancelled() or `release` is set; CAP prevents any hang.
struct SlowNode : Node
{
  ParameterList params;
  std::atomic<bool> *release;
  std::atomic<bool> finished{false};
  explicit SlowNode(std::atomic<bool> *r) : release(r) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("Slow");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    i.isCacheable = false; // always re-run so the second pull recomputes
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    const long CAP = 2000000000L;
    for (long i = 0; i < CAP; ++i) {
      if (ctx.cancelled())
        return; // bail without setOutput
      if (release->load())
        break;
    }
    auto out = std::make_shared<float>(1.0f);
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
    finished.store(true);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator cancels a running pull cooperatively",
    "[graph-asynccancel]")
{
  std::atomic<bool> release{false};
  Graph g;
  auto id = g.addNode(std::make_unique<SlowNode>(&release));
  auto *slow = static_cast<SlowNode *>(g.node(id)->impl.get());

  Evaluator e(g);

  WHEN("a running pull is cancelled before release")
  {
    PullHandle h = e.pullAsync(id);
    e.cancel();   // cooperative: SlowNode sees cancelled() and bails
    e.waitIdle(); // worker observed cancel and returned
    THEN("the pull did not succeed and the node did not finish")
    {
      REQUIRE(e.isReady(h));
      REQUIRE_FALSE(e.result(h));
      REQUIRE_FALSE(slow->finished.load());
    }
    THEN("the worker recovers: a released blocking pull completes")
    {
      release.store(true);
      REQUIRE(e.pull(id));
      const Value *out = e.output(id, Token("out"), hostResidency());
      REQUIRE(out != nullptr);
      REQUIRE(slow->finished.load());
    }
  }
}
```
Register `add_test(NAME tsd::graph::AsyncCancel COMMAND ${PROJECT_NAME} "[graph-asynccancel]")`.

- [ ] **Step 2: Build + run**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph::AsyncCancel' --output-on-failure
```
Expected: PASS — cancelled pull has `result==false`, node not finished; a later released `pull()` succeeds. (No new engine code needed — this exercises Task 1.)

- [ ] **Step 3: Commit**
```bash
clang-format -i tsd/tests/test_graph_AsyncCancel.cpp
jj commit tsd/tests/test_graph_AsyncCancel.cpp tsd/tests/CMakeLists.txt -m "test(graph): cooperative cancellation of an in-flight pull"
```

---

## Task 3: Edit-cancels-in-flight (supersession)

**Files:**
- Test: `tsd/tests/test_graph_AsyncSupersede.cpp`

- [ ] **Step 1: Write the test.** Start a slow pull, then (with the
`cancel()`+`waitIdle()` mutation protocol) edit a param and pull again; the first
handle is superseded, the second produces the updated value.

`tsd/tests/test_graph_AsyncSupersede.cpp`:
```cpp
// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Evaluator.hpp"
// std
#include <atomic>
#include <memory>

using tsd::core::Token;
using tsd::graph::EvalContext;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::PullHandle;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

// Emits param "v"; spins until cancelled or released so a pull can be superseded.
struct SlowSource : Node
{
  ParameterList params;
  std::atomic<bool> *release;
  std::atomic<int> evals{0};
  explicit SlowSource(std::atomic<bool> *r) : release(r) {}
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("SlowSource");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    const long CAP = 2000000000L;
    for (long i = 0; i < CAP; ++i) {
      if (ctx.cancelled())
        return;
      if (release->load())
        break;
    }
    evals++;
    auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator supersedes an in-flight pull after an edit",
    "[graph-supersede]")
{
  std::atomic<bool> release{false};
  Graph g;
  auto id = g.addNode(std::make_unique<SlowSource>(&release));
  auto *src = static_cast<SlowSource *>(g.node(id)->impl.get());
  g.node(id)->impl->parameters().set(Token("v"), 1.0f);

  Evaluator e(g);

  WHEN("a first pull is in flight, then a param edit + second pull happen")
  {
    PullHandle h1 = e.pullAsync(id); // starts running, spins (not released)

    // Mutation protocol: cancel + waitIdle before touching the Graph.
    e.cancel();
    e.waitIdle();
    g.node(id)->impl->parameters().set(Token("v"), 2.0f);
    g.markDirty(id);

    release.store(true);             // let the next run finish
    PullHandle h2 = e.pullAsync(id);
    e.waitIdle();

    THEN("the first pull was superseded and the second produced the new value")
    {
      REQUIRE_FALSE(e.result(h1));
      REQUIRE(e.result(h2));
      const Value *out = e.output(id, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 2.0f);
    }
    THEN("the superseded run did not publish: exactly one successful eval")
    {
      REQUIRE(src->evals.load() == 1);
    }
  }
}
```
Register `add_test(NAME tsd::graph::AsyncSupersede COMMAND ${PROJECT_NAME} "[graph-supersede]")`.

- [ ] **Step 2: Build + run**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph::AsyncSupersede' --output-on-failure
```
Expected: PASS — `result(h1)` false, `result(h2)` true, value 2, exactly one successful eval.

- [ ] **Step 3: Commit**
```bash
clang-format -i tsd/tests/test_graph_AsyncSupersede.cpp
jj commit tsd/tests/test_graph_AsyncSupersede.cpp tsd/tests/CMakeLists.txt -m "test(graph): edit-cancels-in-flight supersession"
```

---

## Task 4: Error isolation across async

**Files:**
- Test: `tsd/tests/test_graph_AsyncError.cpp`

- [ ] **Step 1: Write the test.** A node sets its own `Error` (via a failed required
input from a disconnected port) — reuse the established failure path: a sink with a
required input left unconnected fails to materialize. Simpler and deterministic: a
node that sets an error by reading a missing required input.

`tsd/tests/test_graph_AsyncError.cpp`:
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
using tsd::graph::EvalState;
using tsd::graph::Evaluator;
using tsd::graph::Graph;
using tsd::graph::Node;
using tsd::graph::NodeTypeInfo;
using tsd::graph::ParameterList;
using tsd::graph::PortType;
using tsd::graph::PullHandle;
using tsd::graph::Value;
using tsd::graph::hostResidency;

namespace {

struct ConstSource : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("ConstSource");
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto out = std::make_shared<float>(params.getOr<float>(Token("v"), 0.0f));
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

// Reads a required input that is intentionally left unconnected -> materialize
// fails -> EvalContext sets this node to Error.
struct NeedsInput : Node
{
  ParameterList params;
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("NeedsInput");
    i.inputs.push_back({Token("in"), PortType{Token("scalar")}, true, {}});
    i.outputs.push_back({Token("out"), PortType{Token("scalar")}, true, {}});
    return i;
  }
  ParameterList &parameters() override { return params; }
  void evaluate(EvalContext &ctx) override
  {
    auto in = ctx.input(Token("in"), hostResidency()); // unconnected -> invalid
    auto out = std::make_shared<float>(in.valid() ? 1.0f : 0.0f);
    Value v;
    v.type = PortType{Token("scalar")};
    v.residency = hostResidency();
    v.payload = out;
    ctx.setOutput(Token("out"), v);
  }
};

} // namespace

SCENARIO("tsd::graph::Evaluator isolates a node error in an async pull",
    "[graph-asyncerror]")
{
  Graph g;
  auto good = g.addNode(std::make_unique<ConstSource>());
  g.node(good)->impl->parameters().set(Token("v"), 7.0f);
  auto bad = g.addNode(std::make_unique<NeedsInput>()); // "in" left unconnected

  Evaluator e(g);

  WHEN("pulling the node whose required input is unconnected")
  {
    PullHandle h = e.pullAsync(bad);
    e.waitIdle();
    THEN("the pull fails and the node is in Error")
    {
      REQUIRE(e.isReady(h));
      REQUIRE_FALSE(e.result(h));
      REQUIRE(g.node(bad)->state == EvalState::Error);
    }
  }

  WHEN("pulling an unrelated healthy node afterward")
  {
    e.pull(bad); // leaves `bad` in Error
    THEN("the healthy branch still resolves")
    {
      REQUIRE(e.pull(good));
      const Value *out = e.output(good, Token("out"), hostResidency());
      REQUIRE(*std::static_pointer_cast<float>(out->payload) == 7.0f);
    }
  }
}
```
Register `add_test(NAME tsd::graph::AsyncError COMMAND ${PROJECT_NAME} "[graph-asyncerror]")`.

- [ ] **Step 2: Build + run**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph::AsyncError' --output-on-failure
```
Expected: PASS — bad node → Error, `result` false; healthy node still pulls to 7.

- [ ] **Step 3: Commit**
```bash
clang-format -i tsd/tests/test_graph_AsyncError.cpp
jj commit tsd/tests/test_graph_AsyncError.cpp tsd/tests/CMakeLists.txt -m "test(graph): error isolation across async pulls"
```

---

## Task 5: Full-suite gate

- [ ] **Step 1: Run the entire graph suite**
```bash
cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests
ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R 'tsd::graph' --output-on-failure
```
Expected: all `tsd::graph::*` tests PASS (16 Phase 1 + Async, AsyncCancel, AsyncSupersede, AsyncError = 20). Report the summary line and confirm `.envrc` is still uncommitted (`jj status`).

- [ ] **Step 2: No new commit if nothing changed** (this is a verification gate). If a flake or fix is needed, address it in the relevant task's files and commit with explicit paths.

---

## Phase 2 completion checklist

- [ ] `Evaluator` owns a single `tsd::core::TaskQueue` worker; `m_worker` declared last
- [ ] `pullAsync` + `PullHandle` + `isReady`/`result` + optional `onComplete`
- [ ] `cancel()` + `waitIdle()`; blocking `pull()` retained (Phase 1 tests unchanged)
- [ ] `ensure(id, epoch)` checks `m_cancel`/epoch before each node and before finalizing
- [ ] `EvalContext::cancelled()` for cooperative node bail
- [ ] edit-cancels-in-flight via cancel-before-enqueue + epoch supersession (no snapshot)
- [ ] error isolation preserved across async
- [ ] `~Evaluator` cancels + waits idle; worker joined safely (member order)
- [ ] full suite green (20 tests)

## Out of scope (unchanged from spec)

Concurrent multi-node evaluation / thread pool; contentTag short-circuit; cache
eviction; real CUDA + kernel cancellation; UI debounce; render bridge/viewports.

## Self-review notes

- **Threading correctness rests on:** single FIFO worker + cancel-set-before-enqueue
  (so the new task can't start until the prior bailed) + epoch supersession. The
  worker is the only writer of node cache/state and `m_report`/`m_transferCache`;
  the main thread reads only after `isReady`. No snapshot, no per-node tasks.
- **Member-destruction order** is load-bearing: `m_worker` is declared **last** so it
  (and its thread `join`) tears down before the atomics/graph reference it touches.
  `~Evaluator` also explicitly `cancel()`+`waitIdle()` first.
- **Tests can't hang:** every spinning test node has a hard iteration `CAP` and bails
  on `cancelled()`/`release`; no real sleeps or clocks (deterministic, no
  `Date.now`-style nondeterminism).
- **`output()`/`lastReport()` are read only after `waitIdle()`/`isReady()`** in every
  test, honoring the completion-gated-publish contract.
