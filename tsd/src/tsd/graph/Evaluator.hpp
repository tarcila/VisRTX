// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TaskQueue.hpp"
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/TransferRegistry.hpp"
// std
#include <atomic>
#include <cstdint>
#include <functional>
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
class Evaluator
{
 public:
  explicit Evaluator(Graph &g,
      const TransferRegistry *transfers = nullptr,
      const ConversionRegistry *conversions = nullptr);
  ~Evaluator();

  PullHandle pullAsync(NodeId id, std::function<void(bool)> onComplete = {});
  bool isReady(PullHandle h) const;
  bool result(PullHandle h) const;
  void cancel();
  void waitIdle();
  bool pull(NodeId id);

  const Value *output(
      NodeId id, tsd::core::Token port, const Residency &want) const;

  const EvalReport &lastReport() const
  {
    return m_report;
  }

  bool cancelRequested() const
  {
    return m_cancelEpoch.load() >= m_epoch.load();
  }

 private:
  friend class EvalContext;
  bool ensure(NodeId id, uint64_t epoch);
  // Materialize a producer's output as `wantType` in residency `want`,
  // inserting a conversion and/or transfer and recording each in the
  // EvalReport. Transfer results are cached. Returns false on failure (no path
  // / no conversion).
  bool materializeForInput(const Connection &c,
      PortType wantType,
      const Residency &want,
      Value &out);

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

  std::atomic<uint64_t> m_epoch{0}; // bumped per pullAsync
  std::atomic<uint64_t> m_doneEpoch{0}; // highest epoch whose task has finished
  std::atomic<bool> m_doneOk{false}; // success of the most-recent finished task
  // Epoch up to which cancellation has been requested. A task for epoch e is
  // cancelled iff m_cancelEpoch >= e. cancel() sets this to the current epoch;
  // pullAsync supersedes the prior epoch by bumping m_epoch, so no explicit
  // "clear cancel" is needed in the worker — an old cancel only affects epochs
  // <= the old value.
  std::atomic<uint64_t> m_cancelEpoch{0};
  tsd::core::Future m_lastFuture; // future of the most-recently enqueued task
  tsd::core::TaskQueue m_worker{8}; // MUST be declared last (joins on destruct)
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
  bool cancelled() const;

  // Mark this node as failed (Error state) with a message; evaluate() should
  // return immediately after calling this. The evaluator short-circuits
  // downstream consumers.
  void fail(const std::string &msg);

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
