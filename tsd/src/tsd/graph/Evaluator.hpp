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
