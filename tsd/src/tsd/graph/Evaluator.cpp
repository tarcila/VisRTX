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

Value EvalContext::inputOr(
    tsd::core::Token name, const Residency &want, Value alt)
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
