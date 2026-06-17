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
