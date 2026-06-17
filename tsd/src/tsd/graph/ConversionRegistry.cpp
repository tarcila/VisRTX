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

const ConversionEntry *ConversionRegistry::find(
    PortType from, PortType to) const
{
  for (const auto &e : m_entries) {
    if (e.from == from && e.to == to)
      return &e;
  }
  return nullptr;
}

} // namespace tsd::graph
