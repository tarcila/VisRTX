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
