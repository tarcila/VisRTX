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
