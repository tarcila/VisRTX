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
