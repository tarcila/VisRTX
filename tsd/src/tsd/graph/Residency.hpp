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
// pointer) then deviceId, so a value on CUDA device 0 is distinct from
// device 1.
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
