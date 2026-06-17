// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/PortType.hpp"
#include "tsd/graph/Residency.hpp"
// std
#include <cstdint>
#include <memory>
#include <optional>

namespace tsd::graph {

// A value carried on a wire. Cache invalidation uses `version` (a monotonic
// stamp bumped by the producer on each re-emit) — never a content scan.
// `contentTag` is an optional equality tag populated only by cheap host-side
// producers where "identical output despite recompute" is worth detecting.
//
// The `payload`'s shared_ptr deleter is responsible for freeing with the
// correct backend allocator. A payload is valid only on its own `residency`.
struct Value
{
  PortType type;
  Residency residency;
  std::shared_ptr<void> payload;
  uint64_t producerNodeId{0};
  uint64_t version{0};
  std::optional<uint64_t> contentTag;

  bool valid() const
  {
    return static_cast<bool>(payload);
  }
};

} // namespace tsd::graph
