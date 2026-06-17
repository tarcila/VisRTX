// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/TransferRegistry.hpp"

namespace tsd::graph {

// Registers host<->test transfers for the "array" PortType. The "test" backend
// uses the same std::vector<float> payload as host; transfers only retag
// residency (and, for realism, copy the buffer). Used by CI to exercise the
// residency machinery without CUDA.
void registerTestBackendTransfers(TransferRegistry &reg);

} // namespace tsd::graph
