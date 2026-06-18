// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/NodeRegistry.hpp"

namespace tsd::graph_nodes {

void registerBuiltinNodes(tsd::graph::NodeRegistry &reg);
void registerBuiltinNodes();

void registerGenerateNoiseVolume(tsd::graph::NodeRegistry &reg);
void registerScalarRange(tsd::graph::NodeRegistry &reg);
void registerTransferFunction(tsd::graph::NodeRegistry &reg);
void registerDisplayVolume(tsd::graph::NodeRegistry &reg);
void registerBoundingBox(tsd::graph::NodeRegistry &reg);
void registerDisplaySurface(tsd::graph::NodeRegistry &reg);

} // namespace tsd::graph_nodes
