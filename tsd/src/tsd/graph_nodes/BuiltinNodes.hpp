// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/graph/NodeRegistry.hpp"

namespace tsd::graph_nodes {

void registerBuiltinNodes(tsd::graph::NodeRegistry &reg);
void registerBuiltinNodes();

void registerGenerateNoiseVolume(tsd::graph::NodeRegistry &reg);
void registerGenerateGyroid(tsd::graph::NodeRegistry &reg);
void registerGenerateTurbulence(tsd::graph::NodeRegistry &reg);
void registerGenerateMetaballs(tsd::graph::NodeRegistry &reg);
void registerScalarRange(tsd::graph::NodeRegistry &reg);
void registerTransferFunction(tsd::graph::NodeRegistry &reg);
void registerDisplayVolume(tsd::graph::NodeRegistry &reg);
void registerBoundingBox(tsd::graph::NodeRegistry &reg);
void registerDisplaySurface(tsd::graph::NodeRegistry &reg);
void registerDisplayLight(tsd::graph::NodeRegistry &reg);
#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
void registerIsosurfaceExtract(tsd::graph::NodeRegistry &reg);
void registerCrossSection(tsd::graph::NodeRegistry &reg);
#endif

} // namespace tsd::graph_nodes
