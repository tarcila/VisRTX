// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/BuiltinNodes.hpp"

namespace tsd::graph_nodes {

void registerBuiltinNodes(tsd::graph::NodeRegistry &reg)
{
  registerGenerateNoiseVolume(reg);
  registerGenerateGyroid(reg);
  registerGenerateTurbulence(reg);
  registerGenerateMetaballs(reg);
  registerScalarRange(reg);
  registerTransferFunction(reg);
  registerDisplayVolume(reg);
  registerBoundingBox(reg);
  registerDisplaySurface(reg);
  registerDisplayLight(reg);
#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
  registerIsosurfaceExtract(reg);
#endif
}

void registerBuiltinNodes()
{
  registerBuiltinNodes(tsd::graph::GlobalNodeRegistry());
}

} // namespace tsd::graph_nodes
