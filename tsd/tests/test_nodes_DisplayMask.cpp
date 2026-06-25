// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/DisplayMask.hpp"
// std
#include <algorithm>

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::collectDisplayMasks;
using tsd::graph_nodes::kDefaultViewportMask;

namespace {
const tsd::graph_nodes::DisplayMask *find(
    const std::vector<tsd::graph_nodes::DisplayMask> &v, NodeId id)
{
  for (const auto &dm : v)
    if (dm.node == id)
      return &dm;
  return nullptr;
}
} // namespace

SCENARIO("collectDisplayMasks reports display nodes and their masks",
    "[display-mask]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);

  WHEN("masks are read from a fresh demo graph")
  {
    auto masks = collectDisplayMasks(g);
    THEN("exactly the display nodes appear, each at the default mask")
    {
      // The demo graph has 2 display nodes unconditionally (DisplayVolume +
      // DisplaySurface) plus a third DisplaySurface under
      // TSD_GRAPH_NODES_HAVE_VISKORES when the isosurface branch is compiled in.
#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
      REQUIRE(masks.size() == 3);
#else
      REQUIRE(masks.size() == 2);
#endif
      REQUIRE(find(masks, d.volumeDisplay) != nullptr);
      REQUIRE(find(masks, d.surfaceDisplay) != nullptr);
      REQUIRE(
          find(masks, d.volumeDisplay)->mask == uint64_t(kDefaultViewportMask));
      REQUIRE(find(masks, d.surfaceDisplay)->mask
          == uint64_t(kDefaultViewportMask));
    }
    THEN("non-display nodes are excluded")
    {
      REQUIRE(find(masks, d.source) == nullptr); // GenerateNoiseVolume
    }
  }

  WHEN("one display's viewportMask param is changed to 0b11")
  {
    g.node(d.volumeDisplay)
        ->impl->parameters()
        .set(Token("viewportMask"), 0b11);
    auto masks = collectDisplayMasks(g);
    THEN("the helper reports the new mask for it, default for the other")
    {
      REQUIRE(find(masks, d.volumeDisplay)->mask == uint64_t(0b11));
      REQUIRE(find(masks, d.surfaceDisplay)->mask
          == uint64_t(kDefaultViewportMask));
    }
  }
}
