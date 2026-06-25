// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph_nodes/DemoGraph.hpp"
// std
#include <cassert>

namespace tsd::graph_nodes {

using tsd::core::Token;
using tsd::graph::NodeId;

DemoDisplays buildVolumeSurfaceDemo(
    tsd::graph::Graph &g, tsd::graph::NodeRegistry &reg)
{
  auto add = [&](const char *t) -> NodeId {
    return g.addNode(reg.create(Token(t)));
  };
  // Graph::connect returns LinkResult; a mistyped port must fail loudly, not
  // silently drop the edge (which would surface as an opaque render miss).
  auto link = [&](NodeId f, const char *fp, NodeId t, const char *tp) {
    const bool ok = g.connect(f, Token(fp), t, Token(tp)).ok;
    assert(ok && "buildVolumeSurfaceDemo: connect failed");
    (void)ok;
  };

  const NodeId src = add("GenerateNoiseVolume");
  const NodeId sr = add("ScalarRange");
  const NodeId tf = add("TransferFunction");
  const NodeId dv = add("DisplayVolume");
  const NodeId bb = add("BoundingBox");
  const NodeId ds = add("DisplaySurface");

  link(src, "out", sr, "in");
  link(sr, "out", tf, "in");
  link(src, "out", dv, "field"); // fan-out
  link(tf, "out", dv, "tf"); // multi-input
  link(src, "out", bb, "in");
  link(bb, "out", ds, "in");

#ifdef TSD_GRAPH_NODES_HAVE_VISKORES
  // Isosurface path: field -> IsosurfaceExtract -> its own DisplaySurface.
  const NodeId iso = add("IsosurfaceExtract");
  const NodeId dsIso = add("DisplaySurface");
  link(src, "out", iso, "in");
  link(iso, "out", dsIso, "in");
#endif

  return DemoDisplays{src, dv, ds};
}

} // namespace tsd::graph_nodes
