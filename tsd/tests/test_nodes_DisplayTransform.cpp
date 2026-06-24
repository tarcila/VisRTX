// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph/Graph.hpp"
#include "tsd/graph/NodeRegistry.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/DemoGraph.hpp"
#include "tsd/graph_nodes/DisplayTransform.hpp"
#include "tsd/graph_nodes/TransformableNode.hpp"

using tsd::core::Token;
using namespace tsd::graph;
using tsd::graph_nodes::collectDisplayTransforms;
using mat4 = tsd::core::math::mat4;

namespace {
const tsd::graph_nodes::DisplayTransform *find(
    const std::vector<tsd::graph_nodes::DisplayTransform> &v, NodeId id)
{
  for (const auto &dt : v)
    if (dt.node == id)
      return &dt;
  return nullptr;
}
} // namespace

SCENARIO("collectDisplayTransforms reports display transforms",
    "[display-transform]")
{
  NodeRegistry reg;
  tsd::graph_nodes::registerBuiltinNodes(reg);
  Graph g;
  auto d = tsd::graph_nodes::buildVolumeSurfaceDemo(g, reg);

  WHEN("transforms are default")
  {
    auto xs = collectDisplayTransforms(g);
    THEN("both display nodes appear at identity")
    {
      REQUIRE(xs.size() == 2);
      REQUIRE(find(xs, d.volumeDisplay) != nullptr);
      REQUIRE(find(xs, d.surfaceDisplay) != nullptr);
      REQUIRE(find(xs, d.volumeDisplay)->xfm == tsd::core::math::IDENTITY_MAT4);
    }
    THEN("non-display nodes are excluded")
    {
      REQUIRE(find(xs, d.source) == nullptr);
    }
  }

  WHEN("one display's transform is set via ITransformableNode")
  {
    mat4 m = tsd::core::math::IDENTITY_MAT4;
    m[3].x = 5.f; // translate +5 in x (column-major: column 3 = translation)
    auto *itf = dynamic_cast<tsd::graph_nodes::ITransformableNode *>(
        g.node(d.volumeDisplay)->impl.get());
    REQUIRE(itf != nullptr);
    itf->transform() = m;
    auto xs = collectDisplayTransforms(g);
    THEN("the helper reports it; the other stays identity")
    {
      REQUIRE(find(xs, d.volumeDisplay)->xfm == m);
      REQUIRE(
          find(xs, d.surfaceDisplay)->xfm == tsd::core::math::IDENTITY_MAT4);
    }
  }
}
