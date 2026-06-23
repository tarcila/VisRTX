// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// catch
#include "catch.hpp"
// tsd
#include "tsd/graph_nodes/GraphEditModel.hpp"

using tsd::core::ColorPoint;
using tsd::core::OpacityPoint;
using tsd::graph_nodes::GraphEditModel;
using float4 = tsd::core::math::float4;

SCENARIO("sampleColormap interpolates color and opacity over [0,1]",
    "[tfn-sampling]")
{
  // ColorPoint is {position, R, G, B}; OpacityPoint is {position, opacity}.
  std::vector<ColorPoint> colors = {
      {0.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 0.f}}; // black->red
  std::vector<OpacityPoint> opac = {{0.f, 0.f}, {1.f, 1.f}}; // 0 -> 1 ramp

  auto cm = GraphEditModel::sampleColormap(colors, opac, 3);

  THEN("there are `samples` entries")
  {
    REQUIRE(cm.size() == 3);
  }
  THEN("endpoints match the control points")
  {
    REQUIRE(cm.front().x == Approx(0.f)); // R at t=0
    REQUIRE(cm.front().w == Approx(0.f)); // A at t=0
    REQUIRE(cm.back().x == Approx(1.f)); // R at t=1
    REQUIRE(cm.back().w == Approx(1.f)); // A at t=1
  }
  THEN("the midpoint is halfway")
  {
    REQUIRE(cm[1].x == Approx(0.5f).margin(0.01)); // R at t=0.5
    REQUIRE(cm[1].w == Approx(0.5f).margin(0.01)); // A at t=0.5
    REQUIRE(cm[1].y == Approx(0.f)); // G stays 0
  }
}
