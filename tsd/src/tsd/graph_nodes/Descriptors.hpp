// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/AnyArray.hpp"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/Token.hpp"
#include "tsd/graph/Renderable.hpp"
// std
#include <memory>

namespace tsd::graph_nodes {

struct Field
{
  tsd::core::Token subtype{tsd::core::Token("structuredRegular")};
  tsd::core::math::uint3 dims{0u, 0u, 0u};
  tsd::core::math::float3 origin{-1.f, -1.f, -1.f};
  tsd::core::math::float3 spacing{1.f, 1.f, 1.f};
  tsd::core::AnyArray data;
};

struct TransferFunctionData
{
  tsd::core::AnyArray colormap;
  tsd::core::math::float2 valueRange{0.f, 1.f};
};

struct SurfaceData
{
  tsd::core::Token geomSubtype{tsd::core::Token("triangle")};
  tsd::graph::RenderableParams prim;
  tsd::graph::RenderableParams appearance;
};

inline tsd::core::Token portField()
{
  return tsd::core::Token("field");
}
inline tsd::core::Token portRange()
{
  return tsd::core::Token("range");
}
inline tsd::core::Token portTF()
{
  return tsd::core::Token("transferFunction");
}
inline tsd::core::Token portSurface()
{
  return tsd::core::Token("surface");
}
inline tsd::core::Token portRenderable()
{
  return tsd::core::Token("renderable");
}

} // namespace tsd::graph_nodes
