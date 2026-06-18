// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/AnyArray.hpp"
#include "tsd/core/Token.hpp"
// std
#include <utility>
#include <vector>

namespace tsd::graph {

// Named scalar and array parameters, mapped 1:1 onto tsd Object::setParameter /
// setParameterObject by the render bridge. Backend-agnostic (host data only).
struct RenderableParams
{
  std::vector<std::pair<tsd::core::Token, tsd::core::Any>> scalars;
  std::vector<std::pair<tsd::core::Token, tsd::core::AnyArray>> arrays;
};

// A backend-agnostic description of one renderable thing. The bridge turns it
// into a tsd::Surface (geometry+material) or tsd::Volume (spatial field + TF).
// For a structuredRegular volume, `prim` carries a `dims` (float3) scalar
// alongside the `data` array so the bridge can build a 3D data array.
struct Renderable
{
  enum class Kind
  {
    Surface,
    Volume
  };
  Kind kind{Kind::Surface};
  tsd::core::Token primSubtype; // geometry subtype, or spatial-field subtype
  RenderableParams prim; // geometry params, or spatial-field params
  RenderableParams appearance; // material params, or volume/TF params
};

} // namespace tsd::graph
