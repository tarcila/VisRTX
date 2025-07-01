// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ClearBuffersPass.h"
// std
#include <algorithm>
#include <limits>

#include "detail/parallel_for.h"

namespace tsd::rendering {

ClearBuffersPass::ClearBuffersPass() = default;
ClearBuffersPass::~ClearBuffersPass() = default;

void ClearBuffersPass::render(RenderBuffers &b, int /*stageId*/)
{
  auto size = getDimensions();
  const size_t totalSize = size_t(size.x) * size_t(size.y);
#if ENABLE_CUDA
  thrust::fill(b.color, b.color + totalSize, 0u);
  thrust::fill(
      b.depth, b.depth + totalSize, std::numeric_limits<float>::infinity());
  thrust::fill(b.objectId, b.objectId + totalSize, ~0u);
#else
  std::fill(b.color, b.color + totalSize, 0u);
  std::fill(
      b.depth, b.depth + totalSize, std::numeric_limits<float>::infinity());
  std::fill(b.objectId, b.objectId + totalSize, ~0u);
#endif
}

} // namespace tsd::rendering
