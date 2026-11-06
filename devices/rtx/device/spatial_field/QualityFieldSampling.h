// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "StoredFieldPreviewSampler.h"

namespace visrtx {
namespace quality_field {

// A different interpolant, not the canonical reference sampling graph.
// Clamp rounding back into the endpoint hull for ordinary values. Fast-math
// may flush interior subnormals to zero: the scene's ordinary raw-hull
// endpoints then necessarily enclose zero. The caller MUST require
// previewFloatEligible; texture availability alone is not a certificate for
// arbitrary FP32 optics.
VISRTX_HOST_DEVICE float blend(float a, float b, float w)
{
  const float value = a * (1.f - w) + b * w;
  return fminf(fmaxf(value, fminf(a, b)), fmaxf(a, b));
}

VISRTX_HOST_DEVICE bool eligible(const StoredFieldReferenceGPUData &field)
{
  if (!field.previewPointTexture || field.customProvider || field.rectilinear
      || !field.linear
      || (field.sparse && field.gridType != nanovdb::GridType::Float))
    return false;
  for (int c = 0; c < 3; ++c)
    if (field.indexMin[c] < -262144 || field.indexMax[c] > 262144)
      return false;
  return true;
}

#if defined(__CUDACC__)
VISRTX_DEVICE VISRTX_NOINLINE float sample(
    const StoredFieldReferenceGPUData &field, const vec3 &p);
#endif
#if defined(__CUDA_ARCH__)
VISRTX_DEVICE VISRTX_NOINLINE float sample(
    const StoredFieldReferenceGPUData &field, const vec3 &p)
{
  float x[3];
  if (field.sparse)
    preview_math::sparseIndexCoordinates<float>(field, p, x);
  else {
    for (int c = 0; c < 3; ++c)
      x[c] = (p[c] - field.origin[c]) / field.spacing[c]
          - (field.cellCentered ? .5f : 0.f);
  }
  int lo[3], hi[3];
  float w[3];
  for (int c = 0; c < 3; ++c) {
    x[c] = preview_math::coordinate(
        x[c], float(field.indexMin[c]), float(field.indexMax[c]));
    lo[c] = int(floorf(x[c]));
    hi[c] = lo[c] < field.indexMax[c] ? lo[c] + 1 : lo[c];
    w[c] = x[c] - float(lo[c]);
  }
  preview_math::PointTextureFetch fetch{
      field.previewPointTexture, field.indexMin};
  const float a =
      blend(fetch(lo[0], lo[1], lo[2]), fetch(hi[0], lo[1], lo[2]), w[0]);
  const float b =
      blend(fetch(lo[0], hi[1], lo[2]), fetch(hi[0], hi[1], lo[2]), w[0]);
  const float c =
      blend(fetch(lo[0], lo[1], hi[2]), fetch(hi[0], lo[1], hi[2]), w[0]);
  const float d =
      blend(fetch(lo[0], hi[1], hi[2]), fetch(hi[0], hi[1], hi[2]), w[0]);
  return blend(blend(a, b, w[1]), blend(c, d, w[1]), w[2]);
}
#endif
} // namespace quality_field
} // namespace visrtx
