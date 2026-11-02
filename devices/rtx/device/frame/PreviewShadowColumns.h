// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "gpu/renderer/previewFloatLattice.h"
#include "spatial_field/StoredFieldPreviewSampler.h"

namespace visrtx {
namespace preview_columns {
VISRTX_DEVICE float previousBoundary(
    const PreviewDensityIntegral &q, float upper, const float *cuts, int count)
{
  const float index = floorf((upper - q.anchor) / q.step);
  float lower = q.anchor + index * q.step;
  if (!(lower < upper))
    lower = q.anchor + (index - 1.f) * q.step;
  if (!(lower < upper) || lower < q.lower)
    lower = q.lower;
  for (int i = 0; i < count; ++i)
    if (cuts[i] < upper && cuts[i] > lower)
      lower = cuts[i];
  return lower;
}
template <typename Sample>
VISRTX_DEVICE float density(
    const PreviewDensityIntegral &q, float t, Sample &sample)
{
  const vec3 p(q.origin[0] + q.direction[0] * t,
      q.origin[1] + q.direction[1] * t,
      q.origin[2] + q.direction[2] * t);
  if (!preview_math::contains(q.volumeDomain, p))
    return 0.f;
  return q.scale
      * (preview_math::contains(q.fieldDomain, p) ? sample(p) : q.outside);
}
template <typename Sample>
VISRTX_DEVICE void integrate(float *output,
    uint32_t n,
    size_t column,
    float zLower,
    float zUpper,
    const PreviewDensityIntegral &q,
    const float *cuts,
    int count,
    Sample &sample)
{
  float upper = q.upper;
  float lower = previousBoundary(q, upper, cuts, count);
  float suffix = 0.f;
  for (int z = int(n) - 1; z >= 0; --z) {
    const float node = zLower + (float(z) / float(n - 1)) * (zUpper - zLower);
    float value = 0.f;
    if (node < q.upper) {
      const float target = fmaxf(node, q.lower);
      while (target < lower) {
        suffix +=
            density(q, lower + (upper - lower) * .5f, sample) * (upper - lower);
        upper = lower;
        lower = previousBoundary(q, upper, cuts, count);
      }
      // Re-evaluate the first partial cell exactly as the per-node marcher;
      // do not approximate its density by the full cell's midpoint sample.
      value = suffix
          + density(q, target + (upper - target) * .5f, sample)
              * (upper - target);
    }
    output[column + size_t(n) * n * size_t(z)] = value;
  }
}
template <typename BuildT>
VISRTX_DEVICE void sparse(float *output,
    uint32_t n,
    size_t column,
    float zLower,
    float zUpper,
    const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &q,
    const float *cuts,
    int count)
{
  const auto &grid =
      *static_cast<const nanovdb::NanoGrid<BuildT> *>(field.data);
  preview_math::DensitySample<BuildT> sample{field, {grid.getAccessor()}};
  integrate(output, n, column, zLower, zUpper, q, cuts, count, sample);
}
VISRTX_GLOBAL void buildPreviewShadowColumns(float *output,
    uint32_t n,
    AggregateVolumeSceneGPUData scene,
    mat4 cacheToWorld,
    vec3 cacheLower,
    vec3 cacheUpper,
    float rate)
{
  const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= size_t(n) * n)
    return;
  const auto &o = scene.occurrences[0];
  const auto &b = o.volume.principled->bindings;
  const vec3 p(cacheLower.x
          + float(column % n) / float(n - 1) * (cacheUpper.x - cacheLower.x),
      cacheLower.y
          + float(column / n) / float(n - 1) * (cacheUpper.y - cacheLower.y),
      0.f);
  const vec3 origin = vec3(cacheToWorld * vec4(p, 1.f));
  const vec3 direction = vec3(cacheToWorld[2]);
  preview_volume::Ray ray;
  for (int c = 0; c < 3; ++c) {
    ray.origin[c] = origin[c];
    ray.direction[c] = direction[c];
  }
  ray.lower = -INFINITY;
  ray.upper = INFINITY;
  const auto prepared = practical_preview::prepareRay(scene, ray, rate);
  if (!prepared.hit) {
    for (uint32_t z = 0; z < n; ++z)
      output[column + size_t(n) * n * z] = 0.f;
    return;
  }
  PreviewDensityIntegral q;
  for (int c = 0; c < 3; ++c) {
    q.origin[c] = prepared.local.origin[c];
    q.direction[c] = prepared.local.direction[c];
  }
  q.lower = prepared.full.lower;
  q.upper = prepared.full.upper;
  q.anchor = q.lower;
  q.step = prepared.step;
  q.volumeDomain = b.bounds;
  q.fieldDomain = b.density.domain;
  q.scale = b.density.scale;
  q.outside = b.density.outside;
  // Capacity follows the eleven binding roles, not an occurrence/hit cap.
  float cuts[22];
  int count = 0;
  for (int role = 0; role < 11; ++role) {
    const auto *domain = preview_volume::bindingDomain(b, role);
    preview_volume::Interval cut;
    if (domain
        && preview_volume::intersectDomain(
            *domain, prepared.local, q.lower, q.upper, cut)) {
      if (cut.lower > q.lower && cut.lower < q.upper)
        cuts[count++] = cut.lower;
      if (cut.upper > q.lower && cut.upper < q.upper)
        cuts[count++] = cut.upper;
    }
  }
  const auto &field = o.volume.principled->fields[b.density.field];
  if (!field.sparse) {
    preview_math::DenseDensitySample sample{field};
    integrate(
        output, n, column, cacheLower.z, cacheUpper.z, q, cuts, count, sample);
    return;
  }
  switch (field.gridType) {
  case nanovdb::GridType::Float:
    sparse<float>(
        output, n, column, cacheLower.z, cacheUpper.z, field, q, cuts, count);
    break;
  case nanovdb::GridType::Fp4:
    sparse<nanovdb::Fp4>(
        output, n, column, cacheLower.z, cacheUpper.z, field, q, cuts, count);
    break;
  case nanovdb::GridType::Fp8:
    sparse<nanovdb::Fp8>(
        output, n, column, cacheLower.z, cacheUpper.z, field, q, cuts, count);
    break;
  case nanovdb::GridType::Fp16:
    sparse<nanovdb::Fp16>(
        output, n, column, cacheLower.z, cacheUpper.z, field, q, cuts, count);
    break;
  case nanovdb::GridType::FpN:
    sparse<nanovdb::FpN>(
        output, n, column, cacheLower.z, cacheUpper.z, field, q, cuts, count);
    break;
  default:
    break; // Prepared scalar descriptors cannot reach this case.
  }
}
} // namespace preview_columns
} // namespace visrtx
