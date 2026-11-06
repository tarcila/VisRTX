// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "PreviewDensityIntegral.h"
#include "StoredFieldReferenceSampler.h"
#include "gpu/volumePreviewMath.h"

namespace visrtx {
namespace preview_math {
VISRTX_HOST_DEVICE float coordinate(float x, float lo, float hi)
{
  return fminf(fmaxf(x, lo), hi);
}
VISRTX_HOST_DEVICE float axisLUT(const float *lut, float normalized)
{
  const float x = coordinate(normalized * 8192.f - .5f, 0.f, 8191.f);
  const int lo = int(x), hi = lo < 8191 ? lo + 1 : lo;
  return lerp(lut[lo], lut[hi], x - float(lo));
}
#ifdef __CUDA_ARCH__
struct PointTextureFetch
{
  cudaTextureObject_t texture;
  ivec3 indexMin;
  VISRTX_DEVICE float operator()(int x, int y, int z) const;
};
VISRTX_DEVICE float PointTextureFetch::operator()(int x, int y, int z) const
{
  return tex3D<float>(texture,
      float(x - indexMin.x) + .5f,
      float(y - indexMin.y) + .5f,
      float(z - indexMin.z) + .5f);
}
#endif

template <typename Fetch>
VISRTX_HOST_DEVICE auto filter(const StoredFieldReferenceGPUData &field,
    const float (&coordinates)[3],
    Fetch &&fetch)
{
  int lo[3], hi[3];
  float w[3];
  for (int c = 0; c < 3; ++c) {
    const float x = coordinate(
        coordinates[c], float(field.indexMin[c]), float(field.indexMax[c]));
    lo[c] = int(floorf(field.linear ? x : x + .5f));
    hi[c] = field.linear && lo[c] < field.indexMax[c] ? lo[c] + 1 : lo[c];
    w[c] = x - float(lo[c]);
  }
  if (!field.linear)
    return fetch(lo[0], lo[1], lo[2]);
  const auto a =
      lerp(fetch(lo[0], lo[1], lo[2]), fetch(hi[0], lo[1], lo[2]), w[0]);
  const auto b =
      lerp(fetch(lo[0], hi[1], lo[2]), fetch(hi[0], hi[1], lo[2]), w[0]);
  const auto c =
      lerp(fetch(lo[0], lo[1], hi[2]), fetch(hi[0], lo[1], hi[2]), w[0]);
  const auto d =
      lerp(fetch(lo[0], hi[1], hi[2]), fetch(hi[0], hi[1], hi[2]), w[0]);
  return lerp(lerp(a, b, w[1]), lerp(c, d, w[1]), w[2]);
}
template <typename T>
VISRTX_HOST_DEVICE T dense(const StoredFieldReferenceGPUData &field, vec3 p)
{
  float x[3];
  for (int c = 0; c < 3; ++c) {
    if (field.rectilinear) {
      const float normalized = (p[c] - field.axisLower[c])
          / (field.axisUpper[c] - field.axisLower[c]);
      const float segments =
          float(field.indexMax[c]) + (field.cellCentered ? 1.f : 0.f);
      x[c] = axisLUT(field.axisLUT[c], normalized) * segments;
    } else
      x[c] = (p[c] - field.origin[c]) / field.spacing[c];
    if (field.cellCentered)
      x[c] -= .5f;
  }
#ifdef __CUDA_ARCH__
  if constexpr (std::is_same_v<T, float>)
    if (field.previewPointTexture)
      return filter(field,
          x,
          PointTextureFetch{field.previewPointTexture, field.indexMin});
#endif
  return filter(field, x, optical_reference::DenseReferenceFetch<T>{field});
}
template <typename BuildT>
struct CachedNanoFetch
{
  typename nanovdb::NanoGrid<BuildT>::AccessorType accessor;
  VISRTX_HOST_DEVICE auto operator()(int x, int y, int z) const;
};
template <typename BuildT>
VISRTX_HOST_DEVICE auto CachedNanoFetch<BuildT>::operator()(
    int x, int y, int z) const
{
  return accessor.template get<optical_reference::ReferenceNanoValue<BuildT>>(
      nanovdb::Coord(x, y, z));
}

template <typename BuildT>
VISRTX_HOST_DEVICE void sparseIndexCoordinates(
    const StoredFieldReferenceGPUData &field, vec3 p, float (&x)[3])
{
  const auto &grid =
      *static_cast<const nanovdb::NanoGrid<BuildT> *>(field.data);

  const auto index = grid.worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
  for (int c = 0; c < 3; ++c) {
    const float first = float(field.indexMin[c]);
    const float dims = float(field.indexMax[c]) - first + 1.f;
    const float relative = index[c] - first;
    if (field.rectilinear) {
      const float span = fmaxf(dims - 1.f, 1.f);
      x[c] = axisLUT(field.axisLUT[c],
                 relative / (field.cellCentered ? span : dims))
              * span
          + first;
    } else
      x[c] = (field.cellCentered ? relative : relative * ((dims - 1.f) / dims))
          + first;
    if (field.cellCentered)
      x[c] -= .5f;
  }
}
template <typename BuildT, typename Fetch>
VISRTX_HOST_DEVICE auto sparseWithCache(
    const StoredFieldReferenceGPUData &field, vec3 p, Fetch &fetch)
{
  float x[3];
  sparseIndexCoordinates<BuildT>(field, p, x);
  // Preserve packed storage decoding; only its tiny decoder retains FP64.
  return filter(field, x, fetch);
}
template <typename BuildT>
VISRTX_HOST_DEVICE auto sparse(const StoredFieldReferenceGPUData &field, vec3 p)
{
  const auto &grid =
      *static_cast<const nanovdb::NanoGrid<BuildT> *>(field.data);
  CachedNanoFetch<BuildT> fetch{grid.getAccessor()};
  return sparseWithCache<BuildT>(field, p, fetch);
}
#ifdef __CUDA_ARCH__
template <typename BuildT>
VISRTX_DEVICE VISRTX_NOINLINE float pointSparse(
    const StoredFieldReferenceGPUData &field, vec3 p)
{
  PointTextureFetch fetch{field.previewPointTexture, field.indexMin};
  return sparseWithCache<BuildT>(field, p, fetch);
}
#endif
template <typename BuildT>
VISRTX_HOST_DEVICE auto sparseMaybePoint(
    const StoredFieldReferenceGPUData &field, vec3 p)
{
#ifdef __CUDA_ARCH__
  if (field.previewPointTexture)
    return pointSparse<BuildT>(field, p);
#endif
  return sparse<BuildT>(field, p);
}

template <typename BuildT>
struct DensitySample
{
  const StoredFieldReferenceGPUData &field;
  CachedNanoFetch<BuildT> fetch;
  VISRTX_HOST_DEVICE float operator()(vec3 p);
};
template <typename BuildT>
VISRTX_HOST_DEVICE float DensitySample<BuildT>::operator()(vec3 p)
{
  return sparseWithCache<BuildT>(field, p, fetch);
}
template <typename Fetch>
VISRTX_HOST_DEVICE uint32_t constantTileDimension(
    Fetch &, const nanovdb::Coord &)
{
  return 1;
}
template <typename BuildT>
VISRTX_HOST_DEVICE uint32_t constantTileDimension(
    CachedNanoFetch<BuildT> &fetch, const nanovdb::Coord &coord)
{
  const auto dim = fetch.accessor.template get<nanovdb::GetDim<BuildT>>(coord);
  return dim == 0 ? 4096 : dim; // absent root tile: constant background
}
#ifdef __CUDACC__
VISRTX_DEVICE VISRTX_NOINLINE float textureDensityIntegral(
    const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &query);
#endif
#ifdef __CUDA_ARCH__
VISRTX_DEVICE float textureIntegralGeneral(
    const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &q,
    const vec3 &indexOrigin,
    const vec3 &indexDirection)
{
  float integral = 0.f, t = q.lower;
  while (t < q.upper) {
    const float cell = floorf((t - q.anchor) / q.step);
    float next = q.anchor + (cell + 1.f) * q.step;
    if (!(next > t))
      next = q.anchor + (cell + 2.f) * q.step;
    if (!(next > t) || next > q.upper)
      next = q.upper;
    const float midpoint = t + (next - t) * .5f;
    const vec3 p(q.origin[0] + midpoint * q.direction[0],
        q.origin[1] + midpoint * q.direction[1],
        q.origin[2] + midpoint * q.direction[2]);
    if (contains(q.volumeDomain, p)) {
      float value = q.outside;
      if (contains(q.fieldDomain, p))
        value = tex3D<float>(field.previewTexture,
            indexOrigin.x + midpoint * indexDirection.x
                - float(field.indexMin.x) + .5f,
            indexOrigin.y + midpoint * indexDirection.y
                - float(field.indexMin.y) + .5f,
            indexOrigin.z + midpoint * indexDirection.z
                - float(field.indexMin.z) + .5f);
      integral += (q.scale * value) * (next - t);
    }
    t = next;
  }
  return integral;
}
VISRTX_DEVICE float textureIntegral(const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &q,
    const vec3 &indexOrigin,
    const vec3 &indexDirection)
{
  // Requests are already split at every volume/field-domain endpoint. Domain
  // membership is constant over this span; do not repeat slab tests per sample.
  const float middle = q.lower + (q.upper - q.lower) * .5f;
  const vec3 p(q.origin[0] + middle * q.direction[0],
      q.origin[1] + middle * q.direction[1],
      q.origin[2] + middle * q.direction[2]);
  if (!contains(q.volumeDomain, p))
    return 0.f;
  if (!contains(q.fieldDomain, p))
    return q.scale * q.outside * (q.upper - q.lower);
  const float startIndex = floorf((q.lower - q.anchor) / q.step) + 1.f;
  const float endIndex = floorf((q.upper - q.anchor) / q.step);
  if (!(startIndex >= -0x1p20f && endIndex <= 0x1p20f))
    return textureIntegralGeneral(field, q, indexOrigin, indexDirection);
  auto sample = [&](float t) {
    return tex3D<float>(field.previewTexture,
        indexOrigin.x + t * indexDirection.x - float(field.indexMin.x) + .5f,
        indexOrigin.y + t * indexDirection.y - float(field.indexMin.y) + .5f,
        indexOrigin.z + t * indexDirection.z - float(field.indexMin.z) + .5f);
  };
  int first = int(startIndex), last = int(endIndex);
  float boundary = fmaf(float(first), q.step, q.anchor);
  if (!(boundary > q.lower))
    boundary = fmaf(float(++first), q.step, q.anchor);
  if (boundary >= q.upper)
    return q.scale * sample(middle) * (q.upper - q.lower);
  float integral =
      sample(q.lower + (boundary - q.lower) * .5f) * (boundary - q.lower);
  float end = fmaf(float(last), q.step, q.anchor);
  if (end > q.upper)
    end = fmaf(float(--last), q.step, q.anchor);
  if (last < first) {
    last = first;
    end = boundary;
  }
  vec3 base, slope;
  for (int c = 0; c < 3; ++c) {
    base[c] = indexOrigin[c] + (q.anchor + .5f * q.step) * indexDirection[c]
        - float(field.indexMin[c]) + .5f;
    slope[c] = q.step * indexDirection[c];
  }
  auto at = [&](int i) {
    return tex3D<float>(field.previewTexture,
        fmaf(float(i), slope.x, base.x),
        fmaf(float(i), slope.y, base.y),
        fmaf(float(i), slope.z, base.z));
  };
  float sum = 0.f;
  for (int i = first; i < last; ++i)
    sum += at(i);
  integral += sum * q.step;
  if (end < q.upper)
    integral += sample(end + (q.upper - end) * .5f) * (q.upper - end);
  return q.scale * integral;
}
VISRTX_DEVICE VISRTX_NOINLINE float textureDensityIntegral(
    const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &query)
{
  vec3 origin, direction;
  if (field.sparse) {
    const auto &map = static_cast<const nanovdb::GridData *>(field.data)->mMap;
    const auto o = map.applyInverseMapF(
        nanovdb::Vec3f(query.origin[0], query.origin[1], query.origin[2]));
    const auto d = map.applyInverseJacobianF(nanovdb::Vec3f(
        query.direction[0], query.direction[1], query.direction[2]));
    for (int c = 0; c < 3; ++c) {
      const float first = float(field.indexMin[c]);
      const float dims = float(field.indexMax[c]) - first + 1.f;
      const float factor = field.cellCentered ? 1.f : (dims - 1.f) / dims;
      origin[c] =
          (o[c] - first) * factor + first - (field.cellCentered ? .5f : 0.f);
      direction[c] = d[c] * factor;
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      origin[c] = (query.origin[c] - field.origin[c]) / field.spacing[c]
          - (field.cellCentered ? .5f : 0.f);
      direction[c] = query.direction[c] / field.spacing[c];
    }
  }
  return textureIntegral(field, query, origin, direction);
}
#endif

template <typename Fetch>
VISRTX_HOST_DEVICE float regularIntegral(
    const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &q,
    const vec3 &indexOrigin,
    const vec3 &indexDirection,
    Fetch &fetch)
{
#ifdef __CUDA_ARCH__
  if (field.previewTexture && q.allowTexture)
    return textureIntegral(field, q, indexOrigin, indexDirection);
#endif
  float integral = 0.f, t = q.lower;
  while (t < q.upper) {
    const float cell = floorf((t - q.anchor) / q.step);
    float next = q.anchor + (cell + 1.f) * q.step;
    if (!(next > t))
      next = q.anchor + (cell + 2.f) * q.step;
    if (!(next > t) || next > q.upper)
      next = q.upper;
    const float midpoint = t + (next - t) * .5f;
    const vec3 p(q.origin[0] + midpoint * q.direction[0],
        q.origin[1] + midpoint * q.direction[1],
        q.origin[2] + midpoint * q.direction[2]);
    if (contains(q.volumeDomain, p)) {
      float value = q.outside;
      if (contains(q.fieldDomain, p)) {
        float x[3];
        int lo[3];
        bool interior = true;
        for (int c = 0; c < 3; ++c) {
          x[c] = indexOrigin[c] + midpoint * indexDirection[c];
          interior &= x[c] >= float(field.indexMin[c])
              && x[c] <= float(field.indexMax[c]);
          lo[c] = int(floorf(coordinate(
              x[c], float(field.indexMin[c]), float(field.indexMax[c]))));
        }
        const uint32_t dim = interior
            ? constantTileDimension(fetch, nanovdb::Coord(lo[0], lo[1], lo[2]))
            : 1;
        int tile[3];
        bool constant = dim > 1;
        for (int c = 0; c < 3; ++c) {
          tile[c] = lo[c] & ~int(dim - 1);
          constant &= lo[c] < tile[c] + int(dim) - 1;
        }
        value = constant ? fetch(lo[0], lo[1], lo[2]) : filter(field, x, fetch);
        // Only density-integral consumers use this skip. A constant zero tile
        // is not inferred from its active mask or active-only min/max stats.
        if (constant && value == 0.f) {
          float exit = q.upper;
          bool safe = true;
          for (int c = 0; c < 3; ++c) {
            // A half-voxel inset is a conservative acceleration halo, not an
            // interval epsilon: skipped quadrature cells remain inside the
            // all-zero interpolation footprint.
            const float lower = float(tile[c]) + .5f,
                        upper = float(tile[c] + int(dim)) - 1.5f;
            safe &= x[c] >= lower && x[c] <= upper;
            if (indexDirection[c] > 0.f)
              exit = fminf(exit, midpoint + (upper - x[c]) / indexDirection[c]);
            if (indexDirection[c] < 0.f)
              exit = fminf(exit, midpoint + (lower - x[c]) / indexDirection[c]);
          }
          if (safe && exit > next) {
            const float index = floorf((exit - q.anchor) / q.step);
            float boundary = q.anchor + index * q.step;
            if (boundary > exit)
              boundary = q.anchor + (index - 1.f) * q.step;
            if (exit == q.upper)
              boundary = q.upper;
            if (boundary > next)
              next = boundary;
          }
        }
      }
      integral += (q.scale * value) * (next - t);
    }
    t = next;
  }
  return integral;
}
template <typename BuildT>
VISRTX_HOST_DEVICE float sparseIntegral(
    const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &query)
{
  const auto &grid =
      *static_cast<const nanovdb::NanoGrid<BuildT> *>(field.data);
  if (!field.rectilinear) {
    const auto origin = grid.worldToIndexF(
        nanovdb::Vec3f(query.origin[0], query.origin[1], query.origin[2]));
    const auto direction = grid.applyInverseJacobianF(nanovdb::Vec3f(
        query.direction[0], query.direction[1], query.direction[2]));
    vec3 indexOrigin, indexDirection;
    for (int c = 0; c < 3; ++c) {
      const float first = float(field.indexMin[c]);
      const float dims = float(field.indexMax[c]) - first + 1.f;
      const float factor = field.cellCentered ? 1.f : (dims - 1.f) / dims;
      indexOrigin[c] = (origin[c] - first) * factor + first
          - (field.cellCentered ? .5f : 0.f);
      indexDirection[c] = direction[c] * factor;
    }
    CachedNanoFetch<BuildT> fetch{grid.getAccessor()};
    return regularIntegral(field, query, indexOrigin, indexDirection, fetch);
  }
  DensitySample<BuildT> sample{field, {grid.getAccessor()}};
  return integratePreviewDensity(query, sample);
}
struct DenseDensitySample
{
  const StoredFieldReferenceGPUData &field;
  VISRTX_HOST_DEVICE float operator()(vec3 p);
};
VISRTX_HOST_DEVICE float DenseDensitySample::operator()(vec3 p)
{
  return dense<float>(field, p);
}
VISRTX_HOST_DEVICE VISRTX_NOINLINE float densityIntegral(
    const StoredFieldReferenceGPUData &field,
    const PreviewDensityIntegral &query)
{
  if (!field.sparse) {
    if (!field.rectilinear) {
      vec3 origin, direction;
      for (int c = 0; c < 3; ++c) {
        origin[c] = (query.origin[c] - field.origin[c]) / field.spacing[c]
            - (field.cellCentered ? .5f : 0.f);
        direction[c] = query.direction[c] / field.spacing[c];
      }
      optical_reference::DenseReferenceFetch<float> fetch{field};
      return regularIntegral(field, query, origin, direction, fetch);
    }
    DenseDensitySample sample{field};
    return integratePreviewDensity(query, sample);
  }
  switch (field.gridType) {
  case nanovdb::GridType::Float:
    return sparseIntegral<float>(field, query);
  case nanovdb::GridType::Fp4:
    return sparseIntegral<nanovdb::Fp4>(field, query);
  case nanovdb::GridType::Fp8:
    return sparseIntegral<nanovdb::Fp8>(field, query);
  case nanovdb::GridType::Fp16:
    return sparseIntegral<nanovdb::Fp16>(field, query);
  case nanovdb::GridType::FpN:
    return sparseIntegral<nanovdb::FpN>(field, query);
  default:
    return 0.f;
  }
}
} // namespace preview_math

struct StoredFieldPreviewSampler
{
  const StoredFieldReferenceGPUData *fields;
  template <typename T>
  VISRTX_HOST_DEVICE VISRTX_NOINLINE T sample(
      int32_t index, const vec3 &p) const;
};
template <typename T>
VISRTX_HOST_DEVICE VISRTX_NOINLINE T StoredFieldPreviewSampler::sample(
    int32_t index, const vec3 &p) const
{
  const auto &field = fields[index];
  if (field.customProvider)
    return StoredFieldReferenceSampler{fields}.sample<T>(index, p);
  if (!field.sparse)
    return preview_math::dense<T>(field, p);
  if constexpr (std::is_same_v<T, vec3>)
    return preview_math::sparse<nanovdb::Vec3f>(field, p);
  else {
    switch (field.gridType) {
    case nanovdb::GridType::Float:
      return preview_math::sparseMaybePoint<float>(field, p);
    case nanovdb::GridType::Fp4:
      return preview_math::sparseMaybePoint<nanovdb::Fp4>(field, p);
    case nanovdb::GridType::Fp8:
      return preview_math::sparseMaybePoint<nanovdb::Fp8>(field, p);
    case nanovdb::GridType::Fp16:
      return preview_math::sparseMaybePoint<nanovdb::Fp16>(field, p);
    case nanovdb::GridType::FpN:
      return preview_math::sparseMaybePoint<nanovdb::FpN>(field, p);
    default:
      return T(0.f);
    }
  }
}
} // namespace visrtx
