// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdexcept>
#include "StoredFieldReference.h"
#include "TypedCustomFieldDispatch.h"

namespace visrtx {
namespace optical_reference {

VISRTX_HOST_DEVICE float canonicalNano(float value)
{
  return value;
}
VISRTX_HOST_DEVICE vec3 canonicalNano(const nanovdb::Vec3f &value)
{
  return vec3(value[0], value[1], value[2]);
}

// These offsets/bit widths follow the vendored NanoVDB leaf layout. The same
// routine decodes every represented leaf endpoint for host ranges and sampling.
// No active-mask test: inactive stored values are real interpolated endpoints.
template <typename BuildT>
VISRTX_HOST_DEVICE auto decodeReferenceLeaf(
    const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t i)
{
  if constexpr (std::is_same_v<BuildT, float>
      || std::is_same_v<BuildT, nanovdb::Vec3f>) {
    return canonicalNano(leaf.getValue(i));
  } else {
    const auto *d = leaf.data();
    uint32_t code;
    if constexpr (std::is_same_v<BuildT, nanovdb::Fp4>)
      code = (d->mCode[i >> 1] >> ((i & 1) * 4)) & 15;
    else if constexpr (std::is_same_v<BuildT, nanovdb::Fp8>
        || std::is_same_v<BuildT, nanovdb::Fp16>)
      code = d->mCode[i];
    else {
      static_assert(std::is_same_v<BuildT, nanovdb::FpN>);
      const auto *values = reinterpret_cast<const uint8_t *>(d + 1);
      const uint32_t width = d->bitWidth();
      if (width == 16)
        code = reinterpret_cast<const uint16_t *>(values)[i];
      else
        code = (values[(i * width) >> 3] >> ((i * width) & 7))
            & ((1u << width) - 1);
    }
    return narrow(add(
        multiply(double(code), promote(d->mQuantum)), promote(d->mMinimum)));
  }
}

VISRTX_HOST_DEVICE double clampCoordinate(double x, double lo, double hi)
{
  return x < lo ? lo : (x > hi ? hi : x);
}

VISRTX_HOST_DEVICE float referenceLUT(const float *lut, double normalized)
{
  // Same normalized texture-center convention as the legacy 8192-texel LUT;
  // weights are now computed in double then RN32, not hardware 8-bit fractions.
  const double x =
      clampCoordinate(subtract(multiply(normalized, 8192.), .5), 0., 8191.);
  const int lo = int(x);
  const int hi = lo < 8191 ? lo + 1 : lo;
  return canonicalLerp(lut[lo], lut[hi], narrow(subtract(x, double(lo))));
}

template <typename T>
struct DenseReferenceFetch
{
  const StoredFieldReferenceGPUData &field;
  VISRTX_HOST_DEVICE T operator()(int x, int y, int z) const;
};

template <typename T>
VISRTX_HOST_DEVICE T DenseReferenceFetch<T>::operator()(
    int x, int y, int z) const
{
  const size_t nx = size_t(field.indexMax.x) + 1;
  const size_t ny = size_t(field.indexMax.y) + 1;
  return static_cast<const T *>(
      field.data)[size_t(x) + nx * (size_t(y) + ny * size_t(z))];
}

// Use NanoVDB's traversal operation seam so even the generated fallback code
// cannot invoke its fast-math quantized leaf decoder. Tile/background loads and
// the controlled leaf decode are selected in one traversal.
template <typename BuildT>
struct ReferenceNanoValue
{
  using Type =
      std::conditional_t<std::is_same_v<BuildT, nanovdb::Vec3f>, vec3, float>;
  static constexpr int LEVEL = 0;
  VISRTX_HOST_DEVICE static Type get(const nanovdb::NanoRoot<BuildT> &root);
  VISRTX_HOST_DEVICE static Type get(
      const typename nanovdb::NanoRoot<BuildT>::Tile &tile);
  VISRTX_HOST_DEVICE static Type get(
      const nanovdb::NanoUpper<BuildT> &node, uint32_t i);
  VISRTX_HOST_DEVICE static Type get(
      const nanovdb::NanoLower<BuildT> &node, uint32_t i);
  VISRTX_HOST_DEVICE static Type get(
      const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t i);
};

template <typename BuildT>
VISRTX_HOST_DEVICE typename ReferenceNanoValue<BuildT>::Type
ReferenceNanoValue<BuildT>::get(const nanovdb::NanoRoot<BuildT> &root)
{
  return canonicalNano(root.background());
}

template <typename BuildT>
VISRTX_HOST_DEVICE typename ReferenceNanoValue<BuildT>::Type
ReferenceNanoValue<BuildT>::get(
    const typename nanovdb::NanoRoot<BuildT>::Tile &tile)
{
  return canonicalNano(tile.value);
}

template <typename BuildT>
VISRTX_HOST_DEVICE typename ReferenceNanoValue<BuildT>::Type
ReferenceNanoValue<BuildT>::get(
    const nanovdb::NanoUpper<BuildT> &node, uint32_t i)
{
  return canonicalNano(node.data()->mTable[i].value);
}

template <typename BuildT>
VISRTX_HOST_DEVICE typename ReferenceNanoValue<BuildT>::Type
ReferenceNanoValue<BuildT>::get(
    const nanovdb::NanoLower<BuildT> &node, uint32_t i)
{
  return canonicalNano(node.data()->mTable[i].value);
}

template <typename BuildT>
VISRTX_HOST_DEVICE typename ReferenceNanoValue<BuildT>::Type
ReferenceNanoValue<BuildT>::get(
    const nanovdb::NanoLeaf<BuildT> &leaf, uint32_t i)
{
  return decodeReferenceLeaf<BuildT>(leaf, i);
}

template <typename BuildT>
struct NanoReferenceFetch
{
  typename nanovdb::NanoGrid<BuildT>::AccessorType accessor;
  VISRTX_HOST_DEVICE auto operator()(int x, int y, int z) const;
};

template <typename BuildT>
VISRTX_HOST_DEVICE auto NanoReferenceFetch<BuildT>::operator()(
    int x, int y, int z) const
{
  const nanovdb::Coord ijk(x, y, z);
  return accessor.template get<ReferenceNanoValue<BuildT>>(ijk);
}

template <typename Fetch>
VISRTX_HOST_DEVICE auto filterReference(
    const StoredFieldReferenceGPUData &field,
    const double (&coordinates)[3],
    Fetch fetch)
{
  int lo[3], hi[3];
  float w[3];
  for (int c = 0; c < 3; ++c) {
    const double x = clampCoordinate(
        coordinates[c], double(field.indexMin[c]), double(field.indexMax[c]));
    if (!field.linear) {
      lo[c] = int(floor(add(x, .5)));
      hi[c] = lo[c];
      w[c] = 0.f;
    } else {
      lo[c] = int(floor(x));
      hi[c] = lo[c] < field.indexMax[c] ? lo[c] + 1 : lo[c];
      w[c] = narrow(subtract(x, double(lo[c])));
    }
  }
  if (!field.linear)
    return fetch(lo[0], lo[1], lo[2]);
  const auto a = canonicalLerp(
      fetch(lo[0], lo[1], lo[2]), fetch(hi[0], lo[1], lo[2]), w[0]);
  const auto b = canonicalLerp(
      fetch(lo[0], hi[1], lo[2]), fetch(hi[0], hi[1], lo[2]), w[0]);
  const auto c = canonicalLerp(
      fetch(lo[0], lo[1], hi[2]), fetch(hi[0], lo[1], hi[2]), w[0]);
  const auto d = canonicalLerp(
      fetch(lo[0], hi[1], hi[2]), fetch(hi[0], hi[1], hi[2]), w[0]);
  return canonicalLerp(
      canonicalLerp(a, b, w[1]), canonicalLerp(c, d, w[1]), w[2]);
}

template <typename T>
VISRTX_HOST_DEVICE T sampleDenseReference(
    const StoredFieldReferenceGPUData &field, const vec3 &p)
{
  double x[3];
  for (int c = 0; c < 3; ++c) {
    if (field.rectilinear) {
      const double normalized = divide(
          subtract(promote(p[c]), promote(field.axisLower[c])),
          subtract(promote(field.axisUpper[c]), promote(field.axisLower[c])));
      const double segments =
          double(field.indexMax[c]) + (field.cellCentered ? 1. : 0.);
      x[c] = multiply(
          promote(referenceLUT(field.axisLUT[c], normalized)), segments);
    } else
      x[c] = divide(subtract(promote(p[c]), promote(field.origin[c])),
          promote(field.spacing[c]));
    if (field.cellCentered)
      x[c] = subtract(x[c], .5);
  }
  return filterReference(field, x, DenseReferenceFetch<T>{field});
}

template <typename BuildT>
VISRTX_HOST_DEVICE auto sampleNanoReference(
    const StoredFieldReferenceGPUData &field, const vec3 &p)
{
  const auto &grid =
      *static_cast<const nanovdb::NanoGrid<BuildT> *>(field.data);
  // Mapping is kept separate from the value enclosure. Double coordinates avoid
  // overflowing a float reciprocal before clamping to the represented index
  // box.
  const auto index = grid.worldToIndex(
      nanovdb::Vec3d(promote(p.x), promote(p.y), promote(p.z)));
  double x[3];
  for (int c = 0; c < 3; ++c) {
    const double first = double(field.indexMin[c]);
    const double dims = double(field.indexMax[c]) - first + 1.;
    const double relative = subtract(index[c], first);
    if (field.rectilinear) {
      const double span = dims > 1. ? dims - 1. : 1.;
      const double normalized =
          divide(relative, field.cellCentered ? span : dims);
      x[c] = add(
          multiply(promote(referenceLUT(field.axisLUT[c], normalized)), span),
          first);
    } else {
      x[c] =
          add(field.cellCentered ? relative
                                 : multiply(relative, divide(dims - 1., dims)),
              first);
    }
    if (field.cellCentered)
      x[c] = subtract(x[c], .5);
  }
  // Cache hierarchy traversal across the interpolation footprint, while
  // retaining the exact represented endpoint decoder and filtering graph.
  return filterReference(
      field, x, NanoReferenceFetch<BuildT>{grid.getAccessor()});
}

} // namespace optical_reference

struct StoredFieldReferenceSampler
{
  const StoredFieldReferenceGPUData *fields;
  template <typename T>
  VISRTX_HOST_DEVICE VISRTX_NOINLINE T sample(
      int32_t index, const vec3 &p) const;
};

template <typename T>
VISRTX_HOST_DEVICE VISRTX_NOINLINE T StoredFieldReferenceSampler::sample(
    int32_t index, const vec3 &p) const
{
  const auto &field = fields[index];
  if (field.customProvider) {
#if defined(__CUDA_ARCH__)
    return sampleTypedCustom<T>(field.customProvider, field.data, p);
#else
    throw std::logic_error(
        "typed custom sampling requires a declared GPU execution context");
#endif
  }
  if (!field.sparse)
    return optical_reference::sampleDenseReference<T>(field, p);
  if constexpr (std::is_same_v<T, vec3>) {
    return optical_reference::sampleNanoReference<nanovdb::Vec3f>(field, p);
  } else {
    static_assert(std::is_same_v<T, float>);
    switch (field.gridType) {
    case nanovdb::GridType::Float:
      return optical_reference::sampleNanoReference<float>(field, p);
    case nanovdb::GridType::Fp4:
      return optical_reference::sampleNanoReference<nanovdb::Fp4>(field, p);
    case nanovdb::GridType::Fp8:
      return optical_reference::sampleNanoReference<nanovdb::Fp8>(field, p);
    case nanovdb::GridType::Fp16:
      return optical_reference::sampleNanoReference<nanovdb::Fp16>(field, p);
    case nanovdb::GridType::FpN:
      return optical_reference::sampleNanoReference<nanovdb::FpN>(field, p);
    default:
      break;
    }
  }
  // Unreachable for a successfully prepared, arity-matched stored descriptor.
  return T(0.f);
}

} // namespace visrtx
