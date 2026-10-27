// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "StoredFieldReferenceSampler.h"
#include "StoredFieldTypes.h"
#include "utility/ScopedReferenceRounding.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace visrtx {
namespace {
namespace ref = optical_reference;

void finishUploads()
{
  // A synchronous host-to-device copy can stage pageable memory before DMA is
  // complete. Join the actual producer stream before publishing a snapshot to
  // a renderer using a different (possibly nonblocking) stream.
  const auto error = cudaStreamSynchronize(nullptr);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

void copyChecked(DeviceBuffer &buffer, const void *source, size_t bytes)
{
  buffer.reserve(bytes);
  const auto error = cudaMemcpy(buffer.ptr(), source, bytes, cudaMemcpyDefault);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

template <typename T>
T load(const std::byte *source, size_t i)
{
  T value;
  std::memcpy(&value, source + i * sizeof(T), sizeof(T));
  return value;
}

size_t elementBytes(ANARIDataType type)
{
  switch (type) {
  case ANARI_FIXED8:
  case ANARI_UFIXED8:
    return 1;
  case ANARI_FIXED16:
  case ANARI_UFIXED16:
    return 2;
  case ANARI_FLOAT32:
    return 4;
  case ANARI_FLOAT64:
    return 8;
  case ANARI_FLOAT32_VEC3:
    return sizeof(vec3);
  default:
    return 0;
  }
}

float decodeDense(const std::byte *source, size_t i, ANARIDataType type)
{
  switch (type) {
  case ANARI_FLOAT32:
    return load<float>(source, i);
  case ANARI_FLOAT64:
    return ref::narrow(load<double>(source, i));
  case ANARI_FIXED8:
    return ref::narrow(
        std::max(-1., ref::divide(double(load<int8_t>(source, i)), 127.)));
  case ANARI_FIXED16:
    return ref::narrow(
        std::max(-1., ref::divide(double(load<int16_t>(source, i)), 32767.)));
  case ANARI_UFIXED8:
    return ref::narrow(ref::divide(double(load<uint8_t>(source, i)), 255.));
  case ANARI_UFIXED16:
    return ref::narrow(ref::divide(double(load<uint16_t>(source, i)), 65535.));
  default:
    return 0.f; // validated type before conversion
  }
}

template <typename BuildT, typename Node, typename T>
void includeNode(const Node &node, ref::RawRange<T> &range)
{
  if constexpr (Node::LEVEL == 0) {
    for (uint32_t i = 0; i < 512; ++i)
      range =
          ref::extendRange(range, ref::decodeReferenceLeaf<BuildT>(node, i));
  } else {
    for (auto it = node.cbeginValueAll(); it; ++it)
      range = ref::extendRange(range, ref::canonicalNano(*it));
    for (auto it = node.cbeginChild(); it; ++it)
      includeNode<BuildT>(*it, range);
  }
}

template <typename BuildT>
auto sparseRange(const void *data)
{
  const auto &grid = *static_cast<const nanovdb::NanoGrid<BuildT> *>(data);
  using T = decltype(ref::canonicalNano(grid.tree().background()));
  ref::RawRange<T> range;
  range = ref::extendRange(range, ref::canonicalNano(grid.tree().background()));
  includeNode<BuildT>(grid.tree().root(), range);
  return range;
}

bool validKeyDomain(ref::FieldRevision key, const box3 &domain)
{
  return key.identity && key.revision && key.contract
      && ref::validDomainMetadata(domain);
}

} // namespace

std::unique_ptr<StoredFieldReference> StoredFieldReference::dense(
    const void *data,
    ANARIDataType type,
    uvec3 dims,
    vec3 origin,
    vec3 spacing,
    bool cellCentered,
    bool linear,
    box3 domain,
    ref::FieldRevision key,
    const ReferenceAxes *axes)
{
  const ScopedReferenceRounding rounding;
  const size_t bytesPerElement = elementBytes(type);
  if (!data || !bytesPerElement || !validKeyDomain(key, domain))
    return {};
  size_t count = 1;
  for (int c = 0; c < 3; ++c) {
    if (!dims[c] || dims[c] > uint32_t(std::numeric_limits<int>::max())
        || count > std::numeric_limits<size_t>::max() / dims[c])
      return {};
    count *= dims[c];
    if (!axes
        && !(ref::promote(spacing[c]) > 0.
            && ref::promote(spacing[c])
                <= double(std::numeric_limits<float>::max())))
      return {};
    if (axes && (axes->count[c] != size_t(dims[c]) + (cellCentered ? 1 : 0)))
      return {};
  }
  if (count > std::numeric_limits<size_t>::max()
          / std::max(bytesPerElement, sizeof(vec3)))
    return {};
  auto result = std::make_unique<StoredFieldReference>();
  auto &view = result->m_gpu;
  view.origin = origin;
  view.spacing = spacing;
  view.indexMax = ivec3(dims) - ivec3(1);
  view.cellCentered = cellCentered;
  view.linear = linear;
  view.rectilinear = axes != nullptr;
  result->m_descriptor = {key,
      type == ANARI_FLOAT32_VEC3 ? FieldValueType::RGB : FieldValueType::Scalar,
      domain};
  std::vector<std::byte> raw(count * bytesPerElement);
  const auto error =
      cudaMemcpy(raw.data(), data, raw.size(), cudaMemcpyDefault);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  if (type == ANARI_FLOAT32_VEC3) {
    std::vector<vec3> canonical(count);
    for (size_t i = 0; i < count; ++i)
      canonical[i] = load<vec3>(raw.data(), i);
    result->m_rgbRange =
        ref::canonicalRange(canonical.data(), canonical.size());
    copyChecked(
        result->m_values, canonical.data(), canonical.size() * sizeof(vec3));
  } else {
    std::vector<float> canonical(count);
    for (size_t i = 0; i < count; ++i)
      canonical[i] = decodeDense(raw.data(), i, type);
    result->m_scalarRange =
        ref::canonicalRange(canonical.data(), canonical.size());
    copyChecked(
        result->m_values, canonical.data(), canonical.size() * sizeof(float));
  }
  view.data = result->m_values.ptr();
  if (axes && !result->buildAxes(*axes))
    return {};
  finishUploads();
  return result;
}

bool StoredFieldReference::buildAxes(const ReferenceAxes &axes)
{
  constexpr size_t count = 8192;
  for (int axis = 0; axis < 3; ++axis) {
    const size_t n = axes.count[axis];
    const float *values = axes.values[axis];
    if (!values || n < 2)
      return false;
    const double first = ref::promote(values[0]);
    const double last = ref::promote(values[n - 1]);
    if (!(first < last))
      return false;
    // Axis ordering is the existing input contract, already handled by legacy
    // finalization. This constructs the LUT, not another coordinate validation
    // scan.
    std::vector<float> lut(count);
    lut.front() = 0.f;
    lut.back() = 1.f;
    size_t segment = 0;
    const double extent = ref::subtract(last, first);
    for (size_t i = 1; i + 1 < count; ++i) {
      const double x = ref::add(first,
          ref::multiply(extent, ref::divide(double(i), double(count - 1))));
      while (segment + 2 < n && x > ref::promote(values[segment + 1]))
        ++segment;
      const double fraction =
          ref::divide(ref::subtract(x, ref::promote(values[segment])),
              ref::subtract(ref::promote(values[segment + 1]),
                  ref::promote(values[segment])));
      lut[i] = ref::narrow(
          ref::divide(ref::add(double(segment), fraction), double(n - 1)));
    }
    copyChecked(m_axes[axis], lut.data(), count * sizeof(float));
    m_gpu.axisLUT[axis] = m_axes[axis].ptrAs<float>();
    m_gpu.axisLower[axis] = values[0];
    m_gpu.axisUpper[axis] = values[n - 1];
  }
  return true;
}

std::unique_ptr<StoredFieldReference> StoredFieldReference::sparse(
    const void *hostGrid,
    size_t bytes,
    bool cellCentered,
    bool linear,
    box3 domain,
    ref::FieldRevision key,
    const ReferenceAxes *axes)
{
  const ScopedReferenceRounding rounding;
  if (!hostGrid || bytes < sizeof(nanovdb::GridMetaData)
      || !validKeyDomain(key, domain))
    return {};
  const auto *data = static_cast<const nanovdb::GridData *>(hostGrid);
  if (!data->isValid() || !supportedNanoVDBFieldType(data->mGridType)
      || data->mGridSize > bytes
      || data->mGridSize < sizeof(nanovdb::GridMetaData))
    return {};
  const nanovdb::GridMetaData metadata(data);
  if (metadata.gridCount() != 1 || metadata.indexBBox().empty())
    return {};
  auto result = std::make_unique<StoredFieldReference>();
  auto &view = result->m_gpu;
  view.sparse = true;
  view.rectilinear = axes != nullptr;
  view.gridType = metadata.gridType();
  view.cellCentered = cellCentered;
  view.linear = linear;
  for (int c = 0; c < 3; ++c) {
    view.indexMin[c] = metadata.indexBBox().min()[c];
    view.indexMax[c] = metadata.indexBBox().max()[c];
    const uint64_t dims =
        uint64_t(int64_t(view.indexMax[c]) - view.indexMin[c] + 1);
    if (axes && axes->count[c] != dims + (cellCentered ? 1 : 0))
      return {};
  }
  result->m_descriptor = {key,
      view.gridType == nanovdb::GridType::Vec3f ? FieldValueType::RGB
                                                : FieldValueType::Scalar,
      domain};
  switch (view.gridType) {
  case nanovdb::GridType::Float:
    result->m_scalarRange = sparseRange<float>(hostGrid);
    break;
  case nanovdb::GridType::Fp4:
    result->m_scalarRange = sparseRange<nanovdb::Fp4>(hostGrid);
    break;
  case nanovdb::GridType::Fp8:
    result->m_scalarRange = sparseRange<nanovdb::Fp8>(hostGrid);
    break;
  case nanovdb::GridType::Fp16:
    result->m_scalarRange = sparseRange<nanovdb::Fp16>(hostGrid);
    break;
  case nanovdb::GridType::FpN:
    result->m_scalarRange = sparseRange<nanovdb::FpN>(hostGrid);
    break;
  case nanovdb::GridType::Vec3f:
    result->m_rgbRange = sparseRange<nanovdb::Vec3f>(hostGrid);
    break;
  default:
    return {};
  }
  copyChecked(result->m_values, hostGrid, data->mGridSize);
  view.data = result->m_values.ptr();
  if (axes && !result->buildAxes(*axes))
    return {};
  finishUploads();
  return result;
}

} // namespace visrtx
