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

std::unique_ptr<StoredFieldReference> StoredFieldReference::customRaw(
    const TypedCustomDescriptor &request,
    const void *payload,
    size_t bytes,
    size_t alignment,
    FieldValueType type,
    const optical_reference::RawRange<float> *scalar,
    const optical_reference::RawRange<vec3> *rgb,
    box3 domain,
    optical_reference::FieldRevision key,
    cudaStream_t stream,
    std::shared_ptr<const void> resources,
    TypedCustomError *error)
{
  auto fail = [&](TypedCustomError e) -> std::unique_ptr<StoredFieldReference> {
    if (error)
      *error = e;
    return {};
  };
  if (request.structSize < sizeof(TypedCustomDescriptor))
    return fail(TypedCustomError::DescriptorSize);
  if (request.version != TYPED_CUSTOM_CONTRACT)
    return fail(TypedCustomError::Version);
  const auto *compiled = compiledTypedCustomProvider(request.provider);
  const auto status = validateTypedCustom(request,
      compiled,
      bytes,
      alignment,
      type,
      scalar || rgb,
      bool(resources));
  if (status != TypedCustomError::None)
    return fail(status);
  if (!payload)
    return fail(TypedCustomError::MissingPayload);
  for (int c = 0; c < 3; ++c)
    if (!std::isfinite(domain.lower[c]) || !std::isfinite(domain.upper[c]))
      return fail(TypedCustomError::InvalidDomain);
  const bool empty = scalar ? scalar->empty : rgb->empty;
  if (empty && optical_reference::nonemptyDomain(domain))
    return fail(TypedCustomError::InvalidRange);
  if (!empty)
    for (int c = 0; c < (scalar ? 1 : 3); ++c) {
      const float lo = scalar ? scalar->lower : rgb->lower[c];
      const float hi = scalar ? scalar->upper : rgb->upper[c];
      if (!std::isfinite(lo) || !std::isfinite(hi) || hi < lo)
        return fail(TypedCustomError::InvalidRange);
    }
  // Providers publish ready resources or enqueue their uploads on this stream.
  // Join before CUDA guide preparation can consume them on the default stream.
  const auto ready = cudaStreamSynchronize(stream);
  if (ready != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(ready));
  auto result = std::make_unique<StoredFieldReference>();
  copyChecked(result->m_values, payload, bytes);
  finishUploads();
  result->m_gpu.data = result->m_values.ptr();
  result->m_gpu.customProvider = compiled->provider;
  result->m_gpu.cudaSampling = (compiled->capabilities & TypedCustomCuda) != 0;
  result->m_descriptor = {key, type, domain};
  if (scalar)
    result->m_scalarRange = *scalar;
  else
    result->m_rgbRange = *rgb;
  result->m_previewMap = false;
  result->m_customStream = stream;
  result->m_customResources = std::move(resources);
  if (error)
    *error = TypedCustomError::None;
  return result;
}

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
  // Metadata-only selection, not a scan or a new validity requirement.
  for (int i = 0; i < 9; ++i) {
    const double x = std::abs(data->mMap.mInvMatD[i]);
    result->m_previewMap &=
        x <= 0x1p20 && (x == 0. || x >= std::numeric_limits<float>::min());
    result->m_previewMap &= std::isfinite(data->mMap.mInvMatF[i]);
  }
  double inverseNorm = 0., forwardNorm = 0.;
  for (int row = 0; row < 3; ++row) {
    double a = 0., b = 0.;
    for (int col = 0; col < 3; ++col) {
      a += std::abs(data->mMap.mInvMatD[3 * row + col]);
      b += std::abs(data->mMap.mMatD[3 * row + col]);
    }
    inverseNorm = std::max(inverseNorm, a);
    forwardNorm = std::max(forwardNorm, b);
  }
  result->m_previewMap &=
      inverseNorm >= 0x1p-20 && inverseNorm * forwardNorm <= 64.;
  for (int c = 0; c < 3; ++c)
    result->m_previewMap &= std::abs(data->mMap.mVecD[c]) <= 0x1p16;
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

bool StoredFieldReference::supportsFloatPreview() const
{
  if (!m_previewMap)
    return false;
  for (int c = 0; c < 3; ++c) {
    if (std::abs(double(m_gpu.indexMin[c])) > 0x1p18
        || std::abs(double(m_gpu.indexMax[c])) > 0x1p18)
      return false;
    const double extent = double(m_descriptor.domain.upper[c])
        - double(m_descriptor.domain.lower[c]);
    const double magnitude =
        std::max(std::abs(double(m_descriptor.domain.lower[c])),
            std::abs(double(m_descriptor.domain.upper[c])));
    if (!(extent >= 0x1p-20 && extent <= 0x1p20) || magnitude > extent * 0x1p10)
      return false;
    if (!m_gpu.sparse && !m_gpu.rectilinear
        && !(double(m_gpu.spacing[c]) >= 0x1p-20
            && double(m_gpu.spacing[c]) <= 0x1p20))
      return false;
    if (m_gpu.rectilinear) {
      const double span =
          double(m_gpu.axisUpper[c]) - double(m_gpu.axisLower[c]);
      if (!(span >= 0x1p-20 && span <= 0x1p20))
        return false;
    }
  }
  return true;
}

} // namespace visrtx
