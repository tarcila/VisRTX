// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <stdexcept>
#include "PreviewFieldTexture.h"
#include "StoredFieldReferenceSampler.h"
namespace visrtx {
namespace {
void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Temporary
{
  float *values{};
  ~Temporary();
};
Temporary::~Temporary()
{
  cudaFree(values);
}
template <typename BuildT>
__global__ void expand(
    StoredFieldReferenceGPUData field, ivec3 dims, float *values)
{
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(dims.x) * dims.y * dims.z)
    return;
  const auto &grid =
      *static_cast<const nanovdb::NanoGrid<BuildT> *>(field.data);
  const int x = int(i % dims.x) + field.indexMin.x;
  const int y = int((i / dims.x) % dims.y) + field.indexMin.y;
  const int z = int(i / (size_t(dims.x) * dims.y)) + field.indexMin.z;
  values[i] = optical_reference::NanoReferenceFetch<BuildT>{grid}(x, y, z);
}
} // namespace
PreviewFieldTexture::~PreviewFieldTexture()
{
  if (m_pointTexture)
    cudaDestroyTextureObject(m_pointTexture);
  if (m_texture)
    cudaDestroyTextureObject(m_texture);
  if (m_array)
    cudaFreeArray(m_array);
}
cudaTextureObject_t PreviewFieldTexture::texture() const
{
  return m_texture;
}
cudaTextureObject_t PreviewFieldTexture::pointTexture() const
{
  return m_pointTexture;
}
bool PreviewFieldTexture::prepare(const StoredFieldReferenceGPUData &field)
{
  if (m_attempted)
    return m_texture != 0;
  m_attempted = true;
  ivec3 dims;
  for (int c = 0; c < 3; ++c) {
    const int64_t count =
        int64_t(field.indexMax[c]) - int64_t(field.indexMin[c]) + 1;
    if (count <= 0 || count > 512)
      return false;
    dims[c] = int(count);
  }
  if (field.rectilinear || !field.linear
      || size_t(dims.x) * dims.y * dims.z > (size_t(1) << 24)
      || glm::max(dims.x, glm::max(dims.y, dims.z)) < 32)
    return false;
  if (field.sparse && field.gridType != nanovdb::GridType::Float
      && field.gridType != nanovdb::GridType::Fp4
      && field.gridType != nanovdb::GridType::Fp8
      && field.gridType != nanovdb::GridType::Fp16
      && field.gridType != nanovdb::GridType::FpN)
    return false;
  auto descriptor = cudaCreateChannelDesc<float>();
  auto error = cudaMalloc3DArray(
      &m_array, &descriptor, make_cudaExtent(dims.x, dims.y, dims.z));
  if (error == cudaErrorMemoryAllocation) {
    cudaGetLastError();
    return false;
  }
  checked(error);
  Temporary buffer;
  const float *source = static_cast<const float *>(field.data);
  if (field.sparse) {
    const size_t count = size_t(dims.x) * dims.y * dims.z;
    error = cudaMalloc(&buffer.values, count * sizeof(float));
    if (error == cudaErrorMemoryAllocation) {
      cudaGetLastError();
      cudaFreeArray(m_array);
      m_array = nullptr;
      return false;
    }
    checked(error);
    switch (field.gridType) {
    case nanovdb::GridType::Float:
      expand<float><<<(count + 255) / 256, 256>>>(field, dims, buffer.values);
      break;
    case nanovdb::GridType::Fp4:
      expand<nanovdb::Fp4>
          <<<(count + 255) / 256, 256>>>(field, dims, buffer.values);
      break;
    case nanovdb::GridType::Fp8:
      expand<nanovdb::Fp8>
          <<<(count + 255) / 256, 256>>>(field, dims, buffer.values);
      break;
    case nanovdb::GridType::Fp16:
      expand<nanovdb::Fp16>
          <<<(count + 255) / 256, 256>>>(field, dims, buffer.values);
      break;
    case nanovdb::GridType::FpN:
      expand<nanovdb::FpN>
          <<<(count + 255) / 256, 256>>>(field, dims, buffer.values);
      break;
    default:
      break;
    }
    checked(cudaGetLastError());
    source = buffer.values;
  }
  cudaMemcpy3DParms copy{};
  copy.srcPtr = make_cudaPitchedPtr(const_cast<float *>(source),
      size_t(dims.x) * sizeof(float),
      dims.x,
      dims.y);
  copy.dstArray = m_array;
  copy.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  copy.kind = cudaMemcpyDeviceToDevice;
  checked(cudaMemcpy3D(&copy));
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = m_array;
  cudaTextureDesc texture{};
  texture.normalizedCoords = 0;
  texture.filterMode = cudaFilterModeLinear;
  texture.readMode = cudaReadModeElementType;
  for (int c = 0; c < 3; ++c)
    texture.addressMode[c] = cudaAddressModeClamp;
  checked(cudaCreateTextureObject(&m_texture, &resource, &texture, nullptr));
  // The second object shares the immutable array. Point fetches retain full
  // scalar endpoints; optical interpolation still uses software FP32 weights.
  texture.filterMode = cudaFilterModePoint;
  error =
      cudaCreateTextureObject(&m_pointTexture, &resource, &texture, nullptr);
  if (error == cudaErrorMemoryAllocation) {
    cudaGetLastError();
    m_pointTexture = 0; // optional optimization; ordinary sampler remains
  } else
    checked(error);
  return true;
}
} // namespace visrtx
