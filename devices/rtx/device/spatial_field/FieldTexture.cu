/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "FieldTexture.h"
#include "gpu/volumeOpticsReference.h"
#include "utility/DeviceBuffer.h"

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <limits>

namespace visrtx {
namespace {

struct ExpandRGB
{
  VISRTX_HOST_DEVICE float4 operator()(const float3 &v) const;
};

VISRTX_HOST_DEVICE float4 ExpandRGB::operator()(const float3 &v) const
{
  return make_float4(v.x, v.y, v.z, 1.f);
}

struct NarrowDouble
{
  VISRTX_HOST_DEVICE float operator()(double value) const;
};

VISRTX_HOST_DEVICE float NarrowDouble::operator()(double value) const
{
  return optical_reference::narrow(value);
}

} // namespace

cudaError_t uploadDoubleFieldArray(
    cudaArray_t &array, const void *data, const uvec3 &dims)
{
  if (!data || !dims.x || !dims.y || !dims.z)
    return cudaErrorInvalidValue;
  const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
  if (size_t(dims.x) > limit / dims.y
      || size_t(dims.x) * dims.y > limit / dims.z)
    return cudaErrorInvalidValue;
  const size_t count = size_t(dims.x) * dims.y * dims.z;
  DeviceBuffer source, converted;
  source.reserve(count * sizeof(double));
  converted.reserve(count * sizeof(float));
  auto error =
      cudaMemcpy(source.ptr(), data, count * sizeof(double), cudaMemcpyDefault);
  if (error != cudaSuccess)
    return error;
  auto begin = thrust::device_pointer_cast(source.ptrAs<double>());
  thrust::transform(begin,
      begin + count,
      thrust::device_pointer_cast(converted.ptrAs<float>()),
      NarrowDouble{});
  if (!array) {
    const auto desc = cudaCreateChannelDesc<float>();
    error = cudaMalloc3DArray(
        &array, &desc, make_cudaExtent(dims.x, dims.y, dims.z));
    if (error != cudaSuccess)
      return error;
  }
  cudaMemcpy3DParms copy{};
  copy.srcPtr = make_cudaPitchedPtr(
      converted.ptr(), size_t(dims.x) * sizeof(float), dims.x, dims.y);
  copy.dstArray = array;
  copy.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  copy.kind = cudaMemcpyDeviceToDevice;
  return cudaMemcpy3D(&copy);
}

cudaError_t uploadRGBFieldArray(
    cudaArray_t &array, const void *data, const uvec3 &dims)
{
  if (!data || dims.x == 0 || dims.y == 0 || dims.z == 0)
    return cudaErrorInvalidValue;
  const size_t maxElements =
      std::numeric_limits<size_t>::max() / sizeof(float4);
  if (size_t(dims.x) > maxElements / dims.y
      || size_t(dims.x) * dims.y > maxElements / dims.z)
    return cudaErrorInvalidValue;
  const size_t count = size_t(dims.x) * dims.y * dims.z;
  DeviceBuffer source, expanded;
  source.reserve(count * sizeof(float3));
  expanded.reserve(count * sizeof(float4));
  auto error =
      cudaMemcpy(source.ptr(), data, count * sizeof(float3), cudaMemcpyDefault);
  if (error != cudaSuccess)
    return error;
  auto begin = thrust::device_pointer_cast(source.ptrAs<float3>());
  thrust::transform(begin,
      begin + count,
      thrust::device_pointer_cast(expanded.ptrAs<float4>()),
      ExpandRGB{});
  if (!array) {
    const auto desc = cudaCreateChannelDesc<float4>();
    error = cudaMalloc3DArray(
        &array, &desc, make_cudaExtent(dims.x, dims.y, dims.z));
    if (error != cudaSuccess)
      return error;
  }
  cudaMemcpy3DParms copy{};
  copy.srcPtr = make_cudaPitchedPtr(
      expanded.ptr(), size_t(dims.x) * sizeof(float4), dims.x, dims.y);
  copy.dstArray = array;
  copy.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  copy.kind = cudaMemcpyDeviceToDevice;
  return cudaMemcpy3D(&copy);
}

} // namespace visrtx
