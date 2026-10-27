/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "spatial_field/FieldTexture.h"
#include "spatial_field/StoredFieldSampler.h"
#include "utility/DeviceBuffer.h"

#include <nanovdb/tools/CreateNanoGrid.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace visrtx;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const auto e = (call);                                                     \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e));          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Texture
{
  cudaArray_t array{};
  cudaTextureObject_t texture{};
  Texture() = default;
  Texture(const Texture &) = delete;
  Texture &operator=(const Texture &) = delete;
  ~Texture();
  void create(bool linear, bool normalized = false);
};

Texture::~Texture()
{
  if (texture)
    cudaDestroyTextureObject(texture);
  if (array)
    cudaFreeArray(array);
}

void Texture::create(bool linear, bool normalized)
{
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = array;
  cudaTextureDesc desc{};
  desc.addressMode[0] = desc.addressMode[1] = desc.addressMode[2] =
      cudaAddressModeClamp;
  desc.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
  desc.readMode = cudaReadModeElementType;
  desc.normalizedCoords = normalized;
  CUDA_CHECK(cudaCreateTextureObject(&texture, &resource, &desc, nullptr));
}

VISRTX_GLOBAL void fetch(SpatialFieldGPUData field,
    FieldBindingGPUData<vec3> binding,
    vec3 p,
    vec3 *result)
{
  StoredFieldSampler sampler{&field};
  *result = evaluateFieldBinding(binding, p, sampler);
}

VISRTX_GLOBAL void fetchScalar(SpatialFieldGPUData field, vec3 p, float *result)
{
  StoredFieldSampler sampler{&field};
  *result = sampler.sample<float>(0, p);
}

static int failures = 0;
static void checkScalar(const SpatialFieldGPUData &field,
    const vec3 &p,
    float expected,
    const char *label)
{
  DeviceBuffer output;
  output.reserve(sizeof(float));
  fetchScalar<<<1, 1>>>(field, p, output.ptrAs<float>());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float value;
  output.download(&value);
  if (fabsf(value - expected) > 2e-3f) {
    std::printf("FAIL %s scalar: got %g expected %g\n", label, value, expected);
    ++failures;
  }
}

static void checkSample(const SpatialFieldGPUData &field,
    const box3 &domain,
    const vec3 &p,
    const vec3 &expected,
    const char *label)
{
  FieldBindingGPUData<vec3> binding{vec3(1.f), vec3(5.f)};
  binding.field = 0;
  binding.domain = domain;
  DeviceBuffer output;
  output.reserve(sizeof(vec3));
  fetch<<<1, 1>>>(field, binding, p, output.ptrAs<vec3>());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  vec3 value;
  output.download(&value);
  for (int c = 0; c < 3; ++c) {
    if (fabsf(value[c] - expected[c]) > 2e-3f) {
      std::printf("FAIL %s channel %d: got %g expected %g\n",
          label,
          c,
          value[c],
          expected[c]);
      ++failures;
    }
  }
}

static void identityLUT(Texture &lut)
{
  std::vector<float> values(8192);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = float(i) / float(values.size() - 1);
  const auto desc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaMallocArray(&lut.array, &desc, values.size()));
  CUDA_CHECK(cudaMemcpy2DToArray(lut.array,
      0,
      0,
      values.data(),
      values.size() * sizeof(float),
      values.size() * sizeof(float),
      1,
      cudaMemcpyHostToDevice));
  lut.create(true, true);
}

int main()
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
    return 77;
  Texture lut;
  identityLUT(lut);
  std::vector<float3> values;
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        values.push_back(make_float3(float(x + 1), float(y + 2), float(z + 3)));
  DeviceBuffer deviceInput;
  deviceInput.upload(values);

  for (bool deviceSource : {false, true})
    for (bool linear : {false, true})
      for (bool cellCentered : {false, true}) {
        Texture tex;
        const void *source = deviceSource ? deviceInput.ptr() : values.data();
        CUDA_CHECK(uploadRGBFieldArray(tex.array, source, uvec3(2)));
        tex.create(linear);
        SpatialFieldGPUData field{};
        field.valueType = FieldValueType::RGB;
        field.storageType = StoredFieldType::StructuredRegular;
        auto &r = field.data.structuredRegular;
        r.texObj = tex.texture;
        r.origin = vec3(0);
        r.invSpacing = vec3(1);
        r.dims = ivec3(2);
        r.cellCentered = cellCentered;
        r.filter =
            linear ? SpatialFieldFilter::Linear : SpatialFieldFilter::Nearest;
        const float extent = cellCentered ? 2.f : 1.f;
        const box3 domain(vec3(0), vec3(extent));
        const vec3 point(linear ? extent * .5f : 0.f);
        const vec3 expected = linear ? vec3(1.5f, 2.5f, 3.5f) : vec3(1, 2, 3);
        checkSample(field, domain, point, expected, "dense regular");
        checkSample(field, domain, vec3(-1), vec3(5), "dense outside");
        field.storageType = StoredFieldType::StructuredRectilinear;
        auto &rr = field.data.structuredRectilinear;
        rr.texObj = tex.texture;
        rr.dims = vec3(2);
        rr.cellCentered = cellCentered;
        rr.axisBoundsMin = vec3(0);
        rr.axisBoundsMax = vec3(extent);
        rr.filter =
            linear ? SpatialFieldFilter::Linear : SpatialFieldFilter::Nearest;
        for (int i = 0; i < 3; ++i)
          rr.axisLUT[i] = rr.invAxisLUT[i] = lut.texture;
        checkSample(field, domain, point, expected, "dense rectilinear");
      }

  nanovdb::tools::build::Grid<nanovdb::Vec3f> source(
      nanovdb::Vec3f(10, 20, 30));
  auto accessor = source.getAccessor();
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        if (x || y || z)
          accessor.setValue(nanovdb::Coord(x, y, z),
              nanovdb::Vec3f(float(x + 1), float(y + 2), float(z + 3)));
  auto handle = nanovdb::tools::createNanoGrid(source);
  DeviceBuffer grid;
  grid.upload(static_cast<const uint8_t *>(handle.data()), handle.bufferSize());
  for (bool linear : {false, true})
    for (bool cellCentered : {false, true}) {
      SpatialFieldGPUData field{};
      field.valueType = FieldValueType::RGB;
      field.storageType = StoredFieldType::NanoVDB;
      auto &r = field.data.nvdbRegular;
      r.gridData = grid.ptr();
      r.gridType = nanovdb::GridType::Vec3f;
      r.cellCentered = cellCentered;
      r.filter =
          linear ? SpatialFieldFilter::Linear : SpatialFieldFilter::Nearest;
      r.invTwoVoxelSize = vec3(.5f);
      const box3 domain(vec3(0), vec3(2));
      const vec3 point(linear ? 1.f : 0.f);
      const vec3 expected =
          linear ? vec3(2.625f, 4.75f, 6.875f) : vec3(10, 20, 30);
      checkSample(field,
          domain,
          point,
          expected,
          "NanoVDB including sparse background");
      checkSample(field,
          domain,
          vec3(3),
          vec3(5),
          "NanoVDB outside distinct from background");
      field.storageType = StoredFieldType::NanoVDBRectilinear;
      auto &rr = field.data.nvdbRectilinear;
      rr.gridData = grid.ptr();
      rr.gridType = nanovdb::GridType::Vec3f;
      rr.cellCentered = cellCentered;
      rr.filter =
          linear ? SpatialFieldFilter::Linear : SpatialFieldFilter::Nearest;
      rr.invAvgVoxelSize = vec3(1);
      for (int i = 0; i < 3; ++i)
        rr.axisLUT[i] = rr.invAxisLUT[i] = lut.texture;
      checkSample(field, domain, point, expected, "NanoVDB rectilinear");
    }
  // Compile and exercise the scalar adapter too: no RGB broadcast/reduction.
  Texture scalarTexture;
  const float scalarValues[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  const auto desc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(
      cudaMalloc3DArray(&scalarTexture.array, &desc, make_cudaExtent(2, 2, 2)));
  cudaMemcpy3DParms copy{};
  copy.srcPtr = make_cudaPitchedPtr(
      const_cast<float *>(scalarValues), 2 * sizeof(float), 2, 2);
  copy.dstArray = scalarTexture.array;
  copy.extent = make_cudaExtent(2, 2, 2);
  copy.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpy3D(&copy));
  scalarTexture.create(true);
  SpatialFieldGPUData scalar{};
  scalar.storageType = StoredFieldType::StructuredRegular;
  auto &sr = scalar.data.structuredRegular;
  sr.texObj = scalarTexture.texture;
  sr.origin = vec3(0);
  sr.invSpacing = vec3(1);
  sr.dims = ivec3(2);
  sr.cellCentered = false;
  sr.filter = SpatialFieldFilter::Linear;
  checkScalar(scalar, vec3(.5f), 4.5f, "dense regular");
  // The accepted FLOAT64 storage format must be converted to float texels;
  // CUDA does not support a 64-bit floating texture element descriptor.
  const std::vector<double> doubleValues{1, 2, 3, 4, 5, 6, 7, 8};
  DeviceBuffer doubleInput;
  doubleInput.upload(doubleValues);
  for (bool deviceSource : {false, true}) {
    Texture doubleTexture;
    const void *source = deviceSource ? doubleInput.ptr() : doubleValues.data();
    CUDA_CHECK(uploadDoubleFieldArray(doubleTexture.array, source, uvec3(2)));
    doubleTexture.create(true);
    auto doubleField = scalar;
    doubleField.data.structuredRegular.texObj = doubleTexture.texture;
    checkScalar(doubleField, vec3(.5f), 4.5f, "FLOAT64 converted texture");
  }
  scalar.storageType = StoredFieldType::StructuredRectilinear;
  auto &srr = scalar.data.structuredRectilinear;
  srr.texObj = scalarTexture.texture;
  srr.dims = vec3(2);
  srr.cellCentered = false;
  srr.axisBoundsMin = vec3(0);
  srr.axisBoundsMax = vec3(1);
  srr.filter = SpatialFieldFilter::Linear;
  for (int i = 0; i < 3; ++i)
    srr.axisLUT[i] = srr.invAxisLUT[i] = lut.texture;
  checkScalar(scalar, vec3(.5f), 4.5f, "dense rectilinear");
  nanovdb::tools::build::Grid<float> scalarSource(0.f);
  auto scalarAcc = scalarSource.getAccessor();
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        scalarAcc.setValue(
            nanovdb::Coord(x, y, z), scalarValues[x + 2 * y + 4 * z]);
  auto scalarHandle = nanovdb::tools::createNanoGrid(scalarSource);
  DeviceBuffer scalarGrid;
  scalarGrid.upload(static_cast<const uint8_t *>(scalarHandle.data()),
      scalarHandle.bufferSize());
  scalar.storageType = StoredFieldType::NanoVDB;
  auto &sn = scalar.data.nvdbRegular;
  sn.gridData = scalarGrid.ptr();
  sn.gridType = nanovdb::GridType::Float;
  sn.cellCentered = true;
  sn.filter = SpatialFieldFilter::Linear;
  sn.invTwoVoxelSize = vec3(.5f);
  checkScalar(scalar, vec3(1), 4.5f, "NanoVDB");
  scalar.storageType = StoredFieldType::NanoVDBRectilinear;
  auto &snr = scalar.data.nvdbRectilinear;
  snr.gridData = scalarGrid.ptr();
  snr.gridType = nanovdb::GridType::Float;
  snr.cellCentered = true;
  snr.filter = SpatialFieldFilter::Linear;
  snr.invAvgVoxelSize = vec3(1);
  for (int i = 0; i < 3; ++i)
    snr.axisLUT[i] = snr.invAxisLUT[i] = lut.texture;
  checkScalar(scalar, vec3(1), 4.5f, "NanoVDB rectilinear");

  SpatialFieldGPUData custom{};
  if (supportsStoredFieldSampling(custom, FieldValueType::Scalar))
    ++failures;
  custom.storageType = StoredFieldType::StructuredRegular;
  if (supportsStoredFieldSampling(custom, FieldValueType::RGB))
    ++failures;
  std::printf("Stored-field sampling: %d failures\n", failures);
  return failures ? 1 : 0;
}
