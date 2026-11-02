// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include "gpu/aggregateVolumeData.h"
#include "gpu/sbt.h"
#include "renderer/Renderer.h"
#include "utility/DeviceBuffer.h"

namespace quality_probe {
using namespace visrtx;
inline void optixChecked(OptixResult result)
{
  if (result != OPTIX_SUCCESS)
    throw std::runtime_error(optixGetErrorName(result));
}
inline void cudaChecked(cudaError_t result)
{
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record
{
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// Own only the probe module/PGs/pipeline/records. The production Quality and
// reference-field modules are borrowed from the live device, never recompiled
// from a test copy of the estimator. Destroy this before the ANARI device.
struct OptixProbe
{
  explicit OptixProbe(DeviceGlobalState &state);
  ~OptixProbe();
  void initialize();
  OptixProbe(const OptixProbe &) = delete;
  OptixProbe &operator=(const OptixProbe &) = delete;
  void launch(AggregateVolumeSceneGPUData scene, void *output, unsigned paths);

 private:
  DeviceGlobalState &m_state;
  OptixModule m_module{};
  std::array<OptixProgramGroup, 4> m_groups{};
  OptixPipeline m_pipeline{};
  DeviceBuffer m_raygen, m_callables, m_scene, m_params;
  OptixShaderBindingTable m_sbt{};
};

inline OptixProbe::OptixProbe(DeviceGlobalState &state) : m_state(state) {}

inline void OptixProbe::initialize()
{
  auto &state = m_state;
  optixChecked(optixInit());
  std::ifstream stream(QUALITY_VOLUME_PROBE_PTX, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open Quality probe PTX");
  const std::string ptx{
      std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  auto options = makeVisRTXOptixPipelineCompileOptions();
  OptixModuleCompileOptions moduleOptions{};
  moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  char log[8192];
  size_t logSize = sizeof(log);
  auto result = optixModuleCreate(state.optixContext,
      &moduleOptions,
      &options,
      ptx.data(),
      ptx.size(),
      log,
      &logSize,
      &m_module);
  if (result != OPTIX_SUCCESS)
    std::fprintf(stderr, "probe module: %s\n", log);
  optixChecked(result);
  std::array<OptixProgramGroupDesc, 4> descriptions{};
  descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  descriptions[0].raygen.module = m_module;
  descriptions[0].raygen.entryFunctionName = "__raygen__qualityVolumeProbe";
  for (int i = 1; i < 4; ++i)
    descriptions[i].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  descriptions[1].callables.moduleDC = state.rendererModules.quality;
  descriptions[1].callables.entryFunctionNameDC =
      "__direct_callable__qualityVolume";
  descriptions[2].callables.moduleDC = state.fieldSamplers.structuredRegular;
  descriptions[2].callables.entryFunctionNameDC =
      "__direct_callable__sampleReferenceScalar";
  descriptions[3].callables.moduleDC = state.fieldSamplers.structuredRegular;
  descriptions[3].callables.entryFunctionNameDC =
      "__direct_callable__sampleReferenceRGB";
  OptixProgramGroupOptions groupOptions{};
  logSize = sizeof(log);
  result = optixProgramGroupCreate(state.optixContext,
      descriptions.data(),
      4,
      &groupOptions,
      log,
      &logSize,
      m_groups.data());
  if (result != OPTIX_SUCCESS)
    std::fprintf(stderr, "probe groups: %s\n", log);
  optixChecked(result);
  OptixPipelineLinkOptions linkOptions{};
  linkOptions.maxTraceDepth = 1;
  logSize = sizeof(log);
  result = optixPipelineCreate(state.optixContext,
      &options,
      &linkOptions,
      m_groups.data(),
      4,
      log,
      &logSize,
      &m_pipeline);
  if (result != OPTIX_SUCCESS)
    std::fprintf(stderr, "probe link: %s\n", log);
  optixChecked(result);
  OptixStackSizes stacks{};
  for (auto group : m_groups)
    optixChecked(optixUtilAccumulateStackSizes(group, &stacks, m_pipeline));
  unsigned traversal, stateStack, continuation;
  optixChecked(optixUtilComputeStackSizes(
      &stacks, 1, 0, 2, &traversal, &stateStack, &continuation));
  optixChecked(optixPipelineSetStackSize(
      m_pipeline, traversal, stateStack, continuation, 2));
  Record raygen{};
  optixChecked(optixSbtRecordPackHeader(m_groups[0], &raygen));
  m_raygen.upload(&raygen);
  m_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(m_raygen.ptr());
  std::vector<Record> records(size_t(SbtCallableEntryPoints::Last));
  for (auto &record : records)
    optixChecked(optixSbtRecordPackHeader(m_groups[2], &record));
  optixChecked(optixSbtRecordPackHeader(
      m_groups[1], &records[size_t(SbtCallableEntryPoints::QualityVolume)]));
  optixChecked(optixSbtRecordPackHeader(m_groups[3],
      &records[size_t(SbtCallableEntryPoints::ReferenceFieldRGB)]));
  m_callables.upload(records);
  m_sbt.callablesRecordBase = reinterpret_cast<CUdeviceptr>(m_callables.ptr());
  m_sbt.callablesRecordStrideInBytes = sizeof(Record);
  m_sbt.callablesRecordCount = unsigned(records.size());
}
inline OptixProbe::~OptixProbe()
{
  cudaStreamSynchronize(m_state.stream);
  if (m_pipeline)
    optixPipelineDestroy(m_pipeline);
  for (auto group : m_groups)
    if (group)
      optixProgramGroupDestroy(group);
  if (m_module)
    optixModuleDestroy(m_module);
}
inline void OptixProbe::launch(
    AggregateVolumeSceneGPUData scene, void *output, unsigned paths)
{
  m_scene.upload(&scene);
  FrameGPUData params{};
  params.referenceVolumes = m_scene.ptrAs<AggregateVolumeSceneGPUData>();
  params.fb.buffers.colorAccumulation = static_cast<vec4 *>(output);
  m_params.upload(&params);
  cudaChecked(cudaStreamSynchronize(nullptr));
  optixChecked(optixLaunch(m_pipeline,
      m_state.stream,
      reinterpret_cast<CUdeviceptr>(m_params.ptr()),
      sizeof(params),
      &m_sbt,
      paths,
      1,
      1));
  cudaChecked(cudaStreamSynchronize(m_state.stream));
}
} // namespace quality_probe
