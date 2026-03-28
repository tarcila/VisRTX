// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tsd::algorithms::metal {

struct MetalContext
{
  static MetalContext &instance();

  MTL::Device *device();
  MTL::CommandQueue *defaultQueue();
  void setQueue(MTL::CommandQueue *queue);
  MTL::ComputePipelineState *pipelineState(const char *kernelName);

 private:
  MetalContext();
  ~MetalContext();

  MTL::Device *m_device{nullptr};
  MTL::CommandQueue *m_queue{nullptr};
  MTL::CommandQueue *m_externalQueue{nullptr};
  MTL::Library *m_library{nullptr};
  std::unordered_map<std::string, MTL::ComputePipelineState *> m_pipelines;
  std::mutex m_mutex;
};

} // namespace tsd::algorithms::metal
