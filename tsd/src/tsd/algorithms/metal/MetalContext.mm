// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#define METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "MetalContext.h"

#include <Foundation/Foundation.hpp>

#include "GeneratedShaderSource.h"

namespace tsd::algorithms::metal {

MetalContext &MetalContext::instance()
{
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext()
{
    m_device = MTL::CreateSystemDefaultDevice();
    m_queue = m_device->newCommandQueue();

    auto *src =
        NS::String::string(kShaderSource.data(), NS::UTF8StringEncoding);
    auto *opts = MTL::CompileOptions::alloc()->init();
    NS::Error *error = nullptr;
    m_library = m_device->newLibrary(src, opts, &error);
    opts->release();
}

MetalContext::~MetalContext()
{
    for (auto &[_, pso] : m_pipelines)
        pso->release();
    m_library->release();
    m_queue->release();
    m_device->release();
}

MTL::Device *MetalContext::device() { return m_device; }

MTL::CommandQueue *MetalContext::defaultQueue()
{
    return m_externalQueue ? m_externalQueue : m_queue;
}

void MetalContext::setQueue(MTL::CommandQueue *queue)
{
    m_externalQueue = queue;
}

MTL::ComputePipelineState *MetalContext::pipelineState(const char *kernelName)
{
    std::lock_guard lock(m_mutex);
    auto it = m_pipelines.find(kernelName);
    if (it != m_pipelines.end())
        return it->second;

    auto *fn = m_library->newFunction(
        NS::String::string(kernelName, NS::ASCIIStringEncoding));
    NS::Error *error = nullptr;
    auto *pso = m_device->newComputePipelineState(fn, &error);
    fn->release();

    m_pipelines[kernelName] = pso;
    return pso;
}

} // namespace tsd::algorithms::metal
