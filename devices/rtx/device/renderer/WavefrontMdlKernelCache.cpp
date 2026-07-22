/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "WavefrontMdlKernelCache.h"
// embedded relocatable shell PTX
#include "WavefrontMdlShell_ptx.h"
// nvJitLink
#include <nvJitLink.h>
// std
#include <cstdio>
#include <vector>

namespace visrtx {

namespace {

// The shell exports this kernel; the per-material link must expose it.
constexpr const char *kMdlShadeKernelName = "wavefrontMdlShade";

} // namespace

WavefrontMdlKernelCache::~WavefrontMdlKernelCache()
{
  release();
}

const std::string &WavefrontMdlKernelCache::arch()
{
  if (!m_arch.empty())
    return m_arch;
  CUdevice dev = 0;
  int major = 0, minor = 0;
  if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
    cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  }
  m_arch = std::to_string(major) + std::to_string(minor);
  return m_arch;
}

bool WavefrontMdlKernelCache::contains(uint64_t key) const
{
  return m_kernels.find(key) != m_kernels.end();
}

WavefrontMdlKernel WavefrontMdlKernelCache::getOrBuild(
    uint64_t key, nonstd::span<const char> materialPtx)
{
  if (auto it = m_kernels.find(key); it != m_kernels.end())
    return it->second;

  // Link the wavefront MDL shell against this material's PTX. The material blob
  // is self-contained (texture runtime + material code), and neither references
  // OptiX, so nvJitLink resolves everything into a loadable cubin.
  const std::string archOpt = "-arch=sm_" + arch();
  const char *linkOpts[] = {archOpt.c_str()};

  nvJitLinkHandle linker{};
  if (nvJitLinkCreate(&linker, 1, linkOpts) != NVJITLINK_SUCCESS) {
    fprintf(stderr, "[wavefront MDL] nvJitLinkCreate failed\n");
    return {};
  }

  bool ok = nvJitLinkAddData(linker,
                NVJITLINK_INPUT_PTX,
                reinterpret_cast<const char *>(WavefrontMdlShell_ptx),
                sizeof(WavefrontMdlShell_ptx),
                "wavefront_mdl_shell")
      == NVJITLINK_SUCCESS;
  ok = ok
      && nvJitLinkAddData(linker,
             NVJITLINK_INPUT_PTX,
             materialPtx.data(),
             materialPtx.size(),
             "mdl_material")
          == NVJITLINK_SUCCESS;

  if (!ok || nvJitLinkComplete(linker) != NVJITLINK_SUCCESS) {
    size_t logSize = 0;
    nvJitLinkGetErrorLogSize(linker, &logSize);
    std::string log(logSize, '\0');
    if (logSize)
      nvJitLinkGetErrorLog(linker, log.data());
    fprintf(stderr, "[wavefront MDL] link failed for key %llu:\n%s\n",
        static_cast<unsigned long long>(key), log.c_str());
    nvJitLinkDestroy(&linker);
    return {};
  }

  size_t cubinSize = 0;
  nvJitLinkGetLinkedCubinSize(linker, &cubinSize);
  std::vector<char> cubin(cubinSize);
  nvJitLinkGetLinkedCubin(linker, cubin.data());
  nvJitLinkDestroy(&linker);

  WavefrontMdlKernel kernel{};
  if (cuModuleLoadData(&kernel.module, cubin.data()) != CUDA_SUCCESS) {
    fprintf(stderr, "[wavefront MDL] cuModuleLoadData failed for key %llu\n",
        static_cast<unsigned long long>(key));
    return {};
  }
  if (cuModuleGetFunction(&kernel.function, kernel.module, kMdlShadeKernelName)
      != CUDA_SUCCESS) {
    fprintf(stderr, "[wavefront MDL] kernel '%s' not found for key %llu\n",
        kMdlShadeKernelName, static_cast<unsigned long long>(key));
    cuModuleUnload(kernel.module);
    return {};
  }

  m_kernels.emplace(key, kernel);
  return kernel;
}

void WavefrontMdlKernelCache::release()
{
  for (auto &[key, kernel] : m_kernels) {
    (void)key;
    if (kernel.module)
      cuModuleUnload(kernel.module);
  }
  m_kernels.clear();
}

} // namespace visrtx
