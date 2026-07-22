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
// MDL texture runtime PTX (shared by all materials)
#include "mdl/ptx.h"
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

  // Link the wavefront MDL shade shell against this material's RAW PTX plus the
  // shared MDL texture runtime — the same three-way link the linking spike
  // proved. None reference OptiX (the raw material code, unlike the stitched
  // OptiX blob), so nvJitLink resolves everything into a loadable cubin. The
  // stitched getPtxBlobs() blob can't be used here: it bundles the OptiX
  // surface-eval callable and localizes the mdl* symbols the shell imports.
  const std::string archOpt = "-arch=sm_" + arch();
  const char *linkOpts[] = {archOpt.c_str()};

  nvJitLinkHandle linker{};
  if (nvJitLinkCreate(&linker, 1, linkOpts) != NVJITLINK_SUCCESS) {
    fprintf(stderr, "[wavefront MDL] nvJitLinkCreate failed\n");
    return {};
  }

  // nvJitLink's PTX parser expects NUL-terminated text; the shell array and the
  // material/texture blobs have no guaranteed trailing NUL, so copy each into a
  // std::string (whose data() is NUL-terminated). Passing a bare span fails
  // with "does not match type NVJITLINK_INPUT_PTX".
  const std::string shellText(
      reinterpret_cast<const char *>(WavefrontMdlShell_ptx),
      sizeof(WavefrontMdlShell_ptx));
  const std::string materialText(materialPtx.data(), materialPtx.size());
  const std::string textureText(
      reinterpret_cast<const char *>(mdl::ptx::MDLTexture.ptr),
      mdl::ptx::MDLTexture.size);

  const auto add = [&](const std::string &ptx, const char *name) {
    return nvJitLinkAddData(
               linker, NVJITLINK_INPUT_PTX, ptx.data(), ptx.size(), name)
        == NVJITLINK_SUCCESS;
  };
  bool ok = add(shellText, "wavefront_mdl_shell");
  ok = ok && add(materialText, "mdl_material");
  ok = ok && add(textureText, "mdl_texture");

  if (!ok || nvJitLinkComplete(linker) != NVJITLINK_SUCCESS) {
    size_t logSize = 0;
    nvJitLinkGetErrorLogSize(linker, &logSize);
    std::string log(logSize, '\0');
    if (logSize)
      nvJitLinkGetErrorLog(linker, log.data());
    fprintf(stderr,
        "[wavefront MDL] link failed for key %llu:\n%s\n",
        static_cast<unsigned long long>(key),
        log.c_str());
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
    fprintf(stderr,
        "[wavefront MDL] cuModuleLoadData failed for key %llu\n",
        static_cast<unsigned long long>(key));
    return {};
  }
  if (cuModuleGetFunction(&kernel.function, kernel.module, kMdlShadeKernelName)
      != CUDA_SUCCESS) {
    fprintf(stderr,
        "[wavefront MDL] kernel '%s' not found for key %llu\n",
        kMdlShadeKernelName,
        static_cast<unsigned long long>(key));
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

bool launchWavefrontMdlShade(const WavefrontMdlKernel &kernel,
    CUstream stream,
    CUdeviceptr frameData,
    uint32_t callableBaseIndex,
    uint32_t liveSlots)
{
  if (!kernel || liveSlots == 0)
    return false;

  constexpr unsigned int kThreadsPerBlock = 256;
  const unsigned int blocks =
      (liveSlots + kThreadsPerBlock - 1) / kThreadsPerBlock;

  void *args[] = {&frameData, &callableBaseIndex, &liveSlots};
  const CUresult r = cuLaunchKernel(kernel.function,
      blocks,
      1,
      1,
      kThreadsPerBlock,
      1,
      1,
      0,
      stream,
      args,
      nullptr);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "[wavefront MDL] cuLaunchKernel failed (%d)\n", int(r));
    return false;
  }
  return true;
}

} // namespace visrtx
