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

#pragma once

#include <cuda.h>
// std
#include <cstdint>
#include <string>
#include <unordered_map>
// anari
#include <nonstd/span.hpp>

namespace visrtx {

// One per-material MDL shade kernel: a cubin (module) linked from the wavefront
// MDL shade shell + that material's PTX, and the resolved kernel entry.
struct WavefrontMdlKernel
{
  CUmodule module{nullptr};
  CUfunction function{nullptr};
  explicit operator bool() const
  {
    return function != nullptr;
  }
};

// Builds and caches per-material MDL shade kernels. At MDL material commit the
// device hands each material's PTX blob (from MaterialRegistry::getPtxBlobs)
// and a stable key (the compiled-material UUID) here; the cache nvJitLinks the
// shell against it into a loadable cubin the shade stage can launch. Keyed by
// UUID so the registry's dedup carries through (one kernel per distinct
// compiled material). The nvJitLink toolchain is the one proven by
// TestMdlLinkingSpike.
class WavefrontMdlKernelCache
{
 public:
  WavefrontMdlKernelCache() = default;
  ~WavefrontMdlKernelCache();
  WavefrontMdlKernelCache(const WavefrontMdlKernelCache &) = delete;
  WavefrontMdlKernelCache &operator=(const WavefrontMdlKernelCache &) = delete;

  // Return the cached kernel for `key`, or link one from `materialPtx` (the
  // RAW MDL-generated material PTX from MaterialRegistry::getMaterialPtxBlobs)
  // — nvJitLinked against the embedded shade shell + the shared MDL texture
  // runtime. A null-function result is returned on link/load failure, uncached.
  WavefrontMdlKernel getOrBuild(
      uint64_t key, nonstd::span<const char> materialPtx);

  bool contains(uint64_t key) const;
  void release(); // destroy all cached modules

  // Number of live cached kernels (for tests / diagnostics).
  size_t size() const
  {
    return m_kernels.size();
  }

 private:
  const std::string &arch(); // "89", lazily queried from the current device

  std::string m_arch;
  std::unordered_map<uint64_t, WavefrontMdlKernel> m_kernels;
};

// Launch a per-material MDL shade kernel over its compacted slot list.
// `frameData` is the device FrameGPUData pointer; `packed` is the shared packed
// slot-index array; `offset` / `count` point at this material's words in the
// compaction output (read on-device, so no host readback). `gridUpperBound`
// (>= the material's slot count) sizes a conservative grid; threads past the
// device-side count early-out. Enqueued on `stream`; false on a launch error.
bool launchWavefrontMdlShade(const WavefrontMdlKernel &kernel,
    CUstream stream,
    CUdeviceptr frameData,
    CUdeviceptr packed,
    CUdeviceptr offset,
    CUdeviceptr count,
    uint32_t gridUpperBound);

} // namespace visrtx
