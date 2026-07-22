/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// MDL linking spike (ticket 10 / 03), step 1: prove the toolchain that a
// per-material wavefront MDL kernel needs — compile a shade shell that calls an
// *undefined external* device function to relocatable PTX, compile that
// function's definition to a separate PTX, link the two with nvJitLink into a
// cubin, load it with the driver API, launch, and confirm the cross-blob call
// resolved. This is exactly the shape of (shade-shell PTX + MDL material PTX)
// linking; step 2 substitutes a real MDL blob for the external definition.

#include <cuda.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CU_CHECK(call)                                                          \
  do {                                                                          \
    CUresult _e = (call);                                                       \
    if (_e != CUDA_SUCCESS) {                                                   \
      const char *_s = nullptr;                                                 \
      cuGetErrorString(_e, &_s);                                                \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,          \
          __LINE__, _s ? _s : "?");                                            \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

static std::string compileToPtx(
    const char *src, const char *name, const std::string &arch)
{
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, src, name, 0, nullptr, nullptr)
      != NVRTC_SUCCESS) {
    fprintf(stderr, "nvrtcCreateProgram failed\n");
    std::exit(1);
  }
  // -rdc=true: relocatable device code, so the external symbol stays unresolved
  // for the link step (mirrors the shade shell referencing MDL functions).
  const std::string archOpt = "--gpu-architecture=compute_" + arch;
  const char *opts[] = {"--relocatable-device-code=true", archOpt.c_str()};
  const nvrtcResult r = nvrtcCompileProgram(prog, 2, opts);
  size_t logSize = 0;
  nvrtcGetProgramLogSize(prog, &logSize);
  if (logSize > 1) {
    std::string log(logSize, '\0');
    nvrtcGetProgramLog(prog, log.data());
    fprintf(stderr, "[nvrtc %s]\n%s\n", name, log.c_str());
  }
  if (r != NVRTC_SUCCESS) {
    fprintf(stderr, "nvrtcCompileProgram(%s) failed\n", name);
    std::exit(1);
  }
  size_t ptxSize = 0;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);
  return ptx;
}

int main()
{
  CU_CHECK(cuInit(0));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU_CHECK(cuCtxCreate(&ctx, 0, dev));

  // Target the actual device: a linked cubin is arch-specific SASS, so it must
  // match the GPU or cuModuleLoadData rejects it.
  int ccMajor = 0, ccMinor = 0;
  CU_CHECK(cuDeviceGetAttribute(
      &ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CU_CHECK(cuDeviceGetAttribute(
      &ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  const std::string arch = std::to_string(ccMajor) + std::to_string(ccMinor);

  // The "shade shell": a kernel that calls an external device function it does
  // not define (as the wavefront MDL shell would call the MDL bsdf functions).
  const char *shellSrc =
      "extern \"C\" __device__ float spikeExternal(float x);\n"
      "extern \"C\" __global__ void spikeKernel(float *out) {\n"
      "  out[0] = spikeExternal(41.0f);\n"
      "}\n";
  // The separately-compiled definition (stands in for the MDL material PTX).
  const char *externSrc =
      "extern \"C\" __device__ float spikeExternal(float x) {\n"
      "  return x + 1.0f;\n"
      "}\n";

  const std::string shellPtx = compileToPtx(shellSrc, "shell.cu", arch);
  const std::string externPtx = compileToPtx(externSrc, "extern.cu", arch);

  // Link the two PTX blobs into a cubin.
  nvJitLinkHandle linker;
  const std::string archOpt = "-arch=sm_" + arch;
  const char *linkOpts[] = {archOpt.c_str()};
  if (nvJitLinkCreate(&linker, 1, linkOpts) != NVJITLINK_SUCCESS) {
    fprintf(stderr, "nvJitLinkCreate failed\n");
    return 1;
  }
  auto addPtx = [&](const std::string &ptx, const char *name) {
    if (nvJitLinkAddData(linker, NVJITLINK_INPUT_PTX, ptx.data(), ptx.size(),
            name)
        != NVJITLINK_SUCCESS) {
      fprintf(stderr, "nvJitLinkAddData(%s) failed\n", name);
      std::exit(1);
    }
  };
  addPtx(shellPtx, "shell");
  addPtx(externPtx, "extern");

  if (nvJitLinkComplete(linker) != NVJITLINK_SUCCESS) {
    size_t logSize = 0;
    nvJitLinkGetErrorLogSize(linker, &logSize);
    std::string log(logSize, '\0');
    nvJitLinkGetErrorLog(linker, log.data());
    fprintf(stderr, "FAIL: nvJitLinkComplete could not resolve the link:\n%s\n",
        log.c_str());
    return 1;
  }

  size_t cubinSize = 0;
  if (nvJitLinkGetLinkedCubinSize(linker, &cubinSize) != NVJITLINK_SUCCESS
      || cubinSize == 0) {
    fprintf(stderr, "FAIL: no linked cubin produced\n");
    return 1;
  }
  std::vector<char> cubin(cubinSize);
  nvJitLinkGetLinkedCubin(linker, cubin.data());
  nvJitLinkDestroy(&linker);

  // Load and launch through the driver API.
  CUmodule mod;
  CU_CHECK(cuModuleLoadData(&mod, cubin.data()));
  CUfunction fn;
  CU_CHECK(cuModuleGetFunction(&fn, mod, "spikeKernel"));
  CUdeviceptr outDev;
  CU_CHECK(cuMemAlloc(&outDev, sizeof(float)));
  void *args[] = {&outDev};
  CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  CU_CHECK(cuCtxSynchronize());
  float result = 0.f;
  CU_CHECK(cuMemcpyDtoH(&result, outDev, sizeof(float)));

  CU_CHECK(cuMemFree(outDev));
  CU_CHECK(cuModuleUnload(mod));
  CU_CHECK(cuCtxDestroy(ctx));

  printf("MDL linking spike: cubin %zu bytes, kernel result %.1f (expect 42)\n",
      cubinSize,
      result);
  if (result != 42.0f) {
    fprintf(stderr,
        "FAIL: cross-blob call did not resolve (got %.1f) — nvJitLink toolchain "
        "cannot link a shell PTX against a separately-compiled function\n",
        result);
    return 1;
  }

  printf("PASS: nvJitLink links a shade shell against a separate PTX blob and "
         "the cross-module call resolves — the MDL per-material link path is "
         "viable at the toolchain level.\n");
  return 0;
}
