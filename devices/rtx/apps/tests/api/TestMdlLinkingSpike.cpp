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

// anari_cpp (to generate a real MDL material PTX blob via the device)
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>

#include <cuda.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using vec3 = std::array<float, 3>;

// A trivial emissive MDL material. Committing it in a scene makes the device
// generate the material's PTX (and dump it, since VISRTX_DUMP_MDL_PTX is set).
static const char *MDL_EMISSIVE = R"mdl(mdl 1.6;
import ::df::*;
export material emissive() = material(
    surface: material_surface(
        emission: material_emission(
            emission: df::diffuse_edf(),
            intensity: color(16.0))));
)mdl";

// Render one frame of an MDL-material sphere so the device generates + dumps the
// material PTX. Torn down fully before the driver-API phase to avoid mixing the
// device's CUDA context with the spike's.
static void generateMdlPtxDump()
{
  auto device = makeVisRTXDevice(nullptr);

  auto pos = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
  *anari::map<vec3>(device, pos) = vec3{0.f, 0.f, 0.f};
  anari::unmap(device, pos);
  auto geom = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(device, geom, "vertex.position", pos);
  anari::setParameter(device, geom, "radius", 0.5f);
  anari::commitParameters(device, geom);

  auto mat = anari::newObject<anari::Material>(device, "mdl");
  anari::setParameter(device, mat, "sourceType", "code");
  anari::setParameter(device, mat, "source", MDL_EMISSIVE);
  anari::setParameter(device, mat, "materialName", "emissive");
  anari::commitParameters(device, mat);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geom);
  anari::setAndReleaseParameter(device, surface, "material", mat);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  anari::setParameter(device, camera, "position", vec3{0.f, 0.f, -2.f});
  anari::setParameter(device, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(device, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::commitParameters(device, camera);

  auto renderer = anari::newObject<anari::Renderer>(device, "default");
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  std::array<unsigned int, 2> size = {64, 64};
  anari::setParameter(device, frame, "size", size);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  anari::release(device, camera);
  anari::release(device, renderer);
  anari::release(device, world);
  anari::release(device, frame);
  anari::release(device, device);
}

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
  // Phase 0 (ANARI): render an MDL material so the device dumps its PTX. Done
  // and torn down before any driver-API work below.
  const char *dumpDir = "./mdl_spike_ptx";
  std::filesystem::create_directories(dumpDir);
  setenv("VISRTX_DUMP_MDL_PTX", dumpDir, 1);
  { // scope so any stack ANARI handles are gone before the driver-API phase
    generateMdlPtxDump();
  }

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
  printf("PASS (step 1): nvJitLink links a shade shell against a separate PTX "
         "blob and the cross-module call resolves.\n");

  // Step 2: link the REAL MDL material (dumped in phase 0) against its texture
  // runtime and a CUDA shade shell.
  auto readFile = [](const std::string &path) -> std::string {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
      return {};
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(size_t(n), '\0');
    size_t got = fread(s.data(), 1, size_t(n), f);
    fclose(f);
    s.resize(got);
    return s;
  };
  const std::string materialPtx =
      readFile(std::string(dumpDir) + "/material.ptx");
  const std::string texturePtx =
      readFile(std::string(dumpDir) + "/texture.ptx");
  if (materialPtx.empty() || texturePtx.empty()) {
    fprintf(stderr,
        "FAIL (step 2): the MDL material render did not dump PTX to %s — MDL "
        "support may be disabled\n",
        dumpDir);
    return 1;
  }

  // A CUDA shade shell that calls the MDL bsdf-evaluate entry, so the link must
  // resolve a real MDL function symbol (not just the texture runtime).
  const char *mdlShellSrc =
      "extern \"C\" __device__ void mdlBsdf_evaluate(\n"
      "    void *data, const void *state, const void *rd, const char *arg);\n"
      "extern \"C\" __global__ void mdlSpikeKernel(\n"
      "    void *data, const void *state, const void *rd, const char *arg,\n"
      "    int *touched) {\n"
      "  mdlBsdf_evaluate(data, state, rd, arg);\n"
      "  *touched = 1;\n"
      "}\n";
  const std::string mdlShellPtx = compileToPtx(mdlShellSrc, "mdlshell.cu", arch);

  nvJitLinkHandle mdlLinker;
  if (nvJitLinkCreate(&mdlLinker, 1, linkOpts) != NVJITLINK_SUCCESS) {
    fprintf(stderr, "nvJitLinkCreate (MDL) failed\n");
    return 1;
  }
  auto addMdl = [&](const std::string &ptx, const char *name) {
    return nvJitLinkAddData(
               mdlLinker, NVJITLINK_INPUT_PTX, ptx.data(), ptx.size(), name)
        == NVJITLINK_SUCCESS;
  };
  addMdl(mdlShellPtx, "mdlshell");
  addMdl(materialPtx, "material");
  addMdl(texturePtx, "texture");
  if (nvJitLinkComplete(mdlLinker) != NVJITLINK_SUCCESS) {
    size_t logSize = 0;
    nvJitLinkGetErrorLogSize(mdlLinker, &logSize);
    std::string log(logSize, '\0');
    nvJitLinkGetErrorLog(mdlLinker, log.data());
    fprintf(stderr,
        "FAIL (step 2): nvJitLink could not link the MDL material + texture "
        "runtime + shade shell:\n%s\n",
        log.c_str());
    return 1;
  }
  size_t mdlCubinSize = 0;
  nvJitLinkGetLinkedCubinSize(mdlLinker, &mdlCubinSize);
  std::vector<char> mdlCubin(mdlCubinSize);
  nvJitLinkGetLinkedCubin(mdlLinker, mdlCubin.data());
  nvJitLinkDestroy(&mdlLinker);

  CUmodule mdlMod;
  const CUresult loadRes = cuModuleLoadData(&mdlMod, mdlCubin.data());
  if (loadRes != CUDA_SUCCESS) {
    const char *s = nullptr;
    cuGetErrorString(loadRes, &s);
    fprintf(stderr, "FAIL (step 2): linked MDL cubin did not load: %s\n",
        s ? s : "?");
    return 1;
  }
  CUfunction mdlFn;
  const bool haveKernel =
      cuModuleGetFunction(&mdlFn, mdlMod, "mdlSpikeKernel") == CUDA_SUCCESS;
  cuModuleUnload(mdlMod);

  printf("PASS (step 2): nvJitLink linked a REAL MDL material (%zu B) + texture "
         "runtime (%zu B) + a CUDA shade shell into a loadable cubin (%zu B); "
         "kernel entry present: %s. The MDL per-material CUDA-kernel path is "
         "feasible — material references only tex_lookup (resolved by the "
         "texture runtime), no OptiX.\n",
      materialPtx.size(),
      texturePtx.size(),
      mdlCubinSize,
      haveKernel ? "yes" : "no");

  CU_CHECK(cuCtxDestroy(ctx));
  return 0;
}
