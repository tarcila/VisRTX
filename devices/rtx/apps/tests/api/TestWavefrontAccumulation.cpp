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

// Multi-sample accumulation for the wavefront renderer's Path Pool cycle.
// pixelSamples=N drives N waves through the host cycle loop in a single render:
// the regenerate stage assigns each wave a distinct per-pixel sample ordinal,
// each casting fresh sub-pixel jitter, and the results accumulate. This asserts
// (a) the frame reports N accumulated samples (the pool dispatched every wave)
// and (b) the N-sample image is meaningfully anti-aliased relative to the
// 1-sample image (the waves produced distinct samples, not N copies of one).
// The single-sample smoke can't see this — only the multi-wave path can.

// anari_cpp
#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
// VisRTX
#include <anari/ext/visrtx/makeVisRTXDevice.h>
// std
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_FATAL_ERROR
      || severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR][%p] %s\n", source, message);
    std::exit(1);
  }
}

static anari::World generateScene(anari::Device device)
{
  auto positionsArray = anari::newArray1D(device, ANARI_FLOAT32_VEC3, 1);
  {
    auto *positions = anari::map<vec3>(device, positionsArray);
    positions[0] = vec3{0.f, 0.f, 0.f};
    anari::unmap(device, positionsArray);
  }

  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", positionsArray);
  anari::setParameter(device, geometry, "radius", 0.9f);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", vec3{0.8f, 0.2f, 0.2f});
  anari::commitParameters(device, material);

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);

  auto world = anari::newObject<anari::World>(device);
  anari::setParameterArray1D(device, world, "surface", &surface, 1);
  anari::release(device, surface);
  anari::commitParameters(device, world);

  return world;
}

// Render a single frame at the given pixelSamples and return (image, numSamples).
static std::vector<uint32_t> renderAtSpp(anari::Device device,
    anari::World world,
    anari::Camera camera,
    const uvec2 &imageSize,
    int spp,
    int &numSamplesOut)
{
  auto renderer = anari::newObject<anari::Renderer>(device, "wavefront");
  anari::setParameter(device, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(device, renderer, "pixelSamples", spp);
  anari::commitParameters(device, renderer);

  auto frame = anari::newObject<anari::Frame>(device);
  anari::setParameter(device, frame, "size", imageSize);
  anari::setParameter(device, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(device, frame, "world", world);
  anari::setParameter(device, frame, "camera", camera);
  anari::setParameter(device, frame, "renderer", renderer);
  anari::commitParameters(device, frame);

  anari::render(device, frame);
  anari::wait(device, frame);

  numSamplesOut = -1;
  anari::getProperty(device, frame, "numSamples", numSamplesOut, ANARI_WAIT);

  auto fb = anari::map<uint32_t>(device, frame, "channel.color");
  std::vector<uint32_t> out(fb.data, fb.data + size_t(fb.width) * fb.height);
  anari::unmap(device, frame, "channel.color");

  anari::release(device, renderer);
  anari::release(device, frame);
  return out;
}

int main()
{
  auto device = makeVisRTXDevice(statusFunc);
  auto world = generateScene(device);

  auto camera = anari::newObject<anari::Camera>(device, "perspective");
  const vec3 eye = {0.f, 0.f, -2.f};
  const vec3 dir = {0.f, 0.f, 1.f};
  const vec3 up = {0.f, 1.f, 0.f};
  const uvec2 imageSize = {256, 256};
  anari::setParameter(device, camera, "position", eye);
  anari::setParameter(device, camera, "direction", dir);
  anari::setParameter(device, camera, "up", up);
  anari::setParameter(
      device, camera, "aspect", imageSize[0] / float(imageSize[1]));
  anari::commitParameters(device, camera);

  constexpr int kSpp = 16;

  int ns1 = 0;
  auto oneSample = renderAtSpp(device, world, camera, imageSize, 1, ns1);
  int nsN = 0;
  auto manySamples = renderAtSpp(device, world, camera, imageSize, kSpp, nsN);

  anari::release(device, camera);
  anari::release(device, world);
  anari::release(device, device);

  int status = 0;

  // The pool must dispatch one wave per requested sample.
  if (ns1 != 1 || nsN != kSpp) {
    fprintf(stderr,
        "FAIL: numSamples wrong — spp=1 reported %d, spp=%d reported %d\n",
        ns1,
        kSpp,
        nsN);
    status = 1;
  }

  size_t changed = 0;
  for (size_t i = 0; i < oneSample.size() && i < manySamples.size(); ++i) {
    if (oneSample[i] != manySamples[i])
      ++changed;
  }
  const double changedFraction = double(changed) / double(oneSample.size());
  printf("wavefront accumulation: numSamples %d vs %d; %zu/%zu pixels differ "
         "1-sample vs %d-sample (%.2f%%)\n",
      ns1,
      nsN,
      changed,
      oneSample.size(),
      kSpp,
      100.0 * changedFraction);

  // Anti-aliasing the sphere silhouette perturbs a ring of edge pixels; require
  // a small but unambiguous fraction so N identical samples (a broken pool that
  // reused one sample) would fail here.
  if (changedFraction < 0.005) {
    fprintf(stderr,
        "FAIL: %d-sample image barely differs from 1-sample (%.3f%%) — the "
        "pool is not accumulating distinct samples across waves\n",
        kSpp,
        100.0 * changedFraction);
    status = 1;
  }

  // Atomic accumulation guard. At kSpp the pool runs ~kSpp samples of each pixel
  // concurrently in one wave (65536 px x 16 = the 2^20 pool capacity), so this
  // exercises the atomic scatter-add. Averaging is unbiased, so the mean lit
  // brightness must match the 1-sample render within noise — a race that
  // DROPPED samples would divide the same divisor into fewer deposits and
  // darken the many-sample image.
  auto meanLitLuminance = [](const std::vector<uint32_t> &img) {
    double sum = 0.0;
    size_t n = 0;
    for (uint32_t px : img) {
      if ((px & 0x00ffffffu) == 0)
        continue;
      const double r = px & 0xff, g = (px >> 8) & 0xff, b = (px >> 16) & 0xff;
      sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
      ++n;
    }
    return n ? sum / double(n) : 0.0;
  };
  const double mean1 = meanLitLuminance(oneSample);
  const double meanN = meanLitLuminance(manySamples);
  const double meanDrift = mean1 > 0.0 ? std::fabs(meanN - mean1) / mean1 : 1.0;
  printf("wavefront accumulation: mean lit luminance %.2f (1 spp) vs %.2f (%d "
         "spp), drift %.1f%%\n",
      mean1,
      meanN,
      kSpp,
      100.0 * meanDrift);
  // 3% cleanly separates the correct atomic path (~0.8% drift, Monte Carlo
  // noise between the 1-spp and N-spp means) from a non-atomic scatter-add,
  // which loses ~6% of the deposits to read-modify-write races here.
  if (meanDrift > 0.03) {
    fprintf(stderr,
        "FAIL: mean brightness drifted %.1f%% under concurrent accumulation — "
        "atomic scatter-add is losing samples\n",
        100.0 * meanDrift);
    status = 1;
  }

  if (status == 0)
    printf("PASS\n");
  return status;
}
